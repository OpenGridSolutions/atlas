// SPDX-License-Identifier: AGPL-3.0-only

//! Whisper audio frontend (multiclass Plan.md W.1) — CPU preprocessing
//! mirroring HF `WhisperFeatureExtractor` for whisper-large-v3:
//!
//!   bytes (wav/mp3/flac/m4a/ogg, via symphonia) → f32 mono
//!   → 16 kHz (linear resample) → 30 s windows (480 000 samples,
//!   zero-padded tail) → per window: STFT (n_fft 400, hop 160, periodic
//!   Hann, centered/reflect-padded) → |·|² → 128-bin Slaney mel
//!   filterbank (fmax 8 kHz; `num_mel_bins: 128` per the HF config —
//!   large-v3 uses 128, NOT the 80 of earlier Whispers)
//!   → log10(clamp ≥1e-10) → max(x, global_max − 8) → (x + 4) / 4.
//!
//! Output per window: `MelChunk` = `[N_FRAMES=3000][N_MELS=128]` f32
//! (HF computes 3001 frames on the padded window and drops the last).
//!
//! Chunking is STAGE 1 (fixed 30 s slicing, Plan W.2 note); the
//! `Vec<MelChunk>` API keeps overlapping/merging an additive change.

use anyhow::{Context, Result, bail};
use rustfft::{FftPlanner, num_complex::Complex};

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const N_MELS: usize = 128;
pub const CHUNK_SAMPLES: usize = 30 * SAMPLE_RATE; // 480_000
pub const N_FRAMES: usize = CHUNK_SAMPLES / HOP; // 3000
const FMAX: f64 = 8_000.0;
const N_FREQS: usize = N_FFT / 2 + 1; // 201

/// One 30 s log-mel window, row-major `[N_FRAMES][N_MELS]`.
pub struct MelChunk(pub Vec<f32>);

/// Decode arbitrary container/codec bytes to mono f32 @ 16 kHz.
pub fn decode_audio(bytes: Vec<u8>) -> Result<Vec<f32>> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::probe::Hint;

    let mss = MediaSourceStream::new(Box::new(std::io::Cursor::new(bytes)), Default::default());
    let probed = symphonia::default::get_probe()
        .format(&Hint::new(), mss, &Default::default(), &Default::default())
        .context("unrecognized audio container")?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .context("audio file has no default track")?;
    let track_id = track.id;
    let src_rate = track
        .codec_params
        .sample_rate
        .context("audio track declares no sample rate")? as usize;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &Default::default())
        .context("unsupported audio codec")?;

    let mut mono: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(symphonia::core::errors::Error::ResetRequired) => break,
            Err(e) => return Err(e).context("reading audio packet"),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // Tolerate isolated bad frames (common in truncated mp3s).
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(e).context("decoding audio packet"),
        };
        let spec = *decoded.spec();
        let ch = spec.channels.count().max(1);
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buf.copy_interleaved_ref(decoded);
        let samples = buf.samples();
        mono.extend(
            samples
                .chunks_exact(ch)
                .map(|frame| frame.iter().sum::<f32>() / ch as f32),
        );
    }
    if mono.is_empty() {
        bail!("audio decoded to zero samples");
    }
    Ok(resample_linear(&mono, src_rate, SAMPLE_RATE))
}

/// Linear-interpolation resampler — adequate for speech recognition
/// (Whisper is robust to mild resampling artifacts).
pub fn resample_linear(input: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let out_len = (input.len() as u64 * to as u64 / from as u64) as usize;
    let ratio = from as f64 / to as f64;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let i0 = pos as usize;
            let frac = (pos - i0 as f64) as f32;
            let a = input[i0.min(input.len() - 1)];
            let b = input[(i0 + 1).min(input.len() - 1)];
            a + (b - a) * frac
        })
        .collect()
}

/// Slaney hz→mel (librosa `htk=False`).
fn hz_to_mel(hz: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / (6.4f64.ln() / 27.0)
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * (6.4f64.ln() / 27.0)).exp()
    }
}

/// Slaney-normalized triangular mel filterbank `[N_MELS][N_FREQS]`
/// (librosa `mel(sr=16000, n_fft=400, n_mels=128, fmax=8000)`).
fn mel_filterbank() -> Vec<f32> {
    let fft_freqs: Vec<f64> = (0..N_FREQS)
        .map(|i| i as f64 * SAMPLE_RATE as f64 / N_FFT as f64)
        .collect();
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(FMAX);
    let f_pts: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    let mut fb = vec![0f32; N_MELS * N_FREQS];
    for m in 0..N_MELS {
        let (lo, ctr, hi) = (f_pts[m], f_pts[m + 1], f_pts[m + 2]);
        let enorm = 2.0 / (hi - lo); // Slaney area normalization
        for (k, &f) in fft_freqs.iter().enumerate() {
            let lower = (f - lo) / (ctr - lo);
            let upper = (hi - f) / (hi - ctr);
            let w = lower.min(upper).max(0.0);
            fb[m * N_FREQS + k] = (w * enorm) as f32;
        }
    }
    fb
}

/// 16 kHz mono samples → fixed 30 s log-mel chunks (stage-1 chunking).
pub fn log_mel_chunks(samples: &[f32]) -> Vec<MelChunk> {
    let n_chunks = samples.len().div_ceil(CHUNK_SAMPLES).max(1);
    let fb = mel_filterbank();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    // Periodic Hann window (numpy hanning is symmetric; HF/librosa use
    // periodic for STFT).
    let window: Vec<f32> = (0..N_FFT)
        .map(|i| {
            let x = (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos();
            (0.5 * (1.0 - x)) as f32
        })
        .collect();

    let mut chunks = Vec::with_capacity(n_chunks);
    for c in 0..n_chunks {
        // 30 s slice, zero-padded to exactly CHUNK_SAMPLES, then
        // reflect-padded n_fft/2 on both sides (centered STFT).
        let start = c * CHUNK_SAMPLES;
        let end = (start + CHUNK_SAMPLES).min(samples.len());
        let mut padded = vec![0f32; CHUNK_SAMPLES + N_FFT];
        let body = &mut padded[N_FFT / 2..N_FFT / 2 + CHUNK_SAMPLES];
        body[..end - start].copy_from_slice(&samples[start..end]);
        // Reflect pad the leading edge from the chunk body (trailing
        // edge reflects zeros/zero-pad — matches HF, which pads the
        // 30 s window with zeros BEFORE the reflective STFT pad).
        for i in 0..N_FFT / 2 {
            padded[N_FFT / 2 - 1 - i] = padded[N_FFT / 2 + 1 + i];
            padded[N_FFT / 2 + CHUNK_SAMPLES + i] = padded[N_FFT / 2 + CHUNK_SAMPLES - 2 - i];
        }

        // Power spectrogram → mel. HF computes 3001 frames and drops
        // the last; frame f starts at f*HOP in the padded signal.
        let mut mel = vec![0f32; N_FRAMES * N_MELS];
        let mut buf = vec![Complex::new(0f32, 0f32); N_FFT];
        let mut power = vec![0f32; N_FREQS];
        let mut log_max = f32::MIN;
        for f in 0..N_FRAMES {
            let off = f * HOP;
            for i in 0..N_FFT {
                buf[i] = Complex::new(padded[off + i] * window[i], 0.0);
            }
            fft.process(&mut buf);
            for (k, p) in power.iter_mut().enumerate() {
                *p = buf[k].norm_sqr();
            }
            for m in 0..N_MELS {
                let row = &fb[m * N_FREQS..(m + 1) * N_FREQS];
                let e: f32 = row.iter().zip(&power).map(|(w, p)| w * p).sum();
                let v = e.max(1e-10).log10();
                mel[f * N_MELS + m] = v;
                log_max = log_max.max(v);
            }
        }
        let floor = log_max - 8.0;
        for v in &mut mel {
            *v = (v.max(floor) + 4.0) / 4.0;
        }
        chunks.push(MelChunk(mel));
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_maps_to_minus_1_5() {
        // clamp(0, 1e-10) → log10 = −10; global max = −10 so the −8 dB
        // floor is inert; (−10 + 4) / 4 = −1.5 everywhere.
        let chunks = log_mel_chunks(&vec![0f32; SAMPLE_RATE]);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0.len(), N_FRAMES * N_MELS);
        assert!(chunks[0].0.iter().all(|&v| (v + 1.5).abs() < 1e-4));
    }

    #[test]
    fn sine_1khz_peaks_in_the_right_mel_band() {
        let samples: Vec<f32> = (0..CHUNK_SAMPLES)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        let mel = &log_mel_chunks(&samples)[0].0;
        // Mid-chunk frame, find the hottest mel bin.
        let frame = &mel[1500 * N_MELS..1501 * N_MELS];
        let argmax = frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        // Slaney scale: 1 kHz = mel 15.0 of 45.25 total → filter ≈ 42.
        assert!(
            (36..=48).contains(&argmax),
            "1 kHz peak landed in mel bin {argmax}"
        );
    }

    #[test]
    fn chunking_is_fixed_30s_slices() {
        assert_eq!(log_mel_chunks(&vec![0f32; CHUNK_SAMPLES]).len(), 1);
        assert_eq!(log_mel_chunks(&vec![0f32; CHUNK_SAMPLES + 1]).len(), 2);
        assert_eq!(log_mel_chunks(&vec![0f32; 3 * CHUNK_SAMPLES]).len(), 3);
    }

    #[test]
    fn resample_halves_and_preserves_length_ratio() {
        let input: Vec<f32> = (0..32_000).map(|i| (i % 100) as f32).collect();
        let out = resample_linear(&input, 32_000, 16_000);
        assert_eq!(out.len(), 16_000);
    }

    #[test]
    fn filterbank_rows_are_nonempty_and_bounded() {
        let fb = mel_filterbank();
        for m in 0..N_MELS {
            let row = &fb[m * N_FREQS..(m + 1) * N_FREQS];
            assert!(row.iter().any(|&w| w > 0.0), "empty mel filter {m}");
            assert!(row.iter().all(|&w| w.is_finite() && w >= 0.0));
        }
    }
}

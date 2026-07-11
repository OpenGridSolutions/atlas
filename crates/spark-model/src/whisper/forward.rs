// SPDX-License-Identifier: AGPL-3.0-only

//! Whisper forward passes (multiclass Plan.md W.2): encoder over one
//! 30 s mel chunk, per-chunk cross-attention K/V precompute, and the
//! greedy decoder loop. All BF16; pre-LN residual structure per the HF
//! reference. Buffers/launch geometry documented in `whisper.cu` and
//! the common GEMM microtests.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::KernelLaunch;

use super::*;
use crate::audio_preprocess::{MelChunk, N_FRAMES};

const CONV1_COLS: usize = N_MELS * 3; // 384
const CONV2_COLS: usize = D_MODEL * 3; // 3840

impl Whisper {
    fn gemm(&self, a: DevicePtr, b: DevicePtr, c: DevicePtr, m: usize, n: usize, k: usize) -> Result<()> {
        let g = self.gpu.as_ref();
        KernelLaunch::new(g, self.kern.gemm)
            .grid([(n as u32).div_ceil(16), (m as u32).div_ceil(16), 1])
            .block([16, 16, 1])
            .arg_ptr(a)
            .arg_ptr(b)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(g.default_stream())?;
        Ok(())
    }

    fn gemv(&self, a: DevicePtr, b: DevicePtr, c: DevicePtr, n: usize, k: usize) -> Result<()> {
        let g = self.gpu.as_ref();
        KernelLaunch::new(g, self.kern.gemv)
            .grid([(n as u32).div_ceil(4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(b)
            .arg_ptr(c)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(g.default_stream())?;
        Ok(())
    }

    fn layernorm(&self, x: DevicePtr, ln: &WB, out: DevicePtr, tokens: usize) -> Result<()> {
        let g = self.gpu.as_ref();
        KernelLaunch::new(g, self.kern.layernorm)
            .grid([tokens as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(ln.w)
            .arg_ptr(ln.b)
            .arg_ptr(out)
            .arg_u32(D_MODEL as u32)
            .arg_f32(LN_EPS)
            .launch(g.default_stream())?;
        Ok(())
    }

    fn bias(&self, data: DevicePtr, bias: DevicePtr, tokens: usize, n: usize, gelu: bool) -> Result<()> {
        let g = self.gpu.as_ref();
        let kern = if gelu { self.kern.bias_gelu } else { self.kern.bias_add };
        KernelLaunch::new(g, kern)
            .grid([tokens as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(data)
            .arg_ptr(bias)
            .arg_u32(n as u32)
            .launch(g.default_stream())?;
        Ok(())
    }

    fn add(&self, dst: DevicePtr, src: DevicePtr, n: usize) -> Result<()> {
        let g = self.gpu.as_ref();
        KernelLaunch::new(g, self.kern.add_inplace)
            .grid([(n as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(dst)
            .arg_ptr(src)
            .arg_i32(n as i32)
            .launch(g.default_stream())?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        q: DevicePtr,
        k: DevicePtr,
        v: DevicePtr,
        o: DevicePtr,
        nq: usize,
        nk: usize,
        causal: bool,
    ) -> Result<()> {
        let g = self.gpu.as_ref();
        KernelLaunch::new(g, self.kern.attention)
            .grid([nq as u32, N_HEADS as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(q)
            .arg_ptr(k)
            .arg_ptr(v)
            .arg_ptr(o)
            .arg_u32(nq as u32)
            .arg_u32(nk as u32)
            .arg_u32(N_HEADS as u32)
            .arg_u32(HEAD_DIM as u32)
            .arg_f32(1.0 / (HEAD_DIM as f32).sqrt())
            .arg_u32(causal as u32)
            .launch(g.default_stream())?;
        Ok(())
    }

    /// q/k/v projection + attention + out_proj for `tokens` rows of
    /// `x_normed`, result ADDED to `residual`.
    #[allow(clippy::too_many_arguments)]
    fn attn_block(
        &self,
        x_normed: DevicePtr,
        w: &AttnWeights,
        residual: DevicePtr,
        tokens: usize,
        k_src: DevicePtr,
        v_src: DevicePtr,
        nk: usize,
        causal: bool,
    ) -> Result<()> {
        let s = &self.scratch;
        self.gemm(x_normed, w.q.w, s.q, tokens, D_MODEL, D_MODEL)?;
        self.bias(s.q, w.q.b, tokens, D_MODEL, false)?;
        self.attention(s.q, k_src, v_src, s.o, tokens, nk, causal)?;
        self.gemm(s.o, w.out.w, s.k, tokens, D_MODEL, D_MODEL)?;
        self.bias(s.k, w.out.b, tokens, D_MODEL, false)?;
        self.add(residual, s.k, tokens * D_MODEL)
    }

    /// MLP (fc1 → GELU → fc2) for `tokens` rows, ADDED to `residual`.
    fn mlp_block(&self, x_normed: DevicePtr, fc1: &WB, fc2: &WB, residual: DevicePtr, tokens: usize) -> Result<()> {
        let s = &self.scratch;
        self.gemm(x_normed, fc1.w, s.h_c, tokens, FFN_DIM, D_MODEL)?;
        self.bias(s.h_c, fc1.b, tokens, FFN_DIM, true)?;
        self.gemm(s.h_c, fc2.w, s.k, tokens, D_MODEL, FFN_DIM)?;
        self.bias(s.k, fc2.b, tokens, D_MODEL, false)?;
        self.add(residual, s.k, tokens * D_MODEL)
    }

    /// Run the audio encoder over one 30 s mel chunk into `self.enc_out`
    /// and refresh every decoder layer's cross-attention K/V.
    pub fn encode(&mut self, mel: &MelChunk) -> Result<()> {
        let g = self.gpu.as_ref();
        let s = &self.scratch;

        // ── conv1 (k3 s1 p1) via host im2col + GEMM + bias-GELU ──
        let mut cols = vec![0f32; N_FRAMES * CONV1_COLS];
        for f in 0..N_FRAMES {
            for t in 0..3usize {
                let src = f as isize + t as isize - 1;
                if src < 0 || src >= N_FRAMES as isize {
                    continue; // zero pad
                }
                let src = src as usize;
                for c in 0..N_MELS {
                    cols[f * CONV1_COLS + t * N_MELS + c] = mel.0[src * N_MELS + c];
                }
            }
        }
        let bytes: Vec<u8> = cols
            .iter()
            .flat_map(|v| super::f32_to_bf16_bits(*v).to_le_bytes())
            .collect();
        g.copy_h2d(&bytes, s.im2col)?;
        self.gemm(s.im2col, self.conv1.w, s.h_a, N_FRAMES, D_MODEL, CONV1_COLS)?;
        self.bias(s.h_a, self.conv1.b, N_FRAMES, D_MODEL, true)?;

        // ── conv2 (k3 s2 p1): host roundtrip for the strided im2col ──
        g.synchronize(g.default_stream())?;
        let mut c1raw = vec![0u8; N_FRAMES * D_MODEL * 2];
        g.copy_d2h(s.h_a, &mut c1raw)?;
        let c1: Vec<f32> = c1raw
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect();
        let mut cols2 = vec![0f32; N_AUDIO_CTX * CONV2_COLS];
        for f in 0..N_AUDIO_CTX {
            for t in 0..3usize {
                let src = 2 * f as isize + t as isize - 1;
                if src < 0 || src >= N_FRAMES as isize {
                    continue;
                }
                let src = src as usize;
                cols2[f * CONV2_COLS + t * D_MODEL..f * CONV2_COLS + (t + 1) * D_MODEL]
                    .copy_from_slice(&c1[src * D_MODEL..(src + 1) * D_MODEL]);
            }
        }
        let bytes2: Vec<u8> = cols2
            .iter()
            .flat_map(|v| super::f32_to_bf16_bits(*v).to_le_bytes())
            .collect();
        g.copy_h2d(&bytes2, s.im2col)?;
        self.gemm(s.im2col, self.conv2.w, s.h_a, N_AUDIO_CTX, D_MODEL, CONV2_COLS)?;
        self.bias(s.h_a, self.conv2.b, N_AUDIO_CTX, D_MODEL, true)?;

        // ── + sinusoidal positions, 32 pre-LN layers, final LN ──
        self.add(s.h_a, self.enc_pos, N_AUDIO_CTX * D_MODEL)?;
        for i in 0..N_LAYERS {
            let l = &self.enc_layers[i];
            self.layernorm(s.h_a, &l.ln1, s.h_b, N_AUDIO_CTX)?;
            // k/v projections for self-attn read the SAME normed rows.
            self.gemm(s.h_b, l.attn.k.w, s.k, N_AUDIO_CTX, D_MODEL, D_MODEL)?;
            self.gemm(s.h_b, l.attn.v.w, s.v, N_AUDIO_CTX, D_MODEL, D_MODEL)?;
            self.bias(s.v, l.attn.v.b, N_AUDIO_CTX, D_MODEL, false)?;
            // q + attention + out_proj + residual (attn_block reuses s.q/s.o
            // and writes the projection into s.k AFTER attention consumed it).
            self.gemm(s.h_b, l.attn.q.w, s.q, N_AUDIO_CTX, D_MODEL, D_MODEL)?;
            self.bias(s.q, l.attn.q.b, N_AUDIO_CTX, D_MODEL, false)?;
            self.attention(s.q, s.k, s.v, s.o, N_AUDIO_CTX, N_AUDIO_CTX, false)?;
            self.gemm(s.o, l.attn.out.w, s.h_c, N_AUDIO_CTX, D_MODEL, D_MODEL)?;
            self.bias(s.h_c, l.attn.out.b, N_AUDIO_CTX, D_MODEL, false)?;
            self.add(s.h_a, s.h_c, N_AUDIO_CTX * D_MODEL)?;
            self.layernorm(s.h_a, &l.ln2, s.h_b, N_AUDIO_CTX)?;
            self.mlp_block(s.h_b, &l.fc1, &l.fc2, s.h_a, N_AUDIO_CTX)?;
        }
        self.layernorm(s.h_a, &self.enc_ln, self.enc_out, N_AUDIO_CTX)?;

        // ── per-layer cross-attention K/V for this chunk ──
        for l in &self.dec_layers {
            self.gemm(self.enc_out, l.cross_attn.k.w, l.k_cross, N_AUDIO_CTX, D_MODEL, D_MODEL)?;
            self.gemm(self.enc_out, l.cross_attn.v.w, l.v_cross, N_AUDIO_CTX, D_MODEL, D_MODEL)?;
            self.bias(l.v_cross, l.cross_attn.v.b, N_AUDIO_CTX, D_MODEL, false)?;
        }
        g.synchronize(g.default_stream())?;
        Ok(())
    }

    /// One decoder step: token id at cache position `pos` → raw logits
    /// (host f32, `VOCAB` entries).
    fn decode_step(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        let g = self.gpu.as_ref();
        let s = &self.scratch;
        let row = D_MODEL * 2;
        // x = tok_emb[token] + dec_pos[pos]
        g.copy_d2d(self.tok_emb.offset(token as usize * row), s.h_a, row)?;
        self.add(s.h_a, self.dec_pos.offset(pos * row), D_MODEL)?;

        for i in 0..N_LAYERS {
            let l = &self.dec_layers[i];
            // Self-attention: project k/v of THIS token, append to cache.
            self.layernorm(s.h_a, &l.ln1, s.h_b, 1)?;
            self.gemv(s.h_b, l.self_attn.k.w, s.h_c, D_MODEL, D_MODEL)?;
            g.copy_d2d(s.h_c, l.k_cache.offset(pos * row), row)?;
            self.gemv(s.h_b, l.self_attn.v.w, s.h_c, D_MODEL, D_MODEL)?;
            self.bias(s.h_c, l.self_attn.v.b, 1, D_MODEL, false)?;
            g.copy_d2d(s.h_c, l.v_cache.offset(pos * row), row)?;
            self.attn_block(s.h_b, &l.self_attn, s.h_a, 1, l.k_cache, l.v_cache, pos + 1, true)?;
            // Cross-attention over the chunk's encoder output.
            self.layernorm(s.h_a, &l.ln2, s.h_b, 1)?;
            self.attn_block(s.h_b, &l.cross_attn, s.h_a, 1, l.k_cross, l.v_cross, N_AUDIO_CTX, false)?;
            // MLP.
            self.layernorm(s.h_a, &l.ln3, s.h_b, 1)?;
            self.mlp_block(s.h_b, &l.fc1, &l.fc2, s.h_a, 1)?;
        }
        self.layernorm(s.h_a, &self.dec_ln, s.h_b, 1)?;
        // Tied lm head: logits = h @ tok_emb^T.
        self.gemv(s.h_b, self.tok_emb, s.logits, VOCAB, D_MODEL)?;
        g.synchronize(g.default_stream())?;
        let mut raw = vec![0u8; VOCAB * 2];
        g.copy_d2h(s.logits, &mut raw)?;
        Ok(raw
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect())
    }

    /// Greedy transcription of ONE 30 s chunk. `prompt` is the Whisper
    /// SOT sequence (sot, lang, task, notimestamps); `suppress` /
    /// `begin_suppress` follow generation_config semantics. Returns the
    /// generated text token ids (prompt and EOT excluded).
    pub fn transcribe_chunk(
        &mut self,
        mel: &MelChunk,
        prompt: &[u32],
        suppress: &[u32],
        begin_suppress: &[u32],
        eot: u32,
        max_new: usize,
    ) -> Result<Vec<u32>> {
        self.encode(mel)?;
        let mut pos = 0usize;
        let mut logits = Vec::new();
        for &t in prompt {
            logits = self.decode_step(t, pos)?;
            pos += 1;
        }
        let mut out = Vec::new();
        let max_new = max_new.min(N_TEXT_CTX - prompt.len() - 1);
        for step in 0..max_new {
            for &id in suppress {
                if (id as usize) < logits.len() {
                    logits[id as usize] = f32::NEG_INFINITY;
                }
            }
            if step == 0 {
                for &id in begin_suppress {
                    if (id as usize) < logits.len() {
                        logits[id as usize] = f32::NEG_INFINITY;
                    }
                }
            }
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(eot);
            if next == eot {
                break;
            }
            out.push(next);
            logits = self.decode_step(next, pos)?;
            pos += 1;
        }
        Ok(out)
    }
}

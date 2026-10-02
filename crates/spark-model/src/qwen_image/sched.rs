// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side math of the Qwen-Image-2.1 pipeline: the flow-matching Euler
//! sigma schedule (dynamic exponential shift + terminal stretch, as
//! `FlowMatchEulerDiscreteScheduler` with the checkpoint's
//! scheduler_config.json), the 3-axis RoPE tables of the joint sequence,
//! the sinusoidal timestep projection and the latent noise generator.

/// scheduler/scheduler_config.json
const BASE_IMAGE_SEQ_LEN: f64 = 256.0;
const MAX_IMAGE_SEQ_LEN: f64 = 8192.0;
const BASE_SHIFT: f64 = 0.5;
const MAX_SHIFT: f64 = 0.9;
const SHIFT_TERMINAL: f64 = 0.02;

/// transformer/config.json `axes_dims_rope` (frame, height, width).
pub const ROPE_AXES: [usize; 3] = [16, 56, 56];
const ROPE_THETA: f64 = 10_000.0;

/// Resolution-dependent shift (`calculate_shift`); linear in the number of
/// target tokens and deliberately NOT clamped above `MAX_IMAGE_SEQ_LEN`.
pub fn shift_mu(image_tokens: usize) -> f64 {
    let m = (MAX_SHIFT - BASE_SHIFT) / (MAX_IMAGE_SEQ_LEN - BASE_IMAGE_SEQ_LEN);
    let b = BASE_SHIFT - m * BASE_IMAGE_SEQ_LEN;
    image_tokens as f64 * m + b
}

/// `steps + 1` sigmas, from 1.0 down to the appended terminal 0.0.
pub fn sigmas(steps: usize, image_tokens: usize) -> Vec<f32> {
    let mu = shift_mu(image_tokens);
    // np.linspace(1.0, 1/steps, steps) cast to float32 before shifting.
    let mut s: Vec<f64> = (0..steps)
        .map(|i| {
            let lin = if steps == 1 {
                1.0
            } else {
                1.0 + (1.0 / steps as f64 - 1.0) * i as f64 / (steps - 1) as f64
            };
            lin as f32 as f64
        })
        .collect();
    for v in &mut s {
        *v = (mu.exp() / (mu.exp() + (1.0 / *v - 1.0))) as f32 as f64;
    }
    // stretch_shift_to_terminal
    let one_minus_last = 1.0 - s[steps - 1];
    let scale = one_minus_last / (1.0 - SHIFT_TERMINAL);
    let mut out: Vec<f32> = s.iter().map(|v| (1.0 - (1.0 - v) / scale) as f32).collect();
    out.push(0.0);
    out
}

/// Sinusoidal timestep projection (`QwenImage21TemporalTimesteps`, dim 256):
/// cos in the first half, sin in the second. `sigma` is the pipeline's
/// `timestep / 1000`; the embedder multiplies it back by 1000.
pub fn timestep_proj(sigma: f32) -> Vec<f32> {
    const HALF: usize = 128;
    let t = 1000.0f32 * sigma;
    let mut out = vec![0f32; 2 * HALF];
    for i in 0..HALF {
        let freq = (-(10_000f32.ln()) * i as f32 / HALF as f32).exp();
        let arg = t * freq;
        out[i] = arg.cos();
        out[HALF + i] = arg.sin();
    }
    out
}

/// Per-token (frame, height, width) position indices of the joint sequence
/// for text-to-image: `text_len` text tokens followed by one `lat_h × lat_w`
/// target block (`QwenImage21Rope.forward`). Text advances a shared
/// position on all three axes; the image block freezes the frame axis at
/// the position reached by the text and centres its grid on zero.
pub fn t2i_positions(text_len: usize, lat_h: usize, lat_w: usize) -> Vec<[i32; 3]> {
    let mut pos = Vec::with_capacity(text_len + lat_h * lat_w);
    for i in 0..text_len as i32 {
        pos.push([i, i, i]);
    }
    let frame = text_len as i32;
    let (h, w) = (lat_h as i32, lat_w as i32);
    for y in -(h - h / 2)..h / 2 {
        for x in -(w - w / 2)..w / 2 {
            pos.push([frame, y, x]);
        }
    }
    pos
}

/// cos/sin tables `[tokens, 64]` for the adjacent-pair (complex) rotation:
/// per axis `dim/2` frequencies `theta^(-2i/dim)`, axes concatenated.
pub fn rope_tables(positions: &[[i32; 3]]) -> (Vec<f32>, Vec<f32>) {
    let half: usize = ROPE_AXES.iter().map(|d| d / 2).sum();
    let mut cos = vec![0f32; positions.len() * half];
    let mut sin = vec![0f32; positions.len() * half];
    for (t, p) in positions.iter().enumerate() {
        let mut col = 0;
        for (axis, dim) in ROPE_AXES.iter().enumerate() {
            for i in 0..dim / 2 {
                // Mirror torch: frequency and angle are both f32.
                let freq = 1.0 / (ROPE_THETA as f32).powf((2 * i) as f32 / *dim as f32);
                let ang = p[axis] as f32 * freq;
                cos[t * half + col] = ang.cos();
                sin[t * half + col] = ang.sin();
                col += 1;
            }
        }
    }
    (cos, sin)
}

/// Deterministic standard-normal generator (splitmix64 + Box–Muller). Not
/// bit-compatible with torch's generator — a seed reproduces an image on
/// Atlas, not the diffusers sample for the same seed.
pub struct NormalRng {
    state: u64,
    spare: Option<f32>,
}

impl NormalRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        // (0, 1]: never 0, so ln() is finite.
        ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    pub fn sample(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let r = (-2.0 * self.uniform().ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * self.uniform();
        self.spare = Some((r * theta.sin()) as f32);
        (r * theta.cos()) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn working_memory_estimate_scales_with_resolution() {
        use crate::qwen_image::QwenImage;
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        let small = QwenImage::working_bytes(1024, 1024, 64);
        let large = QwenImage::working_bytes(2048, 2048, 64);
        assert!(gib(small) > 1.0 && gib(small) < 4.0, "{}", gib(small));
        assert!(gib(large) > 6.0 && gib(large) < 9.0, "{}", gib(large));
    }

    #[test]
    fn sigmas_match_diffusers_reference() {
        // FlowMatchEulerDiscreteScheduler, 4 steps, 256 target tokens (mu 0.5).
        let s = sigmas(4, 256);
        let want = [1.0, 0.744_611_4, 0.426_673_47, 0.019_999_98, 0.0];
        assert_eq!(s.len(), want.len());
        for (a, b) in s.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        assert!((shift_mu(256) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn t2i_positions_centre_the_image_grid() {
        let p = t2i_positions(3, 4, 6);
        assert_eq!(p.len(), 3 + 24);
        assert_eq!(p[2], [2, 2, 2]);
        assert_eq!(p[3], [3, -2, -3]);
        assert_eq!(p[3 + 23], [3, 1, 2]);
    }

    #[test]
    fn normal_rng_is_deterministic_and_standard() {
        let mut a = NormalRng::new(42);
        let mut b = NormalRng::new(42);
        let xs: Vec<f32> = (0..20_000).map(|_| a.sample()).collect();
        assert!(
            xs.iter()
                .zip((0..20_000).map(|_| b.sample()))
                .all(|(x, y)| *x == y)
        );
        let mean = xs.iter().sum::<f32>() / xs.len() as f32;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / xs.len() as f32;
        assert!(
            mean.abs() < 0.03 && (var - 1.0).abs() < 0.05,
            "mean {mean} var {var}"
        );
    }
}

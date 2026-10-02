// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen-Image-2.1 text-to-image pipeline as a RESIDENT SIDECAR MODULE
//! (Todo Phase 3.2) — like Whisper, deliberately OUTSIDE the `Model` trait /
//! scheduler / hot-swap stack: diffusion has no autoregressive decode and no
//! paged KV, so it is driven synchronously through its own kernel target.
//!
//! Pipeline (`QwenImage21Pipeline`, text-to-image):
//!   prompt tokens → Qwen3-VL-8B text tower (pre-final-norm hidden states,
//!   system prefix dropped) → 32-block single-stream DiT under a
//!   flow-matching Euler sampler → VAE decoder → RGBA pixels.
//!
//! Precision: BF16 weights and activations with FP32 accumulation; the
//! sampler state (latents) and the final pixels stay FP32. NO NVFP4
//! quantization anywhere in this path.
//!
//! Image conditioning (`/v1/images/edits`): condition images run through
//! the Qwen3-VL vision tower (lazily loaded `EditStack`, deepstack taps
//! included) so the prompt is visually grounded, their VAE-encoded
//! latents join the DiT prefix as bidirectional blocks, and the target
//! denoises against the cached joint prefix. Not implemented:
//! classifier-free guidance (the model is meant to be sampled without it).

mod dit;
mod ops;
pub mod sched;
pub mod st;
mod text;
mod vae;
mod vae_enc;
mod vision;

use std::path::Path;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub use dit::Trace;
use dit::{Dit, LATENT_CH};
use ops::{Ops, Pool, WeightSet};
use std::path::PathBuf;
use text::{TE_HIDDEN, TextEncoder};
use vae::VaeDecoder;
use vae_enc::VaeEncoder;
use vision::VisionTower;

/// System prompt the checkpoint was trained with.
const SYS_PROMPT: &str = "Comprehend and analyze the provided prompt.";

/// One latent token covers a 16×16 pixel tile; sizes snap to 32 px so the
/// latent grid stays even (the joint sequence reserves 2×2 token slots).
pub const PIXEL_MULTIPLE: usize = 32;
const VAE_SCALE: usize = 16;

/// The system-role prefix whose tokens are dropped from the text encoder
/// output (tokenize it to get the drop count).
pub fn system_prefix() -> String {
    format!("<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n")
}

/// The raw text-to-image template; tokenize WITHOUT adding special tokens.
pub fn t2i_prompt(prompt: &str) -> String {
    // Qwen has no BOS token, so an empty prompt would leave nothing to read.
    let prompt = if prompt.is_empty() { " " } else { prompt };
    format!(
        "{}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        system_prefix()
    )
}

/// The image-conditioned (ti2i) template with one
/// `<imageK><|vision_start|><|image_pad|><|vision_end|>` group per
/// condition image (reference `prompt_template_ti2i`). The single
/// `<|image_pad|>` per group is expanded to the image's merged ViT token
/// count by the caller after tokenization.
pub fn ti2i_prompt(prompt: &str, n_images: usize) -> String {
    let prompt = if prompt.is_empty() { " " } else { prompt };
    let mut vision = String::new();
    for i in 1..=n_images.max(1) {
        if i > 1 {
            vision.push(' ');
        }
        vision.push_str(&format!(
            "<image{i}><|vision_start|><|image_pad|><|vision_end|>"
        ));
    }
    format!(
        "{}<|im_start|>user\n{vision}{prompt}<|im_end|>\n<|im_start|>assistant\n",
        system_prefix()
    )
}

/// One condition image for [`QwenImage::generate_edit`].
pub struct EditImage {
    /// ViT patch pixels + grid from `vision_preprocess` (the RGBA source
    /// composited over white, per the reference).
    pub vit_pixels: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
    /// VAE input: RGBA NHWC f32 in [-1, 1] at (vae_height, vae_width),
    /// both 32-aligned (the same resize the ViT copy uses).
    pub vae_pixels: Vec<f32>,
    pub vae_height: usize,
    pub vae_width: usize,
    /// This image's expanded `<|image_pad|>` run in the token sequence:
    /// (start, len), including the system prefix offset.
    pub span: (usize, usize),
}

struct EditStack {
    vision: VisionTower,
    vae_enc: VaeEncoder,
}

/// One condition block for [`QwenImage::denoise_edit_host`], host-side:
/// normalized `[lat_h*lat_w, 64]` latents plus the image's pad span in
/// the (system-dropped) VLM row space.
pub struct HostCond<'a> {
    pub vlm_start: usize,
    pub vlm_len: usize,
    pub lat_h: usize,
    pub lat_w: usize,
    pub latents: &'a [f32],
}

pub struct GenParams {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub seed: u64,
}

pub struct GeneratedImage {
    pub width: usize,
    pub height: usize,
    /// Channels per pixel (4 = RGBA, the VAE's native output).
    pub channels: usize,
    /// Row-major, interleaved 8-bit channels.
    pub pixels: Vec<u8>,
}

pub struct QwenImage {
    gpu: Box<dyn GpuBackend>,
    ops: Ops,
    text: TextEncoder,
    dit: Dit,
    vae: VaeDecoder,
    /// ViT tower + VAE encoder, loaded lazily for `/v1/images/edits`.
    edit: Option<EditStack>,
    /// Snapshot dir, kept for the lazy edit-stack load.
    model_dir: PathBuf,
    /// Every resident allocation (weights + cuBLASLt workspace), released
    /// on drop.
    weights: Vec<DevicePtr>,
}

impl Drop for QwenImage {
    fn drop(&mut self) {
        let g = self.gpu.as_ref();
        let _ = g.synchronize(g.default_stream());
        for p in self.weights.drain(..) {
            let _ = g.free(p);
        }
    }
}

impl QwenImage {
    /// Load the pipeline from a diffusers snapshot directory
    /// (`text_encoder/`, `transformer/`, `vae/`) onto `gpu`, a backend bound
    /// to the qwen-image kernel target. ~29 GiB BF16 resident.
    pub fn load(model_dir: &Path, gpu: Box<dyn GpuBackend>) -> Result<Self> {
        let g = gpu.as_ref();
        let mut ws = WeightSet::new(g);
        let ops = Ops::resolve(g, ws.alloc(ops::CUBLAS_WS_BYTES)?)?;
        let t0 = std::time::Instant::now();
        let dit = Dit::load(&model_dir.join("transformer"), &mut ws)?;
        tracing::info!(
            "qwen-image: DiT loaded ({:.1} GiB, {:.1}s)",
            dit.weights_bytes as f64 / (1u64 << 30) as f64,
            t0.elapsed().as_secs_f64()
        );
        let t1 = std::time::Instant::now();
        let text = TextEncoder::load(&model_dir.join("text_encoder"), &mut ws)?;
        tracing::info!(
            "qwen-image: text encoder loaded ({:.1} GiB, {:.1}s)",
            text.weights_bytes as f64 / (1u64 << 30) as f64,
            t1.elapsed().as_secs_f64()
        );
        let vae = VaeDecoder::load(&model_dir.join("vae"), &mut ws)?;
        tracing::info!(
            "qwen-image: VAE decoder loaded ({:.2} GiB)",
            vae.weights_bytes as f64 / (1u64 << 30) as f64
        );
        let weights = ws.finish();
        Ok(Self {
            gpu,
            ops,
            text,
            dit,
            vae,
            edit: None,
            model_dir: model_dir.to_path_buf(),
            weights,
        })
    }

    /// The sidecar's GPU backend (for cross-thread context binding).
    pub fn gpu_backend(&self) -> &dyn GpuBackend {
        self.gpu.as_ref()
    }

    /// Resident device weight bytes (for /v1/stats).
    pub fn weights_bytes(&self) -> usize {
        self.dit.weights_bytes + self.text.weights_bytes + self.vae.weights_bytes
    }

    /// Snap a requested size to the latent grid: `(pixels, latent tokens)`.
    pub fn snap(pixels: usize) -> (usize, usize) {
        let p = (pixels / PIXEL_MULTIPLE).max(1) * PIXEL_MULTIPLE;
        (p, p / VAE_SCALE)
    }

    /// Device working memory one generation needs on top of the resident
    /// weights (an upper estimate, for the caller's free-memory preflight).
    /// The DiT scratch is released before the VAE decodes, so the two
    /// phases do not add up.
    pub fn working_bytes(width: usize, height: usize, prompt_tokens: usize) -> usize {
        let (width, lat_w) = Self::snap(width);
        let (height, lat_h) = Self::snap(height);
        let n = lat_h * lat_w;
        let keys = ops::pad_keys(n + prompt_tokens);
        // DiT: attention scores (f32) + probabilities (bf16), eleven
        // hidden-sized buffers (residual, norm, q/kv, head-major Q/K/V, …)
        // and the two MLP activations.
        let dit = n * keys * 6 + 11 * n * 4096 * 2 + 2 * n * 12288 * 2;
        // VAE: the widest stage holds the 2×-upsampled 288-channel map, its
        // convolution output and the half-resolution block input, plus one
        // im2col band.
        let px = width * height;
        let vae = px * 288 * 2 * 2 + px / 4 * (576 + 288) * 2 + (768 << 20);
        dit.max(vae)
    }

    /// Text-to-image. `tokens` is the tokenized [`t2i_prompt`], `sys_tokens`
    /// the token count of [`system_prefix`]. `on_step(done, total)` fires after
    /// every denoising step.
    pub fn generate(
        &self,
        tokens: &[u32],
        sys_tokens: usize,
        p: &GenParams,
        mut on_step: impl FnMut(usize, usize),
    ) -> Result<GeneratedImage> {
        ensure!(
            tokens.len() > sys_tokens,
            "prompt has no tokens after the system prefix"
        );
        ensure!(p.steps >= 1, "steps must be >= 1");
        let g = self.gpu.as_ref();
        let (width, lat_w) = Self::snap(p.width);
        let (height, lat_h) = Self::snap(p.height);
        let n = lat_h * lat_w;

        let mut pool = Pool::new(g);
        let hidden = self.text.forward(g, &self.ops, &mut pool, tokens, None)?;
        let prompt = hidden.offset(sys_tokens * TE_HIDDEN * 2);
        let tp = tokens.len() - sys_tokens;

        let mut rng = sched::NormalRng::new(p.seed);
        let mut latents: Vec<f32> = (0..n * LATENT_CH).map(|_| rng.sample()).collect();
        let sigmas = sched::sigmas(p.steps, n);
        self.dit.denoise(
            g,
            &self.ops,
            prompt,
            tp,
            &[],
            lat_h,
            lat_w,
            &mut latents,
            &sigmas,
            None,
            |s| on_step(s + 1, p.steps),
        )?;
        drop(pool);

        let img = self
            .vae
            .decode(g, &self.ops, &latents, lat_h, lat_w, None)?;
        let pixels = img
            .iter()
            .map(|v| ((v / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        Ok(GeneratedImage {
            width,
            height,
            channels: self.vae.out_channels,
            pixels,
        })
    }

    /// Load the edit stack (ViT tower + VAE encoder) if not yet resident.
    /// Lazy so text-to-image deployments skip its ~1.5 GiB of weights and
    /// the encoder scratch; `/v1/images/edits` calls this on first use.
    pub fn ensure_edit_stack(&mut self) -> Result<()> {
        if self.edit.is_some() {
            return Ok(());
        }
        let g = self.gpu.as_ref();
        let mut ws = WeightSet::new(g);
        let t0 = std::time::Instant::now();
        let vision = VisionTower::load(self.text.st(), &mut ws, g)?;
        let vae_enc = VaeEncoder::load(&self.model_dir.join("vae"), &mut ws)?;
        tracing::info!(
            "qwen-image: edit stack loaded (ViT {:.2} GiB + VAE encoder {:.2} GiB, {:.1}s)",
            vision.weights_bytes as f64 / (1u64 << 30) as f64,
            vae_enc.weights_bytes as f64 / (1u64 << 30) as f64,
            t0.elapsed().as_secs_f64(),
        );
        self.weights.extend(ws.finish());
        self.edit = Some(EditStack { vision, vae_enc });
        Ok(())
    }

    /// True once [`Self::ensure_edit_stack`] has run.
    pub fn edit_ready(&self) -> bool {
        self.edit.is_some()
    }

    /// Image-conditioned generation (`/v1/images/edits`). `tokens` is the
    /// tokenized ti2i template with the `<|image_pad|>` runs already
    /// expanded; `images` carry each condition image's ViT pixels and VAE
    /// pixels plus its pad span. Call [`Self::ensure_edit_stack`] first.
    pub fn generate_edit(
        &self,
        tokens: &[u32],
        sys_tokens: usize,
        images: &[EditImage],
        p: &GenParams,
        mut on_step: impl FnMut(usize, usize),
    ) -> Result<GeneratedImage> {
        let edit = self
            .edit
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("edit stack not loaded — call ensure_edit_stack"))?;
        ensure!(
            !images.is_empty(),
            "edits need at least one condition image"
        );
        ensure!(p.steps >= 1, "steps must be >= 1");
        let g = self.gpu.as_ref();
        let (width, lat_w) = Self::snap(p.width);
        let (height, lat_h) = Self::snap(p.height);
        let n = lat_h * lat_w;

        let mut pool = Pool::new(g);
        let vit_images: Vec<(&[f32], usize, usize)> = images
            .iter()
            .map(|i| (i.vit_pixels.as_slice(), i.grid_h, i.grid_w))
            .collect();
        let vl = text::VlConditioning {
            tower: &edit.vision,
            images: vit_images,
            spans: images.iter().map(|i| i.span).collect(),
        };
        let hidden = self
            .text
            .forward(g, &self.ops, &mut pool, tokens, Some(&vl))?;
        let prompt = hidden.offset(sys_tokens * TE_HIDDEN * 2);
        let tv = tokens.len() - sys_tokens;

        // VAE-encode each condition image to normalized latents on device.
        let mut cond = Vec::with_capacity(images.len());
        for img in images {
            ensure!(
                img.span.0 >= sys_tokens,
                "condition image span inside the system prefix"
            );
            let lat = edit.vae_enc.encode(
                g,
                &self.ops,
                &img.vae_pixels,
                img.vae_height,
                img.vae_width,
            )?;
            let dev = pool.upload(&st::f32s_to_bf16_bytes(&lat))?;
            cond.push(dit::CondLatents {
                latents: dev,
                lat_h: img.vae_height / VAE_SCALE,
                lat_w: img.vae_width / VAE_SCALE,
                vlm_start: img.span.0 - sys_tokens,
                vlm_len: img.span.1,
            });
        }

        let mut rng = sched::NormalRng::new(p.seed);
        let mut latents: Vec<f32> = (0..n * LATENT_CH).map(|_| rng.sample()).collect();
        let sigmas = sched::sigmas(p.steps, n);
        self.dit.denoise(
            g,
            &self.ops,
            prompt,
            tv,
            &cond,
            lat_h,
            lat_w,
            &mut latents,
            &sigmas,
            None,
            |s| on_step(s + 1, p.steps),
        )?;
        drop(pool);

        let img = self
            .vae
            .decode(g, &self.ops, &latents, lat_h, lat_w, None)?;
        let pixels = img
            .iter()
            .map(|v| ((v / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        Ok(GeneratedImage {
            width,
            height,
            channels: self.vae.out_channels,
            pixels,
        })
    }

    // ── stage-level entry points for the golden harness ──

    /// Text tower hidden states `[tokens, 4096]` (pre-final-norm), host f32.
    pub fn text_hidden(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let g = self.gpu.as_ref();
        let mut pool = Pool::new(g);
        let h = self.text.forward(g, &self.ops, &mut pool, tokens, None)?;
        let mut raw = vec![0u8; tokens.len() * TE_HIDDEN * 2];
        g.copy_d2h(h, &mut raw)?;
        Ok(st::bf16_bytes_to_f32s(&raw))
    }

    /// Sampler over caller-supplied prompt embeddings `[tp, 4096]` and
    /// initial `latents` `[lat_h*lat_w, 64]` (updated in place).
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_host(
        &self,
        prompt_embeds: &[f32],
        lat_h: usize,
        lat_w: usize,
        latents: &mut [f32],
        steps: usize,
        trace: Option<&mut Trace>,
    ) -> Result<()> {
        let g = self.gpu.as_ref();
        let tp = prompt_embeds.len() / TE_HIDDEN;
        let mut pool = Pool::new(g);
        let prompt = pool.upload(&st::f32s_to_bf16_bytes(prompt_embeds))?;
        let sigmas = sched::sigmas(steps, lat_h * lat_w);
        self.dit.denoise(
            g,
            &self.ops,
            prompt,
            tp,
            &[],
            lat_h,
            lat_w,
            latents,
            &sigmas,
            trace,
            |_| {},
        )
    }

    /// Text tower hidden states WITH condition images (edit path),
    /// `[tokens, 4096]` pre-final-norm, host f32. Needs
    /// [`Self::ensure_edit_stack`].
    pub fn text_hidden_vl(&self, tokens: &[u32], images: &[EditImage]) -> Result<Vec<f32>> {
        let edit = self
            .edit
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("edit stack not loaded"))?;
        let g = self.gpu.as_ref();
        let mut pool = Pool::new(g);
        let vit_images: Vec<(&[f32], usize, usize)> = images
            .iter()
            .map(|i| (i.vit_pixels.as_slice(), i.grid_h, i.grid_w))
            .collect();
        let vl = text::VlConditioning {
            tower: &edit.vision,
            images: vit_images,
            spans: images.iter().map(|i| i.span).collect(),
        };
        let h = self
            .text
            .forward(g, &self.ops, &mut pool, tokens, Some(&vl))?;
        let mut raw = vec![0u8; tokens.len() * TE_HIDDEN * 2];
        g.copy_d2h(h, &mut raw)?;
        Ok(st::bf16_bytes_to_f32s(&raw))
    }

    /// Vision-tower features for the golden harness: `(final, deepstack)`
    /// with `final` the merged `[Σmp, 4096]` splice features and
    /// `deepstack[k]` the tap-k features, host f32. Needs
    /// [`Self::ensure_edit_stack`].
    pub fn vision_features_host(&self, images: &[EditImage]) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let edit = self
            .edit
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("edit stack not loaded"))?;
        let g = self.gpu.as_ref();
        let vit_images: Vec<(&[f32], usize, usize)> = images
            .iter()
            .map(|i| (i.vit_pixels.as_slice(), i.grid_h, i.grid_w))
            .collect();
        let (feats, total) = edit.vision.forward(&vit_images, g, g.default_stream())?;
        g.synchronize(g.default_stream())?;
        let read = |p: spark_runtime::gpu::DevicePtr, rows: usize| -> Result<Vec<f32>> {
            let mut raw = vec![0u8; rows * TE_HIDDEN * 2];
            g.copy_d2h(p, &mut raw)?;
            Ok(st::bf16_bytes_to_f32s(&raw))
        };
        let mut final_out = Vec::new();
        let mut ds_out = vec![Vec::new(); edit.vision.deepstack_taps()];
        for f in &feats {
            final_out.extend(read(edit.vision.final_ptr(f), f.merged_p)?);
            for (k, out) in ds_out.iter_mut().enumerate() {
                out.extend(read(edit.vision.deepstack_ptr(k, total, f), f.merged_p)?);
            }
        }
        Ok((final_out, ds_out))
    }

    /// VAE-encode one condition image (`[h*w, 4]` f32 RGBA in [-1, 1]) to
    /// normalized mode latents `[lat_h*lat_w * 64]`, host f32. Needs
    /// [`Self::ensure_edit_stack`].
    pub fn encode_cond_host(
        &self,
        vae_pixels: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>> {
        let edit = self
            .edit
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("edit stack not loaded"))?;
        edit.vae_enc
            .encode(self.gpu.as_ref(), &self.ops, vae_pixels, height, width)
    }

    /// Edit sampler over caller-supplied prompt embeddings (VLM rows incl.
    /// image slots) and condition latents — isolates the DiT stage for the
    /// golden harness.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_edit_host(
        &self,
        prompt_embeds: &[f32],
        cond: &[HostCond<'_>],
        lat_h: usize,
        lat_w: usize,
        latents: &mut [f32],
        steps: usize,
        trace: Option<&mut Trace>,
    ) -> Result<()> {
        let g = self.gpu.as_ref();
        let tv = prompt_embeds.len() / TE_HIDDEN;
        let mut pool = Pool::new(g);
        let prompt = pool.upload(&st::f32s_to_bf16_bytes(prompt_embeds))?;
        let mut cond_dev = Vec::with_capacity(cond.len());
        for c in cond {
            ensure!(
                c.latents.len() == c.lat_h * c.lat_w * LATENT_CH,
                "condition latents shape mismatch"
            );
            cond_dev.push(dit::CondLatents {
                latents: pool.upload(&st::f32s_to_bf16_bytes(c.latents))?,
                lat_h: c.lat_h,
                lat_w: c.lat_w,
                vlm_start: c.vlm_start,
                vlm_len: c.vlm_len,
            });
        }
        let sigmas = sched::sigmas(steps, lat_h * lat_w);
        self.dit.denoise(
            g,
            &self.ops,
            prompt,
            tv,
            &cond_dev,
            lat_h,
            lat_w,
            latents,
            &sigmas,
            trace,
            |_| {},
        )
    }

    /// VAE decode of normalized latents → `[H*W, channels]` f32 in [-1, 1].
    pub fn decode_host(
        &self,
        latents: &[f32],
        lat_h: usize,
        lat_w: usize,
        trace: Option<&mut Trace>,
    ) -> Result<Vec<f32>> {
        self.vae
            .decode(self.gpu.as_ref(), &self.ops, latents, lat_h, lat_w, trace)
    }
}

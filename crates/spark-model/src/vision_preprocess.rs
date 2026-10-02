// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-side image preprocessing for Qwen3-VL vision inputs.
//!
//! Decodes base64 JPEG/PNG images, resizes to a grid snapped to
//! `patch_size × spatial_merge_size`, normalizes with ImageNet stats,
//! and produces a flat `f32` tensor ready for the GPU vision encoder.

use anyhow::{Context, Result};
use atlas_core::config::VisionConfig;
use image::{DynamicImage, ImageFormat};

/// SigLIP normalization — matches HF's Qwen2VLImageProcessor
/// (`image_mean = image_std = (0.5, 0.5, 0.5)` → pixels mapped to [-1, 1]).
const MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Hard bound on ViT patches per image — MUST match the vision
/// encoder's scratch-buffer sizing (`enc_impl/init.rs` `p_max`, which
/// allocates the [seq, seq] SDPA score matrix from it). The preprocessing
/// pixel budget below derives from this so no admissible image can
/// overflow the encoder buffers. 16384 patches = 4.19M px = 4096 vision
/// tokens after the 2×2 merge (a 2500-px-wide document page passes
/// through natively — the document-extraction use case).
pub const VISION_MAX_PATCHES: usize = 16_384;

/// Default/maximum pixel budget: `VISION_MAX_PATCHES × 16×16`. An
/// explicit `max_pixels` request below this is honored; above it is
/// clamped (buffer safety). Replaces the old 1280-px long-side cap,
/// which silently starved wide document images (2500×1200 → 760 tokens
/// instead of the 2964 the HF reference produces).
const HARD_MAX_PIXELS: u32 = (VISION_MAX_PATCHES as u32) * 256;

/// Minimum pixel budget (OCR floor). The checkpoint's
/// preprocessor_config ships `size.shortest_edge: 65536` (the HF
/// default MIN-pixels bound), but at that floor a small formula
/// screenshot lands at ~75 vision tokens — measured too coarse for OCR
/// (2026-07-12: 264×86 formula read as a single glyph at the HF floor).
/// 262144 px = 256 vision tokens minimum, the Qwen OCR guidance band.
const MIN_PIXELS: u32 = 262_144;

/// Sanity guard on a single side (pathological aspect ratios / u32
/// arithmetic). Multiple of every grid_unit in the family.
const MAX_DIM: u32 = 16_384;

/// Decode a base64 data URI or raw base64 string into a `DynamicImage`.
fn decode_image(data_uri: &str) -> Result<DynamicImage> {
    // Strip optional "data:image/<fmt>;base64," prefix.
    let b64 = if let Some(pos) = data_uri.find(",base64,") {
        &data_uri[pos + 8..]
    } else if data_uri.starts_with("data:") {
        // "data:image/jpeg;base64,..."
        data_uri
            .find(',')
            .map(|p| &data_uri[p + 1..])
            .unwrap_or(data_uri)
    } else {
        data_uri
    };

    let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64.trim())
        .context("base64 decode failed")?;

    // Probe format from magic bytes.
    let fmt = image::guess_format(&bytes).unwrap_or(ImageFormat::Jpeg);
    image::load_from_memory_with_format(&bytes, fmt).context("image decode failed")
}

/// Compute the target (H, W) following HF `smart_resize`
/// (transformers `Qwen2VLImageProcessor`): round both sides to the
/// `grid_unit`, then rescale once if the area violates the pixel
/// budget — FLOOR-rounded when shrinking to `max_pixels`, CEIL-rounded
/// when growing to the `MIN_PIXELS` OCR floor. Aspect ratio is
/// preserved up to grid rounding.
fn target_size_with_max_pixels(
    orig_h: u32,
    orig_w: u32,
    grid_unit: u32,
    max_pixels: Option<usize>,
) -> (u32, u32) {
    let factor = grid_unit as f32;
    let h = orig_h as f32;
    let w = orig_w as f32;
    // Effective max: explicit request wins below the hard buffer bound.
    let max_px = max_pixels
        .filter(|&p| p > 0)
        .map(|p| (p as u32).min(HARD_MAX_PIXELS))
        .unwrap_or(HARD_MAX_PIXELS) as f32;
    // An explicit max_pixels below the OCR floor wins over the floor.
    let min_px = (MIN_PIXELS as f32).min(max_px);

    let mut h_bar = (h / factor).round().max(1.0) * factor;
    let mut w_bar = (w / factor).round().max(1.0) * factor;
    if h_bar * w_bar > max_px {
        let beta = ((h * w) / max_px).sqrt();
        h_bar = (h / beta / factor).floor().max(1.0) * factor;
        w_bar = (w / beta / factor).floor().max(1.0) * factor;
    } else if h_bar * w_bar < min_px {
        let beta = (min_px / (h * w)).sqrt();
        h_bar = (h * beta / factor).ceil() * factor;
        w_bar = (w * beta / factor).ceil() * factor;
    }
    // Buffer-safety backstop: ceil-growth toward a max_pixels that sits
    // below the OCR floor may overshoot by a grid row — shrink back.
    if h_bar * w_bar > HARD_MAX_PIXELS as f32 {
        let beta = ((h_bar * w_bar) / HARD_MAX_PIXELS as f32).sqrt();
        h_bar = (h_bar / beta / factor).floor().max(1.0) * factor;
        w_bar = (w_bar / beta / factor).floor().max(1.0) * factor;
    }
    ((h_bar as u32).min(MAX_DIM), (w_bar as u32).min(MAX_DIM))
}

/// Preprocess a single base64-encoded image for the Qwen3-VL encoder.
///
/// Returns:
/// - `pixels`: flat `f32` tensor shaped `[P, C × T × H_p × W_p]` where:
///   - `P = (H/patch_size) × (W/patch_size)` — number of patches
///   - `C = 3` channels, `T = temporal_patch_size` (image duplicated), `H_p = W_p = patch_size`
/// - `grid_h`: number of patches along height
/// - `grid_w`: number of patches along width
pub fn preprocess_image(data_uri: &str, vcfg: &VisionConfig) -> Result<(Vec<f32>, usize, usize)> {
    preprocess_image_with_max_pixels(data_uri, vcfg, None)
}

/// Preprocess an image with an optional max-pixels cap, matching vLLM-style
/// multimodal processor controls. `None` uses the default budgets
/// (`MIN_PIXELS` OCR floor .. `HARD_MAX_PIXELS` buffer bound).
pub fn preprocess_image_with_max_pixels(
    data_uri: &str,
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
) -> Result<(Vec<f32>, usize, usize)> {
    let img = decode_image(data_uri)?;
    preprocess_rgb_with_max_pixels(img.to_rgb8(), vcfg, max_pixels)
}

/// Patch-preprocess an already-decoded RGB image. Split out of
/// [`preprocess_image_with_max_pixels`] for callers that transform the
/// image before patching (the /v1/ocr pipeline applies Chandra's
/// min-dim upscale + scale_to_fit first) — avoids a PNG re-encode
/// round-trip through the data-URI path.
pub fn preprocess_rgb_with_max_pixels(
    img: image::RgbImage,
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
) -> Result<(Vec<f32>, usize, usize)> {
    let (orig_w, orig_h) = (img.width(), img.height());

    let grid_unit = (vcfg.patch_size * vcfg.spatial_merge_size) as u32;
    let (th, tw) = target_size_with_max_pixels(orig_h, orig_w, grid_unit, max_pixels);

    // Resize with CatmullRom — closest BICUBIC match in the `image` crate,
    // matching HF's `Qwen2VLImageProcessor` which uses PIL resample=3 (BICUBIC).
    let img = image::imageops::resize(&img, tw, th, image::imageops::FilterType::CatmullRom);

    let ps = vcfg.patch_size;
    let tp = vcfg.temporal_patch_size;
    let grid_h = (th as usize) / ps;
    let grid_w = (tw as usize) / ps;
    let num_patches = grid_h * grid_w;
    // Flattened patch dim: C × temporal_patch_size × patch_size × patch_size
    let patch_dim = 3 * tp * ps * ps;
    let mut pixels = vec![0.0f32; num_patches * patch_dim];

    // Build patches. The temporal dimension is handled by duplicating the image `tp` times.
    // Layout: [P, C, T, Hp, Wp] → stored as [P, C*T*Hp*Wp] in row-major order.
    for ph in 0..grid_h {
        for pw in 0..grid_w {
            let patch_idx = ph * grid_w + pw;
            for c in 0..3usize {
                for t in 0..tp {
                    for py in 0..ps {
                        for px in 0..ps {
                            let pixel_y = ph * ps + py;
                            let pixel_x = pw * ps + px;
                            let raw =
                                img.get_pixel(pixel_x as u32, pixel_y as u32)[c] as f32 / 255.0;
                            let norm = (raw - MEAN[c]) / STD[c];
                            // Offset into patch_dim: c*(T*Hp*Wp) + t*(Hp*Wp) + py*Wp + px
                            let off = c * (tp * ps * ps) + t * (ps * ps) + py * ps + px;
                            pixels[patch_idx * patch_dim + off] = norm;
                        }
                    }
                }
            }
        }
    }

    Ok((pixels, grid_h, grid_w))
}

/// Number of image pad tokens produced per image after the vision
/// encoder's spatial merger. Qwen3-VL / Qwen3.6 fold a 2×2 patch block
/// into a single token, so the embedding stream has
/// `(grid_h / sms) * (grid_w / sms)` rows — not `grid_h * grid_w`.
pub fn image_pad_count(grid_h: usize, grid_w: usize, spatial_merge_size: usize) -> usize {
    let sms = spatial_merge_size.max(1);
    (grid_h / sms) * (grid_w / sms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_size_min_pixels_upscale() {
        // OCR floor: a small formula screenshot (264×86 = 22.7k px) is
        // upscaled to ≥ 262144 px → ~290 vision tokens (was 24 pre-fix,
        // 70 at the HF 65536 floor — both read as a single glyph).
        let (h, w) = target_size_with_max_pixels(86, 264, 32, None);
        assert!(h * w >= MIN_PIXELS, "area {h}×{w} below the OCR floor");
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
        let tokens = (h / 32) * (w / 32);
        assert!(
            (250..=400).contains(&tokens),
            "expected a few hundred vision tokens, got {tokens}"
        );
        // Aspect ratio roughly preserved.
        let ar = (w as f32) / (h as f32);
        assert!((ar - 264.0 / 86.0).abs() < 0.7, "aspect drifted: {ar}");
    }

    #[test]
    fn test_target_size_reference_800x1440() {
        // Qwen3-VL reference diagram: 800×1440 → 1125 vision tokens
        // (no rescale, 1 token per 32×32 px).
        let (h, w) = target_size_with_max_pixels(800, 1440, 32, None);
        assert_eq!((h, w), (800, 1440));
        assert_eq!((h / 32) * (w / 32), 1125);
    }

    #[test]
    fn test_target_size_wide_document_passes_natively() {
        // The document-extraction case the old 1280-px long-side cap
        // broke: a 2500×1200 page stays native → ~2964 tokens.
        let (h, w) = target_size_with_max_pixels(1200, 2500, 32, None);
        assert!(h * w >= 2_900_000, "page was shrunk: {h}×{w}");
        assert!((h / 32) * (w / 32) > 1000);
    }

    #[test]
    fn test_target_size_downscale_to_hard_budget() {
        // 2000×3000 = 6M px > 4.19M budget → floor-scaled under it, and
        // the patch count never exceeds the ViT scratch bound.
        let (h, w) = target_size_with_max_pixels(2000, 3000, 32, None);
        assert!(h * w <= HARD_MAX_PIXELS);
        assert!(((h / 16) * (w / 16)) as usize <= VISION_MAX_PATCHES);
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
    }

    #[test]
    fn test_target_size_max_pixels() {
        let (h, w) = target_size_with_max_pixels(1254, 1254, 32, Some(512 * 512));
        assert_eq!((h, w), (512, 512));
    }

    #[test]
    fn test_target_size_extreme_banner_within_patch_budget() {
        // 4000×10 banner: min-floor upscale + budget caps must keep the
        // patch count inside the ViT scratch bound.
        let (h, w) = target_size_with_max_pixels(10, 4000, 32, None);
        assert!(((h / 16) * (w / 16)) as usize <= VISION_MAX_PATCHES);
        assert!(w <= MAX_DIM);
    }

    #[test]
    fn test_image_pad_count_2x2_merge() {
        // Standard Qwen3-VL: 2×2 spatial merger folds a patch block
        // into one embedding token.
        assert_eq!(image_pad_count(64, 64, 2), 32 * 32);
        assert_eq!(image_pad_count(40, 80, 2), 20 * 40);
    }

    #[test]
    fn test_image_pad_count_no_merge() {
        // spatial_merge_size=1 → identity (each patch → one token).
        assert_eq!(image_pad_count(64, 64, 1), 64 * 64);
        assert_eq!(image_pad_count(8, 12, 1), 96);
    }

    #[test]
    fn test_image_pad_count_zero_sms_clamps_to_one() {
        // sms=0 is invalid; clamps to 1 so we never divide by zero.
        assert_eq!(image_pad_count(64, 64, 0), 64 * 64);
    }

    #[test]
    fn test_image_pad_count_non_divisible_floors() {
        // Integer division truncates: 65/2 = 32 (not 33).
        assert_eq!(image_pad_count(65, 64, 2), 32 * 32);
    }
}

// SPDX-License-Identifier: AGPL-3.0-only

//! `POST /v1/images/edits` — image-conditioned generation (Todo 3.2a).
//!
//! OpenAI-compatible multipart/form-data (`image` file parts — up to 10
//! condition images — plus the text fields), with a JSON body accepted as
//! an Atlas extension (`image` as one base64 string or an array of them).
//! `/v1/images/variations` stays 501: the model has no prompt-free
//! variation mode.
//!
//! Per condition image, mirroring the reference pipeline: resize to the
//! ~1 MP 32-aligned size derived from its aspect ratio (Lanczos); the
//! vision-tower copy is the RGBA composited over white; the VAE copy
//! keeps all four channels in [-1, 1]. The output size defaults to the
//! LAST image's derived dimensions. The prompt template interleaves one
//! `<imageK><|vision_start|><|image_pad|><|vision_end|>` group per image,
//! each pad expanded to the image's merged ViT token count.

use std::sync::Arc;

use axum::extract::{FromRequest, Multipart, Request, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::{Json, Response};
use base64::Engine;
use serde::Deserialize;
use spark_model::qwen_image::{self, EditImage, GenParams, QwenImage};

use super::compact::openai_error_response;
use super::images::{
    Background, DEFAULT_STEPS, MAX_PROMPT_TOKENS, MAX_STEPS, Sidecar, bad_request, check_dims,
    dims_for_aspect, encode_image, ensure_free_memory, failure, parse_size, sidecar_with_eviction,
};
use crate::main_modules::AppState;

/// Model-card limit on condition images.
const MAX_COND_IMAGES: usize = 10;
/// Reference `output_resolution`: target area and condition-resize area.
const OUTPUT_RESOLUTION: usize = 1024;
/// First edit request additionally loads the edit stack (ViT + VAE
/// encoder weights + encoder scratch) — preflight headroom for it.
const EDIT_STACK_GIB: f64 = 4.0;

#[derive(Default, Deserialize)]
struct EditFields {
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    model: String,
    n: Option<usize>,
    size: Option<String>,
    aspect_ratio: Option<String>,
    response_format: Option<String>,
    output_format: Option<String>,
    seed: Option<u64>,
    num_inference_steps: Option<usize>,
}

#[derive(Deserialize)]
struct EditJson {
    /// One base64/data-URI image or an array of them.
    image: serde_json::Value,
    #[serde(flatten)]
    fields: EditFields,
}

/// `calculate_dimensions(target_area, ratio)` — the reference's rounding
/// (width from the unrounded value, both snapped to 32).
fn calculate_dimensions(target_area: usize, ratio: f64) -> (usize, usize) {
    let width = (target_area as f64 * ratio).sqrt();
    let height = width / ratio;
    let snap = |v: f64| ((v / 32.0).round() as usize).max(1) * 32;
    (snap(width), snap(height))
}

fn decode_b64_field(field: &str) -> Result<Vec<u8>, String> {
    let b64 = match field.strip_prefix("data:") {
        Some(rest) => rest
            .split_once("base64,")
            .map(|(_, p)| p)
            .ok_or("data: URI without base64 payload")?,
        None => field,
    };
    base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| format!("invalid base64 image: {e}"))
}

async fn parse_multipart(mut mp: Multipart) -> Result<(Vec<Vec<u8>>, EditFields), String> {
    let mut images = Vec::new();
    let mut f = EditFields::default();
    while let Some(field) = mp.next_field().await.map_err(|e| e.to_string())? {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "image" | "image[]" => {
                let bytes = field.bytes().await.map_err(|e| e.to_string())?;
                images.push(bytes.to_vec());
            }
            "prompt" => f.prompt = field.text().await.map_err(|e| e.to_string())?,
            "model" => f.model = field.text().await.map_err(|e| e.to_string())?,
            "n" => f.n = field.text().await.map_err(|e| e.to_string())?.parse().ok(),
            "size" => f.size = Some(field.text().await.map_err(|e| e.to_string())?),
            "aspect_ratio" => {
                f.aspect_ratio = Some(field.text().await.map_err(|e| e.to_string())?);
            }
            "response_format" => {
                f.response_format = Some(field.text().await.map_err(|e| e.to_string())?);
            }
            "output_format" => {
                f.output_format = Some(field.text().await.map_err(|e| e.to_string())?);
            }
            "seed" => f.seed = field.text().await.map_err(|e| e.to_string())?.parse().ok(),
            "num_inference_steps" => {
                f.num_inference_steps = field.text().await.map_err(|e| e.to_string())?.parse().ok();
            }
            // `mask` would select inpainting — not supported by this model.
            "mask" => return Err("'mask' is not supported by Qwen-Image-2.1 edits".into()),
            _ => {}
        }
    }
    Ok((images, f))
}

/// One prepared condition image (host-side, ready for the pipeline).
struct PreparedImage {
    edit: EditImage,
    merged: usize,
}

fn prepare_image(
    bytes: &[u8],
    vcfg: &atlas_core::config::VisionConfig,
) -> Result<PreparedImage, String> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| format!("undecodable condition image: {e}"))?
        .to_rgba8();
    let (w0, h0) = (img.width() as usize, img.height() as usize);
    if w0 == 0 || h0 == 0 {
        return Err("empty condition image".into());
    }
    let (iw, ih) =
        calculate_dimensions(OUTPUT_RESOLUTION * OUTPUT_RESOLUTION, w0 as f64 / h0 as f64);
    let resized = image::imageops::resize(
        &img,
        iw as u32,
        ih as u32,
        image::imageops::FilterType::Lanczos3,
    );

    // Vision-tower copy: alpha composited over white, RGB.
    let mut rgb = image::RgbImage::new(iw as u32, ih as u32);
    for (x, y, p) in resized.enumerate_pixels() {
        let a = p[3] as f32 / 255.0;
        let mix = |c: u8| ((c as f32 * a) + 255.0 * (1.0 - a)).round() as u8;
        rgb.put_pixel(x, y, image::Rgb([mix(p[0]), mix(p[1]), mix(p[2])]));
    }
    let (vit_pixels, grid_h, grid_w) =
        spark_model::vision_preprocess::preprocess_rgb_exact(&rgb, vcfg)
            .map_err(|e| format!("vision preprocess failed: {e:#}"))?;
    if (grid_h * 16, grid_w * 16) != (ih, iw) {
        return Err(format!(
            "vision grid {grid_h}x{grid_w} does not match the {ih}x{iw} resize"
        ));
    }

    // VAE copy: RGBA in [-1, 1], NHWC.
    let mut vae_pixels = Vec::with_capacity(iw * ih * 4);
    for p in resized.pixels() {
        for c in 0..4 {
            vae_pixels.push(p[c] as f32 / 255.0 * 2.0 - 1.0);
        }
    }

    let merged = (grid_h / 2) * (grid_w / 2);
    Ok(PreparedImage {
        edit: EditImage {
            vit_pixels,
            grid_h,
            grid_w,
            vae_pixels,
            vae_height: ih,
            vae_width: iw,
            span: (0, 0), // filled after tokenization
        },
        merged,
    })
}

pub async fn edits(State(state): State<Arc<AppState>>, req: Request) -> Response {
    let content_type = req
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let (raw_images, fields) = if content_type.starts_with("multipart/form-data") {
        match Multipart::from_request(req, &state).await {
            Ok(mp) => match parse_multipart(mp).await {
                Ok(v) => v,
                Err(e) => return bad_request(format!("invalid multipart body: {e}")),
            },
            Err(e) => return bad_request(format!("invalid multipart body: {e}")),
        }
    } else {
        let bytes = match axum::body::to_bytes(req.into_body(), 64 << 20).await {
            Ok(b) => b,
            Err(e) => return bad_request(format!("unreadable body: {e}")),
        };
        let parsed: EditJson = match serde_json::from_slice(&bytes) {
            Ok(p) => p,
            Err(e) => {
                return bad_request(format!(
                    "Invalid request JSON: {e} — expected {{\"image\": <base64 or [base64,…]>, \
                     \"prompt\": <text>}} (or multipart/form-data)"
                ));
            }
        };
        let list = match &parsed.image {
            serde_json::Value::String(s) => vec![s.clone()],
            serde_json::Value::Array(a) => a
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect(),
            _ => return bad_request("'image' must be a base64 string or an array".into()),
        };
        let mut images = Vec::with_capacity(list.len());
        for s in &list {
            match decode_b64_field(s) {
                Ok(b) => images.push(b),
                Err(e) => return bad_request(e),
            }
        }
        (images, parsed.fields)
    };

    if raw_images.is_empty() {
        return bad_request("at least one condition 'image' is required".into());
    }
    if raw_images.len() > MAX_COND_IMAGES {
        return bad_request(format!(
            "too many condition images: {} (limit {MAX_COND_IMAGES})",
            raw_images.len()
        ));
    }
    let model_id =
        match super::images::resolve_image_model(&state, &fields.model, "/v1/images/edits") {
            Ok(id) => id,
            Err(resp) => return *resp,
        };

    let n = fields.n.unwrap_or(1);
    if !(1..=super::images::MAX_IMAGES).contains(&n) {
        return bad_request(format!(
            "'n' must be between 1 and {}",
            super::images::MAX_IMAGES
        ));
    }
    let steps = fields.num_inference_steps.unwrap_or(DEFAULT_STEPS);
    if !(1..=MAX_STEPS).contains(&steps) {
        return bad_request(format!(
            "'num_inference_steps' must be between 1 and {MAX_STEPS}"
        ));
    }
    let response_format = fields.response_format.as_deref().unwrap_or("b64_json");
    if !matches!(response_format, "b64_json" | "url") {
        return bad_request(format!(
            "response_format '{response_format}' not supported (b64_json|url)"
        ));
    }
    let output_format = match fields.output_format.as_deref().unwrap_or("png") {
        "jpg" => "jpeg",
        f @ ("png" | "jpeg" | "webp") => f,
        other => {
            return bad_request(format!(
                "output_format '{other}' not supported (png|jpeg|webp)"
            ));
        }
    }
    .to_string();

    // Prepare the condition images (CPU-side) before touching the GPU.
    let vcfg = atlas_core::config::VisionConfig {
        depth: 27,
        hidden_size: 1152,
        num_heads: 16,
        patch_size: 16,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        intermediate_size: 4304,
        out_hidden_size: 4096,
        deepstack_visual_indexes: vec![8, 16, 24],
        image_pad_token_id: 0,
    };
    let mut prepared = Vec::with_capacity(raw_images.len());
    for (i, bytes) in raw_images.iter().enumerate() {
        match prepare_image(bytes, &vcfg) {
            Ok(p) => prepared.push(p),
            Err(e) => return bad_request(format!("condition image {}: {e}", i + 1)),
        }
    }

    // Output size: explicit size/aspect, else derived from the LAST image.
    let dims = match (fields.size.as_deref(), fields.aspect_ratio.as_deref()) {
        (Some(s), _) if s != "auto" => parse_size(s),
        (_, Some(a)) => dims_for_aspect(a),
        _ => {
            let last = &prepared[prepared.len() - 1].edit;
            check_dims(last.vae_width, last.vae_height)
        }
    };
    let (width, height) = match dims {
        Ok(d) => d,
        Err(e) => return bad_request(e),
    };

    let sc: Arc<Sidecar> = match sidecar_with_eviction(&state, &model_id).await {
        Ok(s) => s,
        Err(e) => return failure("image sidecar init failed", &e),
    };
    if sc.model_id != model_id {
        return bad_request(format!(
            "image model '{}' is resident; only one image pipeline can be loaded per server",
            sc.model_id
        ));
    }

    // Tokenize the ti2i template and expand each single `<|image_pad|>`
    // to the image's merged ViT token count, recording the spans.
    let template = qwen_image::ti2i_prompt(&fields.prompt, prepared.len());
    let base_tokens: Vec<u32> = match sc.tokenizer.encode(template, false) {
        Ok(enc) => enc.get_ids().to_vec(),
        Err(e) => return bad_request(format!("tokenizing the prompt failed: {e}")),
    };
    let Some(pad_id) = sc.tokenizer.token_to_id("<|image_pad|>") else {
        return failure(
            "tokenizer is missing <|image_pad|>",
            &anyhow::anyhow!("bad tokenizer"),
        );
    };
    let mut tokens = Vec::with_capacity(
        base_tokens.len() + prepared.iter().map(|p| p.merged - 1).sum::<usize>(),
    );
    let mut img_idx = 0usize;
    for &t in &base_tokens {
        if t == pad_id {
            if img_idx >= prepared.len() {
                return bad_request("more <|image_pad|> tokens than images".into());
            }
            let merged = prepared[img_idx].merged;
            prepared[img_idx].edit.span = (tokens.len(), merged);
            tokens.extend(std::iter::repeat_n(pad_id, merged));
            img_idx += 1;
        } else {
            tokens.push(t);
        }
    }
    if img_idx != prepared.len() {
        return failure(
            "template/image mismatch",
            &anyhow::anyhow!("{img_idx} pads for {} images", prepared.len()),
        );
    }
    let text_tokens = tokens.len() - prepared.iter().map(|p| p.merged).sum::<usize>();
    if text_tokens.saturating_sub(sc.sys_tokens) > MAX_PROMPT_TOKENS {
        return bad_request(format!(
            "prompt is too long: {} text tokens (limit {MAX_PROMPT_TOKENS})",
            text_tokens - sc.sys_tokens
        ));
    }
    let base_seed = fields.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
    });

    let sc2 = sc.clone();
    let fmt = output_format.clone();
    let edit_images: Vec<EditImage> = prepared.into_iter().map(|p| p.edit).collect();
    let work = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<(u64, Vec<u8>)>> {
        let mut pipeline = sc2.pipeline.blocking_lock();
        pipeline.gpu_backend().bind_to_thread()?;
        let _untracked = spark_runtime::cuda_backend::UntrackedAllocs::enter();
        // The joint sequence adds 4 latent tokens per VLM image slot; the
        // prefix K/V and score rows scale with it.
        let cond_tokens: usize = edit_images.iter().map(|i| i.span.1 * 4).sum();
        let stack_overhead = if pipeline.edit_ready() {
            0
        } else {
            super::images::gib_bytes(EDIT_STACK_GIB)
        };
        ensure_free_memory(
            pipeline.gpu_backend(),
            QwenImage::working_bytes(width, height, tokens.len() + 3 * cond_tokens / 4)
                + stack_overhead,
            "for this edit",
        )?;
        pipeline.ensure_edit_stack()?;
        let pipeline = &*pipeline;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let seed = base_seed.wrapping_add(i as u64);
            let t0 = std::time::Instant::now();
            let params = GenParams {
                width,
                height,
                steps,
                seed,
            };
            let img = pipeline.generate_edit(
                &tokens,
                sc2.sys_tokens,
                &edit_images,
                &params,
                |done, total| {
                    if done % 10 == 0 || done == total {
                        tracing::debug!("image edit: step {done}/{total}");
                    }
                },
            )?;
            tracing::info!(
                "Image edited: {}x{} {steps} steps seed {seed} in {:.1}s \
                 ({} text + {cond_tokens} condition tokens)",
                img.width,
                img.height,
                t0.elapsed().as_secs_f64(),
                text_tokens,
            );
            out.push((seed, encode_image(&img, &fmt, Background::Opaque)?));
        }
        Ok(out)
    })
    .await;
    let images = match work {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return failure("image edit failed", &e),
        Err(e) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("image edit task failed: {e}"),
            );
        }
    };

    let b64 = base64::engine::general_purpose::STANDARD;
    let data: Vec<serde_json::Value> = images
        .iter()
        .map(|(seed, bytes)| {
            let encoded = b64.encode(bytes);
            if response_format == "url" {
                serde_json::json!({
                    "url": format!("data:image/{output_format};base64,{encoded}"),
                    "seed": seed,
                })
            } else {
                serde_json::json!({ "b64_json": encoded, "seed": seed })
            }
        })
        .collect();
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    Json(serde_json::json!({
        "created": created,
        "model": model_id,
        "size": format!("{width}x{height}"),
        "output_format": output_format,
        "data": data,
    }))
    .into_response()
}

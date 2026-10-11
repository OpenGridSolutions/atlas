// SPDX-License-Identifier: AGPL-3.0-only

//! `POST /v1/images/generations` — Qwen-Image-2.1 text-to-image (Todo 3.2a).
//! Replaces the 501 stub for servers that list an image model in
//! `ATLAS_PRESEED_IMAGE`.
//!
//! The pipeline runs as a RESIDENT SIDECAR (the Whisper pattern): loaded
//! once on first request via its own kernel target + backend, never
//! entering the scheduler / hot-swap arena — generating an image does not
//! evict the active chat model. One generation runs at a time; further
//! requests queue on the sidecar lock.
//!
//! OpenAI-Images-compatible request: `prompt`, `model`, `n`, `size`
//! (`"WxH"` or `"auto"`), `response_format` (`b64_json` | `url` — the URL is
//! a `data:` URI, Atlas hosts no files), `output_format` (`png` | `jpeg` |
//! `webp`), `background` (`auto` | `opaque` | `transparent`). Extensions:
//! `seed`, `num_inference_steps`, `aspect_ratio` (`"16:9"`, used when no
//! `size` is given).

use std::sync::Arc;

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use base64::Engine;
use serde::Deserialize;

use crate::main_modules::app_state::AppState;
use spark_model::qwen_image::{self, GenParams, GeneratedImage, QwenImage};

use super::compact::openai_error_response;

pub(super) const DEFAULT_SIDE: usize = 1024;
const MIN_SIDE: usize = 256;
/// 2048×2048 / 2752×1536 class — the model card's largest presets.
const MAX_PIXELS: usize = 2752 * 1536;
pub(super) const DEFAULT_STEPS: usize = 40;
pub(super) const MAX_STEPS: usize = 100;
pub(super) const MAX_IMAGES: usize = 4;
pub(super) const MAX_PROMPT_TOKENS: usize = 2048;

/// Resident weights of the pipeline (DiT + text tower + VAE decoder), GiB.
const WEIGHTS_RESERVE_GIB: f64 = 27.0;
/// Working memory of one generation at the largest size (attention scores,
/// MLP activations, VAE feature maps), GiB.
const WORKING_RESERVE_GIB: f64 = 7.0;
/// Free memory that must remain after the sidecar took its share — other
/// users of the unified memory (the chat model's lazily touched KV pool,
/// page cache, co-tenants) keep allocating.
const FREE_MARGIN_GIB: f64 = 4.0;

#[derive(Deserialize)]
pub struct ImageRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    prompt: String,
    n: Option<usize>,
    size: Option<String>,
    aspect_ratio: Option<String>,
    response_format: Option<String>,
    output_format: Option<String>,
    background: Option<String>,
    seed: Option<u64>,
    #[serde(alias = "steps")]
    num_inference_steps: Option<usize>,
    /// Atlas extension, see `api::admission`.
    #[serde(default)]
    accepted_processing_time: Option<f64>,
}

pub(super) struct Sidecar {
    pub(super) model_id: String,
    pub(super) pipeline: tokio::sync::Mutex<QwenImage>,
    pub(super) tokenizer: tokenizers::Tokenizer,
    /// Token count of the system prefix dropped from the encoder output.
    pub(super) sys_tokens: usize,
}

static SIDECAR: tokio::sync::OnceCell<Arc<Sidecar>> = tokio::sync::OnceCell::const_new();

/// Resident device weight bytes, set once at load. 0 = not loaded.
static SIDECAR_WEIGHTS_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// /v1/stats residency probe: `Some(weights_bytes)` once the image sidecar
/// is resident, `None` before its first request.
pub fn sidecar_stats() -> Option<usize> {
    match SIDECAR_WEIGHTS_BYTES.load(std::sync::atomic::Ordering::Relaxed) {
        0 => None,
        b => Some(b),
    }
}

pub(super) fn gib_bytes(gib: f64) -> usize {
    (gib * (1u64 << 30) as f64) as usize
}

/// Refuse work the unified memory cannot hold instead of driving the host
/// into the OOM killer: `need` plus a safety margin must be free right now.
pub(super) fn ensure_free_memory(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    need: usize,
    what: &str,
) -> anyhow::Result<()> {
    // MemAvailable without swap: device memory is pinned and cannot be
    // swapped out, so swap space is no room for it.
    let free = match spark_runtime::meminfo::read() {
        Some(m) => m.mem_available,
        None => gpu.free_memory()?,
    };
    let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
    anyhow::ensure!(
        free >= need + gib_bytes(FREE_MARGIN_GIB),
        "not enough free memory {what}: need {:.1} GiB + {FREE_MARGIN_GIB:.0} GiB margin, \
         {:.1} GiB free",
        gib(need),
        gib(free),
    );
    Ok(())
}

/// Best-effort eviction BEFORE a generation's authoritative memory gate:
/// when `need` + margin exceeds `MemAvailable` and an arena model is
/// resident, unload it — it lazy-reloads on its next request. Large
/// sizes (2048² needs ~7.4 GiB working memory, VAE decode dominating)
/// otherwise 503 whenever a chat model's KV pool sits on the box, which
/// is exactly the "swap models automatically" case the sidecar load
/// already handles. Unlike the cross-process driver-pool situation,
/// MemAvailable reflects an in-process unload immediately, so the
/// ensure_free_memory gate inside the worker re-checks honestly.
pub(super) async fn maybe_evict_for(state: &AppState, need: usize, what: &str) {
    let Some(ctx) = state.hotswap.as_ref() else {
        return;
    };
    let Some(free) = spark_runtime::meminfo::read().map(|m| m.mem_available) else {
        return;
    };
    if free >= need + gib_bytes(FREE_MARGIN_GIB) {
        return;
    }
    let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
    tracing::warn!(
        "{what} needs {:.1} GiB + {FREE_MARGIN_GIB:.0} GiB margin but {:.1} GiB is available — \
         evicting the arena model to make room (it lazy-reloads on its next request)",
        gib(need),
        gib(free),
    );
    match super::lazy_swap::unload_active(ctx).await {
        Ok(_) => {}
        Err(e) => tracing::warn!("arena eviction for {what} failed: {e:#}"),
    }
}

/// Memory the KV-budget sizing must leave free while the sidecar is
/// configured but not resident (`ATLAS_IMAGE_RESERVE_GB` overrides the
/// weights + working-memory default).
pub fn pending_reserve_bytes() -> usize {
    let gib = std::env::var("ATLAS_IMAGE_RESERVE_GB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|g| *g >= 0.0)
        .unwrap_or(WEIGHTS_RESERVE_GIB + WORKING_RESERVE_GIB);
    gib_bytes(gib)
}

/// Load the sidecar for the handlers. When the memory preflight refuses
/// the load and an arena model is resident, evict it — the lazy-swap
/// machinery reloads it on its next request, by then with the small
/// post-load reserve — and retry ONCE without the preflight: the GB10
/// driver pool hands freshly freed allocations to new CUDA allocations
/// immediately while `MemAvailable` lags minutes behind, so a re-check
/// would false-negative. The loader's per-shard OOM guard remains as the
/// safety net.
pub(super) async fn sidecar_with_eviction(
    state: &AppState,
    model_id: &str,
) -> anyhow::Result<Arc<Sidecar>> {
    let first = sidecar(model_id, false).await;
    let Err(e) = first else {
        return first;
    };
    let memory_bound = format!("{e:#}").contains("not enough free memory");
    let Some(ctx) = state.hotswap.as_ref().filter(|_| memory_bound) else {
        return Err(e);
    };
    tracing::warn!(
        "image sidecar load refused by the memory preflight ({e:#}); evicting arena \
         model '{}' to make room (it lazy-reloads on its next request)",
        ctx.active_model().model_id,
    );
    if !super::lazy_swap::unload_active(ctx).await? {
        // Arena was already empty — eviction cannot free anything, the
        // preflight verdict stands.
        return Err(e);
    }
    sidecar(model_id, true).await
}

pub(super) async fn sidecar(model_id: &str, skip_preflight: bool) -> anyhow::Result<Arc<Sidecar>> {
    let id = model_id.to_string();
    SIDECAR
        .get_or_try_init(|| async {
            tokio::task::spawn_blocking(move || -> anyhow::Result<Arc<Sidecar>> {
                let t0 = std::time::Instant::now();
                // Sidecar allocations must never land in a hot-swap build's
                // watermark set (they would be freed at that model's unload).
                let _untracked = spark_runtime::cuda_backend::UntrackedAllocs::enter();
                let model_dir = crate::model_resolver::resolve_diffusers_dir(&id, None)?;
                let set = atlas_kernels::ptx_for_model_id(&id, "qwen_image", 4096)
                    .filter(|t| t.target.model == "qwen-image-2.1")
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "no compiled kernel target for image model '{id}' \
                             (supported: Qwen/Qwen-Image-2.1)"
                        )
                    })?;
                let key = format!(
                    "{}/{}/{}",
                    set.target.arch, set.target.model, set.target.quant
                );
                let mut backend = spark_runtime::cuda_backend::AtlasCudaBackend::new_for_target(
                    &key,
                    &set.modules,
                )?;
                // Fairness: the sidecar renders on its OWN stream so a
                // 40-step generation no longer FIFO-starves chat decode
                // launches on the shared registry stream (measured 112 s
                // for a 4.8 s chat reply before this).
                backend.use_private_stream()?;
                use spark_runtime::gpu::GpuBackend;
                backend.bind_to_thread()?;
                if skip_preflight {
                    tracing::info!(
                        "image sidecar: skipping the MemAvailable preflight after arena \
                         eviction (freed allocations are CUDA-reusable before MemAvailable \
                         reflects them)"
                    );
                } else {
                    ensure_free_memory(
                        &backend,
                        gib_bytes(WEIGHTS_RESERVE_GIB)
                            + QwenImage::working_bytes(DEFAULT_SIDE, DEFAULT_SIDE, 0),
                        "to load the image pipeline",
                    )?;
                }
                let pipeline = QwenImage::load(&model_dir, Box::new(backend))?;
                SIDECAR_WEIGHTS_BYTES.store(
                    pipeline.weights_bytes(),
                    std::sync::atomic::Ordering::Relaxed,
                );
                // The weights are now visible to the KV budget as Atlas-own
                // memory; only the per-generation working set stays reserved.
                let pending = pending_reserve_bytes();
                spark_runtime::gpu::set_sidecar_pending_reserve_bytes(
                    pending.min(gib_bytes(WORKING_RESERVE_GIB)),
                );

                let tokenizer = tokenizers::Tokenizer::from_file(
                    model_dir.join("processor").join("tokenizer.json"),
                )
                .map_err(|e| anyhow::anyhow!("image tokenizer: {e}"))?;
                let sys_tokens = tokenizer
                    .encode(qwen_image::system_prefix(), false)
                    .map_err(|e| anyhow::anyhow!("image tokenizer: {e}"))?
                    .len();
                tracing::info!(
                    "Qwen-Image sidecar ready in {:.1}s ({:.1} GiB resident)",
                    t0.elapsed().as_secs_f64(),
                    pipeline.weights_bytes() as f64 / (1u64 << 30) as f64,
                );
                Ok(Arc::new(Sidecar {
                    model_id: id,
                    pipeline: tokio::sync::Mutex::new(pipeline),
                    tokenizer,
                    sys_tokens,
                }))
            })
            .await?
        })
        .await
        .cloned()
}

/// `"WxH"` → (width, height), snapped down to the 32 px latent grid.
pub(super) fn parse_size(size: &str) -> Result<(usize, usize), String> {
    let (w, h) = size
        .split_once(['x', 'X'])
        .and_then(|(w, h)| {
            Some((
                w.trim().parse::<usize>().ok()?,
                h.trim().parse::<usize>().ok()?,
            ))
        })
        .ok_or_else(|| format!("size '{size}' is not of the form \"WIDTHxHEIGHT\""))?;
    check_dims(QwenImage::snap(w).0, QwenImage::snap(h).0)
}

/// `"W:H"` → dimensions with the default 1024² pixel area, snapped to 32 px
/// (the reference pipeline's `calculate_dimensions`).
pub(super) fn dims_for_aspect(aspect: &str) -> Result<(usize, usize), String> {
    let (a, b) = aspect
        .split_once(':')
        .and_then(|(a, b)| Some((a.trim().parse::<f64>().ok()?, b.trim().parse::<f64>().ok()?)))
        .filter(|(a, b)| *a > 0.0 && *b > 0.0)
        .ok_or_else(|| format!("aspect_ratio '{aspect}' is not of the form \"W:H\""))?;
    let ratio = a / b;
    let width = ((DEFAULT_SIDE * DEFAULT_SIDE) as f64 * ratio).sqrt();
    let snap = |v: f64| ((v / 32.0).round() as usize).max(1) * 32;
    check_dims(snap(width), snap(width / ratio))
}

pub(super) fn check_dims(w: usize, h: usize) -> Result<(usize, usize), String> {
    if w < MIN_SIDE || h < MIN_SIDE {
        return Err(format!(
            "image side below the {MIN_SIDE} px minimum ({w}x{h})"
        ));
    }
    if w * h > MAX_PIXELS {
        return Err(format!(
            "{w}x{h} exceeds the {MAX_PIXELS} pixel limit (largest presets: 2048x2048, 2752x1536)"
        ));
    }
    Ok((w, h))
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Background {
    Auto,
    Opaque,
    Transparent,
}

/// Encode the pipeline's RGBA output. The VAE always produces an alpha
/// channel; it is kept for `transparent`, dropped for `opaque`, and under
/// `auto` kept only when the image actually is transparent somewhere.
pub(super) fn encode_image(
    img: &GeneratedImage,
    format: &str,
    bg: Background,
) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(
        img.channels == 4,
        "expected RGBA output, got {} channels",
        img.channels
    );
    let px = img.width * img.height;
    let keep_alpha = match bg {
        Background::Transparent => true,
        Background::Opaque => false,
        Background::Auto => {
            let clear = img.pixels.chunks_exact(4).filter(|p| p[3] < 128).count();
            clear * 100 >= px
        }
    } && format != "jpeg";
    let fmt = match format {
        "png" => image::ImageFormat::Png,
        "jpeg" => image::ImageFormat::Jpeg,
        "webp" => image::ImageFormat::WebP,
        other => anyhow::bail!("unsupported output_format '{other}'"),
    };
    let (w, h) = (img.width as u32, img.height as u32);
    let mut out = std::io::Cursor::new(Vec::new());
    if keep_alpha {
        image::RgbaImage::from_raw(w, h, img.pixels.clone())
            .ok_or_else(|| anyhow::anyhow!("pixel buffer size mismatch"))?
            .write_to(&mut out, fmt)?;
    } else {
        let rgb: Vec<u8> = img
            .pixels
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect();
        image::RgbImage::from_raw(w, h, rgb)
            .ok_or_else(|| anyhow::anyhow!("pixel buffer size mismatch"))?
            .write_to(&mut out, fmt)?;
    }
    Ok(out.into_inner())
}

/// The model card's prompt format for transparent (RGBA) generation.
fn transparent_prompt(prompt: &str) -> String {
    if prompt.contains("RGBA") {
        return prompt.to_string();
    }
    format!(
        "This is an RGBA image with transparency. {prompt} The image has alpha channel and \
         the background is transparent."
    )
}

pub(super) fn bad_request(msg: String) -> Response {
    openai_error_response(StatusCode::BAD_REQUEST, msg)
}

/// Resolve the request's model id against the image registrations.
/// An explicitly requested model that is unknown on this server answers
/// the canonical 404 `model_not_found` (the shape of the chat endpoint's
/// lazy-swap gate) — also when NO image model is configured at all; a
/// known model of another category gets the 400 category hint. The 501
/// "not enabled" hint is reserved for requests that leave `model` empty
/// on a server without an image model, where 404 would mislead.
pub(super) fn resolve_image_model(
    state: &AppState,
    requested: &str,
    endpoint: &str,
) -> Result<String, Box<Response>> {
    if requested.is_empty() {
        return match state.image_models.first() {
            Some(m) => Ok(m.clone()),
            None => Err(Box::new(openai_error_response(
                StatusCode::NOT_IMPLEMENTED,
                "Image API is not enabled: no image model configured — add one to \
                 ATLAS_PRESEED_IMAGE (supported: Qwen/Qwen-Image-2.1)"
                    .into(),
            ))),
        };
    }
    if state.image_models.iter().any(|m| m == requested) {
        return Ok(requested.to_string());
    }
    if let Some(reg) = state
        .hotswap
        .as_ref()
        .and_then(|ctx| ctx.registered(requested).cloned())
    {
        return Err(Box::new(bad_request(format!(
            "model '{requested}' is categorized as {} and not available on \
             {endpoint}; use {} instead",
            reg.category.as_str(),
            reg.category.endpoint(),
        ))));
    }
    let available: Vec<&str> = state.image_models.iter().map(String::as_str).collect();
    let body = serde_json::json!({
        "error": {
            "message": format!(
                "model '{requested}' is not available on {endpoint}; configured \
                 image models: [{}]",
                available.join(", ")
            ),
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found",
        },
        "available_models": available,
    });
    Err(Box::new(
        (StatusCode::NOT_FOUND, Json(body)).into_response(),
    ))
}

/// A failed load / generation: 503 when the memory preflight refused it
/// (retryable once memory is free), 500 otherwise.
pub(super) fn failure(what: &str, e: &anyhow::Error) -> Response {
    let msg = format!("{what}: {e:#}");
    let status = if msg.contains("not enough free memory") {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    };
    openai_error_response(status, msg)
}

pub async fn generations(
    State(state): State<Arc<AppState>>,
    req: Result<Json<ImageRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return bad_request(format!(
                "Invalid request JSON: {e} — expected {{\"prompt\": <text>, \"size\"?: \"1024x1024\"}}"
            ));
        }
    };
    let model_id = match resolve_image_model(&state, &req.model, "/v1/images/generations") {
        Ok(id) => id,
        Err(resp) => return *resp,
    };

    if req.prompt.trim().is_empty() {
        return bad_request("'prompt' is required".into());
    }
    let _admission = match super::admission::admit(
        &state,
        &model_id,
        crate::arena_sched::WorkProfile {
            prompt_tokens: 0,
            output_tokens: 0,
            pages: 0,
        },
        req.accepted_processing_time,
    ) {
        Ok(a) => a,
        Err(r) => return super::admission::openai_refusal(&model_id, r),
    };
    let n = req.n.unwrap_or(1);
    if !(1..=MAX_IMAGES).contains(&n) {
        return bad_request(format!("'n' must be between 1 and {MAX_IMAGES}"));
    }
    let steps = req.num_inference_steps.unwrap_or(DEFAULT_STEPS);
    if !(1..=MAX_STEPS).contains(&steps) {
        return bad_request(format!(
            "'num_inference_steps' must be between 1 and {MAX_STEPS}"
        ));
    }
    let response_format = req.response_format.as_deref().unwrap_or("b64_json");
    if !matches!(response_format, "b64_json" | "url") {
        return bad_request(format!(
            "response_format '{response_format}' not supported (b64_json|url)"
        ));
    }
    let output_format = match req.output_format.as_deref().unwrap_or("png") {
        "jpg" => "jpeg",
        f @ ("png" | "jpeg" | "webp") => f,
        other => {
            return bad_request(format!(
                "output_format '{other}' not supported (png|jpeg|webp)"
            ));
        }
    }
    .to_string();
    let background = match req.background.as_deref().unwrap_or("auto") {
        "auto" => Background::Auto,
        "opaque" => Background::Opaque,
        "transparent" => Background::Transparent,
        other => {
            return bad_request(format!(
                "background '{other}' not supported (auto|opaque|transparent)"
            ));
        }
    };
    if background == Background::Transparent && output_format == "jpeg" {
        return bad_request("background 'transparent' needs output_format png or webp".into());
    }
    let dims = match (req.size.as_deref(), req.aspect_ratio.as_deref()) {
        (Some(s), _) if s != "auto" => parse_size(s),
        (_, Some(a)) => dims_for_aspect(a),
        _ => Ok((DEFAULT_SIDE, DEFAULT_SIDE)),
    };
    let (width, height) = match dims {
        Ok(d) => d,
        Err(e) => return bad_request(e),
    };

    let sc = match sidecar_with_eviction(&state, &model_id).await {
        Ok(s) => s,
        Err(e) => return failure("image sidecar init failed", &e),
    };
    if sc.model_id != model_id {
        return bad_request(format!(
            "image model '{}' is resident; only one image pipeline can be loaded per server",
            sc.model_id
        ));
    }

    let prompt = if background == Background::Transparent {
        transparent_prompt(&req.prompt)
    } else {
        req.prompt.clone()
    };
    let tokens: Vec<u32> = match sc.tokenizer.encode(qwen_image::t2i_prompt(&prompt), false) {
        Ok(enc) => enc.get_ids().to_vec(),
        Err(e) => return bad_request(format!("tokenizing the prompt failed: {e}")),
    };
    if tokens.len() - sc.sys_tokens.min(tokens.len()) > MAX_PROMPT_TOKENS {
        return bad_request(format!(
            "prompt is too long: {} tokens (limit {MAX_PROMPT_TOKENS})",
            tokens.len() - sc.sys_tokens
        ));
    }
    let base_seed = req.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
    });

    maybe_evict_for(
        &state,
        QwenImage::working_bytes(width, height, tokens.len()),
        "this image size",
    )
    .await;

    let sc2 = sc.clone();
    let fmt = output_format.clone();
    let work = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<(u64, Vec<u8>)>> {
        let pipeline = sc2.pipeline.blocking_lock();
        pipeline.gpu_backend().bind_to_thread()?;
        let _untracked = spark_runtime::cuda_backend::UntrackedAllocs::enter();
        ensure_free_memory(
            pipeline.gpu_backend(),
            QwenImage::working_bytes(width, height, tokens.len()),
            "for this image size",
        )?;
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
            let img = pipeline.generate(&tokens, sc2.sys_tokens, &params, |done, total| {
                if done % 10 == 0 || done == total {
                    tracing::debug!("image: step {done}/{total}");
                }
            })?;
            tracing::info!(
                "Image generated: {}x{} {steps} steps seed {seed} in {:.1}s ({} prompt tokens)",
                img.width,
                img.height,
                t0.elapsed().as_secs_f64(),
                tokens.len() - sc2.sys_tokens,
            );
            out.push((seed, encode_image(&img, &fmt, background)?));
        }
        Ok(out)
    })
    .await;
    let images = match work {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return failure("image generation failed", &e),
        Err(e) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("image generation task failed: {e}"),
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

/// `POST /v1/images/edits` and `/v1/images/variations` — image-conditioned
/// generation is not implemented yet (needs the Qwen3-VL vision tower in
/// the text encoder and the VAE encoder).
pub async fn edits_unsupported() -> Response {
    openai_error_response(
        StatusCode::NOT_IMPLEMENTED,
        "Image editing / variations are not supported yet; Atlas serves text-to-image on \
         /v1/images/generations only."
            .into(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_is_snapped_to_the_latent_grid_and_bounded() {
        assert_eq!(parse_size("1024x1024"), Ok((1024, 1024)));
        assert_eq!(parse_size("1000x700"), Ok((992, 672)));
        assert_eq!(parse_size("2752x1536"), Ok((2752, 1536)));
        assert!(parse_size("4096x4096").is_err());
        assert!(parse_size("128x128").is_err());
        assert!(parse_size("big").is_err());
    }

    #[test]
    fn aspect_ratio_keeps_the_default_pixel_area() {
        assert_eq!(dims_for_aspect("1:1"), Ok((1024, 1024)));
        // calculate_dimensions(1024², 16/9): 1365.3 → 1376, 768
        assert_eq!(dims_for_aspect("16:9"), Ok((1376, 768)));
        assert_eq!(dims_for_aspect("9:16"), Ok((768, 1376)));
        assert!(dims_for_aspect("0:1").is_err());
        assert!(dims_for_aspect("wide").is_err());
    }

    #[test]
    fn alpha_is_dropped_unless_the_image_is_transparent() {
        let opaque = GeneratedImage {
            width: 2,
            height: 2,
            channels: 4,
            pixels: vec![255; 16],
        };
        let rgb = encode_image(&opaque, "png", Background::Auto).unwrap();
        assert_eq!(
            image::load_from_memory(&rgb).unwrap().color(),
            image::ColorType::Rgb8
        );
        let rgba = encode_image(&opaque, "png", Background::Transparent).unwrap();
        assert_eq!(
            image::load_from_memory(&rgba).unwrap().color(),
            image::ColorType::Rgba8
        );

        let mut clear = opaque.pixels.clone();
        clear[3] = 0;
        let clear = GeneratedImage {
            width: 2,
            height: 2,
            channels: 4,
            pixels: clear,
        };
        let auto = encode_image(&clear, "png", Background::Auto).unwrap();
        assert_eq!(
            image::load_from_memory(&auto).unwrap().color(),
            image::ColorType::Rgba8
        );
        let forced = encode_image(&clear, "png", Background::Opaque).unwrap();
        assert_eq!(
            image::load_from_memory(&forced).unwrap().color(),
            image::ColorType::Rgb8
        );
    }

    #[test]
    fn transparent_prompt_uses_the_model_card_format_once() {
        let p = transparent_prompt("A dragon sticker.");
        assert!(p.starts_with("This is an RGBA image with transparency. A dragon sticker."));
        assert_eq!(transparent_prompt(&p), p);
    }
}

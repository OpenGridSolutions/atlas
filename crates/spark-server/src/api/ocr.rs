// SPDX-License-Identifier: AGPL-3.0-only

//! `POST /v1/ocr` — document OCR via Chandra-2 (Todo 3.3b/c).
//!
//! Single-model category endpoint (like /v1/translate): the request is
//! served by the one `ATLAS_PRESEED_OCR` registration, category-locked
//! so no other model can be driven here and the OCR model answers 400
//! everywhere else.
//!
//! Pipeline per page (mirrors the Chandra reference):
//!   1. decode → RGB, min-dim Lanczos upscale (input.rs::load_image)
//!   2. scale_to_fit into the training pixel budget (model/util.py)
//!   3. checkpoint chat template over one [image, prompt] user turn +
//!      image-pad expansion (the same frame vLLM renders upstream)
//!   4. blocking inference at temperature 0.0 / top_p 0.1; on a
//!      degenerate tail (detect_repeat_token) retry at
//!      temperature+0.2·n (cap 0.8), top_p 0.95 — up to `max_retries`
//!   5. output.rs: layout chunks (pixel bboxes on the step-1 image),
//!      filtered HTML, WebP bbox crops; markdown.rs: HTML→Markdown
//!
//! PDF input (pdfium rendering + `page_range`) is the remaining 3.3c
//! piece — it needs libpdfium in the gb10 image.

use std::sync::Arc;

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use base64::Engine;
use serde::{Deserialize, Serialize};

use super::compact::openai_error_response;
use super::inference_types::InferenceRequest;
use crate::main_modules::AppState;
use crate::main_modules::hotswap::ModelCategory;
use crate::ocr;

#[derive(Deserialize)]
pub struct OcrRequest {
    /// Document as a `data:` URI or raw base64 — an image (PNG/JPEG/
    /// WebP) or a PDF (sniffed via the %PDF- magic; pages are rendered
    /// through pdfium exactly like the reference pipeline).
    pub image: String,
    /// PDF only: 0-based page selection, reference syntax ("0,2-5,9").
    /// Absent/empty = all pages.
    #[serde(default)]
    pub page_range: Option<String>,
    /// `ocr_layout` (default) or `ocr`.
    #[serde(default)]
    pub prompt_type: Option<String>,
    /// Optional: must name the OCR-category model when set. Empty = the
    /// (single) registered OCR model.
    #[serde(default)]
    pub model: String,
    /// Cap on generated tokens (default 12384, the Chandra reference
    /// MAX_OUTPUT_TOKENS; clamped to context).
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Degeneration retry budget (default 6 = reference
    /// MAX_VLLM_RETRIES; 0 disables retries).
    #[serde(default)]
    pub max_retries: Option<usize>,
    /// Keep Page-Header/Page-Footer blocks in html/markdown (default
    /// false, like the reference).
    #[serde(default)]
    pub include_headers_footers: bool,
    /// Include Image/Figure blocks (default true).
    #[serde(default = "default_true")]
    pub include_images: bool,
}

fn default_true() -> bool {
    true
}

/// One OCR'd page — matches what the Python lib produces per page:
/// raw layout HTML, filtered+converted markdown, labeled bbox chunks,
/// and WebP crops of Image/Figure blocks (base64, keyed by the
/// reference naming scheme).
#[derive(Serialize)]
pub struct OcrPage {
    pub page: usize,
    pub html: String,
    pub markdown: String,
    pub chunks: Vec<ocr::output::LayoutBlock>,
    pub images: std::collections::BTreeMap<String, String>,
    /// Degeneration retries this page needed (0 = clean first pass).
    pub retries: usize,
}

#[derive(Serialize)]
pub struct OcrUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Serialize)]
pub struct OcrResponse {
    pub model: String,
    pub pages: Vec<OcrPage>,
    pub usage: OcrUsage,
}

/// One blocking generation pass at the given sampling point.
async fn generate_once(
    state: &Arc<AppState>,
    prompt_tokens: &Arc<Vec<u32>>,
    image_pixels: &[(Vec<f32>, usize, usize)],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> Result<Vec<u32>, Response> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let session_hash = crate::session_manager::compute_session_hash(prompt_tokens);
    let request = InferenceRequest::Blocking {
        prompt_tokens: Arc::clone(prompt_tokens),
        session_hash,
        image_pixels: image_pixels.to_vec(),
        max_tokens,
        min_tokens: 0,
        temperature,
        top_k: 0,
        top_p,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        lz_penalty: 0.0,
        logit_bias: Vec::new(),
        stop_tokens: Vec::new(),
        enable_thinking: false,
        thinking_budget: None,
        repetition_detection: None,
        require_tool_call: false,
        tools_present: false,
        suppress_tool_call: false,
        disable_mtp: false,
        grammar_spec: None,
        seed: None,
        top_logprobs: None,
        timeout_at: {
            let secs = state.request_timeout as f32;
            if secs > 0.0 {
                Some(std::time::Instant::now() + std::time::Duration::from_secs_f32(secs))
            } else {
                None
            }
        },
        logit_probe: None,
        capture_hidden: false,
        response_tx: tx,
    };
    if state.request_tx.send(request).await.is_err() {
        return Err(openai_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "Scheduler queue full".to_string(),
        ));
    }
    match rx.await {
        Ok(Ok(r)) => Ok(r.output_tokens),
        Ok(Err(e)) => Err(openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Inference error: {e}"),
        )),
        Err(_) => Err(openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Inference cancelled".to_string(),
        )),
    }
}

pub async fn ocr(
    State(state): State<Arc<AppState>>,
    req: Result<Json<OcrRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "Invalid request JSON: {e} — expected {{\"image\": <base64 or data URI>, \
                     \"prompt_type\"?: \"ocr_layout\"|\"ocr\"}}"
                ),
            );
        }
    };
    if req.image.is_empty() {
        return openai_error_response(StatusCode::BAD_REQUEST, "'image' is empty".into());
    }
    let prompt_type = req.prompt_type.as_deref().unwrap_or("ocr_layout");
    let Some(prompt_text) = ocr::prompts::prompt_for(prompt_type) else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "unknown prompt_type '{prompt_type}' — valid: {:?}",
                ocr::prompts::PROMPT_TYPES
            ),
        );
    };

    // Resolve the model: explicit id or the single OCR-category
    // registration.
    let model_id = if req.model.is_empty() {
        let Some(id) = state.hotswap.as_ref().and_then(|ctx| {
            ctx.registry
                .iter()
                .find(|m| m.category == ModelCategory::Ocr)
                .map(|m| m.model_id.clone())
        }) else {
            return openai_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "no OCR model registered — add one to ATLAS_PRESEED_OCR".into(),
            );
        };
        id
    } else {
        req.model.clone()
    };

    // Category gate + lazy swap (Plan.md Phase K).
    let swap_ms =
        match super::lazy_swap::ensure_model_active(&state, &model_id, ModelCategory::Ocr).await {
            Ok(ms) => ms,
            Err(resp) => return resp,
        };

    let rt = state.runtime();
    let Some(vcfg) = rt.vision_config.as_ref() else {
        return openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("OCR model '{model_id}' has no vision tower loaded — cannot consume images"),
        );
    };

    // Decode the payload and sniff PDF vs image (reference: filetype).
    let bytes = match ocr::input::decode_request_bytes(&req.image) {
        Ok(b) => b,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Payload decode error: {e:#}"),
            );
        }
    };
    let page_images: Vec<(usize, image::RgbImage)> = if ocr::input::is_pdf(&bytes) {
        let page_range = match req.page_range.as_deref() {
            Some(s) if !s.trim().is_empty() => match ocr::input::parse_range_str(s) {
                Ok(r) => r,
                Err(e) => {
                    return openai_error_response(
                        StatusCode::BAD_REQUEST,
                        format!("invalid page_range: {e:#}"),
                    );
                }
            },
            _ => Vec::new(),
        };
        // pdfium work is CPU-bound and runs under a process-wide lock —
        // keep it off the async runtime.
        match tokio::task::spawn_blocking(move || ocr::input::load_pdf_images(&bytes, &page_range))
            .await
        {
            Ok(Ok(pages)) if !pages.is_empty() => pages,
            Ok(Ok(_)) => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    "PDF has no pages in the requested page_range".into(),
                );
            }
            Ok(Err(e)) => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("PDF render error: {e:#}"),
                );
            }
            Err(e) => {
                return openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("PDF render task failed: {e}"),
                );
            }
        }
    } else {
        match image::load_from_memory(&bytes) {
            Ok(img) => vec![(0, img.to_rgb8())],
            Err(e) => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("Image decode error: {e}"),
                );
            }
        }
    };

    let max_retries = req.max_retries.unwrap_or(ocr::MAX_RETRIES);
    let mut pages = Vec::with_capacity(page_images.len());
    let mut total_prompt = 0usize;
    let mut total_completion = 0usize;
    for (page_idx, raw_image) in page_images {
        // PDF pages are already rendered at ≥1024 px short side; the
        // reference applies the MIN_IMAGE_DIM upscale only to plain
        // images (load_image vs load_pdf_images).
        let page_image = ocr::input::load_image(raw_image, ocr::input::MIN_IMAGE_DIM);
        match process_page(
            &state,
            &rt,
            vcfg,
            &prompt_text,
            &page_image,
            req.max_tokens,
            max_retries,
            req.include_headers_footers,
            req.include_images,
        )
        .await
        {
            Ok((mut page, prompt_toks, completion_toks)) => {
                page.page = page_idx + 1;
                total_prompt += prompt_toks;
                total_completion += completion_toks;
                pages.push(page);
            }
            Err(resp) => return resp,
        }
    }

    let resp = Json(OcrResponse {
        model: model_id,
        pages,
        usage: OcrUsage {
            prompt_tokens: total_prompt,
            completion_tokens: total_completion,
        },
    })
    .into_response();
    super::lazy_swap::attach_swap_header(resp, swap_ms)
}

/// OCR one page image: scale_to_fit → vision preprocess → chat-template
/// prompt → generation with the reference degeneration retry → layout/
/// HTML/Markdown/crop assembly. Returns (page, prompt_tokens,
/// completion_tokens); `page.page` is filled by the caller.
#[allow(clippy::too_many_arguments)]
async fn process_page(
    state: &Arc<AppState>,
    rt: &Arc<crate::main_modules::ModelRuntime>,
    vcfg: &atlas_core::config::VisionConfig,
    prompt_text: &str,
    page_image: &image::RgbImage,
    max_tokens_req: Option<usize>,
    max_retries: usize,
    include_headers_footers: bool,
    include_images: bool,
) -> Result<(OcrPage, usize, usize), Response> {
    let model_image = ocr::input::scale_to_fit(page_image.clone());
    let (pixels, grid_h, grid_w) =
        match spark_model::vision_preprocess::preprocess_rgb_with_max_pixels(
            model_image,
            vcfg,
            state.vision_max_pixels,
        ) {
            Ok(p) => p,
            Err(e) => {
                return Err(openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("Image preprocess error: {e:#}"),
                ));
            }
        };
    let pad_count =
        spark_model::vision_preprocess::image_pad_count(grid_h, grid_w, vcfg.spatial_merge_size);
    let image_pixels = vec![(pixels, grid_h, grid_w)];
    let image_pad_counts = vec![pad_count];

    // Chat-template frame over one [image, text] user turn — identical
    // in shape to what the reference pipeline feeds vLLM.
    let json_messages = vec![serde_json::json!({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ],
    })];
    let prompt_tokens = match rt.tokenizer.apply_chat_template_openai(
        &json_messages,
        None,
        false,
        rt.behavior.disable_tool_steering,
    ) {
        Ok(t) => t,
        Err(e) => {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Tokenization error: {e}"),
            ));
        }
    };
    let prompt_tokens = rt
        .tokenizer
        .expand_image_pads(prompt_tokens, &image_pad_counts);
    let prompt_len = prompt_tokens.len();
    if prompt_len >= state.max_seq_len {
        return Err(openai_error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "page produces {prompt_len} prompt tokens, at/over the context limit \
                 {} — reduce the image resolution",
                state.max_seq_len,
            ),
        ));
    }
    let prompt_tokens = Arc::new(prompt_tokens);

    let max_tokens = max_tokens_req
        .unwrap_or(ocr::MAX_OUTPUT_TOKENS)
        .min(state.max_seq_len.saturating_sub(prompt_len));

    // Generation + degeneration retry, mirroring vllm.py::process_item:
    // pass 0 at (0.0, 0.1); retry n at (min(0.2·n, 0.8), 0.95).
    let mut retries = 0usize;
    let mut completion_tokens;
    let html = loop {
        let (temperature, top_p) = if retries == 0 {
            (0.0, 0.1)
        } else {
            ((0.2 * retries as f32).min(0.8), 0.95)
        };
        let output_tokens = generate_once(
            state,
            &prompt_tokens,
            &image_pixels,
            max_tokens,
            temperature,
            top_p,
        )
        .await?;
        completion_tokens = output_tokens.len();
        let text = match rt.tokenizer.decode(&output_tokens) {
            Ok(t) => t.trim().to_string(),
            Err(e) => {
                return Err(openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Decode error: {e}"),
                ));
            }
        };
        if retries < max_retries && ocr::has_degenerate_tail(&text) {
            retries += 1;
            tracing::info!(
                "OCR degenerate tail detected — retry {retries}/{max_retries} at \
                 temperature {:.1}",
                (0.2 * retries as f32).min(0.8),
            );
            continue;
        }
        break text;
    };

    // Output assembly (output.rs / markdown.rs) on the page image's
    // pixel space.
    let (img_w, img_h) = (page_image.width(), page_image.height());
    let chunks = ocr::output::parse_layout(&html, img_w, img_h);
    let filtered_html = ocr::output::parse_html(&html, include_headers_footers, include_images);
    let markdown = ocr::markdown::convert(&filtered_html);
    let images = ocr::output::extract_images(&html, &chunks, page_image)
        .into_iter()
        .map(|(name, bytes)| {
            (
                name,
                base64::engine::general_purpose::STANDARD.encode(bytes),
            )
        })
        .collect();

    Ok((
        OcrPage {
            page: 0,
            html,
            markdown,
            chunks,
            images,
            retries,
        },
        prompt_len,
        completion_tokens,
    ))
}

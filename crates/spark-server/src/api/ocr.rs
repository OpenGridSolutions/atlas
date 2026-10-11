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
//! Steps 2–4 live in `ocr::pipeline` (shared with the Datalab-compatible
//! `/convert` surface, api/convert/); this file keeps the request shape,
//! the PDF/image decode and the per-page output assembly.

use std::sync::Arc;

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use base64::Engine;
use serde::{Deserialize, Serialize};

use super::compact::{openai_error_response, openai_error_response_with_param};
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
    /// Atlas extension: seconds the client accepts from submit to
    /// completion. The server estimates up front and answers 503
    /// `capacity_exceeded` when it cannot meet it; an accepted request's
    /// deadline is clamped to it.
    #[serde(default)]
    pub accepted_processing_time: Option<f64>,
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

/// The single OCR-category registration (ATLAS_PRESEED_OCR), if any.
/// Shared with the Datalab-compatible `/convert` surface.
pub(crate) fn resolve_ocr_model(state: &AppState) -> Option<String> {
    state.hotswap.as_ref().and_then(|ctx| {
        ctx.registry
            .iter()
            .find(|m| m.category == ModelCategory::Ocr)
            .map(|m| m.model_id.clone())
    })
}

pub async fn ocr(
    State(state): State<Arc<AppState>>,
    prio: Option<axum::extract::Extension<crate::main_modules::middleware::RequestPriority>>,
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
    // Page count for the estimate: the base64 payload is decoded here
    // once more than strictly needed, but only when a budget is given
    // (otherwise one page is assumed for the informational header).
    let est_model = if req.model.is_empty() {
        resolve_ocr_model(&state).unwrap_or_default()
    } else {
        req.model.clone()
    };
    let pages = if req.accepted_processing_time.is_some() {
        match ocr::input::decode_request_bytes(&req.image) {
            Ok(b) if ocr::input::is_pdf(&b) => {
                let range = req.page_range.clone();
                tokio::task::spawn_blocking(move || {
                    ocr::input::pdf_page_count(&b).map(|n| {
                        crate::api::convert::worker::select_pages(n, range.as_deref(), None).len()
                    })
                })
                .await
                .ok()
                .and_then(Result::ok)
                .unwrap_or(1)
            }
            _ => 1,
        }
    } else {
        1
    };
    let admission = match super::admission::admit(
        &state,
        &est_model,
        crate::arena_sched::WorkProfile {
            prompt_tokens: 0,
            output_tokens: 0,
            pages: pages.max(1),
        },
        req.accepted_processing_time,
    ) {
        Ok(a) => a,
        Err(r) => return super::admission::openai_refusal(&est_model, r),
    };
    let resp = ocr_admitted(&state, prio, req, &admission).await;
    admission.attach_header(&state, resp)
}

async fn ocr_admitted(
    state: &Arc<AppState>,
    prio: Option<axum::extract::Extension<crate::main_modules::middleware::RequestPriority>>,
    req: OcrRequest,
    admission: &super::admission::Admission,
) -> Response {
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
        let Some(id) = resolve_ocr_model(state) else {
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
    let swap_ms = match super::lazy_swap::ensure_model_active(
        state,
        &model_id,
        ModelCategory::Ocr,
        prio.map(|p| p.0.0).unwrap_or(false),
        false,
    )
    .await
    {
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
            state,
            &rt,
            vcfg,
            &prompt_text,
            &page_image,
            req.max_tokens,
            max_retries,
            req.include_headers_footers,
            req.include_images,
            admission.deadline,
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

/// OCR one page image through the shared pipeline, then assemble
/// layout/HTML/Markdown/crops on the page image's pixel space. Returns
/// (page, prompt_tokens, completion_tokens); `page.page` is filled by
/// the caller. Pipeline errors map to OpenAI-shaped responses carrying
/// the pipeline's `error_code`; a truncated page is an error, never a
/// silently shortened success.
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
    deadline: Option<std::time::Instant>,
) -> Result<(OcrPage, usize, usize), Response> {
    let out = ocr::pipeline::ocr_page(ocr::pipeline::PageJob {
        state,
        rt,
        vcfg,
        prompt_text,
        page_image,
        max_tokens: max_tokens_req,
        max_retries,
        seed_base: None,
        deadline,
        cancel: None,
    })
    .await
    .map_err(page_error_response)?;
    let html = out.html;

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
            retries: out.retries,
        },
        out.prompt_tokens,
        out.completion_tokens,
    ))
}

/// Pipeline error → OpenAI-shaped response. Document faults are 400,
/// capacity 503, everything else 500; `code` carries the stable name.
fn page_error_response(e: ocr::pipeline::PageError) -> Response {
    use ocr::pipeline::PageError;
    let status = match &e {
        PageError::Document(_) => StatusCode::BAD_REQUEST,
        PageError::Capacity(_) => StatusCode::SERVICE_UNAVAILABLE,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    openai_error_response_with_param(status, e.to_string(), None, Some(e.error_code()))
}

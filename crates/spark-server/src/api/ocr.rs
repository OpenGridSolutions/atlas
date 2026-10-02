// SPDX-License-Identifier: AGPL-3.0-only

//! `POST /v1/ocr` — document OCR via Chandra-2 (Todo 3.3b/c).
//!
//! Single-model category endpoint (like /v1/translate): the request is
//! served by the one `ATLAS_PRESEED_OCR` registration, category-locked
//! so no other model can be driven here and the OCR model answers 400
//! everywhere else.
//!
//! The prompt frame is the checkpoint's own chat template rendered over
//! one user turn of `[image, prompt_text]` — the exact message shape
//! the Chandra reference pipeline sends through vLLM — followed by the
//! standard image-pad expansion. Sampling mirrors
//! `chandra/model/vllm.py`: temperature 0.0, top_p 0.1, up to 12384
//! output tokens.
//!
//! CURRENT SCOPE (3.3b): single image (data URI or raw base64) in,
//! `pages[0].html` out. The 3.3c pipeline port fills in: PDF rendering
//! via pdfium (+ `page_range`), the Lanczos min-dim upscale from
//! chandra/input.py, layout-chunk parsing + bbox crops + HTML→Markdown
//! from chandra/output.py, and the temperature-bump degeneration retry.

use std::sync::Arc;

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};

use super::compact::openai_error_response;
use super::inference_types::InferenceRequest;
use crate::main_modules::AppState;
use crate::main_modules::hotswap::ModelCategory;
use crate::ocr;

#[derive(Deserialize)]
pub struct OcrRequest {
    /// Image as a `data:` URI or raw base64 (PNG/JPEG/WebP). PDF input
    /// lands with the 3.3c pdfium port.
    pub image: String,
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
}

/// One OCR'd page. `markdown`, `chunks` and `images` are filled by the
/// 3.3c output port; until then `html` carries the raw model output and
/// the other fields are empty (stable schema, additive completion).
#[derive(Serialize)]
pub struct OcrPage {
    pub page: usize,
    pub html: String,
    pub markdown: String,
    pub chunks: Vec<OcrChunk>,
    pub images: std::collections::BTreeMap<String, String>,
}

#[derive(Serialize)]
pub struct OcrChunk {
    pub label: String,
    pub bbox: [u32; 4],
    pub content: String,
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

    // Accept both a full data URI and bare base64 payload. The decoder
    // sniffs the actual image format from the bytes, so the exact mime
    // in the synthesized prefix is irrelevant.
    let data_uri = if req.image.starts_with("data:") {
        req.image.clone()
    } else {
        format!("data:image/png;base64,{}", req.image)
    };
    // TODO(3.3c input.rs): Chandra's min-dimension Lanczos upscale
    // (MIN_IMAGE_DIM=1536) before patching — small scans currently get
    // fewer vision tokens than the reference pipeline would produce.
    let (pixels, grid_h, grid_w) = match spark_model::vision_preprocess::preprocess_image_with_max_pixels(
        &data_uri,
        vcfg,
        state.vision_max_pixels,
    ) {
        Ok(p) => p,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Image decode error: {e}"),
            );
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
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Tokenization error: {e}"),
            );
        }
    };
    let prompt_tokens = rt
        .tokenizer
        .expand_image_pads(prompt_tokens, &image_pad_counts);
    let prompt_len = prompt_tokens.len();
    if prompt_len >= state.max_seq_len {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "image produces {prompt_len} prompt tokens, at/over the context limit \
                 {} — reduce the image resolution",
                state.max_seq_len,
            ),
        );
    }

    let max_tokens = req
        .max_tokens
        .unwrap_or(ocr::MAX_OUTPUT_TOKENS)
        .min(state.max_seq_len.saturating_sub(prompt_len));

    let (tx, rx) = tokio::sync::oneshot::channel();
    let session_hash = crate::session_manager::compute_session_hash(&prompt_tokens);
    let request = InferenceRequest::Blocking {
        prompt_tokens: Arc::new(prompt_tokens),
        session_hash,
        image_pixels,
        max_tokens,
        min_tokens: 0,
        // chandra/model/vllm.py defaults: deterministic transcription.
        // TODO(3.3c): degeneration retry at temperature+0.2 (cap 0.8,
        // top_p 0.95, up to MAX_VLLM_RETRIES) like the reference.
        temperature: 0.0,
        top_k: 0,
        top_p: 0.1,
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
        return openai_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "Scheduler queue full".to_string(),
        );
    }
    let response = match rx.await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Inference error: {e}"),
            );
        }
        Err(_) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Inference cancelled".to_string(),
            );
        }
    };
    let html = match rt.tokenizer.decode(&response.output_tokens) {
        Ok(t) => t.trim().to_string(),
        Err(e) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Decode error: {e}"),
            );
        }
    };

    let resp = Json(OcrResponse {
        model: model_id,
        pages: vec![OcrPage {
            page: 1,
            html,
            markdown: String::new(),
            chunks: Vec::new(),
            images: std::collections::BTreeMap::new(),
        }],
        usage: OcrUsage {
            prompt_tokens: prompt_len,
            completion_tokens: response.output_tokens.len(),
        },
    })
    .into_response();
    super::lazy_swap::attach_swap_header(resp, swap_ms)
}

// SPDX-License-Identifier: AGPL-3.0-only

//! Per-page OCR pipeline shared by `/v1/ocr` and `/convert`.
//!
//! One page image in → raw Chandra layout HTML out, with the reference
//! degeneration retry (`vllm.py::process_item`) and typed errors so each
//! surface maps them to its own wire format (OpenAI-shaped vs. Datalab-
//! shaped). The pipeline never touches HTTP types.
//!
//! Truncation is an error here: a page whose generation ended on
//! `finish_reason != "stop"` (max_tokens, request timeout, job deadline,
//! model swap) is reported as [`PageError::Truncated`] instead of being
//! returned as text — the caller decides whether that fails the whole
//! document (`/convert`: yes, `metadata.failed_pages`).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::api::inference_types::InferenceRequest;
use crate::main_modules::{AppState, ModelRuntime};

/// Why a page could not be OCR'd. `error_code()` is the stable,
/// machine-readable name shared by both surfaces.
#[derive(Debug, Clone)]
pub enum PageError {
    /// The document/page itself is the problem (too large for the
    /// context, undecodable pixels). Clients must not fall back.
    Document(String),
    /// Scheduler queue is full — capacity, retry later.
    Capacity(String),
    /// Scheduler reported an inference failure.
    Inference(String),
    /// The oneshot was dropped (scheduler shutdown / drain).
    Cancelled,
    /// Generation stopped before EOS (`finish_reason`), output discarded.
    Truncated {
        finish_reason: String,
        tokens: usize,
    },
    /// Token decode failure.
    Decode(String),
}

impl PageError {
    pub fn error_code(&self) -> &'static str {
        match self {
            PageError::Document(_) => "unprocessable_document",
            PageError::Capacity(_) => "capacity_exceeded",
            PageError::Inference(_) => "inference_error",
            PageError::Cancelled => "cancelled",
            PageError::Truncated { .. } => "incomplete_document",
            PageError::Decode(_) => "internal_error",
        }
    }

    /// True when the client's document is at fault (no fallback).
    pub fn is_document_fault(&self) -> bool {
        matches!(self, PageError::Document(_))
    }
}

impl std::fmt::Display for PageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PageError::Document(m) => write!(f, "{m}"),
            PageError::Capacity(m) => write!(f, "{m}"),
            PageError::Inference(m) => write!(f, "inference error: {m}"),
            PageError::Cancelled => write!(f, "inference cancelled"),
            PageError::Truncated {
                finish_reason,
                tokens,
            } => write!(
                f,
                "page output truncated after {tokens} tokens (finish_reason={finish_reason})"
            ),
            PageError::Decode(m) => write!(f, "decode error: {m}"),
        }
    }
}

/// Everything one page needs. Borrowed, so a caller fanning pages out
/// clones only the `Arc`s.
pub struct PageJob<'a> {
    pub state: &'a Arc<AppState>,
    pub rt: &'a Arc<ModelRuntime>,
    pub vcfg: &'a atlas_core::config::VisionConfig,
    pub prompt_text: &'a str,
    pub page_image: &'a image::RgbImage,
    /// Requested cap on generated tokens (None = reference default).
    pub max_tokens: Option<usize>,
    pub max_retries: usize,
    /// Deterministic seed base for the degeneration retries; retry `n`
    /// samples with `seed_base + n`. None = unseeded (legacy `/v1/ocr`).
    pub seed_base: Option<u64>,
    /// Absolute deadline for this page (job deadline). None = no deadline.
    pub deadline: Option<Instant>,
    /// Cooperative cancel flag checked before every generation pass.
    pub cancel: Option<&'a AtomicBool>,
}

/// Raw result of one page: Chandra layout HTML plus accounting.
pub struct PageOutput {
    pub html: String,
    pub retries: usize,
    /// True when the retry budget is exhausted and the tail is still
    /// degenerate. The reference returns such output anyway; `/convert`
    /// treats it as a failed page (`degenerate_output`).
    pub degenerate: bool,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

/// One blocking generation pass at the given sampling point. Returns
/// `(output_tokens, finish_reason)`.
#[allow(clippy::too_many_arguments)]
async fn generate_once(
    state: &Arc<AppState>,
    prompt_tokens: &Arc<Vec<u32>>,
    image_pixels: &[(Vec<f32>, usize, usize)],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: Option<u64>,
    deadline: Option<Instant>,
    cancel: Option<&AtomicBool>,
) -> Result<(Vec<u32>, String), PageError> {
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
        seed,
        top_logprobs: None,
        timeout_at: deadline,
        logit_probe: None,
        capture_hidden: false,
        response_tx: tx,
    };
    if state.request_tx.send(request).await.is_err() {
        return Err(PageError::Capacity("scheduler queue full".into()));
    }
    // Poll the cancel flag while waiting; dropping `rx` closes the
    // oneshot, which the scheduler treats as "client gone" and finishes
    // the sequence (frees the batch slot and KV).
    let mut rx = rx;
    loop {
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            drop(rx);
            return Err(PageError::Cancelled);
        }
        match tokio::time::timeout(std::time::Duration::from_millis(250), &mut rx).await {
            Ok(Ok(Ok(r))) => return Ok((r.output_tokens, r.finish_reason)),
            Ok(Ok(Err(e))) => return Err(PageError::Inference(e.to_string())),
            Ok(Err(_)) => return Err(PageError::Cancelled),
            Err(_) => continue,
        }
    }
}

/// OCR one page image: scale_to_fit → vision preprocess → chat-template
/// prompt → generation with the reference degeneration retry. The
/// returned HTML is the raw model output; layout/markdown assembly is
/// the caller's (output.rs / markdown.rs).
pub async fn ocr_page(job: PageJob<'_>) -> Result<PageOutput, PageError> {
    let model_image = crate::ocr::input::scale_to_fit(job.page_image.clone());
    let (pixels, grid_h, grid_w) = spark_model::vision_preprocess::preprocess_rgb_with_max_pixels(
        model_image,
        job.vcfg,
        job.state.vision_max_pixels,
    )
    .map_err(|e| PageError::Document(format!("image preprocess error: {e:#}")))?;
    let pad_count = spark_model::vision_preprocess::image_pad_count(
        grid_h,
        grid_w,
        job.vcfg.spatial_merge_size,
    );
    let image_pixels = vec![(pixels, grid_h, grid_w)];

    // Chat-template frame over one [image, text] user turn — identical
    // in shape to what the reference pipeline feeds vLLM.
    let json_messages = vec![serde_json::json!({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": job.prompt_text},
        ],
    })];
    let prompt_tokens = job
        .rt
        .tokenizer
        .apply_chat_template_openai(
            &json_messages,
            None,
            false,
            job.rt.behavior.disable_tool_steering,
        )
        .map_err(|e| PageError::Document(format!("tokenization error: {e}")))?;
    let prompt_tokens = job
        .rt
        .tokenizer
        .expand_image_pads(prompt_tokens, &[pad_count]);
    let prompt_len = prompt_tokens.len();
    if prompt_len >= job.state.max_seq_len() {
        return Err(PageError::Document(format!(
            "page produces {prompt_len} prompt tokens, at/over the context limit {} — \
             reduce the image resolution",
            job.state.max_seq_len(),
        )));
    }
    let prompt_tokens = Arc::new(prompt_tokens);
    let max_tokens = job
        .max_tokens
        .unwrap_or(crate::ocr::MAX_OUTPUT_TOKENS)
        .min(job.state.max_seq_len().saturating_sub(prompt_len));

    // Generation + degeneration retry, mirroring vllm.py::process_item:
    // pass 0 at (0.0, 0.1); retry n at (min(0.2·n, 0.8), 0.95).
    let mut retries = 0usize;
    loop {
        if job.cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Err(PageError::Cancelled);
        }
        if job.deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(PageError::Truncated {
                finish_reason: "deadline".into(),
                tokens: 0,
            });
        }
        let (temperature, top_p) = if retries == 0 {
            (0.0, 0.1)
        } else {
            ((0.2 * retries as f32).min(0.8), 0.95)
        };
        let seed = job.seed_base.map(|b| b.wrapping_add(retries as u64));
        let (output_tokens, finish_reason) = generate_once(
            job.state,
            &prompt_tokens,
            &image_pixels,
            max_tokens,
            temperature,
            top_p,
            seed,
            job.deadline,
            job.cancel,
        )
        .await?;
        if finish_reason != "stop" {
            return Err(PageError::Truncated {
                finish_reason,
                tokens: output_tokens.len(),
            });
        }
        let completion_tokens = output_tokens.len();
        let text = job
            .rt
            .tokenizer
            .decode(&output_tokens)
            .map_err(|e| PageError::Decode(e.to_string()))?
            .trim()
            .to_string();
        let degenerate = crate::ocr::has_degenerate_tail(&text);
        if degenerate && retries < job.max_retries {
            retries += 1;
            tracing::info!(
                "OCR degenerate tail detected — retry {retries}/{} at temperature {:.1}",
                job.max_retries,
                (0.2 * retries as f32).min(0.8),
            );
            continue;
        }
        return Ok(PageOutput {
            html: text,
            retries,
            degenerate,
            prompt_tokens: prompt_len,
            completion_tokens,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_codes_are_stable() {
        assert_eq!(
            PageError::Document("x".into()).error_code(),
            "unprocessable_document"
        );
        assert!(PageError::Document("x".into()).is_document_fault());
        assert_eq!(
            PageError::Truncated {
                finish_reason: "length".into(),
                tokens: 3
            }
            .error_code(),
            "incomplete_document"
        );
        assert!(!PageError::Capacity("q".into()).is_document_fault());
    }
}

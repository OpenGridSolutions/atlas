// SPDX-License-Identifier: AGPL-3.0-only

//! `POST /v1/rerank` — Qwen3-Reranker scoring (multiclass Plan.md 1.4, R2).
//!
//! The Qwen3-Reranker is a causal LM whose relevance score is defined as
//! softmax over the yes/no token logits at the last prompt position of a
//! fixed judge prompt (model card reference implementation). Per
//! (query, document) pair this handler renders that prompt, enqueues a
//! blocking request with `max_tokens = 1` (immediate finish — prefill
//! only, KV freed at once) and `logit_probe = [yes_id, no_id]`, and
//! computes `score = sigmoid(yes_logit − no_logit)` — identical to the
//! pair softmax. All pairs are enqueued before any is awaited, so the
//! scheduler's batched-prefill paths can co-schedule them.

use std::sync::Arc;

use axum::extract::{FromRequest, Json, State};
use axum::http::StatusCode;
use axum::response::Response;
use axum::response::IntoResponse;

use crate::main_modules::app_state::AppState;
use crate::openai::rerank::{
    RerankDocument, RerankRequest, RerankResponse, RerankResult, RerankUsage,
};

use super::compact::openai_error_response;
use super::inference_types::InferenceRequest;

/// Prompt frame from the Qwen3-Reranker model card. The empty
/// `<think></think>` block is part of the official format — the score
/// is read at the FIRST generated position after it.
const SYSTEM_PREFIX: &str = "<|im_start|>system\nJudge whether the Document meets the \
     requirements based on the Query and the Instruct provided. Note that the answer can \
     only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
const ASSISTANT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const DEFAULT_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

/// Relevance score from the raw yes/no logit pair:
/// softmax([yes, no])[yes] = sigmoid(yes − no). Returns `None` when the
/// pair is unusable (probe failed, or both logits are non-finite).
pub(crate) fn relevance_score(probed: &[f32]) -> Option<f32> {
    let [yes, no] = *probed.first_chunk::<2>()?;
    if !yes.is_finite() && !no.is_finite() {
        return None;
    }
    Some(1.0 / (1.0 + (no - yes).exp()))
}

/// Resolve a word to its single token id; the Qwen3-Reranker score
/// definition requires "yes"/"no" to be atomic tokens (they are, in the
/// Qwen vocab — guard anyway so a foreign tokenizer fails loudly).
fn single_token_id(rt: &crate::main_modules::app_state::ModelRuntime, word: &str) -> Option<u32> {
    match rt.tokenizer.encode(word) {
        Ok(ids) if ids.len() == 1 => Some(ids[0]),
        _ => None,
    }
}

/// Content-Type dispatch (Plan.md Phase U): JSON body as before, or
/// `multipart/form-data` where each `documents` file part is an image
/// document and each plain `documents` part a text document. Both
/// paths build the same [`RerankRequest`] and share one core.
pub async fn rerank(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
) -> Response {
    let req = if super::multimodal_input::is_multipart(&request) {
        let multipart = match axum::extract::Multipart::from_request(request, &()).await {
            Ok(mp) => mp,
            Err(e) => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("multipart parse error: {e}"),
                );
            }
        };
        match super::multimodal_input::rerank_from_multipart(multipart).await {
            Ok(r) => r,
            Err((status, msg)) => return openai_error_response(status, msg),
        }
    } else {
        match Json::<RerankRequest>::from_request(request, &()).await {
            Ok(Json(r)) => r,
            Err(e) => return openai_error_response(e.status(), e.body_text()),
        }
    };
    rerank_core(state, req).await
}

async fn rerank_core(state: Arc<AppState>, req: RerankRequest) -> Response {
    if req.documents.is_empty() {
        return openai_error_response(StatusCode::BAD_REQUEST, "documents must not be empty".into());
    }
    // Lazy multi-model swap BEFORE the runtime snapshot — the requested
    // model's tokenizer must be the one that encodes these prompts.
    let swap_ms = match super::lazy_swap::ensure_model_active(
        &state,
        &req.model,
        crate::main_modules::hotswap::ModelCategory::Rerank,
    )
    .await {
        Ok(ms) => ms,
        Err(resp) => return resp,
    };
    let rt = state.runtime();

    let (Some(yes_id), Some(no_id)) = (single_token_id(&rt, "yes"), single_token_id(&rt, "no"))
    else {
        return openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "active model's tokenizer has no atomic yes/no tokens — not a Qwen3-Reranker-class \
             model"
                .into(),
        );
    };
    let probe: Arc<[u32]> = Arc::from([yes_id, no_id].as_slice());

    let instruction = req
        .instruction
        .as_deref()
        .unwrap_or(DEFAULT_INSTRUCTION);

    let timeout_at = if state.request_timeout > 0 {
        Some(std::time::Instant::now() + std::time::Duration::from_secs(state.request_timeout as u64))
    } else {
        None
    };

    // Enqueue every (query, document) pair before awaiting any reply so
    // concurrent prefills can batch.
    let mut pending = Vec::with_capacity(req.documents.len());
    let mut prompt_tokens_total = 0usize;
    crate::metrics::REQUESTS_TOTAL.inc();
    crate::metrics::REQUESTS_ACTIVE.inc();
    for (index, doc) in req.documents.iter().enumerate() {
        let (doc_text, doc_image) = doc.parts();
        // Image documents get the vision frame inside <Document>: (2.3).
        let encoded =
            super::multimodal_input::encode_item(&state, &rt, doc_image, |vision| {
                format!(
                    "{SYSTEM_PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: \
                     {vision}{doc_text}{ASSISTANT_SUFFIX}",
                    query = req.query,
                )
            });
        let encoded = match encoded {
            Ok(e) => e,
            Err((status, msg)) => {
                crate::metrics::REQUESTS_ACTIVE.dec();
                return openai_error_response(status, format!("document {index}: {msg}"));
            }
        };
        let prompt_tokens = encoded.tokens;
        if prompt_tokens.len() >= state.max_seq_len {
            crate::metrics::REQUESTS_ACTIVE.dec();
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "document {index}: rerank prompt is {} tokens, exceeds max_seq_len {}",
                    prompt_tokens.len(),
                    state.max_seq_len
                ),
            );
        }
        prompt_tokens_total += prompt_tokens.len();
        let session_hash = crate::session_manager::compute_session_hash(&prompt_tokens);
        let (tx, rx) = tokio::sync::oneshot::channel();
        let request = InferenceRequest::Blocking {
            prompt_tokens: Arc::new(prompt_tokens),
            session_hash,
            image_pixels: encoded.image_pixels,
            max_tokens: 1, // immediate finish: prefill + one discarded sample
            min_tokens: 0,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
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
            disable_mtp: true,
            grammar_spec: None,
            seed: None,
            top_logprobs: None,
            timeout_at,
            logit_probe: Some(probe.clone()),
            capture_hidden: false,
            response_tx: tx,
        };
        if state.request_tx.send(request).await.is_err() {
            crate::metrics::REQUESTS_ACTIVE.dec();
            return openai_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "Scheduler queue full".into(),
            );
        }
        pending.push((index, rx));
    }

    let mut results = Vec::with_capacity(pending.len());
    for (index, rx) in pending {
        let response = match rx.await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                crate::metrics::REQUESTS_ACTIVE.dec();
                return openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("scoring document {index}: {e}"),
                );
            }
            Err(_) => {
                crate::metrics::REQUESTS_ACTIVE.dec();
                return openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("scoring document {index}: inference cancelled"),
                );
            }
        };
        let Some(score) = relevance_score(&response.probed_logits) else {
            crate::metrics::REQUESTS_ACTIVE.dec();
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "document {index}: logit probe returned no usable yes/no pair \
                     (got {:?})",
                    response.probed_logits
                ),
            );
        };
        results.push(RerankResult {
            index,
            relevance_score: score,
            document: req
                .return_documents
                .then(|| RerankDocument { text: req.documents[index].text_owned() }),
        });
    }
    crate::metrics::REQUESTS_ACTIVE.dec();

    results.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(n) = req.top_n {
        results.truncate(n);
    }

    let n_docs = req.documents.len();
    let resp = Json(RerankResponse {
        model: req.model,
        results,
        usage: RerankUsage {
            prompt_tokens: prompt_tokens_total,
            // One sampled-and-discarded token per document.
            total_tokens: prompt_tokens_total + n_docs,
        },
    })
    .into_response();
    super::lazy_swap::attach_swap_header(resp, swap_ms)
}

#[cfg(test)]
mod tests {
    use super::relevance_score;

    #[test]
    fn score_is_pair_softmax() {
        // yes >> no → ~1; no >> yes → ~0; equal → 0.5.
        assert!(relevance_score(&[10.0, -10.0]).unwrap() > 0.999);
        assert!(relevance_score(&[-10.0, 10.0]).unwrap() < 0.001);
        assert!((relevance_score(&[1.5, 1.5]).unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn degenerate_pairs_are_rejected() {
        assert!(relevance_score(&[]).is_none());
        assert!(relevance_score(&[1.0]).is_none());
        assert!(relevance_score(&[f32::NEG_INFINITY, f32::NEG_INFINITY]).is_none());
    }

    #[test]
    fn one_sided_infinity_still_scores() {
        // Out-of-vocab "no" id degrades to -inf: score saturates at 1.
        assert!((relevance_score(&[5.0, f32::NEG_INFINITY]).unwrap() - 1.0).abs() < 1e-6);
    }
}

// SPDX-License-Identifier: AGPL-3.0-only

//! `POST /v1/embeddings` — Qwen3-Embedding serving (multiclass Plan.md
//! 1.4, E2). Replaces the former 501 stub.
//!
//! Qwen3-Embedding is a causal LM whose text embedding is the
//! post-final-norm hidden state of the trailing EOS token (last-token
//! pooling), L2-normalized. Per input this handler appends
//! `<|endoftext|>`, enqueues a blocking request with `max_tokens = 1`
//! (prefill-only immediate finish) and `capture_hidden = true`, then
//! normalizes host-side. Optional `dimensions` performs Matryoshka
//! truncation + re-normalization.

use std::sync::Arc;

use axum::extract::{FromRequest, Json, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::Response;

use crate::main_modules::app_state::AppState;
use crate::openai::embeddings::{
    EmbeddingObject, EmbeddingsRequest, EmbeddingsResponse, EmbeddingsUsage,
};

use super::compact::openai_error_response;
use super::inference_types::InferenceRequest;

/// Qwen3-Embedding pools the trailing EOS token; append it explicitly
/// (the tokenizer maps the literal to its single special-token id).
const EOS_SUFFIX: &str = "<|endoftext|>";

/// L2-normalize in place; optionally Matryoshka-truncate first.
/// Returns `None` for degenerate vectors (empty or zero norm).
pub(crate) fn normalize(mut v: Vec<f32>, dimensions: Option<usize>) -> Option<Vec<f32>> {
    if let Some(d) = dimensions {
        if d == 0 || d > v.len() {
            return None;
        }
        v.truncate(d);
    }
    let norm = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    if !(norm.is_finite() && norm > 0.0) || v.is_empty() {
        return None;
    }
    for x in &mut v {
        *x = ((*x as f64) / norm) as f32;
    }
    Some(v)
}

/// Content-Type dispatch (Plan.md Phase U): JSON body as before, or
/// `multipart/form-data` with `file` fields carrying image inputs the
/// same way audio goes to /v1/audio/transcriptions. Both paths build
/// the same [`EmbeddingsRequest`] and share one core.
pub async fn embeddings(
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
        match super::multimodal_input::embeddings_from_multipart(multipart).await {
            Ok(r) => r,
            Err((status, msg)) => return openai_error_response(status, msg),
        }
    } else {
        match Json::<EmbeddingsRequest>::from_request(request, &()).await {
            Ok(Json(r)) => r,
            Err(e) => return openai_error_response(e.status(), e.body_text()),
        }
    };
    embeddings_core(state, req).await
}

async fn embeddings_core(state: Arc<AppState>, req: EmbeddingsRequest) -> Response {
    if let Some(fmt) = req.encoding_format.as_deref()
        && fmt != "float"
    {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("encoding_format '{fmt}' not supported (only \"float\")"),
        );
    }
    let inputs = req.input.into_items();
    if inputs.is_empty() || inputs.iter().any(|i| i.is_empty()) {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "input must not be empty (each item needs text and/or image)".into(),
        );
    }

    // Lazy multi-model swap BEFORE the runtime snapshot.
    let swap_ms = match super::lazy_swap::ensure_model_active(
        &state,
        &req.model,
        crate::main_modules::hotswap::ModelCategory::Embed,
    )
    .await {
        Ok(ms) => ms,
        Err(resp) => return resp,
    };
    let rt = state.runtime();

    let timeout_at = if state.request_timeout > 0 {
        Some(
            std::time::Instant::now()
                + std::time::Duration::from_secs(state.request_timeout as u64),
        )
    } else {
        None
    };

    crate::metrics::REQUESTS_TOTAL.inc();
    crate::metrics::REQUESTS_ACTIVE.inc();
    let mut pending = Vec::with_capacity(inputs.len());
    let mut prompt_tokens_total = 0usize;
    for (index, item) in inputs.iter().enumerate() {
        let (text, image) = item.parts();
        // Model-card format: queries carry an instruct prefix, documents
        // embed raw; the caller decides via the `instruction` field. An
        // image item gets the vision frame before the text (2.3).
        let encoded = super::multimodal_input::encode_item(&state, &rt, image, |vision| {
            match req.instruction.as_deref() {
                Some(instr) => format!("Instruct: {instr}\nQuery: {vision}{text}{EOS_SUFFIX}"),
                None => format!("{vision}{text}{EOS_SUFFIX}"),
            }
        });
        let encoded = match encoded {
            Ok(e) => e,
            Err((status, msg)) => {
                crate::metrics::REQUESTS_ACTIVE.dec();
                return openai_error_response(status, format!("input {index}: {msg}"));
            }
        };
        let prompt_tokens = encoded.tokens;
        if prompt_tokens.len() >= state.max_seq_len {
            crate::metrics::REQUESTS_ACTIVE.dec();
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "input {index}: {} tokens exceeds max_seq_len {}",
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
            logit_probe: None,
            capture_hidden: true,
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

    let mut data = Vec::with_capacity(pending.len());
    for (index, rx) in pending {
        let response = match rx.await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                crate::metrics::REQUESTS_ACTIVE.dec();
                return openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("embedding input {index}: {e}"),
                );
            }
            Err(_) => {
                crate::metrics::REQUESTS_ACTIVE.dec();
                return openai_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("embedding input {index}: inference cancelled"),
                );
            }
        };
        let Some(embedding) = normalize(response.pooled_hidden, req.dimensions) else {
            crate::metrics::REQUESTS_ACTIVE.dec();
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "input {index}: pooled-hidden capture returned no usable vector \
                     (dimensions={:?})",
                    req.dimensions
                ),
            );
        };
        data.push(EmbeddingObject {
            object: "embedding",
            index,
            embedding,
        });
    }
    crate::metrics::REQUESTS_ACTIVE.dec();

    let resp = Json(EmbeddingsResponse {
        object: "list",
        data,
        model: req.model,
        usage: EmbeddingsUsage {
            prompt_tokens: prompt_tokens_total,
            total_tokens: prompt_tokens_total,
        },
    })
    .into_response();
    super::lazy_swap::attach_swap_header(resp, swap_ms)
}

#[cfg(test)]
mod tests {
    use super::normalize;

    #[test]
    fn normalizes_to_unit_length() {
        let v = normalize(vec![3.0, 4.0], None).unwrap();
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn matryoshka_truncates_then_renormalizes() {
        let v = normalize(vec![3.0, 4.0, 100.0], Some(2)).unwrap();
        assert_eq!(v.len(), 2);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn degenerate_vectors_are_rejected() {
        assert!(normalize(Vec::new(), None).is_none());
        assert!(normalize(vec![0.0, 0.0], None).is_none());
        assert!(normalize(vec![1.0, 2.0], Some(5)).is_none());
        assert!(normalize(vec![1.0, 2.0], Some(0)).is_none());
    }
}

// SPDX-License-Identifier: AGPL-3.0-only

//! Overflow handling of the chat pipeline (E7): synchronous compaction
//! (CMP-10), the `reject` policy (CTX-2), the not-compactable refusal
//! (CTX-6) and the CTX-4 response headers.

use std::sync::Arc;

use axum::http::{HeaderValue, StatusCode};
use axum::response::Response;

use super::prepare::{Prepared, prepare_prompt};
use crate::AppState;
use crate::api::compact::{openai_error_response, openai_error_response_with_param};
use crate::compaction::request::{
    COMPACTED_HEADER, COMPACTION_ID_HEADER, CompactError, CompactionInfo, RequestCompaction,
};
use crate::main_modules::app_state::ModelRuntime;
use crate::openai::ChatCompletionRequest;

pub(super) const CODE: &str = "context_length_exceeded";

/// CTX-2: the client (or the operator default) chose `reject`.
pub(super) fn reject_response(state: &AppState, prompt_len: usize) -> Response {
    state
        .compaction_index
        .stats
        .rejected_policy
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    crate::metrics::CONTEXT_REJECTED_TOTAL
        .with_label_values(&["policy_reject"])
        .inc();
    openai_error_response_with_param(
        StatusCode::BAD_REQUEST,
        format!(
            "Prompt too long: {prompt_len} tokens exceeds max_seq_len {} (context policy: reject; \
             leave room for output tokens)",
            state.max_seq_len()
        ),
        Some("messages"),
        Some(CODE),
    )
}

/// CTX-6: even head + summary + newest message would not fit.
fn not_compactable(state: &AppState, max_tokens: usize, err: &CompactError) -> Response {
    let CompactError::NothingToFold {
        head_tokens,
        newest_tokens,
        summary_tokens,
    } = err
    else {
        unreachable!("caller filters NothingToFold");
    };
    state
        .compaction_index
        .stats
        .rejected_ctx6
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    crate::metrics::CONTEXT_REJECTED_TOTAL
        .with_label_values(&["not_compactable"])
        .inc();
    let reserve = state.compaction.reserve_for(max_tokens);
    let msg = format!(
        "Request does not fit the {}-token context window even after compaction: head {head_tokens} \
         tokens + newest message {newest_tokens} tokens + summary {summary_tokens} tokens + reserved \
         output {reserve} tokens. Shorten the newest message or the system prompt.",
        state.max_seq_len()
    );
    crate::observe::emit(
        crate::observe::Event::new(2, "compaction_rejected", msg.clone()).detail(
            serde_json::json!({
                "head_tokens": head_tokens, "newest_tokens": newest_tokens,
                "summary_tokens": summary_tokens, "reserve": reserve, "window": state.max_seq_len(),
            }),
        ),
    );
    openai_error_response_with_param(StatusCode::BAD_REQUEST, msg, Some("messages"), Some(CODE))
}

/// Create a new compaction entry for the request and render the prompt
/// again (CMP-10). Errors are complete HTTP responses.
#[allow(clippy::result_large_err)]
pub(super) async fn resolve_overflow(
    state: &Arc<AppState>,
    rt: &Arc<ModelRuntime>,
    cmp: &mut RequestCompaction,
    req: &mut ChatCompletionRequest,
    tools_active: bool,
) -> Result<Prepared, Response> {
    let max_tokens = req.max_tokens;
    match cmp.compact(state, rt, max_tokens).await {
        Ok(messages) => req.messages = messages,
        Err(e @ CompactError::NothingToFold { .. }) => {
            return Err(not_compactable(state, max_tokens, &e));
        }
        Err(CompactError::Summary(e)) => {
            tracing::error!(error = %e, "compaction: summary generation failed");
            return Err(openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Context compaction failed: {e}"),
            ));
        }
    }
    let prepared = prepare_prompt(state, rt, req, tools_active)?;
    let prompt_len = prepared.prompt_tokens.len();
    if !state
        .compaction
        .fits(prompt_len, max_tokens, state.max_seq_len())
    {
        tracing::warn!(
            prompt_len,
            window = state.max_seq_len(),
            "compaction: effective context still does not fit after a fresh entry"
        );
        return Err(not_compactable(state, max_tokens, &cmp.ctx6(rt)));
    }
    Ok(prepared)
}

/// CTX-4 headers on the chat response.
pub(super) fn attach_compaction_headers(
    mut resp: Response,
    info: Option<&CompactionInfo>,
) -> Response {
    if let Some(info) = info {
        let h = resp.headers_mut();
        h.insert(COMPACTED_HEADER, HeaderValue::from_static("true"));
        if let Ok(v) = HeaderValue::from_str(&info.id.to_hex()) {
            h.insert(COMPACTION_ID_HEADER, v);
        }
    }
    resp
}

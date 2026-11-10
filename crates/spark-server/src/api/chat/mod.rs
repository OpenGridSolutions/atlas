// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

//! `/v1/chat/completions` orchestrator.
//!
//! Wave-4g extraction (2026-05-03): the original 1121-LoC `chat.rs`
//! held one async fn (`chat_completions_inner`) where every phase
//! shared a function-local `MsgEntry` struct + ~25 carry-through
//! locals. This module now coordinates:
//!
//! - `msg_entry`      — `MsgEntry` + `build_msg_entries` (req →
//!                      tokenisable shape, image preprocessing,
//!                      cwd extraction)
//! - `loop_detect`    — generic loop / spinning detection +
//!                      task-pin re-anchor
//! - `thinking`       — `(enable_thinking, thinking_budget)`
//!                      resolution
//! - `template`       — JSON-message build, auto-compact,
//!                      Jinja apply, image-pad expand,
//!                      template-forced-thinking detection
//! - `sampling_setup` — preset / penalty / stop-token / grammar /
//!                      timeout / logprobs resolution

mod finish;
mod loop_detect;
mod msg_entry;
mod overflow;
mod prepare;
mod sampling_setup;
mod template;
mod thinking;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use std::sync::Arc;

use crate::AppState;
use crate::compaction::{ContextPolicy, RequestCompaction};
use crate::openai::ChatCompletionRequest;

use super::compact::openai_error_response;

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    prio: Option<axum::extract::Extension<crate::main_modules::middleware::RequestPriority>>,
    policy: Option<axum::extract::Extension<crate::compaction::ContextPolicy>>,
    body: axum::body::Bytes,
) -> Response {
    // Parse the body ourselves (instead of using axum's `Json`
    // extractor) so the same bytes can feed both the deserialized
    // handler path and the `--dump` raw-capture path without
    // cloning the struct or cascading `Serialize` through every
    // request type.
    let req: ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };

    // --dump: record the incoming request body verbatim.
    let dump_seq = state.dump_writer.as_ref().and_then(|d| {
        match serde_json::from_slice::<serde_json::Value>(&body) {
            Ok(v) => {
                let seq = d.next_seq();
                d.dump_request("/v1/chat/completions", seq, &v);
                Some(seq)
            }
            Err(_) => None,
        }
    });

    // Lazy multi-model swap (Task 6 Phase 3): make the requested model the
    // active one BEFORE the per-request ModelRuntime snapshot inside
    // `chat_completions_inner` — otherwise this request would render with
    // the previous model's tokenizer.
    // accepted_processing_time: estimate + reserve BEFORE anything is
    // enqueued, so a budget the server cannot meet is a synchronous 503.
    let admission = match super::admission::admit(
        &state,
        &req.model,
        crate::arena_sched::WorkProfile {
            prompt_tokens: super::admission::prompt_tokens_guess(&req.messages),
            output_tokens: req.max_tokens,
            pages: 0,
        },
        req.accepted_processing_time,
    ) {
        Ok(a) => a,
        Err(r) => return super::admission::openai_refusal(&req.model, r),
    };
    let est_header = admission.header_secs(&state);
    let priority = prio.map(|p| p.0.0).unwrap_or(false);
    let policy = policy.map(|p| p.0);
    let resp = match super::lazy_swap::prepare_model_gate(
        &state,
        &req.model,
        crate::main_modules::hotswap::ModelCategory::Generative,
        priority,
        req.stream,
    )
    .await
    {
        Ok(super::lazy_swap::GatePass::Ready(ms)) => {
            let resp = chat_completions_inner(state, req_ctx, req, dump_seq, policy).await;
            super::lazy_swap::attach_swap_header(resp, ms)
        }
        Ok(super::lazy_swap::GatePass::Wait(ticket)) if req.stream => {
            // WAIT-1/2/5: answer 200 + SSE immediately; the real handler
            // runs once the dispatcher grants the ticket. The reservation
            // rides along until the inner handler returns.
            let keepalive = state.sse_keepalive;
            super::queued_sse::respond(
                ticket,
                keepalive,
                Box::pin(async move {
                    let _hold = admission;
                    chat_completions_inner(state, req_ctx, req, dump_seq, policy).await
                }),
            )
        }
        Ok(super::lazy_swap::GatePass::Wait(ticket)) => match ticket.wait().await {
            crate::arena_sched::TicketOutcome::Granted => {
                let ms = ticket.waited().as_millis() as u64;
                let resp = chat_completions_inner(state, req_ctx, req, dump_seq, policy).await;
                super::lazy_swap::attach_swap_header(resp, Some(ms))
            }
            outcome => super::lazy_swap::wait_outcome_response(outcome, &req.model),
        },
        Err(resp) => resp,
    };
    super::admission::attach_estimate_header(resp, est_header)
}

/// Internal entry for the parsed-request path. Called by
/// [`chat_completions`] after body capture, and by the Responses
/// API adapter (which builds a `ChatCompletionRequest` in-memory
/// and skips HTTP body bytes). `dump_seq` is `Some` only on the
/// public handler path. `policy` is the request's context policy
/// (CTX-1); `None` means the operator default.
///
/// E7 flow: the message chain over the client's history is computed
/// first and the latest matching compaction entry applied (CMP-2); the
/// prompt is then rendered once. If it does not fit the window (fit
/// rule in `CompactionConfig::fits`) the policy decides: `reject` →
/// 400, otherwise a new entry is created synchronously (CMP-10) and the
/// prompt rendered again; a streaming client receives `: compacting`
/// keep-alives meanwhile (WAIT-2).
pub(crate) async fn chat_completions_inner(
    state: Arc<AppState>,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    mut req: ChatCompletionRequest,
    dump_seq: Option<u64>,
    policy: Option<ContextPolicy>,
) -> Response {
    crate::metrics::REQUESTS_TOTAL.inc();
    crate::metrics::REQUESTS_ACTIVE.inc();

    // One consistent per-model snapshot for this whole request (Task 6
    // Phase 1.1) — a hot-swap mid-request must not mix two models' state.
    let rt = state.runtime();

    // ── Input validation + cross-turn F-feature guards ──
    if let Err(resp) = super::chat_phases::validate_input(&req) {
        return resp;
    }

    // Tool-active gating.
    let tools_active = rt.tool_call_parser.is_some()
        && req.tools.as_ref().is_some_and(|t| !t.is_empty())
        && !req.tool_choice.as_ref().is_some_and(|tc| tc.is_none());

    let policy = policy.unwrap_or(state.compaction.default_policy);
    let mut cmp = RequestCompaction::new(&state, &rt, &mut req, policy);

    let mut prepared = match prepare::prepare_prompt(&state, &rt, &mut req, tools_active) {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    let window = state.max_seq_len();
    let prompt_len = prepared.prompt_tokens.len();
    let fits = match policy {
        // CTX-2: `reject` keeps the pre-E7 rule (output is clamped).
        ContextPolicy::Reject => prompt_len < window,
        _ => state.compaction.fits(prompt_len, req.max_tokens, window),
    };
    if !fits {
        if policy == ContextPolicy::Reject {
            return overflow::reject_response(&state, prompt_len);
        }
        if req.stream {
            let keepalive = state.sse_keepalive;
            return super::queued_sse::respond_while_compacting(
                keepalive,
                Box::pin(async move {
                    let prepared = match overflow::resolve_overflow(
                        &state,
                        &rt,
                        &mut cmp,
                        &mut req,
                        tools_active,
                    )
                    .await
                    {
                        Ok(p) => p,
                        Err(resp) => return resp,
                    };
                    let info = cmp.info();
                    let resp = finish::finish_request(
                        state,
                        rt,
                        req_ctx,
                        req,
                        dump_seq,
                        tools_active,
                        prepared,
                    )
                    .await;
                    overflow::attach_compaction_headers(resp, info.as_ref())
                }),
            );
        }
        prepared =
            match overflow::resolve_overflow(&state, &rt, &mut cmp, &mut req, tools_active).await {
                Ok(p) => p,
                Err(resp) => return resp,
            };
    }

    let info = cmp.info();
    let resp =
        finish::finish_request(state, rt, req_ctx, req, dump_seq, tools_active, prepared).await;
    overflow::attach_compaction_headers(resp, info.as_ref())
}

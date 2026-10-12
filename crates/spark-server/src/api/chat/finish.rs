// SPDX-License-Identifier: AGPL-3.0-only

//! Phases 6–7 of the chat pipeline: sampling resolution, the output
//! clamp to the remaining window, and the streaming / blocking dispatch.

use std::sync::Arc;

use axum::response::Response;

use super::prepare::Prepared;
use super::sampling_setup;
use crate::AppState;
use crate::main_modules::app_state::ModelRuntime;
use crate::openai::ChatCompletionRequest;

pub(super) async fn finish_request(
    state: Arc<AppState>,
    rt: Arc<ModelRuntime>,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    req: ChatCompletionRequest,
    dump_seq: Option<u64>,
    tools_active: bool,
    prepared: Prepared,
) -> Response {
    let Prepared {
        cwd_hint,
        image_pixels,
        prompt_tokens,
        enable_thinking,
        thinking_budget,
        loops,
    } = prepared;
    let suppress_tool_call = loops.suppress_tool_call;
    let tool_call_repeat_count = loops.tool_call_repeat_count;

    let session_hash = crate::session_manager::compute_session_hash(&prompt_tokens);
    let tools_count = req.tools.as_ref().map_or(0, |t| t.len());
    tracing::info!(
        "Session {session_hash:#x}: {prompt_tokens} prompt tokens, tools={tools_active} ({tools_count} defined)",
        prompt_tokens = prompt_tokens.len()
    );
    let prompt_len = prompt_tokens.len();
    // The orchestrator guarantees `prompt_len < max_seq_len` (fit rule or
    // the `reject` policy's legacy check); keep the invariant explicit.
    debug_assert!(prompt_len < state.max_seq_len);

    // ── Phase 6: sampling preset / stop / grammar / timeout ─────
    let sampling_setup::SamplingSetup {
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        lz_penalty,
        logit_bias,
        max_tokens,
        stop_tokens,
        tool_choice_required,
        grammar_spec,
        timeout_at,
        top_logprobs,
    } = match sampling_setup::build_sampling(
        &state,
        &rt,
        &req,
        enable_thinking,
        tools_active,
        suppress_tool_call,
        tool_call_repeat_count,
    ) {
        Ok(s) => s,
        Err(resp) => return resp,
    };

    // Clamp the generation budget to the context that is actually left
    // (Plan.md 2.1). Nothing downstream bounds prompt + max_tokens: an
    // unclamped budget runs the sequence past the context window until
    // the KV pool starves — which kills the whole decode batch with an
    // untyped stream error instead of a clean `finish_reason: "length"`.
    let requested_max_tokens = max_tokens;
    let max_tokens = max_tokens.min(state.max_seq_len.saturating_sub(prompt_len).max(1));
    if max_tokens < requested_max_tokens {
        tracing::info!(
            requested = requested_max_tokens,
            clamped = max_tokens,
            prompt_len,
            "max_tokens clamped to remaining context (max_seq_len {})",
            state.max_seq_len
        );
    }

    // ── Phase 7: dispatch streaming or blocking ─────────────────
    if req.stream {
        return super::super::chat_stream_dispatch::dispatch_streaming(
            state,
            rt,
            &req,
            req_ctx,
            dump_seq,
            prompt_tokens,
            session_hash,
            image_pixels,
            max_tokens,
            temperature,
            top_k,
            top_p,
            top_n_sigma,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            lz_penalty,
            logit_bias.clone(),
            enable_thinking,
            thinking_budget,
            tools_active,
            tool_choice_required,
            suppress_tool_call,
            cwd_hint.clone(),
            stop_tokens,
            grammar_spec.clone(),
            top_logprobs,
            timeout_at,
        )
        .await;
    }

    super::super::chat_blocking::run_blocking_path(super::super::chat_blocking::BlockingPathArgs {
        state,
        rt,
        req,
        req_ctx,
        dump_seq,
        prompt_tokens,
        session_hash,
        image_pixels,
        max_tokens,
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        lz_penalty,
        logit_bias,
        stop_tokens,
        enable_thinking,
        thinking_budget,
        tools_active,
        tool_choice_required,
        suppress_tool_call,
        grammar_spec,
        top_logprobs,
        timeout_at,
        cwd_hint,
        prompt_len,
    })
    .await
}

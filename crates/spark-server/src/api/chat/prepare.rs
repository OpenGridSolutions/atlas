// SPDX-License-Identifier: AGPL-3.0-only

//! Phases 1–5 of the chat pipeline, bundled so the orchestrator can run
//! them again after a context compaction rewrote `req.messages`:
//! tool-prompt injection, `MsgEntry` build + image preprocessing,
//! default template kwargs, thinking resolution, loop detection and the
//! Jinja render that yields the prompt tokens.

use std::sync::Arc;

use axum::response::Response;

use super::{loop_detect, msg_entry, template, thinking};
use crate::AppState;
use crate::main_modules::app_state::ModelRuntime;
use crate::openai::ChatCompletionRequest;

/// Everything the dispatch phase needs from the prompt build.
pub(super) struct Prepared {
    pub(super) cwd_hint: Option<String>,
    pub(super) image_pixels: Vec<(Vec<f32>, usize, usize)>,
    pub(super) prompt_tokens: Vec<u32>,
    pub(super) enable_thinking: bool,
    pub(super) thinking_budget: Option<u32>,
    pub(super) loops: loop_detect::LoopDetectOut,
}

#[allow(clippy::result_large_err)]
pub(super) fn prepare_prompt(
    state: &Arc<AppState>,
    rt: &ModelRuntime,
    req: &mut ChatCompletionRequest,
    tools_active: bool,
) -> Result<Prepared, Response> {
    // ST-995 fix: restore the parser-specific behavioral system prompt #90 removed.
    // For the hermes parser this is the canonical NousResearch function-calling
    // prompt ("you MAY call one or more functions... don't make assumptions"),
    // which the GDN model needs to correctly DECLINE on irrelevance prompts. With
    // it (and compact tool-JSON) hallucination returns to ~96 (vs 30/64 without).
    // (The 2026-05-25 attempt to inject the qwen3_coder prompt as well was
    // reverted: two competing tool-format guidances made the model emit
    // literal `<tool_call>` XML as content.)
    if tools_active && let Some(ref parser) = rt.tool_call_parser {
        let default_choice = crate::tool_parser::ToolChoice::Mode("auto".to_string());
        let tool_choice = req.tool_choice.as_ref().unwrap_or(&default_choice);
        let tool_prompt = parser.system_prompt(req.tools.as_deref().unwrap_or(&[]), tool_choice);
        if let Some(first) = req.messages.first_mut().filter(|m| m.role == "system") {
            first.content.text = format!("{}\n\n{}", tool_prompt, first.content.text);
        } else {
            req.messages.insert(
                0,
                crate::openai::IncomingMessage::synthetic_system(tool_prompt),
            );
        }
    }

    tracing::info!(
        "Request: model={}, messages={}, tools={}, tools_active={}, tool_choice={:?}, stream={}, temp={:?}, max_tokens={}, freq_pen={:?}, rep_pen={:?}",
        req.model,
        req.messages.len(),
        req.tools.as_ref().map_or(0, |t| t.len()),
        tools_active,
        req.tool_choice,
        req.stream,
        req.temperature,
        req.max_tokens,
        req.frequency_penalty,
        req.repetition_penalty,
    );

    // ── Phase 1: build MsgEntry vec + image preprocess + cwd ────
    let msg_entry::BuildOut {
        messages,
        cwd_hint,
        image_pixels,
        image_pad_counts,
    } = msg_entry::build_msg_entries(state, rt, req, tools_active)?;

    // ── Phase 1.5: merge server-level chat_template_kwargs default ─
    // When the client sends no thinking parameters and the server has a
    // --default-chat-template-kwargs flag set, inject those kwargs into
    // the request so the existing resolve_thinking() chain sees them as
    // normal request-body fields.
    if let Some(ref default_kw) = state.default_chat_template_kwargs
        && !req.thinking_explicitly_requested()
    {
        req.chat_template_kwargs = Some(default_kw.clone());
    }

    // ── Phase 2: thinking resolution (pre-template) ─────────────
    let (enable_thinking, thinking_budget) =
        thinking::resolve_thinking(state, rt, req, tools_active);

    // ── Phase 4: generic loop / spinning detection + task pin ───
    let loops = loop_detect::check_loops(req, tools_active);

    // ── Phase 5: render Jinja template + image-pad expansion ────
    let template::TemplateOut {
        prompt_tokens,
        enable_thinking,
        thinking_budget,
    } = template::render_template(
        state,
        rt,
        req,
        &messages,
        &image_pad_counts,
        enable_thinking,
        thinking_budget,
        tools_active,
    )?;

    Ok(Prepared {
        cwd_hint,
        image_pixels,
        prompt_tokens,
        enable_thinking,
        thinking_budget,
        loops,
    })
}

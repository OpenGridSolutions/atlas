// SPDX-License-Identifier: AGPL-3.0-only
//
// Chat-template rendering: build the JSON-message array, run the
// optional auto-compact, apply the Jinja template, expand image
// pad-tokens, and detect template-forced thinking.
//
// Lifted out of `chat::chat_completions_inner` (wave 4g).

use axum::http::StatusCode;
use axum::response::Response;
use std::sync::Arc;

use crate::openai::ChatCompletionRequest;
use crate::{AppState, ModelRuntime};

use super::super::compact::openai_error_response;
use super::msg_entry::MsgEntry;

/// Outputs of [`render_template`]. Threaded into the streaming /
/// blocking dispatch.
pub(super) struct TemplateOut {
    pub(super) prompt_tokens: Vec<u32>,
    /// Possibly overridden by template-forced-thinking detection.
    pub(super) enable_thinking: bool,
    pub(super) thinking_budget: Option<u32>,
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::result_large_err)]
pub(super) fn render_template(
    _state: &Arc<AppState>,
    rt: &ModelRuntime,
    req: &ChatCompletionRequest,
    messages: &[MsgEntry],
    image_pad_counts: &[usize],
    enable_thinking: bool,
    thinking_budget: Option<u32>,
    tools_active: bool,
) -> Result<TemplateOut, Response> {
    // Use closed thinking when client doesn't explicitly enable it.
    let template_thinking = enable_thinking;

    // Build JSON messages with structured tool_calls for Jinja.
    let json_messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| {
            let content_val = if m.image_count > 0 {
                let mut items: Vec<serde_json::Value> = Vec::with_capacity(m.image_count + 1);
                for _ in 0..m.image_count {
                    items.push(serde_json::json!({"type": "image"}));
                }
                if !m.content.is_empty() {
                    items.push(serde_json::json!({"type": "text", "text": m.content}));
                }
                serde_json::Value::Array(items)
            } else {
                serde_json::Value::String(m.content.clone())
            };
            let mut msg = serde_json::json!({"role": m.role, "content": content_val});
            if let Some(ref tcs) = m.tool_calls {
                msg["tool_calls"] = serde_json::Value::Array(tcs.clone());
            }
            // F1: forward historical reasoning trace to the Jinja
            // template. The template at qwen3_5_moe.jinja:90-104
            // reads `message.reasoning_content` and rehydrates the
            // `<think>` block for the historical assistant turns.
            // Without this, every historical assistant message
            // rendered an empty `<think>\n\n</think>\n\n` wrapper
            // (empty-think poisoning, → premature `<|im_end|>`).
            if let Some(ref rc) = m.reasoning_content {
                msg["reasoning_content"] = serde_json::Value::String(rc.clone());
            }
            msg
        })
        .collect();
    // When TSCG is enabled the parser's `system_prompt()` has already
    // placed the compact tool signatures into messages[0]; passing
    // `tools` to Jinja as well would re-render the full JSON schema and
    // defeat the compaction. Pass `None` so the template's `{% if tools
    // %}` branch falls through — the tool-call format instructions
    // still come from `system_prompt()`.
    let jinja_tools: Option<Vec<serde_json::Value>> =
        if tools_active && !crate::tscg::tscg_enabled() {
            req.tools.as_ref().map(|ts| {
                ts.iter()
                    .map(|t| serde_json::to_value(t).unwrap_or_default())
                    .collect()
            })
        } else {
            None
        };

    let prompt_tokens = match rt.tokenizer.apply_chat_template_openai(
        &json_messages,
        jinja_tools.as_deref(),
        template_thinking,
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

    // Expand image pads when needed.
    let prompt_tokens = if image_pad_counts.iter().any(|&c| c > 1) {
        rt.tokenizer
            .expand_image_pads(prompt_tokens, image_pad_counts)
    } else {
        prompt_tokens
    };

    // Template-forced thinking detection.
    let (enable_thinking, thinking_budget) = if let Some(think_start) = rt.think_start_token_id {
        let tail = &prompt_tokens[prompt_tokens.len().saturating_sub(8)..];
        let last_start = tail.iter().rposition(|t| *t == think_start);
        let has_unclosed_think = match (last_start, rt.think_end_token_id) {
            (Some(si), Some(end_tok)) => !tail[si + 1..].contains(&end_tok),
            (Some(_), None) => true,
            (None, _) => false,
        };
        if has_unclosed_think && !enable_thinking {
            tracing::info!(
                "Template-forced thinking detected (unclosed \\<think\\> in prompt tail) — \
                 overriding enable_thinking=true with budget={}",
                rt.behavior.max_thinking_budget,
            );
            (true, Some(rt.behavior.max_thinking_budget))
        } else {
            (enable_thinking, thinking_budget)
        }
    } else {
        (enable_thinking, thinking_budget)
    };

    Ok(TemplateOut {
        prompt_tokens,
        enable_thinking,
        thinking_budget,
    })
}

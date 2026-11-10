// SPDX-License-Identifier: AGPL-3.0-only

//! Summary generation (CMP-7, CMP-3): the loaded arena model writes the
//! summary at temperature 0 with at most `summary_max_tokens`. The input
//! is the previous summary plus the messages that fall out of the
//! window; when it does not fit the window it is folded in several
//! cumulative rounds instead of being cut, so nothing is silently lost.

use std::sync::Arc;

use anyhow::{Context, Result};

use super::CompactionConfig;
use crate::AppState;
use crate::api::inference_types::InferenceRequest;
use crate::main_modules::app_state::ModelRuntime;
use crate::openai::IncomingMessage;

/// Built-in summarisation prompt (`--compaction-prompt-file` overrides).
pub const DEFAULT_PROMPT: &str = "You are compacting the earlier part of a long conversation so that it can \
continue within a limited context window. Write a dense summary that preserves: facts and numbers; \
decisions and the reasons behind them; code identifiers (file paths, function names, flags, commands, \
error messages) verbatim; open tasks, unanswered questions and pending requests; constraints and \
preferences the user stated; and the content of any images or documents that were described. \
If a previous summary is given, fold it and the new messages into one updated summary and keep \
everything from the previous summary that is still relevant. Write in the language of the \
conversation. Output only the summary, without preamble or commentary.";

/// Template + prompt overhead reserved on top of the folded messages.
const PROMPT_OVERHEAD_TOKENS: usize = 512;
/// Rough chars-per-token used only when a single message must be
/// clipped to fit one summarisation round.
const CHARS_PER_TOKEN: usize = 4;

/// Plain-text rendering of one message for the summariser.
pub fn render_message(m: &IncomingMessage) -> String {
    let mut s = String::new();
    s.push_str("### ");
    s.push_str(&m.role);
    if let Some(name) = &m.name {
        s.push_str(" (");
        s.push_str(name);
        s.push(')');
    }
    s.push('\n');
    if !m.content.text.is_empty() {
        s.push_str(&m.content.text);
        s.push('\n');
    }
    for _ in &m.content.images {
        s.push_str("[image attached; its content is not available to the summariser]\n");
    }
    if let Some(tcs) = &m.tool_calls {
        for tc in tcs {
            s.push_str("[tool call ");
            s.push_str(&tc.function.name);
            s.push('(');
            s.push_str(&tc.function.arguments);
            s.push_str(")]\n");
        }
    }
    s
}

fn user_turn(previous: Option<&str>, rendered: &[String]) -> String {
    let mut s = String::new();
    match previous {
        Some(p) if !p.trim().is_empty() => {
            s.push_str("<previous_summary>\n");
            s.push_str(p.trim());
            s.push_str("\n</previous_summary>\n\n");
        }
        _ => {}
    }
    s.push_str("<messages>\n");
    for r in rendered {
        s.push_str(r);
        s.push('\n');
    }
    s.push_str("</messages>\n\nWrite the updated summary.");
    s
}

/// Strip a leading `<think>…</think>` block a template-forced thinking
/// model may emit before the summary.
pub fn strip_think(s: &str) -> String {
    let t = s.trim_start();
    if let Some(rest) = t.strip_prefix("<think>")
        && let Some(end) = rest.find("</think>")
    {
        return rest[end + "</think>".len()..].trim().to_string();
    }
    s.trim().to_string()
}

/// Fold `msgs` (with per-message token sizes `sizes`) into `previous`.
pub async fn summarize(
    state: &AppState,
    rt: &ModelRuntime,
    cfg: &CompactionConfig,
    previous: Option<&str>,
    msgs: &[IncomingMessage],
    sizes: &[usize],
) -> Result<String> {
    let window = state.max_seq_len();
    let mut summary: Option<String> = previous.map(str::to_string);
    let mut i = 0;
    while i < msgs.len() {
        // Budget for this round: window − output − prompt/template
        // overhead − previous summary (bounded by its token budget).
        let prev_tokens = summary.as_deref().map_or(0, |s| s.len() / CHARS_PER_TOKEN);
        let budget = window
            .saturating_sub(cfg.summary_max_tokens)
            .saturating_sub(PROMPT_OVERHEAD_TOKENS)
            .saturating_sub(prev_tokens.min(cfg.summary_max_tokens * 2))
            .max(256);
        let mut rendered = Vec::new();
        let mut used = 0;
        while i < msgs.len() && (rendered.is_empty() || used + sizes[i] <= budget) {
            let mut r = render_message(&msgs[i]);
            if rendered.is_empty() && sizes[i] > budget {
                // One message alone exceeds a round: clip it (CMP-3
                // keeps the rest cumulative; this is the only lossy
                // step and it is logged).
                let keep = (budget * CHARS_PER_TOKEN).min(r.len());
                let cut = r
                    .char_indices()
                    .map(|(b, _)| b)
                    .take_while(|&b| b <= keep)
                    .last()
                    .unwrap_or(0);
                tracing::warn!(
                    message_tokens = sizes[i],
                    budget,
                    "compaction: single message exceeds one summarisation round; clipped"
                );
                r.truncate(cut);
                r.push_str("\n[message clipped]\n");
            }
            used += sizes[i];
            rendered.push(r);
            i += 1;
        }
        let turn = user_turn(summary.as_deref(), &rendered);
        summary = Some(generate(state, rt, cfg, &turn).await?);
    }
    Ok(summary.unwrap_or_default())
}

async fn generate(
    state: &AppState,
    rt: &ModelRuntime,
    cfg: &CompactionConfig,
    user_turn: &str,
) -> Result<String> {
    let messages = vec![
        serde_json::json!({"role": "system", "content": &*cfg.prompt}),
        serde_json::json!({"role": "user", "content": user_turn}),
    ];
    let prompt_tokens = rt
        .tokenizer
        .apply_chat_template_openai(&messages, None, false, rt.behavior.disable_tool_steering)
        .context("render summarisation prompt")?;
    let prompt_len = prompt_tokens.len();
    // The window is per model: read it from the caller's snapshot.
    let window = rt.max_seq_len;
    anyhow::ensure!(
        prompt_len < window,
        "summarisation prompt ({prompt_len} tokens) exceeds the window ({window})"
    );
    let max_tokens = cfg.summary_max_tokens.min(window - prompt_len).max(1);
    let stop_tokens: Vec<u32> = rt
        .tokenizer
        .inner()
        .token_to_id("<end_of_turn>")
        .into_iter()
        .collect();
    let (tx, rx) = tokio::sync::oneshot::channel();
    let session_hash = crate::session_manager::compute_session_hash(&prompt_tokens);
    let request = InferenceRequest::Blocking {
        prompt_tokens: Arc::new(prompt_tokens),
        session_hash,
        image_pixels: Vec::new(),
        max_tokens,
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
        stop_tokens,
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
        timeout_at: None,
        logit_probe: None,
        capture_hidden: false,
        response_tx: tx,
    };
    state
        .request_tx
        .send(request)
        .await
        .map_err(|_| anyhow::anyhow!("scheduler queue closed"))?;
    let response = rx
        .await
        .context("summariser response channel closed")?
        .context("summarisation inference failed")?;
    let text = rt.tokenizer.decode(&response.output_tokens)?;
    let text = strip_think(&text);
    anyhow::ensure!(!text.is_empty(), "summariser produced an empty summary");
    Ok(text)
}

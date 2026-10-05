// SPDX-License-Identifier: AGPL-3.0-only

//! `POST /v1/translate` — text translation via TranslateGemma (multiclass
//! Plan.md Phase T).
//!
//! The request is STRUCTURED (`source_lang_code`, `target_lang_code`,
//! `text`); the server renders the TranslateGemma prompt frame itself,
//! byte-for-byte what the checkpoint's own `chat_template.jinja` produces
//! for a single text user turn. The generic chat-template path is
//! deliberately bypassed: the shipped template `raise_exception`s on any
//! standard OpenAI-shaped message ("User role must provide `content` as
//! an iterable with exactly one item"), so /v1/chat/completions-style
//! requests CANNOT drive this model — the forced template is the only
//! path (and the model is category-locked to this endpoint anyway).
//!
//! Language names come from the `languages` map embedded at the top of
//! the checkpoint's chat_template.jinja, parsed once per model dir and
//! cached — no hardcoded language list to drift out of sync.
//!
//! Context limit: the TRANSLATION task is trained at ~2K tokens. Rendered
//! prompts over `PROMPT_TOKEN_LIMIT` are REJECTED with a clear 400
//! (chunking would splice sentences and silently degrade quality —
//! documented decision, Plan.md Phase T).

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

/// Rendered-prompt budget (tokens). The task context is 2K; leave room
/// for the translation itself (target languages can run ~1.5× longer
/// than the source).
const PROMPT_TOKEN_LIMIT: usize = 1800;

#[derive(Deserialize)]
pub struct TranslateRequest {
    pub source_lang_code: String,
    pub target_lang_code: String,
    pub text: String,
    /// Optional: must name the TRANSLATE-category model when set. Empty
    /// = the (single) registered translate model.
    #[serde(default)]
    pub model: String,
    /// Cap on generated tokens (default 1024, clamped to context).
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

#[derive(Serialize)]
pub struct TranslateResponse {
    pub text: String,
    pub model: String,
    pub source_lang: String,
    pub target_lang: String,
    pub usage: TranslateUsage,
}

#[derive(Serialize)]
pub struct TranslateUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

/// code → language name, parsed from the checkpoint's own
/// chat_template.jinja `{%- set languages = {...} %}` header. Cached per
/// process (the map is identical across TranslateGemma revisions; a
/// second translate model would need a keyed cache — YAGNI for now).
static LANGUAGES: std::sync::OnceLock<std::collections::HashMap<String, String>> =
    std::sync::OnceLock::new();

fn language_map(
    model_id: &str,
) -> anyhow::Result<&'static std::collections::HashMap<String, String>> {
    if let Some(m) = LANGUAGES.get() {
        return Ok(m);
    }
    let model_dir = crate::model_resolver::resolve_model_dir(model_id, None)?;
    let tpl = std::fs::read_to_string(model_dir.join("chat_template.jinja"))
        .map_err(|e| anyhow::anyhow!("reading chat_template.jinja: {e}"))?;
    let start = tpl
        .find("set languages")
        .and_then(|i| tpl[i..].find('{').map(|j| i + j))
        .ok_or_else(|| anyhow::anyhow!("chat_template.jinja has no `set languages = {{` header"))?;
    let end = start
        + tpl[start..]
            .find("%}")
            .ok_or_else(|| anyhow::anyhow!("unterminated languages block"))?;
    // The block is a sequence of "code": "Name" pairs — collect the
    // quoted strings pairwise (tolerates trailing commas that would
    // break a strict JSON parse).
    let mut strings: Vec<String> = Vec::new();
    let mut cur: Option<String> = None;
    for c in tpl[start..end].chars() {
        match (&mut cur, c) {
            (None, '"') => cur = Some(String::new()),
            (Some(s), '"') => {
                strings.push(std::mem::take(s));
                cur = None;
            }
            (Some(s), c) => s.push(c),
            (None, _) => {}
        }
    }
    if strings.len() < 2 || !strings.len().is_multiple_of(2) {
        anyhow::bail!(
            "could not parse the languages map ({} quoted strings)",
            strings.len()
        );
    }
    let map: std::collections::HashMap<String, String> = strings
        .chunks(2)
        .map(|p| (p[0].clone(), p[1].clone()))
        .collect();
    tracing::info!(
        "TranslateGemma language map loaded: {} codes from chat_template.jinja",
        map.len()
    );
    Ok(LANGUAGES.get_or_init(|| map))
}

/// The exact single-turn text prompt the checkpoint's chat template
/// renders (minus the leading `bos_token`, which is prepended as a token
/// id after encoding — the tokenizer encodes with
/// `add_special_tokens=false`).
fn render_prompt(
    source_lang: &str,
    source_code: &str,
    target_lang: &str,
    target_code: &str,
    text: &str,
) -> String {
    format!(
        "<start_of_turn>user\n\
         You are a professional {source_lang} ({source_code}) to {target_lang} ({target_code}) \
         translator. Your goal is to accurately convey the meaning and nuances of the original \
         {source_lang} text while adhering to {target_lang} grammar, vocabulary, and cultural \
         sensitivities.\n\
         Produce only the {target_lang} translation, without any additional explanations or \
         commentary. Please translate the following {source_lang} text into {target_lang}:\
         \n\n\n{}<end_of_turn>\n<start_of_turn>model\n",
        text.trim(),
    )
}

pub async fn translate(
    State(state): State<Arc<AppState>>,
    req: Result<Json<TranslateRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "Invalid request JSON: {e} — expected \
                     {{\"source_lang_code\", \"target_lang_code\", \"text\"}}"
                ),
            );
        }
    };
    if req.text.trim().is_empty() {
        return openai_error_response(StatusCode::BAD_REQUEST, "'text' is empty".into());
    }

    // Resolve the model: explicit id or the single TRANSLATE-category
    // registration.
    let model_id = if req.model.is_empty() {
        let Some(id) = state.hotswap.as_ref().and_then(|ctx| {
            ctx.registry
                .iter()
                .find(|m| m.category == ModelCategory::Translate)
                .map(|m| m.model_id.clone())
        }) else {
            return openai_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "no translate model registered — add one to ATLAS_PRESEED_TRANSLATE".into(),
            );
        };
        id
    } else {
        req.model.clone()
    };

    // Language map + code validation BEFORE the (potentially expensive)
    // model swap.
    let languages = match language_map(&model_id) {
        Ok(m) => m,
        Err(e) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("language map unavailable: {e:#}"),
            );
        }
    };
    let norm = |code: &str| code.replace('_', "-");
    let source_code = norm(&req.source_lang_code);
    let target_code = norm(&req.target_lang_code);
    let Some(source_lang) = languages.get(&source_code) else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "unknown source_lang_code '{source_code}' — use a BCP-47 code from the \
                 model's language list (e.g. \"de\", \"en\", \"fr\", \"pt-BR\")"
            ),
        );
    };
    let Some(target_lang) = languages.get(&target_code) else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "unknown target_lang_code '{target_code}' — use a BCP-47 code from the \
                 model's language list (e.g. \"de\", \"en\", \"fr\", \"pt-BR\")"
            ),
        );
    };

    // Category gate + lazy swap (Plan.md Phase K).
    let swap_ms = match super::lazy_swap::ensure_model_active(
        &state,
        &model_id,
        ModelCategory::Translate,
        false,
    )
    .await
    {
        Ok(ms) => ms,
        Err(resp) => return resp,
    };

    let rt = state.runtime();
    let prompt = render_prompt(
        source_lang,
        &source_code,
        target_lang,
        &target_code,
        &req.text,
    );
    let mut prompt_tokens = match rt.tokenizer.encode(&prompt) {
        Ok(t) => t,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Tokenization error: {e}"),
            );
        }
    };
    // The template starts with `{{ bos_token }}`; encode() adds no
    // special tokens, so prepend <bos> explicitly.
    if let Some(bos) = rt.tokenizer.inner().token_to_id("<bos>") {
        prompt_tokens.insert(0, bos);
    }
    let prompt_len = prompt_tokens.len();
    if prompt_len > PROMPT_TOKEN_LIMIT {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "input too long for the translation task: rendered prompt is {prompt_len} \
                 tokens, limit {PROMPT_TOKEN_LIMIT} (TranslateGemma is trained at ~2K \
                 context) — split the text at sentence/paragraph boundaries and translate \
                 the parts separately"
            ),
        );
    }

    let max_tokens = req
        .max_tokens
        .unwrap_or(1024)
        .min(state.max_seq_len.saturating_sub(prompt_len));
    // <end_of_turn> terminates the model turn (generation_config eos
    // [1, 106]; the scheduler's implicit eos covers id 1).
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
        // Deterministic translation: greedy.
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
    let output_text = match rt.tokenizer.decode(&response.output_tokens) {
        Ok(t) => t,
        Err(e) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Decode error: {e}"),
            );
        }
    };
    // Belt-and-braces: strip a decoded end-of-turn marker if the
    // tokenizer surfaced it.
    let translation = output_text
        .trim_end()
        .trim_end_matches("<end_of_turn>")
        .trim()
        .to_string();

    let resp = Json(TranslateResponse {
        text: translation,
        model: model_id,
        source_lang: source_lang.clone(),
        target_lang: target_lang.clone(),
        usage: TranslateUsage {
            prompt_tokens: prompt_len,
            completion_tokens: response.output_tokens.len(),
        },
    })
    .into_response();
    super::lazy_swap::attach_swap_header(resp, swap_ms)
}

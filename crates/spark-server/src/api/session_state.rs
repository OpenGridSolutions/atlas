// SPDX-License-Identifier: AGPL-3.0-only

//! `POST /v1/atlas/session/state` — KV-cache tracking for one session.
//!
//! The server is content-addressed (E2): there are no session IDs, so
//! the client presents its conversation (same shapes as `/tokenize`:
//! `messages` or a raw `prompt`) and the server reports what state that
//! session's KV would be in RIGHT NOW, across every tier:
//!
//! - **context**: how full the session is against the serving window
//!   (`--max-seq-len`) and how much room remains for new context + output.
//! - **ram_cache**: tokens the radix prefix cache would serve without
//!   prefill, plus the deepest RAM SSM resume point (hybrid models).
//! - **nvme**: persisted chain length in the block store, tail record
//!   presence, deepest persisted SSM resume point, and whether the
//!   extension beyond RAM clears `--nvme-restore-min-tokens` (BAT-7).
//!
//! Read-only by construction: only `peek_*` and index probes run —
//! never `lookup`/`insert`, which mutate refcounts and LRU state and
//! belong to the scheduler thread.
//!
//! Scope: the probe answers for the ACTIVE arena model only — token IDs
//! and chain seed are model-specific, and another model's tokenizer is
//! not loaded. Trust note: in a shared deployment this endpoint reveals
//! whether specific content is cached (content-addressed probing); it
//! sits behind the global bearer auth like every `/v1` route and the
//! deployment is single-trust-domain.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use super::compact::openai_error_response;
use crate::main_modules::AppState;
use crate::main_modules::hotswap::SwapState;
use spark_storage::block_store::Kind;

#[derive(Deserialize)]
pub struct SessionStateRequest {
    pub model: String,
    /// Chat-shaped conversation (rendered through the active chat
    /// template, without the generation prompt — the probe matches the
    /// shared prefix of a real request up to the last block).
    #[serde(default)]
    pub messages: Option<Vec<serde_json::Value>>,
    /// Raw prompt alternative (encoded verbatim).
    #[serde(default)]
    pub prompt: Option<String>,
    /// Planned output budget; folded into the slot-fit answer.
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

/// Index-only NVMe chain probe: persisted prefix length, tail record,
/// deepest persisted SSM resume point, and BAT-7 restore eligibility.
fn nvme_probe(
    store: &spark_storage::block_store::BlockStore,
    view: &spark_runtime::block_keys::BlockKeysView<'_>,
    total_tokens: usize,
    ram_cached: usize,
    restore_min_tokens: usize,
) -> serde_json::Value {
    let block_size = view.block_size().max(1);
    let keys = view.full_keys();
    let persisted_blocks = keys
        .iter()
        .take_while(|k| store.contains(&k.0, Kind::Kv))
        .count();
    let tail_persisted = view
        .tail_key()
        .map(|k| store.contains(&k.0, Kind::Tail))
        .unwrap_or(false);
    let mut ssm_resume = 0usize;
    for (i, key) in keys.iter().enumerate().take(persisted_blocks).rev() {
        if let Some(e) = store.entry(&key.0, Kind::Ssm) {
            ssm_resume = (e.token_count as usize).min((i + 1) * block_size);
            break;
        }
    }
    let persisted_tokens = persisted_blocks * block_size
        + if tail_persisted {
            total_tokens % block_size
        } else {
            0
        };
    let extension = persisted_tokens.saturating_sub(ram_cached);
    serde_json::json!({
        "persisted_tokens": persisted_tokens,
        "persisted_blocks": persisted_blocks,
        "tail_persisted": tail_persisted,
        "ssm_resume_tokens": ssm_resume,
        "restore_extension_tokens": extension,
        "restore_eligible": extension >= restore_min_tokens,
    })
}

pub async fn session_state(
    State(state): State<Arc<AppState>>,
    req: Result<Json<SessionStateRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };

    // The probe needs the active model's tokenizer + chain seed.
    let Some((params, cache)) = crate::main_modules::cache_registry::active() else {
        return openai_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "no model active yet (cache registry empty)".into(),
        );
    };
    let arena_running = state
        .hotswap
        .as_ref()
        .map(|c| c.state() == SwapState::Running)
        .unwrap_or(true);
    if !arena_running || req.model != params.model_id {
        return Json(serde_json::json!({
            "model": req.model,
            "active": false,
            "active_model": if arena_running { Some(params.model_id) } else { None },
            "note": "state probe requires the requested model to be the loaded arena \
                     model (token ids and chain keys are model-specific)",
        }))
        .into_response();
    }

    // ── Tokenize exactly like a request would (sans generation prompt) ──
    let rt = state.runtime();
    let mut media_present = false;
    let tokens: Vec<u32> = if let Some(ref prompt) = req.prompt {
        match rt.tokenizer.encode(prompt) {
            Ok(t) => t,
            Err(e) => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("tokenization error: {e:#}"),
                );
            }
        }
    } else if let Some(ref messages) = req.messages {
        let mut json_messages = Vec::with_capacity(messages.len());
        for m in messages {
            let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            // String content passes through; array content (multimodal)
            // keeps its text parts — the probe cannot evaluate media
            // digests, which the response flags via `media_evaluated`.
            let content = match m.get("content") {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(serde_json::Value::Array(parts)) => {
                    let mut text = String::new();
                    for p in parts {
                        match p.get("type").and_then(|v| v.as_str()) {
                            Some("text") => {
                                text.push_str(p.get("text").and_then(|v| v.as_str()).unwrap_or(""));
                            }
                            Some(_) => media_present = true,
                            None => {}
                        }
                    }
                    text
                }
                _ => String::new(),
            };
            json_messages.push(serde_json::json!({"role": role, "content": content}));
        }
        match rt.tokenizer.apply_chat_template_jinja(
            &json_messages,
            None,
            false,
            rt.behavior.disable_tool_steering,
        ) {
            Ok(t) => t,
            Err(e) => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("template/tokenization error: {e:#}"),
                );
            }
        }
    } else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "either 'messages' or 'prompt' is required".into(),
        );
    };

    // ── Chain + RAM probe ──
    let chain = spark_runtime::block_keys::BlockKeyChain::new(
        params.seed,
        params.block_size,
        &tokens,
        params.vision_pad_token_id,
        &[], // no digests from a text probe — media chains report uncacheable
    );
    let cacheable = chain.is_cacheable();
    let view = chain.view(&tokens, tokens.len());
    let session_hash = crate::session_manager::compute_session_hash(&tokens);
    let ram_cached = cache.peek_matched_tokens(&view);
    let ram_ssm = cache
        .peek_snapshot(&view, tokens.len(), session_hash)
        .map(|(_, tc)| tc)
        .unwrap_or(0);

    // ── NVMe probe (index-only, no file I/O) ──
    let nvme = crate::kv_tier::with_active(|tier| {
        nvme_probe(
            tier.store(),
            &view,
            tokens.len(),
            ram_cached,
            tier.restore_min_tokens,
        )
    });

    // ── Context fit: the per-sequence bound is the serving window
    //    (--max-seq-len); capacity across sequences is block-based ──
    let context_tokens = state.max_seq_len;
    let planned_output = req.max_tokens.unwrap_or(0);
    let remaining = context_tokens.saturating_sub(tokens.len());

    // Best reuse: RAM first; the NVMe extension counts when it clears
    // the restore threshold (BAT-7).
    let best_reuse = match &nvme {
        Some(n)
            if n["restore_eligible"].as_bool() == Some(true)
                && n["persisted_tokens"].as_u64().unwrap_or(0) as usize > ram_cached =>
        {
            n["persisted_tokens"].as_u64().unwrap_or(0) as usize
        }
        _ => ram_cached,
    };

    Json(serde_json::json!({
        "model": params.model_id,
        "active": true,
        "tokens": tokens.len(),
        "block_size": params.block_size,
        "cacheable": cacheable,
        "media_evaluated": !media_present && cacheable,
        "context": {
            "max_tokens": context_tokens,
            "source": "max_seq_len",
            "used_tokens": tokens.len(),
            "remaining_tokens": remaining,
            "fits_with_max_tokens": tokens.len() + planned_output <= context_tokens,
        },
        "ram_cache": {
            "cached_tokens": ram_cached,
            "cached_blocks": ram_cached / params.block_size,
            "ssm_resume_tokens": ram_ssm,
        },
        "nvme": nvme,
        "estimated_prefill_tokens": tokens.len().saturating_sub(best_reuse),
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::block_keys::{BlockKeyChain, seed};
    use spark_runtime::prefix_cache::PrefixCache;
    use spark_storage::block_store::{BlockStore, BlockStoreConfig};

    const BS: usize = 16;

    fn chain(tokens: &[u32]) -> BlockKeyChain {
        BlockKeyChain::new(seed(b"probe-model", &[1u8; 32]), BS, tokens, 0, &[])
    }

    fn store(dir: &std::path::Path) -> BlockStore {
        BlockStore::open(BlockStoreConfig {
            root: dir.to_path_buf(),
            model_hash: [7; 32],
            layout_hash: [9; 32],
            session_ttl: std::time::Duration::from_secs(3600),
            min_free_pct: 0,
            write_budget_per_day: None,
        })
        .unwrap()
    }

    #[test]
    fn nvme_probe_reports_chain_tail_and_ssm() {
        let dir = tempfile::tempdir().unwrap();
        let st = store(dir.path());
        // 40 tokens = 2 full blocks + 8-token tail.
        let tokens: Vec<u32> = (0..40).collect();
        let c = chain(&tokens);
        let view = c.view(&tokens, 40);
        let keys = view.full_keys().to_vec();

        // Persist both blocks, the tail, and an SSM record at block 1.
        st.put(Kind::Kv, keys[0].0, [0; 16], 16, &[1; 64]).unwrap();
        st.put(Kind::Kv, keys[1].0, keys[0].0, 16, &[2; 64])
            .unwrap();
        st.put(Kind::Ssm, keys[1].0, keys[1].0, 32, &[3; 64])
            .unwrap();
        let tail = view.tail_key().unwrap();
        st.put(Kind::Tail, tail.0, keys[1].0, 8, &[4; 64]).unwrap();

        let p = nvme_probe(&st, &view, 40, 0, 512);
        assert_eq!(p["persisted_blocks"], 2);
        assert_eq!(p["persisted_tokens"], 40); // 32 + 8-token tail
        assert_eq!(p["tail_persisted"], true);
        assert_eq!(p["ssm_resume_tokens"], 32);
        assert_eq!(p["restore_eligible"], false); // 40 < 512
    }

    #[test]
    fn nvme_probe_extension_counts_beyond_ram() {
        let dir = tempfile::tempdir().unwrap();
        let st = store(dir.path());
        let tokens: Vec<u32> = (0..64).collect(); // 4 blocks
        let c = chain(&tokens);
        let view = c.view(&tokens, 64);
        let keys = view.full_keys().to_vec();
        let mut pred = [0u8; 16];
        for k in &keys {
            st.put(Kind::Kv, k.0, pred, 16, &[0; 32]).unwrap();
            pred = k.0;
        }
        // RAM already covers 32 tokens → extension 32, eligible at min 32.
        let p = nvme_probe(&st, &view, 64, 32, 32);
        assert_eq!(p["persisted_tokens"], 64);
        assert_eq!(p["restore_extension_tokens"], 32);
        assert_eq!(p["restore_eligible"], true);
    }

    #[test]
    fn nvme_probe_stops_at_chain_gap() {
        let dir = tempfile::tempdir().unwrap();
        let st = store(dir.path());
        let tokens: Vec<u32> = (0..48).collect(); // 3 blocks
        let c = chain(&tokens);
        let view = c.view(&tokens, 48);
        let keys = view.full_keys().to_vec();
        // Persist blocks 0 and 2, but NOT 1 — the contiguous prefix ends at 1.
        st.put(Kind::Kv, keys[0].0, [0; 16], 16, &[0; 32]).unwrap();
        st.put(Kind::Kv, keys[2].0, keys[1].0, 16, &[0; 32])
            .unwrap();
        let p = nvme_probe(&st, &view, 48, 0, 16);
        assert_eq!(p["persisted_blocks"], 1);
        assert_eq!(p["persisted_tokens"], 16);
    }

    #[test]
    fn ram_peek_matches_inserted_prefix_without_mutation() {
        // The probe path uses peek_* only — verify a peek neither takes
        // refs nor changes what a later peek sees.
        let tree = spark_runtime::radix_tree::RadixTree::new();
        let tokens: Vec<u32> = (0..48).collect();
        let c = chain(&tokens);
        tree.insert(&c.view(&tokens, 48), &[10, 20, 30], &[], 0);

        let probe: Vec<u32> = (0..64).collect();
        let pc = chain(&probe);
        let v = pc.view(&probe, 64);
        assert_eq!(tree.peek_matched_tokens(&v), 48);
        assert_eq!(tree.peek_matched_tokens(&v), 48); // idempotent
    }
}

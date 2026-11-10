// SPDX-License-Identifier: AGPL-3.0-only

//! Per-request compaction state used by the chat pipeline: the message
//! chain of the client's history, the entry that applies to it (CMP-2)
//! and the creation of a new entry on overflow (CMP-5/7/10, CTX-3).

use std::sync::Arc;

use spark_runtime::block_keys::BlockKey;

use super::entry::{CompactionEntry, FORMAT_VERSION, now_secs};
use super::{CompactionConfig, ContextPolicy, chain, effective, head, plan, summarize};
use crate::AppState;
use crate::main_modules::app_state::ModelRuntime;
use crate::openai::{ChatCompletionRequest, IncomingMessage};

/// CTX-4 response headers.
pub const COMPACTED_HEADER: &str = "x-atlas-compacted";
pub const COMPACTION_ID_HEADER: &str = "x-atlas-compaction-id";

/// Rough prompt cost of one image when sizes are estimated before the
/// vision preprocessor ran; the final fit check uses the real prompt.
const IMAGE_TOKEN_ESTIMATE: usize = 1024;

/// What the response reports (CTX-4) and telemetry counts.
#[derive(Clone, Debug)]
pub struct CompactionInfo {
    pub id: BlockKey,
    pub replaced_messages: usize,
    pub replaced_tokens: usize,
    pub policy: ContextPolicy,
    /// Created by this request (synchronous compaction) rather than
    /// recognised from an earlier one.
    pub fresh: bool,
}

/// Why a new entry could not be created.
#[derive(Debug)]
pub enum CompactError {
    /// CTX-6: head + summary + newest message do not fit the window.
    NothingToFold {
        head_tokens: usize,
        newest_tokens: usize,
        summary_tokens: usize,
    },
    /// The summariser failed (scheduler or model error).
    Summary(anyhow::Error),
}

pub struct RequestCompaction {
    pub model: String,
    pub raw: Vec<IncomingMessage>,
    pub keys: Vec<BlockKey>,
    pub head_len: usize,
    pub applied: Option<(usize, Arc<CompactionEntry>)>,
    pub policy: ContextPolicy,
    fresh: bool,
}

impl RequestCompaction {
    /// Compute the chain over the client's messages and, unless the
    /// policy is `reject`, apply the latest matching entry to `req`.
    pub fn new(
        state: &AppState,
        rt: &ModelRuntime,
        req: &mut ChatCompletionRequest,
        policy: ContextPolicy,
    ) -> Self {
        let cfg = &state.compaction;
        let model = rt.model_name.clone();
        let keys = chain::message_chain(&model, &req.messages);
        let head_len = head::head_len(&req.messages, cfg.keep_first_user);
        let raw = req.messages.clone();
        let applied = if policy == ContextPolicy::Reject {
            None
        } else {
            state.compaction_index.find_latest(&keys, head_len)
        };
        if let Some((c, e)) = &applied {
            req.messages = effective::build_effective(&raw, head_len.min(*c), *c, &e.summary);
            state
                .compaction_index
                .stats
                .applied_requests
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            crate::metrics::COMPACTION_APPLIED_TOTAL.inc();
            tracing::info!(
                id = %e.key,
                replaced = e.replaced_messages,
                "compaction: entry applied to request"
            );
        }
        Self {
            model,
            raw,
            keys,
            head_len,
            applied,
            policy,
            fresh: false,
        }
    }

    pub fn info(&self) -> Option<CompactionInfo> {
        self.applied.as_ref().and_then(|(_, e)| {
            Some(CompactionInfo {
                id: e.key_bytes()?,
                replaced_messages: e.replaced_messages,
                replaced_tokens: e.replaced_tokens,
                policy: self.policy,
                fresh: self.fresh,
            })
        })
    }

    /// First raw index a new cut may land on.
    fn first(&self) -> usize {
        self.applied
            .as_ref()
            .map_or(self.head_len, |(c, _)| c + 1)
            .max(self.head_len)
    }

    /// Per-message prompt-token estimate of the raw history.
    pub fn sizes(&self, rt: &ModelRuntime) -> Vec<usize> {
        self.raw
            .iter()
            .map(|m| {
                let mut n = rt.tokenizer.encode(&m.content.text).map_or(0, |t| t.len());
                if let Some(tcs) = &m.tool_calls {
                    for tc in tcs {
                        n += rt
                            .tokenizer
                            .encode(&format!("{}{}", tc.function.name, tc.function.arguments))
                            .map_or(0, |t| t.len());
                    }
                }
                n + m.content.images.len() * IMAGE_TOKEN_ESTIMATE + 4
            })
            .collect()
    }

    fn summary_tokens(&self, rt: &ModelRuntime) -> usize {
        self.applied.as_ref().map_or(0, |(_, e)| {
            rt.tokenizer.encode(&e.summary).map_or(0, |t| t.len())
        })
    }

    /// CTX-6 figures: head, newest message and current summary sizes.
    pub fn ctx6(&self, rt: &ModelRuntime) -> CompactError {
        let sizes = self.sizes(rt);
        CompactError::NothingToFold {
            head_tokens: sizes[..self.head_len.min(sizes.len())].iter().sum(),
            newest_tokens: sizes.last().copied().unwrap_or(0),
            summary_tokens: self.summary_tokens(rt),
        }
    }

    /// Create a new entry (synchronous compaction, CMP-10) and return
    /// the effective messages. `max_tokens` is the request's output ask.
    pub async fn compact(
        &mut self,
        state: &AppState,
        rt: &ModelRuntime,
        max_tokens: usize,
    ) -> Result<Vec<IncomingMessage>, CompactError> {
        let cfg: &CompactionConfig = &state.compaction;
        let window = state.max_seq_len();
        let sizes = self.sizes(rt);
        let roles: Vec<&str> = self.raw.iter().map(|m| m.role.as_str()).collect();
        let first = self.first();
        let head_tokens: usize = sizes[..self.head_len.min(sizes.len())]
            .iter()
            .sum::<usize>()
            + self.summary_tokens(rt);
        let Some(cut) = plan::choose_cut(
            &roles,
            &sizes,
            first,
            cfg.tail_target(window),
            cfg.tail_budget(window, max_tokens, head_tokens),
        ) else {
            return Err(self.ctx6(rt));
        };
        let previous = self
            .applied
            .as_ref()
            .map(|(_, e)| e.summary.as_str())
            .filter(|s| !s.trim().is_empty());
        let fold = &self.raw[first..=cut.cut];
        let t0 = std::time::Instant::now();
        let summary = match self.policy {
            ContextPolicy::Truncate => String::new(),
            _ => summarize::summarize(state, rt, cfg, previous, fold, &sizes[first..=cut.cut])
                .await
                .map_err(CompactError::Summary)?,
        };
        let secs = t0.elapsed().as_secs_f64();
        let replaced_tokens =
            cut.replaced_tokens + self.applied.as_ref().map_or(0, |(_, e)| e.replaced_tokens);
        let entry = Arc::new(CompactionEntry {
            version: FORMAT_VERSION,
            key: self.keys[cut.cut].to_hex(),
            model: self.model.clone(),
            policy: self.policy.as_str().to_string(),
            head_len: self.head_len,
            cut: cut.cut,
            replaced_messages: cut.cut + 1 - self.head_len.min(cut.cut + 1),
            replaced_tokens,
            summary: summary.clone(),
            pred: self.applied.as_ref().map(|(_, e)| e.key.clone()),
            created_at: now_secs(),
        });
        state.compaction_index.insert(Arc::clone(&entry));
        self.record(state, &entry, secs, cut.tail_tokens);
        self.applied = Some((cut.cut, entry));
        self.fresh = true;
        Ok(effective::build_effective(
            &self.raw,
            self.head_len,
            cut.cut,
            &summary,
        ))
    }

    fn record(&self, state: &AppState, e: &CompactionEntry, secs: f64, tail_tokens: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        let st = &state.compaction_index.stats;
        match self.policy {
            ContextPolicy::Truncate => st.created_truncate.fetch_add(1, Relaxed),
            _ => st.created_summarize.fetch_add(1, Relaxed),
        };
        st.replaced_tokens
            .fetch_add(e.replaced_tokens as u64, Relaxed);
        st.summary_millis.fetch_add((secs * 1000.0) as u64, Relaxed);
        crate::metrics::COMPACTIONS_TOTAL
            .with_label_values(&[self.policy.as_str(), "sync"])
            .inc();
        crate::metrics::COMPACTION_SECONDS.observe(secs);
        crate::metrics::COMPACTION_REPLACED_TOKENS_TOTAL.inc_by(e.replaced_tokens as u64);
        tracing::info!(
            id = %e.key,
            policy = self.policy.as_str(),
            replaced_messages = e.replaced_messages,
            replaced_tokens = e.replaced_tokens,
            tail_tokens,
            summary_chars = e.summary.len(),
            secs = format_args!("{secs:.2}"),
            "compaction: new entry"
        );
        crate::observe::emit(
            crate::observe::Event::new(
                1,
                "compaction",
                format!(
                    "compacted {} messages ({} tokens) in {secs:.1}s",
                    e.replaced_messages, e.replaced_tokens
                ),
            )
            .model(&self.model)
            .detail(serde_json::json!({
                "id": e.key, "policy": e.policy, "replaced_messages": e.replaced_messages,
                "replaced_tokens": e.replaced_tokens, "tail_tokens": tail_tokens, "secs": secs,
            })),
        );
    }
}

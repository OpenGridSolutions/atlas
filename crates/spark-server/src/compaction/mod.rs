// SPDX-License-Identifier: AGPL-3.0-only

//! E7 context compaction (`docs/design/context-compaction.md`).
//!
//! When a conversation outgrows the context window the server replaces
//! the messages between the head (leading system/developer messages and
//! the first user turn, CMP-4) and the most recent turns (CMP-5) with a
//! summary written by the arena model (CMP-7). The cut is remembered as
//! a compaction entry keyed by the chained message key `m_c` of the last
//! replaced message (CMP-1, CMP-11); every follow-up request that
//! carries the same history is filtered automatically (CMP-2) — the
//! client keeps sending the full history, the model sees head + summary
//! + recent turns. Later compactions fold the previous summary and the
//! newly dropped messages into a new entry (CMP-3).
//!
//! Module map:
//! - `canonical` — canonical JSON of one message (CMP-1, decision 4)
//! - `chain`     — the BLAKE3-128 message chain m_1..m_n
//! - `entry`     — the `.cmp` record (CMP-11)
//! - `index`     — RAM index + NVMe persistence of entries
//! - `head`      — head detection (CMP-4)
//! - `plan`      — cut selection at message / tool-pair boundaries (CMP-5)
//! - `effective` — head + summary + tail assembly (CMP-2, CMP-6)
//! - `summarize` — summary generation on the arena model (CMP-7, CMP-3)
//! - `policy`    — `X-Atlas-Context-Policy` (CTX-1) + request middleware
//! - `request`   — per-request state: chain, applied entry, new entries

pub mod canonical;
pub mod chain;
pub mod effective;
pub mod entry;
pub mod head;
pub mod index;
pub mod plan;
pub mod policy;
pub mod request;
pub mod summarize;

pub use index::CompactionIndex;
pub use policy::ContextPolicy;
pub use request::RequestCompaction;

/// Operator configuration (requirements §14 with decision 3 applied:
/// trigger 70 %, summary 768 tokens).
#[derive(Clone, Debug)]
pub struct CompactionConfig {
    /// Behaviour without `X-Atlas-Context-Policy` (CTX-1).
    pub default_policy: ContextPolicy,
    /// Fill level (percent of the window) at which a background
    /// compaction is prepared (CMP-8).
    pub trigger_pct: u8,
    /// Minimum share of the window kept as recent turns (CMP-5).
    pub tail_pct: u8,
    /// Output budget of the summary (CMP-7).
    pub summary_max_tokens: usize,
    /// Output reserve of the fit rule, the block-based replacement of
    /// the removed `slot_buffer_tokens`: a request fits when
    /// `prompt + min(max_tokens, output_reserve) <= window`.
    pub output_reserve: usize,
    /// The first user message belongs to the head (CMP-4).
    pub keep_first_user: bool,
    /// Summarisation prompt (CMP-7): built-in or `--compaction-prompt-file`.
    pub prompt: std::sync::Arc<str>,
}

impl CompactionConfig {
    pub fn from_args(args: &crate::cli::ServeArgs) -> anyhow::Result<Self> {
        for (name, v) in [
            ("--compaction-trigger-pct", args.compaction_trigger_pct),
            ("--compaction-tail-pct", args.compaction_tail_pct),
        ] {
            anyhow::ensure!((1..=99).contains(&v), "{name} must be in 1..=99, got {v}");
        }
        anyhow::ensure!(
            args.compaction_summary_max_tokens > 0,
            "--compaction-summary-max-tokens must be > 0"
        );
        let prompt: std::sync::Arc<str> = match &args.compaction_prompt_file {
            Some(path) => std::fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("read --compaction-prompt-file {path}: {e}"))?
                .into(),
            None => summarize::DEFAULT_PROMPT.into(),
        };
        Ok(Self {
            default_policy: args.default_context_policy,
            trigger_pct: args.compaction_trigger_pct,
            tail_pct: args.compaction_tail_pct,
            summary_max_tokens: args.compaction_summary_max_tokens,
            output_reserve: args.compaction_output_reserve,
            keep_first_user: !args.compaction_no_keep_first_user,
            prompt,
        })
    }

    /// Output tokens the fit rule reserves for a request asking for
    /// `max_tokens` (bounded by `output_reserve`).
    pub fn reserve_for(&self, max_tokens: usize) -> usize {
        max_tokens.min(self.output_reserve).max(1)
    }

    /// The fit rule (requirements SLOT-4 without slots): the prompt plus
    /// the reserved output must fit the window.
    pub fn fits(&self, prompt_len: usize, max_tokens: usize, window: usize) -> bool {
        prompt_len.saturating_add(self.reserve_for(max_tokens)) <= window
    }

    /// Token budget available for the recent turns when a new cut is
    /// chosen: window − reserve − head − summary budget.
    pub fn tail_budget(&self, window: usize, max_tokens: usize, head_tokens: usize) -> usize {
        window
            .saturating_sub(self.reserve_for(max_tokens))
            .saturating_sub(head_tokens)
            .saturating_sub(self.summary_max_tokens)
    }

    /// CMP-5: minimum tail size in tokens.
    pub fn tail_target(&self, window: usize) -> usize {
        window * self.tail_pct as usize / 100
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

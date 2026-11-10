// SPDX-License-Identifier: AGPL-3.0-only

//! Context window of a model build. Either pinned by the operator
//! (`--max-seq-len`) or derived from MODEL.toml `[serving].sessions`: the
//! KV pool is split into that many equal windows, so every warm session
//! can fill its window without evicting another one.
//!
//! The session count follows one rule (documented next to each
//! `[serving]` section): take the model's native window, count how often
//! it fits into the KV pool, round up. Rounding up shrinks the window
//! below the native one so the pool divides evenly.

use anyhow::{Result, bail};

/// Blocks `TransformerModel::new` takes from the pool for good (the
/// zeroed dummy block behind out-of-bounds-safe paged reads). They hold
/// no session tokens.
const RESERVED_KV_BLOCKS: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextWindow {
    /// Window in tokens, pinned by the operator.
    Fixed(usize),
    /// `sessions` equal windows out of the KV pool, each capped at the
    /// model's native window `native_max` (tokens).
    Sessions { sessions: usize, native_max: usize },
}

impl ContextWindow {
    /// Largest window this build can end up with. Known before the KV
    /// pool is sized — for buffers allocated ahead of the pool.
    pub fn upper_bound(self) -> usize {
        match self {
            Self::Fixed(tokens) => tokens,
            Self::Sessions { native_max, .. } => native_max,
        }
    }

    /// The window in tokens for a pool of `num_kv_blocks` blocks.
    pub fn resolve(self, num_kv_blocks: usize, kv_block_size: usize) -> Result<usize> {
        let (sessions, native_max) = match self {
            Self::Fixed(tokens) => return Ok(tokens),
            Self::Sessions {
                sessions,
                native_max,
            } => (sessions, native_max),
        };
        if sessions == 0 || native_max == 0 {
            bail!(
                "context window needs at least 1 session and a native window \
                 (got sessions={sessions}, native window={native_max} tokens)"
            );
        }
        let blocks_per_session = usable_blocks(num_kv_blocks) / sessions;
        if blocks_per_session == 0 {
            bail!(
                "KV pool of {num_kv_blocks} block(s) is too small for {sessions} session(s): \
                 lower [serving].sessions in MODEL.toml or free memory for the KV pool"
            );
        }
        let window = (blocks_per_session * kv_block_size).min(native_max);
        tracing::info!(
            "Context window: the native window ({native_max} tokens) fits {:.2}x into the KV \
             pool → rule: {} session(s); MODEL.toml [serving].sessions = {sessions} → \
             {window} tokens per session",
            (usable_blocks(num_kv_blocks) * kv_block_size) as f64 / native_max as f64,
            rule_sessions(num_kv_blocks, kv_block_size, native_max),
        );
        Ok(window)
    }
}

/// Session count the rule yields for this pool: how often the native
/// window fits, rounded up.
fn rule_sessions(num_kv_blocks: usize, kv_block_size: usize, native_max: usize) -> usize {
    (usable_blocks(num_kv_blocks) * kv_block_size).div_ceil(native_max.max(1))
}

fn usable_blocks(num_kv_blocks: usize) -> usize {
    num_kv_blocks.saturating_sub(RESERVED_KV_BLOCKS)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: usize = 16;

    #[test]
    fn fixed_window_ignores_the_pool() {
        let window = ContextWindow::Fixed(8192);
        assert_eq!(window.upper_bound(), 8192);
        assert_eq!(window.resolve(10, BLOCK).unwrap(), 8192);
    }

    #[test]
    fn sessions_split_the_pool_into_equal_block_aligned_windows() {
        // 100 000 usable blocks / 7 sessions = 14 285 blocks each.
        let window = ContextWindow::Sessions {
            sessions: 7,
            native_max: 262_144,
        };
        assert_eq!(window.upper_bound(), 262_144);
        assert_eq!(window.resolve(100_001, BLOCK).unwrap(), 14_285 * BLOCK);
    }

    #[test]
    fn the_dummy_block_holds_no_session_tokens() {
        let window = ContextWindow::Sessions {
            sessions: 2,
            native_max: 262_144,
        };
        assert_eq!(window.resolve(21, BLOCK).unwrap(), 10 * BLOCK);
        assert_eq!(window.resolve(20, BLOCK).unwrap(), 9 * BLOCK);
    }

    #[test]
    fn a_window_never_exceeds_the_native_one() {
        let window = ContextWindow::Sessions {
            sessions: 2,
            native_max: 40_960,
        };
        assert_eq!(window.resolve(100_001, BLOCK).unwrap(), 40_960);
    }

    #[test]
    fn a_pool_smaller_than_the_session_count_is_an_error() {
        let window = ContextWindow::Sessions {
            sessions: 10,
            native_max: 262_144,
        };
        let err = window.resolve(6, BLOCK).unwrap_err().to_string();
        assert!(err.contains("too small for 10 session(s)"), "{err}");
    }

    #[test]
    fn a_missing_native_window_or_session_count_is_an_error() {
        let no_native = ContextWindow::Sessions {
            sessions: 4,
            native_max: 0,
        };
        assert!(no_native.resolve(100_001, BLOCK).is_err());
        let no_sessions = ContextWindow::Sessions {
            sessions: 0,
            native_max: 262_144,
        };
        assert!(no_sessions.resolve(100_001, BLOCK).is_err());
    }

    #[test]
    fn the_rule_rounds_the_fit_count_up() {
        // 1 600 000 pool tokens / 262 144 = 6.1 → 7 sessions.
        assert_eq!(rule_sessions(100_001, BLOCK, 262_144), 7);
        // Exact fit: 30 blocks × 16 = 480 tokens = 3 × 160.
        assert_eq!(rule_sessions(31, BLOCK, 160), 3);
    }
}

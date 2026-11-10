// SPDX-License-Identifier: AGPL-3.0-only

//! Where a model load gets its context window from. Precedence: an
//! explicit `--max-seq-len` wins, else MODEL.toml `[serving].sessions`
//! (the KV pool split into that many equal windows), else the legacy
//! window for kernel targets without a `[serving]` section.

use anyhow::{Result, bail};
use spark_model::factory::ContextWindow;

/// Window for kernel targets without `[serving].sessions` — the value
/// `--max-seq-len` defaulted to before the window became per-model.
const LEGACY_MAX_SEQ_LEN: usize = 32768;

/// `sessions` is the kernel target's `[serving].sessions`, `native_max`
/// the checkpoint's `max_position_embeddings`.
pub(crate) fn resolve_context_window(
    max_seq_len_flag: Option<usize>,
    sessions: Option<u32>,
    native_max: usize,
    max_batch_size: usize,
    high_speed_swap: bool,
) -> Result<ContextWindow> {
    if let Some(tokens) = max_seq_len_flag {
        if let Some(sessions) = sessions {
            tracing::warn!(
                "Context window: {tokens} tokens (--max-seq-len) — MODEL.toml \
                 [serving].sessions = {sessions} is ignored; drop --max-seq-len to split \
                 the KV pool into {sessions} equal windows",
            );
        }
        return Ok(ContextWindow::Fixed(tokens));
    }
    let Some(sessions) = sessions.map(|s| s as usize) else {
        tracing::info!(
            "Context window: {LEGACY_MAX_SEQ_LEN} tokens (MODEL.toml has no \
             [serving].sessions; override with --max-seq-len)",
        );
        return Ok(ContextWindow::Fixed(LEGACY_MAX_SEQ_LEN));
    };
    if high_speed_swap {
        // The HBM-shrunk pool is sized FROM the window, so there is no
        // pool to split.
        tracing::info!(
            "Context window: {LEGACY_MAX_SEQ_LEN} tokens (--high-speed-swap ignores \
             MODEL.toml [serving].sessions; override with --max-seq-len)",
        );
        return Ok(ContextWindow::Fixed(LEGACY_MAX_SEQ_LEN));
    }
    if native_max == 0 {
        bail!(
            "MODEL.toml [serving].sessions = {sessions}, but the checkpoint's config.json \
             declares no max_position_embeddings to cap the window at — pass --max-seq-len"
        );
    }
    if sessions < max_batch_size {
        bail!(
            "MODEL.toml [serving].sessions = {sessions} is below --max-batch-size \
             {max_batch_size}: every request running in parallel needs its own window. \
             Raise [serving].sessions, lower --max-batch-size, or pass --max-seq-len"
        );
    }
    Ok(ContextWindow::Sessions {
        sessions,
        native_max,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_flag_wins_over_model_toml() {
        let window = resolve_context_window(Some(8192), Some(11), 262_144, 4, false).unwrap();
        assert_eq!(window, ContextWindow::Fixed(8192));
    }

    #[test]
    fn model_toml_sessions_split_the_pool() {
        let window = resolve_context_window(None, Some(11), 262_144, 4, false).unwrap();
        assert_eq!(
            window,
            ContextWindow::Sessions {
                sessions: 11,
                native_max: 262_144
            }
        );
    }

    #[test]
    fn target_without_sessions_keeps_the_legacy_window() {
        let window = resolve_context_window(None, None, 262_144, 4, false).unwrap();
        assert_eq!(window, ContextWindow::Fixed(32768));
    }

    #[test]
    fn high_speed_swap_keeps_the_legacy_window() {
        let window = resolve_context_window(None, Some(11), 262_144, 4, true).unwrap();
        assert_eq!(window, ContextWindow::Fixed(32768));
    }

    #[test]
    fn sessions_without_a_native_window_fail_fast() {
        let err = resolve_context_window(None, Some(11), 0, 4, false).unwrap_err();
        assert!(err.to_string().contains("max_position_embeddings"), "{err}");
    }

    #[test]
    fn fewer_sessions_than_parallel_requests_fail_fast() {
        let err = resolve_context_window(None, Some(3), 131_072, 4, false).unwrap_err();
        assert!(
            err.to_string().contains("below --max-batch-size 4"),
            "{err}"
        );
    }
}

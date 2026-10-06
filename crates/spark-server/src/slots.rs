// SPDX-License-Identifier: AGPL-3.0-only

//! Per-model session slots (requirements E6, SLOT-1..6, MOD-12, BAT-8).
//!
//! A slot is the PLANNING unit of the arena: admission takes at most
//! `slots_total` sequences, and each admitted sequence may grow to
//! `slot_tokens` without displacing another one — preemption becomes a
//! backstop instead of a steady-state mechanism (BAT-5). Slots are
//! logical: blocks stay demand-paged, and blocks no active sequence
//! uses keep serving the prefix cache (SLOT-3).
//!
//! Resolution order for `slot_tokens`: `--slot-tokens` CLI override
//! (`model=N` entries win over a bare global `N`) → MODEL.toml
//! `[behavior].slot_tokens` → `--max-seq-len`. `slot_buffer_tokens`:
//! MODEL.toml → 25 % of the effective slot. MOD-12 validates the result
//! against `--max-seq-len` and the model's position limit at load time
//! and refuses the load on violation.
//!
//! The active plan is published at boot and on every hot-swap load
//! (after the KV pool exists — `slots_total` derives from the real pool
//! geometry), read by the scheduler admission, the request-fit check in
//! the handlers (SLOT-4), the `X-Atlas-Context-Remaining` header
//! (SLOT-6), `/internal/state` and the session-state probe.

use std::sync::Mutex;

use anyhow::{Result, bail};

/// Resolved slot configuration of one model (pre-pool, MOD-12 checked).
#[derive(Clone, Copy, Debug)]
pub struct SlotConfig {
    pub slot_tokens: usize,
    pub slot_buffer_tokens: usize,
}

/// Published plan of the ACTIVE model (post-pool).
#[derive(Clone, Debug)]
pub struct SlotPlan {
    pub model_id: String,
    pub slot_tokens: usize,
    pub slot_buffer_tokens: usize,
    /// ⌊pool ÷ blocks-per-slot⌋ (SLOT-2) — capacity, not reservation.
    pub slots_total: usize,
    /// PRI-7: slots normal requests may not occupy.
    pub priority_reserved: usize,
    /// True when `slot_tokens` came from MODEL.toml/CLI rather than the
    /// `--max-seq-len` fallback (feeds `slot.source` in the API).
    pub explicit: bool,
}

static ACTIVE: Mutex<Option<SlotPlan>> = Mutex::new(None);

pub fn set_active(plan: SlotPlan) {
    tracing::info!(
        model = %plan.model_id,
        slot_tokens = plan.slot_tokens,
        slot_buffer_tokens = plan.slot_buffer_tokens,
        slots_total = plan.slots_total,
        priority_reserved = plan.priority_reserved,
        "slot plan published (E6)"
    );
    *ACTIVE.lock().expect("slot plan lock") = Some(plan);
}

pub fn active() -> Option<SlotPlan> {
    ACTIVE.lock().expect("slot plan lock").clone()
}

/// Parse the repeatable `--slot-tokens` entries: `model=N` wins for its
/// model, a bare `N` is the global override.
fn cli_override(entries: &[String], model_id: &str) -> Result<Option<usize>> {
    let mut global: Option<usize> = None;
    let mut per_model: Option<usize> = None;
    for e in entries {
        match e.split_once('=') {
            Some((m, n)) => {
                let n: usize = n
                    .trim()
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--slot-tokens '{e}': '{n}' is not a number"))?;
                if m.trim() == model_id {
                    per_model = Some(n);
                }
            }
            None => {
                global = Some(e.trim().parse().map_err(|_| {
                    anyhow::anyhow!("--slot-tokens '{e}' is neither N nor model=N")
                })?);
            }
        }
    }
    Ok(per_model.or(global))
}

/// SLOT-1 resolution + MOD-12 validation. `behavior_slot`/`behavior_buffer`
/// are the MODEL.toml values (0 = unset); `position_limit` is the model's
/// `max_position_embeddings` (0 = unknown → not enforced).
pub fn resolve(
    cli_entries: &[String],
    model_id: &str,
    behavior_slot: usize,
    behavior_buffer: usize,
    max_seq_len: usize,
    position_limit: usize,
) -> Result<(SlotConfig, bool)> {
    let cli = cli_override(cli_entries, model_id)?;
    let explicit = cli.is_some() || behavior_slot > 0;
    let slot_tokens = cli.unwrap_or(if behavior_slot > 0 {
        behavior_slot
    } else {
        max_seq_len
    });
    if slot_tokens == 0 {
        bail!("slot_tokens resolved to 0 for '{model_id}'");
    }
    // MOD-12: the slot must fit the serving window and the model.
    if slot_tokens > max_seq_len {
        bail!(
            "slot_tokens {slot_tokens} for '{model_id}' exceeds --max-seq-len {max_seq_len} \
             (MOD-12)"
        );
    }
    if position_limit > 0 && slot_tokens > position_limit {
        bail!(
            "slot_tokens {slot_tokens} for '{model_id}' exceeds the model's position limit \
             {position_limit} (MOD-12)"
        );
    }
    let slot_buffer_tokens = if behavior_buffer > 0 {
        behavior_buffer
    } else {
        (slot_tokens / 4).max(1)
    };
    if slot_buffer_tokens > slot_tokens {
        bail!(
            "slot_buffer_tokens {slot_buffer_tokens} exceeds slot_tokens {slot_tokens} for \
             '{model_id}'"
        );
    }
    Ok((
        SlotConfig {
            slot_tokens,
            slot_buffer_tokens,
        },
        explicit,
    ))
}

/// SLOT-2: slots the live pool can plan with. One slot needs the blocks
/// of a full `slot_tokens` sequence (ceil), each admitted sequence may
/// use them all without displacing another (SLOT-3).
pub fn slots_total(total_blocks: usize, block_size: usize, slot_tokens: usize) -> usize {
    if block_size == 0 || slot_tokens == 0 {
        return 0;
    }
    let blocks_per_slot = slot_tokens.div_ceil(block_size);
    total_blocks / blocks_per_slot.max(1)
}

/// SLOT-2/MOD-12: derive `slots_total` from the freshly built pool and
/// publish the plan. Refuses the load when not even one slot fits the
/// pool — the model would stall every request at admission.
pub fn publish_from_pool(
    model_id: &str,
    cfg: SlotConfig,
    explicit: bool,
    priority_reserved: usize,
    total_blocks: usize,
    block_size: usize,
) -> Result<()> {
    let total = slots_total(total_blocks, block_size, cfg.slot_tokens);
    if total == 0 {
        bail!(
            "slot_tokens {} for '{model_id}' does not fit the KV pool ({total_blocks} blocks × \
             {block_size} tokens) even once (MOD-12)",
            cfg.slot_tokens
        );
    }
    set_active(SlotPlan {
        model_id: model_id.to_string(),
        slot_tokens: cfg.slot_tokens,
        slot_buffer_tokens: cfg.slot_buffer_tokens,
        slots_total: total,
        // PRI-7: at least one slot stays reachable for normal traffic.
        priority_reserved: priority_reserved.min(total.saturating_sub(1)),
        explicit,
    });
    Ok(())
}

/// SLOT-4 outcome.
pub enum Fit {
    /// Granted output budget (requested, clamped to the slot buffer and
    /// the remaining slot room).
    Ok { granted_max_tokens: usize },
    /// Prompt alone (plus one output token) exceeds the slot — HTTP 400
    /// `context_length_exceeded` until the compaction stage shrinks it.
    PromptTooLarge { slot_tokens: usize },
}

/// SLOT-4: clamp `max_tokens` to `slot_buffer_tokens`, then to the room
/// the slot has left. Falls back to (`max_seq_len`, no buffer cap) when
/// no plan is active — byte-identical to the previous clamp.
pub fn fit_request(prompt_tokens: usize, requested_max_tokens: usize, max_seq_len: usize) -> Fit {
    let (slot_tokens, buffer_cap) = match active() {
        Some(p) => (p.slot_tokens, p.slot_buffer_tokens),
        None => (max_seq_len, usize::MAX),
    };
    if prompt_tokens + 1 > slot_tokens {
        return Fit::PromptTooLarge { slot_tokens };
    }
    let granted = requested_max_tokens
        .min(buffer_cap)
        .min(slot_tokens - prompt_tokens)
        .max(1);
    Fit::Ok {
        granted_max_tokens: granted,
    }
}

/// The hard per-sequence bound every handler must enforce: the active
/// slot size, or `max_seq_len` without a plan. The attention block
/// table (`max_blocks_per_seq`, MOD-12) is dimensioned on this — a
/// longer sequence would overflow it.
pub fn prompt_limit(max_seq_len: usize) -> usize {
    active().map(|p| p.slot_tokens).unwrap_or(max_seq_len)
}

/// SLOT-6: free tokens in the slot after `used_tokens` (prompt + output).
pub fn remaining_after(used_tokens: usize, max_seq_len: usize) -> usize {
    let slot = active().map(|p| p.slot_tokens).unwrap_or(max_seq_len);
    slot.saturating_sub(used_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_override_prefers_per_model_over_global() {
        let entries = vec!["8192".to_string(), "m/a=16384".to_string()];
        assert_eq!(cli_override(&entries, "m/a").unwrap(), Some(16384));
        assert_eq!(cli_override(&entries, "m/b").unwrap(), Some(8192));
        assert_eq!(cli_override(&[], "m/a").unwrap(), None);
        assert!(cli_override(&["m/a=abc".to_string()], "m/a").is_err());
    }

    #[test]
    fn resolve_order_cli_toml_maxseq() {
        // CLI wins.
        let (c, explicit) = resolve(&["m=4096".to_string()], "m", 8192, 0, 32768, 131072).unwrap();
        assert_eq!((c.slot_tokens, explicit), (4096, true));
        // MODEL.toml next.
        let (c, explicit) = resolve(&[], "m", 8192, 2048, 32768, 131072).unwrap();
        assert_eq!(
            (c.slot_tokens, c.slot_buffer_tokens, explicit),
            (8192, 2048, true)
        );
        // Fallback: max-seq-len, buffer 25 %.
        let (c, explicit) = resolve(&[], "m", 0, 0, 32768, 131072).unwrap();
        assert_eq!(
            (c.slot_tokens, c.slot_buffer_tokens, explicit),
            (32768, 8192, false)
        );
    }

    #[test]
    fn mod12_rejects_oversized_slots() {
        // Beyond --max-seq-len.
        assert!(resolve(&[], "m", 65536, 0, 32768, 131072).is_err());
        // Beyond the model's position limit.
        assert!(resolve(&[], "m", 16384, 0, 32768, 8192).is_err());
        // Buffer larger than the slot.
        assert!(resolve(&[], "m", 8192, 9000, 32768, 131072).is_err());
    }

    #[test]
    fn slots_total_floors_by_blocks_per_slot() {
        // 1000 blocks × 16 tokens; 32k slot → 2048 blocks/slot → 0 slots.
        assert_eq!(slots_total(1000, 16, 32768), 0);
        // 8k slot → 512 blocks/slot → 1 slot; 4k → 3 slots.
        assert_eq!(slots_total(1000, 16, 8192), 1);
        assert_eq!(slots_total(1000, 16, 4096), 3);
    }

    #[test]
    fn fit_clamps_to_buffer_and_remaining_room() {
        set_active(SlotPlan {
            model_id: "m".into(),
            slot_tokens: 1000,
            slot_buffer_tokens: 200,
            slots_total: 4,
            priority_reserved: 0,
            explicit: true,
        });
        // Buffer caps the request (SLOT-4).
        match fit_request(100, 500, 32768) {
            Fit::Ok { granted_max_tokens } => assert_eq!(granted_max_tokens, 200),
            _ => panic!("expected Ok"),
        }
        // Remaining room caps below the buffer near the slot end.
        match fit_request(950, 500, 32768) {
            Fit::Ok { granted_max_tokens } => assert_eq!(granted_max_tokens, 50),
            _ => panic!("expected Ok"),
        }
        // Prompt alone too large → 400 (CTX-6 class).
        assert!(matches!(
            fit_request(1000, 10, 32768),
            Fit::PromptTooLarge { .. }
        ));
        assert_eq!(remaining_after(400, 32768), 600);
        *ACTIVE.lock().unwrap() = None;
    }
}

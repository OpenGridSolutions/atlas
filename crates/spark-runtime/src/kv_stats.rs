// SPDX-License-Identifier: AGPL-3.0-only

//! Process-global KV-pool residency counters for /v1/stats (Plan.md
//! Phase S).
//!
//! Exactly one `PagedKvCache` exists at a time (single model in the
//! weights arena; the whisper sidecar has no paged pool), so a global
//! is the honest representation — the same pattern as
//! `gpu::baseline_free_bytes`. The pool publishes its totals at
//! construction, the free-block count on every free-list mutation
//! (single mutator thread — the scheduler — so Relaxed stores suffice
//! and the value is always `free_blocks.len()`, never an incremental
//! delta that could drift), and clears on drop. The /v1/stats handler
//! reads a snapshot from any thread.

use std::sync::atomic::{AtomicUsize, Ordering};

static TOTAL_BLOCKS: AtomicUsize = AtomicUsize::new(0);
static FREE_BLOCKS: AtomicUsize = AtomicUsize::new(0);
static POOL_BYTES: AtomicUsize = AtomicUsize::new(0);
// What the LAST KV-budget computation saw (factory::build): the free-now
// input and the resulting budget. THIS pair is what the follow-up
// /proc/meminfo budget fix moves — /v1/stats exposes it so the fix has a
// before/after metric at the exact decision point. Persists across
// unload (deliberately: it documents the last decision, not residency).
static BUDGET_FREE_NOW: AtomicUsize = AtomicUsize::new(0);
static BUDGET_BYTES: AtomicUsize = AtomicUsize::new(0);

/// One live reading of the resident pool. `used_blocks` moves with
/// active sequences.
#[derive(Debug, Clone, Copy)]
pub struct KvPoolSnapshot {
    pub total_blocks: usize,
    pub used_blocks: usize,
    pub reserved_bytes: usize,
}

/// Called by `PagedKvCache::new` with the block count and the exact
/// device bytes it allocated across all layer pools.
pub fn publish_pool(total_blocks: usize, pool_bytes: usize) {
    TOTAL_BLOCKS.store(total_blocks, Ordering::Relaxed);
    FREE_BLOCKS.store(total_blocks, Ordering::Relaxed);
    POOL_BYTES.store(pool_bytes, Ordering::Relaxed);
}

/// Called after every `free_blocks` mutation with the current length.
pub fn set_free_blocks(n: usize) {
    FREE_BLOCKS.store(n, Ordering::Relaxed);
}

/// Called by `PagedKvCache::drop` — no pool, no numbers.
pub fn clear() {
    TOTAL_BLOCKS.store(0, Ordering::Relaxed);
    FREE_BLOCKS.store(0, Ordering::Relaxed);
    POOL_BYTES.store(0, Ordering::Relaxed);
}

// Todo.md 0.5: the FULL gate decision of the last budget computation, so
// the test bench can assert 0.1–0.4 from outside via /v1/stats.
static BUDGET_USED_SO_FAR: AtomicUsize = AtomicUsize::new(0);
static BUDGET_TOTAL: AtomicUsize = AtomicUsize::new(0);
static BUDGET_BASELINE: AtomicUsize = AtomicUsize::new(0);
static BUDGET_COTENANTS: AtomicUsize = AtomicUsize::new(0);
// 1-based so 0 = "no decision published yet" (matches BUDGET_FREE_NOW).
static BUDGET_SOURCE: AtomicUsize = AtomicUsize::new(0);

/// How the last budget's `used_so_far` was derived (mirror of
/// `spark_model::factory::kv_budget::UsedSource`, passed as a code so the
/// crates stay decoupled).
#[derive(Debug, Clone, Copy)]
pub struct BudgetDecision {
    pub used_so_far: usize,
    pub total_budget: usize,
    /// 0 = no baseline was available.
    pub baseline_free: usize,
    pub cotenants: usize,
    /// "raw" | "manual" | "auto_baseline" | "auto_implausible".
    pub source: &'static str,
}

/// Called by `factory::build` each time the KV budget is computed.
pub fn publish_budget_inputs(free_now_bytes: usize, kv_budget_bytes: usize) {
    BUDGET_FREE_NOW.store(free_now_bytes, Ordering::Relaxed);
    BUDGET_BYTES.store(kv_budget_bytes, Ordering::Relaxed);
}

/// Companion to `publish_budget_inputs`: the gate decision itself.
/// `source_code`: 1=raw, 2=manual, 3=auto_baseline, 4=auto_implausible.
pub fn publish_budget_decision(
    used_so_far: usize,
    total_budget: usize,
    baseline_free: Option<usize>,
    cotenants: usize,
    source_code: u8,
) {
    BUDGET_USED_SO_FAR.store(used_so_far, Ordering::Relaxed);
    BUDGET_TOTAL.store(total_budget, Ordering::Relaxed);
    BUDGET_BASELINE.store(baseline_free.unwrap_or(0), Ordering::Relaxed);
    BUDGET_COTENANTS.store(cotenants, Ordering::Relaxed);
    BUDGET_SOURCE.store(source_code as usize, Ordering::Relaxed);
}

/// The last gate decision; `None` before the first budget computation.
pub fn budget_decision() -> Option<BudgetDecision> {
    let source = match BUDGET_SOURCE.load(Ordering::Relaxed) {
        1 => "raw",
        2 => "manual",
        3 => "auto_baseline",
        4 => "auto_implausible",
        _ => return None,
    };
    Some(BudgetDecision {
        used_so_far: BUDGET_USED_SO_FAR.load(Ordering::Relaxed),
        total_budget: BUDGET_TOTAL.load(Ordering::Relaxed),
        baseline_free: BUDGET_BASELINE.load(Ordering::Relaxed),
        cotenants: BUDGET_COTENANTS.load(Ordering::Relaxed),
        source,
    })
}

/// `(free_now_bytes, kv_budget_bytes)` of the last budget computation;
/// `None` before the first model load.
pub fn budget_inputs() -> Option<(usize, usize)> {
    match BUDGET_FREE_NOW.load(Ordering::Relaxed) {
        0 => None,
        f => Some((f, BUDGET_BYTES.load(Ordering::Relaxed))),
    }
}

/// `None` when no pool is resident.
pub fn snapshot() -> Option<KvPoolSnapshot> {
    let total = TOTAL_BLOCKS.load(Ordering::Relaxed);
    if total == 0 {
        return None;
    }
    let free = FREE_BLOCKS.load(Ordering::Relaxed).min(total);
    Some(KvPoolSnapshot {
        total_blocks: total,
        used_blocks: total - free,
        reserved_bytes: POOL_BYTES.load(Ordering::Relaxed),
    })
}

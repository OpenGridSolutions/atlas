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

/// Called by `factory::build` each time the KV budget is computed.
pub fn publish_budget_inputs(free_now_bytes: usize, kv_budget_bytes: usize) {
    BUDGET_FREE_NOW.store(free_now_bytes, Ordering::Relaxed);
    BUDGET_BYTES.store(kv_budget_bytes, Ordering::Relaxed);
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

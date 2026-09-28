// SPDX-License-Identifier: AGPL-3.0-only

//! Per-request, per-phase KV-cache observability (`ATLAS_MEM_DEBUG=1`).
//!
//! Emits structured tracing events on the dedicated target `memdebug` so
//! `RUST_LOG="off,memdebug=debug" ATLAS_MEM_DEBUG=1 ./serve.sh` shows ONLY
//! these lines. Observability only — never changes scheduling, caching or
//! eviction behavior. Zero cost when the env var is unset: every per-seq
//! hook takes an `Option<Box<SeqMemDebug>>` that is `None` when disabled.
//!
//! Event stream per request:
//!   admission        pool stats + GPU free/RSS (+ drift vs first request)
//!   prefill          prompt tokens, prefix-cache reuse, new blocks, held
//!   block_alloc      decode-time growth, attributed to phase
//!                    ("thinking"/"generation" via `inside_thinking`)
//!   decode_summary   every 128 output tokens
//!   phase_transition thinking → generation (on the flag flip itself)
//!   finish           release accounting + per-phase breakdown + mem deltas
//!   idle             leak invariant when the scheduler goes idle
//!
//! Accounting notes:
//!   * `unaccounted = total − free − active_used − cached_evictable` can go
//!     NEGATIVE: a block reused from the prefix cache is held by the active
//!     sequence AND ref'd by the cache, so it is counted in both terms.
//!   * The pool is never empty at idle: model build allocates permanent
//!     blocks (e.g. the CUDA-graph padding dummy block). The idle invariant
//!     therefore compares against a permanent-block baseline captured at
//!     the first admission per pool (re-captured after a hot-swap).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use spark_model::traits::{Model, SequenceState};

/// Periodic decode summary stride (output tokens).
const SUMMARY_STRIDE: usize = 128;
/// Sentinel for "not captured yet" in the atomics below.
const UNSET: usize = usize::MAX;

/// `ATLAS_MEM_DEBUG=1` gate, read once.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_MEM_DEBUG").as_deref() == Ok("1"))
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static BLOCK_SIZE: AtomicUsize = AtomicUsize::new(0);
// Permanent pool overhead (blocks that are used while zero sequences and an
// empty prefix cache exist), captured at the first admission per pool.
// `BASELINE_TOTAL` keys the capture: a hot-swap rebuilds the pool with a
// (usually) different block count, which re-arms the capture.
static BASELINE_USED: AtomicUsize = AtomicUsize::new(UNSET);
static BASELINE_TOTAL: AtomicUsize = AtomicUsize::new(UNSET);
// First-instrumented-request memory readings: cumulative drift anchor for
// CUDA-graph / allocator leak detection (a new graph is captured per
// request — monotonically sinking gpu_drift_mib means graphs or other GPU
// allocations leak).
static FIRST_GPU_FREE_MIB: AtomicUsize = AtomicUsize::new(UNSET);
static FIRST_RSS_MIB: AtomicUsize = AtomicUsize::new(UNSET);
// Busy→idle edge detector so the idle invariant logs once per transition.
static WAS_BUSY: AtomicBool = AtomicBool::new(false);

/// One-time init from `scheduler::run` (block size for reused-block math).
pub fn init(block_size: usize) {
    if enabled() {
        BLOCK_SIZE.store(block_size, Ordering::Relaxed);
        tracing::debug!(target: "memdebug", block_size, "memdebug enabled (ATLAS_MEM_DEBUG=1)");
    }
}

/// `(free_mib, total_mib)` from raw `cuMemGetInfo`; `(-1, -1)` when no CUDA
/// context is available (mock backend, early boot).
fn gpu_free_total_mib() -> (i64, i64) {
    match spark_runtime::cuda_backend::cuda_mem_get_info_raw() {
        Some((free, total)) => ((free >> 20) as i64, (total >> 20) as i64),
        None => (-1, -1),
    }
}

/// Process RSS in MiB from `/proc/self/status` (`VmRSS` is reported in kB,
/// so no page-size assumption); `-1` when unreadable.
fn rss_mib() -> i64 {
    let Ok(s) = std::fs::read_to_string("/proc/self/status") else {
        return -1;
    };
    s.lines()
        .find_map(|l| l.strip_prefix("VmRSS:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|kb| kb.parse::<i64>().ok())
        .map_or(-1, |kb| kb / 1024)
}

/// Drift vs the first instrumented request; captures the anchor on first use.
fn drift_vs_first(anchor: &AtomicUsize, now: i64) -> i64 {
    if now < 0 {
        return 0;
    }
    let _ = anchor.compare_exchange(UNSET, now as usize, Ordering::Relaxed, Ordering::Relaxed);
    let first = anchor.load(Ordering::Relaxed);
    if first == UNSET {
        0
    } else {
        now - first as i64
    }
}

struct PoolView {
    total: usize,
    free: usize,
    cached_evictable: usize,
}

fn pool_view(model: &dyn Model) -> PoolView {
    let (total, free) = match spark_runtime::kv_stats::snapshot() {
        Some(s) => (s.total_blocks, s.total_blocks - s.used_blocks),
        None => (0, 0),
    };
    let (_entries, cached_evictable) = model.prefix_cache_stats();
    PoolView {
        total,
        free,
        cached_evictable,
    }
}

/// Boot complete: preseed finished and the HTTP listener is bound. Mirrors
/// the info-level "Listening on …" readiness onto the memdebug target so a
/// `RUST_LOG=off,memdebug=debug` session shows when testing can start.
pub fn on_server_ready(addr: &str, registered_models: usize) {
    if !enabled() {
        return;
    }
    tracing::debug!(
        target: "memdebug",
        event = "server_ready",
        addr,
        registered_models,
        "server ready — preseed complete, listener bound (models lazy-load on first request)"
    );
}

/// Request admission: pool stats + GPU/RSS context. `active_used` is the sum
/// of block-table lengths over active + prefilling sequences (scheduler view).
pub fn on_request_admitted(model: &dyn Model, prompt_tokens: usize, active_used: usize) {
    if !enabled() {
        return;
    }
    WAS_BUSY.store(true, Ordering::Relaxed);
    let p = pool_view(model);
    let used = p.total.saturating_sub(p.free);
    if BASELINE_TOTAL.load(Ordering::Relaxed) != p.total {
        let permanent = used
            .saturating_sub(active_used)
            .saturating_sub(p.cached_evictable);
        BASELINE_USED.store(permanent, Ordering::Relaxed);
        BASELINE_TOTAL.store(p.total, Ordering::Relaxed);
        tracing::debug!(
            target: "memdebug",
            permanent_blocks = permanent,
            pool_total = p.total,
            "captured permanent-block baseline (model-build allocations, e.g. padding dummy block)"
        );
    }
    let (gpu_free, gpu_total) = gpu_free_total_mib();
    let rss = rss_mib();
    tracing::debug!(
        target: "memdebug",
        event = "admission",
        prompt_tokens,
        pool_total = p.total,
        pool_free = p.free,
        active_used,
        cached_evictable = p.cached_evictable,
        unaccounted =
            p.total as i64 - p.free as i64 - active_used as i64 - p.cached_evictable as i64,
        gpu_free_mib = gpu_free,
        gpu_total_mib = gpu_total,
        rss_mib = rss,
        gpu_drift_mib = drift_vs_first(&FIRST_GPU_FREE_MIB, gpu_free),
        rss_drift_mib = drift_vs_first(&FIRST_RSS_MIB, rss),
        "request admitted"
    );
}

/// Per-sequence debug state, carried on `PrefillInProgress` → `ActiveSeq`.
/// `None` when memdebug is off. The `Drop` backstop flags any sequence that
/// is dropped without going through a finish/abort/swap-out hook — every
/// legitimate exit path consumes the guard, so a Drop-time error means a
/// release path was genuinely skipped.
pub struct SeqMemDebug {
    id: u64,
    resumed: bool,
    start_gpu_free_mib: i64,
    start_rss_mib: i64,
    prompt_tokens: usize,
    prefix_hit_tokens: usize,
    reused_blocks: usize,
    prefill_blocks_new: usize,
    /// Block-table length last seen by `on_decode_tick` (growth detector).
    known_blocks: usize,
    thinking_blocks: usize,
    generation_blocks: usize,
    was_inside_thinking: Option<bool>,
    next_summary_at: usize,
    closed: bool,
}

impl Drop for SeqMemDebug {
    fn drop(&mut self) {
        if !self.closed {
            tracing::error!(
                target: "memdebug",
                seq = self.id,
                blocks_known = self.known_blocks,
                "sequence dropped without finish/abort accounting — a release path was skipped"
            );
        }
    }
}

fn new_seq_inner(resumed: bool) -> Option<Box<SeqMemDebug>> {
    if !enabled() {
        return None;
    }
    let (gpu_free, _) = gpu_free_total_mib();
    Some(Box::new(SeqMemDebug {
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        resumed,
        start_gpu_free_mib: gpu_free,
        start_rss_mib: rss_mib(),
        prompt_tokens: 0,
        prefix_hit_tokens: 0,
        reused_blocks: 0,
        prefill_blocks_new: 0,
        known_blocks: 0,
        thinking_blocks: 0,
        generation_blocks: 0,
        was_inside_thinking: None,
        next_summary_at: SUMMARY_STRIDE,
        closed: false,
    }))
}

/// Create the per-sequence guard at prefill start. `None` when disabled.
pub fn new_seq() -> Option<Box<SeqMemDebug>> {
    new_seq_inner(false)
}

/// Fresh guard for a sequence resumed from disk swap (fresh memdebug ID,
/// `resumed=true`; per-phase counters restart — the pre-swap portion was
/// already logged by `on_swap_out`).
pub fn new_seq_resumed() -> Option<Box<SeqMemDebug>> {
    new_seq_inner(true)
}

/// Prefill completed: prefix-cache reuse + net new blocks for the prompt.
pub fn on_prefill_complete(
    mem: &mut Option<Box<SeqMemDebug>>,
    seq: &SequenceState,
    prompt_tokens: usize,
) {
    let Some(m) = mem else { return };
    let bs = BLOCK_SIZE.load(Ordering::Relaxed).max(1);
    m.prompt_tokens = prompt_tokens;
    m.prefix_hit_tokens = seq.cached_prefix_tokens;
    m.reused_blocks = seq.cached_prefix_tokens / bs;
    m.known_blocks = seq.block_table.len();
    m.prefill_blocks_new = m.known_blocks.saturating_sub(m.reused_blocks);
    tracing::debug!(
        target: "memdebug",
        seq = m.id,
        resumed = m.resumed,
        event = "prefill",
        prompt_tokens,
        prefix_hit_tokens = m.prefix_hit_tokens,
        reused_blocks = m.reused_blocks,
        new_blocks = m.prefill_blocks_new,
        blocks_held = m.known_blocks,
        "prefill complete"
    );
}

/// Swap-in restore: sync the growth detector to the restored block table.
pub fn on_resumed(mem: &mut Option<Box<SeqMemDebug>>, seq: &SequenceState) {
    let Some(m) = mem else { return };
    m.known_blocks = seq.block_table.len();
    tracing::debug!(
        target: "memdebug",
        seq = m.id,
        event = "resume",
        blocks_held = m.known_blocks,
        seq_len = seq.seq_len,
        "sequence restored from disk swap"
    );
}

/// Once-per-scheduler-step decode hook (covers MTP, non-MTP and mixed
/// paths — called before `retire_finished_sequences`). Detects block-table
/// growth and attributes it to the current phase, logs the thinking →
/// generation transition on the `inside_thinking` flip itself, and emits a
/// periodic summary every 128 output tokens.
pub fn on_decode_tick(
    mem: &mut Option<Box<SeqMemDebug>>,
    seq: &SequenceState,
    output_tokens: usize,
    thinking_tokens: u32,
    inside_thinking: bool,
    model: &dyn Model,
) {
    let Some(m) = mem else { return };
    let phase = if inside_thinking {
        "thinking"
    } else {
        "generation"
    };
    if m.was_inside_thinking == Some(true) && !inside_thinking {
        tracing::debug!(
            target: "memdebug",
            seq = m.id,
            event = "phase_transition",
            thinking_tokens,
            thinking_blocks = m.thinking_blocks,
            "thinking -> generation"
        );
    }
    m.was_inside_thinking = Some(inside_thinking);

    let bt_len = seq.block_table.len();
    if bt_len > m.known_blocks {
        let grew = bt_len - m.known_blocks;
        if inside_thinking {
            m.thinking_blocks += grew;
        } else {
            m.generation_blocks += grew;
        }
        tracing::debug!(
            target: "memdebug",
            seq = m.id,
            event = "block_alloc",
            phase,
            new_blocks = ?&seq.block_table[m.known_blocks..],
            blocks_held = bt_len,
            output_tokens,
            thinking_tokens,
            pool_free = model.num_free_blocks(),
            "block(s) allocated during decode"
        );
        m.known_blocks = bt_len;
    } else if bt_len < m.known_blocks {
        // HSS sliding window / watchdog rollback shrank the table — resync
        // without counting anything (releases are accounted at finish).
        m.known_blocks = bt_len;
    }

    if output_tokens >= m.next_summary_at {
        m.next_summary_at = (output_tokens / SUMMARY_STRIDE + 1) * SUMMARY_STRIDE;
        tracing::debug!(
            target: "memdebug",
            seq = m.id,
            event = "decode_summary",
            phase,
            output_tokens,
            thinking_tokens,
            generation_tokens = output_tokens.saturating_sub(thinking_tokens as usize),
            blocks_held = bt_len,
            thinking_blocks = m.thinking_blocks,
            generation_blocks = m.generation_blocks,
            pool_free = model.num_free_blocks(),
            "decode progress"
        );
    }
}

/// Block-release numbers measured around `free_sequence` by the caller.
pub struct FinishBlockAccounting {
    /// `block_table.len()` before the release.
    pub held_before: usize,
    /// Pool free-list growth across the release (blocks whose refcount hit
    /// 0; the remainder stayed alive under prefix-cache references).
    pub freed_to_pool: usize,
    /// `block_table.len()` after the release — must be 0.
    pub held_after: usize,
}

/// Sequence finished (any reason: stop/length/tool_calls/model_swap/error/
/// cancel/receiver-dropped). Logs the full release + per-phase breakdown and
/// consumes the guard.
pub fn on_finish(
    mem: &mut Option<Box<SeqMemDebug>>,
    reason: &str,
    acct: FinishBlockAccounting,
    output_tokens: usize,
    thinking_tokens: u32,
    model: &dyn Model,
) {
    let Some(m) = mem.as_mut() else { return };
    m.closed = true;
    if acct.held_after > 0 {
        tracing::error!(
            target: "memdebug",
            seq = m.id,
            held_after = acct.held_after,
            "KV LEAK: sequence still holds blocks after free_sequence"
        );
    }
    let p = pool_view(model);
    let (gpu_free, _) = gpu_free_total_mib();
    let rss = rss_mib();
    let gpu_delta = if m.start_gpu_free_mib >= 0 && gpu_free >= 0 {
        gpu_free - m.start_gpu_free_mib
    } else {
        0
    };
    let rss_delta = if m.start_rss_mib >= 0 && rss >= 0 {
        rss - m.start_rss_mib
    } else {
        0
    };
    tracing::debug!(
        target: "memdebug",
        seq = m.id,
        resumed = m.resumed,
        event = "finish",
        reason,
        blocks_released = acct.held_before,
        freed_to_pool = acct.freed_to_pool,
        kept_by_prefix_cache = acct.held_before.saturating_sub(acct.freed_to_pool),
        held_after = acct.held_after,
        prefill_prompt_tokens = m.prompt_tokens,
        prefill_prefix_hit_tokens = m.prefix_hit_tokens,
        prefill_new_blocks = m.prefill_blocks_new,
        thinking_tokens,
        thinking_blocks = m.thinking_blocks,
        generation_tokens = output_tokens.saturating_sub(thinking_tokens as usize),
        generation_blocks = m.generation_blocks,
        pool_total = p.total,
        pool_free = p.free,
        cached_evictable = p.cached_evictable,
        gpu_free_mib = gpu_free,
        gpu_delta_mib = gpu_delta,
        gpu_drift_mib = drift_vs_first(&FIRST_GPU_FREE_MIB, gpu_free),
        rss_mib = rss,
        rss_delta_mib = rss_delta,
        rss_drift_mib = drift_vs_first(&FIRST_RSS_MIB, rss),
        "sequence finished"
    );
    *mem = None;
}

/// Non-finish exit that still released blocks correctly (prefill error,
/// hot-swap drain of an in-progress prefill, shutdown drain). Consumes the
/// guard so the Drop backstop stays reserved for genuinely missed releases.
pub fn on_aborted(mem: &mut Option<Box<SeqMemDebug>>, why: &str) {
    let Some(m) = mem.as_mut() else { return };
    m.closed = true;
    tracing::debug!(
        target: "memdebug",
        seq = m.id,
        event = "abort",
        why,
        "sequence aborted; blocks released by the abort path"
    );
    *mem = None;
}

/// Swap-out to disk: blocks are freed, state continues later under a fresh
/// guard (`new_seq_resumed`). Consumes this guard.
pub fn on_swap_out(mem: &mut Option<Box<SeqMemDebug>>, blocks: usize) {
    let Some(m) = mem.as_mut() else { return };
    m.closed = true;
    tracing::debug!(
        target: "memdebug",
        seq = m.id,
        event = "swap_out",
        blocks_released = blocks,
        thinking_blocks = m.thinking_blocks,
        generation_blocks = m.generation_blocks,
        "sequence swapped out to disk"
    );
    *mem = None;
}

/// Idle invariant: called each loop iteration where the scheduler has zero
/// live sequences (active, prefilling and swapped all empty). Logs once per
/// busy→idle transition. `leaked = used − cached_evictable − permanent`
/// must be 0 — anything above means blocks are held with no owner.
pub fn on_idle(model: &dyn Model) {
    if !enabled() || !WAS_BUSY.swap(false, Ordering::Relaxed) {
        return;
    }
    let p = pool_view(model);
    if BASELINE_TOTAL.load(Ordering::Relaxed) != p.total {
        return; // no baseline for this pool yet (no instrumented request)
    }
    let permanent = BASELINE_USED.load(Ordering::Relaxed);
    let used = p.total.saturating_sub(p.free);
    let leaked = used as i64 - p.cached_evictable as i64 - permanent as i64;
    if leaked > 0 {
        tracing::error!(
            target: "memdebug",
            leaked_blocks = leaked,
            active_used = 0,
            used_blocks = used,
            cached_evictable = p.cached_evictable,
            permanent_blocks = permanent,
            pool_total = p.total,
            "KV LEAK: blocks held by sequences while scheduler is idle"
        );
    } else {
        tracing::debug!(
            target: "memdebug",
            used_blocks = used,
            cached_evictable = p.cached_evictable,
            permanent_blocks = permanent,
            unaccounted = leaked,
            pool_total = p.total,
            "idle invariant ok"
        );
    }
}

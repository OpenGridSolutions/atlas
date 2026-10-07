// SPDX-License-Identifier: AGPL-3.0-only

//! Scheduler: batched concurrent decode on a single GPU thread.
//! Architecture:
//! - Receiver thread: blocks on request channel, pushes to pending queue,
//!   signals condvar (instantaneous wake, zero polling).
//! - Scheduler thread: prefills new requests sequentially, then runs
//!   batched decode steps via `model.decode_batch()` — weights loaded once
//!   per step for all active sequences.
//!
//! When idle (no active sequences): blocks on condvar (zero CPU).
//! When busy: drains pending queue (mutex lock) after each decode step.

// ── Submodules (split for ≤500 LoC files) ──────────────────────────────────
mod confidence;
mod decode_logits_content;
mod decode_logits_seq;
mod decode_logits_step;
mod decode_step;
mod emit_step;
mod fast_greedy;
mod helpers;
mod hotswap_ctl;
mod lifecycle;
mod logit_dump;
mod logit_processors;
mod logprobs;
mod mod_helpers;
mod mtp_gate;
mod mtp_step;
pub(crate) mod mtp_timing;
mod phase_continue_prefills;
mod phase_promote_prefills;
mod phase_start_prefills;
mod prefill_a_step;
mod prefill_a_step_params;
mod prefill_b_step;
mod repetition;
mod rollback;
mod sample_step;
mod spec_step;
mod ssm_decode_ring;
mod types;
mod verify_dflash_step;
mod verify_k2_step;
mod verify_k3_step;
mod verify_k4_step;
mod verify_pipeline_helper;

use confidence::*;
use decode_logits_content::*;
use decode_logits_seq::*;
use decode_logits_step::*;
use decode_step::*;
use emit_step::*;
pub use helpers::disable_watchdogs;
pub use helpers::set_boundary_token_mask;
pub use helpers::set_enable_loop_watchdog;
pub use helpers::set_fence_token_mask;
pub use helpers::set_im_start_hard_stop;
pub use helpers::set_mid_word_token_mask;
pub use helpers::set_numeric_token_mask;
pub use helpers::set_tool_response_hard_stop;
use helpers::*;
pub use helpers::{CONTENT_LOOP_PERIOD_MAX, CONTENT_LOOP_PERIOD_MIN};
pub use helpers::{DigitTok, set_numeric_token_values};
pub use helpers::{WatchdogParams, set_watchdog_params};
use lifecycle::*;
use logprobs::*;
use mod_helpers::*;
use mtp_step::*;
use phase_continue_prefills::continue_in_progress_prefills;
use phase_start_prefills::start_new_requests;
use prefill_a_step::*;
use prefill_b_step::*;
use repetition::*;
use rollback::{RollbackOutcome, rollback_to_boundary};
use sample_step::*;
use spec_step::*;
use ssm_decode_ring::SsmDecodeRing;
pub use types::ControlMsg;
use types::*;
use verify_dflash_step::*;
use verify_k2_step::*;
use verify_k3_step::*;
use verify_k4_step::*;
// verify_pipeline_helper is referenced via fully-qualified
// `crate::scheduler::verify_pipeline_helper::...` from sibling step
// files (verify_k2/k3/k4/dflash + spec_step), so no `use` import.

// Re-exports threaded through `use super::*;` in sibling step files —
// keep these imports here even though `run` itself doesn't reference all
// of them directly (see scheduler/decode_step.rs etc.).
use anyhow::Result;
use parking_lot::{Condvar, Mutex};
use spark_model::traits::{Model, SequenceState};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_spill::KvSpillManager;
use spark_runtime::sampler::{
    SamplingParams, apply_penalties_and_bias, sample_with_params, sample_with_params_history,
};

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::Instant;

/// Live scheduler occupancy (active + in-progress prefills), published
/// once per loop iteration for the arena dispatcher's N_active term
/// (BAT-11). Process-global like the kv_stats counters.
static SCHED_LOAD: AtomicUsize = AtomicUsize::new(0);

pub fn scheduler_load() -> usize {
    SCHED_LOAD.load(AtomicOrdering::Relaxed)
}

use crate::api::{GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent};
use crate::grammar::{GrammarEngine, GrammarState};
use crate::ngram::NgramProposer;
use crate::scheduling_policy::SchedulingPolicy;

/// Everything per-model the scheduler loop holds besides the model itself.
/// Handed over as one unit on a hot-swap load so a different model installs
/// ITS tokenizer-derived tokens and grammar engine (Task 6 Phase 1.3) —
/// keeping model A's `</think>` id while decoding model B would be silently
/// wrong sampling.
pub struct SchedulerModelParams {
    pub eos_tokens: Vec<u32>,
    pub think_end_token: Option<u32>,
    pub think_start_token: Option<u32>,
    pub code_fence_token: Option<u32>,
    pub tool_call_start_token: Option<u32>,
    pub tool_call_end_token: Option<u32>,
    pub grammar_engine: Option<GrammarEngine>,
    pub spontaneous_think_budget: u32,
}

/// Result of a hot-swap load: the rebuilt model, its scheduler params, and
/// the JSON stats answered to the admin endpoint.
pub struct SwapOutcome {
    pub model: Box<dyn Model>,
    pub params: SchedulerModelParams,
    pub stats: serde_json::Value,
    /// NVMe KV tier bound to the swapped-in model (None = tier off).
    pub kv_tier: Option<Arc<crate::kv_tier::KvTier>>,
}

/// Rebuilds a model for a hot-swap `load` (constructed in serve.rs, runs
/// on the scheduler thread). `None` reloads the model that occupied the
/// arena; `Some(id)` swaps to a different model from its snapshot.
pub type ModelReloader = Box<dyn Fn(Option<&str>) -> Result<SwapOutcome> + Send + Sync>;

/// External handle onto the scheduler's queue: lets the admin endpoints
/// push hot-swap control commands and wake the (possibly idle) loop.
#[derive(Clone)]
pub struct SchedulerHandle {
    shared: Arc<(Mutex<PendingQueue>, Condvar)>,
}

impl Default for SchedulerHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl SchedulerHandle {
    pub fn new() -> Self {
        Self {
            shared: Arc::new((
                Mutex::new(PendingQueue {
                    requests: Vec::new(),
                    closed: false,
                    control: Vec::new(),
                }),
                Condvar::new(),
            )),
        }
    }

    pub fn push_control(&self, msg: ControlMsg) {
        self.shared.0.lock().control.push(msg);
        self.shared.1.notify_one();
    }
}

/// Run the scheduler loop on the current thread.
#[allow(clippy::too_many_arguments)]
pub fn run(
    model: Box<dyn Model>,
    request_rx: tokio::sync::mpsc::Receiver<InferenceRequest>,
    handle: SchedulerHandle,
    hotswap: Option<Arc<crate::main_modules::hotswap::HotswapCtx>>,
    model_ready: Arc<std::sync::atomic::AtomicBool>,
    reload: ModelReloader,
    params: SchedulerModelParams,
    max_batch_size: usize,
    use_speculative: bool,
    num_drafts: usize,
    policy: Box<dyn SchedulingPolicy>,
    max_prefill_tokens: usize,
    max_batch_tokens: usize,
    use_self_speculative: bool,
    use_ngram_speculative: bool,
    swap_space_gb: usize,
    high_speed_swap_cfg: Option<spark_storage::HighSpeedSwapConfig>,
    block_size: usize,
    adaptive_sampling: bool,
    mut session_manager: crate::session_manager::SessionSsmManager,
    kv_tier: Option<Arc<crate::kv_tier::KvTier>>,
    drain_grace: std::time::Duration,
    growth_reserve_pct: u8,
) {
    // BAT-3: a slice of the pool stays free for running sequences'
    // growth; admission (and its spill guard) must leave it untouched.
    let growth_reserve_blocks = spark_runtime::kv_stats::snapshot()
        .map(|p| p.total_blocks * (growth_reserve_pct.min(90) as usize) / 100)
        .unwrap_or(0);
    // NVMe tier for the boot model; re-installed per hot-swap load.
    crate::kv_tier::set_active(kv_tier);
    // Per-model scheduler state — every binding here is REBOUND when a
    // hot-swap load installs a different model (see the swap arm below).
    let SchedulerModelParams {
        mut eos_tokens,
        mut think_end_token,
        mut think_start_token,
        mut code_fence_token,
        mut tool_call_start_token,
        mut tool_call_end_token,
        mut grammar_engine,
        mut spontaneous_think_budget,
    } = params;
    let mut model = model;
    model
        .bind_gpu_to_thread()
        .expect("Failed to bind CUDA context to scheduler thread");
    let use_mtp = use_speculative && model.has_proposer();
    let num_drafts = if use_mtp || use_self_speculative || use_ngram_speculative {
        num_drafts.max(1)
    } else {
        0
    };
    let chunked = max_prefill_tokens > 0;
    // Throughput-aware MTP gate: when MTP is requested, measure the verify-step
    // cost multiplier over the first decode steps of the first lone-sequence
    // session and auto-disable MTP if it is provably net-negative. Only armed
    // for the pure-MTP path (not ngram/self/dflash, which have their own
    // economics and proposers).
    let mut mtp_gate = if use_mtp && !mtp_timing::gate_forced() {
        Some(mtp_gate::MtpGate::new(num_drafts))
    } else {
        if use_mtp && mtp_timing::gate_forced() {
            tracing::warn!(
                "ATLAS_MTP_GATE_FORCE=1: MTP throughput gate DISARMED (diagnostic; \
                 verify runs even where the gate would measure it net-negative)"
            );
        }
        None
    };
    let mut ngram_proposer = if use_ngram_speculative {
        Some(NgramProposer::new(4)) // 4-gram context
    } else {
        None
    };
    tracing::info!(
        "Scheduler started (batched mode, max_batch={max_batch_size}, mtp={}, ngram={}, num_drafts={num_drafts}, policy={}, chunked_prefill={}, max_prefill_tokens={})",
        use_mtp,
        use_ngram_speculative,
        policy.name(),
        chunked,
        if chunked { max_prefill_tokens } else { 0 },
    );

    // Holo "always-on fused mixed step" gate (default OFF). When OFF the
    // scheduler behaves EXACTLY as today (binary should_prefill, no slice
    // budget). When ON, an active decode + an in-progress prefill always
    // takes a fused mixed step sized by the policy's prefill_slice_budget
    // so decode never starves during a prefill burst. Read once at startup.
    let always_mixed = std::env::var("ATLAS_HOLO_ALWAYS_MIXED")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if always_mixed {
        tracing::info!("ATLAS_HOLO_ALWAYS_MIXED=on: fused mixed step always-on (slice-budget)");
    }

    // Shared queue lives in the handle so admin endpoints can push
    // hot-swap control commands and wake this loop.
    let pending = Arc::clone(&handle.shared);

    // Receiver thread: blocks on channel, signals scheduler via condvar.
    let p = Arc::clone(&pending);
    std::thread::spawn(move || {
        let mut rx = request_rx;
        while let Some(req) = rx.blocking_recv() {
            p.0.lock().requests.push(req);
            p.1.notify_one();
        }
        p.0.lock().closed = true;
        p.1.notify_one();
    });

    // Dedicated CUDA stream + event for prefill compute-copy overlap.
    // `mut`: recreated when a hot-swap load installs a fresh model.
    let mut prefill_stream = model
        .create_stream()
        .expect("Failed to create prefill CUDA stream");
    let mut prefill_event = model
        .create_event()
        .expect("Failed to create prefill CUDA event");

    let mut active: Vec<ActiveSeq> = Vec::new();
    let mut prefilling: Vec<PrefillInProgress> = Vec::new();
    let mut swapped: Vec<SwappedSeq> = Vec::new();
    let mut spill_manager: Option<KvSpillManager> = if swap_space_gb > 0 {
        let max_bytes = swap_space_gb as u64 * 1024 * 1024 * 1024;
        match KvSpillManager::new(PathBuf::from("/tmp/atlas-swap"), max_bytes) {
            Ok(mgr) => {
                tracing::info!("Swap space: {swap_space_gb} GB at /tmp/atlas-swap/");
                Some(mgr)
            }
            Err(e) => {
                tracing::error!("Failed to initialize swap space: {e:#}");
                None
            }
        }
    } else {
        None
    };

    install_high_speed_swap(&*model, high_speed_swap_cfg);

    // ATLAS_MEM_DEBUG: block size for prefix-reuse math (no-op when off).
    crate::memdebug::init(block_size);

    let mut swap = hotswap_ctl::SwapCtl::new(hotswap.clone(), model_ready.clone(), drain_grace);

    // ── NVMe-restore staging (BAT-6) ──
    // Requests whose prompt has a restorable prefix on the NVMe tier are
    // routed through a blocking fetch thread; the loop imports the staged
    // payloads into fresh KV blocks + the radix cache and only then lets
    // the request enter prefill. The batch never waits on disk.
    struct StagedRestore {
        req: InferenceRequest,
        prompt: Arc<Vec<u32>>,
        chain: spark_runtime::block_keys::BlockKeyChain,
        payloads: Vec<(usize, Vec<u8>)>,
        /// Hybrid models: restored SSM resume point `(token_count, bytes)`
        /// (SSM-4), registered after the KV import succeeds.
        ssm: Option<(usize, Vec<u8>)>,
    }
    let (staged_tx, staged_rx) = std::sync::mpsc::channel::<StagedRestore>();
    let mut staging_in_flight: usize = 0;
    // Bounded concurrency: the brief admission overshoot a completed
    // staging causes is capped by this.
    const MAX_STAGING: usize = 2;
    // On import failure (pool pressure) stop staging for a while instead
    // of looping stage→fail→stage on the same request.
    let mut staging_backoff_until: Option<Instant> = None;
    // Completed restores bypass the staging filter on re-entry.
    let mut restored_ready: Vec<InferenceRequest> = Vec::new();

    loop {
        // ── Hot-swap: drain-and-unload control (plan Task 4/5) ──
        if let Some(unload_reply) = swap.poll(
            &pending,
            &*model,
            &mut active,
            &mut prefilling,
            &mut swapped,
            &mut spill_manager,
        ) {
            let ctx = hotswap
                .as_ref()
                .expect("SwapCtl only accepts unload with a hotswap ctx");
            match hotswap_ctl::unload_and_park(
                model,
                unload_reply,
                &pending,
                ctx,
                &mut session_manager,
                reload.as_ref(),
            ) {
                Some(outcome) => {
                    model = outcome.model;
                    crate::kv_tier::set_active(outcome.kv_tier);
                    // Install the (possibly different) model's scheduler
                    // params — tokenizer-derived token ids, EOS set,
                    // grammar engine, thinking budget (Task 6 Phase 1.3).
                    let p = outcome.params;
                    eos_tokens = p.eos_tokens;
                    think_end_token = p.think_end_token;
                    think_start_token = p.think_start_token;
                    code_fence_token = p.code_fence_token;
                    tool_call_start_token = p.tool_call_start_token;
                    tool_call_end_token = p.tool_call_end_token;
                    grammar_engine = p.grammar_engine;
                    spontaneous_think_budget = p.spontaneous_think_budget;
                    model
                        .bind_gpu_to_thread()
                        .expect("Failed to re-bind CUDA context after hot-swap load");
                    // Per-run objects tied to the old model.
                    prefill_stream = model
                        .create_stream()
                        .expect("Failed to recreate prefill stream after hot-swap load");
                    prefill_event = model
                        .create_event()
                        .expect("Failed to recreate prefill event after hot-swap load");
                    model_ready.store(true, std::sync::atomic::Ordering::Relaxed);
                    tracing::info!("Hot-swap: scheduler reopened with reloaded model");
                }
                None => {
                    tracing::info!("Scheduler stopped (shutdown while unloaded)");
                    return;
                }
            }
        }

        // NVMe tier: re-offer writes the bounded queue rejected (cheap
        // no-op when the retry buffer is empty).
        crate::kv_tier::with_active(|t| t.pump_retries());

        // ── NVMe-restore completions: import + release into admission ──
        while let Ok(mut st) = staged_rx.try_recv() {
            staging_in_flight = staging_in_flight.saturating_sub(1);
            match model.import_cached_blocks(&st.prompt, &mut st.chain, &st.payloads) {
                Ok(covered) => {
                    if covered > 0 {
                        let bs = st.chain.block_size().max(1);
                        crate::metrics::NVME_RESTORED_TOKENS_TOTAL
                            .inc_by((st.payloads.len() * bs) as u64);
                        // The upload runs on the model's restore stream;
                        // prefill must not read the blocks before it lands.
                        if let Err(e) = model.restore_fence(prefill_stream) {
                            tracing::warn!("nvme-restore: prefill fence failed: {e:#}");
                        }
                    }
                    if covered == 0 && !st.payloads.is_empty() {
                        // Pool pressure or RAM-match drift — don't re-stage
                        // this wave; the request prefills normally.
                        staging_backoff_until =
                            Some(Instant::now() + std::time::Duration::from_secs(5));
                    }
                    // SSM resume point (SSM-4): only valid when the KV
                    // import actually covers it.
                    if let Some((tokens, payload)) = st.ssm.take()
                        && covered >= tokens
                        && let Err(e) =
                            model.import_ssm_snapshot(&st.prompt, &mut st.chain, tokens, &payload)
                    {
                        tracing::warn!("nvme-restore: ssm import failed: {e:#}");
                    }
                }
                Err(e) => {
                    tracing::warn!("nvme-restore import failed: {e:#}");
                    staging_backoff_until =
                        Some(Instant::now() + std::time::Duration::from_secs(5));
                }
            }
            restored_ready.push(st.req);
        }

        // ── Drain pending → start prefill (chunked or full) ──
        // Admission is batch-bounded: at most max_batch_size sequences,
        // and an NVMe restore in staging (or imported, awaiting re-entry)
        // already holds its place (BAT-6). Capacity itself is block-based:
        // the pool, LRU cache reclaim and the disk spill decide how many
        // sequences fit.
        let cap = max_batch_size.saturating_sub(staging_in_flight + restored_ready.len());
        // Work that arrives without a client signal — swapped-out sequences,
        // NVMe restores in flight (they complete on an mpsc channel) and
        // restores already imported — must keep the loop polling instead
        // of parking on the request condvar.
        let background_work =
            !swapped.is_empty() || staging_in_flight > 0 || !restored_ready.is_empty();
        let mut new_reqs = drain_pending_requests(
            &pending,
            &active,
            &prefilling,
            &*policy,
            cap,
            background_work,
        );

        // ── NVMe-restore staging filter (BAT-6/BAT-7) ──
        let staging_open = staging_in_flight < MAX_STAGING
            && staging_backoff_until.is_none_or(|t| Instant::now() >= t);
        if staging_open {
            crate::kv_tier::with_active(|tier| {
                let min_tokens = tier.restore_min_tokens.max(1);
                let mut i = 0;
                while i < new_reqs.len() && staging_in_flight < MAX_STAGING {
                    let req = &new_reqs[i];
                    let prompt = req.prompt_tokens_arc();
                    let digests = req.media_digests();
                    let chain = model.build_block_key_chain(&prompt, &digests);
                    let (start_block, keys): (usize, Vec<[u8; 16]>) = {
                        if !chain.is_cacheable() {
                            i += 1;
                            continue;
                        }
                        let view = chain.view(&prompt, prompt.len());
                        let bs = view.block_size();
                        if bs == 0 {
                            i += 1;
                            continue;
                        }
                        let ram_blocks = model.peek_prefix_cached_tokens(&view) / bs;
                        let keys: Vec<[u8; 16]> = view.full_keys()
                            [ram_blocks.min(view.full_keys().len())..]
                            .iter()
                            .map(|k| k.0)
                            .collect();
                        let avail = tier.probe_restore(&keys);
                        if avail * bs < min_tokens {
                            i += 1;
                            continue;
                        }
                        (ram_blocks, keys[..avail].to_vec())
                    };
                    // Stage: hand the request to the fetch thread.
                    let req = new_reqs.remove(i);
                    staging_in_flight += 1;
                    let store = Arc::clone(tier.store());
                    let tx = staged_tx.clone();
                    let hybrid = model.has_recurrent_state();
                    let bs = chain.block_size();
                    std::thread::spawn(move || {
                        // Hybrid models (SSM-4): pick the deepest `.ssm`
                        // resume point within the restorable range and load
                        // KV only up to it; without one, load everything
                        // and let the Marconi replay recompute SSM state.
                        let mut kv_keys: &[[u8; 16]] = &keys;
                        let mut ssm: Option<(usize, Vec<u8>)> = None;
                        if hybrid {
                            use spark_storage::block_store::Kind;
                            for j in (0..keys.len()).rev() {
                                if !store.contains(&keys[j], Kind::Ssm) {
                                    continue;
                                }
                                match store.get(&keys[j], Kind::Ssm) {
                                    Some(rec) => {
                                        // Token coverage from the record —
                                        // interior points store the aligned
                                        // count, end points the exact one.
                                        let tokens = (rec.header.token_count as usize)
                                            .min((start_block + j + 1) * bs);
                                        ssm = Some((tokens, rec.payload));
                                        kv_keys = &keys[..=j];
                                        break;
                                    }
                                    // Checksum/binding failure: the record
                                    // was deleted — fall back to the next
                                    // earlier resume point (SSM-5).
                                    None => continue,
                                }
                            }
                        }
                        let payloads =
                            crate::kv_tier::KvTier::fetch_blocks(&store, start_block, kv_keys);
                        let _ = tx.send(StagedRestore {
                            req,
                            prompt,
                            chain,
                            payloads,
                            ssm,
                        });
                    });
                }
            });
        }
        // Completed restores enter admission ahead of fresh arrivals —
        // their blocks are RAM-resident now (BAT-6's "erst dann in den
        // Batch").
        if !restored_ready.is_empty() {
            restored_ready.extend(new_reqs);
            new_reqs = std::mem::take(&mut restored_ready);
        }
        // ATLAS_MEM_DEBUG: admission-time pool stats + GPU/RSS context.
        if crate::memdebug::enabled() && !new_reqs.is_empty() {
            let active_used: usize = active
                .iter()
                .map(|a| a.seq.block_table.len())
                .chain(prefilling.iter().map(|p| p.seq.block_table.len()))
                .sum();
            for req in &new_reqs {
                crate::memdebug::on_request_admitted(&*model, req.prompt_len(), active_used);
            }
        }
        if new_reqs.is_empty() && active.is_empty() && prefilling.is_empty() {
            // Receiver thread was closed (shutdown).
            let pending_closed = pending.0.lock().closed;
            if pending_closed {
                break;
            }
        }

        // ── Swap-out: evict active sequences to disk when blocks run low ──
        if let Some(ref mut spill) = spill_manager {
            for req in &new_reqs {
                let prompt_len = req.prompt_len();
                // BAT-3(a): admission may not consume the growth reserve.
                let blocks_needed = prompt_len / block_size + 1 + growth_reserve_blocks;
                while model.reclaim_free_blocks(blocks_needed) < blocks_needed && !active.is_empty()
                {
                    let victim_idx = active
                        .iter()
                        .enumerate()
                        .filter(|(_, a)| a.grammar_state.is_none())
                        .max_by_key(|(_, a)| a.seq.block_table.len())
                        .map(|(i, _)| i);
                    let Some(victim_idx) = victim_idx else {
                        tracing::warn!("No swappable sequences (all grammar-active)");
                        break;
                    };
                    match swap_out_sequence(&*model, &mut active, victim_idx, spill) {
                        Ok(s) => {
                            tracing::info!(
                                "Swap-out: evicted seq (seq_len={}, blocks={}) to disk",
                                s.seq_len,
                                s.num_blocks,
                            );
                            swapped.push(s);
                        }
                        Err(e) => {
                            tracing::error!("Swap-out failed: {e:#}");
                            break;
                        }
                    }
                }
            }
        }

        // ── Start new requests ──
        start_new_requests(
            &*model,
            new_reqs,
            chunked,
            always_mixed,
            max_prefill_tokens,
            max_batch_tokens,
            &eos_tokens,
            prefill_stream,
            prefill_event,
            &mut grammar_engine,
            spontaneous_think_budget,
            think_end_token,
            think_start_token,
            code_fence_token,
            tool_call_start_token,
            tool_call_end_token,
            &mut active,
            &mut prefilling,
        );

        // ── Decode-time pool guard (Todo.md 0.6 / case A6) ──
        // The admission-time swap-out above only runs when NEW requests
        // arrive. Sequences that grow the pool dry between arrivals used
        // to hit the block allocator mid-step, and the decode_batch error
        // path kills the WHOLE batch. Worst case per step is one fresh
        // block per active sequence (a step appends ≤ num_drafts+1 << 16
        // tokens), so preemptively swap out the largest non-grammar
        // sequence until the step is covered. Never the last one: a lone
        // sequence that outgrew the pool gains nothing from swapping
        // itself out (its restore needs the same blocks back).
        if let Some(ref mut spill) = spill_manager {
            while active.len() > 1 && model.reclaim_free_blocks(active.len()) < active.len() {
                let victim_idx = active
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| a.grammar_state.is_none())
                    .max_by_key(|(_, a)| a.seq.block_table.len())
                    .map(|(i, _)| i);
                let Some(victim_idx) = victim_idx else {
                    tracing::warn!(
                        "Decode-time guard: pool low ({} free blocks, {} active) but no \
                         swappable sequence (all grammar-active) — decode may fail this step",
                        model.num_free_blocks(),
                        active.len()
                    );
                    break;
                };
                match swap_out_sequence(&*model, &mut active, victim_idx, spill) {
                    Ok(s) => {
                        tracing::info!(
                            "Decode-time guard: KV pool low — swapped out seq \
                             (seq_len={}, blocks={}) to disk; {} active remain",
                            s.seq_len,
                            s.num_blocks,
                            active.len(),
                        );
                        swapped.push(s);
                    }
                    Err(e) => {
                        tracing::error!("Decode-time guard: swap-out failed: {e:#}");
                        break;
                    }
                }
            }
        }

        // ── Continue in-progress prefills ──
        let did_mixed_step = continue_in_progress_prefills(
            &*model,
            &*policy,
            &mut active,
            &mut prefilling,
            max_prefill_tokens,
            max_batch_tokens,
            always_mixed,
            prefill_stream,
            prefill_event,
            use_mtp,
            use_self_speculative,
            use_ngram_speculative,
            think_end_token,
            think_start_token,
            code_fence_token,
            tool_call_start_token,
            tool_call_end_token,
            adaptive_sampling,
        );

        if active.is_empty() {
            // Swapped-out sequences must be able to come back without a
            // new arrival — the swap-in below the decode step is skipped
            // by this `continue`, so run it here (it reclaims cache blocks
            // itself; before this, an idle arena with only swapped
            // sequences hung forever).
            if let Some(ref mut spill) = spill_manager
                && !swapped.is_empty()
            {
                swap_in_ready(
                    &*model,
                    spill,
                    &mut swapped,
                    &mut active,
                    max_batch_size,
                    think_end_token,
                    think_start_token,
                );
                if !active.is_empty() {
                    SCHED_LOAD.store(active.len() + prefilling.len(), AtomicOrdering::Relaxed);
                    continue;
                }
            }
            SCHED_LOAD.store(prefilling.len(), AtomicOrdering::Relaxed);
            // ATLAS_MEM_DEBUG: a prefill error can end the last live request
            // here (before the end-of-loop check is reached via `continue`).
            if prefilling.is_empty() && swapped.is_empty() {
                crate::memdebug::on_idle(&*model);
            }
            continue;
        }

        // Skip decode when mixed_forward already processed decode logits.
        if !did_mixed_step {
            // Ensure any in-flight prefill work on the prefill stream is complete
            // before decode starts on the default stream.
            if !prefilling.is_empty() {
                let _ = model.record_event(prefill_event, prefill_stream);
                let _ = model.stream_wait_event(model.default_stream(), prefill_event);
            }

            // Build the verify-time LogitsContext once per step: the
            // tokenizer special-token IDs the verify pipeline needs to
            // run the same 8-stage logits processors the non-MTP path
            // applies (mid-word/post-close/tool-during-think/forced-
            // think-end/pin-tool-call/forced-token/grammar). Without
            // this context the MTP/spec verify path emits unmasked
            // GPU-argmax tokens (Phase C-2 root cause, 2026-05-24).
            let verify_ctx = crate::scheduler::logit_processors::LogitsContext {
                think_end_token,
                think_start_token,
                tool_call_start_token,
                tool_call_end_token,
            };
            if use_ngram_speculative && active.len() == 1 && active[0].grammar_state.is_none() {
                // N-gram speculative: CPU proposer + CUDA-graphed K=2 verify.
                if let Some(ref mut proposer) = ngram_proposer {
                    step_ngram(&*model, &mut active, proposer, &verify_ctx);
                }
            } else if use_self_speculative && active.len() == 1 && active[0].grammar_state.is_none()
            {
                // Self-speculative: draft via layer-skipping, verify with full model.
                step_self_spec(&*model, &mut active, num_drafts, &verify_ctx);
            } else if use_mtp
                && active.len() == 1
                && !active[0].inside_thinking
                && !active[0].suppress_tool_call
                && !active[0].disable_mtp
            {
                // Throughput-aware MTP gate: while still measuring, run the
                // step type the gate asks for and time it. Both step types emit
                // real, correct tokens (MTP verify and plain decode are greedy-
                // equivalent), so the measurement window does not waste work.
                //
                // The verify/decode multiplier is DEPTH-DEPENDENT (weight-bound
                // at short context vs KV/SSM-bound at depth — see mtp_gate module
                // docs), so a decision is only valid for the regime it was
                // measured in: re-open measurement when the live depth leaves it.
                if let Some(gate) = mtp_gate.as_mut() {
                    gate.maybe_remeasure(active[0].seq.seq_len);
                }
                if let Some(gate) = mtp_gate.as_mut().filter(|g| g.is_measuring()) {
                    gate.note_depth(active[0].seq.seq_len);
                    match gate.next_step() {
                        mtp_gate::GateStep::MeasureDecode => {
                            let t0 = std::time::Instant::now();
                            step_decode_only(
                                &*model,
                                &mut active,
                                think_end_token,
                                think_start_token,
                                code_fence_token,
                                tool_call_start_token,
                                tool_call_end_token,
                                adaptive_sampling,
                            );
                            mtp_gate
                                .as_mut()
                                .expect("gate present")
                                .record_decode(t0.elapsed());
                        }
                        mtp_gate::GateStep::MeasureVerify => {
                            // Only a true verify forward (drafts already
                            // proposed) is a representative sample; a bootstrap-
                            // only step proposes the first draft and is skipped.
                            let had_draft = !active[0].pending_drafts.is_empty();
                            let t0 = std::time::Instant::now();
                            step_mtp(&*model, &mut active, num_drafts, &verify_ctx);
                            if had_draft {
                                mtp_gate
                                    .as_mut()
                                    .expect("gate present")
                                    .record_verify(t0.elapsed());
                            }
                        }
                    }
                    // One-time transition work for a freshly-reached DISABLE:
                    // drop pending drafts and order the draft-head state resync
                    // before the next plain decode reads it. NOT permanent —
                    // `maybe_remeasure` above re-opens measurement when the
                    // depth regime changes, so MTP can come back exactly where
                    // it pays (deep agentic contexts).
                    if mtp_gate
                        .as_mut()
                        .and_then(mtp_gate::MtpGate::take_fresh_decision)
                        == Some(mtp_gate::GateDecision::DisableMtp)
                    {
                        for a in active.iter_mut() {
                            a.pending_drafts.clear();
                        }
                        if let Err(e) = model.sync_secondary() {
                            tracing::error!("mtp-gate→decode sync_secondary: {e:#}");
                        }
                    }
                } else if mtp_gate
                    .as_ref()
                    .is_some_and(|g| g.decision() == Some(mtp_gate::GateDecision::DisableMtp))
                {
                    // Measured net-negative in the CURRENT depth regime: plain
                    // decode. Consulted every step (not a latched kill switch)
                    // so the regime re-measurement above can flip it back.
                    step_decode_only(
                        &*model,
                        &mut active,
                        think_end_token,
                        think_start_token,
                        code_fence_token,
                        tool_call_start_token,
                        tool_call_end_token,
                        adaptive_sampling,
                    );
                } else {
                    // MTP wins in this regime (or no gate): speculative decode.
                    step_mtp(&*model, &mut active, num_drafts, &verify_ctx);
                }
            } else {
                // Batch decode (no MTP). Clear stale drafts when transitioning out of MTP mode.
                if use_mtp {
                    for a in active.iter_mut() {
                        a.pending_drafts.clear();
                    }
                    // MTP→decode-only transition: the last verify commit's
                    // live-state restore runs async on the secondary stream;
                    // order it before this decode reads h_state/conv_state
                    // (GPU-side event wait, zero CPU cost).
                    if let Err(e) = model.sync_secondary() {
                        tracing::error!("mtp→decode sync_secondary: {e:#}");
                    }
                }
                step_decode_only(
                    &*model,
                    &mut active,
                    think_end_token,
                    think_start_token,
                    code_fence_token,
                    tool_call_start_token,
                    tool_call_end_token,
                    adaptive_sampling,
                );
            }
        }

        // ATLAS_MEM_DEBUG: once-per-step decode tick BEFORE retirement so a
        // finishing step's growth/transition is still observed. Covers the
        // MTP, non-MTP and fused-mixed paths (all converge here). No-op per
        // sequence when the guard is None.
        for a in active.iter_mut() {
            crate::memdebug::on_decode_tick(
                &mut a.mem,
                &a.seq,
                a.output_tokens.len(),
                a.thinking_tokens,
                a.inside_thinking,
                &*model,
            );
        }

        retire_finished_sequences(&*model, &mut active);

        // ── Swap-in: resume swapped sequences when blocks free up ──
        if let Some(ref mut spill) = spill_manager {
            swap_in_ready(
                &*model,
                spill,
                &mut swapped,
                &mut active,
                max_batch_size,
                think_end_token,
                think_start_token,
            );
        }

        SCHED_LOAD.store(active.len() + prefilling.len(), AtomicOrdering::Relaxed);

        // ATLAS_MEM_DEBUG: idle invariant — zero live sequences after this
        // step's retirement. Logs once per busy→idle transition; the next
        // loop iteration blocks on the condvar until new work arrives.
        if active.is_empty() && prefilling.is_empty() && swapped.is_empty() {
            crate::memdebug::on_idle(&*model);
        }
    }

    // Periodic session eviction: free SSM snapshots for expired sessions.
    {
        let freed_slots = session_manager.evict_expired();
        if !freed_slots.is_empty() {
            tracing::info!(
                "Session eviction: freed {} SSM snapshot slot(s), {} sessions active",
                freed_slots.len(),
                session_manager.session_count()
            );
        }
    }

    // Drain any remaining active sequences on shutdown.
    for mut a in active {
        finish_sequence(&*model, &mut a);
    }
    if let Some(ref mut spill) = spill_manager {
        for s in swapped {
            let _ = spill.remove_file(s.swap_id);
        }
    }
    for mut p in prefilling {
        // Regular shutdown exit — consume the memdebug guard (blocks are
        // freed right below); the Drop backstop stays reserved for
        // genuinely missed releases.
        crate::memdebug::on_aborted(&mut p.mem, "shutdown");
        let mut seq = p.seq;
        let _ = model.free_sequence(&mut seq);
        let _ = model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF1);
    }
    // Shutdown applies to every slot the worker has; seq_id is ignored.
    let _ = model.ep_broadcast_cmd_for_seq(0, 0xFFFFFFFF);
    tracing::info!("Scheduler stopped");
}

// SPDX-License-Identifier: AGPL-3.0-only

//! Hot-swap control inside the scheduler loop (HOTSWAP_MVP_PLAN.md Tasks
//! 4/5): drain → unload → park → reload, all on the scheduler thread, which
//! owns the model and has the CUDA context bound.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use parking_lot::{Condvar, Mutex};
use serde_json::json;
use spark_model::traits::Model;

use super::lifecycle::{finish_sequence_with_reason, send_error_to_sink};
use super::types::{ActiveSeq, ControlMsg, PendingQueue, PrefillInProgress, SwappedSeq};
use crate::api::{InferenceRequest, StreamEvent};
use crate::main_modules::hotswap::{HotswapCtx, SwapState};

// Drain grace is configurable since the pressure-scheduler stage
// (`--drain-grace-ms`, DRN-1); SwapCtl carries the value.

type Shared = Arc<(Mutex<PendingQueue>, Condvar)>;
type Reply = tokio::sync::oneshot::Sender<Result<serde_json::Value>>;

/// Reply to a request that cannot be served because the model is draining
/// or unloaded.
pub(super) fn fail_request(req: InferenceRequest, msg: &str) {
    match req {
        InferenceRequest::Blocking { response_tx, .. } => {
            let _ = response_tx.send(Err(anyhow::anyhow!("{msg}")));
        }
        InferenceRequest::Streaming { token_tx, .. } => {
            let _ = token_tx.blocking_send(StreamEvent::Error(msg.to_string()));
        }
    }
}

/// Per-run hot-swap state machine, polled at the top of the scheduler loop.
pub(super) struct SwapCtl {
    hotswap: Option<Arc<HotswapCtx>>,
    model_ready: Arc<AtomicBool>,
    /// DRN-1: grace before force-finishing running sequences.
    drain_grace: Duration,
    /// In-progress drain: reply channel + force-finish deadline.
    draining: Option<(Reply, Instant)>,
}

impl SwapCtl {
    pub fn new(
        hotswap: Option<Arc<HotswapCtx>>,
        model_ready: Arc<AtomicBool>,
        drain_grace: Duration,
    ) -> Self {
        Self {
            hotswap,
            model_ready,
            drain_grace,
            draining: None,
        }
    }

    /// Poll control commands and advance a drain. Returns the unload reply
    /// channel once the model must be torn down NOW (drain complete or
    /// grace expired; all sequences already finished/failed).
    pub fn poll(
        &mut self,
        pending: &Shared,
        model: &dyn Model,
        active: &mut Vec<ActiveSeq>,
        prefilling: &mut Vec<PrefillInProgress>,
        swapped: &mut Vec<SwappedSeq>,
        spill_manager: &mut Option<spark_runtime::kv_spill::KvSpillManager>,
    ) -> Option<Reply> {
        // Drain the control queue (and, while draining, fail queued
        // requests before the admission path can see them).
        let msgs: Vec<ControlMsg> = {
            let mut g = pending.0.lock();
            if self.draining.is_some() {
                for req in g.requests.drain(..) {
                    fail_request(req, "model is unloading (state: draining); retry shortly");
                }
            }
            g.control.drain(..).collect()
        };
        for msg in msgs {
            match msg {
                ControlMsg::Unload { reply } => {
                    if self.hotswap.is_none() {
                        let _ = reply.send(Err(anyhow::anyhow!(
                            "hot-swap unavailable (weights arena disabled)"
                        )));
                        continue;
                    }
                    if self.draining.is_some() {
                        let _ = reply.send(Err(anyhow::anyhow!("unload already in progress")));
                        continue;
                    }
                    tracing::info!(
                        "Hot-swap: unload requested ({} active, {} prefilling, {} swapped)",
                        active.len(),
                        prefilling.len(),
                        swapped.len()
                    );
                    self.model_ready.store(false, Ordering::Relaxed);
                    if let Some(ctx) = &self.hotswap {
                        ctx.set_state(SwapState::Draining);
                    }
                    self.draining = Some((reply, Instant::now() + self.drain_grace));
                }
                ControlMsg::Load { reply, .. } => {
                    let _ = reply.send(Err(anyhow::anyhow!(
                        "model is loaded (state: {}); unload first",
                        self.hotswap
                            .as_ref()
                            .map(|c| c.state().as_str())
                            .unwrap_or("running")
                    )));
                }
            }
        }

        let (_, deadline) = self.draining.as_ref()?;
        let drained = active.is_empty() && prefilling.is_empty() && swapped.is_empty();
        if !drained && Instant::now() < *deadline {
            return None; // keep stepping; sequences may finish in time
        }
        if !drained {
            tracing::warn!(
                "Hot-swap: drain grace expired — force-finishing {} active, {} prefilling, \
                 {} swapped with finish_reason=model_swap",
                active.len(),
                prefilling.len(),
                swapped.len()
            );
        }
        for mut a in active.drain(..) {
            finish_sequence_with_reason(model, &mut a, Some("model_swap"));
        }
        for mut p in prefilling.drain(..) {
            send_error_to_sink(&mut p.sink, "request aborted: model_swap");
            // Regular drain exit — consume the memdebug guard so the Drop
            // backstop stays reserved for genuinely missed releases.
            crate::memdebug::on_aborted(&mut p.mem, "model_swap");
            if let Err(e) = model.free_sequence(&mut p.seq) {
                tracing::error!("hot-swap drain: free_sequence(prefilling): {e:#}");
            }
        }
        for mut s in swapped.drain(..) {
            send_error_to_sink(&mut s.sink, "request aborted: model_swap");
            if let Some(spill) = spill_manager {
                let _ = spill.remove_file(s.swap_id);
            }
        }
        // NVM-2 / DRN-3: the force-finished sequences above enqueued their
        // persists (finish_sequence_with_reason → KvTier::persist_sequence);
        // drain the write queue bounded by persist_timeout before the
        // teardown invalidates nothing — the store outlives the model, but
        // the payload exports must reach disk before we stop pumping.
        crate::kv_tier::with_active(|t| {
            let queued_gib = t.stats().1 as f64 / (1u64 << 30) as f64;
            let t0 = Instant::now();
            t.flush();
            if let Some(ctx) = &self.hotswap {
                crate::swap_costs::observe_persist(
                    &ctx.active_model().model_id,
                    t0.elapsed().as_secs_f64() * 1000.0,
                    queued_gib,
                );
            }
        });
        let (reply, _) = self.draining.take().expect("draining checked above");
        Some(reply)
    }
}

/// Tear down the model's device allocations, drop it, and reset the arena.
/// Consumes the model; runs on the scheduler thread.
fn unload_model(model: Box<dyn Model>, hotswap: &HotswapCtx) -> Result<serde_json::Value> {
    let t0 = Instant::now();
    // Release accounting (Todo.md case B1): meminfo + raw CUDA before and
    // after teardown prove the swap actually returned memory to the kernel
    // — the 2026-09-11 failure mode was exactly this not happening (arena
    // pages stayed resident, follow-up loads found no KV budget). Logged on
    // the `hotswap` target AND returned in the unload reply so
    // scripts/hotswap_release_test.sh can assert it from outside.
    let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
    let mem_before = spark_runtime::meminfo::read();
    let cuda_before = spark_runtime::cuda_backend::cuda_mem_get_info_raw();
    let pool_before = spark_runtime::kv_stats::snapshot();
    let arena_used = hotswap.arena.used();
    // Attribute the outgoing model's residency time (Task 6 §5 metrics).
    {
        let active = hotswap.active_model();
        let since = *hotswap.active_since.lock();
        crate::metrics::MODEL_RESIDENCY_SECONDS_TOTAL
            .with_label_values(&[active.model_id.as_str()])
            .inc_by(since.elapsed().as_secs());
    }
    let (freed, free_failed, allocs) = {
        let gpu = model
            .gpu_backend()
            .context("model exposes no GPU backend — cannot tear down device allocations")?;
        gpu.device_synchronize()?;
        // Read before the drain: what the build allocated, and what was
        // allocated after it (never part of the KV budget, not freed here).
        let allocs = gpu.alloc_stats();
        // Free everything build_model allocated and never freed (KV pool
        // slabs, SSM pools, transposes, dequants, buffers — trace.md §5).
        // The arena free-guard makes any arena view in the registry a no-op,
        // and arena offsets are never allocated through alloc() anyway.
        let tracked = gpu.take_tracked_allocs();
        let mut freed = 0usize;
        let mut failed = 0usize;
        for ptr in tracked {
            match gpu.free(ptr) {
                Ok(()) => freed += 1,
                Err(e) => {
                    failed += 1;
                    tracing::error!("hot-swap unload: free({ptr}): {e:#}");
                }
            }
        }
        gpu.set_alloc_tracking(false);
        gpu.device_synchronize()?;
        (freed, failed, allocs)
    };
    // Drops the WeightStore index views and pinned host staging; all device
    // memory was released via the registry above. Weights live on in the
    // arena mapping, which is never returned to the kernel.
    drop(model);
    hotswap.arena.reset();
    let duration_ms = t0.elapsed().as_millis() as u64;
    // MOD-5/8: refine the outgoing model's unload cost.
    crate::swap_costs::observe_unload(&hotswap.active_model().model_id, duration_ms as f64);
    tracing::info!(
        "Hot-swap: model unloaded ({freed} device allocations freed, {free_failed} failed, \
         {duration_ms} ms); arena reset (base {} unchanged)",
        hotswap.arena.base()
    );
    tracing::info!(
        "Hot-swap: device allocations of the unloaded model — build {:.2} GiB (freed); \
         after the build: peak {:.1} MiB, still live {:.1} MiB in {} allocation(s) (not freed)",
        gib(allocs.build_live_bytes),
        allocs.late_peak_bytes as f64 / (1024.0 * 1024.0),
        allocs.late_live_bytes as f64 / (1024.0 * 1024.0),
        allocs.late_live_count,
    );
    let mem_after = spark_runtime::meminfo::read();
    let cuda_after = spark_runtime::cuda_backend::cuda_mem_get_info_raw();
    // Signed: co-tenants allocating during the teardown can legitimately
    // push this below the arena payload; a value near zero after a big
    // model is the leak signal (was permanently ~0 before the
    // MADV_DONTNEED fix in WeightsArena::reset).
    let reclaimed_gb = match (mem_before, mem_after) {
        (Some(b), Some(a)) => Some(gib(a.available_for_alloc) - gib(b.available_for_alloc)),
        _ => None,
    };
    // Todo.md 0.3: the boot-time baseline is stale after a swap (see the
    // --kv-pool-gb note in factory::build). Re-arm it at the post-teardown
    // state — with the arena pages just released this IS "our footprint =
    // zero" for the next load's self-relative budget. Same max(cuda,
    // meminfo) rule as gpu.free_memory().
    let free_after = match (cuda_after, mem_after) {
        (Some((f, _)), Some(m)) => Some(f.max(m.available_for_alloc)),
        (Some((f, _)), None) => Some(f),
        (None, Some(m)) => Some(m.available_for_alloc),
        (None, None) => None,
    };
    if let Some(free) = free_after {
        spark_runtime::gpu::set_baseline_free_bytes(free);
        tracing::info!(
            target: "hotswap",
            baseline_free_gib = gib(free),
            "KV-budget baseline re-armed post-unload"
        );
    }
    tracing::info!(
        target: "hotswap",
        event = "unload_release",
        freed_allocs = freed,
        free_failed,
        pool_used_blocks = pool_before.map_or(-1i64, |p| p.used_blocks as i64),
        pool_total_blocks = pool_before.map_or(-1i64, |p| p.total_blocks as i64),
        arena_used_gib = gib(arena_used),
        meminfo_avail_gib_before = mem_before.map_or(-1.0, |m| gib(m.available_for_alloc)),
        meminfo_avail_gib_after = mem_after.map_or(-1.0, |m| gib(m.available_for_alloc)),
        cuda_free_gib_before = cuda_before.map_or(-1.0, |(f, _)| gib(f)),
        cuda_free_gib_after = cuda_after.map_or(-1.0, |(f, _)| gib(f)),
        reclaimed_gib = reclaimed_gb.unwrap_or(-1.0),
        duration_ms,
        "hot-swap unload release accounting"
    );
    Ok(json!({
        "state": "unloaded",
        "freed_allocs": freed,
        "free_failed": free_failed,
        "duration_ms": duration_ms,
        "arena": {
            "base": format!("{:#x}", hotswap.arena.base().0),
            "capacity_bytes": hotswap.arena.capacity(),
            "used_bytes": hotswap.arena.used(),
        },
        "released": {
            "arena_used_bytes_before": arena_used,
            "pool_used_blocks_before": pool_before.map(|p| p.used_blocks),
            "pool_total_blocks_before": pool_before.map(|p| p.total_blocks),
            "meminfo_available_gb_before": mem_before.map(|m| gib(m.available_for_alloc)),
            "meminfo_available_gb_after": mem_after.map(|m| gib(m.available_for_alloc)),
            "cuda_free_gb_before": cuda_before.map(|(f, _)| gib(f)),
            "cuda_free_gb_after": cuda_after.map(|(f, _)| gib(f)),
            "reclaimed_gb": reclaimed_gb,
        },
    }))
}

/// Unload the model, answer `reply`, then park until a `Load` command or
/// shutdown. Returns the swap outcome (`Load`, Task 5 / Task 6 Phase 1.3 —
/// possibly a DIFFERENT model with its own scheduler params) or `None` on
/// shutdown.
pub(super) fn unload_and_park(
    model: Box<dyn Model>,
    reply: Reply,
    pending: &Shared,
    hotswap: &Arc<HotswapCtx>,
    session_manager: &mut crate::session_manager::SessionSsmManager,
    reload: &dyn Fn(Option<&str>) -> Result<super::SwapOutcome>,
) -> Option<super::SwapOutcome> {
    // Session snapshots reference SSM pool slots of the model being dropped.
    session_manager.clear();

    let result = unload_model(model, hotswap);
    let ok = result.is_ok();
    let _ = reply.send(result);
    if !ok {
        // Teardown failed mid-way: the model is gone regardless (consumed).
        // Park as unloaded — a Load can still recover the process.
        tracing::error!("Hot-swap: unload finished with errors; parked as unloaded");
    }
    hotswap.set_state(SwapState::NoModel);

    // ── Parked: no model. Fail requests, wait for Load or shutdown. ──
    loop {
        let msgs: Vec<ControlMsg> = {
            let mut g = pending.0.lock();
            while g.requests.is_empty() && g.control.is_empty() && !g.closed {
                pending.1.wait(&mut g);
            }
            for req in g.requests.drain(..) {
                fail_request(
                    req,
                    "no model loaded (state: unloaded); POST /admin/models/load",
                );
            }
            if g.closed && g.control.is_empty() {
                return None;
            }
            g.control.drain(..).collect()
        };
        for msg in msgs {
            match msg {
                ControlMsg::Unload { reply } => {
                    let _ = reply.send(Err(anyhow::anyhow!("already unloaded")));
                }
                ControlMsg::Load { model, reply } => {
                    hotswap.set_state(SwapState::Loading);
                    // One model in memory at a time: a resident sidecar
                    // goes first. Waits for its in-flight requests — they
                    // cannot be cancelled.
                    crate::sidecars::unload_all_blocking();
                    match reload(model.as_deref()) {
                        Ok(mut outcome) => {
                            hotswap.set_state(SwapState::Running);
                            let stats = std::mem::take(&mut outcome.stats);
                            let _ = reply.send(Ok(stats));
                            return Some(outcome);
                        }
                        Err(e) => {
                            hotswap.set_state(SwapState::NoModel);
                            tracing::error!("Hot-swap: load failed: {e:#}");
                            let _ = reply.send(Err(e));
                        }
                    }
                }
            }
        }
    }
}

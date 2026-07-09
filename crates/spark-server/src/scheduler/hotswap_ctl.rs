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

/// Max grace for running sequences after an unload request; afterwards they
/// are force-finished with finish_reason "model_swap" (plan Task 4).
const DRAIN_GRACE: Duration = Duration::from_millis(2000);

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
    /// In-progress drain: reply channel + force-finish deadline.
    draining: Option<(Reply, Instant)>,
}

impl SwapCtl {
    pub fn new(hotswap: Option<Arc<HotswapCtx>>, model_ready: Arc<AtomicBool>) -> Self {
        Self {
            hotswap,
            model_ready,
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
                    self.draining = Some((reply, Instant::now() + DRAIN_GRACE));
                }
                ControlMsg::Load { reply } => {
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
        let (reply, _) = self.draining.take().expect("draining checked above");
        Some(reply)
    }
}

/// Tear down the model's device allocations, drop it, and reset the arena.
/// Consumes the model; runs on the scheduler thread.
fn unload_model(model: Box<dyn Model>, hotswap: &HotswapCtx) -> Result<serde_json::Value> {
    let t0 = Instant::now();
    let (freed, free_failed) = {
        let gpu = model
            .gpu_backend()
            .context("model exposes no GPU backend — cannot tear down device allocations")?;
        gpu.device_synchronize()?;
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
        (freed, failed)
    };
    // Drops the WeightStore index views and pinned host staging; all device
    // memory was released via the registry above. Weights live on in the
    // arena mapping, which is never returned to the kernel.
    drop(model);
    hotswap.arena.reset();
    let duration_ms = t0.elapsed().as_millis() as u64;
    tracing::info!(
        "Hot-swap: model unloaded ({freed} device allocations freed, {free_failed} failed, \
         {duration_ms} ms); arena reset (base {} unchanged)",
        hotswap.arena.base()
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
    }))
}

/// Unload the model, answer `reply`, then park until a `Load` command or
/// shutdown. Returns the new model (`Load`, Task 5) or `None` on shutdown.
pub(super) fn unload_and_park(
    model: Box<dyn Model>,
    reply: Reply,
    pending: &Shared,
    hotswap: &Arc<HotswapCtx>,
    session_manager: &mut crate::session_manager::SessionSsmManager,
    reload: &dyn Fn() -> Result<(Box<dyn Model>, serde_json::Value)>,
) -> Option<Box<dyn Model>> {
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
                fail_request(req, "no model loaded (state: unloaded); POST /admin/models/load");
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
                ControlMsg::Load { reply } => {
                    hotswap.set_state(SwapState::Loading);
                    match reload() {
                        Ok((new_model, stats)) => {
                            hotswap.set_state(SwapState::Running);
                            let _ = reply.send(Ok(stats));
                            return Some(new_model);
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

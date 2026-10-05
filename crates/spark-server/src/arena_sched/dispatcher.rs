// SPDX-License-Identifier: AGPL-3.0-only

//! The arena dispatcher (requirements §7/§10): a Tokio task that owns
//! the swap decision and its execution, replacing the synchronous
//! first-request-pays lazy swap.
//!
//! Flow per request (SCH-2/SCH-7): the gate in `api::lazy_swap` passes
//! resident-model requests straight through (λ noted), queues a ticket
//! for everything else, and pokes this task. The task evaluates the pure
//! decision every `scheduler_interval` AND on every enqueue (SCH-4 via
//! `decision::decide`), performs a decided swap through the existing
//! `ControlMsg` drain/unload/load machinery (serialised with the admin
//! endpoints by the same `admin_lock`), and then grants the target's
//! tickets — priority tier first (SCH-6). A failed load restores the
//! previous model and rejects the target's waiters (MOD-10). Overdue
//! waiters expire per PRI-4/WAIT-4.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::decision::{ActiveStats, CandidateStats, Decision, DecisionConfig, decide};
use super::queues::ArenaQueues;
use crate::main_modules::hotswap::{HotswapCtx, SwapState};

pub struct DispatcherConfig {
    pub decision: DecisionConfig,
    pub interval: Duration,
    pub priority_max_wait: Duration,
    pub nonstream_max_wait: Duration,
}

pub struct ArenaDispatcher {
    pub queues: Arc<ArenaQueues>,
    cfg: DispatcherConfig,
    hotswap: Arc<HotswapCtx>,
    submit_notify: tokio::sync::Notify,
    swapping: AtomicBool,
}

impl ArenaDispatcher {
    pub fn spawn(
        queues: Arc<ArenaQueues>,
        cfg: DispatcherConfig,
        hotswap: Arc<HotswapCtx>,
    ) -> Arc<ArenaDispatcher> {
        let d = Arc::new(ArenaDispatcher {
            queues,
            cfg,
            hotswap,
            submit_notify: tokio::sync::Notify::new(),
            swapping: AtomicBool::new(false),
        });
        let task = Arc::clone(&d);
        tokio::spawn(async move { task.run().await });
        d
    }

    /// Gate-side: wake the evaluation after an enqueue (SCH-4).
    pub fn poke(&self) {
        self.submit_notify.notify_one();
    }

    /// True while a drain/unload/load cycle is in flight (SCH-7: requests
    /// for the OLD model queue instead of fast-pathing during the drain).
    pub fn swap_in_progress(&self) -> bool {
        self.swapping.load(Ordering::Acquire)
    }

    async fn run(self: Arc<Self>) {
        loop {
            tokio::select! {
                _ = self.submit_notify.notified() => {}
                _ = tokio::time::sleep(self.cfg.interval) => {}
            }

            let loaded = self.loaded_model();
            self.queues.expire_sweep(
                loaded.as_deref(),
                self.cfg.priority_max_wait,
                self.cfg.nonstream_max_wait,
            );

            // Grant stragglers for the resident model (e.g. tickets queued
            // during a drain for the model that then stayed/returned).
            if let Some(ref m) = loaded {
                let (p, n) = self.queues.grant_all(m);
                if p + n > 0 {
                    tracing::debug!("arena-dispatcher: granted {p}+{n} waiters of resident '{m}'");
                }
            }

            match self.evaluate(loaded.as_deref()) {
                Decision::Stay => {}
                Decision::Swap {
                    target,
                    trigger,
                    pressure,
                    threshold,
                } => {
                    self.perform_swap(
                        loaded.as_deref(),
                        &target,
                        trigger.as_str(),
                        pressure,
                        threshold,
                    )
                    .await;
                }
            }
        }
    }

    fn loaded_model(&self) -> Option<String> {
        match self.hotswap.state() {
            SwapState::Running => Some(self.hotswap.active_model().model_id),
            _ => None,
        }
    }

    /// Snapshot the statistics and run the pure decision.
    fn evaluate(&self, loaded: Option<&str>) -> Decision {
        if self.swapping.load(Ordering::Acquire) {
            return Decision::Stay;
        }
        // Only decide in steady states; Draining/Loading settle first.
        match self.hotswap.state() {
            SwapState::Running | SwapState::NoModel => {}
            _ => return Decision::Stay,
        }

        let snapshots = self.queues.snapshot_all();
        let active_snap = loaded
            .and_then(|m| snapshots.iter().find(|(id, _)| id == m))
            .map(|(_, s)| s.clone())
            .unwrap_or_default();
        let active = ActiveStats {
            n_active: crate::scheduler::scheduler_load(),
            n_queued: active_snap.priority_len + active_snap.normal_len,
            lambda: active_snap.lambda,
            resident_secs: match loaded {
                Some(_) => self.hotswap.active_since.lock().elapsed().as_secs_f64(),
                None => f64::MAX / 4.0, // empty arena: hysteresis never blocks
            },
        };

        // Pending persist volume raises every candidate's T_swap (§6).
        let persist_gib =
            crate::kv_tier::with_active(|t| t.stats().1).unwrap_or(0) as f64 / (1u64 << 30) as f64;

        let candidates: Vec<CandidateStats> = snapshots
            .iter()
            .filter(|(id, s)| Some(id.as_str()) != loaded && s.normal_len > 0)
            .filter(|(id, _)| self.hotswap.registered(id).is_some())
            .map(|(id, s)| {
                let est = crate::swap_costs::estimate(id, self.snapshot_bytes(id));
                CandidateStats {
                    model_id: id.clone(),
                    normal_queued: s.normal_len,
                    pressure_secs: s.normal_pressure_secs,
                    oldest_normal_wait_secs: s.oldest_normal_wait_secs,
                    t_swap_secs: est.total(persist_gib).as_secs_f64(),
                }
            })
            .collect();

        decide(&active, &candidates, &self.cfg.decision)
    }

    /// WAIT-3: coarse queue-wait estimate for `target` — the hysteresis
    /// remainder of the resident model plus the target's T_swap. Zero
    /// when the target is resident and no swap is in flight.
    pub fn estimate_wait_secs(&self, target: &str) -> f64 {
        let loaded = self.loaded_model();
        if loaded.as_deref() == Some(target) && !self.swap_in_progress() {
            return 0.0;
        }
        let est = crate::swap_costs::estimate(target, self.snapshot_bytes(target));
        let t_swap = est.total(0.0).as_secs_f64();
        let residency_hold = match loaded {
            Some(_) => {
                let resident = self.hotswap.active_since.lock().elapsed().as_secs_f64();
                (self.cfg.decision.min_residency_factor * t_swap - resident).max(0.0)
            }
            None => 0.0,
        };
        residency_hold + t_swap
    }

    fn snapshot_bytes(&self, model_id: &str) -> Option<u64> {
        let reg = self.hotswap.registered(model_id)?;
        std::fs::metadata(self.hotswap.snapshot_path_for(model_id, &reg.model_hash))
            .ok()
            .map(|m| m.len())
    }

    /// §10 swap sequence via the existing drain/unload/load machinery.
    async fn perform_swap(
        &self,
        from: Option<&str>,
        target: &str,
        trigger: &str,
        pressure: f64,
        threshold: f64,
    ) {
        self.swapping.store(true, Ordering::Release);
        let residency = self.hotswap.active_since.lock().elapsed();
        let t0 = Instant::now();
        tracing::info!(
            target: "hotswap",
            event = "swap_decision",
            from = from.unwrap_or("<none>"),
            to = target,
            trigger,
            pressure_s = pressure,
            threshold_s = threshold,
            queued = self.queues.queued_for(target),
            "arena-dispatcher: swap decided"
        );
        let result = crate::api::lazy_swap::swap_to(&self.hotswap, target).await;
        let swap_ms = t0.elapsed().as_millis() as u64;
        match result {
            Ok(()) => {
                let (p, n) = self.queues.grant_all(target);
                // TEL-4: the structured swap event — κ/hysteresis tuning
                // reads these fields.
                tracing::info!(
                    target: "hotswap",
                    event = "swap_complete",
                    from = from.unwrap_or("<none>"),
                    to = target,
                    trigger,
                    pressure_s = pressure,
                    threshold_s = threshold,
                    duration_ms = swap_ms,
                    residency_s = residency.as_secs_f64(),
                    granted_priority = p,
                    granted_normal = n,
                    "arena-dispatcher: swap complete"
                );
            }
            Err(_) => {
                // MOD-10: restore the previous model; the target's waiters
                // get a 503 via their rejected tickets.
                let rejected = self.queues.reject_all(target);
                tracing::error!(
                    target: "hotswap",
                    event = "swap_failed",
                    from = from.unwrap_or("<none>"),
                    to = target,
                    trigger,
                    rejected,
                    "arena-dispatcher: load failed — restoring previous model"
                );
                if let Some(prev) = from
                    && crate::api::lazy_swap::swap_to(&self.hotswap, prev)
                        .await
                        .is_err()
                {
                    tracing::error!(
                        "arena-dispatcher: restoring previous model '{prev}' ALSO failed"
                    );
                }
                if let Some(m) = self.loaded_model() {
                    self.queues.grant_all(&m);
                }
            }
        }
        self.swapping.store(false, Ordering::Release);
    }
}

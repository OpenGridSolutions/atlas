// SPDX-License-Identifier: AGPL-3.0-only

//! Periodic anomaly detectors (design doc §6) for conditions no single
//! log line expresses: a scheduler that stopped stepping, a swap that
//! never ends, a 5xx or rejection rate, watchdog storms, NVMe write-queue
//! rejections. Each tick reads counters/gauges, compares with the last
//! tick, and emits events; the observer's throttle keeps a persisting
//! condition from spamming.

use std::sync::Arc;
use std::time::{Duration, Instant};

use super::Event;
use crate::main_modules::AppState;

#[derive(Clone, Debug)]
pub struct DetectorConfig {
    pub interval: Duration,
    /// Seconds without a scheduler loop iteration while work is active.
    pub stall_secs: f64,
    /// A swap in flight longer than this is stuck.
    pub swap_stuck_secs: f64,
    /// 5xx responses per minute that count as an incident.
    pub http_5xx_per_min: u64,
    /// Admission rejections per minute (with an idle GPU) that mean the
    /// estimate itself is broken.
    pub rejections_per_min: u64,
    /// Watchdog stops per minute worth a note.
    pub watchdog_stops_per_min: u64,
}

fn counter_sum(c: &prometheus::IntCounterVec) -> u64 {
    use prometheus::core::Collector;
    c.collect()
        .iter()
        .flat_map(|mf| {
            mf.get_metric()
                .iter()
                .map(|m| m.get_counter().get_value() as u64)
        })
        .sum()
}

/// Pure per-tick evaluation over a small sample so the thresholds are
/// unit-testable without a server.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sample {
    pub secs_since_last_step: f64,
    pub scheduler_load: usize,
    pub swapping_for_secs: f64,
    /// Counter deltas over the sliding `elapsed_secs` window (≈ 60 s).
    pub http_5xx_delta: u64,
    pub rejections_delta: u64,
    pub admissions_delta: u64,
    pub watchdog_delta: u64,
    pub nvme_rejections_delta: u64,
    pub elapsed_secs: f64,
}

/// Sliding window of per-tick counter deltas (6 ticks × 10 s = 60 s).
#[derive(Default)]
struct Window {
    deltas: std::collections::VecDeque<[u64; 5]>,
    last: Option<[u64; 5]>,
}

impl Window {
    const LEN: usize = 6;
    /// Feed the current absolute counters; returns the window sums.
    fn push(&mut self, cur: [u64; 5]) -> [u64; 5] {
        if let Some(prev) = self.last {
            let d: [u64; 5] = std::array::from_fn(|i| cur[i].saturating_sub(prev[i]));
            self.deltas.push_back(d);
            if self.deltas.len() > Self::LEN {
                self.deltas.pop_front();
            }
        }
        self.last = Some(cur);
        let mut sum = [0u64; 5];
        for d in &self.deltas {
            for i in 0..5 {
                sum[i] += d[i];
            }
        }
        sum
    }
    fn secs(&self, tick: Duration) -> f64 {
        self.deltas.len().max(1) as f64 * tick.as_secs_f64()
    }
}

pub fn evaluate(s: &Sample, cfg: &DetectorConfig) -> Vec<Event> {
    let mut out = Vec::new();
    let per_min = |d: u64| d as f64 * 60.0 / s.elapsed_secs.max(1.0);
    if s.scheduler_load > 0 && s.secs_since_last_step > cfg.stall_secs {
        out.push(
            Event::new(
                5,
                "scheduler_stall",
                format!(
                    "scheduler loop has not iterated for {:.0}s with {} active sequences",
                    s.secs_since_last_step, s.scheduler_load
                ),
            )
            .detail(
                serde_json::json!({"since_s": s.secs_since_last_step, "load": s.scheduler_load}),
            ),
        );
    }
    if s.swapping_for_secs > cfg.swap_stuck_secs {
        out.push(Event::new(
            5,
            "swap_stuck",
            format!("arena swap in flight for {:.0}s", s.swapping_for_secs),
        ));
    }
    if s.http_5xx_delta >= cfg.http_5xx_per_min.max(1)
        && per_min(s.http_5xx_delta) >= cfg.http_5xx_per_min as f64
    {
        out.push(
            Event::new(
                4,
                "http_5xx_rate",
                format!(
                    "{} 5xx responses in {:.0}s",
                    s.http_5xx_delta, s.elapsed_secs
                ),
            )
            .detail(serde_json::json!({"count": s.http_5xx_delta, "window_s": s.elapsed_secs})),
        );
    }
    if s.scheduler_load == 0
        && s.admissions_delta == 0
        && s.rejections_delta >= cfg.rejections_per_min
        && per_min(s.rejections_delta) >= cfg.rejections_per_min as f64
    {
        out.push(
            Event::new(
                5,
                "estimate_broken",
                format!(
                    "{} admission rejections and no admission in {:.0}s while the GPU is \
                     idle — the estimate refuses everything",
                    s.rejections_delta, s.elapsed_secs
                ),
            )
            .detail(serde_json::json!({"rejections": s.rejections_delta})),
        );
    }
    if s.watchdog_delta > 0 && per_min(s.watchdog_delta) >= cfg.watchdog_stops_per_min as f64 {
        out.push(Event::new(
            2,
            "watchdog_stops",
            format!(
                "{} watchdog stops in {:.0}s",
                s.watchdog_delta, s.elapsed_secs
            ),
        ));
    }
    if s.nvme_rejections_delta > 0 {
        out.push(Event::new(
            3,
            "nvme_write_queue_full",
            format!(
                "{} blocks rejected by the NVMe write queue in {:.0}s",
                s.nvme_rejections_delta, s.elapsed_secs
            ),
        ));
    }
    out
}

pub fn spawn(state: Arc<AppState>, cfg: DetectorConfig) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(cfg.interval);
        let mut window = Window::default();
        let mut swapping_since: Option<Instant> = None;
        loop {
            tick.tick().await;
            let now = Instant::now();
            let swapping = state.arena.as_ref().is_some_and(|d| d.swap_in_progress());
            swapping_since = match (swapping, swapping_since) {
                (true, None) => Some(now),
                (true, Some(t)) => Some(t),
                (false, _) => None,
            };
            let sums = window.push([
                counter_sum(&crate::metrics::REQUEST_FAILURES_TOTAL),
                counter_sum(&crate::metrics::ADMISSION_REJECTED_TOTAL),
                counter_sum(&crate::metrics::ADMISSION_ACCEPTED_TOTAL),
                counter_sum(&crate::metrics::WATCHDOG_STOPS_TOTAL),
                nvme_rejections(),
            ]);
            let sample = Sample {
                secs_since_last_step: crate::scheduler::secs_since_last_step(),
                scheduler_load: crate::scheduler::scheduler_load(),
                swapping_for_secs: swapping_since
                    .map(|t| now.duration_since(t).as_secs_f64())
                    .unwrap_or(0.0),
                http_5xx_delta: sums[0],
                rejections_delta: sums[1],
                admissions_delta: sums[2],
                watchdog_delta: sums[3],
                nvme_rejections_delta: sums[4],
                elapsed_secs: window.secs(cfg.interval),
            };
            for ev in evaluate(&sample, &cfg) {
                super::emit(ev);
            }
        }
    });
}

fn nvme_rejections() -> u64 {
    crate::kv_tier::with_active(|t| t.stats().2).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> DetectorConfig {
        DetectorConfig {
            interval: Duration::from_secs(10),
            stall_secs: 60.0,
            swap_stuck_secs: 600.0,
            http_5xx_per_min: 10,
            rejections_per_min: 10,
            watchdog_stops_per_min: 20,
        }
    }

    fn kinds(evs: &[Event]) -> Vec<&str> {
        evs.iter().map(|e| e.kind.as_str()).collect()
    }

    #[test]
    fn quiet_sample_emits_nothing() {
        let s = Sample {
            elapsed_secs: 10.0,
            ..Default::default()
        };
        assert!(evaluate(&s, &cfg()).is_empty());
        // Idle scheduler with an old last step is not a stall.
        let idle = Sample {
            secs_since_last_step: 1000.0,
            scheduler_load: 0,
            elapsed_secs: 10.0,
            ..Default::default()
        };
        assert!(evaluate(&idle, &cfg()).is_empty());
    }

    #[test]
    fn thresholds() {
        let s = Sample {
            secs_since_last_step: 61.0,
            scheduler_load: 2,
            swapping_for_secs: 601.0,
            http_5xx_delta: 12, // ≥ 10 absolute and ≥ 10/min over 60 s
            rejections_delta: 0,
            admissions_delta: 0,
            watchdog_delta: 24,
            nvme_rejections_delta: 1,
            elapsed_secs: 60.0,
        };
        let evs = evaluate(&s, &cfg());
        let k = kinds(&evs);
        assert_eq!(
            k,
            vec![
                "scheduler_stall",
                "swap_stuck",
                "http_5xx_rate",
                "watchdog_stops",
                "nvme_write_queue_full"
            ]
        );
        // A couple of 5xx or 503 back-pressure never trips the rate.
        let few = Sample {
            http_5xx_delta: 2,
            elapsed_secs: 10.0,
            ..Default::default()
        };
        assert!(kinds(&evaluate(&few, &cfg())).is_empty());
        // Rejections only mean "estimate broken" with an idle GPU AND no
        // admission at all in the window (absurd budgets alone don't).
        let busy = Sample {
            scheduler_load: 1,
            rejections_delta: 12,
            elapsed_secs: 60.0,
            ..Default::default()
        };
        assert!(kinds(&evaluate(&busy, &cfg())).is_empty());
        let mixed = Sample {
            scheduler_load: 0,
            rejections_delta: 12,
            admissions_delta: 1,
            elapsed_secs: 60.0,
            ..Default::default()
        };
        assert!(kinds(&evaluate(&mixed, &cfg())).is_empty());
        let broken = Sample {
            scheduler_load: 0,
            rejections_delta: 12,
            admissions_delta: 0,
            elapsed_secs: 60.0,
            ..Default::default()
        };
        assert_eq!(kinds(&evaluate(&broken, &cfg())), vec!["estimate_broken"]);
        // Window bookkeeping.
        let mut w = Window::default();
        assert_eq!(w.push([0; 5]), [0; 5]);
        assert_eq!(w.push([5, 1, 0, 0, 0]), [5, 1, 0, 0, 0]);
        for _ in 0..7 {
            w.push([5, 1, 0, 0, 0]);
        }
        assert_eq!(
            w.push([6, 1, 0, 0, 0])[0],
            1,
            "old deltas fell out of the window"
        );
    }
}

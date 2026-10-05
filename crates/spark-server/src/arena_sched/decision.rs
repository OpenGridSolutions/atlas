// SPDX-License-Identifier: AGPL-3.0-only

//! The arena-swap decision (requirements §7) as a PURE function (SCH-5):
//! inputs are a snapshot of queue statistics, swap costs and the
//! configuration; the output is `Stay` or `Swap { target }`. No clocks,
//! no I/O, no globals — recorded load profiles replay through it in
//! unit tests.
//!
//! Pressure model: a switch to B happens once the accumulated waiting
//! time of B's NORMAL requests exceeds the waiting time the switch
//! inflicts on A's users:
//!
//! ```text
//! D_B = Σ wait_r                         (normal queue of B)
//! S_B = κ · T_swap · (1 + N_active,A + N_queued,A + λ_A · T_swap)
//! ```
//!
//! Decision order (SCH-4, binding): candidate by highest D/S → starvation
//! → idle → hysteresis → pressure → stay. Starvation outranks hysteresis.
//! Priority requests never appear in D or the starvation clock (PRI-2);
//! they DO count into N_active/N_queued of the loaded model (PRI-3).

/// Statistics of one NOT-loaded model's queue, as of the decision instant.
#[derive(Clone, Debug)]
pub struct CandidateStats {
    pub model_id: String,
    /// Normal-tier queue length (priority excluded — PRI-2).
    pub normal_queued: usize,
    /// Σ wait of the normal tier (seconds) — `D_B`.
    pub pressure_secs: f64,
    /// Oldest NORMAL request's wait (seconds).
    pub oldest_normal_wait_secs: f64,
    /// `T_swap(A→B)` estimate in seconds (persist + unload + load + warmup).
    pub t_swap_secs: f64,
}

/// Statistics of the loaded model.
#[derive(Clone, Debug)]
pub struct ActiveStats {
    /// Current batch size (BAT-11), priority requests included (PRI-3).
    pub n_active: usize,
    /// Queued requests FOR the loaded model, priority included (PRI-3).
    pub n_queued: usize,
    /// Arrival rate of the loaded model (EWMA, req/s).
    pub lambda: f64,
    /// How long the model has been resident (seconds).
    pub resident_secs: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct DecisionConfig {
    /// Tuning factor of the switch threshold (κ = 1 ⇒ break-even).
    pub kappa: f64,
    /// Minimum residency in multiples of `T_swap` (hysteresis).
    pub min_residency_factor: f64,
    /// Starvation bound for normal requests (seconds).
    pub max_wait_hard_secs: f64,
    /// Expected A-arrivals during the swap below which A counts as idle.
    pub idle_epsilon: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwapTrigger {
    Pressure,
    Starvation,
    Idle,
}

impl SwapTrigger {
    pub fn as_str(&self) -> &'static str {
        match self {
            SwapTrigger::Pressure => "pressure",
            SwapTrigger::Starvation => "starvation",
            SwapTrigger::Idle => "idle",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Decision {
    Stay,
    Swap {
        target: String,
        trigger: SwapTrigger,
        /// D and S of the winning candidate at decision time (TEL-4).
        pressure: f64,
        threshold: f64,
    },
}

/// Switch threshold `S_B` for one candidate.
pub fn threshold(active: &ActiveStats, t_swap: f64, kappa: f64) -> f64 {
    let t = t_swap.max(0.001);
    kappa.max(0.0)
        * t
        * (1.0 + active.n_active as f64 + active.n_queued as f64 + active.lambda.max(0.0) * t)
}

/// The decision (§7, verbindliche Reihenfolge).
pub fn decide(
    active: &ActiveStats,
    candidates: &[CandidateStats],
    cfg: &DecisionConfig,
) -> Decision {
    // 1. Candidate = highest D/S among models with ≥1 normal waiter —
    //    the RATIO prefers cheaply loadable models over absolute pressure.
    let mut best: Option<(&CandidateStats, f64, f64)> = None; // (cand, ratio, s)
    for c in candidates {
        if c.normal_queued == 0 {
            continue;
        }
        let s = threshold(active, c.t_swap_secs, cfg.kappa);
        let ratio = if s > 0.0 {
            c.pressure_secs / s
        } else {
            f64::INFINITY
        };
        if best.as_ref().is_none_or(|(_, r, _)| ratio > *r) {
            best = Some((c, ratio, s));
        }
    }
    let Some((cand, ratio, s)) = best else {
        return Decision::Stay;
    };

    // 2. Starvation beats everything, including the hysteresis (SCH-4).
    if cand.oldest_normal_wait_secs > cfg.max_wait_hard_secs {
        return Decision::Swap {
            target: cand.model_id.clone(),
            trigger: SwapTrigger::Starvation,
            pressure: cand.pressure_secs,
            threshold: s,
        };
    }

    // 3. Idle: nothing running, nothing queued for A, and A-traffic during
    //    the swap is expected to be negligible.
    if active.n_active == 0
        && active.n_queued == 0
        && active.lambda.max(0.0) * cand.t_swap_secs < cfg.idle_epsilon
    {
        return Decision::Swap {
            target: cand.model_id.clone(),
            trigger: SwapTrigger::Idle,
            pressure: cand.pressure_secs,
            threshold: s,
        };
    }

    // 4. Hysteresis: A must stay resident for min_residency_factor × T_swap.
    if active.resident_secs < cfg.min_residency_factor * cand.t_swap_secs {
        return Decision::Stay;
    }

    // 5. Pressure break-even.
    if ratio >= 1.0 {
        return Decision::Swap {
            target: cand.model_id.clone(),
            trigger: SwapTrigger::Pressure,
            pressure: cand.pressure_secs,
            threshold: s,
        };
    }

    Decision::Stay
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> DecisionConfig {
        DecisionConfig {
            kappa: 1.0,
            min_residency_factor: 2.0,
            max_wait_hard_secs: 60.0,
            idle_epsilon: 0.1,
        }
    }

    fn active(n_active: usize, n_queued: usize, lambda: f64, resident: f64) -> ActiveStats {
        ActiveStats {
            n_active,
            n_queued,
            lambda,
            resident_secs: resident,
        }
    }

    fn cand(id: &str, n: usize, d: f64, oldest: f64, t_swap: f64) -> CandidateStats {
        CandidateStats {
            model_id: id.into(),
            normal_queued: n,
            pressure_secs: d,
            oldest_normal_wait_secs: oldest,
            t_swap_secs: t_swap,
        }
    }

    fn target(d: &Decision) -> Option<(&str, SwapTrigger)> {
        match d {
            Decision::Stay => None,
            Decision::Swap {
                target, trigger, ..
            } => Some((target.as_str(), *trigger)),
        }
    }

    #[test]
    fn no_normal_waiters_means_stay() {
        // Priority-only queues never trigger a switch (PRI-2): the
        // candidate list simply has normal_queued == 0.
        let d = decide(
            &active(0, 0, 0.0, 1000.0),
            &[cand("b", 0, 0.0, 0.0, 6.0)],
            &cfg(),
        );
        assert_eq!(d, Decision::Stay);
    }

    #[test]
    fn idle_arena_switches_immediately() {
        let d = decide(
            &active(0, 0, 0.0, 0.5), // freshly resident — idle beats hysteresis
            &[cand("b", 1, 0.1, 0.1, 6.0)],
            &cfg(),
        );
        assert_eq!(target(&d), Some(("b", SwapTrigger::Idle)));
    }

    #[test]
    fn idle_requires_low_expected_arrivals() {
        // λ_A · T_swap = 0.5 ≥ idle_epsilon → not idle; hysteresis holds.
        let d = decide(
            &active(0, 0, 0.0833, 1.0),
            &[cand("b", 1, 0.1, 0.1, 6.0)],
            &cfg(),
        );
        assert_eq!(d, Decision::Stay);
    }

    #[test]
    fn pressure_break_even_at_kappa_one() {
        // A: 3 active, 1 queued, λ=0.5/s; T_swap=6 →
        // S = 6·(1+3+1+3) = 48. D=47.9 → stay, D=48.1 → swap.
        let a = active(3, 1, 0.5, 1000.0);
        let below = decide(&a, &[cand("b", 4, 47.9, 20.0, 6.0)], &cfg());
        assert_eq!(below, Decision::Stay);
        let above = decide(&a, &[cand("b", 4, 48.1, 20.0, 6.0)], &cfg());
        assert_eq!(target(&above), Some(("b", SwapTrigger::Pressure)));
    }

    #[test]
    fn hysteresis_blocks_fresh_residency() {
        // Residency 10 s < 2 × 6 s → stay despite D >> S.
        let d = decide(
            &active(1, 0, 0.1, 10.0),
            &[cand("b", 8, 500.0, 30.0, 6.0)],
            &cfg(),
        );
        assert_eq!(d, Decision::Stay);
    }

    #[test]
    fn starvation_overrides_hysteresis() {
        // Same fresh residency, but the oldest normal waiter is past
        // max_wait_hard → swap regardless (SCH-4).
        let d = decide(
            &active(1, 0, 0.1, 10.0),
            &[cand("b", 8, 500.0, 61.0, 6.0)],
            &cfg(),
        );
        assert_eq!(target(&d), Some(("b", SwapTrigger::Starvation)));
    }

    #[test]
    fn ratio_prefers_cheaply_loadable_model() {
        // "c" has lower absolute pressure but a much cheaper swap —
        // higher D/S ratio wins (§7 step 1).
        let a = active(0, 0, 0.0, 1000.0);
        let heavy = cand("b", 2, 100.0, 50.0, 60.0); // S = 60, ratio 1.67
        let cheap = cand("c", 2, 20.0, 10.0, 2.0); // S = 2,  ratio 10
        let d = decide(&a, &[heavy, cheap], &cfg());
        // (idle fires here — both qualify; target must be the ratio winner)
        assert_eq!(target(&d).map(|t| t.0), Some("c"));
    }

    #[test]
    fn busy_arena_raises_threshold_with_batch_and_arrivals() {
        // PRI-3/BAT-11: active batch + queued (incl. priority) and λ_A
        // make leaving A more expensive.
        let quiet = active(0, 0, 0.0, 1000.0);
        let busy = active(6, 4, 2.0, 1000.0);
        let c = cand("b", 3, 30.0, 20.0, 6.0);
        assert_eq!(
            target(&decide(&quiet, std::slice::from_ref(&c), &cfg())).map(|t| t.1),
            Some(SwapTrigger::Idle)
        );
        // busy: S = 6·(1+6+4+12) = 138 → ratio ~0.22 → stay.
        assert_eq!(decide(&busy, &[c], &cfg()), Decision::Stay);
    }

    #[test]
    fn kappa_scales_the_threshold() {
        let a = active(3, 1, 0.5, 1000.0);
        let c = cand("b", 4, 30.0, 20.0, 6.0); // S(κ=1)=48 → stay
        assert_eq!(decide(&a, std::slice::from_ref(&c), &cfg()), Decision::Stay);
        let mut lenient = cfg();
        lenient.kappa = 0.5; // S=24 → ratio 1.25 → swap
        assert_eq!(
            target(&decide(&a, &[c], &lenient)).map(|t| t.1),
            Some(SwapTrigger::Pressure)
        );
    }

    #[test]
    fn replayed_ping_pong_profile_stays_put() {
        // Recorded-profile style replay (SCH-5): alternating single
        // requests for A and B with a 6 s swap must NOT ping-pong —
        // after a swap to A (resident 1 s), one B request with 2 s of
        // accumulated wait stays queued.
        let mut resident = 1.0f64;
        let mut swaps = 0;
        for tick in 0..50 {
            let d_b = 2.0 + (tick % 3) as f64; // small oscillating pressure
            let d = decide(
                &active(1, 0, 0.5, resident),
                &[cand("b", 1, d_b, d_b, 6.0)],
                &cfg(),
            );
            if matches!(d, Decision::Swap { .. }) {
                swaps += 1;
                resident = 0.0;
            }
            resident += 0.2;
        }
        assert_eq!(swaps, 0, "small oscillating pressure must not ping-pong");
    }

    #[test]
    fn replayed_burst_profile_switches_once_pressure_accumulates() {
        // B's queue grows steadily while A drains to idle; the decision
        // must flip exactly once (idle or pressure), not oscillate.
        let mut swaps = 0;
        let mut last_was_swap = false;
        for tick in 0..100 {
            let d_b = tick as f64 * 1.5; // growing pressure
            let a = if tick < 40 {
                active(2, 1, 0.5, 500.0)
            } else {
                active(0, 0, 0.01, 500.0 + tick as f64)
            };
            let d = decide(&a, &[cand("b", 3, d_b, d_b / 3.0, 6.0)], &cfg());
            let is_swap = matches!(d, Decision::Swap { .. });
            if is_swap && !last_was_swap {
                swaps += 1;
            }
            last_was_swap = is_swap;
        }
        assert_eq!(swaps, 1, "one coherent switch, no flapping");
    }
}

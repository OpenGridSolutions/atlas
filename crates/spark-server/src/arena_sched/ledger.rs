// SPDX-License-Identifier: AGPL-3.0-only

//! Capacity ledger (docs/design/deadline-admission.md §Frage 4).
//!
//! Admitted-but-unfinished work, so that two simultaneous requests
//! cannot both be accepted when there is room for one: the estimate is
//! computed and the reservation recorded under ONE lock ([`admit`]).
//! A [`Reservation`] releases on drop (success, error, client abort);
//! as a backstop every entry also expires at `started + secs × grace`
//! so a streaming body that outlives its handler cannot pin capacity.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::estimate::{Estimate, EstimateError, EstimateProfile, Snapshot, WorkProfile, estimate};

#[derive(Clone, Debug)]
struct Entry {
    model: String,
    /// Single-sequence seconds this request was estimated to occupy.
    secs: f64,
    started: Instant,
    expires: Instant,
}

#[derive(Default)]
struct State {
    entries: HashMap<u64, Entry>,
    next_id: u64,
}

pub struct CapacityLedger {
    inner: Mutex<State>,
    /// Multiplier on the estimate before comparing with the budget.
    safety_factor: f64,
    /// Entries self-expire after `secs × grace` (+ 5 s) as a backstop.
    expiry_grace: f64,
}

/// Outcome of a successful admission. Dropping it releases the
/// reservation.
pub struct Reservation {
    ledger: Arc<CapacityLedger>,
    id: u64,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.ledger.release(self.id);
    }
}

/// Result of [`CapacityLedger::admit`].
#[derive(Debug)]
pub enum Verdict {
    /// Within budget (or no budget given). The estimate is informational.
    Admitted(Estimate),
    /// Estimate × safety factor exceeds the budget.
    Rejected {
        estimate: Estimate,
        budget_secs: f64,
    },
    /// No estimate possible; admitted when no budget was given,
    /// otherwise the caller answers `unavailable`.
    NoEstimate(EstimateError),
}

impl CapacityLedger {
    pub fn new(safety_factor: f64, expiry_grace: f64) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(State::default()),
            safety_factor: safety_factor.max(1.0),
            expiry_grace: expiry_grace.max(1.0),
        })
    }

    pub fn safety_factor(&self) -> f64 {
        self.safety_factor
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Reserved single-sequence seconds still ahead (remaining = secs −
    /// elapsed, floored at 0), for `model` only. Caller holds the lock.
    fn reserved_secs_locked(st: &State, model: &str, now: Instant) -> f64 {
        st.entries
            .values()
            .filter(|e| e.model == model)
            .map(|e| (e.secs - now.duration_since(e.started).as_secs_f64()).max(0.0))
            .sum()
    }

    fn purge_expired(st: &mut State, now: Instant) {
        st.entries.retain(|_, e| e.expires > now);
    }

    /// Estimate + decide + reserve atomically. `snapshot` is filled in
    /// by the caller EXCEPT `reserved_secs`, which this method reads from
    /// its own entries under the lock. Returns the verdict and, when
    /// admitted with an estimate, the reservation guard.
    pub fn admit(
        self: &Arc<Self>,
        model: &str,
        profile: Option<&EstimateProfile>,
        work: &WorkProfile,
        mut snapshot: Snapshot,
        budget_secs: Option<f64>,
        now: Instant,
    ) -> (Verdict, Option<Reservation>) {
        let mut st = self.lock();
        Self::purge_expired(&mut st, now);
        snapshot.reserved_secs = Self::reserved_secs_locked(&st, model, now);
        let est = match estimate(profile, work, &snapshot) {
            Ok(e) => e,
            Err(e) => return (Verdict::NoEstimate(e), None),
        };
        let predicted = est.total_secs() * self.safety_factor;
        if let Some(budget) = budget_secs
            && predicted > budget
        {
            return (
                Verdict::Rejected {
                    estimate: est,
                    budget_secs: budget,
                },
                None,
            );
        }
        let id = st.next_id;
        st.next_id += 1;
        let secs = est.work_secs.max(0.0);
        st.entries.insert(
            id,
            Entry {
                model: model.to_string(),
                secs,
                started: now,
                expires: now
                    + std::time::Duration::from_secs_f64(
                        (est.total_secs() * self.expiry_grace + 5.0).min(86_400.0),
                    ),
            },
        );
        (
            Verdict::Admitted(est),
            Some(Reservation {
                ledger: Arc::clone(self),
                id,
            }),
        )
    }

    fn release(&self, id: u64) {
        self.lock().entries.remove(&id);
    }

    /// Live reservations (for `/internal/state` and tests).
    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remaining reserved seconds per model, for `/internal/state`.
    pub fn state_json(&self, now: Instant) -> serde_json::Value {
        let st = self.lock();
        let mut per_model: std::collections::BTreeMap<String, (usize, f64)> = Default::default();
        for e in st.entries.values() {
            let rem = (e.secs - now.duration_since(e.started).as_secs_f64()).max(0.0);
            let slot = per_model.entry(e.model.clone()).or_default();
            slot.0 += 1;
            slot.1 += rem;
        }
        serde_json::json!({
            "reservations": st.entries.len(),
            "safety_factor": self.safety_factor,
            "per_model": per_model.iter().map(|(m, (n, s))| serde_json::json!({
                "model": m, "count": n, "remaining_secs": s,
            })).collect::<Vec<_>>(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> EstimateProfile {
        EstimateProfile {
            prefill_tok_s: 1000.0,
            decode_step_ms: [10.0, 10.0, 10.0, 10.0],
            ocr_page_secs_p90: 10.0,
            ocr_page_tokens_p90: 0,
        }
    }

    fn snap(max_batch: usize) -> Snapshot {
        Snapshot {
            target_resident: true,
            model_wait_secs: 0.0,
            n_active: 0,
            max_batch,
            reserved_secs: 0.0,
            queued_ahead: 0,
        }
    }

    #[test]
    fn one_slot_two_requests_only_one_fits() {
        let ledger = CapacityLedger::new(1.0, 2.0);
        let now = Instant::now();
        let work = WorkProfile {
            pages: 3,
            ..Default::default()
        }; // 30 s each
        // Batch width 1 and one sequence already active: the first request
        // is admitted (fits 60 s budget), reserves 30 s; the second sees
        // 30 s ahead + 30 s own = 60 s… budget 50 s → rejected.
        let busy = Snapshot {
            n_active: 1,
            ..snap(1)
        };
        let (v1, r1) = ledger.admit("m", Some(&profile()), &work, busy, Some(60.0), now);
        assert!(matches!(v1, Verdict::Admitted(_)), "{v1:?}");
        assert!(r1.is_some());
        let (v2, r2) = ledger.admit("m", Some(&profile()), &work, busy, Some(50.0), now);
        match v2 {
            Verdict::Rejected { estimate, .. } => {
                assert!((estimate.ahead_secs - 30.0).abs() < 1e-9, "{estimate:?}");
            }
            other => panic!("{other:?}"),
        }
        assert!(r2.is_none());
        assert_eq!(ledger.len(), 1);
        drop(r1);
        assert!(ledger.is_empty(), "drop releases");
        let (v3, _r3) = ledger.admit("m", Some(&profile()), &work, busy, Some(50.0), now);
        assert!(matches!(v3, Verdict::Admitted(_)), "after release: {v3:?}");
    }

    #[test]
    fn safety_factor_and_no_budget() {
        let ledger = CapacityLedger::new(1.5, 2.0);
        let now = Instant::now();
        let work = WorkProfile {
            pages: 1,
            ..Default::default()
        }; // 10 s × 1.5 = 15 s predicted
        let (v, _) = ledger.admit("m", Some(&profile()), &work, snap(4), Some(14.0), now);
        assert!(matches!(v, Verdict::Rejected { .. }));
        let (v, r) = ledger.admit("m", Some(&profile()), &work, snap(4), Some(15.0), now);
        assert!(matches!(v, Verdict::Admitted(_)));
        drop(r);
        // Without a budget everything is admitted and still reserved.
        let (v, r) = ledger.admit("m", Some(&profile()), &work, snap(4), None, now);
        assert!(matches!(v, Verdict::Admitted(_)));
        assert_eq!(ledger.len(), 1);
        drop(r);
    }

    #[test]
    fn no_profile_reserves_nothing() {
        let ledger = CapacityLedger::new(1.0, 2.0);
        let (v, r) = ledger.admit(
            "m",
            None,
            &WorkProfile::default(),
            snap(4),
            Some(1.0),
            Instant::now(),
        );
        assert!(matches!(v, Verdict::NoEstimate(EstimateError::NoProfile)));
        assert!(r.is_none());
        assert!(ledger.is_empty());
    }

    #[test]
    fn reservations_expire_and_decay_and_are_per_model() {
        let ledger = CapacityLedger::new(1.0, 1.0);
        let t0 = Instant::now();
        let work = WorkProfile {
            pages: 1,
            ..Default::default()
        }; // 10 s
        let (_, r) = ledger.admit("a", Some(&profile()), &work, snap(1), None, t0);
        let r = r.unwrap();
        // Another model is not delayed by model a's work.
        let busy = Snapshot {
            n_active: 1,
            ..snap(1)
        };
        let (v, rb) = ledger.admit("b", Some(&profile()), &work, busy, None, t0);
        match v {
            Verdict::Admitted(e) => assert_eq!(e.ahead_secs, 0.0),
            other => panic!("{other:?}"),
        }
        drop(rb);
        // 4 s later, 6 s of a's work remain ahead for the next a-request.
        let t1 = t0 + std::time::Duration::from_secs(4);
        let (v, ra) = ledger.admit("a", Some(&profile()), &work, busy, None, t1);
        match v {
            Verdict::Admitted(e) => assert!((e.ahead_secs - 6.0).abs() < 1e-9, "{e:?}"),
            other => panic!("{other:?}"),
        }
        drop(ra);
        // Past expiry (10 s × 1 + 5 s) the entry is purged even if the
        // guard is still alive (leaked streaming body).
        let t2 = t0 + std::time::Duration::from_secs(16);
        let (v, rc) = ledger.admit("a", Some(&profile()), &work, busy, None, t2);
        match v {
            Verdict::Admitted(e) => assert_eq!(e.ahead_secs, 0.0, "{e:?}"),
            other => panic!("{other:?}"),
        }
        drop(rc);
        drop(r);
        assert!(ledger.is_empty());
    }
}

// SPDX-License-Identifier: AGPL-3.0-only

//! Per-arena-model admission queues (SCH-1/2/3, PRI-1) — TICKET based.
//!
//! Handlers cannot tokenize a prompt before its model's tokenizer is
//! active, so what queues here is not a finished `InferenceRequest` but
//! a *ticket*: the handler registers, awaits the grant, and only then
//! proceeds exactly like today (snapshot the runtime, tokenize, submit
//! to the GPU scheduler). The dispatcher grants tickets when their model
//! becomes resident (priority tier first — SCH-6/PRI-1), expires them
//! per PRI-4/WAIT-4, and rejects them when a load fails (MOD-10).
//!
//! A dropped handler future (client disconnected while waiting) drops
//! its ticket Arc; the queue holds Weaks and sweeps them out (DRN-5).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

const WAITING: u8 = 0;
const GRANTED: u8 = 1;
const EXPIRED: u8 = 2;
const REJECTED: u8 = 3;

/// Handler-visible outcome of a wait.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TicketOutcome {
    /// Model is resident — proceed to tokenize + submit.
    Granted,
    /// Wait bound exceeded (PRI-4 / WAIT-4) → 503 + Retry-After.
    Expired,
    /// The model failed to load (MOD-10) → 503 with the failure note.
    Rejected,
}

pub(crate) struct TicketInner {
    enqueued: Instant,
    priority: bool,
    streaming: bool,
    state: AtomicU8,
    notify: tokio::sync::Notify,
}

/// The handler's side of a queued admission (SCH-1 metadata lives on the
/// inner shared with the queue).
pub struct Ticket {
    inner: Arc<TicketInner>,
}

impl Ticket {
    /// Await the dispatcher's verdict.
    pub async fn wait(&self) -> TicketOutcome {
        loop {
            match self.inner.state.load(Ordering::Acquire) {
                GRANTED => return TicketOutcome::Granted,
                EXPIRED => return TicketOutcome::Expired,
                REJECTED => return TicketOutcome::Rejected,
                _ => self.inner.notify.notified().await,
            }
        }
    }

    pub fn waited(&self) -> Duration {
        self.inner.enqueued.elapsed()
    }
}

#[derive(Default)]
struct ModelQueue {
    priority: VecDeque<Weak<TicketInner>>,
    normal: VecDeque<Weak<TicketInner>>,
    /// Arrival rate EWMA (req/s) over ALL arrivals for this model —
    /// queued AND fast-path (λ in §7).
    lambda: f64,
    last_arrival: Option<Instant>,
}

impl ModelQueue {
    fn sweep(&mut self) {
        for tier in [&mut self.priority, &mut self.normal] {
            tier.retain(|w| {
                w.upgrade()
                    .is_some_and(|t| t.state.load(Ordering::Acquire) == WAITING)
            });
        }
    }

    fn len(&self) -> usize {
        self.priority.len() + self.normal.len()
    }

    fn observe_arrival(&mut self, now: Instant, alpha: f64) {
        if let Some(prev) = self.last_arrival {
            let dt = now.duration_since(prev).as_secs_f64().max(1e-3);
            let sample = 1.0 / dt;
            self.lambda = if self.lambda <= 0.0 {
                sample
            } else {
                alpha * sample + (1.0 - alpha) * self.lambda
            };
        }
        self.last_arrival = Some(now);
    }

    /// Decay λ toward zero in silence (idle detection must not act on a
    /// stale burst-era λ). Halves per 10 s without arrivals.
    fn decayed_lambda(&self, now: Instant) -> f64 {
        match self.last_arrival {
            None => 0.0,
            Some(prev) => {
                let silent = now.duration_since(prev).as_secs_f64();
                self.lambda * 0.5f64.powf(silent / 10.0)
            }
        }
    }
}

/// Per-tier snapshot for the decision function + telemetry (SCH-3, PRI-6).
#[derive(Clone, Debug, Default)]
pub struct QueueSnapshot {
    pub priority_len: usize,
    pub normal_len: usize,
    /// Σ wait of the NORMAL tier (seconds) — D (PRI-2 excludes priority).
    pub normal_pressure_secs: f64,
    pub oldest_normal_wait_secs: f64,
    pub oldest_priority_wait_secs: f64,
    /// λ (EWMA, decayed for silence).
    pub lambda: f64,
}

/// Rejection reason for `submit`.
#[derive(Debug)]
pub enum SubmitError {
    /// SCH-8: queue at `max_queue_length` → HTTP 429 + Retry-After.
    QueueFull { retry_after_secs: u64 },
}

pub struct ArenaQueues {
    inner: Mutex<HashMap<String, ModelQueue>>,
    max_queue_length: usize,
    ewma_alpha: f64,
}

impl ArenaQueues {
    pub fn new(max_queue_length: usize, ewma_alpha: f64) -> ArenaQueues {
        ArenaQueues {
            inner: Mutex::new(HashMap::new()),
            max_queue_length: max_queue_length.max(1),
            ewma_alpha: ewma_alpha.clamp(0.01, 1.0),
        }
    }

    /// Register a waiter for `model` (SCH-1; priority tier per PRI-1).
    pub fn submit(
        &self,
        model: &str,
        priority: bool,
        streaming: bool,
    ) -> Result<Ticket, SubmitError> {
        let now = Instant::now();
        let mut g = self.inner.lock().expect("arena queues lock");
        let q = g.entry(model.to_string()).or_default();
        q.sweep();
        if q.len() >= self.max_queue_length {
            return Err(SubmitError::QueueFull {
                retry_after_secs: 5,
            });
        }
        q.observe_arrival(now, self.ewma_alpha);
        let inner = Arc::new(TicketInner {
            enqueued: now,
            priority,
            streaming,
            state: AtomicU8::new(WAITING),
            notify: tokio::sync::Notify::new(),
        });
        let weak = Arc::downgrade(&inner);
        if priority {
            q.priority.push_back(weak);
        } else {
            q.normal.push_back(weak);
        }
        Ok(Ticket { inner })
    }

    /// λ bookkeeping for fast-path arrivals (request hit the resident
    /// model and never queued).
    pub fn note_arrival(&self, model: &str) {
        let now = Instant::now();
        let mut g = self.inner.lock().expect("arena queues lock");
        g.entry(model.to_string())
            .or_default()
            .observe_arrival(now, self.ewma_alpha);
    }

    fn finish_tier(tier: &mut VecDeque<Weak<TicketInner>>, state: u8) -> usize {
        let mut n = 0;
        for w in tier.drain(..) {
            if let Some(t) = w.upgrade()
                && t.state
                    .compare_exchange(WAITING, state, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                t.notify.notify_waiters();
                n += 1;
            }
        }
        n
    }

    /// Grant every waiter of `model` — priority tier first (SCH-6).
    /// Returns (priority, normal) grant counts.
    pub fn grant_all(&self, model: &str) -> (usize, usize) {
        let mut g = self.inner.lock().expect("arena queues lock");
        let Some(q) = g.get_mut(model) else {
            return (0, 0);
        };
        let p = Self::finish_tier(&mut q.priority, GRANTED);
        let n = Self::finish_tier(&mut q.normal, GRANTED);
        (p, n)
    }

    /// Reject every waiter of `model` (load failed, MOD-10).
    pub fn reject_all(&self, model: &str) -> usize {
        let mut g = self.inner.lock().expect("arena queues lock");
        let Some(q) = g.get_mut(model) else { return 0 };
        Self::finish_tier(&mut q.priority, REJECTED) + Self::finish_tier(&mut q.normal, REJECTED)
    }

    /// Expire overdue waiters: PRIORITY tickets for NOT-loaded models past
    /// `priority_max_wait` (PRI-4) and NON-streaming tickets past
    /// `nonstream_max_wait` (WAIT-4). Returns the number expired.
    pub fn expire_sweep(
        &self,
        loaded_model: Option<&str>,
        priority_max_wait: Duration,
        nonstream_max_wait: Duration,
    ) -> usize {
        let now = Instant::now();
        let mut g = self.inner.lock().expect("arena queues lock");
        let mut expired = 0;
        for (model, q) in g.iter_mut() {
            let foreign = loaded_model != Some(model.as_str());
            for tier in [&mut q.priority, &mut q.normal] {
                tier.retain(|w| {
                    let Some(t) = w.upgrade() else { return false };
                    if t.state.load(Ordering::Acquire) != WAITING {
                        return false;
                    }
                    let waited = now.duration_since(t.enqueued);
                    let overdue = (t.priority && foreign && waited > priority_max_wait)
                        || (!t.streaming && waited > nonstream_max_wait);
                    if overdue
                        && t.state
                            .compare_exchange(WAITING, EXPIRED, Ordering::AcqRel, Ordering::Acquire)
                            .is_ok()
                    {
                        t.notify.notify_waiters();
                        expired += 1;
                        return false;
                    }
                    true
                });
            }
        }
        expired
    }

    /// Snapshot of one model's queue (SCH-3, PRI-6).
    pub fn snapshot(&self, model: &str) -> QueueSnapshot {
        let now = Instant::now();
        let mut g = self.inner.lock().expect("arena queues lock");
        match g.get_mut(model) {
            None => QueueSnapshot::default(),
            Some(q) => {
                q.sweep();
                snapshot_of(q, now)
            }
        }
    }

    /// Snapshots of every known model.
    pub fn snapshot_all(&self) -> Vec<(String, QueueSnapshot)> {
        let now = Instant::now();
        let mut g = self.inner.lock().expect("arena queues lock");
        g.iter_mut()
            .map(|(m, q)| {
                q.sweep();
                (m.clone(), snapshot_of(q, now))
            })
            .collect()
    }

    pub fn queued_for(&self, model: &str) -> usize {
        let mut g = self.inner.lock().expect("arena queues lock");
        g.get_mut(model)
            .map(|q| {
                q.sweep();
                q.len()
            })
            .unwrap_or(0)
    }

    pub fn total_queued(&self) -> usize {
        let mut g = self.inner.lock().expect("arena queues lock");
        g.values_mut()
            .map(|q| {
                q.sweep();
                q.len()
            })
            .sum()
    }
}

fn snapshot_of(q: &ModelQueue, now: Instant) -> QueueSnapshot {
    let wait = |w: &Weak<TicketInner>| {
        w.upgrade()
            .map(|t| now.duration_since(t.enqueued).as_secs_f64())
            .unwrap_or(0.0)
    };
    QueueSnapshot {
        priority_len: q.priority.len(),
        normal_len: q.normal.len(),
        normal_pressure_secs: q.normal.iter().map(wait).sum(),
        oldest_normal_wait_secs: q.normal.front().map(wait).unwrap_or(0.0),
        oldest_priority_wait_secs: q.priority.front().map(wait).unwrap_or(0.0),
        lambda: q.decayed_lambda(now),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn grant_wakes_waiters_priority_first() {
        let q = ArenaQueues::new(64, 0.2);
        let n = q.submit("m", false, false).unwrap();
        let p = q.submit("m", true, false).unwrap();
        let (granted_p, granted_n) = q.grant_all("m");
        assert_eq!((granted_p, granted_n), (1, 1));
        assert_eq!(p.wait().await, TicketOutcome::Granted);
        assert_eq!(n.wait().await, TicketOutcome::Granted);
    }

    #[tokio::test]
    async fn queue_full_rejects_with_429_semantics() {
        let q = ArenaQueues::new(2, 0.2);
        let _a = q.submit("m", false, false).unwrap();
        let _b = q.submit("m", true, false).unwrap();
        assert!(matches!(
            q.submit("m", false, false),
            Err(SubmitError::QueueFull { .. })
        ));
        // Other models are unaffected.
        let _c = q.submit("other", false, false).unwrap();
    }

    #[tokio::test]
    async fn dropped_ticket_leaves_the_queue() {
        let q = ArenaQueues::new(2, 0.2);
        {
            let _dropped = q.submit("m", false, false).unwrap();
        } // client disconnected (DRN-5)
        assert_eq!(q.queued_for("m"), 0);
        // Its slot is reusable.
        let _a = q.submit("m", false, false).unwrap();
        let _b = q.submit("m", false, false).unwrap();
        assert!(q.submit("m", false, false).is_err());
    }

    #[tokio::test]
    async fn pressure_counts_only_normal_tier() {
        let q = ArenaQueues::new(64, 0.2);
        let _p = q.submit("m", true, false).unwrap();
        let _n = q.submit("m", false, false).unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        let s = q.snapshot("m");
        assert_eq!((s.priority_len, s.normal_len), (1, 1));
        assert!(s.normal_pressure_secs > 0.0);
        assert!(s.oldest_priority_wait_secs > 0.0);
        assert!(s.normal_pressure_secs < 2.0 * s.oldest_normal_wait_secs + 1e-6);
    }

    #[tokio::test]
    async fn expiry_hits_foreign_priority_and_nonstream() {
        let q = ArenaQueues::new(64, 0.2);
        let loaded_stream_p = q.submit("loaded", true, true).unwrap();
        let foreign_p = q.submit("other", true, true).unwrap();
        let nonstream = q.submit("other", false, false).unwrap();
        tokio::time::sleep(Duration::from_millis(15)).await;
        let expired = q.expire_sweep(
            Some("loaded"),
            Duration::from_millis(10),
            Duration::from_millis(10),
        );
        assert_eq!(expired, 2);
        assert_eq!(foreign_p.wait().await, TicketOutcome::Expired);
        assert_eq!(nonstream.wait().await, TicketOutcome::Expired);
        // Loaded-model streaming priority ticket survives both rules.
        assert_eq!(q.queued_for("loaded"), 1);
        drop(loaded_stream_p);
    }

    #[tokio::test]
    async fn reject_all_signals_load_failure() {
        let q = ArenaQueues::new(64, 0.2);
        let t = q.submit("m", false, false).unwrap();
        assert_eq!(q.reject_all("m"), 1);
        assert_eq!(t.wait().await, TicketOutcome::Rejected);
    }

    #[tokio::test]
    async fn lambda_rises_with_arrivals() {
        let q = ArenaQueues::new(64, 0.5);
        for _ in 0..5 {
            q.note_arrival("m");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(q.snapshot("m").lambda > 1.0);
    }
}

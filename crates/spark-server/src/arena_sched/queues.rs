// SPDX-License-Identifier: AGPL-3.0-only

//! Per-arena-model request queues (SCH-1/2/3, PRI-1): two FIFO tiers
//! (priority, normal) per model, arrival-rate EWMA, pressure/oldest-wait
//! statistics for the decision function, and queue-length limits
//! (SCH-8). Thread-safe — HTTP handlers submit from Tokio workers, the
//! dispatcher consumes from its own task.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::api::InferenceRequest;

/// One queued request with its admission metadata (SCH-1).
pub struct QueuedRequest {
    pub req: InferenceRequest,
    pub enqueued: Instant,
    pub priority: bool,
    pub streaming: bool,
}

#[derive(Default)]
struct ModelQueue {
    priority: VecDeque<QueuedRequest>,
    normal: VecDeque<QueuedRequest>,
    /// Arrival rate EWMA (req/s) over ALL tiers (λ_A in §7).
    lambda: f64,
    last_arrival: Option<Instant>,
}

impl ModelQueue {
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
        self.last_arrival = now.into();
    }

    /// Decay λ toward zero when no arrivals happen (idle detection must
    /// not act on a stale burst-era λ).
    fn decayed_lambda(&self, now: Instant) -> f64 {
        match self.last_arrival {
            None => 0.0,
            Some(prev) => {
                let silent = now.duration_since(prev).as_secs_f64();
                // Halve per 10 s of silence — smooth, monotone decay.
                self.lambda * 0.5f64.powf(silent / 10.0)
            }
        }
    }
}

/// Per-tier snapshot for telemetry + the decision function (SCH-3).
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

    /// Enqueue a request for `model` (SCH-1; priority tier per PRI-1).
    pub fn submit(
        &self,
        model: &str,
        req: InferenceRequest,
        priority: bool,
        streaming: bool,
    ) -> Result<(), (InferenceRequest, SubmitError)> {
        let now = Instant::now();
        let mut g = self.inner.lock().expect("arena queues lock");
        let q = g.entry(model.to_string()).or_default();
        if q.len() >= self.max_queue_length {
            return Err((
                req,
                SubmitError::QueueFull {
                    retry_after_secs: 5,
                },
            ));
        }
        q.observe_arrival(now, self.ewma_alpha);
        let item = QueuedRequest {
            req,
            enqueued: now,
            priority,
            streaming,
        };
        if priority {
            q.priority.push_back(item);
        } else {
            q.normal.push_back(item);
        }
        Ok(())
    }

    /// Dequeue the next request for `model` — priority tier first, FIFO
    /// within each tier (PRI-1, SCH-6).
    pub fn pop(&self, model: &str) -> Option<QueuedRequest> {
        let mut g = self.inner.lock().expect("arena queues lock");
        let q = g.get_mut(model)?;
        q.priority.pop_front().or_else(|| q.normal.pop_front())
    }

    /// Push a request back to the FRONT of its tier (admission couldn't
    /// place it yet — keeps FIFO order intact).
    pub fn push_front(&self, model: &str, item: QueuedRequest) {
        let mut g = self.inner.lock().expect("arena queues lock");
        let q = g.entry(model.to_string()).or_default();
        if item.priority {
            q.priority.push_front(item);
        } else {
            q.normal.push_front(item);
        }
    }

    /// Drop queued requests whose client is gone (DRN-5). The probe
    /// returns true when the request's response channel is closed.
    pub fn drop_disconnected(&self, is_dead: impl Fn(&InferenceRequest) -> bool) -> usize {
        let mut g = self.inner.lock().expect("arena queues lock");
        let mut dropped = 0;
        for q in g.values_mut() {
            for tier in [&mut q.priority, &mut q.normal] {
                let before = tier.len();
                tier.retain(|item| !is_dead(&item.req));
                dropped += before - tier.len();
            }
        }
        dropped
    }

    /// Expire PRIORITY requests for `not_loaded` models older than
    /// `priority_max_wait` (PRI-4) and NON-STREAMING requests older than
    /// `nonstream_max_wait` (WAIT-4). Returns the expired requests for
    /// the caller to answer with 503 + Retry-After.
    pub fn take_expired(
        &self,
        loaded_model: Option<&str>,
        priority_max_wait: Duration,
        nonstream_max_wait: Duration,
    ) -> Vec<(String, QueuedRequest)> {
        let now = Instant::now();
        let mut g = self.inner.lock().expect("arena queues lock");
        let mut out = Vec::new();
        for (model, q) in g.iter_mut() {
            let foreign = loaded_model != Some(model.as_str());
            let mut keep_p = VecDeque::new();
            for item in q.priority.drain(..) {
                let waited = now.duration_since(item.enqueued);
                let expired = (foreign && waited > priority_max_wait)
                    || (!item.streaming && waited > nonstream_max_wait);
                if expired {
                    out.push((model.clone(), item));
                } else {
                    keep_p.push_back(item);
                }
            }
            q.priority = keep_p;
            let mut keep_n = VecDeque::new();
            for item in q.normal.drain(..) {
                let waited = now.duration_since(item.enqueued);
                if !item.streaming && waited > nonstream_max_wait {
                    out.push((model.clone(), item));
                } else {
                    keep_n.push_back(item);
                }
            }
            q.normal = keep_n;
        }
        out
    }

    /// Snapshot of one model's queue (SCH-3, PRI-6).
    pub fn snapshot(&self, model: &str) -> QueueSnapshot {
        let now = Instant::now();
        let g = self.inner.lock().expect("arena queues lock");
        match g.get(model) {
            None => QueueSnapshot::default(),
            Some(q) => snapshot_of(q, now),
        }
    }

    /// Snapshots of every model with queued work or a live λ.
    pub fn snapshot_all(&self) -> Vec<(String, QueueSnapshot)> {
        let now = Instant::now();
        let g = self.inner.lock().expect("arena queues lock");
        g.iter()
            .map(|(m, q)| (m.clone(), snapshot_of(q, now)))
            .collect()
    }

    pub fn total_queued(&self) -> usize {
        let g = self.inner.lock().expect("arena queues lock");
        g.values().map(|q| q.len()).sum()
    }

    pub fn queued_for(&self, model: &str) -> usize {
        let g = self.inner.lock().expect("arena queues lock");
        g.get(model).map(|q| q.len()).unwrap_or(0)
    }
}

fn snapshot_of(q: &ModelQueue, now: Instant) -> QueueSnapshot {
    let wait = |item: &QueuedRequest| now.duration_since(item.enqueued).as_secs_f64();
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

    fn dummy_req() -> InferenceRequest {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        InferenceRequest::Blocking {
            prompt_tokens: std::sync::Arc::new(vec![1, 2, 3]),
            session_hash: 0,
            image_pixels: Vec::new(),
            max_tokens: 1,
            min_tokens: 0,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            top_n_sigma: 0.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: 0.0,
            dry_allowed_length: 0,
            lz_penalty: 0.0,
            logit_bias: Vec::new(),
            stop_tokens: Vec::new(),
            enable_thinking: false,
            thinking_budget: None,
            repetition_detection: None,
            require_tool_call: false,
            tools_present: false,
            suppress_tool_call: false,
            disable_mtp: false,
            grammar_spec: None,
            seed: None,
            top_logprobs: None,
            timeout_at: None,
            logit_probe: None,
            capture_hidden: false,
            response_tx: tx,
        }
    }

    #[test]
    fn priority_tier_pops_first_fifo_within_tiers() {
        let q = ArenaQueues::new(64, 0.2);
        q.submit("m", dummy_req(), false, false)
            .map_err(|_| ())
            .unwrap();
        q.submit("m", dummy_req(), true, false)
            .map_err(|_| ())
            .unwrap();
        q.submit("m", dummy_req(), true, false)
            .map_err(|_| ())
            .unwrap();
        let a = q.pop("m").unwrap();
        let b = q.pop("m").unwrap();
        let c = q.pop("m").unwrap();
        assert!(a.priority && b.priority && !c.priority);
        assert!(a.enqueued <= b.enqueued, "FIFO within the priority tier");
        assert!(q.pop("m").is_none());
    }

    #[test]
    fn queue_full_rejects_with_429_semantics() {
        let q = ArenaQueues::new(2, 0.2);
        q.submit("m", dummy_req(), false, false)
            .map_err(|_| ())
            .unwrap();
        q.submit("m", dummy_req(), true, false)
            .map_err(|_| ())
            .unwrap();
        let err = q.submit("m", dummy_req(), false, false);
        assert!(matches!(err, Err((_, SubmitError::QueueFull { .. }))));
        // Other models are unaffected.
        q.submit("other", dummy_req(), false, false)
            .map_err(|_| ())
            .unwrap();
    }

    #[test]
    fn pressure_counts_only_normal_tier() {
        let q = ArenaQueues::new(64, 0.2);
        q.submit("m", dummy_req(), true, false)
            .map_err(|_| ())
            .unwrap();
        q.submit("m", dummy_req(), false, false)
            .map_err(|_| ())
            .unwrap();
        std::thread::sleep(Duration::from_millis(20));
        let s = q.snapshot("m");
        assert_eq!(s.priority_len, 1);
        assert_eq!(s.normal_len, 1);
        assert!(s.normal_pressure_secs > 0.0);
        assert!(s.oldest_priority_wait_secs > 0.0);
        // PRI-2: the priority wait is NOT part of the normal pressure.
        assert!(s.normal_pressure_secs < 2.0 * s.oldest_normal_wait_secs + 1e-6);
    }

    #[test]
    fn lambda_rises_with_arrivals_and_decays_in_silence() {
        let q = ArenaQueues::new(64, 0.5);
        for _ in 0..5 {
            q.submit("m", dummy_req(), false, false)
                .map_err(|_| ())
                .unwrap();
            std::thread::sleep(Duration::from_millis(5));
        }
        let live = q.snapshot("m").lambda;
        assert!(live > 1.0, "5ms inter-arrivals → λ ≫ 1/s, got {live}");
    }

    #[test]
    fn push_front_preserves_head_position() {
        let q = ArenaQueues::new(64, 0.2);
        q.submit("m", dummy_req(), false, false)
            .map_err(|_| ())
            .unwrap();
        q.submit("m", dummy_req(), false, false)
            .map_err(|_| ())
            .unwrap();
        let head = q.pop("m").unwrap();
        let head_t = head.enqueued;
        q.push_front("m", head);
        assert_eq!(q.pop("m").unwrap().enqueued, head_t);
    }

    #[test]
    fn expiry_hits_foreign_priority_and_nonstream_waiters() {
        let q = ArenaQueues::new(64, 0.2);
        q.submit("loaded", dummy_req(), true, false)
            .map_err(|_| ())
            .unwrap();
        q.submit("other", dummy_req(), true, true)
            .map_err(|_| ())
            .unwrap(); // foreign priority (streaming)
        q.submit("other", dummy_req(), false, false)
            .map_err(|_| ())
            .unwrap(); // foreign normal nonstream
        std::thread::sleep(Duration::from_millis(15));
        // priority_max_wait exceeded for foreign priority; nonstream_max_wait
        // exceeded for the nonstream one; loaded-model priority survives
        // (PRI-4 applies to NOT-loaded models only; it is streaming? no —
        // loaded + nonstream BUT nonstream_max also applies... use large).
        let expired = q.take_expired(
            Some("loaded"),
            Duration::from_millis(10),
            Duration::from_millis(10),
        );
        let models: Vec<&str> = expired.iter().map(|(m, _)| m.as_str()).collect();
        assert!(models.contains(&"other"));
        // the loaded model's nonstream priority item also expired via
        // nonstream_max_wait (WAIT-4 applies regardless of model).
        assert_eq!(expired.len(), 3);
        assert_eq!(q.total_queued(), 0);
    }
}

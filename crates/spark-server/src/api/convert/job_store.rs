// SPDX-License-Identifier: AGPL-3.0-only

//! In-memory job store for `/convert` (Datalab-style submit → poll).
//!
//! Three bounds keep it finite (design doc §Frage 6): finished results
//! expire after `result_ttl`; a running job nobody polled for
//! `poll_idle` is abandoned (cancelled, `error_code: abandoned`); the
//! stored result bytes and the job count are capped. Everything is a
//! plain `Mutex<HashMap>` — the store is small (tens of jobs) and every
//! operation is O(1) except the periodic sweep.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct JobConfig {
    /// How long a complete/failed result stays retrievable.
    pub result_ttl: Duration,
    /// A pending/processing job without a poll for this long is abandoned.
    pub poll_idle: Duration,
    /// Cap on the sum of stored result bytes.
    pub store_max_bytes: u64,
    /// Cap on jobs held in the store (any state).
    pub max_jobs: usize,
}

#[derive(Clone, Debug)]
pub enum JobState {
    Pending,
    Processing,
    /// Serialized Datalab result document, kept as a shared JSON value.
    Complete(Arc<serde_json::Value>),
    Failed {
        error: String,
        error_code: String,
    },
}

impl JobState {
    pub fn status_str(&self) -> &'static str {
        match self {
            JobState::Pending => "pending",
            JobState::Processing => "processing",
            JobState::Complete(_) => "complete",
            JobState::Failed { .. } => "failed",
        }
    }

    fn is_terminal(&self) -> bool {
        matches!(self, JobState::Complete(_) | JobState::Failed { .. })
    }
}

#[derive(Debug)]
struct Job {
    created: Instant,
    last_poll: Instant,
    finished: Option<Instant>,
    state: JobState,
    cancel: Arc<AtomicBool>,
    deadline: Option<Instant>,
    result_bytes: u64,
}

/// What a poll returns.
#[derive(Clone, Debug)]
pub struct JobView {
    pub state: JobState,
    pub created: Instant,
    pub deadline: Option<Instant>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum CreateError {
    /// `max_jobs` reached.
    TooManyJobs,
    /// `store_max_bytes` reached — results must expire first.
    StoreFull,
}

#[derive(Debug, Default)]
pub struct SweepReport {
    pub expired: usize,
    /// Jobs cancelled because nobody polled them in time.
    pub abandoned: Vec<String>,
    /// Jobs cancelled because their deadline passed while running.
    pub deadline_exceeded: Vec<String>,
}

pub struct JobStore {
    cfg: JobConfig,
    inner: Mutex<HashMap<String, Job>>,
    stored_bytes: AtomicU64,
}

impl JobStore {
    pub fn new(cfg: JobConfig) -> Self {
        Self {
            cfg,
            inner: Mutex::new(HashMap::new()),
            stored_bytes: AtomicU64::new(0),
        }
    }

    pub fn config(&self) -> &JobConfig {
        &self.cfg
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Job>> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Register a new pending job. Returns `(id, cancel_flag)`.
    pub fn create(
        &self,
        id: String,
        now: Instant,
        deadline: Option<Instant>,
    ) -> Result<Arc<AtomicBool>, CreateError> {
        let mut m = self.lock();
        if m.len() >= self.cfg.max_jobs {
            return Err(CreateError::TooManyJobs);
        }
        if self.stored_bytes.load(Ordering::Relaxed) >= self.cfg.store_max_bytes {
            return Err(CreateError::StoreFull);
        }
        let cancel = Arc::new(AtomicBool::new(false));
        m.insert(
            id,
            Job {
                created: now,
                last_poll: now,
                finished: None,
                state: JobState::Pending,
                cancel: Arc::clone(&cancel),
                deadline,
                result_bytes: 0,
            },
        );
        Ok(cancel)
    }

    pub fn set_processing(&self, id: &str) {
        if let Some(j) = self.lock().get_mut(id)
            && matches!(j.state, JobState::Pending)
        {
            j.state = JobState::Processing;
        }
    }

    /// Store the result. Ignored when the job was already finished
    /// (cancelled/abandoned while the worker was still running).
    pub fn complete(&self, id: &str, result: serde_json::Value, bytes: u64, now: Instant) -> bool {
        let mut m = self.lock();
        let Some(j) = m.get_mut(id) else {
            return false;
        };
        if j.state.is_terminal() {
            return false;
        }
        j.state = JobState::Complete(Arc::new(result));
        j.finished = Some(now);
        j.result_bytes = bytes;
        self.stored_bytes.fetch_add(bytes, Ordering::Relaxed);
        true
    }

    /// Mark the job failed. Ignored when already finished.
    pub fn fail(&self, id: &str, error_code: &str, error: String, now: Instant) -> bool {
        let mut m = self.lock();
        let Some(j) = m.get_mut(id) else {
            return false;
        };
        if j.state.is_terminal() {
            return false;
        }
        j.state = JobState::Failed {
            error,
            error_code: error_code.to_string(),
        };
        j.finished = Some(now);
        j.cancel.store(true, Ordering::Relaxed);
        true
    }

    /// Read the job and record the poll (keeps it from being abandoned).
    pub fn poll(&self, id: &str, now: Instant) -> Option<JobView> {
        let mut m = self.lock();
        let j = m.get_mut(id)?;
        j.last_poll = now;
        Some(JobView {
            state: j.state.clone(),
            created: j.created,
            deadline: j.deadline,
        })
    }

    /// `DELETE`: cancel a running job (idempotent). Returns whether the
    /// id was known. A finished job is left as is.
    pub fn cancel(&self, id: &str, now: Instant) -> bool {
        let mut m = self.lock();
        let Some(j) = m.get_mut(id) else {
            return false;
        };
        if !j.state.is_terminal() {
            j.state = JobState::Failed {
                error: "cancelled by client".into(),
                error_code: "cancelled".into(),
            };
            j.finished = Some(now);
        }
        j.cancel.store(true, Ordering::Relaxed);
        true
    }

    /// Periodic maintenance: drop expired results, abandon unpolled
    /// jobs, fail jobs past their deadline. Pure bookkeeping — the
    /// caller logs/alerts from the report.
    pub fn sweep(&self, now: Instant) -> SweepReport {
        let mut report = SweepReport::default();
        let mut m = self.lock();
        let mut freed = 0u64;
        m.retain(|_, j| {
            if let Some(f) = j.finished
                && now.duration_since(f) >= self.cfg.result_ttl
            {
                freed += j.result_bytes;
                report.expired += 1;
                return false;
            }
            true
        });
        if freed > 0 {
            self.stored_bytes.fetch_sub(freed, Ordering::Relaxed);
        }
        for (id, j) in m.iter_mut() {
            if j.state.is_terminal() {
                continue;
            }
            if let Some(d) = j.deadline
                && now >= d
            {
                j.state = JobState::Failed {
                    error: "accepted_processing_time exceeded".into(),
                    error_code: "deadline_exceeded".into(),
                };
                j.finished = Some(now);
                j.cancel.store(true, Ordering::Relaxed);
                report.deadline_exceeded.push(id.clone());
            } else if now.duration_since(j.last_poll) >= self.cfg.poll_idle {
                j.state = JobState::Failed {
                    error: format!(
                        "no status poll for {}s — job abandoned",
                        self.cfg.poll_idle.as_secs()
                    ),
                    error_code: "abandoned".into(),
                };
                j.finished = Some(now);
                j.cancel.store(true, Ordering::Relaxed);
                report.abandoned.push(id.clone());
            }
        }
        report
    }

    pub fn len(&self) -> usize {
        self.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn stored_bytes(&self) -> u64 {
        self.stored_bytes.load(Ordering::Relaxed)
    }

    /// Counts per state, for `/internal/state`.
    pub fn counts(&self) -> (usize, usize, usize, usize) {
        let m = self.lock();
        let mut c = (0, 0, 0, 0);
        for j in m.values() {
            match j.state {
                JobState::Pending => c.0 += 1,
                JobState::Processing => c.1 += 1,
                JobState::Complete(_) => c.2 += 1,
                JobState::Failed { .. } => c.3 += 1,
            }
        }
        c
    }
}

/// 128 random bits as 32 hex chars. Job ids are capability tokens for
/// the result (CWE-330: must not be guessable), so this reads the OS
/// CSPRNG and fails loudly instead of falling back to a weak source.
pub fn new_job_id() -> std::io::Result<String> {
    use std::io::Read;
    let mut buf = [0u8; 16];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut buf)?;
    Ok(buf.iter().map(|b| format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> JobConfig {
        JobConfig {
            result_ttl: Duration::from_secs(100),
            poll_idle: Duration::from_secs(10),
            store_max_bytes: 1000,
            max_jobs: 2,
        }
    }

    #[test]
    fn lifecycle_and_poll_touch() {
        let s = JobStore::new(cfg());
        let t0 = Instant::now();
        let cancel = s.create("a".into(), t0, None).unwrap();
        assert_eq!(s.poll("a", t0).unwrap().state.status_str(), "pending");
        s.set_processing("a");
        assert_eq!(s.poll("a", t0).unwrap().state.status_str(), "processing");
        assert!(s.complete("a", serde_json::json!({"ok": true}), 10, t0));
        assert_eq!(s.stored_bytes(), 10);
        // Second completion / failure is ignored.
        assert!(!s.fail("a", "x", "y".into(), t0));
        assert!(!cancel.load(Ordering::Relaxed));
        // Expiry.
        let r = s.sweep(t0 + Duration::from_secs(100));
        assert_eq!(r.expired, 1);
        assert!(s.is_empty());
        assert_eq!(s.stored_bytes(), 0);
    }

    #[test]
    fn abandoned_when_not_polled() {
        let s = JobStore::new(cfg());
        let t0 = Instant::now();
        let cancel = s.create("a".into(), t0, None).unwrap();
        s.set_processing("a");
        // Poll at t+5 keeps it alive at t+12.
        s.poll("a", t0 + Duration::from_secs(5));
        assert!(s.sweep(t0 + Duration::from_secs(12)).abandoned.is_empty());
        let r = s.sweep(t0 + Duration::from_secs(15));
        assert_eq!(r.abandoned, vec!["a".to_string()]);
        assert!(cancel.load(Ordering::Relaxed));
        match s.poll("a", t0).unwrap().state {
            JobState::Failed { error_code, .. } => assert_eq!(error_code, "abandoned"),
            other => panic!("{other:?}"),
        }
        // Late completion from the worker is dropped.
        assert!(!s.complete("a", serde_json::json!({}), 1, t0));
    }

    #[test]
    fn deadline_and_cancel() {
        let s = JobStore::new(cfg());
        let t0 = Instant::now();
        let cancel = s
            .create("d".into(), t0, Some(t0 + Duration::from_secs(3)))
            .unwrap();
        let r = s.sweep(t0 + Duration::from_secs(3));
        assert_eq!(r.deadline_exceeded, vec!["d".to_string()]);
        assert!(cancel.load(Ordering::Relaxed));

        let c2 = s.create("c".into(), t0, None).unwrap();
        assert!(s.cancel("c", t0));
        assert!(c2.load(Ordering::Relaxed));
        assert!(s.cancel("c", t0), "idempotent");
        assert!(!s.cancel("nope", t0));
        assert_eq!(s.poll("c", t0).unwrap().state.status_str(), "failed");
    }

    #[test]
    fn caps() {
        let s = JobStore::new(cfg());
        let t0 = Instant::now();
        s.create("a".into(), t0, None).unwrap();
        s.create("b".into(), t0, None).unwrap();
        assert_eq!(
            s.create("c".into(), t0, None).unwrap_err(),
            CreateError::TooManyJobs
        );
        s.complete("a", serde_json::json!({}), 1000, t0);
        s.sweep(t0 + Duration::from_secs(100));
        // "a" expired → one slot free, but bytes were freed too.
        assert!(s.create("c".into(), t0, None).is_ok());
        s.complete("c", serde_json::json!({}), 1000, t0);
        assert_eq!(
            s.create("e".into(), t0, None).unwrap_err(),
            CreateError::TooManyJobs
        );
        s.sweep(t0 + Duration::from_secs(50)); // nothing expires yet
        s.cancel("b", t0);
        s.sweep(t0 + Duration::from_secs(150));
        // "b" gone (ttl after finish), "c" gone (ttl) → bytes back to 0.
        assert_eq!(s.stored_bytes(), 0);
    }

    #[test]
    fn job_ids_are_hex_and_unique() {
        let a = new_job_id().unwrap();
        let b = new_job_id().unwrap();
        assert_eq!(a.len(), 32);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b);
    }
}

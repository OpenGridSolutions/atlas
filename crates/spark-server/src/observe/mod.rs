// SPDX-License-Identifier: AGPL-3.0-only

//! Runtime observability (docs/design/deadline-admission.md §6/§7, PR C):
//! anomaly events → ntfy push + SQLite event store, periodic state
//! snapshots, per-request records.
//!
//! SBIO: business code calls [`emit`], [`record_request`] or [`fatal`];
//! everything that touches the network or the database happens on ONE
//! background thread ([`notify::Notifier`], [`store::EventStore`]).
//! `emit` never blocks — a bounded channel drops events when the thread
//! falls behind (and counts the drops). [`fatal`] is the exception: a
//! process about to `exit(1)` or abort has no background thread left, so
//! it posts synchronously with a short timeout.

pub mod detectors;
pub mod layer;
pub mod notify;
pub mod store;

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use notify::Notifier;
use serde::Serialize;

/// One anomaly. `priority` follows ntfy's 1 (min) … 5 (urgent); events
/// below [`ObserverConfig::notify_min_priority`] are stored but never
/// pushed.
#[derive(Clone, Debug, Serialize)]
pub struct Event {
    pub ts_ms: u64,
    pub priority: u8,
    pub kind: String,
    pub model: Option<String>,
    pub request_id: Option<String>,
    pub message: String,
    pub detail: serde_json::Value,
}

impl Event {
    pub fn new(priority: u8, kind: &str, message: impl Into<String>) -> Self {
        Self {
            ts_ms: now_ms(),
            priority: priority.clamp(1, 5),
            kind: kind.to_string(),
            model: None,
            request_id: None,
            message: message.into(),
            detail: serde_json::Value::Null,
        }
    }
    pub fn model(mut self, m: impl Into<String>) -> Self {
        self.model = Some(m.into());
        self
    }
    pub fn request_id(mut self, r: impl Into<String>) -> Self {
        self.request_id = Some(r.into());
        self
    }
    pub fn detail(mut self, d: serde_json::Value) -> Self {
        self.detail = d;
        self
    }
    /// Dedupe key for the notifier: same kind + model within the minimum
    /// interval collapses to one push.
    fn key(&self) -> String {
        match &self.model {
            Some(m) => format!("{}:{m}", self.kind),
            None => self.kind.clone(),
        }
    }
}

/// One finished HTTP request (the `requests` table).
#[derive(Clone, Debug, Serialize)]
pub struct RequestRecord {
    pub ts_ms: u64,
    pub request_id: String,
    pub endpoint: String,
    pub status: u16,
    pub elapsed_ms: u64,
    pub model: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ObserverConfig {
    /// `(server url, topic)`; None = no push, store only.
    pub ntfy: Option<(String, String)>,
    /// Minimum gap between two pushes of the same key.
    pub min_interval: Duration,
    /// Cap on pushes per key per hour.
    pub max_per_key_per_hour: usize,
    /// Events at or above this priority are pushed.
    pub notify_min_priority: u8,
    pub db_path: Option<PathBuf>,
    pub db_max_bytes: u64,
    pub snapshot_interval: Duration,
    /// An event at or above this priority also triggers a snapshot.
    pub snapshot_on_priority: u8,
    /// Shown in push titles.
    pub host_label: String,
}

enum Msg {
    Event(Event),
    Request(RequestRecord),
}

struct Observer {
    tx: SyncSender<Msg>,
    cfg: ObserverConfig,
    fatal_notifier: Arc<dyn Notifier>,
}

static OBSERVER: OnceLock<Observer> = OnceLock::new();
static DROPPED: AtomicU64 = AtomicU64::new(0);
/// Events emitted before `start` (boot-time errors) are kept here and
/// flushed on start, so nothing from the boot path is lost.
static PENDING: Mutex<Vec<Event>> = Mutex::new(Vec::new());
const PENDING_CAP: usize = 256;
const QUEUE_CAP: usize = 4096;

/// Short host name for push titles (`HOSTNAME`, `/etc/hostname`, "atlas").
pub fn host_label() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .or_else(|| std::fs::read_to_string("/etc/hostname").ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "atlas".into())
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Start the background thread. `snapshot` builds the state document
/// that is stored every `snapshot_interval` and on urgent events.
pub fn start(
    cfg: ObserverConfig,
    snapshot: Box<dyn Fn() -> serde_json::Value + Send + Sync>,
) -> anyhow::Result<()> {
    let notifier: Arc<dyn Notifier> = match &cfg.ntfy {
        Some((url, topic)) => Arc::new(notify::NtfyNotifier::new(url, topic)),
        None => Arc::new(notify::NoopNotifier),
    };
    let mut store: Box<dyn store::EventStore> = match &cfg.db_path {
        Some(p) => Box::new(store::SqliteStore::open(p)?),
        None => Box::new(store::NoopStore),
    };
    let (tx, rx) = sync_channel::<Msg>(QUEUE_CAP);
    let worker_cfg = cfg.clone();
    let worker_notifier = Arc::clone(&notifier);
    std::thread::Builder::new()
        .name("atlas-observe".into())
        .spawn(move || run(rx, worker_cfg, worker_notifier, store.as_mut(), snapshot))
        .map_err(|e| anyhow::anyhow!("observe thread: {e}"))?;
    let obs = Observer {
        tx,
        cfg,
        fatal_notifier: notifier,
    };
    if OBSERVER.set(obs).is_err() {
        anyhow::bail!("observer already started");
    }
    let pending: Vec<Event> =
        std::mem::take(&mut *PENDING.lock().unwrap_or_else(|e| e.into_inner()));
    for ev in pending {
        emit(ev);
    }
    Ok(())
}

/// Queue an event. Never blocks; drops (counted) when the queue is full.
pub fn emit(ev: Event) {
    match OBSERVER.get() {
        Some(o) => {
            if let Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) =
                o.tx.try_send(Msg::Event(ev))
            {
                DROPPED.fetch_add(1, Ordering::Relaxed);
            }
        }
        None => {
            let mut p = PENDING.lock().unwrap_or_else(|e| e.into_inner());
            if p.len() < PENDING_CAP {
                p.push(ev);
            }
        }
    }
}

pub fn record_request(r: RequestRecord) {
    if let Some(o) = OBSERVER.get()
        && o.tx.try_send(Msg::Request(r)).is_err()
    {
        DROPPED.fetch_add(1, Ordering::Relaxed);
    }
}

/// Events dropped because the queue was full (for `/internal/state`).
pub fn dropped() -> u64 {
    DROPPED.load(Ordering::Relaxed)
}

/// Synchronous last words for panics and the OOM watchdog: push with a
/// short timeout, then append to the database on a fresh connection.
/// Installed into `spark_runtime::crash::set_fatal_hook`.
pub fn fatal(kind: &str, detail: &str) {
    let Some(o) = OBSERVER.get() else {
        return;
    };
    let ev = Event::new(5, kind, format!("FATAL {kind}: {}", truncate(detail, 900)));
    let title = format!("Atlas {}: {kind}", o.cfg.host_label);
    let _ = o
        .fatal_notifier
        .notify(&title, &ev.message, 5, "rotating_light");
    if let Some(p) = &o.cfg.db_path
        && let Ok(mut s) = store::SqliteStore::open(p)
    {
        store::EventStore::insert_event(&mut s, &ev, true);
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        let mut end = n;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &s[..end])
    }
}

fn tags_for(priority: u8) -> &'static str {
    match priority {
        5 => "rotating_light",
        4 | 3 => "warning",
        _ => "information_source",
    }
}

/// Per-key push throttle: min interval + hourly cap. Pure, so the
/// dedupe policy is unit-testable without a thread.
pub struct Throttle {
    min_interval: Duration,
    per_hour: usize,
    sent: HashMap<String, VecDeque<Instant>>,
}

impl Throttle {
    pub fn new(min_interval: Duration, per_hour: usize) -> Self {
        Self {
            min_interval,
            per_hour: per_hour.max(1),
            sent: HashMap::new(),
        }
    }

    /// True when a push for `key` is allowed now (and records it).
    pub fn allow(&mut self, key: &str, now: Instant) -> bool {
        let q = self.sent.entry(key.to_string()).or_default();
        while q
            .front()
            .is_some_and(|t| now.duration_since(*t) >= Duration::from_secs(3600))
        {
            q.pop_front();
        }
        if q.len() >= self.per_hour {
            return false;
        }
        if q.back()
            .is_some_and(|t| now.duration_since(*t) < self.min_interval)
        {
            return false;
        }
        q.push_back(now);
        true
    }
}

fn run(
    rx: Receiver<Msg>,
    cfg: ObserverConfig,
    notifier: Arc<dyn Notifier>,
    store: &mut dyn store::EventStore,
    snapshot: Box<dyn Fn() -> serde_json::Value + Send + Sync>,
) {
    let mut throttle = Throttle::new(cfg.min_interval, cfg.max_per_key_per_hour);
    let mut last_snapshot = Instant::now() - cfg.snapshot_interval;
    let mut last_size_check = Instant::now();
    let mut suppressed: u64 = 0;
    loop {
        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(Msg::Event(ev)) => {
                let now = Instant::now();
                let push = ev.priority >= cfg.notify_min_priority;
                let allowed = push && throttle.allow(&ev.key(), now);
                if push && !allowed {
                    suppressed += 1;
                }
                store.insert_event(&ev, allowed);
                if allowed {
                    let title = format!("Atlas {}: {}", cfg.host_label, ev.kind);
                    let mut body = truncate(&ev.message, 1200);
                    if let Some(m) = &ev.model {
                        body.push_str(&format!("\nmodel: {m}"));
                    }
                    if let Some(r) = &ev.request_id {
                        body.push_str(&format!("\nrequest: {r}"));
                    }
                    if suppressed > 0 {
                        body.push_str(&format!("\n({suppressed} similar pushes suppressed)"));
                        suppressed = 0;
                    }
                    if let Err(e) =
                        notifier.notify(&title, &body, ev.priority, tags_for(ev.priority))
                    {
                        tracing::warn!(target: "observe", "ntfy push failed: {e}");
                    }
                }
                if ev.priority >= cfg.snapshot_on_priority
                    && now.duration_since(last_snapshot) >= Duration::from_secs(5)
                {
                    store.insert_snapshot(&format!("event:{}", ev.kind), &snapshot());
                    last_snapshot = now;
                }
            }
            Ok(Msg::Request(r)) => store.insert_request(&r),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
        let now = Instant::now();
        if now.duration_since(last_snapshot) >= cfg.snapshot_interval {
            store.insert_snapshot("periodic", &snapshot());
            last_snapshot = now;
        }
        if now.duration_since(last_size_check) >= Duration::from_secs(60) {
            store.enforce_size(cfg.db_max_bytes);
            last_size_check = now;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throttle_min_interval_and_hourly_cap() {
        let mut t = Throttle::new(Duration::from_secs(60), 3);
        let t0 = Instant::now();
        assert!(t.allow("a", t0));
        assert!(
            !t.allow("a", t0 + Duration::from_secs(10)),
            "inside min interval"
        );
        assert!(
            t.allow("b", t0 + Duration::from_secs(10)),
            "other key independent"
        );
        assert!(t.allow("a", t0 + Duration::from_secs(61)));
        assert!(t.allow("a", t0 + Duration::from_secs(122)));
        assert!(
            !t.allow("a", t0 + Duration::from_secs(183)),
            "hourly cap of 3"
        );
        assert!(t.allow("a", t0 + Duration::from_secs(3601)), "window slid");
    }

    #[test]
    fn event_key_and_truncate() {
        let e = Event::new(9, "x", "m").model("m1");
        assert_eq!(e.priority, 5);
        assert_eq!(e.key(), "x:m1");
        assert_eq!(Event::new(0, "y", "m").key(), "y");
        assert_eq!(truncate("abc", 5), "abc");
        assert_eq!(truncate("abcdef", 3), "abc…");
        assert_eq!(truncate("äöü", 3), "ä…");
    }

    #[test]
    fn emit_before_start_is_buffered() {
        // No observer in unit tests: events land in PENDING, capped.
        for i in 0..(PENDING_CAP + 10) {
            emit(Event::new(1, "t", format!("{i}")));
        }
        let n = PENDING.lock().unwrap().len();
        assert!(n <= PENDING_CAP);
    }
}

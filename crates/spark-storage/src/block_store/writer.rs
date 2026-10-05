// SPDX-License-Identifier: AGPL-3.0-only

//! Background write queue of the NVMe block store (NVM-11): the
//! generation path enqueues host-copied payloads and never blocks on
//! disk. A full queue rejects the job — the caller keeps the block in
//! RAM (KV-7) and retries on the next release. The worker thread also
//! owns the periodic index persistence (NVM-8) and the GC cadence
//! (NVM-10), so the store needs no extra timers.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use super::BlockStore;
use super::header::Kind;

/// One queued record write.
pub struct WriteJob {
    pub kind: Kind,
    pub key: [u8; 16],
    pub pred: [u8; 16],
    pub token_count: u32,
    pub payload: Vec<u8>,
}

struct Queue {
    jobs: VecDeque<WriteJob>,
    bytes: u64,
    closed: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    /// Signalled when work arrives or the queue closes.
    work: Condvar,
    /// Signalled when the queue drains (for `flush`).
    drained: Condvar,
    in_flight: AtomicBool,
    dropped_full: AtomicU64,
}

/// Handle to the writer thread. Dropping it closes the queue, drains
/// remaining jobs and joins the thread.
pub struct StoreWriter {
    shared: Arc<Shared>,
    store: Arc<BlockStore>,
    max_queue_bytes: u64,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl StoreWriter {
    pub fn spawn(
        store: Arc<BlockStore>,
        max_queue_bytes: u64,
        gc_interval: Duration,
        index_interval: Duration,
    ) -> StoreWriter {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue {
                jobs: VecDeque::new(),
                bytes: 0,
                closed: false,
            }),
            work: Condvar::new(),
            drained: Condvar::new(),
            in_flight: AtomicBool::new(false),
            dropped_full: AtomicU64::new(0),
        });
        let worker_shared = Arc::clone(&shared);
        let worker_store = Arc::clone(&store);
        let handle = std::thread::Builder::new()
            .name("atlas-block-store".into())
            .spawn(move || worker(worker_shared, worker_store, gc_interval, index_interval))
            .expect("spawn block-store writer");
        StoreWriter {
            shared,
            store,
            max_queue_bytes,
            handle: Some(handle),
        }
    }

    /// Enqueue a record write. Never blocks. Returns the job when the
    /// queue is at its byte limit (NVM-11) — the caller must keep the
    /// block in RAM until a later enqueue succeeds — and `None` when the
    /// job was accepted OR skipped because the key is already persisted
    /// (NVM-3, checked here so duplicate payload copies die early).
    pub fn enqueue(&self, job: WriteJob) -> Option<WriteJob> {
        if self.store.contains(&job.key, job.kind) {
            self.store.touch(&job.key);
            return None;
        }
        let bytes = job.payload.len() as u64;
        let mut q = self.shared.queue.lock().expect("queue lock");
        if q.closed {
            return Some(job);
        }
        if q.bytes + bytes > self.max_queue_bytes {
            drop(q);
            let dropped = self.shared.dropped_full.fetch_add(1, Ordering::Relaxed) + 1;
            if dropped.is_power_of_two() {
                tracing::warn!(
                    queued_mb = self.pending_bytes() / (1 << 20),
                    limit_mb = self.max_queue_bytes / (1 << 20),
                    dropped,
                    "block-store write queue full — block stays in RAM (NVM-11)"
                );
            }
            return Some(job);
        }
        q.bytes += bytes;
        q.jobs.push_back(job);
        drop(q);
        self.shared.work.notify_one();
        None
    }

    /// Bytes currently queued (telemetry; host-RAM the queue pins).
    pub fn pending_bytes(&self) -> u64 {
        self.shared.queue.lock().expect("queue lock").bytes
    }

    /// Jobs rejected because the queue was full (telemetry).
    pub fn dropped_full(&self) -> u64 {
        self.shared.dropped_full.load(Ordering::Relaxed)
    }

    /// Block until the queue is fully drained or `deadline` passes
    /// (pre-swap persist, DRN-3's `persist_timeout`). Returns `true`
    /// when everything queued before the call reached disk.
    pub fn flush(&self, deadline: Instant) -> bool {
        let mut q = self.shared.queue.lock().expect("queue lock");
        loop {
            if q.jobs.is_empty() && !self.shared.in_flight.load(Ordering::Acquire) {
                return true;
            }
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let (guard, timeout) = self
                .shared
                .drained
                .wait_timeout(q, deadline - now)
                .expect("drained wait");
            q = guard;
            if timeout.timed_out()
                && q.jobs.is_empty()
                && !self.shared.in_flight.load(Ordering::Acquire)
            {
                return true;
            }
        }
    }

    pub fn store(&self) -> &Arc<BlockStore> {
        &self.store
    }
}

impl Drop for StoreWriter {
    fn drop(&mut self) {
        {
            let mut q = self.shared.queue.lock().expect("queue lock");
            q.closed = true;
        }
        self.shared.work.notify_all();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn worker(
    shared: Arc<Shared>,
    store: Arc<BlockStore>,
    gc_interval: Duration,
    index_interval: Duration,
) {
    let mut last_gc = Instant::now();
    let mut last_index = Instant::now();
    loop {
        let job = {
            let mut q = shared.queue.lock().expect("queue lock");
            loop {
                if let Some(job) = q.jobs.pop_front() {
                    q.bytes = q.bytes.saturating_sub(job.payload.len() as u64);
                    shared.in_flight.store(true, Ordering::Release);
                    break Some(job);
                }
                if q.closed {
                    break None;
                }
                let (guard, _) = shared
                    .work
                    .wait_timeout(q, Duration::from_millis(250))
                    .expect("work wait");
                q = guard;
                // Periodic duties piggyback on the wait timeout.
                if last_index.elapsed() >= index_interval || last_gc.elapsed() >= gc_interval {
                    break None;
                }
            }
        };

        match job {
            Some(job) => {
                if let Err(e) =
                    store.put(job.kind, job.key, job.pred, job.token_count, &job.payload)
                {
                    tracing::error!(
                        key = %super::hex(&job.key),
                        "block-store write failed: {e:#}"
                    );
                }
                shared.in_flight.store(false, Ordering::Release);
                let q = shared.queue.lock().expect("queue lock");
                if q.jobs.is_empty() {
                    shared.drained.notify_all();
                }
            }
            None => {
                // Timer tick or close.
                if last_index.elapsed() >= index_interval {
                    last_index = Instant::now();
                    if let Err(e) = store.persist_index() {
                        tracing::warn!("block-store index persist failed: {e:#}");
                    }
                }
                if last_gc.elapsed() >= gc_interval {
                    last_gc = Instant::now();
                    store.run_gc();
                }
                let q = shared.queue.lock().expect("queue lock");
                if q.closed && q.jobs.is_empty() {
                    drop(q);
                    let _ = store.persist_index();
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{BlockStore, BlockStoreConfig, Kind};
    use super::*;
    use std::path::Path;

    fn cfg(root: &Path) -> BlockStoreConfig {
        BlockStoreConfig {
            root: root.to_path_buf(),
            model_hash: [7; 32],
            layout_hash: [9; 32],
            session_ttl: Duration::from_secs(24 * 3600),
            min_free_pct: 0,
            write_budget_per_day: None,
        }
    }

    fn job(n: u8, payload_len: usize) -> WriteJob {
        WriteJob {
            kind: Kind::Kv,
            key: [n; 16],
            pred: [0; 16],
            token_count: 16,
            payload: vec![n; payload_len],
        }
    }

    #[test]
    fn enqueue_write_flush_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(BlockStore::open(cfg(dir.path())).unwrap());
        let writer = StoreWriter::spawn(
            Arc::clone(&store),
            1 << 20,
            Duration::from_secs(3600),
            Duration::from_secs(3600),
        );
        assert!(writer.enqueue(job(1, 1000)).is_none());
        assert!(writer.enqueue(job(2, 1000)).is_none());
        assert!(writer.flush(Instant::now() + Duration::from_secs(5)));
        assert!(store.contains(&[1; 16], Kind::Kv));
        assert!(store.contains(&[2; 16], Kind::Kv));
    }

    #[test]
    fn full_queue_rejects_without_blocking() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(BlockStore::open(cfg(dir.path())).unwrap());
        // Tiny limit: the first job may be in flight already, but a burst
        // must eventually bounce instead of blocking.
        let writer = StoreWriter::spawn(
            Arc::clone(&store),
            2048,
            Duration::from_secs(3600),
            Duration::from_secs(3600),
        );
        let mut rejected = 0;
        for n in 0..32 {
            if let Some(j) = writer.enqueue(job(n, 1024)) {
                rejected += 1;
                drop(j); // caller keeps the block in RAM (KV-7)
            }
        }
        assert!(rejected > 0, "burst past the byte limit must bounce");
        assert!(writer.dropped_full() >= rejected as u64);
    }

    #[test]
    fn duplicate_key_skipped_at_enqueue() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(BlockStore::open(cfg(dir.path())).unwrap());
        store.put(Kind::Kv, [1; 16], [0; 16], 16, &[1; 8]).unwrap();
        let writer = StoreWriter::spawn(
            Arc::clone(&store),
            1 << 20,
            Duration::from_secs(3600),
            Duration::from_secs(3600),
        );
        // Already persisted → accepted as a no-op (recency touch only).
        assert!(writer.enqueue(job(1, 8)).is_none());
        assert_eq!(writer.pending_bytes(), 0);
    }

    #[test]
    fn drop_drains_and_persists_index() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(BlockStore::open(cfg(dir.path())).unwrap());
        {
            let writer = StoreWriter::spawn(
                Arc::clone(&store),
                1 << 20,
                Duration::from_secs(3600),
                Duration::from_secs(3600),
            );
            assert!(writer.enqueue(job(9, 500)).is_none());
        } // drop joins
        let reopened = BlockStore::open(cfg(dir.path())).unwrap();
        assert!(reopened.contains(&[9; 16], Kind::Kv));
    }
}

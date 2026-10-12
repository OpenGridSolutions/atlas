// SPDX-License-Identifier: AGPL-3.0-only

//! Entry index: a bounded RAM map (so compaction works without the NVMe
//! tier) in front of the `.cmp` records of the active model's block
//! store (CMP-11). Every use refreshes the record's recency so entries
//! share the TTL/GC life cycle of the KV blocks they belong to.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use spark_runtime::block_keys::BlockKey;

use super::entry::CompactionEntry;

/// Default RAM capacity (entries). A record is ~1–4 KB.
pub const DEFAULT_RAM_ENTRIES: usize = 4096;

/// TEL-8 counters.
#[derive(Default)]
pub struct Stats {
    pub created_summarize: AtomicU64,
    pub created_truncate: AtomicU64,
    pub applied_requests: AtomicU64,
    pub rejected_ctx6: AtomicU64,
    pub rejected_policy: AtomicU64,
    pub replaced_tokens: AtomicU64,
    pub summary_millis: AtomicU64,
}

struct Inner {
    map: HashMap<BlockKey, Arc<CompactionEntry>>,
    order: VecDeque<BlockKey>,
}

pub struct CompactionIndex {
    inner: Mutex<Inner>,
    cap: usize,
    pub stats: Stats,
}

impl CompactionIndex {
    pub fn new(cap: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                order: VecDeque::new(),
            }),
            cap: cap.max(1),
            stats: Stats::default(),
        }
    }

    fn insert_ram(&self, key: BlockKey, e: Arc<CompactionEntry>) {
        let mut g = self.inner.lock().expect("compaction index lock");
        if g.map.insert(key, e).is_none() {
            g.order.push_back(key);
        }
        while g.map.len() > self.cap {
            match g.order.pop_front() {
                Some(old) => {
                    g.map.remove(&old);
                }
                None => break,
            }
        }
    }

    /// Entry for `key`: RAM first, then the active NVMe store.
    pub fn get(&self, key: &BlockKey) -> Option<Arc<CompactionEntry>> {
        let hit = self
            .inner
            .lock()
            .expect("compaction index lock")
            .map
            .get(key)
            .cloned();
        if let Some(e) = hit {
            crate::kv_tier::with_active(|t| t.cmp_touch(&key.0));
            return Some(e);
        }
        let bytes = crate::kv_tier::with_active(|t| t.cmp_get(&key.0)).flatten()?;
        let e = Arc::new(CompactionEntry::from_json_bytes(&bytes)?);
        self.insert_ram(*key, Arc::clone(&e));
        Some(e)
    }

    /// Remember a new entry (RAM + NVMe when the tier is active).
    pub fn insert(&self, e: Arc<CompactionEntry>) {
        let Some(key) = e.key_bytes() else {
            return;
        };
        let pred = e.pred_bytes();
        let payload = e.to_json_bytes();
        self.insert_ram(key, e);
        crate::kv_tier::with_active(|t| t.cmp_put(key.0, pred.0, payload));
    }

    /// CMP-2: the largest index `c >= first` with an entry for `keys[c]`.
    pub fn find_latest(
        &self,
        keys: &[BlockKey],
        first: usize,
    ) -> Option<(usize, Arc<CompactionEntry>)> {
        (first..keys.len())
            .rev()
            .find_map(|c| self.get(&keys[c]).map(|e| (c, e)))
    }

    pub fn ram_entries(&self) -> usize {
        self.inner.lock().expect("compaction index lock").map.len()
    }

    /// TEL-3/TEL-8 fragment for `/internal/state`.
    pub fn state_json(&self) -> serde_json::Value {
        let s = &self.stats;
        let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let created = load(&s.created_summarize) + load(&s.created_truncate);
        serde_json::json!({
            "ram_entries": self.ram_entries(),
            "ram_capacity": self.cap,
            "created": {
                "summarize": load(&s.created_summarize),
                "truncate": load(&s.created_truncate),
            },
            "applied_requests": load(&s.applied_requests),
            "rejected": {
                "context_length_exceeded": load(&s.rejected_ctx6),
                "policy_reject": load(&s.rejected_policy),
            },
            "replaced_tokens": load(&s.replaced_tokens),
            "summary_secs_total": load(&s.summary_millis) as f64 / 1000.0,
            "summary_secs_mean": if created > 0 {
                load(&s.summary_millis) as f64 / 1000.0 / created as f64
            } else { 0.0 },
        })
    }
}

// SPDX-License-Identifier: AGPL-3.0-only

//! Garbage collector of the NVMe block store (NVM-10): TTL expiry,
//! orphan removal (predecessor missing → the chain can never be
//! restored through this record), and free-space pressure eviction in
//! ascending `last_access` order until the configured reserve is free.

use super::header::Kind;
use super::{BlockStore, now_secs};

/// What one GC run did — logged and exported to `/internal/state`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GcOutcome {
    pub expired: usize,
    pub orphaned: usize,
    pub pressure_evicted: usize,
    pub bytes_freed: u64,
}

impl BlockStore {
    /// Run one GC pass. Cheap when nothing qualifies; called every
    /// `gc_interval` by the store writer thread.
    pub fn run_gc(&self) -> GcOutcome {
        let now = now_secs();
        let ttl = self.cfg.session_ttl.as_secs();
        let mut out = GcOutcome::default();

        // Phase 1: TTL expiry.
        let expired: Vec<([u8; 16], Kind)> = {
            let idx = self.index.lock().expect("index lock");
            idx.iter()
                .filter(|(_, e)| now.saturating_sub(e.last_access) > ttl)
                .map(|(k, _)| *k)
                .collect()
        };
        for (key, kind) in expired {
            out.bytes_freed += self.delete_record(&key, kind);
            out.expired += 1;
        }

        // Phase 2: orphans — records whose predecessor KV block is gone.
        // Deleting an orphan can orphan its successors, so iterate to a
        // fixpoint (bounded by the entry count).
        loop {
            let orphans: Vec<([u8; 16], Kind)> = {
                let idx = self.index.lock().expect("index lock");
                idx.iter()
                    .filter(|(_, e)| e.pred != [0u8; 16] && !idx.contains(&e.pred, Kind::Kv))
                    .map(|(k, _)| *k)
                    .collect()
            };
            if orphans.is_empty() {
                break;
            }
            for (key, kind) in orphans {
                out.bytes_freed += self.delete_record(&key, kind);
                out.orphaned += 1;
            }
        }

        // Phase 3: free-space pressure (ascending last_access).
        let (free, total) = (self.free_space)(&self.cfg.root);
        if total != u64::MAX && total > 0 {
            let reserve = total / 100 * self.cfg.min_free_pct as u64;
            if free < reserve {
                let mut need = reserve - free;
                let mut victims: Vec<([u8; 16], Kind, u64, u64)> = {
                    let idx = self.index.lock().expect("index lock");
                    idx.iter()
                        .map(|(k, e)| (k.0, k.1, e.last_access, e.file_len))
                        .collect()
                };
                victims.sort_by_key(|&(_, _, access, _)| access);
                for (key, kind, _, file_len) in victims {
                    if need == 0 {
                        break;
                    }
                    out.bytes_freed += self.delete_record(&key, kind);
                    out.pressure_evicted += 1;
                    need = need.saturating_sub(file_len);
                }
            }
        }

        if out != GcOutcome::default() {
            tracing::info!(
                expired = out.expired,
                orphaned = out.orphaned,
                pressure = out.pressure_evicted,
                freed_mb = out.bytes_freed / (1 << 20),
                "block-store GC"
            );
        }
        out
    }

    /// Delete one record (file + index entry); returns its byte size.
    fn delete_record(&self, key: &[u8; 16], kind: Kind) -> u64 {
        let path = self.record_path(key, kind);
        let _ = std::fs::remove_file(&path);
        self.index
            .lock()
            .expect("index lock")
            .remove(key, kind)
            .map(|e| e.file_len)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{BlockStore, BlockStoreConfig, Kind};
    use std::path::Path;
    use std::time::Duration;

    fn cfg(root: &Path, ttl_secs: u64) -> BlockStoreConfig {
        BlockStoreConfig {
            root: root.to_path_buf(),
            model_hash: [7; 32],
            layout_hash: [9; 32],
            session_ttl: Duration::from_secs(ttl_secs),
            min_free_pct: 10,
            write_budget_per_day: None,
        }
    }

    fn key(n: u8) -> [u8; 16] {
        [n; 16]
    }

    #[test]
    fn ttl_expires_old_records_and_cascades_orphans() {
        let dir = tempfile::tempdir().unwrap();
        let store = BlockStore::open(cfg(dir.path(), 3600)).unwrap();
        // Chain 1 → 2 → 3.
        store.put(Kind::Kv, key(1), [0; 16], 16, &[1; 10]).unwrap();
        store.put(Kind::Kv, key(2), key(1), 16, &[2; 10]).unwrap();
        store.put(Kind::Kv, key(3), key(2), 16, &[3; 10]).unwrap();

        // Age ONLY the chain head below the TTL horizon.
        {
            let mut idx = store.index.lock().unwrap();
            let mut e = *idx.get(&key(1), Kind::Kv).unwrap();
            e.last_access = 1; // ancient
            idx.insert(key(1), e);
        }
        let out = store.run_gc();
        assert_eq!(out.expired, 1, "head expired by TTL");
        assert_eq!(out.orphaned, 2, "successors orphaned transitively");
        assert!(!store.contains(&key(2), Kind::Kv));
        assert!(!store.contains(&key(3), Kind::Kv));
    }

    #[test]
    fn recency_propagation_protects_whole_chain() {
        let dir = tempfile::tempdir().unwrap();
        let store = BlockStore::open(cfg(dir.path(), 3600)).unwrap();
        store.put(Kind::Kv, key(1), [0; 16], 16, &[1; 10]).unwrap();
        store.put(Kind::Kv, key(2), key(1), 16, &[2; 10]).unwrap();
        // Age both, then touch the chain tail — NVM-9 refreshes the head
        // too, so TTL must not take either.
        {
            let mut idx = store.index.lock().unwrap();
            for k in [key(1), key(2)] {
                let mut e = *idx.get(&k, Kind::Kv).unwrap();
                e.last_access = 1;
                idx.insert(k, e);
            }
        }
        store.touch(&key(2));
        let out = store.run_gc();
        assert_eq!(out.expired, 0);
        assert!(store.contains(&key(1), Kind::Kv));
    }

    #[test]
    fn ssm_and_tail_records_orphan_with_their_kv_chain() {
        let dir = tempfile::tempdir().unwrap();
        let store = BlockStore::open(cfg(dir.path(), 3600)).unwrap();
        store.put(Kind::Kv, key(1), [0; 16], 16, &[1; 10]).unwrap();
        store.put(Kind::Ssm, key(5), key(1), 32, &[5; 10]).unwrap();
        store.put(Kind::Tail, key(6), key(1), 4, &[6; 10]).unwrap();

        {
            let mut idx = store.index.lock().unwrap();
            let mut e = *idx.get(&key(1), Kind::Kv).unwrap();
            e.last_access = 1;
            idx.insert(key(1), e);
        }
        let out = store.run_gc();
        assert_eq!(out.expired, 1);
        assert_eq!(out.orphaned, 2);
        assert!(!store.contains(&key(5), Kind::Ssm));
        assert!(!store.contains(&key(6), Kind::Tail));
    }

    #[test]
    fn pressure_evicts_lru_first() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = BlockStore::open(cfg(dir.path(), u64::MAX / 4)).unwrap();
        store.put(Kind::Kv, key(1), [0; 16], 16, &[1; 10]).unwrap();
        store.put(Kind::Kv, key(2), [0; 16], 16, &[2; 10]).unwrap();
        // Make key(1) the LRU.
        {
            let mut idx = store.index.lock().unwrap();
            let mut e = *idx.get(&key(1), Kind::Kv).unwrap();
            e.last_access = 1;
            idx.insert(key(1), e);
        }
        // Simulated filesystem: 1 GiB total, nothing free → reserve unmet.
        store.set_free_space_probe(|_| (0, 1 << 30));
        let out = store.run_gc();
        assert!(out.pressure_evicted >= 1);
        assert!(
            !store.contains(&key(1), Kind::Kv),
            "LRU record evicted first"
        );
    }
}

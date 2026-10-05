// SPDX-License-Identifier: AGPL-3.0-only

//! Radix tree prefix cache for KV block reuse.
//!
//! Nodes are keyed by chained 128-bit block keys ([`crate::block_keys`]):
//! one node per sealed KV block, children addressed by the child block's
//! chain key. Because the key covers the entire causal prefix INCLUDING
//! media digests, key equality implies content equality — vision prompts
//! are cacheable without the former pad-token bypass (issue #58).
//!
//! Thread-safe via `Mutex<RadixTreeInner>`.

use parking_lot::Mutex;

use crate::block_keys::BlockKeysView;
use crate::prefix_cache::{EvictGuard, EvictedBlocks, PrefixCache, PrefixMatch};

mod inner;
mod snapshot;

#[cfg(test)]
mod tests;

use inner::RadixTreeInner;
use snapshot::SsmSnapshotIndex;

/// Thread-safe radix tree prefix cache.
///
/// SSM snapshots are stored in a separate `SsmSnapshotIndex`, decoupled from
/// tree node lifetime. This ensures snapshots survive KV cache eviction.
/// Lock ordering: acquire `inner` first (then release), then `snapshot_index`.
pub struct RadixTree {
    inner: Mutex<RadixTreeInner>,
    snapshot_index: Mutex<SsmSnapshotIndex>,
    /// KV-7: blocks the guard rejects stay in RAM (not yet persisted).
    evict_guard: Mutex<Option<EvictGuard>>,
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

impl RadixTree {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(RadixTreeInner::new()),
            snapshot_index: Mutex::new(SsmSnapshotIndex::new()),
            evict_guard: Mutex::new(None),
        }
    }
}

impl PrefixCache for RadixTree {
    fn lookup(&self, keys: &BlockKeysView<'_>, session_hash: u64) -> PrefixMatch {
        // Uncacheable chain (media without digests / uninitialised):
        // behave as a miss WITHOUT polluting hit/miss counters — these
        // sequences are outside the cache's universe by design.
        if !keys.cacheable() {
            return PrefixMatch::empty();
        }
        let block_size = keys.block_size();
        let tail_len = keys.total_tokens() % block_size;
        // Phase 1: walk tree (lock inner, then release)
        let (matched_blocks, matched_disk_block_ids, matched_tokens) = {
            let mut inner = self.inner.lock();
            let (blocks, disk, matched) =
                inner.walk(keys.full_keys(), keys.tail_key(), tail_len, block_size);
            if matched > 0 {
                inner.inc_refs(keys.full_keys(), matched / block_size);
                crate::prefix_cache::record_cache_hit(matched);
            } else {
                crate::prefix_cache::record_cache_miss();
            }
            (blocks, disk, matched)
        };
        // Phase 2: snapshot lookup (lock snapshot_index, inner NOT held)
        let (ssm_snapshot, ssm_snapshot_tokens) = if matched_tokens > 0 {
            let mut idx = self.snapshot_index.lock();
            match idx.lookup(keys, matched_tokens, session_hash) {
                Some((snap_id, tok_count)) => (Some(snap_id), tok_count),
                None => (None, 0),
            }
        } else {
            (None, 0)
        };
        // Filter disk_block_ids to MAX-free entries when HSS isn't in use, so
        // the caller can check `!matched_disk_block_ids.is_empty()` as the
        // HSS-engaged signal. When HSS *is* in use every entry should be a
        // valid disk_id (not MAX).
        let matched_disk_block_ids = if matched_disk_block_ids.iter().all(|&id| id == u32::MAX) {
            Vec::new()
        } else {
            matched_disk_block_ids
        };
        PrefixMatch {
            matched_blocks,
            matched_disk_block_ids,
            matched_tokens,
            ssm_snapshot,
            ssm_snapshot_tokens,
        }
    }

    fn peek_matched_tokens(&self, keys: &BlockKeysView<'_>) -> usize {
        if !keys.cacheable() {
            return 0;
        }
        let block_size = keys.block_size();
        let tail_len = keys.total_tokens() % block_size;
        self.inner
            .lock()
            .walk(keys.full_keys(), keys.tail_key(), tail_len, block_size)
            .2
    }

    fn insert(
        &self,
        keys: &BlockKeysView<'_>,
        block_table: &[u32],
        disk_block_ids: &[u32],
        matched_tokens: usize,
    ) -> Vec<u32> {
        if !keys.cacheable() {
            return Vec::new();
        }
        self.inner.lock().insert(
            keys.full_keys(),
            keys.tail_key(),
            block_table,
            disk_block_ids,
            keys.block_size(),
            matched_tokens,
        )
    }

    fn insert_with_snapshot(
        &self,
        keys: &BlockKeysView<'_>,
        block_table: &[u32],
        disk_block_ids: &[u32],
        snapshot_id: usize,
        session_hash: u64,
        matched_tokens: usize,
    ) -> (Option<usize>, Vec<u32>) {
        if !keys.cacheable() {
            return (None, Vec::new());
        }
        // Phase 1: insert tree nodes (lock inner, then release)
        let newly_acquired = self.inner.lock().insert(
            keys.full_keys(),
            keys.tail_key(),
            block_table,
            disk_block_ids,
            keys.block_size(),
            matched_tokens,
        );
        // Phase 2: register snapshot in index (lock snapshot_index, inner NOT held)
        let prefix_hash = keys.key_at(keys.total_tokens()).fold_u64();
        let mut idx = self.snapshot_index.lock();
        let displaced = idx.insert(prefix_hash, snapshot_id, session_hash, keys.total_tokens());
        (displaced, newly_acquired)
    }

    fn insert_intermediate_snapshot(
        &self,
        keys: &BlockKeysView<'_>,
        _block_table: &[u32],
        _disk_block_ids: &[u32],
        snapshot_id: usize,
        session_hash: u64,
        _matched_tokens: usize,
    ) -> Option<usize> {
        if !keys.cacheable() {
            return None;
        }
        // Intermediate snapshots go directly into the index with the correct
        // token boundary (keys.total_tokens()). Tree nodes are already
        // inserted by a prior `insert()` call, which handled the ref_count
        // bookkeeping.
        let prefix_hash = keys.key_at(keys.total_tokens()).fold_u64();
        let mut idx = self.snapshot_index.lock();
        idx.insert(prefix_hash, snapshot_id, session_hash, keys.total_tokens())
    }

    fn release(&self, keys: &BlockKeysView<'_>) {
        if !keys.cacheable() {
            return;
        }
        self.inner.lock().dec_refs(keys.full_keys());
    }

    fn set_evict_guard(&self, guard: Option<EvictGuard>) {
        *self.evict_guard.lock() = guard;
    }

    fn evict(&self, num_blocks: usize) -> EvictedBlocks {
        let guard = self.evict_guard.lock().clone();
        let (physical, disk) = self.inner.lock().evict(num_blocks, guard.as_ref());
        // Filter MAX sentinels out — the caller only needs disk_block_ids to
        // dec_disk_ref on, and MAX entries don't correspond to a live HSS ref.
        let disk_block_ids: Vec<u32> = disk.into_iter().filter(|&id| id != u32::MAX).collect();
        EvictedBlocks {
            physical,
            disk_block_ids,
        }
    }

    fn evict_snapshot_lru(&self) -> Option<usize> {
        self.snapshot_index.lock().evict_lru()
    }

    fn snapshot_count(&self) -> usize {
        self.snapshot_index.lock().len()
    }

    fn stats(&self) -> (usize, usize) {
        let inner = self.inner.lock();
        let entries = inner.num_entries();
        (entries, entries)
    }
}

// SPDX-License-Identifier: AGPL-3.0-only

//! Prefix caching trait for KV block reuse (SDD).
//!
//! When multiple requests share a common prompt prefix, previously-computed
//! KV cache blocks can be reused instead of re-running prefill. The cache
//! is indexed by chained 128-bit block keys ([`crate::block_keys`]) at
//! block granularity via a radix tree. The keys fold in the media digests
//! of vision prompts, so two prompts with the same token IDs but different
//! images never share a block (issue #58 — this replaces the former
//! vision-pad cache bypass).
//!
//! Two implementations:
//! - `NoPrefixCaching`: no-ops (zero overhead when disabled)
//! - `RadixTree` (see `crate::radix_tree`): full radix tree with LRU eviction

use std::sync::atomic::{AtomicU64, Ordering};

use crate::block_keys::BlockKeysView;

/// KV-7 eviction guard: `guard(key, is_tail)` returns true when the
/// block identified by `key` may leave RAM (i.e. it is persisted on the
/// NVMe tier, or no tier is configured). Installed by the server when
/// `--nvme-cache-dir` is active; absent → everything is evictable.
pub type EvictGuard = std::sync::Arc<dyn Fn(&[u8; 16], bool) -> bool + Send + Sync>;

// ── Global prefix cache counters (one RadixTree per server) ──

static CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
static CACHE_HIT_TOKENS: AtomicU64 = AtomicU64::new(0);

pub fn record_cache_hit(matched_tokens: usize) {
    CACHE_HITS.fetch_add(1, Ordering::Relaxed);
    CACHE_HIT_TOKENS.fetch_add(matched_tokens as u64, Ordering::Relaxed);
}

pub fn record_cache_miss() {
    CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
}

pub fn cache_hit_count() -> u64 {
    CACHE_HITS.load(Ordering::Relaxed)
}
pub fn cache_miss_count() -> u64 {
    CACHE_MISSES.load(Ordering::Relaxed)
}
pub fn cache_hit_tokens_total() -> u64 {
    CACHE_HIT_TOKENS.load(Ordering::Relaxed)
}

/// Result of evicting LRU cached blocks (Phase 6.1.e).
#[derive(Debug, Clone, Default)]
pub struct EvictedBlocks {
    /// Physical block indices freed (caller calls `PagedKvCache::free_block`).
    pub physical: Vec<u32>,
    /// Parallel disk-block IDs to release (caller calls
    /// `HighSpeedSwap::dec_disk_ref`). Empty when HSS isn't in use.
    pub disk_block_ids: Vec<u32>,
}

impl EvictedBlocks {
    pub fn is_empty(&self) -> bool {
        self.physical.is_empty()
    }

    pub fn len(&self) -> usize {
        self.physical.len()
    }
}

/// Result of looking up a key chain in the prefix cache.
#[derive(Debug, Clone)]
pub struct PrefixMatch {
    /// Physical KV cache block indices to reuse (in order).
    pub matched_blocks: Vec<u32>,
    /// `--high-speed-swap` disk-block IDs parallel to `matched_blocks`
    /// (Phase 6.1.e). Empty when HSS is not in use. Same length as
    /// `matched_blocks` when populated. Caller must `inc_disk_ref` each
    /// before treating them as live (the cache itself does not
    /// retain disk-side refs — see `RadixTree::lookup` for the bump).
    pub matched_disk_block_ids: Vec<u32>,
    /// Number of tokens matched. Block-aligned, except when the view's
    /// exact tail key also matched (then it equals the view length).
    pub matched_tokens: usize,
    /// SSM state snapshot ID at the deepest matched node (Marconi caching).
    /// When `Some`, the caller can restore SSM h_state + conv_state from
    /// this snapshot and skip SSM computation for the matched prefix.
    pub ssm_snapshot: Option<usize>,
    /// Number of tokens covered by `ssm_snapshot` (Marconi intermediate checkpoints).
    /// With leaf-only snapshots this equals `matched_tokens`. With intermediate
    /// checkpoints it may be less — the caller must recompute SSM state for
    /// tokens between `ssm_snapshot_tokens` and `matched_tokens`.
    pub ssm_snapshot_tokens: usize,
}

impl PrefixMatch {
    /// Empty match (no cached prefix found).
    pub fn empty() -> Self {
        Self {
            matched_blocks: Vec::new(),
            matched_disk_block_ids: Vec::new(),
            matched_tokens: 0,
            ssm_snapshot: None,
            ssm_snapshot_tokens: 0,
        }
    }

    /// Whether any prefix was matched.
    pub fn is_empty(&self) -> bool {
        self.matched_tokens == 0
    }
}

/// Trait for prefix caching strategies.
///
/// All methods take `&self` — implementations use interior mutability
/// (e.g., `Mutex`) for thread safety. This allows the prefix cache to be
/// shared between the model (prefill) and scheduler (free_sequence) without
/// requiring `&mut self`.
///
/// Every method that identifies blocks takes a [`BlockKeysView`] — the
/// chained-key view over the sequence's tokens. A view whose
/// `cacheable()` is false (media placeholders without digests, or an
/// uninitialised chain) MUST behave as a miss/no-op; `RadixTree`
/// enforces this internally so callers need no vision gates.
pub trait PrefixCache: Send + Sync {
    /// Whether this implementation is active (i.e., a real cache that
    /// actually inserts/holds refs). `NoPrefixCaching` returns false;
    /// `RadixTree` returns true. Callers use this to skip ref-bookkeeping
    /// that's only meaningful when the cache holds refs (e.g., the manual
    /// `kv_cache.inc_ref` in `cache_sequence` that pairs with eviction's
    /// `return_evicted_block`).
    fn is_active(&self) -> bool {
        true
    }

    /// Look up a key chain and return cached KV blocks for the longest
    /// matching prefix (block-aligned, plus an optional exact-tail hit).
    ///
    /// Increments ref_count on matched nodes so they survive eviction
    /// while the sequence is active. `session_hash` is used for SSM
    /// snapshot isolation (0 = legacy/no session tracking).
    fn lookup(&self, keys: &BlockKeysView<'_>, session_hash: u64) -> PrefixMatch;

    /// Read-only longest-prefix probe: number of tokens `lookup` would
    /// match, WITHOUT taking refs, touching LRU state, or counting a
    /// hit/miss. Used by the prefill tail-checkpoint split to detect
    /// conversation reuse before deciding to pay the extra pass.
    fn peek_matched_tokens(&self, _keys: &BlockKeysView<'_>) -> usize {
        0
    }

    /// Insert a completed prefill's blocks into the cache.
    ///
    /// `block_table[i]` is the physical block for tokens
    /// `[i*block_size .. (i+1)*block_size]` — identified by
    /// `keys.full_keys()[i]`.
    ///
    /// `disk_block_ids` parallels `block_table` for `--high-speed-swap`
    /// (Phase 6.1.e). Empty when HSS is not in use; same length as
    /// `block_table` when populated. The cache stores these alongside the
    /// physical block IDs and returns them in `EvictedBlocks` so the
    /// caller can `dec_disk_ref` the orchestrator's per-block refcount.
    ///
    /// **Disk-ref obligation (Issue #17 fix):** the returned vec lists every
    /// disk_block_id on which this insert call newly took an ownership ref
    /// (a node was created OR an existing node had its `disk_block_id`
    /// populated for the first time). The caller MUST `inc_disk_ref` each
    /// returned ID so the swap allocator's refcount matches the cache's
    /// reachability. Already-cached portions (matched-prefix entries, or
    /// blocks a prior intermediate insert already covered) are NOT in the
    /// returned vec — re-incing them would leak the cache's refcount.
    ///
    /// `matched_tokens` is the number of tokens the inserting sequence
    /// already acquired via `lookup()`'s `inc_refs` (those nodes' ref_count
    /// was bumped by walk and will be balanced by the eventual `release`).
    /// Tokens past this offset are "seq-owned" — the inserting sequence's
    /// eventual `release()` will decrement them — so `insert` must bump
    /// their ref_count to keep the cache's own reference alive after the
    /// release. See the release/lookup dance at the top of `radix_tree.rs`.
    fn insert(
        &self,
        keys: &BlockKeysView<'_>,
        block_table: &[u32],
        disk_block_ids: &[u32],
        matched_tokens: usize,
    ) -> Vec<u32>;

    /// Insert blocks with an SSM state snapshot registered in the snapshot index.
    ///
    /// The snapshot ID references a slot in an external `SsmSnapshotPool`.
    /// On future lookups matching this prefix, the snapshot ID is returned
    /// in `PrefixMatch::ssm_snapshot` so the caller can restore SSM state.
    /// `session_hash` tags the snapshot for session-scoped isolation.
    /// `matched_tokens` has the same semantics as in `insert`.
    /// Returns `(displaced_snapshot_id, newly_acquired_disk_ids)`. The
    /// disk-ref obligation matches `insert`: caller `inc_disk_ref`s each
    /// returned ID.
    #[allow(clippy::too_many_arguments)]
    fn insert_with_snapshot(
        &self,
        keys: &BlockKeysView<'_>,
        block_table: &[u32],
        disk_block_ids: &[u32],
        snapshot_id: usize,
        session_hash: u64,
        matched_tokens: usize,
    ) -> (Option<usize>, Vec<u32>);

    /// Insert an SSM snapshot at an intermediate token boundary.
    ///
    /// `keys` covers the token sequence up to and including the snapshot
    /// point. `block_table` contains the physical block indices for those
    /// tokens. `session_hash` tags the snapshot for session-scoped
    /// isolation. `matched_tokens` has the same semantics as in `insert`.
    /// Returns the displaced snapshot ID if an existing entry was overwritten.
    #[allow(clippy::too_many_arguments)]
    fn insert_intermediate_snapshot(
        &self,
        keys: &BlockKeysView<'_>,
        block_table: &[u32],
        disk_block_ids: &[u32],
        snapshot_id: usize,
        session_hash: u64,
        matched_tokens: usize,
    ) -> Option<usize>;

    /// Release ref_counts on blocks that were acquired via `lookup`.
    ///
    /// Called when a sequence finishes. Decrements ref_count on cache
    /// nodes along the key chain, making them eligible for eviction.
    fn release(&self, keys: &BlockKeysView<'_>);

    /// Read-only probe of the deepest SSM snapshot at or below
    /// `max_tokens` for this chain (no LRU/hit-count mutation). Returns
    /// `(snapshot_id, token_count)`. Used by the NVMe persist path to
    /// export the deepest aligned checkpoint. Default: none.
    fn peek_snapshot(
        &self,
        _keys: &BlockKeysView<'_>,
        _max_tokens: usize,
        _session_hash: u64,
    ) -> Option<(usize, usize)> {
        None
    }

    /// Install (or clear) the KV-7 eviction guard. Default: no-op —
    /// `NoPrefixCaching` has nothing to evict.
    fn set_evict_guard(&self, _guard: Option<EvictGuard>) {}

    /// Evict up to `num_blocks` cached blocks, returning their physical
    /// indices and parallel disk-block IDs (Phase 6.1.e).
    ///
    /// Picks LRU zero-ref leaf nodes. Returns fewer than requested if not
    /// enough evictable blocks exist. The caller is responsible for
    /// `dec_disk_ref`-ing every entry in `disk_block_ids` (releasing the
    /// cache's HSS-side refcount). When HSS isn't in use the `disk_block_ids`
    /// vec is empty.
    fn evict(&self, num_blocks: usize) -> EvictedBlocks;

    /// Evict the least-recently-used SSM snapshot from the snapshot index.
    /// Returns the snapshot ID so the caller can free it in `SsmSnapshotPool`.
    fn evict_snapshot_lru(&self) -> Option<usize>;

    /// Number of SSM snapshots currently stored in the snapshot index.
    fn snapshot_count(&self) -> usize;

    /// (entries, cached_blocks) for logging.
    fn stats(&self) -> (usize, usize);

    /// Physical block the cache maps for each full-block key of `keys`, in
    /// chain order, ending at the first key without a node. A sequence may
    /// hand the cache a KV ref only on blocks the cache actually maps: an
    /// `insert` keeps the block of a node that already existed, so the
    /// sequence's own recomputation of that content (e.g. the blocks past
    /// a Marconi SSM boundary hit) stays sequence-owned and must be freed
    /// with the sequence instead of ref'd on the cache's behalf.
    /// Default: nothing mapped (no cache).
    fn mapped_blocks(&self, _keys: &BlockKeysView<'_>) -> Vec<u32> {
        Vec::new()
    }
}

/// No-op prefix cache (zero overhead when disabled).
pub struct NoPrefixCaching;

impl PrefixCache for NoPrefixCaching {
    fn is_active(&self) -> bool {
        false
    }

    fn lookup(&self, _keys: &BlockKeysView<'_>, _session_hash: u64) -> PrefixMatch {
        PrefixMatch::empty()
    }

    fn insert(
        &self,
        _keys: &BlockKeysView<'_>,
        _block_table: &[u32],
        _disk_block_ids: &[u32],
        _matched_tokens: usize,
    ) -> Vec<u32> {
        Vec::new()
    }

    fn insert_with_snapshot(
        &self,
        _keys: &BlockKeysView<'_>,
        _block_table: &[u32],
        _disk_block_ids: &[u32],
        _snapshot_id: usize,
        _session_hash: u64,
        _matched_tokens: usize,
    ) -> (Option<usize>, Vec<u32>) {
        (None, Vec::new())
    }

    fn insert_intermediate_snapshot(
        &self,
        _keys: &BlockKeysView<'_>,
        _block_table: &[u32],
        _disk_block_ids: &[u32],
        _snapshot_id: usize,
        _session_hash: u64,
        _matched_tokens: usize,
    ) -> Option<usize> {
        None
    }

    fn release(&self, _keys: &BlockKeysView<'_>) {}

    fn evict(&self, _num_blocks: usize) -> EvictedBlocks {
        EvictedBlocks::default()
    }

    fn evict_snapshot_lru(&self) -> Option<usize> {
        None
    }

    fn snapshot_count(&self) -> usize {
        0
    }

    fn stats(&self) -> (usize, usize) {
        (0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_keys::{BlockKeyChain, seed};

    #[test]
    fn test_prefix_match_empty() {
        let m = PrefixMatch::empty();
        assert!(m.is_empty());
        assert_eq!(m.matched_tokens, 0);
        assert!(m.matched_blocks.is_empty());
    }

    #[test]
    fn test_no_prefix_caching_is_noop() {
        let cache = NoPrefixCaching;
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let chain = BlockKeyChain::new(seed(b"m", &[0u8; 32]), 4, &tokens, 0, &[]);
        let block_table = vec![0, 1];
        let disk_block_ids: Vec<u32> = vec![];

        let view = chain.view(&tokens, 8);
        let m = cache.lookup(&view, 0);
        assert!(m.is_empty());

        // These should not panic
        let new_acq = cache.insert(&view, &block_table, &disk_block_ids, 0);
        assert!(new_acq.is_empty());
        cache.release(&view);

        let evicted = cache.evict(10);
        assert!(evicted.is_empty());

        assert_eq!(cache.evict_snapshot_lru(), None);
        assert_eq!(cache.snapshot_count(), 0);

        assert_eq!(cache.stats(), (0, 0));
    }
}

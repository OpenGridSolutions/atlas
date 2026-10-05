// SPDX-License-Identifier: AGPL-3.0-only

//! Single-threaded radix-tree state — wrapped behind a `Mutex` in the
//! [`super::RadixTree`] public API. Nodes are keyed by their chained
//! 128-bit [`BlockKey`] (KV-3): the key is a function of the ENTIRE
//! causal prefix *including media digests*, so key equality implies the
//! block's content is reusable — two prompts with identical token IDs
//! but different images diverge at the first media block and never share
//! a node. Each node holds one physical KV block index.

use std::collections::HashMap;

use crate::block_keys::BlockKey;

type NodeId = usize;

/// Cap on exact-tail entries stored per node. Tails are per-sequence
/// endings (one per released conversation turn); a handful per node
/// covers the realistic reuse window, and the cap bounds the growth of
/// un-refcounted partial blocks (the pre-key code had a single slot
/// whose overwrite had the same dropped-block semantics).
const MAX_TAILS_PER_NODE: usize = 8;

pub(super) struct RadixNode {
    /// Children keyed by the child block's chain key.
    children: HashMap<BlockKey, NodeId>,
    /// Physical KV cache block index stored at this node.
    block_idx: u32,
    /// `--high-speed-swap` disk-block ID (Phase 6.1.e). `u32::MAX` when HSS
    /// is not in use. The cache holds a refcount on this disk_id (bumped
    /// at insert time, dropped at evict time) parallel to its physical-
    /// block ref bookkeeping.
    disk_block_id: u32,
    /// Number of active sequences referencing this node.
    ref_count: u32,
    /// Monotonic access counter for LRU eviction.
    last_access: u64,
    /// Parent node (None for root).
    parent: Option<NodeId>,
    /// Chain key that leads from parent to this node (for cleanup).
    parent_key: Option<BlockKey>,
    /// Exact-tail entries: the trailing sub-block of a cached sequence,
    /// keyed by its tail key (`H(k_i ‖ tail_tokens ‖ tail_digests)`).
    /// Value: `(physical_block, disk_block_id)`. Matched only exactly —
    /// key equality implies identical tail tokens AND media. Insertion
    /// order preserved for the size cap (oldest dropped first).
    tails: Vec<(BlockKey, u32, u32)>,
}

/// Inner state protected by mutex.
pub(super) struct RadixTreeInner {
    nodes: Vec<RadixNode>,
    /// Indices of deleted nodes available for reuse.
    free_nodes: Vec<NodeId>,
    root: NodeId,
    access_counter: u64,
}

impl RadixTreeInner {
    pub(super) fn new() -> Self {
        let root = RadixNode {
            children: HashMap::new(),
            block_idx: u32::MAX,     // sentinel — root has no block
            disk_block_id: u32::MAX, // sentinel — root has no disk slot either
            ref_count: 0,
            last_access: 0,
            parent: None,
            parent_key: None,
            tails: Vec::new(),
        };
        Self {
            nodes: vec![root],
            free_nodes: Vec::new(),
            root: 0,
            access_counter: 0,
        }
    }

    pub(super) fn next_access(&mut self) -> u64 {
        self.access_counter += 1;
        self.access_counter
    }

    pub(super) fn alloc_node(&mut self, node: RadixNode) -> NodeId {
        if let Some(id) = self.free_nodes.pop() {
            self.nodes[id] = node;
            id
        } else {
            let id = self.nodes.len();
            self.nodes.push(node);
            id
        }
    }

    /// Walk the tree along the chain keys. Returns
    /// (matched_blocks, matched_disk_block_ids, matched_tokens). The
    /// disk-id vec parallels matched_blocks; entries are `u32::MAX` on
    /// nodes that were inserted before HSS engaged (caller's
    /// responsibility to filter / treat MAX as "no disk-side ref to
    /// bump"). `tail_key`/`tail_len` describe the view's trailing
    /// sub-block (if any): after the deepest full-block match, an exact
    /// tail entry on the current node extends the match to the full view.
    pub(super) fn walk(
        &self,
        keys: &[BlockKey],
        tail_key: Option<BlockKey>,
        tail_len: usize,
        block_size: usize,
    ) -> (Vec<u32>, Vec<u32>, usize) {
        let mut current = self.root;
        let mut matched_blocks = Vec::new();
        let mut matched_disk = Vec::new();
        let mut matched_tokens = 0;
        let mut complete = true;

        for key in keys {
            match self.nodes[current].children.get(key) {
                Some(&child) if self.nodes[child].ref_count > 0 => {
                    // Key chain matches AND block is still live — safe to reuse.
                    matched_blocks.push(self.nodes[child].block_idx);
                    matched_disk.push(self.nodes[child].disk_block_id);
                    matched_tokens += block_size;
                    current = child;
                }
                _ => {
                    complete = false;
                    break;
                }
            }
        }

        // Exact-tail matching: only when every full block matched does the
        // view's tail key apply at `current`.
        if complete
            && let Some(tk) = tail_key
            && let Some(&(_, block, disk)) = self.nodes[current].tails.iter().find(|t| t.0 == tk)
        {
            matched_blocks.push(block);
            matched_disk.push(disk);
            matched_tokens += tail_len;
        }

        (matched_blocks, matched_disk, matched_tokens)
    }

    /// Increment ref_count on all full-block nodes along the matched path.
    pub(super) fn inc_refs(&mut self, keys: &[BlockKey], num_blocks: usize) {
        let access = self.next_access();
        let mut current = self.root;

        for key in keys.iter().take(num_blocks) {
            if let Some(&child) = self.nodes[current].children.get(key) {
                self.nodes[child].ref_count += 1;
                self.nodes[child].last_access = access;
                current = child;
            } else {
                break;
            }
        }
    }

    /// Decrement ref_count on all full-block nodes along the matched path.
    pub(super) fn dec_refs(&mut self, keys: &[BlockKey]) {
        let mut current = self.root;

        for key in keys {
            if let Some(&child) = self.nodes[current].children.get(key) {
                if self.nodes[child].ref_count > 0 {
                    self.nodes[child].ref_count -= 1;
                }
                current = child;
            } else {
                break;
            }
        }
    }

    /// Insert blocks into the tree. Skips blocks that already exist.
    /// Snapshot storage is handled externally by SsmSnapshotIndex.
    ///
    /// `keys` are the chain keys of the sequence's full blocks;
    /// `tail_key` the exact key of the trailing sub-block (when
    /// `block_table` has one more entry than `keys` covers).
    ///
    /// `matched_tokens` is the prefix length the inserting sequence already
    /// acquired via `lookup()`'s `inc_refs` (those nodes' ref_count was
    /// bumped by walk and will be balanced by the eventual `release`). For
    /// blocks past that offset the inserting sequence holds no walk-ref but
    /// `release` will still dec_ref them — so we pre-bump here to keep the
    /// cache's baseline ref (=1) alive across the release. Without this the
    /// very first sequence to cache a prompt immediately renders its own
    /// cache entry un-findable on exit (`walk` requires `ref_count > 0`).
    pub(super) fn insert(
        &mut self,
        keys: &[BlockKey],
        tail_key: Option<BlockKey>,
        block_table: &[u32],
        disk_block_ids: &[u32],
        block_size: usize,
        matched_tokens: usize,
    ) -> Vec<u32> {
        let access = self.next_access();
        let mut current = self.root;
        let num_blocks = keys.len().min(block_table.len());
        // disk_block_ids may be empty (HSS off) or parallel to block_table.
        // When empty, every node's disk_block_id stays at u32::MAX sentinel.
        let hss_active = !disk_block_ids.is_empty();
        debug_assert!(
            !hss_active || disk_block_ids.len() == block_table.len(),
            "disk_block_ids length mismatch: {} vs {}",
            disk_block_ids.len(),
            block_table.len(),
        );

        // Issue #17: collect disk_ids on which the cache newly takes an
        // ownership ref. Caller `inc_disk_ref`s these so the swap allocator's
        // refcount tracks the cache's reachability. Two acquisition events:
        //   1. New node creation with a real disk_id.
        //   2. Existing node, MAX → real disk_id (first-time HSS population).
        // Re-insertion of an already-cached (node, disk_id) pair is NOT an
        // acquisition — the cache's ref already covers it.
        let mut newly_acquired: Vec<u32> = Vec::new();

        for (i, key) in keys.iter().take(num_blocks).enumerate() {
            let token_start = i * block_size;
            let is_seq_owned = token_start >= matched_tokens;
            let disk_id = if hss_active {
                disk_block_ids[i]
            } else {
                u32::MAX
            };

            if let Some(&child) = self.nodes[current].children.get(key) {
                // Node exists — key equality means identical content, so
                // only refresh access time and ensure the cache's ref is
                // still held (ref_count >= 1).
                self.nodes[child].last_access = access;
                if self.nodes[child].ref_count == 0 {
                    self.nodes[child].ref_count = 1; // restore cache's ref
                }
                if is_seq_owned {
                    self.nodes[child].ref_count += 1;
                }
                // First-time HSS population on a pre-existing node: stash
                // the disk_id. (This happens when the cache was built
                // pre-HSS and a later HSS-enabled sequence touches the
                // same prefix — defensive, shouldn't fire under normal
                // operation since cache_blocks_per_seq is set at startup.)
                if hss_active && self.nodes[child].disk_block_id == u32::MAX && disk_id != u32::MAX
                {
                    self.nodes[child].disk_block_id = disk_id;
                    newly_acquired.push(disk_id);
                }
                current = child;
            } else {
                let node = RadixNode {
                    children: HashMap::new(),
                    block_idx: block_table[i],
                    disk_block_id: disk_id,
                    ref_count: if is_seq_owned { 2 } else { 1 },
                    last_access: access,
                    parent: Some(current),
                    parent_key: Some(*key),
                    tails: Vec::new(),
                };
                let child_id = self.alloc_node(node);
                self.nodes[current].children.insert(*key, child_id);
                if hss_active && disk_id != u32::MAX {
                    newly_acquired.push(disk_id);
                }
                current = child_id;
            }
        }

        // Trailing sub-block: store as an exact-tail entry on the deepest
        // node reached (never on the root — a tail without any full block
        // in front has no stable anchor, matching the old partial_suffix
        // root guard).
        if let Some(tk) = tail_key
            && block_table.len() > num_blocks
            && current != self.root
        {
            let tail_block = block_table[num_blocks];
            let tail_disk = if hss_active && disk_block_ids.len() > num_blocks {
                disk_block_ids[num_blocks]
            } else {
                u32::MAX
            };
            let node = &mut self.nodes[current];
            if let Some(entry) = node.tails.iter_mut().find(|t| t.0 == tk) {
                // Same key = same content; refresh the stored block ids.
                entry.1 = tail_block;
                entry.2 = tail_disk;
            } else {
                if node.tails.len() >= MAX_TAILS_PER_NODE {
                    // Drop the oldest tail. Its physical block stays with
                    // whatever refcount the kv_cache holds — identical
                    // semantics to the pre-key single-slot overwrite.
                    node.tails.remove(0);
                }
                node.tails.push((tk, tail_block, tail_disk));
                if hss_active && tail_disk != u32::MAX {
                    newly_acquired.push(tail_disk);
                }
            }
        }

        newly_acquired
    }

    /// Evict up to `num_blocks` LRU zero-ref leaf nodes.
    /// Returns physical block indices that were freed plus parallel
    /// disk-block IDs (Phase 6.1.e). When HSS isn't in use, every disk_id
    /// in the result is `u32::MAX` and the caller should ignore them; the
    /// public-trait wrapper filters those out into the returned
    /// `EvictedBlocks::disk_block_ids`.
    pub(super) fn evict(&mut self, num_blocks: usize) -> (Vec<u32>, Vec<u32>) {
        let mut freed_phys = Vec::new();
        let mut freed_disk = Vec::new();
        if num_blocks == 0 {
            return (freed_phys, freed_disk);
        }

        loop {
            if freed_phys.len() >= num_blocks {
                break;
            }

            let mut best: Option<(NodeId, u64)> = None;
            for (id, node) in self.nodes.iter().enumerate() {
                if id == self.root {
                    continue;
                }
                if node.ref_count <= 1 && node.children.is_empty() && node.block_idx != u32::MAX {
                    match best {
                        None => best = Some((id, node.last_access)),
                        Some((_, best_access)) if node.last_access < best_access => {
                            best = Some((id, node.last_access));
                        }
                        _ => {}
                    }
                }
            }

            match best {
                Some((node_id, _)) => {
                    let block = self.nodes[node_id].block_idx;
                    let disk = self.nodes[node_id].disk_block_id;
                    freed_phys.push(block);
                    freed_disk.push(disk);

                    for (_, tail_block, tail_disk) in std::mem::take(&mut self.nodes[node_id].tails)
                    {
                        freed_phys.push(tail_block);
                        freed_disk.push(tail_disk);
                    }

                    if let Some(parent_id) = self.nodes[node_id].parent
                        && let Some(key) = self.nodes[node_id].parent_key
                    {
                        self.nodes[parent_id].children.remove(&key);
                    }

                    self.nodes[node_id].block_idx = u32::MAX;
                    self.nodes[node_id].disk_block_id = u32::MAX;
                    self.nodes[node_id].children.clear();
                    self.nodes[node_id].parent = None;
                    self.nodes[node_id].parent_key = None;
                    self.nodes[node_id].tails = Vec::new();
                    self.free_nodes.push(node_id);
                }
                None => break,
            }
        }

        (freed_phys, freed_disk)
    }

    pub(super) fn num_entries(&self) -> usize {
        // Count non-root, non-deleted nodes.
        self.nodes
            .iter()
            .enumerate()
            .filter(|(id, n)| *id != self.root && n.block_idx != u32::MAX)
            .count()
    }
}

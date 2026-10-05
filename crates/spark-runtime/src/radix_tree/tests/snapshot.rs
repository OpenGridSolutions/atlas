// SPDX-License-Identifier: AGPL-3.0-only

//! Snapshot-side tests: intermediate snapshots, exact-tail matching,
//! and the standalone snapshot index LRU/session/overwrite behaviours.

use crate::block_keys::{BlockKeyChain, seed};
use crate::prefix_cache::PrefixCache;
use crate::radix_tree::RadixTree;

use super::super::snapshot::SsmSnapshotIndex;

const BS: usize = 16;

fn chain(tokens: &[u32]) -> BlockKeyChain {
    BlockKeyChain::new(seed(b"test-model", &[0u8; 32]), BS, tokens, 0, &[])
}

#[test]
fn test_insert_without_snapshot() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();
    let c = chain(&tokens);

    tree.insert(&c.view(&tokens, 16), &[10], &[], 0);
    let m = tree.lookup(&c.view(&tokens, 16), 0);
    assert_eq!(m.ssm_snapshot, None);
    assert_eq!(m.ssm_snapshot_tokens, 0);
    tree.release(&c.view(&tokens, 16));
}

#[test]
fn test_intermediate_snapshot_on_partial_match() {
    let tree = RadixTree::new();

    // Insert 4-block sequence
    let tokens: Vec<u32> = (0..64).collect();
    let c = chain(&tokens);
    tree.insert(&c.view(&tokens, 64), &[10, 20, 30, 40], &[], 0);

    // Attach intermediate snapshot at block 2 (token 32)
    tree.insert_intermediate_snapshot(&c.view(&tokens, 32), &[10, 20], &[], 50, 0, 0);

    // Lookup all 4 blocks — should return intermediate snapshot at block 2
    let m = tree.lookup(&c.view(&tokens, 64), 0);
    assert_eq!(m.matched_tokens, 64);
    assert_eq!(m.ssm_snapshot, Some(50));
    assert_eq!(m.ssm_snapshot_tokens, 32);
    tree.release(&c.view(&tokens, 64));
}

#[test]
fn test_intermediate_snapshot_deepest_wins() {
    let tree = RadixTree::new();

    // Insert 4-block sequence with leaf snapshot
    let tokens: Vec<u32> = (0..64).collect();
    let c = chain(&tokens);
    tree.insert_with_snapshot(&c.view(&tokens, 64), &[10, 20, 30, 40], &[], 99, 0, 0);

    // Attach intermediate snapshot at block 2 (token 32)
    tree.insert_intermediate_snapshot(&c.view(&tokens, 32), &[10, 20], &[], 50, 0, 0);

    // Lookup all 4 blocks — leaf snapshot (deeper) wins
    let m = tree.lookup(&c.view(&tokens, 64), 0);
    assert_eq!(m.matched_tokens, 64);
    assert_eq!(m.ssm_snapshot, Some(99));
    assert_eq!(m.ssm_snapshot_tokens, 64);
    tree.release(&c.view(&tokens, 64));
}

#[test]
fn test_intermediate_snapshot_partial_prefix_hit() {
    let tree = RadixTree::new();

    // Insert 4-block sequence
    let tokens: Vec<u32> = (0..64).collect();
    let c = chain(&tokens);
    tree.insert(&c.view(&tokens, 64), &[10, 20, 30, 40], &[], 0);

    // Attach intermediate snapshot at block 2 (token 32)
    tree.insert_intermediate_snapshot(&c.view(&tokens, 32), &[10, 20], &[], 50, 0, 0);

    // New request shares first 48 tokens, diverges at block 4
    let mut tokens_new: Vec<u32> = (0..48).collect();
    tokens_new.extend(200..216);
    let cn = chain(&tokens_new);
    let m = tree.lookup(&cn.view(&tokens_new, 64), 0);
    // Matches 3 blocks (48 tokens), intermediate snapshot at block 2
    assert_eq!(m.matched_tokens, 48);
    assert_eq!(m.ssm_snapshot, Some(50));
    assert_eq!(m.ssm_snapshot_tokens, 32);
    tree.release(&cn.view(&tokens_new, 64));
}

#[test]
fn test_intermediate_snapshot_survives_tree_eviction() {
    let tree = RadixTree::new();

    // Insert 2-block sequence with intermediate snapshot on block 1
    let tokens: Vec<u32> = (0..32).collect();
    let c = chain(&tokens);
    tree.insert(&c.view(&tokens, 32), &[10, 20], &[], 0);
    tree.release(&c.view(&tokens, 32)); // inserting seq exits → nodes evictable

    tree.insert_intermediate_snapshot(&c.view(&tokens, 16), &[10], &[], 50, 0, 0);

    // Evict both tree nodes — snapshot survives in index
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![20]);
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![10]);

    // Snapshot still in index (decoupled from tree)
    assert_eq!(tree.snapshot_count(), 1);
    let snap = tree.evict_snapshot_lru();
    assert_eq!(snap, Some(50));
}

// ── Exact-tail tests (sub-block trailing remainder) ──

#[test]
fn test_tail_insert_and_lookup() {
    let tree = RadixTree::new();
    // 20 tokens = 1 full block (16) + 4 tail
    let tokens: Vec<u32> = (0..20).collect();
    let block_table = vec![10, 20]; // block for full + block for tail
    let c = chain(&tokens);

    tree.insert(&c.view(&tokens, 20), &block_table, &[], 0);
    let m = tree.lookup(&c.view(&tokens, 20), 0);

    // Should match all 20 tokens (16 full + 4 exact tail)
    assert_eq!(m.matched_tokens, 20);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&c.view(&tokens, 20));
}

#[test]
fn test_tail_no_match_different_suffix() {
    let tree = RadixTree::new();
    // Insert 20 tokens
    let tokens_a: Vec<u32> = (0..20).collect();
    let ca = chain(&tokens_a);
    tree.insert(&ca.view(&tokens_a, 20), &[10, 20], &[], 0);

    // Lookup 20 tokens with different suffix (same first 16, different last 4)
    let mut tokens_b: Vec<u32> = (0..16).collect();
    tokens_b.extend(100..104);
    let cb = chain(&tokens_b);
    let m = tree.lookup(&cb.view(&tokens_b, 20), 0);

    // Should match only 16 full-block tokens (tail key differs)
    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
    tree.release(&cb.view(&tokens_b, 20));
}

#[test]
fn test_tail_not_matched_for_full_block_request() {
    let tree = RadixTree::new();
    // Insert 20 tokens (1 full + 4 tail)
    let tokens: Vec<u32> = (0..20).collect();
    let c = chain(&tokens);
    tree.insert(&c.view(&tokens, 20), &[10, 20], &[], 0);

    // Lookup 32 tokens — 2 full blocks in request; no tail in the view
    // (32 % 16 == 0), so the tail entry is skipped.
    let tokens_32: Vec<u32> = (0..32).collect();
    let c32 = chain(&tokens_32);
    let m = tree.lookup(&c32.view(&tokens_32, 32), 0);

    // Only first full block matches (second block [16..32] has no matching tree node)
    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
    tree.release(&c32.view(&tokens_32, 32));
}

#[test]
fn test_tail_eviction_frees_both_blocks() {
    let tree = RadixTree::new();
    // Insert 20 tokens (1 full block + 4 tail) + release inserting seq
    let tokens: Vec<u32> = (0..20).collect();
    let c = chain(&tokens);
    tree.insert(&c.view(&tokens, 20), &[10, 20], &[], 0);
    tree.release(&c.view(&tokens, 20));

    // Evict 1 — should free block 10 (full) AND block 20 (tail entry)
    let evicted = tree.evict(1);
    assert!(evicted.physical.contains(&10));
    assert!(evicted.physical.contains(&20));
}

#[test]
fn test_tail_coexists_with_extension() {
    let tree = RadixTree::new();
    // Insert 20 tokens (1 full + 4 tail)
    let tokens_20: Vec<u32> = (0..20).collect();
    let c20 = chain(&tokens_20);
    tree.insert(&c20.view(&tokens_20, 20), &[10, 20], &[], 0);

    // Insert 32 tokens (2 full blocks, extends past the tail). Tail keys
    // are content-addressed, so the extension does NOT clear the tail —
    // both the 20-token ending and the 32-token path stay reachable.
    let tokens_32: Vec<u32> = (0..32).collect();
    let c32 = chain(&tokens_32);
    tree.insert(&c32.view(&tokens_32, 32), &[10, 30], &[], 0);

    let m = tree.lookup(&c20.view(&tokens_20, 20), 0);
    assert_eq!(m.matched_tokens, 20);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&c20.view(&tokens_20, 20));

    let m = tree.lookup(&c32.view(&tokens_32, 32), 0);
    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.matched_blocks, vec![10, 30]);
    tree.release(&c32.view(&tokens_32, 32));
}

#[test]
fn test_tail_multi_block_prefix() {
    let tree = RadixTree::new();
    // 396 tokens = 24 full blocks + 12 tail
    let tokens: Vec<u32> = (0..396).collect();
    let block_table: Vec<u32> = (0..25).collect();
    let c = chain(&tokens);
    // block_table[24] = tail block

    tree.insert(&c.view(&tokens, 396), &block_table, &[], 0);
    let m = tree.lookup(&c.view(&tokens, 396), 0);

    assert_eq!(m.matched_tokens, 396);
    assert_eq!(m.matched_blocks.len(), 25);
    tree.release(&c.view(&tokens, 396));
}

#[test]
fn test_tail_match_is_exact_not_prefix() {
    // Chained keys are all-or-nothing: a stored 15-token tail can no
    // longer serve a 6-token prefix of itself (the pre-key tree did
    // prefix matching here). The match ends at the last block boundary;
    // sub-block prefix sharing needs copy-on-write (KV-9, later stage).
    let tree = RadixTree::new();
    // Insert 31 tokens (1 full block + 15 tail) — simulates prompt+generation
    let tokens_31: Vec<u32> = (0..31).collect();
    let c31 = chain(&tokens_31);
    tree.insert(&c31.view(&tokens_31, 31), &[10, 20], &[], 0);

    // Lookup 22 tokens (1 full block + 6 tail) — repeat of prompt only
    let tokens_22: Vec<u32> = (0..22).collect();
    let c22 = chain(&tokens_22);
    let m = tree.lookup(&c22.view(&tokens_22, 22), 0);

    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
    tree.release(&c22.view(&tokens_22, 22));
}

#[test]
fn test_full_block_not_matched_by_sub_block_request() {
    // Same exact-only property for stored FULL blocks: a request whose
    // remainder is a strict prefix of a cached full block gets no
    // sub-block hit (pre-key behavior matched here via key-prefix scan).
    let tree = RadixTree::new();
    let tokens_35: Vec<u32> = (0..35).collect();
    let c35 = chain(&tokens_35);
    tree.insert(&c35.view(&tokens_35, 35), &[10, 20, 30], &[], 0);

    let tokens_22: Vec<u32> = (0..22).collect();
    let c22 = chain(&tokens_22);
    let m = tree.lookup(&c22.view(&tokens_22, 22), 0);

    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
    tree.release(&c22.view(&tokens_22, 22));
}

#[test]
fn test_tail_sub_block_only() {
    let tree = RadixTree::new();
    // Only 10 tokens — no full blocks, tail not stored (no anchor node)
    let tokens: Vec<u32> = (0..10).collect();
    let c = chain(&tokens);
    tree.insert(&c.view(&tokens, 10), &[42], &[], 0);

    // No full blocks → nothing cached or matched
    assert_eq!(tree.stats(), (0, 0));
    let m = tree.lookup(&c.view(&tokens, 10), 0);
    assert_eq!(m.matched_tokens, 0);
}

// ── SsmSnapshotIndex tests ──

#[test]
fn test_snapshot_index_insert_lookup_roundtrip() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..32).collect();
    let c = chain(&tokens);
    let prefix_hash = c.view(&tokens, 32).key_at(32).fold_u64();

    assert!(idx.insert(prefix_hash, 42, 100, 32).is_none());
    let result = idx.lookup(&c.view(&tokens, 32), 32, 100);
    assert_eq!(result, Some((42, 32)));
}

#[test]
fn test_snapshot_index_lru_eviction() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens_a: Vec<u32> = (0..16).collect();
    let tokens_b: Vec<u32> = (100..116).collect();
    let ca = chain(&tokens_a);
    let cb = chain(&tokens_b);
    let ha = ca.view(&tokens_a, 16).key_at(16).fold_u64();
    let hb = cb.view(&tokens_b, 16).key_at(16).fold_u64();

    idx.insert(ha, 1, 0, 16); // older
    idx.insert(hb, 2, 0, 16); // newer

    // LRU eviction should evict snapshot 1 (older)
    let evicted = idx.evict_lru();
    assert_eq!(evicted, Some(1));
    assert_eq!(idx.len(), 1);

    // Only snapshot 2 remains
    let evicted = idx.evict_lru();
    assert_eq!(evicted, Some(2));
    assert_eq!(idx.len(), 0);

    // Empty
    assert_eq!(idx.evict_lru(), None);
}

#[test]
fn test_snapshot_index_session_isolation() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..16).collect();
    let c = chain(&tokens);
    let prefix_hash = c.view(&tokens, 16).key_at(16).fold_u64();

    // Insert snapshot for session 100
    idx.insert(prefix_hash, 42, 100, 16);

    // Lookup from session 200 — should NOT match (different session)
    let result = idx.lookup(&c.view(&tokens, 16), 16, 200);
    assert_eq!(result, None);

    // Lookup from session 100 — should match
    let result = idx.lookup(&c.view(&tokens, 16), 16, 100);
    assert_eq!(result, Some((42, 16)));

    // Lookup with session_hash=0 (legacy) — matches any session
    let result = idx.lookup(&c.view(&tokens, 16), 16, 0);
    assert_eq!(result, Some((42, 16)));
}

#[test]
fn test_snapshot_index_overwrite_existing() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..16).collect();
    let c = chain(&tokens);
    let prefix_hash = c.view(&tokens, 16).key_at(16).fold_u64();

    // Insert first
    assert!(idx.insert(prefix_hash, 5, 0, 16).is_none());
    assert_eq!(idx.len(), 1);

    // Overwrite same prefix_hash — returns old snapshot_id
    let old = idx.insert(prefix_hash, 8, 0, 16);
    assert_eq!(old, Some(5));
    assert_eq!(idx.len(), 1); // still 1 entry, not 2

    // Lookup returns new value
    let result = idx.lookup(&c.view(&tokens, 16), 16, 0);
    assert_eq!(result, Some((8, 16)));
}

#[test]
fn test_snapshot_index_deepest_match() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..64).collect();
    let c = chain(&tokens);

    // Snapshot at token 16
    let h16 = c.view(&tokens, 64).key_at(16).fold_u64();
    idx.insert(h16, 10, 0, 16);

    // Snapshot at token 32
    let h32 = c.view(&tokens, 64).key_at(32).fold_u64();
    idx.insert(h32, 20, 0, 32);

    // Lookup with 48 matched tokens — deepest snapshot at 32 wins
    let result = idx.lookup(&c.view(&tokens, 64), 48, 0);
    assert_eq!(result, Some((20, 32)));

    // Lookup with 20 matched tokens — only snapshot at 16 qualifies
    let result = idx.lookup(&c.view(&tokens, 64), 20, 0);
    assert_eq!(result, Some((10, 16)));
}

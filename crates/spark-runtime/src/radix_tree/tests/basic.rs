// SPDX-License-Identifier: AGPL-3.0-only

//! Basic radix-tree tests: insert, lookup, eviction, branching, partial
//! match, sub-block tails, ref counting, media-digest divergence (KV-4),
//! and SSM-snapshot insert/lookup through the [`super::super::RadixTree`]
//! API.

use crate::block_keys::{BlockKeyChain, MediaDigest, seed};
use crate::prefix_cache::PrefixCache;
use crate::radix_tree::RadixTree;

const BS: usize = 16;
const IMAGE_PAD: u32 = 151655; // <|image_pad|>

/// Chain for a text-only prompt (no media).
fn chain(tokens: &[u32]) -> BlockKeyChain {
    BlockKeyChain::new(seed(b"test-model", &[0u8; 32]), BS, tokens, 0, &[])
}

/// Chain for a vision prompt: pad runs of `IMAGE_PAD`, one digest each.
fn vision_chain(tokens: &[u32], digests: &[MediaDigest]) -> BlockKeyChain {
    BlockKeyChain::new(
        seed(b"test-model", &[0u8; 32]),
        BS,
        tokens,
        IMAGE_PAD,
        digests,
    )
}

#[test]
fn test_insert_and_lookup_exact() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect(); // 1 block of 16 tokens
    let block_table = vec![42];
    let c = chain(&tokens);

    tree.insert(&c.view(&tokens, 16), &block_table, &[], 0);
    let m = tree.lookup(&c.view(&tokens, 16), 0);

    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![42]);
}

#[test]
fn test_insert_and_lookup_multi_block() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..48).collect(); // 3 blocks of 16
    let block_table = vec![10, 20, 30];
    let c = chain(&tokens);

    tree.insert(&c.view(&tokens, 48), &block_table, &[], 0);
    let m = tree.lookup(&c.view(&tokens, 48), 0);

    assert_eq!(m.matched_tokens, 48);
    assert_eq!(m.matched_blocks, vec![10, 20, 30]);
}

#[test]
fn test_partial_match() {
    let tree = RadixTree::new();
    // Insert 32 tokens (2 blocks)
    let tokens_a: Vec<u32> = (0..32).collect();
    let ca = chain(&tokens_a);
    tree.insert(&ca.view(&tokens_a, 32), &[10, 20], &[], 0);

    // Lookup 48 tokens (3 blocks) — only first 2 match
    let tokens_b: Vec<u32> = (0..48).collect();
    let cb = chain(&tokens_b);
    let m = tree.lookup(&cb.view(&tokens_b, 48), 0);

    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.matched_blocks, vec![10, 20]);
}

#[test]
fn test_no_match() {
    let tree = RadixTree::new();
    let tokens_a: Vec<u32> = (0..16).collect();
    let ca = chain(&tokens_a);
    tree.insert(&ca.view(&tokens_a, 16), &[10], &[], 0);

    // Different tokens — no match
    let tokens_b: Vec<u32> = (100..116).collect();
    let cb = chain(&tokens_b);
    let m = tree.lookup(&cb.view(&tokens_b, 16), 0);

    assert!(m.is_empty());
}

#[test]
fn test_release_decrements_refcount() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();
    let c = chain(&tokens);
    // Simulate the real cache-miss flow: insert from a seq that did NOT
    // match on walk (matched_tokens=0), then that seq exits (release).
    // After release the cache holds exactly one ref (its baseline).
    tree.insert(&c.view(&tokens, 16), &[42], &[], 0);
    tree.release(&c.view(&tokens, 16));

    // A second seq comes in, walks (hits), then exits.
    let _ = tree.lookup(&c.view(&tokens, 16), 0);
    tree.release(&c.view(&tokens, 16));

    // After both seqs exit, ref_count == cache baseline (1) → evictable.
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![42]);
}

#[test]
fn test_insert_release_lookup_survives() {
    // Regression: the original Marconi/prefix-cache regression (alpha-2.99)
    // was that the first seq to cache a prompt rendered its own cache
    // entry un-findable when it exited — `walk` required `ref_count > 0`
    // but `insert` created nodes at ref_count=1 and `release` decremented
    // them to 0. The fix threads matched_tokens through insert so the
    // inserting seq's pending release doesn't drop the cache's baseline.
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    let c = chain(&tokens);
    tree.insert(&c.view(&tokens, 32), &[10, 20], &[], 0); // cache-miss insert
    tree.release(&c.view(&tokens, 32)); // inserting seq exits

    // Second request with identical prompt — must find the cached entry.
    let m = tree.lookup(&c.view(&tokens, 32), 0);
    assert_eq!(m.matched_tokens, 32, "stale cache entry after seq exit");
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&c.view(&tokens, 32));
}

#[test]
fn test_evict_lru_order() {
    let tree = RadixTree::new();

    // Insert block A (older) + release inserting seq
    let tokens_a: Vec<u32> = (0..16).collect();
    let ca = chain(&tokens_a);
    tree.insert(&ca.view(&tokens_a, 16), &[10], &[], 0);
    tree.release(&ca.view(&tokens_a, 16));

    // Insert block B (newer) + release inserting seq
    let tokens_b: Vec<u32> = (100..116).collect();
    let cb = chain(&tokens_b);
    tree.insert(&cb.view(&tokens_b, 16), &[20], &[], 0);
    tree.release(&cb.view(&tokens_b, 16));

    // Evict 1 — should be A (older LRU)
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![10]);

    // Evict 1 more — should be B
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![20]);
}

#[test]
fn test_evict_skips_referenced_nodes() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();
    let c = chain(&tokens);
    // Cache-miss insert + release: node settles at ref_count=1 (cache baseline).
    tree.insert(&c.view(&tokens, 16), &[42], &[], 0);
    tree.release(&c.view(&tokens, 16));

    // An active seq walks (matches) → ref_count bumps above the baseline.
    let _ = tree.lookup(&c.view(&tokens, 16), 0);

    // Evict should skip while the seq is active (ref_count > 1).
    let evicted = tree.evict(1);
    assert!(evicted.is_empty());

    // Seq exits → ref_count returns to the cache baseline → evictable.
    tree.release(&c.view(&tokens, 16));
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![42]);
}

#[test]
fn test_evict_chain_from_leaf() {
    let tree = RadixTree::new();
    // Insert 3-block chain + release inserting seq so the chain is evictable.
    let tokens: Vec<u32> = (0..48).collect();
    let c = chain(&tokens);
    tree.insert(&c.view(&tokens, 48), &[10, 20, 30], &[], 0);
    tree.release(&c.view(&tokens, 48));

    // Evict 1 — should remove leaf (block 30)
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![30]);

    // Now block 20 is a leaf, evict it
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![20]);

    // Now block 10 is a leaf
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![10]);

    // Tree is empty
    assert_eq!(tree.stats(), (0, 0));
}

#[test]
fn test_insert_idempotent() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();
    let c = chain(&tokens);

    // Insert same sequence twice — should not create duplicate nodes
    tree.insert(&c.view(&tokens, 16), &[42], &[], 0);
    tree.insert(&c.view(&tokens, 16), &[42], &[], 0);

    assert_eq!(tree.stats(), (1, 1));
}

#[test]
fn test_branching_tree() {
    let tree = RadixTree::new();

    // Insert A: [0..32] → blocks [10, 20]
    let tokens_a: Vec<u32> = (0..32).collect();
    let ca = chain(&tokens_a);
    tree.insert(&ca.view(&tokens_a, 32), &[10, 20], &[], 0);

    // Insert B: [0..16, 100..116] → blocks [10, 30]
    // Same first block, different second block
    let mut tokens_b: Vec<u32> = (0..16).collect();
    tokens_b.extend(100..116);
    let cb = chain(&tokens_b);
    tree.insert(&cb.view(&tokens_b, 32), &[10, 30], &[], 0);

    // Lookup A — full match
    let m = tree.lookup(&ca.view(&tokens_a, 32), 0);
    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&ca.view(&tokens_a, 32));

    // Lookup B — full match
    let m = tree.lookup(&cb.view(&tokens_b, 32), 0);
    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.matched_blocks, vec![10, 30]);
    tree.release(&cb.view(&tokens_b, 32));

    // 3 nodes total: block 10 (shared prefix), block 20 (A), block 30 (B)
    assert_eq!(tree.stats(), (3, 3));
}

#[test]
fn test_sub_block_tokens_ignored() {
    let tree = RadixTree::new();
    // Only 10 tokens — less than one block of 16; a tail has no full-block
    // anchor node, so nothing is cached.
    let tokens: Vec<u32> = (0..10).collect();
    let c = chain(&tokens);
    tree.insert(&c.view(&tokens, 10), &[42], &[], 0);

    assert_eq!(tree.stats(), (0, 0));
    assert!(tree.lookup(&c.view(&tokens, 10), 0).is_empty());
}

#[test]
fn test_exact_tail_reuse() {
    let tree = RadixTree::new();
    // 20 tokens: 1 full block + 4-token tail. block_table carries the
    // tail's physical block as the extra entry.
    let tokens: Vec<u32> = (0..20).collect();
    let c = chain(&tokens);
    tree.insert(&c.view(&tokens, 20), &[10, 11], &[], 0);

    // Identical 20-token prompt: full block AND exact tail match.
    let m = tree.lookup(&c.view(&tokens, 20), 0);
    assert_eq!(m.matched_tokens, 20);
    assert_eq!(m.matched_blocks, vec![10, 11]);
    tree.release(&c.view(&tokens, 20));

    // A longer prompt sharing the first 20 tokens matches only the full
    // block — the tail block's KV covers positions 16..20 of a DIFFERENT
    // physical layout and is matched exactly or not at all.
    let longer: Vec<u32> = (0..36).collect();
    let cl = chain(&longer);
    let m = tree.lookup(&cl.view(&longer, 36), 0);
    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
    tree.release(&cl.view(&longer, 36));
}

#[test]
fn test_divergent_tails_coexist() {
    let tree = RadixTree::new();
    // Two sequences share block 0 but end in different tails.
    let mut tokens_a: Vec<u32> = (0..16).collect();
    tokens_a.extend([100, 101]);
    let mut tokens_b: Vec<u32> = (0..16).collect();
    tokens_b.extend([200, 201, 202]);

    let ca = chain(&tokens_a);
    let cb = chain(&tokens_b);
    tree.insert(&ca.view(&tokens_a, 18), &[10, 50], &[], 0);
    tree.insert(&cb.view(&tokens_b, 19), &[10, 60], &[], 0);

    let ma = tree.lookup(&ca.view(&tokens_a, 18), 0);
    assert_eq!(ma.matched_tokens, 18);
    assert_eq!(ma.matched_blocks, vec![10, 50]);
    tree.release(&ca.view(&tokens_a, 18));

    let mb = tree.lookup(&cb.view(&tokens_b, 19), 0);
    assert_eq!(mb.matched_tokens, 19);
    assert_eq!(mb.matched_blocks, vec![10, 60]);
    tree.release(&cb.view(&tokens_b, 19));
}

/// Issue #17 regression: when HSS is active, the cache must report which
/// disk_block_ids it newly takes ownership of so the caller can balance
/// `inc_disk_ref`/`dec_disk_ref`. Without this, the second request panics
/// at `high_speed_swap.rs:167` because the cache holds a stale ID whose
/// refcount has already hit zero.
#[test]
fn test_hss_disk_ref_acquisition_cold_insert() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    let block_table = vec![10, 20];
    let disk_ids = vec![100u32, 101u32];
    let c = chain(&tokens);

    // Cold insert with HSS active: cache takes ownership of both disk_ids.
    let acquired = tree.insert(&c.view(&tokens, 32), &block_table, &disk_ids, 0);
    assert_eq!(acquired, vec![100, 101]);
}

#[test]
fn test_hss_disk_ref_acquisition_re_insert_no_double() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    let block_table = vec![10, 20];
    let disk_ids = vec![100u32, 101u32];
    let c = chain(&tokens);

    // First insert acquires both.
    let acquired1 = tree.insert(&c.view(&tokens, 32), &block_table, &disk_ids, 0);
    assert_eq!(acquired1, vec![100, 101]);

    // Second insert of the same prefix: nothing newly acquired (the cache
    // already owns these). Re-acquiring would over-inc and leak the disk
    // refcount.
    let acquired2 = tree.insert(&c.view(&tokens, 32), &block_table, &disk_ids, 0);
    assert!(
        acquired2.is_empty(),
        "re-insert should not re-acquire disk_ids; got {acquired2:?}"
    );
}

#[test]
fn test_hss_disk_ref_acquisition_extension() {
    let tree = RadixTree::new();
    let tokens_short: Vec<u32> = (0..32).collect();
    let tokens_long: Vec<u32> = (0..48).collect();
    let cs = chain(&tokens_short);
    let cl = chain(&tokens_long);

    // Insert 2 blocks first.
    let acquired1 = tree.insert(&cs.view(&tokens_short, 32), &[10, 20], &[100u32, 101u32], 0);
    assert_eq!(acquired1, vec![100, 101]);

    // Extension: same first 2 blocks (already cached) + 1 new block.
    // Only the new block's disk_id should be reported as acquired.
    let acquired2 = tree.insert(
        &cl.view(&tokens_long, 48),
        &[10, 20, 30],
        &[100u32, 101u32, 102u32],
        0,
    );
    assert_eq!(acquired2, vec![102]);
}

#[test]
fn test_hss_disk_ref_acquisition_no_op_when_hss_inactive() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    let c = chain(&tokens);

    // Empty disk_ids slice ⇒ HSS not active ⇒ nothing acquired regardless
    // of whether nodes are new or pre-existing.
    let acquired = tree.insert(&c.view(&tokens, 32), &[10, 20], &[], 0);
    assert!(acquired.is_empty());
}

#[test]
fn test_ssm_snapshot_insert_and_lookup() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    let c = chain(&tokens);

    // Insert with snapshot on deepest node
    tree.insert_with_snapshot(&c.view(&tokens, 32), &[10, 20], &[], 42, 0, 0);

    // Lookup should return the snapshot at 32 tokens (leaf)
    let m = tree.lookup(&c.view(&tokens, 32), 0);
    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.ssm_snapshot, Some(42));
    assert_eq!(m.ssm_snapshot_tokens, 32);
    tree.release(&c.view(&tokens, 32));
}

#[test]
fn test_ssm_snapshot_partial_match_returns_deepest() {
    let tree = RadixTree::new();

    // Insert 3-block sequence with snapshot on leaf
    let tokens: Vec<u32> = (0..48).collect();
    let c = chain(&tokens);
    tree.insert_with_snapshot(&c.view(&tokens, 48), &[10, 20, 30], &[], 99, 0, 0);

    // Lookup only 2 blocks — snapshot is on block 3 (not matched)
    let tokens_short: Vec<u32> = (0..32).collect();
    let cs = chain(&tokens_short);
    let m = tree.lookup(&cs.view(&tokens_short, 32), 0);
    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.ssm_snapshot, None);
    assert_eq!(m.ssm_snapshot_tokens, 0);
    tree.release(&cs.view(&tokens_short, 32));
}

#[test]
fn test_ssm_snapshot_survives_tree_eviction() {
    // Snapshots are decoupled from tree nodes — evicting tree nodes
    // does NOT destroy the snapshot in the index.
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();
    let c = chain(&tokens);

    tree.insert_with_snapshot(&c.view(&tokens, 16), &[10], &[], 7, 0, 0);
    tree.release(&c.view(&tokens, 16)); // inserting seq exits → node evictable

    // Evict the tree node
    let evicted_blocks = tree.evict(1);
    assert_eq!(evicted_blocks.physical, vec![10]);

    // Snapshot still exists in the index (decoupled from tree)
    assert_eq!(tree.snapshot_count(), 1);

    // Explicit LRU eviction of snapshot index
    let snap = tree.evict_snapshot_lru();
    assert_eq!(snap, Some(7));
    assert_eq!(tree.snapshot_count(), 0);

    // Second LRU eviction returns None
    assert_eq!(tree.evict_snapshot_lru(), None);
}

#[test]
fn test_ssm_snapshot_overwrite_returns_displaced() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();
    let c = chain(&tokens);

    // First insert — no displaced snapshot
    let (displaced, _acquired) =
        tree.insert_with_snapshot(&c.view(&tokens, 16), &[10], &[], 5, 0, 0);
    assert_eq!(displaced, None);

    // Re-insert same path with new snapshot — returns displaced ID 5
    let (displaced, _acquired) =
        tree.insert_with_snapshot(&c.view(&tokens, 16), &[10], &[], 8, 0, 0);
    assert_eq!(displaced, Some(5));

    // Only 1 entry in the index (overwrite, not append)
    assert_eq!(tree.snapshot_count(), 1);

    // Lookup should return new snapshot
    let m = tree.lookup(&c.view(&tokens, 16), 0);
    assert_eq!(m.ssm_snapshot, Some(8));
    assert_eq!(m.ssm_snapshot_tokens, 16);
    tree.release(&c.view(&tokens, 16));
}

/// KV-4 regression (closes issue #58): two vision requests with identical
/// token streams but DIFFERENT images must not share any block from the
/// first media block onward. The media digest enters the chain key, so
/// the walk diverges exactly where the image does — and vision prompts no
/// longer need the all-or-nothing cache bypass.
#[test]
fn test_vision_different_images_never_share_blocks() {
    let tree = RadixTree::new();

    // 3 prompt tokens, pad run of 29 → 32 tokens == 2 blocks of 16.
    // The pad run starts inside block 0, so EVERY block is media-tainted.
    let page: Vec<u32> = [1u32, 2, 3]
        .into_iter()
        .chain(std::iter::repeat_n(IMAGE_PAD, 29))
        .collect();

    let ca = vision_chain(&page, &[[0xAA; 32]]);
    let cb = vision_chain(&page, &[[0xBB; 32]]);

    tree.insert(&ca.view(&page, 32), &[10, 20], &[], 0);

    // Different image: no block may match.
    let m = tree.lookup(&cb.view(&page, 32), 0);
    assert!(
        m.is_empty(),
        "distinct images shared prefix-cache blocks (issue #58): {m:?}"
    );

    // Same image: full match.
    let ca2 = vision_chain(&page, &[[0xAA; 32]]);
    let m = tree.lookup(&ca2.view(&page, 32), 0);
    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&ca2.view(&page, 32));
}

/// KV-4: text blocks BEFORE the first image stay shared; divergence
/// starts at the first block containing pad tokens.
#[test]
fn test_vision_divergence_starts_at_first_media_block() {
    let tree = RadixTree::new();

    // Block 0: pure text. Block 1: text + pad run. Block 2: pads.
    let mut page: Vec<u32> = (0..20).collect(); // tokens 0..20 (block 0 + 4)
    page.extend(std::iter::repeat_n(IMAGE_PAD, 28)); // tokens 20..48

    let ca = vision_chain(&page, &[[0xAA; 32]]);
    let cb = vision_chain(&page, &[[0xBB; 32]]);

    tree.insert(&ca.view(&page, 48), &[10, 20, 30], &[], 0);

    let m = tree.lookup(&cb.view(&page, 48), 0);
    assert_eq!(
        m.matched_tokens, 16,
        "pre-image text block must stay shared; media blocks must not"
    );
    assert_eq!(m.matched_blocks, vec![10]);
    tree.release(&cb.view(&page, 48));
}

/// A chain whose media has no digests (legacy path) must behave exactly
/// like the old vision bypass: no lookup hits, no inserts.
#[test]
fn test_vision_without_digests_bypasses_cache() {
    let tree = RadixTree::new();
    let page: Vec<u32> = [1u32, 2, 3]
        .into_iter()
        .chain(std::iter::repeat_n(IMAGE_PAD, 29))
        .collect();

    let c = vision_chain(&page, &[]); // pad run present, digest missing
    tree.insert(&c.view(&page, 32), &[10, 20], &[], 0);
    assert_eq!(tree.stats(), (0, 0));
    assert!(tree.lookup(&c.view(&page, 32), 0).is_empty());
}

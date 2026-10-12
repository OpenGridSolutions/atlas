// SPDX-License-Identifier: AGPL-3.0-only

//! `mapped_blocks`: the cache reports the physical block it maps per key,
//! which is NOT the inserting sequence's block when the node already
//! existed — the caller must not take a cache ref on its duplicate.

use crate::block_keys::{BlockKeyChain, seed};
use crate::prefix_cache::PrefixCache;
use crate::radix_tree::RadixTree;

fn chain(tokens: &[u32]) -> BlockKeyChain {
    BlockKeyChain::new(seed(b"test-model", &[0u8; 32]), 16, tokens, 0, &[])
}

#[test]
fn mapped_blocks_keep_the_existing_node_block_over_a_duplicate() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..48).collect();
    let c = chain(&tokens);
    // First sequence: three full blocks, all adopted.
    tree.insert(&c.view(&tokens, 48), &[10, 20, 30], &[], 0);
    assert_eq!(tree.mapped_blocks(&c.view(&tokens, 48)), vec![10, 20, 30]);

    // Second sequence shared block 10 (matched 16 tokens) and recomputed
    // the same content into its own blocks 21 and 31.
    tree.insert(&c.view(&tokens, 48), &[10, 21, 31], &[], 16);
    let mapped = tree.mapped_blocks(&c.view(&tokens, 48));
    assert_eq!(mapped, vec![10, 20, 30], "existing nodes keep their blocks");
    let own = [10u32, 21, 31];
    let cache_owned: Vec<u32> = own
        .iter()
        .zip(&mapped)
        .filter(|(b, m)| b == m)
        .map(|(b, _)| *b)
        .collect();
    assert_eq!(cache_owned, vec![10], "21 and 31 stay sequence-owned");

    // A chain the tree has never seen maps nothing.
    let other: Vec<u32> = (100..148).collect();
    let oc = chain(&other);
    assert!(tree.mapped_blocks(&oc.view(&other, 48)).is_empty());
}

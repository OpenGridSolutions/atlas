// SPDX-License-Identifier: AGPL-3.0-only

//! KV-7 eviction guard: persisted blocks leave RAM first; when nothing
//! persisted is left the LRU pass runs unguarded, so a lagging NVMe write
//! queue can never turn into `KV cache exhausted` (T-KV-10).

use std::sync::Arc;

use crate::block_keys::{BlockKeyChain, seed};
use crate::prefix_cache::{EvictGuard, PrefixCache};
use crate::radix_tree::RadixTree;

const BS: usize = 16;

fn chain(tokens: &[u32]) -> BlockKeyChain {
    BlockKeyChain::new(seed(b"test-model", &[0u8; 32]), BS, tokens, 0, &[])
}

#[test]
fn guard_prefers_persisted_blocks_then_falls_back_to_plain_lru() {
    let tree = RadixTree::new();
    let a: Vec<u32> = (0..16).collect();
    let b: Vec<u32> = (100..116).collect();
    let ca = chain(&a);
    let cb = chain(&b);
    tree.insert(&ca.view(&a, 16), &[10], &[], 0);
    tree.release(&ca.view(&a, 16));
    tree.insert(&cb.view(&b, 16), &[20], &[], 0);
    tree.release(&cb.view(&b, 16));

    // Only B counts as persisted.
    let kb = cb.view(&b, 16).full_keys()[0].0;
    let guard: EvictGuard = Arc::new(move |k: &[u8; 16], _tail: bool| *k == kb);
    tree.set_evict_guard(Some(guard));

    // A is the LRU candidate but unpersisted → the persisted B goes first.
    assert_eq!(tree.evict(1).physical, vec![20]);
    // Nothing persisted remains → the fallback pass still frees A.
    assert_eq!(tree.evict(1).physical, vec![10]);
    assert!(tree.evict(1).physical.is_empty());
}

#[test]
fn guard_that_vetoes_everything_does_not_starve_the_allocator() {
    let tree = RadixTree::new();
    let a: Vec<u32> = (0..32).collect();
    let ca = chain(&a);
    tree.insert(&ca.view(&a, 32), &[10, 20], &[], 0);
    tree.release(&ca.view(&a, 32));
    tree.set_evict_guard(Some(Arc::new(|_: &[u8; 16], _: bool| false)));
    // Two blocks requested, none persisted: both still come back, LRU order.
    let freed = tree.evict(2).physical;
    assert_eq!(freed.len(), 2);
    assert!(freed.contains(&10) && freed.contains(&20));
}

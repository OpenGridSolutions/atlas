// SPDX-License-Identifier: AGPL-3.0-only

//! LRU order under NVMe-restore shapes: a chain imported last must be the
//! last to go, whatever the eviction demand (BAT-6 regression probe).

use crate::block_keys::{BlockKeyChain, seed};
use crate::prefix_cache::PrefixCache;
use crate::radix_tree::RadixTree;

const BS: usize = 16;

fn chain(tokens: &[u32]) -> BlockKeyChain {
    BlockKeyChain::new(seed(b"test-model", &[0u8; 32]), BS, tokens, 0, &[])
}

fn tokens(session: u32, blocks: usize) -> Vec<u32> {
    (0..(blocks * BS) as u32)
        .map(|i| session * 100_000 + i)
        .collect()
}

#[test]
fn imported_chain_survives_eviction_of_older_sessions() {
    let tree = RadixTree::new();
    let blocks = 175;
    let mut next_block = 0u32;
    // Seven older sessions, inserted at request end (insert + release).
    for s in 0..7u32 {
        let t = tokens(s, blocks);
        let c = chain(&t);
        let bt: Vec<u32> = (next_block..next_block + blocks as u32).collect();
        next_block += blocks as u32;
        tree.insert(&c.view(&t, t.len()), &bt, &[], 0);
        tree.release(&c.view(&t, t.len()));
    }
    // The restored session: inserted cache-owned (matched_tokens = covered).
    let t = tokens(99, blocks);
    let c = chain(&t);
    let bt: Vec<u32> = (next_block..next_block + blocks as u32).collect();
    let imported: std::collections::HashSet<u32> = bt.iter().copied().collect();
    tree.insert(&c.view(&t, t.len()), &bt, &[], t.len());

    let evicted = tree.evict(177).physical;
    assert_eq!(evicted.len(), 177);
    let lost: Vec<u32> = evicted
        .iter()
        .copied()
        .filter(|b| imported.contains(b))
        .collect();
    assert!(
        lost.is_empty(),
        "{} imported blocks evicted before older sessions",
        lost.len()
    );
    assert_eq!(tree.peek_matched_tokens(&c.view(&t, t.len())), t.len());
}

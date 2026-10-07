// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the chained block keys (KV-3/4/5) and the shrink
//! discipline that keeps rejected draft tokens out of sealed keys (KV-8).

use super::*;

const BS: usize = 4;

fn chain(tokens: &[u32], pad: u32, digests: &[MediaDigest]) -> BlockKeyChain {
    BlockKeyChain::new(seed(b"model-A", &[7u8; 32]), BS, tokens, pad, digests)
}

#[test]
fn deterministic_and_prefix_stable() {
    let toks: Vec<u32> = (0..13).collect();
    let a = chain(&toks, 0, &[]);
    let b = chain(&toks, 0, &[]);
    let va = a.view(&toks, 13);
    let keys_a: Vec<_> = va.full_keys().to_vec();
    let tail_a = va.tail_key();
    let vb = b.view(&toks, 13);
    assert_eq!(keys_a, vb.full_keys());
    assert_eq!(tail_a, vb.tail_key());
    assert_eq!(keys_a.len(), 3);
    assert!(tail_a.is_some());
    // Shorter view exposes a strict prefix of the same keys.
    let v8 = b.view(&toks, 8);
    assert_eq!(&keys_a[..2], v8.full_keys());
    assert_eq!(v8.tail_key(), None);
}

#[test]
fn different_seed_diverges_from_block_zero() {
    let toks: Vec<u32> = (0..8).collect();
    let a = BlockKeyChain::new(seed(b"model-A", &[7u8; 32]), BS, &toks, 0, &[]);
    let b = BlockKeyChain::new(seed(b"model-B", &[7u8; 32]), BS, &toks, 0, &[]);
    assert_ne!(a.view(&toks, 8).full_keys(), b.view(&toks, 8).full_keys());
}

#[test]
fn media_divergence_starts_at_first_media_block() {
    // pad token 99, run at tokens [4..6) → block 1 is the first media block.
    let toks: Vec<u32> = vec![1, 2, 3, 4, 99, 99, 7, 8, 9, 10, 11, 12];
    let a = chain(&toks, 99, &[[1u8; 32]]);
    let b = chain(&toks, 99, &[[2u8; 32]]);
    let va = a.view(&toks, 12);
    let vb = b.view(&toks, 12);
    assert!(va.cacheable() && vb.cacheable());
    // Block 0 (tokens 0..4, no media): shared.
    assert_eq!(va.full_keys()[0], vb.full_keys()[0]);
    // Block 1 (contains the pad run): diverges, and stays diverged.
    assert_ne!(va.full_keys()[1], vb.full_keys()[1]);
    assert_ne!(va.full_keys()[2], vb.full_keys()[2]);
}

#[test]
fn media_run_spanning_blocks_taints_every_touched_block() {
    // run [2..6) touches blocks 0 and 1.
    let toks: Vec<u32> = vec![1, 2, 99, 99, 99, 99, 7, 8];
    let a = chain(&toks, 99, &[[1u8; 32]]);
    let b = chain(&toks, 99, &[[2u8; 32]]);
    let va = a.view(&toks, 8);
    let vb = b.view(&toks, 8);
    assert_ne!(va.full_keys()[0], vb.full_keys()[0]);
    assert_ne!(va.full_keys()[1], vb.full_keys()[1]);
}

#[test]
fn missing_digests_make_chain_uncacheable() {
    let toks: Vec<u32> = vec![1, 99, 99, 4, 5, 6, 7, 8];
    let c = chain(&toks, 99, &[]); // one pad run, zero digests
    assert!(!c.is_cacheable());
    assert!(!c.view(&toks, 8).cacheable());
    // No media, no digests: cacheable.
    let plain: Vec<u32> = (1..9).collect();
    let p = chain(&plain, 99, &[]);
    assert!(p.view(&plain, 8).cacheable());
}

#[test]
fn default_chain_is_inert() {
    let toks: Vec<u32> = (0..8).collect();
    let c = BlockKeyChain::default();
    let v = c.view(&toks, 8);
    assert!(!v.cacheable());
    assert_eq!(v.num_full_blocks(), 0);
}

#[test]
fn key_at_matches_view_boundaries() {
    let toks: Vec<u32> = (0..10).collect();
    let c = chain(&toks, 0, &[]);
    let v = c.view(&toks, 10);
    assert_eq!(v.key_at(4), v.full_keys()[0]);
    assert_eq!(v.key_at(10), v.tail_key().unwrap());
    assert_eq!(v.key_at(0), c.view(&toks, 0).key_at(0));
}

#[test]
fn tail_key_depends_on_tail_media() {
    // 9 tokens, pad run in the tail (tokens 8..9).
    let toks: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 99];
    let a = chain(&toks, 99, &[[1u8; 32]]);
    let b = chain(&toks, 99, &[[2u8; 32]]);
    let ta = a.view(&toks, 9).tail_key();
    let tb = b.view(&toks, 9).tail_key();
    assert_ne!(ta, tb);
}

#[test]
fn hex_roundtrip() {
    let k = seed(b"x", &[3u8; 32]);
    assert_eq!(BlockKey::from_hex(&k.to_hex()), Some(k));
    assert_eq!(BlockKey::from_hex("zz"), None);
}

#[test]
fn decode_extension_is_incremental() {
    // Simulate decode: prompt of 6, then tokens appended one by one.
    let mut toks: Vec<u32> = (0..6).collect();
    let mut c = chain(&toks, 0, &[]);
    let k_prompt = c.view(&toks, 6).full_keys().to_vec();
    for t in 6..14 {
        toks.push(t);
        c.extend_to(&toks);
    }
    let v = c.view(&toks, 14);
    assert_eq!(&v.full_keys()[..1], &k_prompt[..]);
    assert_eq!(v.num_full_blocks(), 3);
    // Same result as a fresh chain over the final tokens.
    let fresh = chain(&toks, 0, &[]);
    assert_eq!(v.full_keys(), fresh.view(&toks, 14).full_keys());
}

// ── KV-8: no key over tokens that were never committed ─────────────

/// T-KV-08. A verify step pushes γ drafts into `tokens`; if an observer
/// (Marconi checkpoint, release) extends the chain in that window and
/// the drafts are then rejected, the key over the draft block must not
/// survive the rewind. The regrown block gets the key of a fresh chain.
#[test]
fn rejected_drafts_never_seal_a_block() {
    let mut toks: Vec<u32> = (0..6).collect();
    let mut c = chain(&toks, 0, &[]);
    // γ=3 drafts complete block 1 (tokens 4..8).
    toks.extend_from_slice(&[100, 101, 102]);
    c.extend_to(&toks);
    let stale = c.view(&toks, toks.len()).full_keys()[1];
    // All drafts rejected: rewind, then regrow with the verified tokens.
    toks.truncate(6);
    c.truncate_to(toks.len());
    toks.extend_from_slice(&[7, 8, 9]);
    c.extend_to(&toks);
    let v = c.view(&toks, 9);
    let fresh = chain(&toks, 0, &[]);
    assert_eq!(v.full_keys(), fresh.view(&toks, 9).full_keys());
    assert_ne!(
        v.full_keys()[1],
        stale,
        "draft-tainted key survived the rewind"
    );
    assert_eq!(v.tail_key(), fresh.view(&toks, 9).tail_key());
}

/// A shrink that the chain observes through `extend_to` (no explicit
/// `truncate_to`) is healed before the chain regrows.
#[test]
fn extend_heals_an_over_extended_chain() {
    let mut toks: Vec<u32> = (0..12).collect();
    let mut c = chain(&toks, 0, &[]);
    assert_eq!(c.view(&toks, 12).num_full_blocks(), 3);
    toks.truncate(5);
    c.extend_to(&toks);
    assert_eq!(c.view(&toks, 5).num_full_blocks(), 1);
    toks.extend_from_slice(&[50, 51, 52, 53, 54, 55, 56]);
    c.extend_to(&toks);
    let fresh = chain(&toks, 0, &[]);
    assert_eq!(
        c.view(&toks, 12).full_keys(),
        fresh.view(&toks, 12).full_keys()
    );
}

/// `truncate_to` keeps every block that is still fully covered and is
/// idempotent at the aligned prefix.
#[test]
fn truncate_to_keeps_the_covered_prefix() {
    let toks: Vec<u32> = (0..13).collect();
    let mut c = chain(&toks, 0, &[]);
    let keys = c.view(&toks, 13).full_keys().to_vec();
    assert_eq!(keys.len(), 3);
    c.truncate_to(9); // blocks 0,1 intact; block 2 (tokens 8..12) gone
    assert_eq!(c.view(&toks[..9], 9).full_keys(), &keys[..2]);
    c.truncate_to(9);
    assert_eq!(c.view(&toks[..9], 9).full_keys(), &keys[..2]);
    c.truncate_to(0);
    assert_eq!(c.view(&toks[..0], 0).num_full_blocks(), 0);
    // Inert chains are untouched.
    let mut inert = BlockKeyChain::default();
    inert.truncate_to(0);
    assert!(!inert.is_cacheable());
}

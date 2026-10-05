// SPDX-License-Identifier: AGPL-3.0-only

//! Chained block keys for KV-cache identity (RAM prefix cache + NVMe store).
//!
//! Every sealed KV block is identified by a 128-bit BLAKE3 chain key:
//!
//! ```text
//! k_0 = H(model_hash ‖ layout_hash)           (the per-model seed)
//! k_i = H(k_{i-1} ‖ token_ids_i ‖ media_digests_i)
//! ```
//!
//! `media_digests_i` are the digests of all media (images) whose
//! placeholder tokens fall into block `i`, in prompt order. Two prompts
//! with identical token IDs but different images therefore diverge at the
//! first media block — the fix that replaces the old all-or-nothing
//! vision-pad cache bypass (issue #58).
//!
//! A *tail key* identifies the trailing sub-block of a sequence (its last
//! `len % block_size` tokens): `H(k_i ‖ tail_token_ids ‖ tail_digests)`.
//! Tail matches are exact — block-aligned sharing plus exact-tail reuse;
//! prefix-of-block sharing needs copy-on-write and is a later stage.
//!
//! BLAKE3 is deliberate: the cache is shared across users, so collisions
//! must not be constructible (a non-cryptographic chain would let one
//! tenant poison another tenant's prefix).

use std::fmt;

/// 128-bit truncated BLAKE3 chain key. Identifies one sealed KV block
/// (or an SSM snapshot boundary) in the RAM radix cache and on NVMe.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockKey(pub [u8; 16]);

impl BlockKey {
    pub const ZERO: BlockKey = BlockKey([0u8; 16]);

    /// Stable 64-bit fold (first 8 bytes, LE) — for indexes that key by
    /// `u64` (SSM snapshot index).
    pub fn fold_u64(&self) -> u64 {
        u64::from_le_bytes(self.0[..8].try_into().expect("8 bytes"))
    }

    /// Lowercase hex, for store filenames and logs.
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(32);
        for b in self.0 {
            use fmt::Write;
            let _ = write!(s, "{b:02x}");
        }
        s
    }

    pub fn from_hex(s: &str) -> Option<BlockKey> {
        if s.len() != 32 {
            return None;
        }
        let mut out = [0u8; 16];
        for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
            let hi = (chunk[0] as char).to_digit(16)?;
            let lo = (chunk[1] as char).to_digit(16)?;
            out[i] = ((hi << 4) | lo) as u8;
        }
        Some(BlockKey(out))
    }
}

impl fmt::Debug for BlockKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlockKey({})", self.to_hex())
    }
}

/// Media digest: BLAKE3 over the raw media bytes plus the preprocessing
/// parameters that shape the spliced embeddings (resolution/grid, merge
/// size). Computed at the API layer, carried per image in prompt order.
pub type MediaDigest = [u8; 32];

/// One placeholder-token run in the prompt (one image).
#[derive(Clone, Copy, Debug)]
struct MediaRun {
    /// First token index of the pad run.
    start: usize,
    /// Number of pad tokens in the run.
    len: usize,
    /// Digest of the media occupying this run.
    digest: MediaDigest,
}

/// Compute the per-model seed `k_0 = H(model_hash ‖ layout_hash)`.
pub fn seed(model_identity: &[u8], layout_hash: &[u8; 32]) -> BlockKey {
    let mut h = blake3::Hasher::new();
    h.update(model_identity);
    h.update(layout_hash);
    truncate(h.finalize())
}

fn truncate(hash: blake3::Hash) -> BlockKey {
    let mut out = [0u8; 16];
    out.copy_from_slice(&hash.as_bytes()[..16]);
    BlockKey(out)
}

fn hash_step(prev: &BlockKey, tokens: &[u32], digests: &[MediaDigest]) -> BlockKey {
    let mut h = blake3::Hasher::new();
    h.update(&prev.0);
    for &t in tokens {
        h.update(&t.to_le_bytes());
    }
    for d in digests {
        h.update(d);
    }
    truncate(h.finalize())
}

/// Incrementally computed chain of block keys for one sequence.
///
/// Owned by `SequenceState`; extended as the sequence grows (decode
/// appends tokens, never media). All key material is CPU-side and cheap
/// (~ns per block); the chain survives disk spill and restore.
pub struct BlockKeyChain {
    seed: BlockKey,
    block_size: usize,
    media_runs: Vec<MediaRun>,
    /// `keys[i]` = chain key after block `i` (first `(i+1)*block_size`
    /// tokens).
    keys: Vec<BlockKey>,
    /// Tokens consumed into `keys` (always a multiple of `block_size`).
    consumed: usize,
    /// False when the prompt contains media placeholders that no digest
    /// covers (or construction was skipped): the sequence must not use
    /// the prefix cache — the key chain cannot represent its content.
    cacheable: bool,
}

impl Default for BlockKeyChain {
    /// Inert chain: uncacheable, zero seed. The safe default for
    /// sequences whose owner never initialised keys (mock models, paths
    /// that predate the key plumbing).
    fn default() -> Self {
        Self {
            seed: BlockKey::ZERO,
            block_size: 0,
            media_runs: Vec::new(),
            keys: Vec::new(),
            consumed: 0,
            cacheable: false,
        }
    }
}

impl BlockKeyChain {
    /// Build the chain for a prompt. `media_digests` are the digests of
    /// the prompt's images in order; pad runs are found by scanning
    /// `prompt_tokens` for maximal runs of `pad_token_id`. When the run
    /// count and the digest count disagree (an image path that doesn't
    /// provide digests yet), the chain is marked uncacheable — the
    /// sequence then behaves exactly like the pre-digest vision bypass.
    pub fn new(
        seed_key: BlockKey,
        block_size: usize,
        prompt_tokens: &[u32],
        pad_token_id: u32,
        media_digests: &[MediaDigest],
    ) -> Self {
        assert!(block_size > 0, "block_size must be non-zero");
        let mut runs: Vec<MediaRun> = Vec::new();
        let mut i = 0usize;
        while i < prompt_tokens.len() {
            if prompt_tokens[i] == pad_token_id && pad_token_id != 0 {
                let start = i;
                while i < prompt_tokens.len() && prompt_tokens[i] == pad_token_id {
                    i += 1;
                }
                runs.push(MediaRun {
                    start,
                    len: i - start,
                    digest: [0u8; 32], // filled below
                });
            } else {
                i += 1;
            }
        }
        let cacheable = runs.len() == media_digests.len();
        if cacheable {
            for (run, digest) in runs.iter_mut().zip(media_digests) {
                run.digest = *digest;
            }
        }
        let mut chain = Self {
            seed: seed_key,
            block_size,
            media_runs: runs,
            keys: Vec::new(),
            consumed: 0,
            cacheable,
        };
        chain.extend(prompt_tokens);
        chain
    }

    pub fn is_cacheable(&self) -> bool {
        self.cacheable
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Digests of media whose pad runs intersect `[start, end)`, in
    /// prompt order.
    fn digests_in(&self, start: usize, end: usize) -> Vec<MediaDigest> {
        self.media_runs
            .iter()
            .filter(|r| r.start < end && r.start + r.len > start)
            .map(|r| r.digest)
            .collect()
    }

    /// Extend `keys` to cover all full blocks of `tokens`.
    fn extend(&mut self, tokens: &[u32]) {
        if self.block_size == 0 {
            return;
        }
        let bs = self.block_size;
        let full = tokens.len() / bs;
        while self.keys.len() < full {
            let i = self.keys.len();
            let start = i * bs;
            let end = start + bs;
            let prev = if i == 0 { self.seed } else { self.keys[i - 1] };
            let digests = self.digests_in(start, end);
            self.keys
                .push(hash_step(&prev, &tokens[start..end], &digests));
            self.consumed = end;
        }
    }

    /// Extend the materialised keys to cover all full blocks of `tokens`.
    /// `new()` covers the prompt; call this after decode appends tokens
    /// (any `&mut` site before a view over the grown sequence works —
    /// the computation is incremental and idempotent).
    pub fn extend_to(&mut self, tokens: &[u32]) {
        self.extend(tokens);
    }

    /// Borrowed view over the first `len` tokens. Requires the chain to
    /// be extended at least that far (`new()` covers the prompt;
    /// `extend_to` covers decode growth) — an under-extended view clamps
    /// to the materialised keys (debug-asserted) rather than exposing
    /// wrong keys.
    pub fn view<'a>(&'a self, tokens: &'a [u32], len: usize) -> BlockKeysView<'a> {
        let mut len = len.min(tokens.len());
        if self.block_size > 0 && len / self.block_size > self.keys.len() {
            debug_assert!(
                false,
                "BlockKeyChain under-extended: view of {} tokens but only {} keys \
                 (call extend_to after appending tokens)",
                len,
                self.keys.len(),
            );
            len = self.keys.len() * self.block_size;
        }
        BlockKeysView {
            chain: self,
            tokens,
            len,
        }
    }
}

/// Read view over a chain prefix — the parameter object every
/// `PrefixCache` method takes. Carries the tokens alongside the keys so
/// positional (sub-block) keys can be derived on demand.
pub struct BlockKeysView<'a> {
    chain: &'a BlockKeyChain,
    tokens: &'a [u32],
    len: usize,
}

impl<'a> BlockKeysView<'a> {
    pub fn cacheable(&self) -> bool {
        self.chain.cacheable && self.chain.block_size > 0
    }

    pub fn block_size(&self) -> usize {
        self.chain.block_size
    }

    /// Number of tokens this view covers.
    pub fn total_tokens(&self) -> usize {
        self.len
    }

    pub fn num_full_blocks(&self) -> usize {
        self.len.checked_div(self.chain.block_size).unwrap_or(0)
    }

    /// Chain keys of the full blocks within the view.
    pub fn full_keys(&self) -> &[BlockKey] {
        &self.chain.keys[..self.num_full_blocks()]
    }

    /// Exact key of the trailing sub-block (`None` when the view is
    /// block-aligned).
    pub fn tail_key(&self) -> Option<BlockKey> {
        let bs = self.chain.block_size;
        if bs == 0 || self.len.is_multiple_of(bs) {
            return None;
        }
        let full = self.len / bs;
        let prev = if full == 0 {
            self.chain.seed
        } else {
            self.chain.keys[full - 1]
        };
        let digests = self.chain.digests_in(full * bs, self.len);
        Some(hash_step(
            &prev,
            &self.tokens[full * bs..self.len],
            &digests,
        ))
    }

    /// Key at an arbitrary position `pos <= total_tokens()` (for the SSM
    /// snapshot index, whose boundaries are not always block-aligned).
    pub fn key_at(&self, pos: usize) -> BlockKey {
        debug_assert!(pos <= self.len);
        let bs = self.chain.block_size;
        if pos == 0 || bs == 0 {
            return self.chain.seed;
        }
        let full = pos / bs;
        let aligned = if full == 0 {
            self.chain.seed
        } else {
            self.chain.keys[full - 1]
        };
        if pos.is_multiple_of(bs) {
            return aligned;
        }
        let digests = self.chain.digests_in(full * bs, pos);
        hash_step(&aligned, &self.tokens[full * bs..pos], &digests)
    }
}

#[cfg(test)]
mod tests {
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
}

// SPDX-License-Identifier: AGPL-3.0-only

//! Test vectors: the constants are the buffers stored in
//! nvidia/Qwen3.8-Flash-Next-FP4-Local (`layers.1.ple.ple_embedding.*`); the
//! expected row ids were computed with an independent implementation of the
//! reference rule (shifted token streams padded with EOS at segment starts).

use super::*;

const EOS: u32 = 248044;
const SHARD_ROWS: usize = 2_500_012;

fn sizes() -> Vec<i64> {
    vec![
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081,
        20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
    ]
}
fn offsets() -> Vec<i64> {
    let mut next = 0;
    sizes()
        .iter()
        .map(|s| {
            let o = next;
            next += s;
            o
        })
        .collect()
}
fn hasher() -> NgramHasher {
    NgramHasher::new(
        vec![23703573157769, 20109073645365, 8052911324071],
        sizes(),
        offsets(),
        8,
        EOS,
        SHARD_ROWS,
        128,
    )
    .expect("checkpoint constants are valid")
}

/// (ids[0], ids[7], ids[8], ids[15]) of every token of the sequence.
const SEQUENCE: [u32; 11] = [1000, 2000, 3000, EOS, 5, 6, EOS, EOS, 7, 8, 9];
const EXPECTED: [[i64; 4]; 11] = [
    [1876085, 142751402, 178377549, 304819575],
    [19790471, 152120809, 172190926, 315915558],
    [17238551, 147589323, 163669071, 302034413],
    [2413878, 141212937, 179088783, 311229572],
    [15389869, 155458159, 179763390, 315720898],
    [15567690, 148013673, 164414634, 307414745],
    [14335253, 147655513, 179956454, 303018794],
    [9663979, 151022725, 170054832, 300986548],
    [2927653, 151352333, 167888935, 304994522],
    [17329723, 156529177, 176740572, 304107632],
    [49828, 153224292, 175994762, 305810447],
];

fn ids_at(h: &NgramHasher, history: &[u32]) -> Vec<i64> {
    let mut out = Vec::new();
    h.row_ids(history, &mut out).expect("non-empty history");
    out
}

#[test]
fn row_ids_match_the_reference_for_a_sequence_with_eos_boundaries() {
    let h = hasher();
    assert_eq!((h.heads(), h.context_len()), (16, 2));
    for (i, want) in EXPECTED.iter().enumerate() {
        let ids = ids_at(&h, &SEQUENCE[..=i]);
        assert_eq!(ids.len(), 16);
        assert_eq!([ids[0], ids[7], ids[8], ids[15]], *want, "token {i}");
    }
}

#[test]
fn only_the_last_context_tokens_matter() {
    let h = hasher();
    let full = ids_at(&h, &SEQUENCE[..=5]);
    let tail = ids_at(&h, &SEQUENCE[3..=5]);
    assert_eq!(full, tail);
}

#[test]
fn start_of_sequence_reads_as_eos() {
    let h = hasher();
    // Token 5 directly after an EOS equals token 5 at position 0.
    assert_eq!(ids_at(&h, &[5]), ids_at(&h, &[1000, EOS, 5]));
    // One real predecessor at position 1: the missing second one is EOS.
    assert_eq!(ids_at(&h, &[5, 6]), ids_at(&h, &[EOS, 5, 6]));
}

#[test]
fn bigram_heads_ignore_the_second_predecessor() {
    let h = hasher();
    let a = ids_at(&h, &[11, 22, 33]);
    let b = ids_at(&h, &[99, 22, 33]);
    assert_eq!(a[..8], b[..8]);
    assert_ne!(a[8..], b[8..]);
}

#[test]
fn every_id_lies_in_its_head_range_and_addresses_a_real_row() {
    let h = hasher();
    let (sizes, offsets) = (sizes(), offsets());
    for tok in [0u32, 1, 4711, 248_319] {
        let ids = ids_at(&h, &[tok.wrapping_mul(7) % 248_320, 17, tok]);
        for (head, id) in ids.iter().enumerate() {
            assert!((offsets[head]..offsets[head] + sizes[head]).contains(id));
            let a = h.address(*id);
            assert!(a.shard < 128 && a.row < SHARD_ROWS);
            assert_eq!((a.shard * SHARD_ROWS + a.row) as i64, *id);
        }
    }
}

#[test]
fn empty_history_is_an_error() {
    assert!(hasher().row_ids(&[], &mut Vec::new()).is_err());
}

#[test]
fn inconsistent_constants_are_rejected() {
    let ok = |m: Vec<i64>, s: Vec<i64>, o: Vec<i64>, shards| {
        NgramHasher::new(m, s, o, 8, EOS, SHARD_ROWS, shards).is_ok()
    };
    let m = vec![3, 5, 7];
    assert!(ok(m.clone(), sizes(), offsets(), 128));
    assert!(!ok(vec![3], sizes(), offsets(), 128), "ngram_size 1");
    assert!(
        !ok(m.clone(), sizes()[..15].to_vec(), offsets(), 128),
        "head count"
    );
    let mut gap = offsets();
    gap[3] += 1;
    assert!(
        !ok(m.clone(), sizes(), gap, 128),
        "offsets are not cumulative"
    );
    let mut zero = sizes();
    zero[0] = 0;
    assert!(!ok(m.clone(), zero, offsets(), 128), "empty head");
    assert!(
        !ok(m, sizes(), offsets(), 127),
        "table smaller than the heads"
    );
}

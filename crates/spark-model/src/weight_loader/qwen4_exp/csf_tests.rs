// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// 16 rows x 4 cols: row r has base 100 + r and codes [1, 2, 3, 4] + r % 3.
fn sample() -> (Vec<u8>, Vec<u8>) {
    let (rows, cols) = (16usize, 4usize);
    let mut fixed = vec![0u8; fixed_shape(rows, cols).iter().product()];
    for r in 0..rows {
        fixed[r] = 100 + r as u8;
        let k = (r % 3) as u8;
        fixed[16 + r * 2] = (1 + k) | ((2 + k) << 4);
        fixed[16 + r * 2 + 1] = (3 + k) | ((4 + k) << 4);
    }
    // row 2 col 1 -> 7, row 15 col 3 -> 200
    let exc: Vec<u8> = [(2u32 * 4 + 1) | (7 << 24), (15 * 4 + 3) | (200 << 24)]
        .iter()
        .flat_map(|r| r.to_le_bytes())
        .collect();
    (fixed, exc)
}

#[test]
fn fixed_shape_matches_the_checkpoint_tensors() {
    // gate/up: [640, 2560] weights -> 160 scale columns; down: [2560, 640] -> 40.
    assert_eq!(fixed_shape(640, 160), [40, 1296]);
    assert_eq!(fixed_shape(2560, 40), [160, 336]);
}

#[test]
fn decode_applies_base_code_and_exceptions() {
    let (fixed, exc) = sample();
    validate("t", 16, 4, &fixed, &exc).expect("valid");
    let out = decode(16, 4, &fixed, &exc);
    assert_eq!(
        &out[0..4],
        &[101, 102, 103, 104],
        "row 0: base 100, low nibble first"
    );
    assert_eq!(
        &out[4..8],
        &[103, 104, 105, 106],
        "row 1: base 101, codes + 1"
    );
    assert_eq!(
        &out[8..12],
        &[105, 7, 107, 108],
        "row 2: exception replaces the byte"
    );
    assert_eq!(out[15 * 4 + 3], 200);
    assert_eq!(
        out[15 * 4 + 2],
        115 + 3,
        "neighbour of an exception is untouched"
    );
}

#[test]
fn decode_without_exceptions() {
    let (fixed, _) = sample();
    validate("t", 16, 4, &fixed, &[]).expect("valid");
    assert_eq!(decode(16, 4, &fixed, &[])[9], 106);
}

#[test]
fn validate_rejects_malformed_streams() {
    let (fixed, exc) = sample();
    let rec = |pos: u32, byte: u32| (pos | (byte << 24)).to_le_bytes().to_vec();
    assert!(
        validate("t", 15, 4, &fixed, &exc).is_err(),
        "rows not a multiple of 16"
    );
    assert!(
        validate("t", 16, 3, &fixed, &exc).is_err(),
        "odd column count"
    );
    assert!(
        validate("t", 16, 4, &fixed[1..], &exc).is_err(),
        "short fixed stream"
    );
    assert!(
        validate("t", 16, 4, &fixed, &exc[1..]).is_err(),
        "partial exception record"
    );
    let mut overflow = fixed.clone();
    overflow[5] = 241;
    assert!(
        validate("t", 16, 4, &overflow, &exc).is_err(),
        "base + 15 would wrap"
    );
    assert!(
        validate("t", 16, 4, &fixed, &rec(64, 1)).is_err(),
        "position out of range"
    );
    let unsorted = [rec(9, 1), rec(9, 2)].concat();
    assert!(
        validate("t", 16, 4, &fixed, &unsorted).is_err(),
        "duplicate position"
    );
    let descending = [rec(10, 1), rec(3, 2)].concat();
    assert!(
        validate("t", 16, 4, &fixed, &descending).is_err(),
        "descending positions"
    );
    assert!(
        validate("t", 4096 * 16, 258, &[], &[]).is_err(),
        "more than 2^24 positions"
    );
}

#[test]
fn base_240_with_code_15_is_the_largest_valid_byte() {
    let (mut fixed, _) = sample();
    fixed[0] = 240;
    fixed[16] = 0xFF;
    validate("t", 16, 4, &fixed, &[]).expect("240 is allowed");
    assert_eq!(decode(16, 4, &fixed, &[])[0], 255);
}

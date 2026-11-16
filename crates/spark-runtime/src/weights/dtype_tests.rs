// SPDX-License-Identifier: AGPL-3.0-only

use super::WeightDtype;

const ALL: [WeightDtype; 7] = [
    WeightDtype::BF16,
    WeightDtype::FP32,
    WeightDtype::FP8E4M3,
    WeightDtype::FP8E8M0,
    WeightDtype::UInt8,
    WeightDtype::Int64,
    WeightDtype::UInt32,
];

#[test]
fn manifest_codes_round_trip_and_are_unique() {
    let mut seen = std::collections::HashSet::new();
    for d in ALL {
        assert_eq!(WeightDtype::from_code(d.to_code()).expect("known code"), d);
        assert!(
            seen.insert(d.to_code()),
            "duplicate manifest code for {d:?}"
        );
    }
    assert!(WeightDtype::from_code(ALL.len() as u8).is_err());
}

#[test]
fn established_manifest_codes_are_unchanged() {
    // Snapshots on disk carry these numbers.
    assert_eq!(WeightDtype::BF16.to_code(), 0);
    assert_eq!(WeightDtype::FP32.to_code(), 1);
    assert_eq!(WeightDtype::FP8E4M3.to_code(), 2);
    assert_eq!(WeightDtype::FP8E8M0.to_code(), 3);
    assert_eq!(WeightDtype::UInt8.to_code(), 4);
    assert_eq!(WeightDtype::Int64.to_code(), 5);
}

#[test]
fn u32_tensors_are_accepted() {
    // Exception records of csf-compressed NVFP4 scales.
    let d = WeightDtype::from_safetensors(safetensors::Dtype::U32).expect("U32 is supported");
    assert_eq!(d, WeightDtype::UInt32);
    assert_eq!(d.byte_size(), 4);
}

#[test]
fn unsupported_dtypes_are_still_rejected() {
    assert!(WeightDtype::from_safetensors(safetensors::Dtype::F16).is_err());
}

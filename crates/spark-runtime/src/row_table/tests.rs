// SPDX-License-Identifier: AGPL-3.0-only

use super::direct::RowEngine;
use super::*;
use std::io::Write;

/// Deterministic file content: byte i = hash of i.
fn content(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i as u64).wrapping_mul(0x9E3779B97F4A7C15) >> 56) as u8)
        .collect()
}

/// Two planes in one file at unaligned offsets (rows of 80 and 10 bytes, like
/// the n-gram table) and a third in a second file ending exactly at EOF.
struct Fixture {
    _dir: tempfile::TempDir,
    planes: Vec<RowPlane>,
    data: Vec<Vec<u8>>, // file contents by plane
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let a = content(3 * 4096 + 123 + 300 * 80 + 300 * 10);
    let b = content(77 + 50 * 4096);
    let (pa, pb) = (dir.path().join("a.bin"), dir.path().join("b.bin"));
    std::fs::File::create(&pa)
        .and_then(|mut f| f.write_all(&a))
        .expect("write a");
    std::fs::File::create(&pb)
        .and_then(|mut f| f.write_all(&b))
        .expect("write b");
    let planes = vec![
        RowPlane {
            path: pa.clone(),
            data_offset: 4096 + 123,
            row_bytes: 80,
            rows: 300,
        },
        RowPlane {
            path: pa,
            data_offset: 4096 + 123 + 300 * 80,
            row_bytes: 10,
            rows: 300,
        },
        RowPlane {
            path: pb,
            data_offset: 77,
            row_bytes: 4096,
            rows: 50,
        },
    ];
    Fixture {
        _dir: dir,
        planes,
        data: vec![a.clone(), a, b],
    }
}

fn expected(fx: &Fixture, reads: &[RowRead]) -> Vec<u8> {
    let mut out = Vec::new();
    for r in reads {
        let p = &fx.planes[r.plane as usize];
        let start = (p.data_offset + r.row * p.row_bytes as u64) as usize;
        out.extend_from_slice(&fx.data[r.plane as usize][start..start + p.row_bytes]);
    }
    out
}

fn reads() -> Vec<RowRead> {
    let mut v = vec![
        RowRead { plane: 0, row: 0 },
        RowRead { plane: 1, row: 299 },
        RowRead { plane: 2, row: 49 }, // ends at EOF
        RowRead { plane: 2, row: 0 },  // spans two blocks
        RowRead { plane: 0, row: 299 },
        RowRead { plane: 0, row: 0 }, // repeated row
    ];
    // More than one queue depth, with rows that straddle block boundaries.
    v.extend((0..200).map(|i| RowRead {
        plane: (i % 2) as u32,
        row: (i * 37 % 300) as u64,
    }));
    v
}

fn read_all(src: &mut dyn RowSource, reads: &[RowRead]) -> Result<Vec<u8>> {
    let len = reads
        .iter()
        .map(|r| src.row_bytes(r.plane))
        .sum::<Result<usize>>()?;
    let mut out = vec![0u8; len];
    src.read_rows(reads, &mut out)?;
    Ok(out)
}

#[test]
fn direct_rows_return_the_file_bytes_with_either_engine() {
    let fx = fixture();
    let reads = reads();
    let want = expected(&fx, &reads);
    let mut src = DirectRows::open(fx.planes.clone()).expect("open");
    let first = src.engine();
    println!("engine selected at open: {first:?}");
    assert_eq!(
        read_all(&mut src, &reads).expect("read"),
        want,
        "engine {first:?}"
    );
    src.use_pread().expect("pread pool");
    assert_eq!(src.engine(), RowEngine::Pread);
    assert_eq!(
        read_all(&mut src, &reads).expect("read"),
        want,
        "pread engine"
    );
}

#[test]
fn memory_rows_agree_with_direct_rows() {
    let fx = fixture();
    let reads = reads();
    let planes = fx
        .planes
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let start = p.data_offset as usize;
            let end = start + p.rows as usize * p.row_bytes;
            (p.row_bytes, fx.data[i][start..end].to_vec())
        })
        .collect();
    let mut mem = MemoryRows::new(planes).expect("planes");
    assert_eq!(
        read_all(&mut mem, &reads).expect("read"),
        expected(&fx, &reads)
    );
}

#[test]
fn empty_request_reads_nothing() {
    let fx = fixture();
    let mut src = DirectRows::open(fx.planes.clone()).expect("open");
    src.read_rows(&[], &mut []).expect("empty read");
}

#[test]
fn bad_requests_are_rejected() {
    let fx = fixture();
    let mut src = DirectRows::open(fx.planes.clone()).expect("open");
    let mut buf = vec![0u8; 80];
    assert!(
        src.read_rows(&[RowRead { plane: 0, row: 300 }], &mut buf)
            .is_err(),
        "row past the plane"
    );
    assert!(
        src.read_rows(&[RowRead { plane: 3, row: 0 }], &mut buf)
            .is_err(),
        "unknown plane"
    );
    assert!(
        src.read_rows(&[RowRead { plane: 0, row: 1 }], &mut buf[..79])
            .is_err(),
        "short buffer"
    );
    assert!(src.row_bytes(9).is_err());
}

#[test]
fn planes_that_do_not_fit_their_file_are_rejected_at_open() {
    let fx = fixture();
    let mut too_long = fx.planes.clone();
    too_long[2].rows = 51;
    assert!(DirectRows::open(too_long).is_err());
    let mut wide = fx.planes.clone();
    wide[0].row_bytes = 4097;
    assert!(DirectRows::open(wide).is_err());
    let mut missing = fx.planes.clone();
    missing[0].path = fx.planes[0].path.with_file_name("absent.bin");
    assert!(DirectRows::open(missing).is_err());
}

#[test]
fn memory_rows_reject_partial_rows() {
    assert!(MemoryRows::new(vec![(80, vec![0u8; 81])]).is_err());
    assert!(MemoryRows::new(vec![(0, vec![])]).is_err());
}

// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the scheduler-facing NVMe tier wrapper: the BAT-6 restore
//! probe, the NVM-6 chain cut on invalid records, the KV-7 evict guard,
//! the NVM-11 retry buffer and the NVM-2 pre-unload flush. The store's
//! own format/GC/index tests live in `spark-storage`.
//! Catalogue: T-NVM-03, T-NVM-04, T-NVM-07 (restart), T-NVM-12 (RAM-side).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use spark_storage::block_store::{HEADER_LEN, Kind, WriteJob};

use super::{KvTier, KvTierIdentity, KvTierParams};

fn params(root: &Path, queue_bytes: u64) -> KvTierParams {
    KvTierParams {
        root: root.to_path_buf(),
        session_ttl: Duration::from_secs(3600),
        gc_interval: Duration::from_secs(3600),
        min_free_pct: 0,
        write_queue_max_bytes: queue_bytes,
        write_budget_per_day: None,
        persist_timeout: Duration::from_secs(5),
        restore_min_tokens: 0,
    }
}

fn identity() -> KvTierIdentity {
    KvTierIdentity {
        model_hash: [7; 32],
        layout_hash: [9; 32],
    }
}

fn open(root: &Path, queue_bytes: u64) -> Arc<KvTier> {
    KvTier::open(&params(root, queue_bytes), &identity()).unwrap()
}

fn key(n: u8) -> [u8; 16] {
    [n; 16]
}

fn payload(n: u8) -> Vec<u8> {
    vec![n; 512]
}

fn put(tier: &KvTier, kind: Kind, n: u8, pred: u8) {
    tier.store()
        .put(kind, key(n), key(pred), 16, &payload(n))
        .unwrap();
}

fn job(n: u8) -> WriteJob {
    WriteJob {
        kind: Kind::Kv,
        key: key(n),
        pred: key(n.saturating_sub(1)),
        token_count: 16,
        payload: payload(n),
    }
}

/// Locate a file under `root` by name predicate (the store's layout is
/// private; the tests only need "the file for key n").
fn find_file(root: &Path, pred: &dyn Fn(&str) -> bool) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let p = entry.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().and_then(|n| n.to_str()).is_some_and(pred) {
                return Some(p);
            }
        }
    }
    None
}

fn record_file(root: &Path, n: u8) -> PathBuf {
    let hex: String = key(n).iter().map(|b| format!("{b:02x}")).collect();
    find_file(root, &|name| name == format!("{hex}.kvb")).expect("record file")
}

/// BAT-6: the probe counts the contiguous KV prefix only — a gap or a
/// non-KV record at a position ends the count.
#[test]
fn probe_restore_counts_only_the_contiguous_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let tier = open(dir.path(), 1 << 20);
    put(&tier, Kind::Kv, 1, 0);
    put(&tier, Kind::Kv, 2, 1);
    put(&tier, Kind::Tail, 3, 2); // a tail record is not a KV block
    put(&tier, Kind::Kv, 4, 3);
    assert_eq!(tier.probe_restore(&[key(1), key(2), key(3), key(4)]), 2);
    assert_eq!(tier.probe_restore(&[key(3), key(4)]), 0);
    assert_eq!(tier.probe_restore(&[key(4)]), 1);
    assert_eq!(tier.probe_restore(&[]), 0);
}

/// T-NVM-03 / T-NVM-04: a flipped payload byte or a truncated file ends
/// the restore chain BEFORE that block, deletes the record, and leaves
/// the rest to recomputation. Nothing invalid is ever returned.
#[test]
fn fetch_blocks_ends_the_chain_at_an_invalid_record() {
    let dir = tempfile::tempdir().unwrap();
    let tier = open(dir.path(), 1 << 20);
    for n in 1..=3 {
        put(&tier, Kind::Kv, n, n - 1);
    }
    // Bit flip inside the payload of block 2.
    let p2 = record_file(dir.path(), 2);
    let mut bytes = std::fs::read(&p2).unwrap();
    bytes[HEADER_LEN + 100] ^= 0x01;
    std::fs::write(&p2, &bytes).unwrap();

    let got = KvTier::fetch_blocks(tier.store(), 10, &[key(1), key(2), key(3)]);
    assert_eq!(got.len(), 1, "chain must end before the corrupted block");
    assert_eq!(got[0].0, 10);
    assert_eq!(got[0].1, payload(1));
    assert!(!tier.store().contains(&key(2), Kind::Kv));
    assert!(!p2.exists(), "corrupted record must be deleted");
    assert!(
        tier.store().contains(&key(3), Kind::Kv),
        "untouched successor stays for the GC"
    );

    // Truncated file (crash artefact): same verdict.
    let p3 = record_file(dir.path(), 3);
    let bytes = std::fs::read(&p3).unwrap();
    std::fs::write(&p3, &bytes[..HEADER_LEN + 100]).unwrap();
    assert!(KvTier::fetch_blocks(tier.store(), 12, &[key(3)]).is_empty());
    assert!(!p3.exists());
    assert!(!tier.store().contains(&key(3), Kind::Kv));
}

/// KV-7: the evict guard answers "is persisted" per record kind.
#[test]
fn evict_guard_is_kind_aware() {
    let dir = tempfile::tempdir().unwrap();
    let tier = open(dir.path(), 1 << 20);
    put(&tier, Kind::Kv, 1, 0);
    put(&tier, Kind::Tail, 2, 1);
    let guard = tier.evict_guard();
    assert!(guard(&key(1), false));
    assert!(!guard(&key(1), true));
    assert!(guard(&key(2), true));
    assert!(!guard(&key(2), false));
    assert!(!guard(&key(9), false));
}

/// NVM-11: a full write queue never blocks the caller — rejected jobs
/// are parked in the retry buffer and re-offered by `pump_retries` until
/// every block is on disk.
#[test]
fn rejected_writes_are_parked_and_pumped_later() {
    let dir = tempfile::tempdir().unwrap();
    // Queue holds at most ~one 512 B payload at a time.
    let tier = open(dir.path(), 600);
    for n in 1..=8 {
        tier.submit(job(n));
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        tier.pump_retries();
        let parked = tier.retry.lock().unwrap().jobs.len();
        if parked == 0 {
            break;
        }
        assert!(Instant::now() < deadline, "retry buffer never drained");
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(tier.flush());
    for n in 1..=8 {
        assert!(tier.store().contains(&key(n), Kind::Kv), "block {n}");
    }
    let retry = tier.retry.lock().unwrap();
    assert_eq!(retry.bytes, 0);
    assert_eq!(retry.dropped, 0);
}

/// NVM-2 / T-NVM-07: `flush` drains the queue within `persist_timeout`
/// and persists the index; a fresh tier on the same directory (process
/// restart) sees the whole chain.
#[test]
fn flush_persists_the_index_for_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    {
        let tier = open(dir.path(), 1 << 20);
        for n in 1..=3 {
            tier.submit(job(n));
        }
        assert!(tier.flush());
        assert!(find_file(dir.path(), &|n| n == "index.atlbsix").is_some());
        let (stats, queued, dropped) = tier.stats();
        assert_eq!(stats.entries, 3);
        assert_eq!((queued, dropped), (0, 0));
    }
    let reopened = open(dir.path(), 1 << 20);
    assert_eq!(reopened.probe_restore(&[key(1), key(2), key(3), key(4)]), 3);
    assert_eq!(reopened.stats().0.entries, 3);
}

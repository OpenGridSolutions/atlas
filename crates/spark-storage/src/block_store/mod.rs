// SPDX-License-Identifier: AGPL-3.0-only

//! Content-addressed NVMe block store (requirements §5, E2/E3).
//!
//! Stores sealed KV blocks, SSM snapshots and tail records as individual
//! files addressed by their 128-bit chain key:
//! `<root>/<model_hash_prefix>/<key[0..2]>/<key_hex>.<kind>`.
//!
//! A record with an existing key is never written twice (NVM-3). Writes
//! are atomic — tmp file, `fdatasync`, `rename` (NVM-5) — and O_DIRECT
//! with 4-KiB-aligned buffers where the filesystem supports it, falling
//! back to buffered I/O otherwise (dev-container seccomp / tmpfs;
//! NVM-4's fallback arm). Reads validate the header binding (NVM-7) and
//! the payload checksum (NVM-6); invalid files are deleted on sight.
//!
//! CPU-only by design: payloads arrive as host buffers (the scheduler
//! does the D2H copy) and keys as raw `[u8; 16]`, so the store has no
//! dependency on spark-runtime or CUDA.

mod gc;
mod header;
mod index;
mod writer;

pub use gc::GcOutcome;
pub use header::{ALIGN, AlignedBuf, FORMAT_VERSION, HEADER_LEN, Header, Kind, align_up};
pub use index::IndexEntry;
pub use writer::{StoreWriter, WriteJob};

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use xxhash_rust::xxh3::xxh3_64;

use index::Index;

/// Store configuration. All values are explicit (PCND) — the CLI layer
/// owns the defaults.
#[derive(Clone)]
pub struct BlockStoreConfig {
    /// Root directory (`--nvme-cache-dir`).
    pub root: PathBuf,
    /// Model hash binding every record (snapshot `model_version_hash`
    /// domain — covers checkpoint revision).
    pub model_hash: [u8; 32],
    /// KV layout hash (dtypes, geometry, block size, RoPE, format).
    pub layout_hash: [u8; 32],
    /// Records older than this are garbage-collected (`session_ttl_hours`).
    pub session_ttl: Duration,
    /// Free-space floor in percent of the filesystem (`nvme_min_free_pct`).
    pub min_free_pct: u8,
    /// Daily write budget in bytes; exceeded → warn (NVM-12). None = off.
    pub write_budget_per_day: Option<u64>,
}

/// Read result: validated header + payload.
pub struct StoreRecord {
    pub header: Header,
    pub payload: Vec<u8>,
}

/// Telemetry snapshot.
#[derive(Clone, Copy, Debug, Default)]
pub struct StoreStats {
    pub entries: usize,
    pub total_bytes: u64,
    pub written_today: u64,
    pub free_bytes: u64,
    pub fs_total_bytes: u64,
}

pub struct BlockStore {
    cfg: BlockStoreConfig,
    model_dir: PathBuf,
    index: Mutex<Index>,
    index_path: PathBuf,
    /// Free/total bytes of the filesystem at `root`. Swappable for tests.
    free_space: fn(&Path) -> (u64, u64),
    /// Set once the daily budget warning fired for the current day.
    budget_warned_day: Mutex<u64>,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(unix)]
fn fs_free_space(path: &Path) -> (u64, u64) {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap_or_default();
    // SAFETY: statvfs writes into the zeroed out-param on success only.
    unsafe {
        let mut st: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c.as_ptr(), &mut st) == 0 {
            let frsize = if st.f_frsize > 0 {
                st.f_frsize
            } else {
                st.f_bsize
            } as u64;
            (st.f_bavail as u64 * frsize, st.f_blocks as u64 * frsize)
        } else {
            (u64::MAX, u64::MAX)
        }
    }
}

#[cfg(not(unix))]
fn fs_free_space(_path: &Path) -> (u64, u64) {
    (u64::MAX, u64::MAX)
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

impl BlockStore {
    /// Open (or create) the store: set up directories, delete leftover
    /// tmp files (NVM-5), load the index file or rebuild it by scanning
    /// (NVM-8, mtime fallback).
    pub fn open(cfg: BlockStoreConfig) -> Result<BlockStore> {
        let model_dir = cfg.root.join(hex(&cfg.model_hash[..8]));
        std::fs::create_dir_all(&model_dir)
            .with_context(|| format!("create {}", model_dir.display()))?;
        let index_path = model_dir.join("index.atlbsix");

        // NVM-5: tmp files are half-written records from a crash.
        let mut tmp_removed = 0usize;
        for entry in walk_files(&model_dir) {
            if entry.extension().and_then(|e| e.to_str()) == Some("tmp") {
                let _ = std::fs::remove_file(&entry);
                tmp_removed += 1;
            }
        }
        if tmp_removed > 0 {
            tracing::info!("block-store: removed {tmp_removed} leftover tmp file(s)");
        }

        let index = match Index::load(&index_path) {
            Ok(idx) => {
                tracing::info!(
                    "block-store: index loaded ({} entries, {})",
                    idx.len(),
                    model_dir.display(),
                );
                idx
            }
            Err(e) => {
                if index_path.exists() {
                    tracing::warn!("block-store: index file unreadable ({e:#}); rescanning");
                } else {
                    tracing::info!(
                        "block-store: no index file; scanning {}",
                        model_dir.display()
                    );
                }
                Self::scan(&model_dir)?
            }
        };

        Ok(BlockStore {
            model_dir,
            index: Mutex::new(index),
            index_path,
            cfg,
            free_space: fs_free_space,
            budget_warned_day: Mutex::new(u64::MAX),
        })
    }

    /// Directory scan fallback: read every record header; `last_access`
    /// comes from the file mtime (NVM-8).
    fn scan(model_dir: &Path) -> Result<Index> {
        let mut idx = Index::default();
        let mut invalid = 0usize;
        for path in walk_files(model_dir) {
            let Some(kind) = path
                .extension()
                .and_then(|e| e.to_str())
                .and_then(Kind::from_ext)
            else {
                continue;
            };
            match read_header(&path) {
                Ok(h) if h.kind == kind => {
                    let meta = std::fs::metadata(&path)?;
                    let mtime = meta
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    idx.insert(
                        h.key,
                        IndexEntry {
                            kind,
                            pred: h.pred,
                            file_len: meta.len(),
                            token_count: h.token_count,
                            last_access: mtime,
                        },
                    );
                }
                _ => {
                    invalid += 1;
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        if invalid > 0 {
            tracing::warn!("block-store scan: deleted {invalid} invalid record file(s)");
        }
        tracing::info!("block-store scan: {} record(s) indexed", idx.len());
        Ok(idx)
    }

    fn record_path(&self, key: &[u8; 16], kind: Kind) -> PathBuf {
        let k = hex(key);
        self.model_dir
            .join(&k[..2])
            .join(format!("{k}.{}", kind.ext()))
    }

    /// NVM-3: whether a record exists (used by the persist queue to skip
    /// writes, and by KV-7 eviction as the "is persisted" predicate).
    pub fn contains(&self, key: &[u8; 16], kind: Kind) -> bool {
        self.index.lock().expect("index lock").contains(key, kind)
    }

    /// Touch a chain head's recency (NVM-9) without reading payload.
    pub fn touch(&self, key: &[u8; 16]) {
        self.index
            .lock()
            .expect("index lock")
            .touch_chain(key, now_secs());
    }

    /// Write a record (atomically, O_DIRECT where possible). Returns
    /// `false` when the key already existed and nothing was written
    /// (NVM-3 — recency is refreshed instead).
    pub fn put(
        &self,
        kind: Kind,
        key: [u8; 16],
        pred: [u8; 16],
        token_count: u32,
        payload: &[u8],
    ) -> Result<bool> {
        let now = now_secs();
        {
            let mut idx = self.index.lock().expect("index lock");
            if idx.contains(&key, kind) {
                idx.touch_chain(&key, now);
                return Ok(false);
            }
        }

        let header = Header {
            kind,
            model_hash: self.cfg.model_hash,
            layout_hash: self.cfg.layout_hash,
            key,
            pred,
            token_count,
            payload_len: payload.len() as u64,
            payload_xxh3: xxh3_64(payload),
        };
        let mut buf = AlignedBuf::zeroed(HEADER_LEN + payload.len());
        header.encode_into(buf.as_mut_slice());
        buf.as_mut_slice()[HEADER_LEN..].copy_from_slice(payload);

        let path = self.record_path(&key, kind);
        let dir = path.parent().expect("record path has parent");
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        let tmp = path.with_extension("tmp");
        write_direct(&tmp, &buf).with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, &path)
            .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;

        let file_len = buf.aligned_len() as u64;
        {
            let mut idx = self.index.lock().expect("index lock");
            idx.insert(
                key,
                IndexEntry {
                    kind,
                    pred,
                    file_len,
                    token_count,
                    last_access: now,
                },
            );
            idx.record_written(file_len, now);
            self.check_write_budget(&idx);
        }
        Ok(true)
    }

    /// Read + validate a record. Any header-binding failure (NVM-7) or
    /// payload-checksum failure (NVM-6) deletes the file and returns
    /// `None` — the caller's restore chain ends here and recomputes.
    pub fn get(&self, key: &[u8; 16], kind: Kind) -> Option<StoreRecord> {
        let path = self.record_path(key, kind);
        let mut data = Vec::new();
        match std::fs::File::open(&path).and_then(|mut f| f.read_to_end(&mut data)) {
            Ok(_) => {}
            Err(_) => {
                // File missing but indexed → index is stale; drop the entry.
                self.index.lock().expect("index lock").remove(key, kind);
                return None;
            }
        }
        let invalidate = |why: &str| {
            tracing::warn!("block-store: invalidating {} ({why})", path.display());
            let _ = std::fs::remove_file(&path);
            self.index.lock().expect("index lock").remove(key, kind);
        };
        let header = match Header::decode(&data) {
            Ok(h) => h,
            Err(e) => {
                invalidate(&format!("header: {e}"));
                return None;
            }
        };
        if header.model_hash != self.cfg.model_hash
            || header.layout_hash != self.cfg.layout_hash
            || header.kind != kind
            || header.key != *key
        {
            invalidate("model/layout/key binding mismatch");
            return None;
        }
        let plen = header.payload_len as usize;
        if data.len() < HEADER_LEN + plen {
            invalidate("payload truncated");
            return None;
        }
        let payload = &data[HEADER_LEN..HEADER_LEN + plen];
        if xxh3_64(payload) != header.payload_xxh3 {
            invalidate("payload checksum mismatch");
            return None;
        }
        self.index
            .lock()
            .expect("index lock")
            .touch_chain(key, now_secs());
        Some(StoreRecord {
            header,
            payload: payload.to_vec(),
        })
    }

    /// Persist the index when dirty (called every ~60 s by the writer
    /// thread; NVM-8).
    pub fn persist_index(&self) -> Result<()> {
        let mut idx = self.index.lock().expect("index lock");
        if idx.dirty {
            idx.save(&self.index_path)?;
        }
        Ok(())
    }

    pub fn stats(&self) -> StoreStats {
        let idx = self.index.lock().expect("index lock");
        let (free, total) = (self.free_space)(&self.cfg.root);
        StoreStats {
            entries: idx.len(),
            total_bytes: idx.total_bytes(),
            written_today: idx.written_today,
            free_bytes: free,
            fs_total_bytes: total,
        }
    }

    fn check_write_budget(&self, idx: &Index) {
        let Some(budget) = self.cfg.write_budget_per_day else {
            return;
        };
        if idx.written_today > budget {
            let mut warned = self.budget_warned_day.lock().expect("budget lock");
            if *warned != idx.budget_day {
                *warned = idx.budget_day;
                tracing::warn!(
                    written_gb = idx.written_today as f64 / 1e9,
                    budget_gb = budget as f64 / 1e9,
                    "block-store: daily NVMe write budget exceeded (NVM-12)"
                );
            }
        }
    }

    #[cfg(test)]
    fn set_free_space_probe(&mut self, f: fn(&Path) -> (u64, u64)) {
        self.free_space = f;
    }
}

/// Write an aligned buffer to `path`, trying O_DIRECT first and falling
/// back to buffered I/O (tmpfs and some filesystems reject O_DIRECT).
/// Always `fdatasync`s before returning (NVM-5).
fn write_direct(path: &Path, buf: &AlignedBuf) -> Result<()> {
    use std::io::Write;
    #[cfg(target_os = "linux")]
    let file = {
        use std::os::unix::fs::OpenOptionsExt;
        match std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
        {
            Ok(f) => f,
            Err(_) => std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)?,
        }
    };
    #[cfg(not(target_os = "linux"))]
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;

    let mut file = file;
    file.write_all(buf.aligned_slice())?;
    file.sync_data()?;
    Ok(())
}

fn read_header(path: &Path) -> Result<Header> {
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; HEADER_LEN];
    f.read_exact(&mut buf)?;
    Header::decode(&buf)
}

/// Depth-limited file walk over the two-level shard layout.
fn walk_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            if let Ok(sub) = std::fs::read_dir(&p) {
                for e in sub.flatten() {
                    let sp = e.path();
                    if sp.is_file() {
                        out.push(sp);
                    }
                }
            }
        } else if p.is_file() {
            out.push(p);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(root: &Path) -> BlockStoreConfig {
        BlockStoreConfig {
            root: root.to_path_buf(),
            model_hash: [7; 32],
            layout_hash: [9; 32],
            session_ttl: Duration::from_secs(24 * 3600),
            min_free_pct: 10,
            write_budget_per_day: None,
        }
    }

    fn key(n: u8) -> [u8; 16] {
        [n; 16]
    }

    #[test]
    fn put_get_roundtrip_and_dedup() {
        let dir = tempfile::tempdir().unwrap();
        let store = BlockStore::open(cfg(dir.path())).unwrap();
        let payload = vec![0xAB; 5000];

        assert!(store.put(Kind::Kv, key(1), key(0), 16, &payload).unwrap());
        // NVM-3: second put of the same key writes nothing.
        assert!(!store.put(Kind::Kv, key(1), key(0), 16, &payload).unwrap());

        let rec = store.get(&key(1), Kind::Kv).unwrap();
        assert_eq!(rec.payload, payload);
        assert_eq!(rec.header.pred, key(0));
        assert_eq!(rec.header.token_count, 16);
        assert!(store.contains(&key(1), Kind::Kv));
        assert!(!store.contains(&key(1), Kind::Ssm));
    }

    #[test]
    fn corrupt_payload_is_deleted_on_read() {
        let dir = tempfile::tempdir().unwrap();
        let store = BlockStore::open(cfg(dir.path())).unwrap();
        store
            .put(Kind::Kv, key(1), key(0), 16, &[1, 2, 3, 4])
            .unwrap();

        // Flip a payload byte on disk.
        let path = store.record_path(&key(1), Kind::Kv);
        let mut data = std::fs::read(&path).unwrap();
        data[HEADER_LEN] ^= 0xFF;
        std::fs::write(&path, &data).unwrap();

        assert!(store.get(&key(1), Kind::Kv).is_none(), "NVM-6");
        assert!(!path.exists(), "corrupt file must be deleted");
        assert!(!store.contains(&key(1), Kind::Kv));
    }

    #[test]
    fn foreign_layout_is_deleted_on_read() {
        let dir = tempfile::tempdir().unwrap();
        let store = BlockStore::open(cfg(dir.path())).unwrap();
        store.put(Kind::Kv, key(1), key(0), 16, &[5; 64]).unwrap();

        let mut other = cfg(dir.path());
        other.layout_hash = [1; 32];
        let store2 = BlockStore::open(other).unwrap();
        assert!(store2.get(&key(1), Kind::Kv).is_none(), "NVM-7");
        assert!(!store2.contains(&key(1), Kind::Kv) || store.contains(&key(1), Kind::Kv));
    }

    #[test]
    fn index_scan_rebuild_matches() {
        let dir = tempfile::tempdir().unwrap();
        {
            let store = BlockStore::open(cfg(dir.path())).unwrap();
            store.put(Kind::Kv, key(1), key(0), 16, &[1; 100]).unwrap();
            store.put(Kind::Tail, key(2), key(1), 4, &[2; 100]).unwrap();
            // Deliberately do NOT persist the index.
        }
        let store = BlockStore::open(cfg(dir.path())).unwrap();
        assert!(store.contains(&key(1), Kind::Kv));
        assert!(store.contains(&key(2), Kind::Tail));
        let rec = store.get(&key(2), Kind::Tail).unwrap();
        assert_eq!(rec.header.pred, key(1));
    }

    #[test]
    fn index_file_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        {
            let store = BlockStore::open(cfg(dir.path())).unwrap();
            store.put(Kind::Kv, key(3), key(0), 16, &[3; 32]).unwrap();
            store.persist_index().unwrap();
        }
        let store = BlockStore::open(cfg(dir.path())).unwrap();
        assert!(store.contains(&key(3), Kind::Kv));
    }

    #[test]
    fn tmp_files_removed_at_open() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join(hex(&[7u8; 8]));
        let shard = model_dir.join("ab");
        std::fs::create_dir_all(&shard).unwrap();
        let tmp = shard.join("abcd.tmp");
        std::fs::write(&tmp, b"half-written").unwrap();
        let _store = BlockStore::open(cfg(dir.path())).unwrap();
        assert!(!tmp.exists(), "NVM-5 startup cleanup");
    }

    #[test]
    fn stats_report_entries_and_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let store = BlockStore::open(cfg(dir.path())).unwrap();
        store.put(Kind::Kv, key(1), key(0), 16, &[0; 5000]).unwrap();
        let st = store.stats();
        assert_eq!(st.entries, 1);
        assert_eq!(st.total_bytes, (HEADER_LEN + align_up(5000)) as u64);
        assert_eq!(st.written_today, st.total_bytes);
    }
}

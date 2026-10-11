// SPDX-License-Identifier: AGPL-3.0-only

//! Content-addressed result cache for `/convert` (design doc §2b.1).
//!
//! The cache is what makes block ids stable across runs of the same
//! document: a hit returns the stored result byte-for-byte, no
//! inference, no GPU rounding. Key = BLAKE3 over (file bytes, model id,
//! prompt type, page selection). `skip_cache=true` bypasses the lookup
//! and refreshes the entry.
//!
//! SBIO: the worker only sees the [`ResultCache`] trait; the filesystem
//! lives in [`FsResultCache`]. Entries are one file each under a 2-hex
//! fan-out directory, expire by mtime after `ttl`, and the directory is
//! pruned oldest-first when it grows past `max_bytes`.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

pub trait ResultCache: Send + Sync {
    fn get(&self, key: &str) -> Option<Vec<u8>>;
    fn put(&self, key: &str, bytes: &[u8]);
}

/// Cache key for one submission. Any input that changes the output is
/// part of it; `max_pages`/`page_range` are folded into `pages`.
pub fn cache_key(file: &[u8], model_id: &str, prompt_type: &str, pages: &[usize]) -> String {
    let mut h = blake3::Hasher::new();
    h.update(b"atlas-convert-v1\0");
    h.update(model_id.as_bytes());
    h.update(b"\0");
    h.update(prompt_type.as_bytes());
    h.update(b"\0");
    for p in pages {
        h.update(&(*p as u64).to_le_bytes());
    }
    h.update(b"\0");
    h.update(file);
    h.finalize().to_hex().to_string()
}

/// Per-page seed base so degeneration retries sample deterministically
/// for the same page bytes (design doc §2b.1, point 3).
pub fn page_seed(file_key: &str, page_idx: usize) -> u64 {
    let mut h = blake3::Hasher::new();
    h.update(file_key.as_bytes());
    h.update(&(page_idx as u64).to_le_bytes());
    let bytes = h.finalize();
    u64::from_le_bytes(bytes.as_bytes()[..8].try_into().expect("8 bytes"))
}

pub struct FsResultCache {
    dir: PathBuf,
    ttl: Duration,
    max_bytes: u64,
}

impl FsResultCache {
    pub fn open(dir: &Path, ttl: Duration, max_bytes: u64) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        // Fail at boot, not at the first job, if the directory is read-only.
        let probe = dir.join(".write-probe");
        std::fs::write(&probe, b"ok")?;
        let _ = std::fs::remove_file(&probe);
        Ok(Self {
            dir: dir.to_path_buf(),
            ttl,
            max_bytes,
        })
    }

    fn path_for(&self, key: &str) -> PathBuf {
        let (a, _) = key.split_at(2.min(key.len()));
        self.dir.join(a).join(format!("{key}.json"))
    }

    /// Oldest-first prune until the directory fits `max_bytes`.
    fn prune(&self) {
        let mut entries: Vec<(SystemTime, u64, PathBuf)> = Vec::new();
        let Ok(fanout) = std::fs::read_dir(&self.dir) else {
            return;
        };
        for sub in fanout.flatten() {
            let Ok(files) = std::fs::read_dir(sub.path()) else {
                continue;
            };
            for f in files.flatten() {
                if let Ok(md) = f.metadata()
                    && md.is_file()
                {
                    let mtime = md.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                    entries.push((mtime, md.len(), f.path()));
                }
            }
        }
        let mut total: u64 = entries.iter().map(|e| e.1).sum();
        if total <= self.max_bytes {
            return;
        }
        entries.sort_by_key(|e| e.0);
        for (_, len, path) in entries {
            if total <= self.max_bytes {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                total = total.saturating_sub(len);
            }
        }
    }
}

impl ResultCache for FsResultCache {
    fn get(&self, key: &str) -> Option<Vec<u8>> {
        let path = self.path_for(key);
        let md = std::fs::metadata(&path).ok()?;
        let age = md.modified().ok()?.elapsed().unwrap_or(Duration::ZERO);
        if age >= self.ttl {
            let _ = std::fs::remove_file(&path);
            return None;
        }
        std::fs::read(&path).ok()
    }

    fn put(&self, key: &str, bytes: &[u8]) {
        let path = self.path_for(key);
        if let Some(parent) = path.parent()
            && std::fs::create_dir_all(parent).is_err()
        {
            return;
        }
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, bytes).is_ok() && std::fs::rename(&tmp, &path).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        self.prune();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_covers_every_input() {
        let k = cache_key(b"pdf", "m", "ocr_layout", &[0, 1]);
        assert_eq!(k.len(), 64);
        assert_ne!(k, cache_key(b"pdf2", "m", "ocr_layout", &[0, 1]));
        assert_ne!(k, cache_key(b"pdf", "m2", "ocr_layout", &[0, 1]));
        assert_ne!(k, cache_key(b"pdf", "m", "ocr", &[0, 1]));
        assert_ne!(k, cache_key(b"pdf", "m", "ocr_layout", &[0]));
        assert_eq!(k, cache_key(b"pdf", "m", "ocr_layout", &[0, 1]));
        assert_ne!(page_seed(&k, 0), page_seed(&k, 1));
        assert_eq!(page_seed(&k, 3), page_seed(&k, 3));
    }

    #[test]
    fn fs_roundtrip_ttl_and_prune() {
        let dir = tempfile::tempdir().unwrap();
        let c = FsResultCache::open(dir.path(), Duration::from_secs(60), 40).unwrap();
        c.put("aa11", b"0123456789");
        assert_eq!(c.get("aa11").unwrap(), b"0123456789");
        assert!(c.get("zz99").is_none());
        // Four 10-byte entries fit exactly; the fifth prunes the oldest.
        for k in ["bb22", "cc33", "dd44"] {
            c.put(k, b"0123456789");
        }
        std::thread::sleep(Duration::from_millis(20));
        c.put("ee55", b"0123456789");
        let present = ["aa11", "bb22", "cc33", "dd44", "ee55"]
            .iter()
            .filter(|k| c.get(k).is_some())
            .count();
        assert_eq!(present, 4);
        assert!(c.get("ee55").is_some(), "newest survives");

        // TTL 0 ⇒ everything is stale.
        let stale = FsResultCache::open(dir.path(), Duration::ZERO, 1 << 20).unwrap();
        assert!(stale.get("ee55").is_none());
    }
}

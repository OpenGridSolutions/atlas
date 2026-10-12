// SPDX-License-Identifier: AGPL-3.0-only

//! In-memory index of the NVMe block store (NVM-8/9): key → (kind, size,
//! predecessor, token count, last_access). Built at startup from the
//! persisted index file, falling back to a directory scan with file
//! mtimes; saved periodically so a crash loses at most one interval of
//! access-recency (never data — the files themselves are the truth).

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result};

use super::header::Kind;

/// Index file magic + version (independent of the record format).
const INDEX_MAGIC: &[u8; 8] = b"ATLBSIX1";
const ENTRY_LEN: usize = 1 + 16 + 16 + 8 + 4 + 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexEntry {
    pub kind: Kind,
    pub pred: [u8; 16],
    /// Total file size in bytes (header + padded payload).
    pub file_len: u64,
    pub token_count: u32,
    /// Unix seconds of the last read/write touching this record or any
    /// successor of it (NVM-9).
    pub last_access: u64,
}

#[derive(Default)]
pub struct Index {
    entries: HashMap<([u8; 16], Kind), IndexEntry>,
    /// Bytes written since `budget_day` began (NVM-12).
    pub written_today: u64,
    /// Day number (unix secs / 86400) the counter belongs to.
    pub budget_day: u64,
    /// True when the in-memory state is newer than the index file.
    pub dirty: bool,
}

impl Index {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn total_bytes(&self) -> u64 {
        self.entries.values().map(|e| e.file_len).sum()
    }

    pub fn contains(&self, key: &[u8; 16], kind: Kind) -> bool {
        self.entries.contains_key(&(*key, kind))
    }

    pub fn get(&self, key: &[u8; 16], kind: Kind) -> Option<&IndexEntry> {
        self.entries.get(&(*key, kind))
    }

    pub fn insert(&mut self, key: [u8; 16], entry: IndexEntry) {
        self.entries.insert((key, entry.kind), entry);
        self.dirty = true;
    }

    pub fn remove(&mut self, key: &[u8; 16], kind: Kind) -> Option<IndexEntry> {
        let e = self.entries.remove(&(*key, kind));
        if e.is_some() {
            self.dirty = true;
        }
        e
    }

    pub fn iter(&self) -> impl Iterator<Item = (&([u8; 16], Kind), &IndexEntry)> {
        self.entries.iter()
    }

    /// Record `bytes` written now, resetting the day counter on day roll.
    pub fn record_written(&mut self, bytes: u64, now_secs: u64) {
        let day = now_secs / 86_400;
        if day != self.budget_day {
            self.budget_day = day;
            self.written_today = 0;
        }
        self.written_today += bytes;
        self.dirty = true;
    }

    /// NVM-9: touch `key` (any kind) and every predecessor of its chain.
    /// A predecessor is thereby never older than its successors. The walk
    /// is bounded by the index size (cycle-safe: strictly decreasing
    /// remaining budget).
    pub fn touch_chain(&mut self, key: &[u8; 16], now_secs: u64) {
        let mut budget = self.entries.len() + 1;
        let mut cur = *key;
        while budget > 0 {
            budget -= 1;
            let mut pred: Option<[u8; 16]> = None;
            let mut any = false;
            for kind in [Kind::Kv, Kind::Tail, Kind::Ssm, Kind::Cmp] {
                if let Some(e) = self.entries.get_mut(&(cur, kind)) {
                    any = true;
                    if e.last_access < now_secs {
                        e.last_access = now_secs;
                        self.dirty = true;
                    }
                    if kind == Kind::Kv || pred.is_none() {
                        pred = Some(e.pred);
                    }
                }
            }
            match pred {
                Some(p) if any && p != [0u8; 16] => cur = p,
                _ => break,
            }
        }
    }

    // ── Persistence ──

    pub fn save(&mut self, path: &Path) -> Result<()> {
        let tmp = path.with_extension("tmp");
        {
            let mut f =
                std::fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
            let mut buf = Vec::with_capacity(8 + 8 + 8 + 8 + self.entries.len() * (16 + ENTRY_LEN));
            buf.extend_from_slice(INDEX_MAGIC);
            buf.extend_from_slice(&(self.entries.len() as u64).to_le_bytes());
            buf.extend_from_slice(&self.budget_day.to_le_bytes());
            buf.extend_from_slice(&self.written_today.to_le_bytes());
            for ((key, kind), e) in &self.entries {
                buf.extend_from_slice(key);
                buf.push(kind.tag());
                buf.extend_from_slice(&e.pred);
                buf.extend_from_slice(&e.file_len.to_le_bytes());
                buf.extend_from_slice(&e.token_count.to_le_bytes());
                buf.extend_from_slice(&e.last_access.to_le_bytes());
                // (ENTRY_LEN accounting: 1 kind is folded in above; pred 16,
                // len 8, tokens 4, access 8 — plus the 16-byte key prefix.)
            }
            f.write_all(&buf)?;
            f.sync_data()?;
        }
        std::fs::rename(&tmp, path)
            .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
        self.dirty = false;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Index> {
        let mut f = std::fs::File::open(path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        if buf.len() < 32 || &buf[..8] != INDEX_MAGIC {
            anyhow::bail!("bad index file");
        }
        let n = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
        let budget_day = u64::from_le_bytes(buf[16..24].try_into().unwrap());
        let written_today = u64::from_le_bytes(buf[24..32].try_into().unwrap());
        let mut entries = HashMap::with_capacity(n);
        let mut off = 32;
        for _ in 0..n {
            let rec_len = 16 + 1 + 16 + 8 + 4 + 8;
            if off + rec_len > buf.len() {
                anyhow::bail!("index file truncated");
            }
            let key: [u8; 16] = buf[off..off + 16].try_into().unwrap();
            off += 16;
            let kind = Kind::from_tag(buf[off]).ok_or_else(|| anyhow::anyhow!("bad kind"))?;
            off += 1;
            let pred: [u8; 16] = buf[off..off + 16].try_into().unwrap();
            off += 16;
            let file_len = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
            off += 8;
            let token_count = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
            off += 4;
            let last_access = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
            off += 8;
            entries.insert(
                (key, kind),
                IndexEntry {
                    kind,
                    pred,
                    file_len,
                    token_count,
                    last_access,
                },
            );
        }
        Ok(Index {
            entries,
            written_today,
            budget_day,
            dirty: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(kind: Kind, pred: [u8; 16], access: u64) -> IndexEntry {
        IndexEntry {
            kind,
            pred,
            file_len: 8192,
            token_count: 16,
            last_access: access,
        }
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.bin");
        let mut idx = Index::default();
        idx.insert([1; 16], e(Kind::Kv, [0; 16], 100));
        idx.insert([2; 16], e(Kind::Ssm, [1; 16], 200));
        idx.record_written(4096, 86_400 * 3 + 5);
        idx.save(&path).unwrap();
        assert!(!idx.dirty);

        let loaded = Index::load(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get(&[1; 16], Kind::Kv), idx.get(&[1; 16], Kind::Kv));
        assert_eq!(loaded.written_today, 4096);
        assert_eq!(loaded.budget_day, 3);
    }

    #[test]
    fn touch_chain_updates_predecessors() {
        let mut idx = Index::default();
        idx.insert([1; 16], e(Kind::Kv, [0; 16], 10));
        idx.insert([2; 16], e(Kind::Kv, [1; 16], 10));
        idx.insert([3; 16], e(Kind::Kv, [2; 16], 10));
        idx.insert([9; 16], e(Kind::Kv, [0; 16], 10)); // unrelated

        idx.touch_chain(&[3; 16], 500);
        assert_eq!(idx.get(&[3; 16], Kind::Kv).unwrap().last_access, 500);
        assert_eq!(idx.get(&[2; 16], Kind::Kv).unwrap().last_access, 500);
        assert_eq!(idx.get(&[1; 16], Kind::Kv).unwrap().last_access, 500);
        assert_eq!(idx.get(&[9; 16], Kind::Kv).unwrap().last_access, 10);
    }

    #[test]
    fn touch_chain_covers_ssm_at_same_key() {
        let mut idx = Index::default();
        idx.insert([1; 16], e(Kind::Kv, [0; 16], 10));
        idx.insert([1; 16], e(Kind::Ssm, [0; 16], 10));
        idx.touch_chain(&[1; 16], 99);
        assert_eq!(idx.get(&[1; 16], Kind::Ssm).unwrap().last_access, 99);
    }

    #[test]
    fn touch_chain_is_cycle_safe() {
        let mut idx = Index::default();
        idx.insert([1; 16], e(Kind::Kv, [2; 16], 10));
        idx.insert([2; 16], e(Kind::Kv, [1; 16], 10)); // artificial cycle
        idx.touch_chain(&[1; 16], 42); // must terminate
        assert_eq!(idx.get(&[2; 16], Kind::Kv).unwrap().last_access, 42);
    }

    #[test]
    fn day_roll_resets_budget() {
        let mut idx = Index::default();
        idx.record_written(100, 86_400);
        idx.record_written(50, 86_400 + 10);
        assert_eq!(idx.written_today, 150);
        idx.record_written(7, 2 * 86_400);
        assert_eq!(idx.written_today, 7);
    }
}

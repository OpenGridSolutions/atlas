// SPDX-License-Identifier: AGPL-3.0-only

//! ATLASRAW model snapshot format (HOTSWAP_MVP_PLAN.md Task 3).
//!
//! Layout:
//! ```text
//! [ header, fixed 4096 B                                    ]
//! [ payload: raw arena bytes [0, used), padded to 4096      ]  ← O_DIRECT
//! [ manifest: postcard-encoded Vec<ArenaTensorEntry>        ]  ← buffered
//! [ footer, last 12 B: manifest_len u32 | manifest_xxh3_lo u32 | "AMFT" ]
//! ```
//!
//! Header fields (little-endian, zero-padded to 4096):
//! `magic "ATLASRAW"` (8) · `format_version u32 = 1` ·
//! `model_version_hash [u8;32]` · `weight_payload_size u64` ·
//! `tensor_count u32` · `manifest_offset u64` · `payload_xxh3 u64`.
//!
//! The payload is written with O_DIRECT in large aligned chunks straight
//! from the arena mapping (the arena base is 4096-aligned and its capacity
//! is a 4096 multiple, so the padded tail never leaves the mapping — no
//! bounce buffer). The manifest + footer live at the unaligned tail and use
//! normal buffered I/O (plan §2).
//!
//! Reader validation order: footer magic → manifest length/checksum →
//! cross-check against header `manifest_offset`/`payload_size` → hard-reject
//! unknown `format_version`.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use xxhash_rust::xxh3::xxh3_64;

use crate::weights_arena::{ArenaTensorEntry, WeightsArena};

pub const SNAPSHOT_MAGIC: &[u8; 8] = b"ATLASRAW";
pub const FOOTER_MAGIC: u32 = u32::from_le_bytes(*b"AMFT");
pub const FORMAT_VERSION: u32 = 1;
pub const HEADER_LEN: usize = 4096;
pub const FOOTER_LEN: usize = 12;
/// Bump when the arena layout rules change (name-sorted, 256-align).
/// Feeds `model_version_hash`, so stale snapshots stop resolving.
pub const ARENA_LAYOUT_VERSION: u32 = 1;

const DIRECT_CHUNK: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotHeader {
    pub format_version: u32,
    pub model_version_hash: [u8; 32],
    pub payload_size: u64,
    pub tensor_count: u32,
    pub manifest_offset: u64,
    pub payload_xxh3: u64,
}

impl SnapshotHeader {
    fn encode(&self) -> [u8; HEADER_LEN] {
        let mut buf = [0u8; HEADER_LEN];
        buf[0..8].copy_from_slice(SNAPSHOT_MAGIC);
        buf[8..12].copy_from_slice(&self.format_version.to_le_bytes());
        buf[12..44].copy_from_slice(&self.model_version_hash);
        buf[44..52].copy_from_slice(&self.payload_size.to_le_bytes());
        buf[52..56].copy_from_slice(&self.tensor_count.to_le_bytes());
        buf[56..64].copy_from_slice(&self.manifest_offset.to_le_bytes());
        buf[64..72].copy_from_slice(&self.payload_xxh3.to_le_bytes());
        buf
    }

    fn decode(buf: &[u8]) -> Result<Self> {
        ensure!(buf.len() >= HEADER_LEN, "snapshot header truncated");
        ensure!(
            &buf[0..8] == SNAPSHOT_MAGIC,
            "bad snapshot magic (not an ATLASRAW file)"
        );
        let format_version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        // Hard reject BEFORE trusting any other field.
        if format_version != FORMAT_VERSION {
            bail!(
                "unsupported snapshot format_version {format_version} \
                 (this build reads only {FORMAT_VERSION})"
            );
        }
        let mut model_version_hash = [0u8; 32];
        model_version_hash.copy_from_slice(&buf[12..44]);
        Ok(Self {
            format_version,
            model_version_hash,
            payload_size: u64::from_le_bytes(buf[44..52].try_into().unwrap()),
            tensor_count: u32::from_le_bytes(buf[52..56].try_into().unwrap()),
            manifest_offset: u64::from_le_bytes(buf[56..64].try_into().unwrap()),
            payload_xxh3: u64::from_le_bytes(buf[64..72].try_into().unwrap()),
        })
    }
}

/// Stable identity for (model id, revision, arena layout): xxh3-128 in the
/// first 16 bytes of the 32-byte field, rest zero.
pub fn model_version_hash(model_id: &str, revision: &str) -> [u8; 32] {
    let input = format!("{model_id}\0{revision}\0layout-v{ARENA_LAYOUT_VERSION}");
    let h = xxhash_rust::xxh3::xxh3_128(input.as_bytes());
    let mut out = [0u8; 32];
    out[..16].copy_from_slice(&h.to_le_bytes());
    out
}

/// `<model_slug>-<hash8>.raw` (slug = HF id with '/' → "--", the same
/// transform the model resolver uses for cache directories).
pub fn snapshot_file_name(model_id: &str, hash: &[u8; 32]) -> String {
    let slug = model_id.replace('/', "--");
    let hash8: String = hash[..4].iter().map(|b| format!("{b:02x}")).collect();
    format!("{slug}-{hash8}.raw")
}

#[derive(Debug)]
pub struct SnapshotWriteStats {
    pub bytes: u64,
    pub payload_xxh3: u64,
    pub duration_ms: u128,
    /// False when the filesystem rejected O_DIRECT and the payload fell
    /// back to buffered writes (tmpfs/overlayfs).
    pub o_direct: bool,
}

/// Write a snapshot of the arena's used prefix plus its manifest.
///
/// Safe to call while the model is serving: weights are read-only during
/// inference. Callers must serialize against unload/reload (the admin lock).
pub fn write_snapshot(
    path: &Path,
    arena: &WeightsArena,
    manifest: &[ArenaTensorEntry],
    model_hash: [u8; 32],
) -> Result<SnapshotWriteStats> {
    let t0 = Instant::now();
    let used = arena.used();
    ensure!(used > 0, "arena is empty — nothing to snapshot");
    let padded = used.div_ceil(HEADER_LEN) * HEADER_LEN;
    ensure!(padded <= arena.capacity(), "padded payload exceeds arena");

    // Safety: read-only view of the mapping; the padded tail stays inside
    // capacity. No writer can be active (admin-op serialization).
    let payload_padded = unsafe { std::slice::from_raw_parts(arena.base().0 as *const u8, padded) };
    let payload_xxh3 = xxh3_64(&payload_padded[..used]);

    let manifest_bytes =
        postcard::to_allocvec(manifest).context("postcard-encoding snapshot manifest")?;
    let manifest_offset = (HEADER_LEN + padded) as u64;

    let header = SnapshotHeader {
        format_version: FORMAT_VERSION,
        model_version_hash: model_hash,
        payload_size: used as u64,
        tensor_count: manifest.len() as u32,
        manifest_offset,
        payload_xxh3,
    };

    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating snapshot dir {}", dir.display()))?;
    }

    // Header + payload via O_DIRECT when the filesystem allows it.
    let (file, o_direct) = match OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
    {
        Ok(f) => (f, true),
        Err(e) => {
            tracing::warn!(
                "snapshot: O_DIRECT open failed for {} ({e}); falling back to buffered payload writes",
                path.display()
            );
            (
                OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path)?,
                false,
            )
        }
    };

    // O_DIRECT needs an aligned source buffer for the header as well; the
    // encoded header array lives on the stack unaligned, so stage it at the
    // start of a page-aligned copy.
    {
        use std::os::unix::fs::FileExt;
        let encoded = header.encode();
        if o_direct {
            let mut staged = AlignedPage::new();
            staged.as_mut().copy_from_slice(&encoded);
            file.write_all_at(staged.as_ref(), 0)
                .context("writing snapshot header (O_DIRECT)")?;
        } else {
            file.write_all_at(&encoded, 0)
                .context("writing snapshot header")?;
        }

        let mut off = 0usize;
        while off < padded {
            let n = (padded - off).min(DIRECT_CHUNK);
            file.write_all_at(&payload_padded[off..off + n], (HEADER_LEN + off) as u64)
                .with_context(|| format!("writing snapshot payload at {off}"))?;
            off += n;
        }
    }
    file.sync_all()?;
    drop(file);

    // Manifest + footer at the unaligned tail: normal buffered I/O.
    let mut tail = OpenOptions::new().write(true).open(path)?;
    tail.seek(SeekFrom::Start(manifest_offset))?;
    tail.write_all(&manifest_bytes)?;
    let manifest_xxh3_lo = (xxh3_64(&manifest_bytes) & 0xFFFF_FFFF) as u32;
    tail.write_all(&(manifest_bytes.len() as u32).to_le_bytes())?;
    tail.write_all(&manifest_xxh3_lo.to_le_bytes())?;
    tail.write_all(&FOOTER_MAGIC.to_le_bytes())?;
    tail.sync_all()?;
    let bytes = tail.metadata()?.len();

    Ok(SnapshotWriteStats {
        bytes,
        payload_xxh3,
        duration_ms: t0.elapsed().as_millis(),
        o_direct,
    })
}

/// Validated header + manifest (everything except the payload bytes).
pub fn read_snapshot_meta(path: &Path) -> Result<(SnapshotHeader, Vec<ArenaTensorEntry>)> {
    let mut file =
        File::open(path).with_context(|| format!("opening snapshot {}", path.display()))?;
    let file_len = file.metadata()?.len();
    ensure!(
        file_len >= (HEADER_LEN + FOOTER_LEN) as u64,
        "snapshot too small ({file_len} B)"
    );

    // 1. Footer magic.
    let mut footer = [0u8; FOOTER_LEN];
    file.seek(SeekFrom::Start(file_len - FOOTER_LEN as u64))?;
    file.read_exact(&mut footer)?;
    let manifest_len = u32::from_le_bytes(footer[0..4].try_into().unwrap()) as u64;
    let manifest_xxh3_lo = u32::from_le_bytes(footer[4..8].try_into().unwrap());
    let footer_magic = u32::from_le_bytes(footer[8..12].try_into().unwrap());
    ensure!(footer_magic == FOOTER_MAGIC, "bad snapshot footer magic");

    // 2. Manifest length + checksum.
    ensure!(
        manifest_len + (HEADER_LEN + FOOTER_LEN) as u64 <= file_len,
        "footer manifest_len {manifest_len} inconsistent with file size {file_len}"
    );
    let manifest_start = file_len - FOOTER_LEN as u64 - manifest_len;
    let mut manifest_bytes = vec![0u8; manifest_len as usize];
    file.seek(SeekFrom::Start(manifest_start))?;
    file.read_exact(&mut manifest_bytes)?;
    let got_lo = (xxh3_64(&manifest_bytes) & 0xFFFF_FFFF) as u32;
    ensure!(
        got_lo == manifest_xxh3_lo,
        "manifest checksum mismatch: got {got_lo:#010x}, footer says {manifest_xxh3_lo:#010x}"
    );

    // 3. Header (decode hard-rejects unknown format_version) + cross-checks.
    let mut header_buf = [0u8; HEADER_LEN];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut header_buf)?;
    let header = SnapshotHeader::decode(&header_buf)?;
    ensure!(
        header.manifest_offset == manifest_start,
        "header manifest_offset {} != actual manifest start {manifest_start}",
        header.manifest_offset
    );
    let padded = header.payload_size.div_ceil(HEADER_LEN as u64) * HEADER_LEN as u64;
    ensure!(
        HEADER_LEN as u64 + padded == header.manifest_offset,
        "header payload_size {} inconsistent with manifest_offset {}",
        header.payload_size,
        header.manifest_offset
    );

    let manifest: Vec<ArenaTensorEntry> =
        postcard::from_bytes(&manifest_bytes).context("decoding snapshot manifest")?;
    ensure!(
        manifest.len() == header.tensor_count as usize,
        "manifest entry count {} != header tensor_count {}",
        manifest.len(),
        header.tensor_count
    );
    Ok((header, manifest))
}

/// One page-aligned page, for O_DIRECT writes of small unaligned sources.
struct AlignedPage(*mut u8);

impl AlignedPage {
    fn new() -> Self {
        let layout = std::alloc::Layout::from_size_align(HEADER_LEN, HEADER_LEN).unwrap();
        let p = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!p.is_null());
        Self(p)
    }
    fn as_ref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.0, HEADER_LEN) }
    }
    fn as_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.0, HEADER_LEN) }
    }
}

impl Drop for AlignedPage {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(HEADER_LEN, HEADER_LEN).unwrap();
        unsafe { std::alloc::dealloc(self.0, layout) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_manifest() -> Vec<ArenaTensorEntry> {
        vec![
            ArenaTensorEntry {
                name: "a.w".into(),
                offset: 0,
                len: 300,
                dtype: 0,
                shape: vec![150],
            },
            ArenaTensorEntry {
                name: "b.w".into(),
                offset: 512,
                len: 100,
                dtype: 4,
                shape: vec![10, 10],
            },
        ]
    }

    fn write_tiny(dir: &Path) -> (std::path::PathBuf, WeightsArena, u64) {
        let arena = WeightsArena::new(1 << 20, false).unwrap();
        arena.alloc(612, 256).unwrap(); // used = 612
        // Deterministic pattern in the used region.
        unsafe {
            let s = std::slice::from_raw_parts_mut(arena.base().0 as *mut u8, 612);
            for (i, b) in s.iter_mut().enumerate() {
                *b = (i % 253) as u8;
            }
        }
        let path = dir.join("t.raw");
        let stats =
            write_snapshot(&path, &arena, &tiny_manifest(), model_version_hash("m/x", "r1"))
                .unwrap();
        (path, arena, stats.payload_xxh3)
    }

    #[test]
    fn roundtrip_header_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let (path, _arena, xxh) = write_tiny(dir.path());
        let (header, manifest) = read_snapshot_meta(&path).unwrap();
        assert_eq!(header.format_version, FORMAT_VERSION);
        assert_eq!(header.payload_size, 612);
        assert_eq!(header.tensor_count, 2);
        assert_eq!(header.payload_xxh3, xxh);
        assert_eq!(header.manifest_offset, (HEADER_LEN + 4096) as u64);
        assert_eq!(manifest, tiny_manifest());
        assert_eq!(
            header.model_version_hash,
            model_version_hash("m/x", "r1")
        );
    }

    #[test]
    fn unknown_format_version_hard_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let (path, _arena, _) = write_tiny(dir.path());
        // Corrupt format_version in place.
        let f = OpenOptions::new().write(true).open(&path).unwrap();
        use std::os::unix::fs::FileExt;
        f.write_all_at(&2u32.to_le_bytes(), 8).unwrap();
        let err = read_snapshot_meta(&path).unwrap_err().to_string();
        assert!(err.contains("format_version"), "{err}");
    }

    #[test]
    fn corrupt_manifest_checksum_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let (path, _arena, _) = write_tiny(dir.path());
        let len = std::fs::metadata(&path).unwrap().len();
        let f = OpenOptions::new().write(true).open(&path).unwrap();
        use std::os::unix::fs::FileExt;
        // Flip a byte inside the manifest region.
        f.write_all_at(&[0xFF], len - FOOTER_LEN as u64 - 3).unwrap();
        let err = read_snapshot_meta(&path).unwrap_err().to_string();
        assert!(err.contains("manifest checksum"), "{err}");
    }

    #[test]
    fn bad_footer_magic_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let (path, _arena, _) = write_tiny(dir.path());
        let len = std::fs::metadata(&path).unwrap().len();
        let f = OpenOptions::new().write(true).open(&path).unwrap();
        use std::os::unix::fs::FileExt;
        f.write_all_at(b"XXXX", len - 4).unwrap();
        let err = read_snapshot_meta(&path).unwrap_err().to_string();
        assert!(err.contains("footer magic"), "{err}");
    }

    #[test]
    fn file_name_and_hash_stable() {
        let h = model_version_hash("nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4", "abc");
        assert_eq!(h, model_version_hash("nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4", "abc"));
        let name = snapshot_file_name("nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4", &h);
        assert!(name.starts_with("nvidia--Qwen3-Next-80B-A3B-Thinking-NVFP4-"));
        assert!(name.ends_with(".raw"));
    }
}

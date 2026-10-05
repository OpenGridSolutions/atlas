// SPDX-License-Identifier: AGPL-3.0-only

//! On-disk record format of the NVMe block store (NVM-5..7 of the
//! requirements doc; design/nvme-tier-analysis.md §2.2).
//!
//! Every record is `HEADER_LEN` bytes of header followed by the payload,
//! padded to `ALIGN` so the whole file can be written with O_DIRECT in
//! one aligned submission. The header binds the payload to its chain key,
//! predecessor, model and KV layout; a mismatch on read invalidates the
//! file (NVM-7), a payload-checksum failure ends the restore chain at
//! this record (NVM-6).

use anyhow::{Result, bail};
use xxhash_rust::xxh3::xxh3_64;

/// O_DIRECT alignment for offsets, lengths and buffer addresses.
pub const ALIGN: usize = 4096;
/// Fixed header size (one aligned sector).
pub const HEADER_LEN: usize = 4096;
/// Record magic.
pub const MAGIC: &[u8; 8] = b"ATLBSTR1";
/// Format version — bump on ANY layout change; old files are deleted.
pub const FORMAT_VERSION: u32 = 1;

/// Record kind — maps 1:1 onto the file extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// One sealed KV block (`.kvb`).
    Kv,
    /// SSM snapshot at a chain position (`.ssm`).
    Ssm,
    /// Tail record: trailing sub-block KV + tail SSM snapshot (`.tail`).
    Tail,
}

impl Kind {
    pub fn ext(self) -> &'static str {
        match self {
            Kind::Kv => "kvb",
            Kind::Ssm => "ssm",
            Kind::Tail => "tail",
        }
    }

    pub fn tag(self) -> u8 {
        match self {
            Kind::Kv => 1,
            Kind::Ssm => 2,
            Kind::Tail => 3,
        }
    }

    pub fn from_tag(t: u8) -> Option<Kind> {
        match t {
            1 => Some(Kind::Kv),
            2 => Some(Kind::Ssm),
            3 => Some(Kind::Tail),
            _ => None,
        }
    }

    pub fn from_ext(e: &str) -> Option<Kind> {
        match e {
            "kvb" => Some(Kind::Kv),
            "ssm" => Some(Kind::Ssm),
            "tail" => Some(Kind::Tail),
            _ => None,
        }
    }
}

/// Decoded record header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub kind: Kind,
    pub model_hash: [u8; 32],
    pub layout_hash: [u8; 32],
    /// Chain key of this record.
    pub key: [u8; 16],
    /// Chain key of the predecessor block (`[0; 16]` for block 0 /
    /// records anchored at the seed).
    pub pred: [u8; 16],
    /// Tokens covered by this record's payload (block_size for full KV
    /// blocks, tail length for tails, chain position fraction for SSM).
    pub token_count: u32,
    pub payload_len: u64,
    pub payload_xxh3: u64,
}

// Byte offsets within the 4096-byte header.
const O_MAGIC: usize = 0; // 8
const O_VERSION: usize = 8; // 4
const O_KIND: usize = 12; // 1 (+3 pad)
const O_MODEL: usize = 16; // 32
const O_LAYOUT: usize = 48; // 32
const O_KEY: usize = 80; // 16
const O_PRED: usize = 96; // 16
const O_TOKENS: usize = 112; // 4 (+4 pad)
const O_PLEN: usize = 120; // 8
const O_PSUM: usize = 128; // 8
const O_HSUM: usize = 136; // 8  (xxh3 over bytes [0..136))

impl Header {
    /// Serialise into the first `HEADER_LEN` bytes of `buf`.
    pub fn encode_into(&self, buf: &mut [u8]) {
        assert!(buf.len() >= HEADER_LEN);
        buf[..HEADER_LEN].fill(0);
        buf[O_MAGIC..O_MAGIC + 8].copy_from_slice(MAGIC);
        buf[O_VERSION..O_VERSION + 4].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf[O_KIND] = self.kind.tag();
        buf[O_MODEL..O_MODEL + 32].copy_from_slice(&self.model_hash);
        buf[O_LAYOUT..O_LAYOUT + 32].copy_from_slice(&self.layout_hash);
        buf[O_KEY..O_KEY + 16].copy_from_slice(&self.key);
        buf[O_PRED..O_PRED + 16].copy_from_slice(&self.pred);
        buf[O_TOKENS..O_TOKENS + 4].copy_from_slice(&self.token_count.to_le_bytes());
        buf[O_PLEN..O_PLEN + 8].copy_from_slice(&self.payload_len.to_le_bytes());
        buf[O_PSUM..O_PSUM + 8].copy_from_slice(&self.payload_xxh3.to_le_bytes());
        let hsum = xxh3_64(&buf[..O_HSUM]);
        buf[O_HSUM..O_HSUM + 8].copy_from_slice(&hsum.to_le_bytes());
    }

    /// Parse and validate the header's own integrity (magic, version,
    /// header checksum). Model/layout binding is the caller's check —
    /// it owns the expected values (NVM-7).
    pub fn decode(buf: &[u8]) -> Result<Header> {
        if buf.len() < HEADER_LEN {
            bail!("header truncated: {} bytes", buf.len());
        }
        if &buf[O_MAGIC..O_MAGIC + 8] != MAGIC {
            bail!("bad magic");
        }
        let version = u32::from_le_bytes(buf[O_VERSION..O_VERSION + 4].try_into().unwrap());
        if version != FORMAT_VERSION {
            bail!("unsupported format version {version} (want {FORMAT_VERSION})");
        }
        let hsum_stored = u64::from_le_bytes(buf[O_HSUM..O_HSUM + 8].try_into().unwrap());
        if xxh3_64(&buf[..O_HSUM]) != hsum_stored {
            bail!("header checksum mismatch");
        }
        let kind = Kind::from_tag(buf[O_KIND]).ok_or_else(|| anyhow::anyhow!("bad kind tag"))?;
        Ok(Header {
            kind,
            model_hash: buf[O_MODEL..O_MODEL + 32].try_into().unwrap(),
            layout_hash: buf[O_LAYOUT..O_LAYOUT + 32].try_into().unwrap(),
            key: buf[O_KEY..O_KEY + 16].try_into().unwrap(),
            pred: buf[O_PRED..O_PRED + 16].try_into().unwrap(),
            token_count: u32::from_le_bytes(buf[O_TOKENS..O_TOKENS + 4].try_into().unwrap()),
            payload_len: u64::from_le_bytes(buf[O_PLEN..O_PLEN + 8].try_into().unwrap()),
            payload_xxh3: u64::from_le_bytes(buf[O_PSUM..O_PSUM + 8].try_into().unwrap()),
        })
    }
}

/// Round `n` up to the next `ALIGN` multiple.
pub fn align_up(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
}

/// Heap buffer aligned to `ALIGN` — required for O_DIRECT writes.
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
    cap: usize,
}

// SAFETY: plain owned heap memory, no interior sharing.
unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    /// Zeroed buffer of `len` bytes rounded up to the alignment.
    pub fn zeroed(len: usize) -> AlignedBuf {
        let cap = align_up(len.max(1));
        let layout = std::alloc::Layout::from_size_align(cap, ALIGN).expect("layout");
        // SAFETY: cap > 0, ALIGN is a power of two.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "aligned alloc of {cap} bytes failed");
        AlignedBuf { ptr, len, cap }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Full aligned capacity (what O_DIRECT writes).
    pub fn aligned_len(&self) -> usize {
        self.cap
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr valid for cap >= len bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: exclusive borrow of owned memory.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// The whole aligned region (len..cap is zero padding).
    pub fn aligned_slice(&self) -> &[u8] {
        // SAFETY: ptr valid for cap bytes, zero-initialised.
        unsafe { std::slice::from_raw_parts(self.ptr, self.cap) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.cap, ALIGN).expect("layout");
        // SAFETY: allocated with this exact layout in `zeroed`.
        unsafe { std::alloc::dealloc(self.ptr, layout) };
    }
}

impl std::fmt::Debug for AlignedBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AlignedBuf(len={}, cap={})", self.len, self.cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hdr() -> Header {
        Header {
            kind: Kind::Kv,
            model_hash: [1; 32],
            layout_hash: [2; 32],
            key: [3; 16],
            pred: [4; 16],
            token_count: 16,
            payload_len: 12345,
            payload_xxh3: 0xDEAD_BEEF,
        }
    }

    #[test]
    fn roundtrip() {
        let mut buf = vec![0u8; HEADER_LEN];
        let h = hdr();
        h.encode_into(&mut buf);
        assert_eq!(Header::decode(&buf).unwrap(), h);
    }

    #[test]
    fn corrupt_header_rejected() {
        let mut buf = vec![0u8; HEADER_LEN];
        hdr().encode_into(&mut buf);
        buf[O_KEY] ^= 0xFF;
        assert!(Header::decode(&buf).is_err());
    }

    #[test]
    fn wrong_version_rejected() {
        let mut buf = vec![0u8; HEADER_LEN];
        hdr().encode_into(&mut buf);
        buf[O_VERSION] = 99;
        assert!(Header::decode(&buf).is_err());
    }

    #[test]
    fn aligned_buf_is_aligned_and_zero_padded() {
        let mut b = AlignedBuf::zeroed(5000);
        assert_eq!(b.as_slice().as_ptr() as usize % ALIGN, 0);
        assert_eq!(b.aligned_len(), 8192);
        b.as_mut_slice()[4999] = 7;
        assert_eq!(b.aligned_slice()[4999], 7);
        assert_eq!(b.aligned_slice()[5000], 0);
    }

    #[test]
    fn align_up_math() {
        assert_eq!(align_up(0), 0);
        assert_eq!(align_up(1), 4096);
        assert_eq!(align_up(4096), 4096);
        assert_eq!(align_up(4097), 8192);
    }
}

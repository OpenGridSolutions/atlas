// SPDX-License-Identifier: AGPL-3.0-only

//! The compaction entry (CMP-11): "messages after the head up to and
//! including index `cut` are replaced by `summary`". Stored as JSON in
//! the RAM index and, when the NVMe tier is active, as a `.cmp` record
//! keyed by `m_cut` whose `pred` is the previous entry of the same
//! conversation (CMP-3).

use serde::{Deserialize, Serialize};
use spark_runtime::block_keys::BlockKey;

/// Bump when a field changes meaning; older records are ignored.
pub const FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionEntry {
    pub version: u32,
    /// Hex of `m_cut`.
    pub key: String,
    /// Model the summary was written for (the chain seed includes it).
    pub model: String,
    /// `summarize` or `truncate` (CTX-3: empty summary).
    pub policy: String,
    /// Messages at the front that were kept verbatim (CMP-4).
    pub head_len: usize,
    /// 0-based index of the last replaced message.
    pub cut: usize,
    /// `cut + 1 - head_len`.
    pub replaced_messages: usize,
    /// Prompt tokens the replaced messages occupied (estimate).
    pub replaced_tokens: usize,
    pub summary: String,
    /// Hex of the previous entry's key (CMP-3), `None` for the first.
    pub pred: Option<String>,
    /// Unix seconds.
    pub created_at: u64,
}

impl CompactionEntry {
    pub fn key_bytes(&self) -> Option<BlockKey> {
        BlockKey::from_hex(&self.key)
    }

    pub fn pred_bytes(&self) -> BlockKey {
        self.pred
            .as_deref()
            .and_then(BlockKey::from_hex)
            .unwrap_or(BlockKey::ZERO)
    }

    pub fn to_json_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("compaction entry serialises")
    }

    /// Parse a stored record; `None` for foreign or newer formats.
    pub fn from_json_bytes(bytes: &[u8]) -> Option<Self> {
        let e: Self = serde_json::from_slice(bytes).ok()?;
        (e.version == FORMAT_VERSION && e.key_bytes().is_some()).then_some(e)
    }
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

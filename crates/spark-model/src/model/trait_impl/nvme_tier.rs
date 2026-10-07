// SPDX-License-Identifier: AGPL-3.0-only

//! NVMe-tier SSM dispatches (requirements §4): export the recurrent
//! state of hybrid models for `.ssm`/`.tail` records and import restored
//! snapshots back into the Marconi pool + index. Pure-attention models
//! short-circuit everywhere (SSM-8 — every block boundary is a resume
//! point, no state to carry).

use anyhow::Result;

use super::super::types::TransformerModel;
use crate::layer::SsmLayerState;
use crate::traits::SequenceState;
use atlas_core::config::LayerType;

impl TransformerModel {
    pub(super) fn has_recurrent_state_dispatch(&self) -> bool {
        self.config.num_ssm_layers() > 0
    }

    /// Live end-of-sequence SSM state (h + conv per linear layer) — the
    /// `.tail`/aligned-end `.ssm` payload (SSM-3). Empty for attention-only
    /// models. Must run before `free_sequence` (the slot is still owned).
    pub(super) fn export_ssm_state_dispatch(&self, seq: &SequenceState) -> Result<Vec<u8>> {
        if self.config.num_ssm_layers() == 0 {
            return Ok(Vec::new());
        }
        // Order after any in-flight secondary-stream commit — same guard
        // as finish_leaf_snapshot (the MTP commit races the copy).
        let _ = self.sync_secondary_dispatch();
        let gpu = self.gpu.as_ref();
        let mut out = Vec::new();
        for (i, layer_state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any()
                    .downcast_ref::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("expected SsmLayerState at layer {i}"))?;
                let mut h_buf = vec![0u8; self.ssm_pool.h_bytes];
                let mut c_buf = vec![0u8; self.ssm_pool.conv_bytes];
                gpu.copy_d2h(ssm.h_state, &mut h_buf)?;
                gpu.copy_d2h(ssm.conv_state, &mut c_buf)?;
                out.extend_from_slice(&h_buf);
                out.extend_from_slice(&c_buf);
            }
        }
        Ok(out)
    }

    /// Deepest BLOCK-ALIGNED Marconi snapshot of this sequence, exported
    /// to host bytes — persisted as an interior `.ssm` resume point so a
    /// later conversation continuation does not have to replay the whole
    /// restored prefix through the SSM layers. `None` when Marconi is
    /// off, the model is attention-only, or no aligned snapshot exists.
    pub(super) fn export_aligned_ssm_snapshots_dispatch(
        &self,
        seq: &SequenceState,
    ) -> Result<Vec<(usize, Vec<u8>)>> {
        if self.config.num_ssm_layers() == 0 || !self.ssm_snapshots.is_enabled() {
            return Ok(Vec::new());
        }
        let bs = self.kv_cache.lock().block_size();
        if bs == 0 {
            return Ok(Vec::new());
        }
        let view = seq.block_keys.view(&seq.tokens, seq.tokens.len());
        let mut out = Vec::new();
        for (snap_id, tokens) in aligned_snapshots(bs, seq.tokens.len(), |max| {
            self.prefix_cache
                .peek_snapshot(&view, max, seq.session_hash)
        }) {
            let bytes = self.ssm_snapshots.export_host(snap_id, self.gpu.as_ref())?;
            out.push((tokens, bytes));
        }
        Ok(out)
    }

    /// Import an NVMe `.ssm` payload: pool slot + Marconi-index entry at
    /// `token_count` (SSM-4's resume point). Reclaims one LRU snapshot on
    /// pool exhaustion before giving up. Returns false when no slot was
    /// available (restore proceeds via the SSM replay path instead).
    pub(super) fn import_ssm_snapshot_dispatch(
        &self,
        prompt_tokens: &[u32],
        chain: &mut spark_runtime::block_keys::BlockKeyChain,
        token_count: usize,
        payload: &[u8],
    ) -> Result<bool> {
        if self.config.num_ssm_layers() == 0 || !self.ssm_snapshots.is_enabled() {
            return Ok(false);
        }
        chain.extend_to(prompt_tokens);
        if !chain.is_cacheable() || token_count > prompt_tokens.len() {
            return Ok(false);
        }
        let gpu = self.gpu.as_ref();
        let slot = match self.ssm_snapshots.import_host(payload, gpu)? {
            Some(s) => Some(s),
            None => {
                if self
                    .ssm_snapshots
                    .reclaim_from_cache(self.prefix_cache.as_ref(), &mut self.kv_cache.lock())
                {
                    self.ssm_snapshots.import_host(payload, gpu)?
                } else {
                    None
                }
            }
        };
        let Some(slot) = slot else {
            return Ok(false);
        };
        let view = chain.view(prompt_tokens, token_count);
        let displaced =
            self.prefix_cache
                .insert_intermediate_snapshot(&view, &[], &[], slot, 0, token_count);
        if let Some(old) = displaced {
            self.ssm_snapshots.free(old);
        }
        tracing::info!("nvme-restore: SSM resume point registered at {token_count} tokens");
        Ok(true)
    }
}

/// Every BLOCK-ALIGNED snapshot at or below `len` tokens, deepest first.
/// `peek(max)` answers "deepest snapshot ≤ max" (any alignment); the
/// chunk-end and finish-leaf snapshots sit at the exact prompt/sequence
/// end, which is almost never aligned, so the walk steps below each hit
/// until the index is exhausted. Only aligned ones are resume points.
/// Without this a hybrid model persisted no `.ssm` record at all and
/// every NVMe restore recomputed the whole prompt; with only the deepest
/// one, a long answer's decode checkpoint hid the prompt-tail checkpoint
/// that a replay of the prompt alone needs.
pub(super) fn aligned_snapshots(
    block_size: usize,
    len: usize,
    peek: impl Fn(usize) -> Option<(usize, usize)>,
) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if block_size == 0 {
        return out;
    }
    let mut max = len;
    while let Some((snap_id, tokens)) = peek(max) {
        if tokens == 0 {
            break;
        }
        if tokens.is_multiple_of(block_size) {
            out.push((snap_id, tokens));
        }
        max = tokens - 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::aligned_snapshots;

    /// SSM-4: the persist path skips the unaligned prompt-end and
    /// finish-leaf snapshots and keeps every interval/tail checkpoint —
    /// the prompt-tail one (7680) AND the deeper decode one (8192).
    #[test]
    fn keeps_every_aligned_checkpoint_deepest_first() {
        // leaf 8205, decode ckpt 8192, chunk end 7705, tail ckpt 7680, 4096.
        let entries = [
            (1usize, 8205usize),
            (5, 8192),
            (2, 7705),
            (3, 7680),
            (4, 4096),
        ];
        let peek = |max: usize| {
            entries
                .iter()
                .filter(|(_, t)| *t <= max)
                .max_by_key(|(_, t)| *t)
                .copied()
        };
        assert_eq!(
            aligned_snapshots(16, 8205, peek),
            vec![(5, 8192), (3, 7680), (4, 4096)]
        );
        assert_eq!(aligned_snapshots(16, 7679, peek), vec![(4, 4096)]);
        assert!(aligned_snapshots(16, 4000, peek).is_empty());
        assert!(aligned_snapshots(0, 8205, peek).is_empty());
        let only_end = |max: usize| (7705 <= max).then_some((2, 7705));
        assert!(aligned_snapshots(16, 7717, only_end).is_empty());
    }
}

// SPDX-License-Identifier: AGPL-3.0-only

//! Support for model families that advance their per-sequence state
//! themselves (`ModelWeightLoader::token_serial`): layers with a decode step
//! and, optionally, a plain chunk prefill.
//!
//! Such a model runs a prompt through `decode` token by token, or — when the
//! family declares a chunk size — through the layers' `prefill` in chunks of
//! one sequence. It serves a batch sequence by sequence and never looks up
//! the prefix cache: the state is only ever advanced in prompt order, so no
//! path can leave it out of step with the KV cache.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, KernelHandle};

use super::block_mgmt::ensure_blocks_through_prefill;
use super::types::TransformerModel;
use crate::layer::{AttnMetadataDev, DecodePrep, ForwardContext};
use crate::layers::ops;
use crate::traits::{Model, PrefillSlice, SequenceState};

/// Offset of the block table in the arena scratch (as in the decode step).
const DECODE_BLOCK_TABLE_AT: usize = 32768 + 256;

impl TransformerModel {
    /// Installs what the weight loader declares about its model family: the
    /// token-serial mode, the largest prefill chunk its layers take (1 = the
    /// layers only decode) and the kernel that stands in for the final norm.
    pub fn set_family_traits(
        &mut self,
        token_serial: bool,
        serial_prefill_rows: usize,
        final_norm_kernel: KernelHandle,
    ) {
        self.token_serial = token_serial;
        self.serial_prefill_rows = serial_prefill_rows;
        self.rms_norm_kernel = final_norm_kernel;
    }

    /// Runs every layer's host-side preparation for processing `tokens` next.
    fn prepare_layers(
        &self,
        seq: &SequenceState,
        tokens: &[u32],
        prefill: bool,
        stream: u64,
    ) -> Result<()> {
        let prep = DecodePrep {
            history: &seq.tokens,
            tokens,
            prefill,
            gpu: self.gpu.as_ref(),
            stream,
        };
        for layer in &self.layers {
            layer.prepare_decode(&prep)?;
        }
        Ok(())
    }

    /// Runs every layer's host-side preparation for decoding `token` next.
    pub(crate) fn prepare_layers_for_decode(
        &self,
        seq: &SequenceState,
        token: u32,
        stream: u64,
    ) -> Result<()> {
        self.prepare_layers(seq, &[token], false, stream)
    }

    /// Appends `tokens` to the sequence through the layers' `prefill` in one
    /// chunk. With `logits` the LM head runs on the last token.
    fn prefill_rows(&self, tokens: &[u32], seq: &mut SequenceState, logits: bool) -> Result<()> {
        // The stream decode uses, so chunks and decode steps stay in order.
        let stream = self.gpu.default_stream();
        let (gpu, n, start) = (self.gpu.as_ref(), tokens.len(), seq.seq_len);
        let h = self.config.hidden_size;
        let hidden = self.buffers.hidden_states();
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        ensure_blocks_through_prefill(
            seq,
            (start + n - 1) / bs,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            gpu,
            stream,
        )?;

        // Metadata: positions u32[n] at 0, KV slots i64[n] behind them (on
        // an 8-byte boundary); the block table where the decode step keeps it.
        let (slot_at, table_at) = ((n * 4).next_multiple_of(8), DECODE_BLOCK_TABLE_AT);
        if n > self.buffers.max_batch_tokens() || slot_at + n * 8 > table_at {
            bail!("prefill chunk of {n} tokens exceeds the buffers");
        }
        let scratch = self.buffers.scratch();
        let mut meta = Vec::with_capacity(n * 12);
        meta.extend((start..start + n).flat_map(|p| (p as u32).to_le_bytes()));
        meta.resize(slot_at, 0);
        for p in start..start + n {
            let block = seq
                .physical_block_for(p / bs)
                .unwrap_or(self.dummy_kv_block);
            let slot = block as i64 * bs as i64 + (p % bs) as i64;
            meta.extend(slot.to_le_bytes());
        }
        gpu.copy_h2d_async(&meta, scratch, stream)?;
        let table: Vec<u8> = seq
            .block_table
            .iter()
            .flat_map(|&b| (b as i32).to_le_bytes())
            .collect();
        gpu.copy_h2d_async(&table, scratch.offset(table_at), stream)?;
        for (i, &token) in tokens.iter().enumerate() {
            self.embed(token, hidden.offset(i * h * 2), stream)?;
        }
        self.prepare_layers(seq, tokens, true, stream)?;

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu,
            config: &self.config,
            attn_metadata: Some(AttnMetadataDev {
                positions: scratch,
                positions_h: scratch,
                positions_w: scratch,
                slot: scratch.offset(slot_at),
                seq_len: scratch,
                block_table: scratch.offset(table_at),
                max_blocks_per_seq: seq.block_table.len() as u32,
                num_seqs: 1,
            }),
            profile: self.profile,
            comm: self.comm_ref(),
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
        };
        for (i, layer) in self.layers.iter().enumerate() {
            layer.prefill(
                hidden,
                self.buffers.residual(),
                n,
                seq.layer_states[i].as_mut(),
                &mut kv_cache,
                start,
                &mut seq.block_table,
                &mut seq.disk_block_ids,
                &mut seq.disk_last_offloaded_per_layer,
                0,
                &ctx,
                stream,
            )?;
        }
        if logits {
            let normed = self.buffers.norm_output();
            ops::rms_norm(
                gpu,
                self.rms_norm_kernel,
                hidden.offset((n - 1) * h * 2),
                &self.final_norm,
                normed,
                1,
                h as u32,
                self.config.rms_norm_eps as f32,
                stream,
            )?;
            self.lm_head(normed, stream)?;
        }
        seq.tokens.extend_from_slice(tokens);
        seq.seq_len += n;
        Ok(())
    }

    /// Prefill of `tokens[start..start + len]` in prompt order; `tokens` is
    /// the whole prompt. Returns the logits of the last token processed.
    pub(crate) fn prefill_token_serial(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        start: usize,
        len: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        if start == 0 {
            seq.cached_prefix_tokens = 0;
            seq.prompt_len = tokens.len();
        }
        let (mut at, end) = (start, start + len);
        while at < end {
            let n = (end - at).min(self.serial_prefill_rows);
            if n == 1 {
                self.decode(tokens[at], seq, stream)?;
            } else {
                self.prefill_rows(&tokens[at..at + n], seq, at + n == end)?;
            }
            at += n;
        }
        seq.kv_valid_tokens = seq.kv_valid_tokens.max(start + len);
        Ok(self.decode_logits_ptr())
    }

    /// Token-serial stand-in for the batched prefill: one stream after the
    /// other. Every stream gets its own logits row, because the caller
    /// samples only after all streams have run.
    pub(crate) fn prefill_batch_token_serial(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<Vec<DevicePtr>> {
        use std::sync::atomic::Ordering;
        let n = streams.len();
        let logits = self.decode_logits_ptr();
        let elem = if self.decode_logits_fp32() { 4 } else { 2 };
        let row_bytes = self.config.vocab_size * elem;
        let rows = self.buffers.max_batch_tokens().min(32);
        if n > rows {
            bail!("{n} concurrent prefills exceed the {rows} logits rows");
        }
        // As in the per-sequence decode fallback: no graph capture while
        // several state slots take turns on one stream.
        let prev_suppress = self.suppress_graphs.swap(n > 1, Ordering::Relaxed);
        let result = (|| -> Result<()> {
            let mut staged = vec![0u8; n * row_bytes];
            for (i, s) in streams.iter_mut().enumerate() {
                self.prefill_token_serial(
                    s.prompt_tokens,
                    s.seq,
                    s.chunk_start,
                    s.chunk_len,
                    stream,
                )?;
                self.gpu.copy_d2h_on_stream(
                    logits,
                    &mut staged[i * row_bytes..(i + 1) * row_bytes],
                    stream,
                )?;
            }
            self.gpu.copy_h2d_async(&staged, logits, stream)?;
            self.gpu.synchronize(stream)
        })();
        self.suppress_graphs.store(prev_suppress, Ordering::Relaxed);
        result?;
        Ok((0..n).map(|i| logits.offset(i * row_bytes)).collect())
    }
}

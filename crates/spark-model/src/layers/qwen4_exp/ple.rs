// SPDX-License-Identifier: AGPL-3.0-only

//! N-gram layer (PLE): fetches a token's table rows from disk and adds the
//! gated, convolved n-gram features to the streams.
//!
//! The fetch is host work and runs before the decode graph
//! ([`Ple::prepare`]); the kernels then read the rows from a fixed staging
//! buffer.

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::row_table::{RowRead, RowSource};

use super::kernels::{Kernels, StreamDims};
use super::prefill::Pass;
use super::scratch::Scratch;
use super::weights::PleWeights;
use crate::layers::ops;
use crate::weight_loader::qwen4_exp::ngram::NgramHasher;
use crate::weight_map::DenseWeight;

pub struct Ple {
    pub weights: PleWeights,
    pub hasher: NgramHasher,
    /// Table planes: `2 * shard` = packed rows, `2 * shard + 1` = row scales.
    pub rows: Mutex<Box<dyn RowSource>>,
    pub head_dim: u32,
    pub taps: u32,
    pub dilation: u32,
}

impl Ple {
    fn state_len(&self) -> u32 {
        (self.taps - 1) * self.dilation
    }

    /// Reads the table rows of `tokens` (which follow `history`) and uploads
    /// them: all packed rows to `rows_dev`, all row scales to `scales_dev`,
    /// token by token.
    pub(super) fn prepare(
        &self,
        history: &[u32],
        tokens: &[u32],
        gpu: &dyn GpuBackend,
        rows_dev: DevicePtr,
        scales_dev: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        // The hash reads a token and its context_len predecessors.
        let ctx = self.hasher.context_len();
        let mut window: Vec<u32> = history[history.len().saturating_sub(ctx)..].to_vec();
        let heads = self.hasher.heads();
        let (row_bytes, scale_bytes) = (self.head_dim as usize / 2, self.head_dim as usize / 16);
        // One read list: the packed rows of every token, then the scales.
        let mut addresses = Vec::with_capacity(tokens.len() * heads);
        let mut ids = Vec::with_capacity(heads);
        for &token in tokens {
            window.push(token);
            let from = window.len().saturating_sub(ctx + 1);
            ids.clear();
            self.hasher.row_ids(&window[from..], &mut ids)?;
            addresses.extend(ids.iter().map(|id| self.hasher.address(*id)));
        }
        let mut reads = Vec::with_capacity(2 * addresses.len());
        for plane in 0..2u32 {
            reads.extend(addresses.iter().map(|a| RowRead {
                plane: 2 * a.shard as u32 + plane,
                row: a.row as u64,
            }));
        }
        let mut bytes = vec![0u8; addresses.len() * (row_bytes + scale_bytes)];
        self.rows.lock().read_rows(&reads, &mut bytes)?;
        let (rows, scales) = bytes.split_at(addresses.len() * row_bytes);
        gpu.copy_h2d_async(rows, rows_dev, stream)?;
        gpu.copy_h2d_async(scales, scales_dev, stream)
    }

    /// The n-gram features of a chunk: `p.s.streams += ple(streams, rows)`
    /// for `p.n` tokens whose rows are staged in the prefill scratch.
    /// `first_position` is the first token's position (u32 on the device).
    pub(super) fn forward_prefill(
        &self,
        p: &Pass<'_>,
        d: StreamDims,
        state: DevicePtr,
        first_position: DevicePtr,
    ) -> Result<()> {
        let (w, s) = (&self.weights, p.s);
        let heads = self.hasher.heads() as u32;
        let emb = heads * self.head_dim;
        let channels = d.hc * d.hidden;
        // key -> gate_u, value -> y: both are free until the first mixer runs.
        let (key, value) = (s.gate_u, s.y);
        KernelLaunch::new(p.gpu, p.k.ple_decode_rows)
            .grid([p.n, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.ple_rows)
            .arg_ptr(s.ple_row_scales)
            .arg_ptr(s.ple_emb)
            .arg_u32(heads)
            .arg_u32(self.head_dim)
            .arg_f32(w.scale2)
            .launch(p.stream)?;
        p.gemm(s.ple_emb, w.key_proj, key, channels, emb)?;
        p.gemm(s.ple_emb, w.value_proj, value, d.hidden, emb)?;
        KernelLaunch::new(p.gpu, p.k.ple_gate)
            .grid([p.n, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(key)
            .arg_ptr(value)
            .arg_ptr(s.streams)
            .arg_ptr(w.w_key)
            .arg_ptr(w.w_query)
            .arg_ptr(w.w_conv)
            .arg_ptr(s.ple_gated)
            .arg_ptr(s.ple_conv_in)
            .arg_u32(d.hidden)
            .arg_u32(d.hc)
            .arg_f32(d.eps)
            .launch(p.stream)?;
        self.conv(
            p.gpu,
            p.k,
            p.s1,
            (s.ple_gated, s.ple_conv_in, s.streams),
            state,
            first_position,
            channels,
            p.n,
            p.stream,
        )
    }

    /// Convolution over the token axis and state update for `n` consecutive
    /// tokens of one sequence; `bufs` = (gated, conv_in, streams).
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        gpu: &dyn GpuBackend,
        k: &Kernels,
        s: &Scratch,
        (gated, conv_in, streams): (DevicePtr, DevicePtr, DevicePtr),
        state: DevicePtr,
        first_position: DevicePtr,
        channels: u32,
        n: u32,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, k.ple_conv_add)
            .grid([channels.div_ceil(256), n, 1])
            .block([256, 1, 1])
            .arg_ptr(gated)
            .arg_ptr(conv_in)
            .arg_ptr(self.weights.conv)
            .arg_ptr(state)
            .arg_ptr(s.zero_u32)
            .arg_ptr(first_position)
            .arg_ptr(streams)
            .arg_u64(0)
            .arg_u32(channels)
            .arg_u32(n)
            .arg_u32(self.taps)
            .arg_u32(self.dilation)
            .launch(stream)?;
        KernelLaunch::new(gpu, k.ple_state_update)
            .grid([channels.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(conv_in)
            .arg_ptr(state)
            .arg_ptr(s.zero_u32)
            .arg_u64(0)
            .arg_u32(channels)
            .arg_u32(n)
            .arg_u32(self.state_len())
            .launch(stream)
    }

    /// `streams += ple(streams, rows)` for one token at `positions[0]`.
    /// `state` is the sequence's convolution history, FP32
    /// `[hc * H, state_len]`, oldest value first.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward(
        &self,
        gpu: &dyn GpuBackend,
        k: &Kernels,
        s: &Scratch,
        d: StreamDims,
        streams: DevicePtr,
        state: DevicePtr,
        positions: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let w = &self.weights;
        let heads = self.hasher.heads() as u32;
        let emb = heads * self.head_dim;
        let channels = d.hc * d.hidden;
        KernelLaunch::new(gpu, k.ple_decode_rows)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.ple_rows)
            .arg_ptr(s.ple_row_scales)
            .arg_ptr(s.ple_emb)
            .arg_u32(heads)
            .arg_u32(self.head_dim)
            .arg_f32(w.scale2)
            .launch(stream)?;
        let (key_w, value_w) = (
            DenseWeight { weight: w.key_proj },
            DenseWeight {
                weight: w.value_proj,
            },
        );
        ops::dense_gemv(
            gpu,
            k.dense_gemv,
            s.ple_emb,
            &key_w,
            s.ple_key,
            channels,
            emb,
            stream,
        )?;
        ops::dense_gemv(
            gpu,
            k.dense_gemv,
            s.ple_emb,
            &value_w,
            s.ple_value,
            d.hidden,
            emb,
            stream,
        )?;
        KernelLaunch::new(gpu, k.ple_gate)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.ple_key)
            .arg_ptr(s.ple_value)
            .arg_ptr(streams)
            .arg_ptr(w.w_key)
            .arg_ptr(w.w_query)
            .arg_ptr(w.w_conv)
            .arg_ptr(s.ple_gated)
            .arg_ptr(s.ple_conv_in)
            .arg_u32(d.hidden)
            .arg_u32(d.hc)
            .arg_f32(d.eps)
            .launch(stream)?;
        self.conv(
            gpu,
            k,
            s,
            (s.ple_gated, s.ple_conv_in, streams),
            state,
            positions,
            channels,
            1,
            stream,
        )
    }
}

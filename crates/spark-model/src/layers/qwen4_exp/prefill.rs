// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill path of the `qwen4_exp` layer: many prompt tokens per call.
//!
//! The math is that of the decode step. What changes is how often weights
//! are read: every projection runs as one GEMM over the chunk, and every
//! routed expert is read once per chunk instead of once per token. MXFP8
//! projections are expanded to BF16 into a scratch matrix for the duration
//! of their GEMM; the resident weights stay in checkpoint format.

mod scratch;
mod sublayers;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub use self::scratch::PREFILL_ROWS;
pub(super) use self::scratch::PrefillScratch;
pub(super) use self::sublayers::{attention, gdn, moe};
use super::kernels::StreamDims;
use super::weights::{Mixer, Mxfp8Linear};
use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// Handles of the kernels only the prefill path launches.
pub(super) struct PrefillKernels {
    // model target
    pub gr_norm_bf16: KernelHandle,
    pub gr_act: KernelHandle,
    pub gr_gate_mix: KernelHandle,
    pub mxfp8_expand: KernelHandle,
    pub gdn_out_norm_bf16: KernelHandle,
    // shared
    pub gemm: KernelHandle,
    pub conv1d: KernelHandle,
    pub l2_norm: KernelHandle,
    pub gdn: KernelHandle,
    pub deinterleave_split: KernelHandle,
    pub attn_paged: KernelHandle,
    pub sigmoid_gate_mul: KernelHandle,
    pub moe_topk: KernelHandle,
    pub moe_sort: KernelHandle,
    pub moe_grouped: KernelHandle,
    pub moe_unpermute: KernelHandle,
    pub moe_blend: KernelHandle,
}

impl PrefillKernels {
    pub(super) fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        let k = |m: &str, f: &str| gpu.kernel(m, f);
        Ok(Self {
            gr_norm_bf16: k("gated_residual", "gr_norm_bf16")?,
            gr_act: k("gated_residual", "gr_act")?,
            gr_gate_mix: k("gated_residual", "gr_gate_mix")?,
            mxfp8_expand: k("mxfp8_gemv", "mxfp8_expand_bf16")?,
            gdn_out_norm_bf16: k("gdn_split", "gdn_out_norm_bf16")?,
            gemm: k("gemm", "dense_gemm_bf16_pipelined")?,
            conv1d: k("causal_conv1d", "causal_conv1d_update_prefill")?,
            l2_norm: k("norm", "l2_norm_bf16")?,
            gdn: k(
                "gated_delta_rule_persistent",
                "gated_delta_rule_prefill_persistent",
            )?,
            deinterleave_split: k("ssm_preprocess", "deinterleave_qg_split")?,
            attn_paged: k("prefill_paged", "inferspark_prefill_paged")?,
            sigmoid_gate_mul: k("residual_add", "sigmoid_gate_mul_batched")?,
            moe_topk: k("moe_topk", "moe_topk_softmax_batched")?,
            moe_sort: k("moe", "moe_sort_by_expert")?,
            moe_grouped: k("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable")?,
            moe_unpermute: k("moe", "moe_unpermute_reduce_indexed")?,
            moe_blend: k("moe", "moe_batched_blend")?,
        })
    }
}

/// What every prefill step needs: the device, the kernels, the working
/// memory, the stream and the number of tokens.
#[derive(Clone, Copy)]
pub(super) struct Pass<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub k: &'a super::kernels::Kernels,
    pub pk: &'a PrefillKernels,
    pub s: &'a PrefillScratch,
    /// Single-token working memory: expert scale slots, selection scratch.
    pub s1: &'a super::scratch::Scratch,
    pub stream: u64,
    pub n: u32,
}

impl Pass<'_> {
    /// `out[t, n] = sum_k x[t, k] * w[n, k]` for a BF16 weight `[n, k]`.
    pub(super) fn gemm(
        &self,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let w = DenseWeight { weight: w };
        ops::dense_gemm_bf16_pipelined(
            self.gpu,
            self.pk.gemm,
            x,
            &w,
            out,
            self.n,
            n,
            k,
            self.stream,
        )
    }

    /// The same for an MXFP8 weight, expanded into the weight scratch first.
    pub(super) fn linear(&self, x: DevicePtr, w: &Mxfp8Linear, out: DevicePtr) -> Result<()> {
        let groups = w.n as u64 * (w.k / 32) as u64;
        KernelLaunch::new(self.gpu, self.pk.mxfp8_expand)
            .grid([groups.div_ceil(256) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(w.weight)
            .arg_ptr(w.scale)
            .arg_ptr(self.s.weight)
            .arg_u32(w.n)
            .arg_u32(w.k)
            .launch(self.stream)?;
        self.gemm(x, self.s.weight, out, w.n, w.k)
    }

    /// Collapses the streams of every token into `mixed` (BF16 `[T, H]`);
    /// with an inject weight the injection weights go to `inj` (`[T, hc]`).
    pub(super) fn mix(
        &self,
        d: StreamDims,
        m: &Mixer,
        inj: DevicePtr,
        mixed: DevicePtr,
    ) -> Result<()> {
        let (s, channels) = (self.s, d.hc * d.hidden);
        KernelLaunch::new(self.gpu, self.pk.gr_norm_bf16)
            .grid([self.n, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.streams)
            .arg_ptr(m.norm_w)
            .arg_ptr(s.normed)
            .arg_u32(d.hidden)
            .arg_u32(d.hc)
            .arg_f32(d.eps)
            .launch(self.stream)?;
        self.gemm(s.normed, m.down, s.act, d.rank, channels)?;
        let (logits, inj) = match m.inject {
            Some(w) => {
                self.gemm(s.normed, w, s.inj_logits, d.hc, channels)?;
                (s.inj_logits, inj)
            }
            None => (DevicePtr::NULL, DevicePtr::NULL),
        };
        KernelLaunch::new(self.gpu, self.pk.gr_act)
            .grid([self.n, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.act)
            .arg_ptr(logits)
            .arg_ptr(inj)
            .arg_u32(d.rank)
            .arg_u32(d.hc)
            .launch(self.stream)?;
        self.gemm(s.act, m.up, s.gate_u, channels, d.rank)?;
        KernelLaunch::new(self.gpu, self.pk.gr_gate_mix)
            .grid([d.hidden.div_ceil(256), self.n, 1])
            .block([256, 1, 1])
            .arg_ptr(s.gate_u)
            .arg_ptr(s.normed)
            .arg_ptr(mixed)
            .arg_u32(d.hidden)
            .arg_u32(d.hc)
            .launch(self.stream)
    }

    /// `streams[t, i] += block_out[t] * inj[t, i]`.
    pub(super) fn post(&self, d: StreamDims, block_out: DevicePtr, inj: DevicePtr) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.gr_post)
            .grid([self.n, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(block_out)
            .arg_ptr(inj)
            .arg_ptr(self.s.streams)
            .arg_u32(d.hidden)
            .arg_u32(d.hc)
            .launch(self.stream)
    }

    /// `streams[t, i] = hidden[t]`.
    pub(super) fn expand(&self, d: StreamDims, hidden: DevicePtr) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.gr_expand)
            .grid([self.n, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(hidden)
            .arg_ptr(self.s.streams)
            .arg_u32(d.hidden)
            .arg_u32(d.hc)
            .launch(self.stream)
    }
}

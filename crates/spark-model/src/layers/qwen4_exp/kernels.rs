// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel handles and launch geometry for the `qwen4_exp` layer.
//!
//! The first group lives in the model's own kernel target
//! (`kernels/gb10/qwen3.8-flash-next/`), the second is shared with other
//! model families and launched through `layers::ops`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::weights::{Mixer, Mxfp8Linear};

/// Handles of every kernel the layer launches.
pub(super) struct Kernels {
    // model target
    pub gr_expand: KernelHandle,
    pub gr_norm_inject: KernelHandle,
    pub gr_down_silu: KernelHandle,
    pub gr_up_mix: KernelHandle,
    pub gr_post: KernelHandle,
    pub mxfp8_gemv: KernelHandle,
    pub swiglu_mul: KernelHandle,
    pub csf_expand_fixed: KernelHandle,
    pub csf_expand_exceptions: KernelHandle,
    pub csf_validate: KernelHandle,
    pub gdn_gates_split: KernelHandle,
    pub gdn_out_norm: KernelHandle,
    pub ple_decode_rows: KernelHandle,
    pub ple_gate: KernelHandle,
    pub ple_conv_add: KernelHandle,
    pub ple_state_update: KernelHandle,
    pub qsa_index_raw: KernelHandle,
    pub qsa_index_pool: KernelHandle,
    pub qsa_score: KernelHandle,
    pub qsa_select: KernelHandle,
    pub qsa_gather: KernelHandle,
    // shared
    pub dense_gemv: KernelHandle,
    pub moe_topk: KernelHandle,
    pub moe_gate_up: KernelHandle,
    pub moe_silu_down: KernelHandle,
    pub moe_blend: KernelHandle,
    pub conv1d_l2norm_f32: KernelHandle,
    pub gdn_decode_f32: KernelHandle,
    pub deinterleave_qg: KernelHandle,
    pub rms_norm: KernelHandle,
    pub rope: KernelHandle,
    pub reshape_and_cache: KernelHandle,
    pub paged_decode: KernelHandle,
    pub sigmoid_gate_mul: KernelHandle,
}

impl Kernels {
    pub(super) fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        let k = |m: &str, f: &str| gpu.kernel(m, f);
        Ok(Self {
            gr_expand: k("gated_residual", "gr_expand")?,
            gr_norm_inject: k("gated_residual", "gr_norm_inject")?,
            gr_down_silu: k("gated_residual", "gr_down_silu")?,
            gr_up_mix: k("gated_residual", "gr_up_mix")?,
            gr_post: k("gated_residual", "gr_post")?,
            mxfp8_gemv: k("mxfp8_gemv", "mxfp8_gemv")?,
            swiglu_mul: k("mxfp8_gemv", "swiglu_mul")?,
            csf_expand_fixed: k("csf_expand", "csf_expand_fixed")?,
            csf_expand_exceptions: k("csf_expand", "csf_expand_exceptions")?,
            csf_validate: k("csf_expand", "csf_validate")?,
            gdn_gates_split: k("gdn_split", "gdn_gates_split")?,
            gdn_out_norm: k("gdn_split", "gdn_out_norm")?,
            ple_decode_rows: k("ple_layer", "ple_decode_rows")?,
            ple_gate: k("ple_layer", "ple_gate")?,
            ple_conv_add: k("ple_layer", "ple_conv_add")?,
            ple_state_update: k("ple_layer", "ple_state_update")?,
            qsa_index_raw: k("qsa", "qsa_index_raw")?,
            qsa_index_pool: k("qsa", "qsa_index_pool")?,
            qsa_score: k("qsa", "qsa_score")?,
            qsa_select: k("qsa", "qsa_select")?,
            qsa_gather: k("qsa", "qsa_gather")?,
            dense_gemv: k("gemv", "dense_gemv_bf16")?,
            moe_topk: k("moe_topk", "moe_topk_softmax")?,
            moe_gate_up: k("moe_expert_gemv_fused", "moe_expert_gemv_gate_up")?,
            moe_silu_down: k("moe_expert_gemv_fused", "moe_expert_gemv_silu_down")?,
            moe_blend: k("moe_expert_gemv", "moe_weighted_sum_blend")?,
            conv1d_l2norm_f32: k("causal_conv1d", "causal_conv1d_update_l2norm_f32")?,
            gdn_decode_f32: k("gated_delta_rule", "gated_delta_rule_decode_f32")?,
            deinterleave_qg: k("ssm_preprocess", "deinterleave_qg")?,
            rms_norm: k("norm", "rms_norm")?,
            rope: k("rope", "rope_forward")?,
            reshape_and_cache: k("reshape_and_cache", "reshape_and_cache_flash")?,
            paged_decode: k("paged_decode", "paged_decode_attn")?,
            sigmoid_gate_mul: k("residual_add", "sigmoid_gate_mul")?,
        })
    }
}

/// `out[t, n] = sum_k act[t, k] * W[n, k]` for an MXFP8 weight; BF16 in and out.
pub(super) fn mxfp8_gemv(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    act: DevicePtr,
    w: &Mxfp8Linear,
    out: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k.mxfp8_gemv)
        .grid([w.n.div_ceil(4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(act)
        .arg_ptr(w.weight)
        .arg_ptr(w.scale)
        .arg_ptr(out)
        .arg_u32(w.n)
        .arg_u32(w.k)
        .launch(stream)
}

/// Stream geometry shared by the mixer launches.
#[derive(Clone, Copy)]
pub(super) struct StreamDims {
    pub hidden: u32,
    pub hc: u32,
    pub rank: u32,
    pub eps: f32,
}

/// Collapses the streams of one token into the sublayer input `mixed`
/// (BF16 `[H]`). With an inject weight the per-stream injection weights go to
/// `inj`. `normed` (FP32 `[hc, H]`) and `act` (FP32 `[rank]`) are scratch.
#[allow(clippy::too_many_arguments)]
pub(super) fn mix(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    d: StreamDims,
    m: &Mixer,
    streams: DevicePtr,
    normed: DevicePtr,
    act: DevicePtr,
    inj: DevicePtr,
    mixed: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k.gr_norm_inject)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(m.norm_w)
        .arg_ptr(m.inject.unwrap_or(DevicePtr::NULL))
        .arg_ptr(normed)
        .arg_ptr(if m.inject.is_some() {
            inj
        } else {
            DevicePtr::NULL
        })
        .arg_u32(d.hidden)
        .arg_u32(d.hc)
        .arg_f32(d.eps)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.gr_down_silu)
        .grid([d.rank, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(normed)
        .arg_ptr(m.down)
        .arg_ptr(act)
        .arg_u32(d.hidden)
        .arg_u32(d.hc)
        .arg_u32(d.rank)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.gr_up_mix)
        .grid([d.hidden.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(normed)
        .arg_ptr(act)
        .arg_ptr(m.up)
        .arg_ptr(mixed)
        .arg_u32(d.hidden)
        .arg_u32(d.hc)
        .arg_u32(d.rank)
        .launch(stream)
}

/// `streams[i] += block_out * inj[i]` for one token.
pub(super) fn post(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    d: StreamDims,
    block_out: DevicePtr,
    inj: DevicePtr,
    streams: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k.gr_post)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(block_out)
        .arg_ptr(inj)
        .arg_ptr(streams)
        .arg_u32(d.hidden)
        .arg_u32(d.hc)
        .launch(stream)
}

/// `streams[i] = hidden` for one token.
pub(super) fn expand(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    d: StreamDims,
    hidden: DevicePtr,
    streams: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k.gr_expand)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(streams)
        .arg_u32(d.hidden)
        .arg_u32(d.hc)
        .launch(stream)
}

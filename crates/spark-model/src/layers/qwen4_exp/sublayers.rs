// SPDX-License-Identifier: AGPL-3.0-only

//! Single-token forward of the gated delta net and the MoE (attention lives
//! in `attention.rs`). Each reads the mixed stream `scratch.x` (BF16 `[H]`) and writes
//! `scratch.y` (BF16 `[H]`).

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::kernels::{Kernels, mxfp8_gemv};
use super::scratch::{Scratch, expert_scale_dims};
use super::weights::{ExpertProj, GdnWeights, MoeWeights};
use crate::layer::SsmLayerState;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// L2-norm epsilon of the query/key heads (reference: `l2norm(..., eps=1e-6)`).
pub(super) const GDN_L2_EPS: f32 = 1e-6;

/// Gated delta net: projections, causal conv, recurrence, gated norm, out.
#[allow(clippy::too_many_arguments)]
pub(super) fn gdn(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    s: &Scratch,
    config: &ModelConfig,
    sigmoid_gate: bool,
    w: &GdnWeights,
    state: &SsmLayerState,
    stream: u64,
) -> Result<()> {
    let (nk, nv) = (
        config.linear_num_key_heads as u32,
        config.linear_num_value_heads as u32,
    );
    let (kd, vd) = (
        config.linear_key_head_dim as u32,
        config.linear_value_head_dim as u32,
    );
    let key_dim = (nk * kd) as usize;
    let conv_dim = 2 * nk * kd + nv * vd;
    let z = s.qkvz.offset(conv_dim as usize * 2);

    mxfp8_gemv(gpu, k, s.x, &w.qkv, s.qkvz, stream)?;
    mxfp8_gemv(gpu, k, s.x, &w.z, z, stream)?;
    mxfp8_gemv(gpu, k, s.x, &w.a, s.gdn_a, stream)?;
    mxfp8_gemv(gpu, k, s.x, &w.b, s.gdn_b, stream)?;
    KernelLaunch::new(gpu, k.gdn_gates_split)
        .grid([1, 1, 1])
        .block([64, 1, 1])
        .arg_ptr(s.gdn_a)
        .arg_ptr(s.gdn_b)
        .arg_ptr(w.a_log)
        .arg_ptr(w.dt_bias)
        .arg_ptr(s.decay)
        .arg_ptr(s.beta)
        .arg_u32(nv)
        .launch(stream)?;
    // Conv over [q | k | v] with SiLU; q and k heads are L2-normalised.
    ops::conv1d_update_l2norm(
        gpu,
        k.conv1d_l2norm_f32,
        state.conv_state,
        s.qkvz,
        &DenseWeight { weight: w.conv },
        s.conv_out,
        conv_dim,
        config.linear_conv_kernel_dim as u32,
        1,
        2 * nk * kd,
        kd,
        GDN_L2_EPS,
        stream,
    )?;
    ops::gdn_decode(
        gpu,
        k.gdn_decode_f32,
        state.h_state,
        s.conv_out,
        s.conv_out.offset(key_dim * 4),
        s.conv_out.offset(2 * key_dim * 4),
        s.decay,
        s.beta,
        s.core,
        1,
        nk,
        nv,
        kd,
        vd,
        stream,
    )?;
    KernelLaunch::new(gpu, k.gdn_out_norm)
        .grid([nv, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(s.core)
        .arg_ptr(z)
        .arg_ptr(w.norm)
        .arg_ptr(s.gdn_normed)
        .arg_u32(vd)
        .arg_f32(config.rms_norm_eps as f32)
        .arg_u32(u32::from(sigmoid_gate))
        .launch(stream)?;
    mxfp8_gemv(gpu, k, s.gdn_normed, &w.out, s.y, stream)
}

/// Which experts of a projection get their block scales expanded.
#[derive(Clone, Copy)]
pub(super) enum Routed {
    /// The `n` expert ids of one token (u32 on the device).
    Ids(DevicePtr, u32),
    /// Every expert with a non-empty range in the i32 `[E + 1]` prefix sums
    /// the MoE sort produces.
    Offsets(DevicePtr, u32),
}

/// Expands the block scales of the routed experts of one projection into
/// their scratch slots.
pub(super) fn expand_scales(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    p: &ExpertProj,
    routed: Routed,
    out: DevicePtr,
    (rows, cols): (usize, usize),
    stream: u64,
) -> Result<()> {
    let (ids, offsets, n) = match routed {
        Routed::Ids(ids, n) => (ids, DevicePtr::NULL, n),
        Routed::Offsets(offsets, n) => (DevicePtr::NULL, offsets, n),
    };
    KernelLaunch::new(gpu, k.csf_expand_fixed)
        .grid([(rows / 16) as u32, n, 1])
        .block([256, 1, 1])
        .arg_ptr(p.fixed_ptrs)
        .arg_ptr(ids)
        .arg_ptr(offsets)
        .arg_ptr(out)
        .arg_u32(rows as u32)
        .arg_u32(cols as u32)
        .launch(stream)?;
    if p.max_exc == 0 {
        return Ok(());
    }
    KernelLaunch::new(gpu, k.csf_expand_exceptions)
        .grid([p.max_exc.div_ceil(256), n, 1])
        .block([256, 1, 1])
        .arg_ptr(p.exc_ptrs)
        .arg_ptr(p.exc_counts)
        .arg_ptr(ids)
        .arg_ptr(offsets)
        .arg_ptr(out)
        .arg_u32(rows as u32)
        .arg_u32(cols as u32)
        .launch(stream)
}

/// MoE: top-k routed NVFP4 experts plus the gated MXFP8 shared expert.
pub(super) fn moe(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    s: &Scratch,
    config: &ModelConfig,
    w: &MoeWeights,
    stream: u64,
) -> Result<()> {
    let h = config.hidden_size as u32;
    let (e, top_k) = (config.num_experts as u32, config.num_experts_per_tok as u32);
    let inter = config.moe_intermediate_size as u32;
    let sh_inter = config.shared_expert_intermediate_size as u32;
    let [gate_up_dims, down_dims] = expert_scale_dims(config);

    let router = DenseWeight { weight: w.router };
    ops::dense_gemv(
        gpu,
        k.dense_gemv,
        s.x,
        &router,
        s.router_logits,
        e,
        h,
        stream,
    )
    .context("MoE router")?;
    ops::moe_topk_softmax(
        gpu,
        k.moe_topk,
        s.router_logits,
        s.expert_ids,
        s.expert_weights,
        e,
        top_k,
        config.norm_topk_prob,
        stream,
    )?;
    expand_scales(
        gpu,
        k,
        &w.gate,
        Routed::Ids(s.expert_ids, top_k),
        s.gate_scales,
        gate_up_dims,
        stream,
    )?;
    expand_scales(
        gpu,
        k,
        &w.up,
        Routed::Ids(s.expert_ids, top_k),
        s.up_scales,
        gate_up_dims,
        stream,
    )?;
    expand_scales(
        gpu,
        k,
        &w.down,
        Routed::Ids(s.expert_ids, top_k),
        s.down_scales,
        down_dims,
        stream,
    )?;
    ops::moe_expert_gemv_gate_up(
        gpu,
        k.moe_gate_up,
        s.x,
        w.gate.packed_ptrs,
        s.gate_scale_ptrs,
        w.gate.scale2,
        s.expert_gate,
        w.up.packed_ptrs,
        s.up_scale_ptrs,
        w.up.scale2,
        s.expert_up,
        s.expert_ids,
        inter,
        h,
        top_k,
        stream,
    )?;
    ops::moe_expert_gemv_silu_down(
        gpu,
        k.moe_silu_down,
        s.expert_gate,
        s.expert_up,
        w.down.packed_ptrs,
        s.down_scale_ptrs,
        w.down.scale2,
        s.expert_down,
        s.expert_ids,
        h,
        inter,
        top_k,
        stream,
    )?;
    // Shared expert: down(silu(gate(x)) * up(x)).
    mxfp8_gemv(gpu, k, s.x, &w.sh_gate, s.sh_gate, stream)?;
    mxfp8_gemv(gpu, k, s.x, &w.sh_up, s.sh_up, stream)?;
    KernelLaunch::new(gpu, k.swiglu_mul)
        .grid([sh_inter.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(s.sh_gate)
        .arg_ptr(s.sh_up)
        .arg_ptr(s.sh_act)
        .arg_u32(sh_inter)
        .launch(stream)?;
    mxfp8_gemv(gpu, k, s.sh_act, &w.sh_down, s.sh_down, stream)?;
    // y = sum_e weight_e * expert_e + sigmoid(shared_gate . x) * shared
    ops::moe_weighted_sum_blend(
        gpu,
        k.moe_blend,
        s.y,
        s.expert_down,
        s.expert_weights,
        s.sh_down,
        s.x,
        w.shared_gate,
        h,
        top_k,
        h,
        stream,
    )
}

// SPDX-License-Identifier: AGPL-3.0-only

//! The three sublayers for a chunk of tokens. Each reads the mixed streams
//! `scratch.x` (BF16 `[T, H]`) and writes `scratch.y` (BF16 `[T, H]`).

use anyhow::{Context, Result};
use atlas_core::config::{ModelConfig, Qwen4ExpConfig};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::PagedKvCache;

use super::Pass;
use crate::layer::{AttnMetadataDev, SsmLayerState};
use crate::layers::ops;
use crate::layers::qwen4_exp::attention::{AttnGeom, index_write, select_attend};
use crate::layers::qwen4_exp::scratch::expert_scale_dims;
use crate::layers::qwen4_exp::sublayers::{GDN_L2_EPS, Routed, expand_scales};
use crate::layers::qwen4_exp::weights::{AttnWeights, GdnWeights, MoeWeights};
use crate::weight_map::DenseWeight;

/// Gated delta net: the recurrence runs over the chunk in one launch and
/// leaves the sequence state where a decode step continues.
pub(in crate::layers::qwen4_exp) fn gdn(
    p: &Pass<'_>,
    config: &ModelConfig,
    sigmoid_gate: bool,
    w: &GdnWeights,
    state: &SsmLayerState,
) -> Result<()> {
    let (gpu, s, stream, n) = (p.gpu, p.s, p.stream, p.n);
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
    // wide = [q | k | v] per token, mid_a = z, mid_b = recurrence output.
    let (qkv, z, core, normed) = (s.wide, s.mid_a, s.mid_b, s.mid_c);

    p.linear(s.x, &w.qkv, qkv)?;
    p.linear(s.x, &w.z, z)?;
    p.linear(s.x, &w.a, s.gdn_a)?;
    p.linear(s.x, &w.b, s.gdn_b)?;
    KernelLaunch::new(gpu, p.k.gdn_gates_split)
        .grid([n, 1, 1])
        .block([64, 1, 1])
        .arg_ptr(s.gdn_a)
        .arg_ptr(s.gdn_b)
        .arg_ptr(w.a_log)
        .arg_ptr(w.dt_bias)
        .arg_ptr(s.decay)
        .arg_ptr(s.beta)
        .arg_u32(nv)
        .launch(stream)?;
    // Conv over [q | k | v] with SiLU, in place; then q and k heads are
    // L2-normalised.
    ops::conv1d_update_prefill(
        gpu,
        p.pk.conv1d,
        state.conv_state,
        qkv,
        &DenseWeight { weight: w.conv },
        DevicePtr::NULL,
        qkv,
        conv_dim,
        config.linear_conv_kernel_dim as u32,
        n,
        conv_dim,
        conv_dim,
        stream,
    )?;
    ops::l2_norm(
        gpu,
        p.pk.l2_norm,
        qkv,
        2 * nk,
        kd,
        GDN_L2_EPS,
        n,
        conv_dim,
        stream,
    )?;
    ops::gdn_prefill_persistent(
        gpu,
        p.pk.gdn,
        state.h_state,
        qkv,
        qkv.offset(key_dim * 2),
        qkv.offset(2 * key_dim * 2),
        s.decay,
        s.beta,
        core,
        1,
        n,
        nk,
        nv,
        kd,
        vd,
        conv_dim,
        conv_dim,
        nv,
        stream,
    )?;
    KernelLaunch::new(gpu, p.pk.gdn_out_norm_bf16)
        .grid([n * nv, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(core)
        .arg_ptr(z)
        .arg_ptr(w.norm)
        .arg_ptr(normed)
        .arg_u32(vd)
        .arg_f32(config.rms_norm_eps as f32)
        .arg_u32(u32::from(sigmoid_gate))
        .launch(stream)?;
    p.linear(normed, &w.out, s.y)
}

/// Attention for tokens at positions `start..start + n`. K/V and index keys
/// of the whole chunk are stored first. Queries that still see the whole
/// context run as one causal attention pass over the cache; beyond that every
/// query selects its own blocks.
pub(in crate::layers::qwen4_exp) fn attention(
    p: &Pass<'_>,
    config: &ModelConfig,
    q: &Qwen4ExpConfig,
    w: &AttnWeights,
    kv: &PagedKvCache,
    meta: &AttnMetadataDev,
    start: usize,
) -> Result<()> {
    let (gpu, s, stream, n) = (p.gpu, p.s, p.stream, p.n);
    let g = AttnGeom::new(config, q, w, kv)?;
    let (q_dim, kv_dim) = (g.nq * g.hd, g.nkv * g.hd);
    let index_width = ((g.heads + 1) * g.dim) as usize;
    // wide = per head [query | gate], then [.. | all gates]; mid_a = queries,
    // mid_b = attention output, mid_c = gated output.
    let (qg, queries, out, gated) = (s.wide, s.mid_a, s.mid_b, s.mid_c);
    let gates = qg.offset(q_dim as usize * 2);

    p.linear(s.x, &w.q, qg)?;
    ops::deinterleave_qg_split(
        gpu,
        p.pk.deinterleave_split,
        qg,
        queries,
        n,
        g.nq,
        g.hd,
        2 * q_dim,
        stream,
    )?;
    p.linear(s.x, &w.k, s.k)?;
    p.linear(s.x, &w.v, s.v)?;
    let (qn, kn) = (
        DenseWeight { weight: w.q_norm },
        DenseWeight { weight: w.k_norm },
    );
    let norm = p.k.rms_norm;
    ops::rms_norm(
        gpu,
        norm,
        queries,
        &qn,
        queries,
        n * g.nq,
        g.hd,
        g.eps,
        stream,
    )?;
    ops::rms_norm(gpu, norm, s.k, &kn, s.k, n * g.nkv, g.hd, g.eps, stream)?;
    ops::rope(
        gpu,
        p.k.rope,
        queries,
        s.k,
        meta.positions,
        n,
        g.nq,
        g.nkv,
        g.hd,
        g.rot,
        g.theta,
        stream,
    )?;
    ops::reshape_and_cache(
        gpu,
        p.k.reshape_and_cache,
        s.k,
        s.v,
        g.k_pool,
        g.v_pool,
        meta.slot,
        n,
        g.nkv,
        g.hd,
        g.bs,
        kv_dim,
        kv_dim,
        g.cache_stride,
        stream,
    )?;
    p.linear(s.x, &w.index_qk, s.index_qk)?;
    index_write(
        gpu,
        p.k,
        &g,
        w,
        s.index_qk,
        meta.block_table,
        meta.positions,
        n,
        stream,
    )?;

    let end = start + n as usize;
    if end <= g.full_attention_tokens() {
        ops::prefill_attention_paged(
            gpu,
            p.pk.attn_paged,
            queries,
            g.k_pool,
            g.v_pool,
            out,
            meta.block_table,
            n,
            end as u32,
            start as u32,
            g.nq,
            g.nkv,
            g.hd,
            g.bs,
            0,
            1.0 / (g.hd as f32).sqrt(),
            stream,
        )?;
    } else {
        for t in 0..n as usize {
            select_attend(
                gpu,
                p.k,
                p.s1,
                &g,
                w,
                queries.offset(t * q_dim as usize * 2),
                s.index_qk.offset(t * index_width * 2),
                meta.block_table,
                meta.positions.offset(t * 4),
                out.offset(t * q_dim as usize * 2),
                stream,
            )?;
        }
    }
    ops::sigmoid_gate_mul_batched(
        gpu,
        p.pk.sigmoid_gate_mul,
        out,
        gates,
        gated,
        q_dim,
        2 * q_dim,
        n,
        stream,
    )?;
    p.linear(gated, &w.o, s.y)
}

/// MoE: tokens are grouped by expert, so each routed expert's weights are
/// read once for all tokens that chose it.
pub(in crate::layers::qwen4_exp) fn moe(
    p: &Pass<'_>,
    config: &ModelConfig,
    w: &MoeWeights,
) -> Result<()> {
    let (gpu, s, s1, stream, n) = (p.gpu, p.s, p.s1, p.stream, p.n);
    let h = config.hidden_size as u32;
    let (e, top_k) = (config.num_experts as u32, config.num_experts_per_tok as u32);
    let inter = config.moe_intermediate_size as u32;
    let sh_inter = config.shared_expert_intermediate_size as u32;
    let [gate_up_dims, down_dims] = expert_scale_dims(config);
    let routed_rows = n * top_k;

    p.gemm(s.x, w.router, s.router, e, h)
        .context("MoE router")?;
    ops::moe_topk_softmax_batched(
        gpu,
        p.pk.moe_topk,
        s.router,
        s.expert_ids,
        s.expert_weights,
        e,
        top_k,
        config.norm_topk_prob,
        n,
        stream,
    )?;
    ops::moe_sort_by_expert(
        gpu,
        p.pk.moe_sort,
        s.expert_ids,
        s.sorted_tokens,
        s.sorted_experts,
        s.expert_offsets,
        s.token_to_perm,
        routed_rows,
        e,
        top_k,
        stream,
    )?;
    let routed = Routed::Offsets(s.expert_offsets, e);
    for (proj, slots, dims) in [
        (&w.gate, s1.gate_scales, gate_up_dims),
        (&w.up, s1.up_scales, gate_up_dims),
        (&w.down, s1.down_scales, down_dims),
    ] {
        expand_scales(gpu, p.k, proj, routed, slots, dims, stream)?;
    }
    // An expert serves at most every token of the chunk once.
    let m_tiles = n.div_ceil(64);
    for (proj, slot_ptrs, out) in [
        (&w.gate, s1.gate_scale_ptrs, s.expert_gate),
        (&w.up, s1.up_scale_ptrs, s.expert_up),
    ] {
        ops::moe_w4a16_grouped_gemm_ptrtable(
            gpu,
            p.pk.moe_grouped,
            s.x,
            proj.packed_ptrs,
            slot_ptrs,
            proj.scale2,
            out,
            s.expert_offsets,
            s.sorted_tokens,
            e,
            inter,
            h,
            m_tiles,
            stream,
        )?;
    }
    let swiglu = |gate: DevicePtr, up: DevicePtr, len: u32| {
        KernelLaunch::new(gpu, p.k.swiglu_mul)
            .grid([len.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(gate)
            .arg_ptr(up)
            .arg_ptr(up)
            .arg_u32(len)
            .launch(stream)
    };
    swiglu(s.expert_gate, s.expert_up, routed_rows * inter)?;
    ops::moe_w4a16_grouped_gemm_ptrtable(
        gpu,
        p.pk.moe_grouped,
        s.expert_up,
        w.down.packed_ptrs,
        s1.down_scale_ptrs,
        w.down.scale2,
        s.expert_down,
        s.expert_offsets,
        DevicePtr::NULL,
        e,
        h,
        inter,
        m_tiles,
        stream,
    )?;
    ops::moe_unpermute_reduce_indexed(
        gpu,
        p.pk.moe_unpermute,
        s.expert_down,
        s.y,
        s.token_to_perm,
        s.expert_weights,
        h,
        n,
        top_k,
        stream,
    )?;
    // Shared expert: down(silu(gate(x)) * up(x)), blended in with its gate.
    p.linear(s.x, &w.sh_gate, s.sh_gate)?;
    p.linear(s.x, &w.sh_up, s.sh_up)?;
    swiglu(s.sh_gate, s.sh_up, n * sh_inter)?;
    p.linear(s.sh_up, &w.sh_down, s.sh_down)?;
    ops::moe_batched_blend(
        gpu,
        p.pk.moe_blend,
        s.y,
        s.sh_down,
        s.x,
        w.shared_gate,
        h,
        n,
        stream,
    )
}

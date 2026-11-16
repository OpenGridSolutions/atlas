// SPDX-License-Identifier: AGPL-3.0-only

//! Gated sparse attention (BF16 KV): a query attends to the tokens of the
//! index blocks it selects plus the incomplete block at the end. While no
//! more blocks are visible than a query may select, that is the whole context.
//!
//! The pieces here take one query at a time; the decode step uses them
//! directly, the prefill path once the context has outgrown full attention.

use anyhow::{Result, bail};
use atlas_core::config::{ModelConfig, Qwen4ExpConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::PagedKvCache;

use super::kernels::{Kernels, mxfp8_gemv};
use super::scratch::{SEL_BLOCK, Scratch, sel_rows};
use super::weights::AttnWeights;
use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// Geometry and cache planes of one attention layer.
#[derive(Clone, Copy)]
pub(super) struct AttnGeom {
    pub nq: u32,
    pub nkv: u32,
    pub hd: u32,
    pub eps: f32,
    pub rot: u32,
    pub theta: f32,
    /// Tokens per KV block, tokens per index block.
    pub bs: u32,
    pub ratio: u32,
    /// Index query heads, their width, blocks a query may select.
    pub heads: u32,
    pub dim: u32,
    pub topk: u32,
    pub max_blocks: u32,
    pub sel_blocks: u32,
    pub aux_block_bytes: u64,
    pub aux: DevicePtr,
    pub k_pool: DevicePtr,
    pub v_pool: DevicePtr,
    pub cache_stride: u64,
}

impl AttnGeom {
    pub(super) fn new(
        config: &ModelConfig,
        q: &Qwen4ExpConfig,
        w: &AttnWeights,
        kv: &PagedKvCache,
    ) -> Result<Self> {
        let (bs, ratio) = (kv.block_size(), q.indexer_compress_ratio);
        let Some(aux) = kv.aux_pool_ptr(w.attn_idx) else {
            bail!("qwen4_exp attention needs the KV cache's index plane");
        };
        if !bs.is_multiple_of(ratio) || kv.aux_block_bytes() != q.index_block_bytes(bs) {
            bail!("KV block size {bs} does not fit index blocks of {ratio} tokens");
        }
        Ok(Self {
            nq: config.num_attention_heads as u32,
            nkv: config.num_key_value_heads as u32,
            hd: config.head_dim as u32,
            eps: config.rms_norm_eps as f32,
            rot: config.rotary_dim() as u32,
            theta: config.rope_theta as f32,
            bs: bs as u32,
            ratio: ratio as u32,
            heads: q.indexer_n_heads as u32,
            dim: q.indexer_head_dim as u32,
            topk: q.indexer_top_blocks() as u32,
            max_blocks: (config.max_position_embeddings / ratio) as u32,
            sel_blocks: sel_rows(q).div_ceil(SEL_BLOCK) as u32,
            aux_block_bytes: kv.aux_block_bytes() as u64,
            aux,
            k_pool: kv.k_pool_ptr(w.attn_idx),
            v_pool: kv.v_pool_ptr(w.attn_idx),
            cache_stride: kv.cache_stride() as u64,
        })
    }

    /// Tokens from the start of a sequence for which every query still
    /// attends the whole context.
    pub(super) fn full_attention_tokens(&self) -> usize {
        ((self.topk + 1) * self.ratio - 1) as usize
    }
}

/// Stores the index keys of `n` tokens: every token's raw key, and the block
/// key of every index block these tokens complete. `index_qk` holds
/// `[n, (heads + 1) * dim]`, `positions` the tokens' positions.
#[allow(clippy::too_many_arguments)]
pub(super) fn index_write(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    g: &AttnGeom,
    w: &AttnWeights,
    index_qk: DevicePtr,
    block_table: DevicePtr,
    positions: DevicePtr,
    n: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k.qsa_index_raw)
        .grid([n, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(index_qk)
        .arg_ptr(g.aux)
        .arg_ptr(block_table)
        .arg_ptr(positions)
        .arg_u64(g.aux_block_bytes)
        .arg_u32(g.bs)
        .arg_u32(g.dim)
        .arg_u32(g.heads)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.qsa_index_pool)
        .grid([n, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(w.index_k_norm)
        .arg_ptr(g.aux)
        .arg_ptr(block_table)
        .arg_ptr(positions)
        .arg_u64(g.aux_block_bytes)
        .arg_u32(g.bs)
        .arg_u32(g.ratio)
        .arg_u32(g.dim)
        .arg_u32(g.rot)
        .arg_f32(g.theta)
        .arg_f32(g.eps)
        .launch(stream)
}

/// One query: scores the visible index blocks, selects, gathers the selected
/// tokens' K/V and attends over them. `query` is `[nq * hd]` (normed,
/// rotated), `index_qk` the token's index queries, `position` its position
/// (u32 on the device); the result goes to `out` (`[nq * hd]`). K/V and index
/// keys up to the token must already be in the cache.
#[allow(clippy::too_many_arguments)]
pub(super) fn select_attend(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    s: &Scratch,
    g: &AttnGeom,
    w: &AttnWeights,
    query: DevicePtr,
    index_qk: DevicePtr,
    block_table: DevicePtr,
    position: DevicePtr,
    out: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k.qsa_score)
        .grid([g.max_blocks.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(index_qk)
        .arg_ptr(w.index_q_norm)
        .arg_ptr(g.aux)
        .arg_ptr(block_table)
        .arg_ptr(position)
        .arg_ptr(s.index_scores)
        .arg_u64(g.aux_block_bytes)
        .arg_u32(g.bs)
        .arg_u32(g.ratio)
        .arg_u32(g.dim)
        .arg_u32(g.heads)
        .arg_u32(g.rot)
        .arg_f32(g.theta)
        .arg_f32(g.eps)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.qsa_select)
        .grid([1, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(s.index_scores)
        .arg_ptr(position)
        .arg_ptr(s.index_selected)
        .arg_ptr(s.index_count)
        .arg_u32(g.ratio)
        .arg_u32(g.topk)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.qsa_gather)
        .grid([g.topk + 1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(s.index_selected)
        .arg_ptr(s.index_count)
        .arg_ptr(position)
        .arg_ptr(block_table)
        .arg_ptr(g.k_pool)
        .arg_ptr(g.v_pool)
        .arg_ptr(s.sel_k)
        .arg_ptr(s.sel_v)
        .arg_ptr(s.sel_tokens)
        .arg_u32(g.bs)
        .arg_u32(g.ratio)
        .arg_u32(g.nkv * g.hd)
        .arg_u32(g.topk)
        .launch(stream)?;
    ops::paged_decode_attn_bf16(
        gpu,
        k.paged_decode,
        query,
        s.sel_k,
        s.sel_v,
        out,
        s.sel_block_table,
        s.sel_tokens,
        g.sel_blocks,
        1,
        g.nq,
        g.nkv,
        g.hd,
        SEL_BLOCK as u32,
        1.0 / (g.hd as f32).sqrt(),
        g.nq * g.hd,
        0,
        stream,
    )
}

/// Attention sublayer for one token: reads `scratch.x`, writes `scratch.y`.
#[allow(clippy::too_many_arguments)]
pub(super) fn attention(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    s: &Scratch,
    config: &ModelConfig,
    q: &Qwen4ExpConfig,
    w: &AttnWeights,
    kv: &PagedKvCache,
    meta: &AttnMetadataDev,
    stream: u64,
) -> Result<()> {
    let g = AttnGeom::new(config, q, w, kv)?;
    let q_dim = (g.nq * g.hd) as usize;
    let gate = s.qg.offset(q_dim * 2);

    mxfp8_gemv(gpu, k, s.x, &w.q, s.qg, stream)?;
    // per head [query | gate] -> [all queries | all gates]
    ops::deinterleave_qg(
        gpu,
        k.deinterleave_qg,
        s.qg,
        1,
        g.nq,
        g.hd,
        2 * g.nq * g.hd,
        stream,
    )?;
    mxfp8_gemv(gpu, k, s.x, &w.k, s.k, stream)?;
    mxfp8_gemv(gpu, k, s.x, &w.v, s.v, stream)?;
    let (qn, kn) = (
        DenseWeight { weight: w.q_norm },
        DenseWeight { weight: w.k_norm },
    );
    ops::rms_norm(gpu, k.rms_norm, s.qg, &qn, s.qg, g.nq, g.hd, g.eps, stream)?;
    ops::rms_norm(gpu, k.rms_norm, s.k, &kn, s.k, g.nkv, g.hd, g.eps, stream)?;
    ops::rope(
        gpu,
        k.rope,
        s.qg,
        s.k,
        meta.positions,
        1,
        g.nq,
        g.nkv,
        g.hd,
        g.rot,
        g.theta,
        stream,
    )?;
    ops::reshape_and_cache(
        gpu,
        k.reshape_and_cache,
        s.k,
        s.v,
        g.k_pool,
        g.v_pool,
        meta.slot,
        1,
        g.nkv,
        g.hd,
        g.bs,
        g.nkv * g.hd,
        g.nkv * g.hd,
        g.cache_stride,
        stream,
    )?;

    mxfp8_gemv(gpu, k, s.x, &w.index_qk, s.index_qk, stream)?;
    index_write(
        gpu,
        k,
        &g,
        w,
        s.index_qk,
        meta.block_table,
        meta.positions,
        1,
        stream,
    )?;
    select_attend(
        gpu,
        k,
        s,
        &g,
        w,
        s.qg,
        s.index_qk,
        meta.block_table,
        meta.positions,
        s.attn_out,
        stream,
    )?;
    ops::sigmoid_gate_mul(
        gpu,
        k.sigmoid_gate_mul,
        s.attn_out,
        gate,
        s.attn_gated,
        g.nq * g.hd,
        stream,
    )?;
    mxfp8_gemv(gpu, k, s.attn_gated, &w.o, s.y, stream)
}

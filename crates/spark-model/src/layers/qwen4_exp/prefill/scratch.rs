// SPDX-License-Identifier: AGPL-3.0-only

//! Working memory of the prefill path: one device allocation for up to
//! [`PREFILL_ROWS`] tokens, shared by all layers. Wide buffers serve the GDN
//! and the attention sublayer alike — a layer has one or the other.

use anyhow::{Context, Result};
use atlas_core::config::{ModelConfig, Qwen4ExpConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Most tokens one prefill call takes. Every expert a chunk routes to is read
/// once per chunk, so larger chunks cost less weight traffic per token; the
/// working memory grows linearly with this number.
pub const PREFILL_ROWS: usize = 512;

pub(in crate::layers::qwen4_exp) struct PrefillScratch {
    base: DevicePtr,
    pub bytes: usize,
    /// FP32 `[T, hc, H]`: the residual streams of the chunk.
    pub streams: DevicePtr,
    // ── stream mixer ──
    /// BF16 `[T, hc * H]` each: group-normed streams, gate logits.
    pub normed: DevicePtr,
    pub gate_u: DevicePtr,
    /// BF16 `[T, rank]`, `[T, hc]`.
    pub act: DevicePtr,
    pub inj_logits: DevicePtr,
    /// FP32 `[T, hc]`: injection weights of the first / second site.
    pub inj_a: DevicePtr,
    pub inj_b: DevicePtr,
    /// BF16 `[T, H]`: sublayer input / output.
    pub x: DevicePtr,
    pub y: DevicePtr,
    // ── GDN / attention ──
    /// BF16 `[T, wide]`, wide = max(GDN conv width, 2 * attention width).
    pub wide: DevicePtr,
    /// BF16 `[T, mid]` x3, mid = max(GDN value width, attention width).
    pub mid_a: DevicePtr,
    pub mid_b: DevicePtr,
    pub mid_c: DevicePtr,
    /// BF16 `[T, nv]` x2, FP32 `[T, nv]` x2.
    pub gdn_a: DevicePtr,
    pub gdn_b: DevicePtr,
    pub decay: DevicePtr,
    pub beta: DevicePtr,
    /// BF16 `[T, nkv * hd]` each.
    pub k: DevicePtr,
    pub v: DevicePtr,
    /// BF16 `[T, (index heads + 1) * index dim]`.
    pub index_qk: DevicePtr,
    // ── MoE ──
    /// BF16 `[T, E]`.
    pub router: DevicePtr,
    /// u32 `[T, top_k]`, f32 `[T, top_k]`.
    pub expert_ids: DevicePtr,
    pub expert_weights: DevicePtr,
    /// i32 `[T * top_k]` x3, i32 `[E + 1]`.
    pub sorted_tokens: DevicePtr,
    pub sorted_experts: DevicePtr,
    pub token_to_perm: DevicePtr,
    pub expert_offsets: DevicePtr,
    /// BF16 `[T * top_k, inter]` x2, `[T * top_k, H]`.
    pub expert_gate: DevicePtr,
    pub expert_up: DevicePtr,
    pub expert_down: DevicePtr,
    /// BF16 `[T, shared_inter]` x2, `[T, H]`.
    pub sh_gate: DevicePtr,
    pub sh_up: DevicePtr,
    pub sh_down: DevicePtr,
    /// BF16: one MXFP8 projection expanded for the duration of a GEMM.
    pub weight: DevicePtr,
    // ── n-gram layer ──
    /// u8 `[T, heads * head_dim / 2]`, `[T, heads * head_dim / 16]`.
    pub ple_rows: DevicePtr,
    pub ple_row_scales: DevicePtr,
    /// BF16 `[T, E]`.
    pub ple_emb: DevicePtr,
    /// FP32 `[T, hc * H]` each.
    pub ple_gated: DevicePtr,
    pub ple_conv_in: DevicePtr,
}

impl PrefillScratch {
    pub(in crate::layers::qwen4_exp) fn new(
        gpu: &dyn GpuBackend,
        config: &ModelConfig,
        q: &Qwen4ExpConfig,
    ) -> Result<Self> {
        let t = PREFILL_ROWS;
        let (h, hc) = (config.hidden_size, config.hc_mult);
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_dim = 2 * key_dim + value_dim;
        let nv = config.linear_num_value_heads;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let (e, top_k) = (config.num_experts, config.num_experts_per_tok);
        let (inter, sh_inter) = (
            config.moe_intermediate_size,
            config.shared_expert_intermediate_size,
        );
        let wide = conv_dim.max(2 * q_dim);
        let mid = value_dim.max(q_dim);
        let index_dim = (q.indexer_n_heads + q.indexer_kv_heads) * q.indexer_head_dim;
        let ple_e = q.ngram_heads() * q.ngram_head_dim();
        // Largest MXFP8 projection, in elements: every one maps to or from H.
        let weight_elems = h * wide.max(mid).max(sh_inter).max(index_dim);

        let mut end = 0usize;
        let mut take = |bytes: usize| {
            let at = end;
            end += bytes.div_ceil(256) * 256;
            at
        };
        let o = [
            take(t * hc * h * 4),        // 0 streams
            take(t * hc * h * 2),        // 1 normed
            take(t * hc * h * 2),        // 2 gate_u
            take(t * q.hc_lowrank * 2),  // 3 act
            take(t * hc * 2),            // 4 inj_logits
            take(t * hc * 4),            // 5 inj_a
            take(t * hc * 4),            // 6 inj_b
            take(t * h * 2),             // 7 x
            take(t * h * 2),             // 8 y
            take(t * wide * 2),          // 9 wide
            take(t * mid * 2),           // 10 mid_a
            take(t * mid * 2),           // 11 mid_b
            take(t * mid * 2),           // 12 mid_c
            take(t * nv * 2),            // 13 gdn_a
            take(t * nv * 2),            // 14 gdn_b
            take(t * nv * 4),            // 15 decay
            take(t * nv * 4),            // 16 beta
            take(t * kv_dim * 2),        // 17 k
            take(t * kv_dim * 2),        // 18 v
            take(t * index_dim * 2),     // 19 index_qk
            take(t * e * 2),             // 20 router
            take(t * top_k * 4),         // 21 expert_ids
            take(t * top_k * 4),         // 22 expert_weights
            take(t * top_k * 4),         // 23 sorted_tokens
            take(t * top_k * 4),         // 24 sorted_experts
            take(t * top_k * 4),         // 25 token_to_perm
            take((e + 1) * 4),           // 26 expert_offsets
            take(t * top_k * inter * 2), // 27 expert_gate
            take(t * top_k * inter * 2), // 28 expert_up
            take(t * top_k * h * 2),     // 29 expert_down
            take(t * sh_inter * 2),      // 30 sh_gate
            take(t * sh_inter * 2),      // 31 sh_up
            take(t * h * 2),             // 32 sh_down
            take(weight_elems * 2),      // 33 weight
            take(t * ple_e / 2),         // 34 ple_rows
            take(t * ple_e / 16),        // 35 ple_row_scales
            take(t * ple_e * 2),         // 36 ple_emb
            take(t * hc * h * 4),        // 37 ple_gated
            take(t * hc * h * 4),        // 38 ple_conv_in
        ];
        let bytes = end;
        let base = gpu
            .alloc(bytes)
            .with_context(|| format!("qwen4_exp prefill working memory ({bytes} bytes)"))?;
        let at = |i: usize| base.offset(o[i]);
        Ok(Self {
            base,
            bytes,
            streams: at(0),
            normed: at(1),
            gate_u: at(2),
            act: at(3),
            inj_logits: at(4),
            inj_a: at(5),
            inj_b: at(6),
            x: at(7),
            y: at(8),
            wide: at(9),
            mid_a: at(10),
            mid_b: at(11),
            mid_c: at(12),
            gdn_a: at(13),
            gdn_b: at(14),
            decay: at(15),
            beta: at(16),
            k: at(17),
            v: at(18),
            index_qk: at(19),
            router: at(20),
            expert_ids: at(21),
            expert_weights: at(22),
            sorted_tokens: at(23),
            sorted_experts: at(24),
            token_to_perm: at(25),
            expert_offsets: at(26),
            expert_gate: at(27),
            expert_up: at(28),
            expert_down: at(29),
            sh_gate: at(30),
            sh_up: at(31),
            sh_down: at(32),
            weight: at(33),
            ple_rows: at(34),
            ple_row_scales: at(35),
            ple_emb: at(36),
            ple_gated: at(37),
            ple_conv_in: at(38),
        })
    }

    /// Frees the allocation (model unload).
    #[allow(dead_code)]
    pub(in crate::layers::qwen4_exp) fn free(&self, gpu: &dyn GpuBackend) {
        gpu.free(self.base).ok();
    }
}

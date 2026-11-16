// SPDX-License-Identifier: AGPL-3.0-only

//! Weight views of one `qwen4_exp` decoder layer. Every pointer refers to a
//! checkpoint tensor in the format it was stored in; nothing is converted.

use spark_runtime::gpu::DevicePtr;

/// MXFP8 linear: E4M3 weight `[n, k]` with one exponent byte per 32 columns.
#[derive(Debug, Clone, Copy)]
pub struct Mxfp8Linear {
    pub weight: DevicePtr,
    pub scale: DevicePtr,
    pub n: u32,
    pub k: u32,
}

/// Stream mixer of one site (before attention/GDN, before the MoE, final).
#[derive(Debug, Clone, Copy)]
pub struct Mixer {
    /// FP32 `[hc * H]`.
    pub norm_w: DevicePtr,
    /// BF16 `[rank, hc * H]`.
    pub down: DevicePtr,
    /// BF16 `[hc * H, rank]`.
    pub up: DevicePtr,
    /// BF16 `[hc, hc * H]`; absent in the final mixer.
    pub inject: Option<DevicePtr>,
}

/// Gated-delta-net sublayer.
#[derive(Debug, Clone, Copy)]
pub struct GdnWeights {
    pub qkv: Mxfp8Linear,
    pub z: Mxfp8Linear,
    pub a: Mxfp8Linear,
    pub b: Mxfp8Linear,
    pub out: Mxfp8Linear,
    /// BF16 `[conv_dim, d_conv]`.
    pub conv: DevicePtr,
    /// BF16 `[num_v_heads]`.
    pub a_log: DevicePtr,
    /// BF16 `[num_v_heads]`.
    pub dt_bias: DevicePtr,
    /// BF16 `[value_head_dim]`.
    pub norm: DevicePtr,
}

/// Attention sublayer.
#[derive(Debug, Clone, Copy)]
pub struct AttnWeights {
    /// `[2 * nq * hd, H]`, per head `[query | gate]`.
    pub q: Mxfp8Linear,
    pub k: Mxfp8Linear,
    pub v: Mxfp8Linear,
    pub o: Mxfp8Linear,
    /// BF16 `[head_dim]`, weight (1 + w).
    pub q_norm: DevicePtr,
    /// BF16 `[head_dim]`, weight (1 + w).
    pub k_norm: DevicePtr,
    /// Indexer projection `[(index heads + 1) * index dim, H]`: the index
    /// queries, then the token's raw index key.
    pub index_qk: Mxfp8Linear,
    /// BF16 `[index dim]`, weight (1 + w).
    pub index_q_norm: DevicePtr,
    /// BF16 `[index dim]`, weight (1 + w).
    pub index_k_norm: DevicePtr,
    /// Index among the attention layers (selects the KV pool).
    pub attn_idx: usize,
}

/// Device tables of one routed-expert projection (gate, up or down).
#[derive(Debug, Clone, Copy)]
pub struct ExpertProj {
    /// u64 `[E]`: packed NVFP4 weights.
    pub packed_ptrs: DevicePtr,
    /// f32 `[E]`: `weight_scale_2`.
    pub scale2: DevicePtr,
    /// u64 `[E]`: csf `fixed` streams.
    pub fixed_ptrs: DevicePtr,
    /// u64 `[E]`: csf exception records.
    pub exc_ptrs: DevicePtr,
    /// u32 `[E]`: number of exception records.
    pub exc_counts: DevicePtr,
    /// Largest entry of `exc_counts`.
    pub max_exc: u32,
}

/// MoE block.
#[derive(Debug, Clone, Copy)]
pub struct MoeWeights {
    /// BF16 `[E, H]`.
    pub router: DevicePtr,
    /// BF16 `[1, H]`.
    pub shared_gate: DevicePtr,
    pub sh_gate: Mxfp8Linear,
    pub sh_up: Mxfp8Linear,
    pub sh_down: Mxfp8Linear,
    pub gate: ExpertProj,
    pub up: ExpertProj,
    pub down: ExpertProj,
}

/// What a layer runs between its two mixers' first halves.
#[derive(Debug, Clone, Copy)]
pub enum Sublayer {
    Gdn(GdnWeights),
    Attn(AttnWeights),
}

/// N-gram layer weights (the table rows stay on disk).
#[derive(Debug, Clone, Copy)]
pub struct PleWeights {
    /// BF16 `[hc * H, E]`.
    pub key_proj: DevicePtr,
    /// BF16 `[H, E]`.
    pub value_proj: DevicePtr,
    /// BF16 `[hc * H, taps]`.
    pub conv: DevicePtr,
    /// FP32 `[hc * H]` each.
    pub w_key: DevicePtr,
    pub w_query: DevicePtr,
    pub w_conv: DevicePtr,
    /// Global scale of the NVFP4 table.
    pub scale2: f32,
}

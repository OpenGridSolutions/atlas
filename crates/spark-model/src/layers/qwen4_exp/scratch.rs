// SPDX-License-Identifier: AGPL-3.0-only

//! Working memory of the `qwen4_exp` layers: one device allocation made at
//! load time, shared by all layers (they run one after another), with a
//! fixed address per buffer so a captured decode graph stays valid.

use anyhow::{Context, Result};
use atlas_core::config::{ModelConfig, Qwen4ExpConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Bump allocator over one device buffer: collects sizes, then hands out
/// offsets. Every buffer is aligned to 256 bytes.
struct Layout(usize);
impl Layout {
    fn take(&mut self, bytes: usize) -> usize {
        let at = self.0;
        self.0 += bytes.div_ceil(256) * 256;
        at
    }
}

/// Buffers for one token. Names follow the data flow of a layer.
pub(super) struct Scratch {
    base: DevicePtr,
    pub bytes: usize,
    // ── stream mixer ──
    /// FP32 `[hc, H]`: group-normed streams.
    pub normed: DevicePtr,
    /// FP32 `[rank]`: bottleneck activations.
    pub act: DevicePtr,
    /// FP32 `[hc]`: injection weights of the first / second site.
    pub inj_a: DevicePtr,
    pub inj_b: DevicePtr,
    /// BF16 `[H]`: sublayer input / output.
    pub x: DevicePtr,
    pub y: DevicePtr,
    // ── GDN ──
    /// BF16 `[2*key_dim + value_dim | value_dim]`: conv input and gate z.
    pub qkvz: DevicePtr,
    /// BF16 `[nv]` each.
    pub gdn_a: DevicePtr,
    pub gdn_b: DevicePtr,
    /// FP32 `[nv]` each.
    pub decay: DevicePtr,
    pub beta: DevicePtr,
    /// FP32 `[2*key_dim + value_dim]`: conv output (q, k, v).
    pub conv_out: DevicePtr,
    /// FP32 `[value_dim]`: recurrence output.
    pub core: DevicePtr,
    /// BF16 `[value_dim]`: gated-normed output.
    pub gdn_normed: DevicePtr,
    // ── attention ──
    /// BF16 `[2 * nq * hd]`: `[Q | gate]` after de-interleaving.
    pub qg: DevicePtr,
    /// BF16 `[nkv * hd]` each.
    pub k: DevicePtr,
    pub v: DevicePtr,
    /// BF16 `[nq * hd]` each.
    pub attn_out: DevicePtr,
    pub attn_gated: DevicePtr,
    // ── sparse-attention indexer ──
    /// BF16 `[(index heads + 1) * index dim]`: index queries, raw key.
    pub index_qk: DevicePtr,
    /// FP32 `[max index blocks]`.
    pub index_scores: DevicePtr,
    /// i32 `[top blocks]`, i32 `[1]`: selected index blocks and their number.
    pub index_selected: DevicePtr,
    pub index_count: DevicePtr,
    /// i32 `[1]`: tokens gathered for the attention kernel.
    pub sel_tokens: DevicePtr,
    /// BF16 `[sel_rows, nkv * hd]` each: K/V of the selected tokens, laid out
    /// like a KV cache whose blocks are numbered by `sel_block_table`.
    pub sel_k: DevicePtr,
    pub sel_v: DevicePtr,
    /// i32 `[sel_blocks]`: 0, 1, 2, ...
    pub sel_block_table: DevicePtr,
    // ── MoE ──
    /// BF16 `[E]`.
    pub router_logits: DevicePtr,
    /// u32 `[top_k]`, f32 `[top_k]`.
    pub expert_ids: DevicePtr,
    pub expert_weights: DevicePtr,
    /// BF16 `[top_k, inter]` each, `[top_k, H]`.
    pub expert_gate: DevicePtr,
    pub expert_up: DevicePtr,
    pub expert_down: DevicePtr,
    /// BF16 `[shared_inter]` x3, `[H]`.
    pub sh_gate: DevicePtr,
    pub sh_up: DevicePtr,
    pub sh_act: DevicePtr,
    pub sh_down: DevicePtr,
    /// E4M3 `[E, rows, cols]`: expanded block scales, one slot per expert.
    pub gate_scales: DevicePtr,
    pub up_scales: DevicePtr,
    pub down_scales: DevicePtr,
    /// u64 `[E]`: address of each expert's slot in the areas above.
    pub gate_scale_ptrs: DevicePtr,
    pub up_scale_ptrs: DevicePtr,
    pub down_scale_ptrs: DevicePtr,
    // ── n-gram layer ──
    /// u8 `[heads, head_dim/2]`, `[heads, head_dim/16]`: table rows of the token.
    pub ple_rows: DevicePtr,
    pub ple_row_scales: DevicePtr,
    /// u32 `[1]`, always 0: slot index for kernels that address state as
    /// base + slot * stride when the base already is the sequence's state.
    pub zero_u32: DevicePtr,
    /// BF16 `[E]`, `[hc*H]`, `[H]`.
    pub ple_emb: DevicePtr,
    pub ple_key: DevicePtr,
    pub ple_value: DevicePtr,
    /// FP32 `[hc*H]` each.
    pub ple_gated: DevicePtr,
    pub ple_conv_in: DevicePtr,
}

/// Block size of the scratch cache the selected tokens are gathered into.
pub(super) const SEL_BLOCK: usize = 16;

/// Most tokens one query attends: the selected blocks plus an incomplete
/// block at the end.
pub(super) fn sel_rows(q: &Qwen4ExpConfig) -> usize {
    q.indexer_budget + q.indexer_compress_ratio - 1
}

/// Rows and scale columns of the gate/up and down projections of an expert.
pub(super) fn expert_scale_dims(config: &ModelConfig) -> [(usize, usize); 2] {
    let (h, inter) = (config.hidden_size, config.moe_intermediate_size);
    [(inter, h / 16), (h, inter / 16)]
}

impl Scratch {
    pub(super) fn new(
        gpu: &dyn GpuBackend,
        config: &ModelConfig,
        q: &Qwen4ExpConfig,
    ) -> Result<Self> {
        let h = config.hidden_size;
        let hc = config.hc_mult;
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
        let [(gu_rows, gu_cols), (d_rows, d_cols)] = expert_scale_dims(config);
        let (heads, head_dim, emb) = (q.ngram_heads(), q.ngram_head_dim(), q.ple_embed_dim);

        let mut l = Layout(0);
        let o_normed = l.take(hc * h * 4);
        let o_act = l.take(q.hc_lowrank * 4);
        let o_inj_a = l.take(hc * 4);
        let o_inj_b = l.take(hc * 4);
        let o_x = l.take(h * 2);
        let o_y = l.take(h * 2);
        let o_qkvz = l.take((conv_dim + value_dim) * 2);
        let o_gdn_a = l.take(nv * 2);
        let o_gdn_b = l.take(nv * 2);
        let o_decay = l.take(nv * 4);
        let o_beta = l.take(nv * 4);
        let o_conv_out = l.take(conv_dim * 4);
        let o_core = l.take(value_dim * 4);
        let o_gdn_normed = l.take(value_dim * 2);
        let o_qg = l.take(2 * q_dim * 2);
        let o_k = l.take(kv_dim * 2);
        let o_v = l.take(kv_dim * 2);
        let o_attn_out = l.take(q_dim * 2);
        let o_attn_gated = l.take(q_dim * 2);
        let max_index_blocks = config.max_position_embeddings / q.indexer_compress_ratio;
        let sel_blocks = sel_rows(q).div_ceil(SEL_BLOCK);
        let o_index_qk = l.take((q.indexer_n_heads + q.indexer_kv_heads) * q.indexer_head_dim * 2);
        let o_index_scores = l.take(max_index_blocks * 4);
        let o_index_selected = l.take(q.indexer_top_blocks() * 4);
        let o_index_count = l.take(4);
        let o_sel_tokens = l.take(4);
        let o_sel_k = l.take(sel_blocks * SEL_BLOCK * kv_dim * 2);
        let o_sel_v = l.take(sel_blocks * SEL_BLOCK * kv_dim * 2);
        let o_sel_bt = l.take(sel_blocks * 4);
        let o_router = l.take(e * 2);
        let o_ids = l.take(top_k * 4);
        let o_weights = l.take(top_k * 4);
        let o_e_gate = l.take(top_k * inter * 2);
        let o_e_up = l.take(top_k * inter * 2);
        let o_e_down = l.take(top_k * h * 2);
        let o_sh_gate = l.take(sh_inter * 2);
        let o_sh_up = l.take(sh_inter * 2);
        let o_sh_act = l.take(sh_inter * 2);
        let o_sh_down = l.take(h * 2);
        let o_gate_scales = l.take(e * gu_rows * gu_cols);
        let o_up_scales = l.take(e * gu_rows * gu_cols);
        let o_down_scales = l.take(e * d_rows * d_cols);
        let o_gate_ptrs = l.take(e * 8);
        let o_up_ptrs = l.take(e * 8);
        let o_down_ptrs = l.take(e * 8);
        let o_rows = l.take(heads * head_dim / 2);
        let o_row_scales = l.take(heads * head_dim / 16);
        let o_slot = l.take(4);
        let o_emb = l.take(emb * 2);
        let o_key = l.take(hc * h * 2);
        let o_value = l.take(h * 2);
        let o_gated = l.take(hc * h * 4);
        let o_conv_in = l.take(hc * h * 4);

        let bytes = l.0;
        let base = gpu
            .alloc(bytes)
            .with_context(|| format!("qwen4_exp working memory ({bytes} bytes)"))?;
        let at = |o: usize| base.offset(o);
        let s = Self {
            base,
            bytes,
            normed: at(o_normed),
            act: at(o_act),
            inj_a: at(o_inj_a),
            inj_b: at(o_inj_b),
            x: at(o_x),
            y: at(o_y),
            qkvz: at(o_qkvz),
            gdn_a: at(o_gdn_a),
            gdn_b: at(o_gdn_b),
            decay: at(o_decay),
            beta: at(o_beta),
            conv_out: at(o_conv_out),
            core: at(o_core),
            gdn_normed: at(o_gdn_normed),
            qg: at(o_qg),
            k: at(o_k),
            v: at(o_v),
            attn_out: at(o_attn_out),
            attn_gated: at(o_attn_gated),
            index_qk: at(o_index_qk),
            index_scores: at(o_index_scores),
            index_selected: at(o_index_selected),
            index_count: at(o_index_count),
            sel_tokens: at(o_sel_tokens),
            sel_k: at(o_sel_k),
            sel_v: at(o_sel_v),
            sel_block_table: at(o_sel_bt),
            router_logits: at(o_router),
            expert_ids: at(o_ids),
            expert_weights: at(o_weights),
            expert_gate: at(o_e_gate),
            expert_up: at(o_e_up),
            expert_down: at(o_e_down),
            sh_gate: at(o_sh_gate),
            sh_up: at(o_sh_up),
            sh_act: at(o_sh_act),
            sh_down: at(o_sh_down),
            gate_scales: at(o_gate_scales),
            up_scales: at(o_up_scales),
            down_scales: at(o_down_scales),
            gate_scale_ptrs: at(o_gate_ptrs),
            up_scale_ptrs: at(o_up_ptrs),
            down_scale_ptrs: at(o_down_ptrs),
            ple_rows: at(o_rows),
            ple_row_scales: at(o_row_scales),
            zero_u32: at(o_slot),
            ple_emb: at(o_emb),
            ple_key: at(o_key),
            ple_value: at(o_value),
            ple_gated: at(o_gated),
            ple_conv_in: at(o_conv_in),
        };
        gpu.copy_h2d(&0u32.to_le_bytes(), s.zero_u32)?;
        let identity: Vec<u8> = (0..sel_blocks as i32).flat_map(i32::to_le_bytes).collect();
        gpu.copy_h2d(&identity, s.sel_block_table)?;
        // Slot addresses of the expanded scales: fixed for the model's lifetime.
        for (table, area, slot) in [
            (s.gate_scale_ptrs, s.gate_scales, gu_rows * gu_cols),
            (s.up_scale_ptrs, s.up_scales, gu_rows * gu_cols),
            (s.down_scale_ptrs, s.down_scales, d_rows * d_cols),
        ] {
            let ptrs: Vec<u8> = (0..e)
                .flat_map(|i| area.offset(i * slot).0.to_le_bytes())
                .collect();
            gpu.copy_h2d(&ptrs, table)?;
        }
        Ok(s)
    }

    /// Frees the allocation (model unload).
    #[allow(dead_code)]
    pub(super) fn free(&self, gpu: &dyn GpuBackend) {
        gpu.free(self.base).ok();
    }
}

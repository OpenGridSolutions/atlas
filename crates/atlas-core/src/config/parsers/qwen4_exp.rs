// SPDX-License-Identifier: AGPL-3.0-only

//! Parser for the `qwen4_exp` family (Qwen3.8-Flash-Next).
//!
//! The text tower sits under `text_config`. On top of the Qwen3-Next
//! building blocks (GDN layers, gated attention, MoE) it has several
//! residual streams with a learned mix (`hc_count`, `hc_lowrank`), sparse
//! attention with a block indexer (`indexer_*`) and a hashed n-gram
//! embedding table injected at `ple_layer_ids` (`ngram_*`, `ple_*`). Those
//! parameters live in [`Qwen4ExpConfig`]; all of them are required.

use anyhow::{Context, Result, bail};

use super::super::{ModelConfig, default_partial_rotary, default_rope_theta, finalize_config};

/// Model-family parameters of `qwen4_exp` that no other family has.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen4ExpConfig {
    /// Rank of the bottleneck in the stream mixer (`hc_lowrank`).
    pub hc_lowrank: usize,
    /// GDN output gate: `true` = sigmoid, `false` = silu (`output_gate_type`).
    pub gdn_output_gate_sigmoid: bool,
    /// Tokens a query may attend through the indexer (`indexer_budget`).
    pub indexer_budget: usize,
    /// Tokens per pooled index key (`indexer_compress_ratio`).
    pub indexer_compress_ratio: usize,
    /// Dimension of an index key / query head (`indexer_head_dim`).
    pub indexer_head_dim: usize,
    /// Index query heads (`indexer_n_heads`).
    pub indexer_n_heads: usize,
    /// Index key heads (`indexer_kv_heads`).
    pub indexer_kv_heads: usize,
    /// Longest n-gram hashed (`ngram_size`); the context is `ngram_size - 1` tokens.
    pub ngram_size: usize,
    /// Hash heads per n-gram order (`heads_per_ngram`).
    pub heads_per_ngram: usize,
    /// Lower bound of the per-head table sizes (`ngram_vocab_size_base`).
    pub ngram_vocab_size_base: usize,
    /// Number of tensors the table is split into (`split_ngram_parts`).
    pub split_ngram_parts: usize,
    /// Row-count granularity of the table (`make_ngram_vocab_size_divisible_by`).
    pub ngram_vocab_divisor: usize,
    /// Zero-based indices of the layers with an n-gram injection
    /// (`ple_layer_ids` is one-based).
    pub ple_layers: Vec<usize>,
    /// Width of the concatenated n-gram embedding (`ple_embed_dim`).
    pub ple_embed_dim: usize,
    /// Kernel size of the n-gram layer's convolution (`ple_conv_kernel_size`).
    pub ple_conv_kernel_size: usize,
}

impl Qwen4ExpConfig {
    /// Hash heads per token: one group of `heads_per_ngram` per n-gram order 2..=ngram_size.
    pub fn ngram_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// Width of one table row.
    pub fn ngram_head_dim(&self) -> usize {
        self.ple_embed_dim / self.ngram_heads()
    }

    /// Dilation of the n-gram layer's convolution (the reference uses `ngram_size`).
    pub fn ple_conv_dilation(&self) -> usize {
        self.ngram_size
    }

    /// Per-channel history the convolution keeps between calls.
    pub fn ple_conv_state_len(&self) -> usize {
        (self.ple_conv_kernel_size - 1) * self.ple_conv_dilation()
    }

    /// Bytes of the n-gram layer's convolution history for one sequence:
    /// `[channels, state_len]` FP32, `channels` = streams x hidden size.
    pub fn ple_conv_state_bytes(&self, channels: usize) -> usize {
        channels * self.ple_conv_state_len() * 4
    }

    /// Bytes of index data per KV block of `kv_block_size` tokens and
    /// attention layer: the raw key of every token followed by the index
    /// key of every `indexer_compress_ratio` tokens, BF16.
    pub fn index_block_bytes(&self, kv_block_size: usize) -> usize {
        (kv_block_size + kv_block_size / self.indexer_compress_ratio) * self.indexer_head_dim * 2
    }

    /// Blocks a query selects through the indexer.
    pub fn indexer_top_blocks(&self) -> usize {
        self.indexer_budget / self.indexer_compress_ratio
    }
}

fn required_usize(text: &serde_json::Value, key: &str) -> Result<usize> {
    text.get(key)
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize)
        .with_context(|| format!("qwen4_exp text_config: missing or non-integer `{key}`"))
}

pub fn parse_qwen4_exp(raw: &serde_json::Value) -> Result<ModelConfig> {
    let text = raw
        .get("text_config")
        .context("qwen4_exp config missing text_config")?;
    let mut config: ModelConfig =
        serde_json::from_value(text.clone()).context("Failed to parse qwen4_exp text_config")?;
    config.model_type = "qwen4_exp".to_string();
    config.nested_config = true;
    config.attn_gated = true;

    if let Some(rope) = text.get("rope_parameters") {
        if config.rope_theta == default_rope_theta()
            && let Some(theta) = rope.get("rope_theta").and_then(serde_json::Value::as_f64)
        {
            config.rope_theta = theta;
        }
        if config.partial_rotary_factor == default_partial_rotary()
            && let Some(prf) = rope
                .get("partial_rotary_factor")
                .and_then(serde_json::Value::as_f64)
        {
            config.partial_rotary_factor = prf;
        }
        if let Some(ms) = rope.get("mrope_section").and_then(|v| v.as_array())
            && ms.len() == 3
        {
            config.mrope_section = [
                ms[0].as_u64().unwrap_or(0) as usize,
                ms[1].as_u64().unwrap_or(0) as usize,
                ms[2].as_u64().unwrap_or(0) as usize,
            ];
        }
        config.mrope_interleaved = rope
            .get("mrope_interleaved")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
    }

    // The reference configuration defaults `norm_topk_prob` to true and the
    // released checkpoints omit the key.
    config.norm_topk_prob = text
        .get("norm_topk_prob")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);

    // The stream buffers are sized by `hc_mult`.
    config.hc_mult = required_usize(text, "hc_count")?;

    let gdn_output_gate_sigmoid = match text.get("output_gate_type").and_then(|v| v.as_str()) {
        Some("sigmoid") => true,
        Some("silu") => false,
        other => bail!("qwen4_exp text_config: unsupported output_gate_type {other:?}"),
    };
    let ple_layers = text
        .get("ple_layer_ids")
        .and_then(|v| v.as_array())
        .context("qwen4_exp text_config: missing `ple_layer_ids`")?
        .iter()
        .map(|v| match v.as_u64() {
            Some(id) if id >= 1 && (id as usize) <= config.num_hidden_layers => Ok(id as usize - 1),
            _ => bail!("qwen4_exp text_config: ple_layer_ids entry {v} is not a layer number"),
        })
        .collect::<Result<Vec<_>>>()?;
    let q = Qwen4ExpConfig {
        hc_lowrank: required_usize(text, "hc_lowrank")?,
        gdn_output_gate_sigmoid,
        indexer_budget: required_usize(text, "indexer_budget")?,
        indexer_compress_ratio: required_usize(text, "indexer_compress_ratio")?,
        indexer_head_dim: required_usize(text, "indexer_head_dim")?,
        indexer_n_heads: required_usize(text, "indexer_n_heads")?,
        indexer_kv_heads: required_usize(text, "indexer_kv_heads")?,
        ngram_size: required_usize(text, "ngram_size")?,
        heads_per_ngram: required_usize(text, "heads_per_ngram")?,
        ngram_vocab_size_base: required_usize(text, "ngram_vocab_size_base")?,
        split_ngram_parts: required_usize(text, "split_ngram_parts")?,
        ngram_vocab_divisor: required_usize(text, "make_ngram_vocab_size_divisible_by")?,
        ple_layers,
        ple_embed_dim: required_usize(text, "ple_embed_dim")?,
        ple_conv_kernel_size: required_usize(text, "ple_conv_kernel_size")?,
    };
    if config.hc_mult == 0 || q.hc_lowrank == 0 {
        bail!("qwen4_exp text_config: hc_count and hc_lowrank must be positive");
    }
    if q.ngram_size < 2 || q.heads_per_ngram == 0 || q.ple_conv_kernel_size < 2 {
        bail!(
            "qwen4_exp text_config: ngram_size/heads_per_ngram/ple_conv_kernel_size out of range"
        );
    }
    if !q.ple_embed_dim.is_multiple_of(q.ngram_heads()) || !q.ngram_head_dim().is_multiple_of(16) {
        bail!(
            "qwen4_exp text_config: ple_embed_dim {} does not split into {} heads of a multiple of 16",
            q.ple_embed_dim,
            q.ngram_heads()
        );
    }
    if q.indexer_compress_ratio == 0
        || !q.indexer_budget.is_multiple_of(q.indexer_compress_ratio)
        || q.indexer_n_heads == 0
        || q.indexer_kv_heads != 1
        || q.indexer_head_dim == 0
    {
        bail!("qwen4_exp text_config: unsupported indexer geometry {q:?}");
    }
    config.qwen4_exp = Some(q);

    finalize_config(&mut config, raw)?;
    Ok(config)
}

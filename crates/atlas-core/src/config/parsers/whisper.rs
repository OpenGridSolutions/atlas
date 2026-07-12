// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `config.rs` for file-size budget. Parser for the
//! Whisper encoder-decoder family (multiclass Plan.md Phase W).
//!
//! Whisper's config.json has NO top-level `hidden_size` — the
//! encoder-decoder schema splits dims (`d_model`, `encoder_layers`,
//! `decoder_layers`, `decoder_attention_heads`, `decoder_ffn_dim`, …),
//! so the generic flat fallback arm died with "missing field
//! hidden_size" and the preseed servability gate mis-reported a parse
//! error instead of the real story (sidecar-served model).
//!
//! This arm maps the schema into `ModelConfig`'s DECODER-side view:
//! hidden_size = d_model, layers = decoder_layers (all full attention).
//! That is enough for everything ModelConfig is consulted for here —
//! kernel-target selection (`ptx_for_config("whisper", 1280)`), quant
//! canonicalization, and the preseed gates. The Whisper SIDECAR itself
//! (spark-model/src/whisper/) reads the HF checkpoint directly and does
//! not consume ModelConfig; encoder-side dims therefore stay out of
//! this struct deliberately.

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;

use super::super::{LayerType, ModelConfig, finalize_config};

/// The encoder-decoder field set, schema-specifically deserialized —
/// `hidden_size` stays REQUIRED for every decoder-only parse.
#[derive(Deserialize)]
struct WhisperRaw {
    d_model: usize,
    decoder_layers: usize,
    decoder_attention_heads: usize,
    #[serde(default)]
    decoder_ffn_dim: usize,
    #[serde(default)]
    vocab_size: usize,
    #[serde(default)]
    max_target_positions: usize,
    #[serde(default)]
    eos_token_id: u64,
}

pub(crate) fn parse_whisper(json: &str, raw: &Value) -> Result<ModelConfig> {
    let w: WhisperRaw =
        serde_json::from_str(json).context("Failed to parse whisper config.json")?;
    // Start from a default-heavy parse of the SAME json so shared
    // optional fields (rope defaults, eps, quantization_config via
    // finalize) behave like everywhere else — but with hidden_size
    // injected from d_model instead of being serde-required.
    let mut config: ModelConfig = serde_json::from_str(
        &{
            let mut v: Value = raw.clone();
            v["hidden_size"] = Value::from(w.d_model);
            v.to_string()
        },
    )
    .context("Failed to lower whisper config into ModelConfig")?;
    config.model_type = "whisper".to_string();
    config.hidden_size = w.d_model;
    config.num_hidden_layers = w.decoder_layers;
    config.num_attention_heads = w.decoder_attention_heads;
    config.num_key_value_heads = w.decoder_attention_heads; // MHA, no GQA
    config.head_dim = w.d_model / w.decoder_attention_heads.max(1);
    config.intermediate_size = w.decoder_ffn_dim;
    config.vocab_size = w.vocab_size;
    config.max_position_embeddings = w.max_target_positions;
    if config.eos_token_id == 0 {
        config.eos_token_id = w.eos_token_id as u32;
    }
    config.attn_gated = false;
    config.layer_types = vec![LayerType::FullAttention; w.decoder_layers];
    finalize_config(&mut config, raw)?;
    Ok(config)
}

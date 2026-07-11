// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `config.rs` for file-size budget. Parsers for the dense
//! Qwen3 family (multiclass Plan.md step 1.1, config half):
//!
//! - `qwen3` (Qwen3-Embedding-8B / Qwen3-Reranker-8B): flat config, all
//!   core fields top-level — but standard UNGATED attention (the generic
//!   flat fallback arm sets `attn_gated = true`, which is wrong here).
//! - `qwen3_vl` (Qwen3-VL-Embedding-8B / Qwen3-VL-Reranker-8B): nested —
//!   `hidden_size` & friends live under `text_config`, the ViT under a
//!   top-level `vision_config`. The flat fallback arm dies with
//!   "missing field hidden_size at line 65" (= EOF of the 65-line
//!   config.json) because there IS no top-level hidden_size. MRoPE params
//!   sit under `text_config.rope_scaling` (NOT `rope_parameters` as in
//!   the Qwen3.5/3.6 family).

use anyhow::{Context, Result};
use serde_json::Value;

use super::super::{LayerType, ModelConfig, finalize_config};
use super::vision::parse_vision_config;

/// Fill `layer_types` for a pure-attention dense model: the config has no
/// layer pattern field — every layer is full attention.
fn fill_dense_layer_types(config: &mut ModelConfig) {
    if config.layer_types.is_empty() {
        config.layer_types = vec![LayerType::FullAttention; config.num_hidden_layers];
    }
}

/// `model_type = "qwen3"` — standard dense attention-only Qwen3
/// (Embedding-8B / Reranker-8B geometry: hidden 4096, 36 L, GQA 32/8,
/// head_dim 128). Flat layout, so serde does the heavy lifting; the two
/// corrections over the generic flat arm are `attn_gated = false` (Qwen3
/// dense has q/k norms but NO attention output gate — only Qwen3-Next
/// gates) and an explicit all-FullAttention layer_types fill.
pub(crate) fn parse_qwen3_dense(json: &str, raw: &Value) -> Result<ModelConfig> {
    let mut config: ModelConfig =
        serde_json::from_str(json).context("Failed to parse qwen3 (dense) config.json")?;
    config.attn_gated = false;
    fill_dense_layer_types(&mut config);
    finalize_config(&mut config, raw)?;
    Ok(config)
}

/// `model_type = "qwen3_vl"` — dense Qwen3-VL (VL-Embedding-8B /
/// VL-Reranker-8B): the same dense 8B text tower nested under
/// `text_config` (its own model_type is "qwen3_vl_text"), plus a
/// top-level `vision_config` (27-block ViT, deepstack [8,16,24]).
///
/// Quirk worth knowing: VL-Reranker-8B ships a stray TOP-LEVEL
/// `hidden_size` alongside the nested one, so the generic flat arm would
/// "successfully" parse it into a garbage config (0 layers, 0 heads, no
/// vision) instead of erroring — this arm reads `text_config`
/// authoritatively for both checkpoints.
pub(crate) fn parse_qwen3_vl_dense(raw: &Value) -> Result<ModelConfig> {
    let text_config = raw
        .get("text_config")
        .context("qwen3_vl config missing text_config")?;
    let mut config: ModelConfig = serde_json::from_value(text_config.clone())
        .context("Failed to parse qwen3_vl text_config")?;
    // text_config declares "qwen3_vl_text"; kernel-target and loader
    // selection key on the top-level type.
    config.model_type = "qwen3_vl".to_string();
    config.attn_gated = false;
    config.nested_config = true;
    if config.eos_token_id == 0 {
        config.eos_token_id = text_config
            .get("eos_token_id")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
    }
    if config.vocab_size == 0 {
        config.vocab_size = raw
            .get("vocab_size")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
    }
    // MRoPE lives under text_config.rope_scaling in this family
    // (rope_type "default" + mrope_interleaved + mrope_section) — the
    // Qwen3.5/3.6 arm reads `rope_parameters`, which does not exist here.
    if let Some(rope) = text_config.get("rope_scaling") {
        if let Some(ms) = rope.get("mrope_section").and_then(Value::as_array)
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
            .and_then(Value::as_bool)
            .unwrap_or(false);
    }
    config.vision = parse_vision_config(raw);
    fill_dense_layer_types(&mut config);
    finalize_config(&mut config, raw)?;
    Ok(config)
}

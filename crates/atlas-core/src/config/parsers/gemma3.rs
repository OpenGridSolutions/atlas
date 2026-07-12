// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `config.rs` for file-size budget. Parser for a model family.
//!
//! Gemma-3 (e.g. google/translategemma-12b-it, Plan.md Phase T): nested
//! `text_config` with NO top-level hidden_size — the flat fallback arm
//! dies with "missing field hidden_size" exactly like qwen3_vl/whisper
//! did. Differs from the gemma4 arm in the RoPE schema: FLAT
//! `rope_theta` (full-attention, 1M) + `rope_local_base_freq` (sliding,
//! 10k) + `rope_parameters.full_attention.factor` (linear scaling 8.0),
//! instead of gemma4's nested per-kind rope_parameters thetas. Layers
//! are homogeneous in shape (uniform head_dim/GQA); the 5-sliding :
//! 1-full pattern comes from `text_config.layer_types` and is carried
//! as `full_attention_interval` (layer_types itself stays all
//! FullAttention so KV-pool sizing covers every layer — sliding is a
//! per-layer runtime override, the gemma4 convention).

#![allow(unused_imports)]

use anyhow::{Context, Result, bail};
use serde_json::Value;

use super::super::{
    LayerType, ModelConfig, QuantizationConfig, VisionConfig, default_conv_kernel, default_one,
    default_one_f64, default_partial_rotary, default_rms_eps, default_rope_theta, finalize_config,
    parse_quantization_config, parse_vision_config, validate_config,
};

pub(crate) fn parse_gemma3_params(raw: &serde_json::Value) -> Result<ModelConfig> {
    let tc = raw
        .get("text_config")
        .context("gemma3 config missing text_config")?;

    let get_usize = |key: &str| -> usize {
        tc.get(key).and_then(serde_json::Value::as_u64).unwrap_or(0) as usize
    };
    let get_f64 = |key: &str, default: f64| -> f64 {
        tc.get(key)
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(default)
    };

    let hidden_size = get_usize("hidden_size");
    let num_hidden_layers = get_usize("num_hidden_layers");

    // Sliding/full pattern: strictly periodic in the Gemma-3 family
    // (every 6th layer full). Derive + verify the period from
    // layer_types so a future aperiodic variant fails loudly instead of
    // silently mis-roping.
    let mut full_attention_interval = 0usize;
    if let Some(lt) = tc.get("layer_types").and_then(Value::as_array) {
        if lt.len() != num_hidden_layers {
            bail!(
                "gemma3 layer_types length {} != num_hidden_layers {}",
                lt.len(),
                num_hidden_layers
            );
        }
        let full_idx: Vec<usize> = lt
            .iter()
            .enumerate()
            .filter_map(|(i, v)| (v.as_str() == Some("full_attention")).then_some(i))
            .collect();
        let first = *full_idx
            .first()
            .context("gemma3 layer_types has no full_attention layer")?;
        full_attention_interval = first + 1;
        for (n, idx) in full_idx.iter().enumerate() {
            if *idx != first + n * full_attention_interval {
                bail!(
                    "gemma3 layer_types is not periodic (full layers at {full_idx:?}) — \
                     the loader's every-Nth rule cannot represent it"
                );
            }
        }
        // Sliding layers must fill the gaps.
        let expected_full = num_hidden_layers / full_attention_interval;
        if full_idx.len() != expected_full {
            bail!(
                "gemma3 layer_types full-attention count {} != layers/interval {}",
                full_idx.len(),
                expected_full
            );
        }
    }
    if full_attention_interval == 0 {
        // Fallback: the family constant (Gemma-3 4B/12B/27B all use 6).
        full_attention_interval = 6;
    }

    // Dual-theta RoPE (flat keys) + linear scaling factor on full layers.
    let rope_theta_full = get_f64("rope_theta", 1_000_000.0);
    let rope_local = get_f64("rope_local_base_freq", 10_000.0);
    let rope_linear_factor = tc
        .get("rope_parameters")
        .and_then(|r| r.get("full_attention"))
        .filter(|fa| {
            fa.get("rope_type").and_then(Value::as_str) == Some("linear")
        })
        .and_then(|fa| fa.get("factor"))
        .and_then(Value::as_f64)
        .or_else(|| {
            // Older transformers serialize the same thing as rope_scaling.
            tc.get("rope_scaling")
                .filter(|rs| rs.get("rope_type").and_then(Value::as_str) == Some("linear"))
                .and_then(|rs| rs.get("factor"))
                .and_then(Value::as_f64)
        })
        .unwrap_or(1.0);

    // Build config from a clean template, resetting all SSM/MoE fields
    // (the gemma4 arm's convention).
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.linear_num_key_heads = 0;
    config.linear_key_head_dim = 0;
    config.linear_num_value_heads = 0;
    config.linear_value_head_dim = 0;
    config.linear_conv_kernel_dim = 0;
    config.num_experts = 0;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = 0;
    config.shared_expert_intermediate_size = 0;
    config.mtp_num_hidden_layers = 0;

    config.hidden_size = hidden_size;
    config.num_hidden_layers = num_hidden_layers;
    config.intermediate_size = get_usize("intermediate_size");
    config.vocab_size = get_usize("vocab_size");
    config.num_attention_heads = get_usize("num_attention_heads");
    config.num_key_value_heads = get_usize("num_key_value_heads");
    config.head_dim = get_usize("head_dim");
    config.partial_rotary_factor = 1.0; // full rotation on every layer
    // All layers FullAttention for KV-pool sizing; the sliding pattern is
    // a per-layer runtime override derived from full_attention_interval.
    config.layer_types = vec![LayerType::FullAttention; num_hidden_layers];
    config.full_attention_interval = full_attention_interval;
    config.sliding_window = tc
        .get("sliding_window")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as u32;
    config.max_position_embeddings = get_usize("max_position_embeddings");
    config.rope_theta = rope_theta_full;
    config.rope_local_base_freq = rope_local;
    config.rope_linear_factor = rope_linear_factor;
    config.rms_norm_eps = get_f64("rms_norm_eps", 1e-6);
    config.tie_word_embeddings = raw
        .get("tie_word_embeddings")
        .or_else(|| tc.get("tie_word_embeddings"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true); // Gemma family ties embeddings by default
    config.model_type = "gemma3".to_string();
    config.attn_gated = false;
    config.nested_config = true;
    config.norm_topk_prob = false;

    // Gemma convention: embeddings *= sqrt(hidden_size).
    config.embed_scale = (hidden_size as f32).sqrt();
    // Gemma-3 has NO logit softcapping (that was Gemma-2/-4).
    config.final_logit_softcapping = tc
        .get("final_logit_softcapping")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0) as f32;

    // TranslateGemma's config.json carries NO bos/eos at all (they live
    // in generation_config.json: bos 2, eos [1, 106]) — fall back to the
    // Gemma3TextConfig defaults. <end_of_turn>=106 is enforced as an
    // additional stop token at the /v1/translate endpoint.
    config.eos_token_id = tc
        .get("eos_token_id")
        .or_else(|| raw.get("eos_token_id"))
        .and_then(|v| {
            v.as_u64()
                .or_else(|| v.as_array().and_then(|a| a.first().and_then(Value::as_u64)))
        })
        .unwrap_or(1) as u32;
    config.bos_token_id = tc
        .get("bos_token_id")
        .or_else(|| raw.get("bos_token_id"))
        .and_then(Value::as_u64)
        .unwrap_or(2) as u32;

    finalize_config(&mut config, raw)?;
    Ok(config)
}

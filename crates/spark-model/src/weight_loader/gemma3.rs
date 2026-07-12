// SPDX-License-Identifier: AGPL-3.0-only

//! Weight loader for Gemma-3 TEXT backbones (`model_type = "gemma3"`) —
//! google/translategemma-12b-it (Plan.md Phase T). Geometry: hidden 3840,
//! 48 layers, GQA 16/8, head_dim 256 uniform, GeGLU MLP, 4 norms per
//! layer, tied lm_head. The checkpoint is multimodal on disk
//! (`vision_tower.*`, `multi_modal_projector.*`) but this loader reads
//! ONLY the `language_model.model.*` text tower — text→text translation
//! is the whole scope; vision tensors are never referenced.
//!
//! Precision policy: stock BF16 checkpoint, served BF16 end to end (the
//! qwen3_dense pattern) — BF16 attention via `set_o_dense_bf16` +
//! BF16 MLP via `set_bf16_weights`; NVFP4 exists only as the required
//! shadowed placeholders. No NVFP4 runtime quantization on the compute
//! path.
//!
//! Divergences from the Gemma-4 loader (why this is a separate file):
//! - Layers are SHAPE-HOMOGENEOUS; the 5-sliding : 1-full pattern comes
//!   from `config.full_attention_interval` (parsed from layer_types),
//!   not from per-layer weight shapes.
//! - Dual-theta RoPE with FULL rotation on every layer: sliding layers
//!   θ = `rope_local_base_freq` (10k), full layers θ = `rope_theta` (1M)
//!   PLUS linear position scaling 1/`rope_linear_factor` (Gemma-3's
//!   32K→128K context extension bakes factor 8 into the weights).
//! - Attention scale stays the default 1/√head_dim
//!   (query_pre_attn_scalar == head_dim); gemma4's 1.0 override and its
//!   v_norm/K=V machinery do not exist in Gemma-3.
//! - Norm weights are zero-centered (`out = x_rms * (1 + w)`) — exactly
//!   the common `rms_norm` kernel's convention, so ALL norms load raw
//!   (no `dense_minus_one`, no per-target kernel override).

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::{DenseFfnLayer, FfnActivation, FfnComponent, Qwen3AttentionLayer};
use crate::weight_map::{
    AttentionWeights, DenseWeight, MtpWeights, QuantizeCtx, dense, dense_auto,
    detect_nvfp4_variant, load_kv_scales, quantize_to_nvfp4, quantized_any,
};

pub struct Gemma3WeightLoader;

/// Text-tower prefix. `auto_detect_weight_prefix` sets
/// `config.weight_prefix = "language_model.model"` for this layout; the
/// store fallback covers a text-only re-export (`model.*`).
fn text_prefix(store: &WeightStore, config: &ModelConfig) -> String {
    if !config.weight_prefix.is_empty() {
        return format!("{}.", config.weight_prefix);
    }
    if store.contains("language_model.model.embed_tokens.weight") {
        "language_model.model.".to_string()
    } else {
        "model.".to_string()
    }
}

impl super::ModelWeightLoader for Gemma3WeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);

        let variant = detect_nvfp4_variant(store, config);
        tracing::info!("Gemma-3 NVFP4 variant: {variant:?} (expected Bf16Raw)");
        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();
        let qctx = QuantizeCtx {
            absmax_k,
            quantize_k,
            stream,
        };
        let h = config.hidden_size;
        let q_out = config.num_attention_heads * config.head_dim;
        let interval = config.full_attention_interval.max(1);
        let theta_full = config.rope_theta as f32;
        let theta_sliding = if config.rope_local_base_freq > 0.0 {
            config.rope_local_base_freq as f32
        } else {
            theta_full
        };
        let pos_scale = if config.rope_linear_factor > 1.0 {
            Some(1.0 / config.rope_linear_factor as f32)
        } else {
            None
        };

        for i in 0..config.num_hidden_layers {
            let lp = config.layer_prefix(i);
            let p = format!("{lp}.self_attn");

            // 4 norms, all zero-centered → raw. Constructor slots follow
            // the gemma4 mapping: input_layernorm → input_norm,
            // pre_feedforward → post_attn_norm (pre-FFN); the two
            // post-sublayer OUTPUT norms install via
            // set_post_sublayer_norms.
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let pre_ffn_norm = dense(store, &format!("{lp}.pre_feedforward_layernorm.weight"))?;
            let post_attn_out_norm =
                dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;
            let post_ffn_out_norm =
                dense(store, &format!("{lp}.post_feedforward_layernorm.weight"))?;

            // Attention projections: BF16 zero-copy aliases; NVFP4 o_proj
            // placeholder shadowed by set_o_dense_bf16 below.
            let q_dense = dense_auto(store, &format!("{p}.q_proj.weight"), gpu)?;
            let k_dense = dense_auto(store, &format!("{p}.k_proj.weight"), gpu)?;
            let v_dense = dense_auto(store, &format!("{p}.v_proj.weight"), gpu)?;
            let o_dense = dense_auto(store, &format!("{p}.o_proj.weight"), gpu)?;
            let o_nvfp4 = quantize_to_nvfp4(&o_dense, h, q_out, gpu, absmax_k, quantize_k, stream)?;
            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
            let attn = AttentionWeights {
                q_proj: q_dense,
                k_proj: k_dense,
                v_proj: v_dense,
                o_proj: o_nvfp4,
                q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
                k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
                q_norm_full: None,
                k_norm_full: None,
                k_scale,
                v_scale,
            };

            // GeGLU MLP: required NVFP4 placeholders + preferred BF16.
            let gate_proj = quantized_any(
                store,
                &format!("{lp}.mlp.gate_proj"),
                config.intermediate_size,
                h,
                gpu,
                variant,
                qctx,
            )?;
            let up_proj = quantized_any(
                store,
                &format!("{lp}.mlp.up_proj"),
                config.intermediate_size,
                h,
                gpu,
                variant,
                qctx,
            )?;
            let down_proj = quantized_any(
                store,
                &format!("{lp}.mlp.down_proj"),
                h,
                config.intermediate_size,
                gpu,
                variant,
                qctx,
            )?;
            let ffn_weights = DenseFfnWeights {
                gate_proj,
                up_proj,
                down_proj,
                gate_proj_t: None,
                up_proj_t: None,
                down_proj_t: None,
            };
            let mut ffn_layer =
                DenseFfnLayer::new_with_activation(ffn_weights, FfnActivation::GeLU, gpu)?;
            ffn_layer.set_bf16_weights(
                dense_auto(store, &format!("{lp}.mlp.gate_proj.weight"), gpu)?,
                dense_auto(store, &format!("{lp}.mlp.up_proj.weight"), gpu)?,
                dense_auto(store, &format!("{lp}.mlp.down_proj.weight"), gpu)?,
            );
            let ffn = FfnComponent::Dense(ffn_layer);

            let mut layer = Qwen3AttentionLayer::new_ungated(
                input_norm,
                attn,
                pre_ffn_norm,
                ffn,
                i,
                None, // q/k/v stay BF16
                None,
                None,
                gpu,
                layer_kv_dtypes[i],
                config.fp8_kv_calibration_tokens,
                config,
            )?;
            layer.set_o_dense_bf16(o_dense);
            layer.set_post_sublayer_norms(post_attn_out_norm, post_ffn_out_norm);

            // 5-sliding : 1-full pattern (periodic; parser verified).
            let is_full_attn = (i + 1) % interval == 0;
            if is_full_attn {
                layer.set_rope_overrides(theta_full, config.head_dim as u32);
                if let Some(scale) = pos_scale {
                    layer.set_rope_pos_scale(scale);
                }
                layer.set_sliding_window(None);
                if i < interval * 2 {
                    tracing::info!(
                        "L{i}: full attn — θ={theta_full}, pos_scale={pos_scale:?}"
                    );
                }
            } else {
                layer.set_rope_overrides(theta_sliding, config.head_dim as u32);
                if config.sliding_window > 0 {
                    layer.set_sliding_window(Some(config.sliding_window));
                }
                if i < 2 {
                    tracing::info!(
                        "L{i}: sliding attn — θ={theta_sliding}, window={}",
                        config.sliding_window
                    );
                }
            }

            layers.push(Box::new(layer));
            if (i + 1).is_multiple_of(12) || i + 1 == config.num_hidden_layers {
                tracing::info!("Loaded layers 0..{}", i + 1);
            }
        }
        gpu.synchronize(stream)?;
        tracing::info!(
            "Gemma-3 weight loader: {} layers ({} full / {} sliding, BF16 attn+MLP, GeGLU)",
            layers.len(),
            config.num_hidden_layers / interval,
            config.num_hidden_layers - config.num_hidden_layers / interval,
        );
        Ok(layers)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        // embed_scale (√hidden) is applied by the embedding forward from
        // config.embed_scale, gemma4 convention.
        dense(store, &format!("{}embed_tokens.weight", text_prefix(store, config)))
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        // Zero-centered like every other Gemma-3 norm → raw.
        dense(store, &format!("{}norm.weight", text_prefix(store, config)))
    }

    fn load_lm_head(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        // Tied embeddings: TranslateGemma ships no lm_head tensor.
        if store.contains("lm_head.weight") {
            return dense(store, "lm_head.weight");
        }
        dense(store, &format!("{}embed_tokens.weight", text_prefix(store, config)))
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }
}

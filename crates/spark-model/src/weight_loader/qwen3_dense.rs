// SPDX-License-Identifier: AGPL-3.0-only

//! Weight loader for standard DENSE attention-only Qwen3 checkpoints
//! (`model_type = "qwen3"`) — Qwen3-Embedding-8B / Qwen3-Reranker-8B
//! (multiclass Plan.md step 1.1). Geometry: hidden 4096, 36 layers,
//! GQA 32/8, head_dim 128, per-head q/k norms, UNGATED attention,
//! SwiGLU dense MLP. Stock BF16 on disk (no quantization_config →
//! `Nvfp4Variant::Bf16Raw`).
//!
//! Precision policy (Plan.md D2 — bf16 for embedding/rerank quality,
//! mirroring the Gemma-4-31B production path):
//! - attention q/k/v/o stay BF16 (`dense_auto` zero-copy aliases; NVFP4
//!   q/k/v passed as `None`, o as a quantized placeholder shadowed by
//!   `set_o_dense_bf16`);
//! - MLP gets BF16 weights installed via `set_bf16_weights` (preferred
//!   by the DenseFfnLayer dispatch, prefill included) over the required
//!   runtime-quantized NVFP4 fallback;
//! - norms/embedding/lm_head stay BF16.
//!
//! Two checkpoint quirks handled here:
//! - **Weight-name prefix**: Reranker-8B uses standard `model.*` names;
//!   Embedding-8B is a sentence-transformers export with NO prefix
//!   (`embed_tokens.weight`, `layers.N.*`). Resolved per store.
//! - **RMSNorm convention**: plain Qwen3 stores STANDARD norm weights
//!   (`out = x_rms * w`), but the common `rms_norm` kernel computes
//!   `x_rms * (1 + w)` (Qwen3-Next zero-centered convention). All norm
//!   weights (input/post/q/k/final) load through `dense_minus_one`
//!   (the DeepSeek-V4 pattern); Gemma instead overrides the kernel
//!   per-target, which a zero-override target deliberately avoids.
//! - **lm_head**: Reranker ships `lm_head.weight` (needed — its score
//!   IS a yes/no logit pair). Embedding ships NONE (untied): fall back
//!   to embed_tokens with a WARN — the pointer keeps build_model's
//!   invariants intact, and the embedding class consumes pooled hidden
//!   states, never logits.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::{DenseFfnLayer, FfnComponent, Qwen3AttentionLayer};
use crate::weight_map::{
    AttentionWeights, DenseWeight, MtpWeights, QuantizeCtx, dense, dense_auto, dense_minus_one,
    detect_nvfp4_variant, load_kv_scales, quantize_to_nvfp4, quantized_any,
};

pub struct Qwen3DenseWeightLoader;

/// Weight-name prefix for this checkpoint: `"model.language_model."`
/// (Qwen3-VL nested layout), `"model."` (standard HF, Reranker-8B) or
/// `""` (sentence-transformers export, Embedding-8B).
fn store_prefix(store: &WeightStore) -> &'static str {
    if store.contains("model.language_model.embed_tokens.weight") {
        "model.language_model."
    } else if store.contains("model.embed_tokens.weight") {
        "model."
    } else {
        ""
    }
}

impl super::ModelWeightLoader for Qwen3DenseWeightLoader {
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
        tracing::info!("Qwen3 dense NVFP4 variant: {variant:?} (expected Bf16Raw)");
        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();
        let qctx = QuantizeCtx {
            absmax_k,
            quantize_k,
            stream,
        };
        let h = config.hidden_size;
        // Homogeneous geometry: attention dims come straight from config
        // (q_out = 32×128 = 4096); only o_proj's shape is needed here.
        let q_out = config.num_attention_heads * config.head_dim;
        let pfx = store_prefix(store);

        for i in 0..config.num_hidden_layers {
            let lp = format!("{pfx}layers.{i}");
            let p = format!("{lp}.self_attn");

            // Norms: standard Qwen3 convention → shift to the kernel's
            // offset-from-1 form (module doc).
            let input_norm = dense_minus_one(store, &format!("{lp}.input_layernorm.weight"), gpu)?;
            let post_attn_norm =
                dense_minus_one(store, &format!("{lp}.post_attention_layernorm.weight"), gpu)?;

            // Attention projections: BF16 zero-copy aliases.
            let q_dense = dense_auto(store, &format!("{p}.q_proj.weight"), gpu)?;
            let k_dense = dense_auto(store, &format!("{p}.k_proj.weight"), gpu)?;
            let v_dense = dense_auto(store, &format!("{p}.v_proj.weight"), gpu)?;
            let o_dense = dense_auto(store, &format!("{p}.o_proj.weight"), gpu)?;
            // AttentionWeights requires an NVFP4 o_proj; the BF16 dense
            // pointer installed below via set_o_dense_bf16 shadows it in
            // the decode/prefill dispatch (Gemma-4 pattern).
            let o_nvfp4 = quantize_to_nvfp4(&o_dense, h, q_out, gpu, absmax_k, quantize_k, stream)?;
            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
            let attn = AttentionWeights {
                q_proj: q_dense,
                k_proj: k_dense,
                v_proj: v_dense,
                o_proj: o_nvfp4,
                q_norm: dense_minus_one(store, &format!("{p}.q_norm.weight"), gpu)?,
                k_norm: dense_minus_one(store, &format!("{p}.k_norm.weight"), gpu)?,
                q_norm_full: None,
                k_norm_full: None,
                k_scale,
                v_scale,
            };

            // MLP: required NVFP4 weights (runtime-quantized from the
            // BF16 tensors under Bf16Raw), then BF16 aliases installed
            // as the preferred path.
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
                // BF16 prefill path — no transposed NVFP4 copies needed.
                gate_proj_t: None,
                up_proj_t: None,
                down_proj_t: None,
            };
            // SwiGLU = default SiLU activation.
            let mut ffn_layer = DenseFfnLayer::new(ffn_weights, gpu)?;
            ffn_layer.set_bf16_weights(
                dense_auto(store, &format!("{lp}.mlp.gate_proj.weight"), gpu)?,
                dense_auto(store, &format!("{lp}.mlp.up_proj.weight"), gpu)?,
                dense_auto(store, &format!("{lp}.mlp.down_proj.weight"), gpu)?,
            );
            let ffn = FfnComponent::Dense(ffn_layer);

            let mut layer = Qwen3AttentionLayer::new_ungated(
                input_norm,
                attn,
                post_attn_norm,
                ffn,
                i,
                None, // q/k/v stay BF16 (dense_gemv/gemm path)
                None,
                None,
                gpu,
                layer_kv_dtypes[i],
                config.fp8_kv_calibration_tokens,
                config,
            )?;
            layer.set_o_dense_bf16(o_dense);
            layers.push(Box::new(layer));
            if (i + 1).is_multiple_of(12) || i + 1 == config.num_hidden_layers {
                tracing::info!("Loaded layers 0..{}", i + 1);
            }
        }
        gpu.synchronize(stream)?;
        tracing::info!(
            "Qwen3 dense weight loader: {} layers (all full attention, BF16 attn+MLP, \
             prefix '{pfx}')",
            layers.len()
        );
        Ok(layers)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, &format!("{}embed_tokens.weight", store_prefix(store)))
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        // Standard-convention norm → shift (module doc).
        dense_minus_one(store, &format!("{}norm.weight", store_prefix(store)), gpu)
    }

    fn load_lm_head(&self, store: &WeightStore, _config: &ModelConfig) -> Result<DenseWeight> {
        if store.contains("lm_head.weight") {
            return dense(store, "lm_head.weight");
        }
        let embed = format!("{}embed_tokens.weight", store_prefix(store));
        tracing::warn!(
            "qwen3 dense checkpoint ships no lm_head.weight (embedding-class export) — \
             falling back to {embed}. Token LOGITS from this model are NOT meaningful; \
             it is served for pooled-embedding outputs only."
        );
        dense(store, &embed)
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }

    /// Qwen3-VL dense (VL-Embedding/Reranker-8B): the ViT lives under
    /// `model.visual.*` in exactly the layout the Qwen3-VL loader reads —
    /// delegate. No-op (`None`) for plain-text checkpoints: the delegate
    /// returns early when `config.vision` is absent.
    fn load_vision_encoder(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<crate::layers::VisionEncoder>> {
        super::Qwen3VLWeightLoader.load_vision_encoder(store, config, gpu)
    }
}

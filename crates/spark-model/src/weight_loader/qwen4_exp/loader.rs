// SPDX-License-Identifier: AGPL-3.0-only

//! Weight loader of the `qwen4_exp` family (Qwen3.8-Flash-Next).

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::inventory::{self, LM};
use super::tensors::table;
use crate::layer::TransformerLayer;
use crate::layers::qwen4_exp::Qwen4ExpShared;
use crate::weight_loader::{ModelWeightLoader, TensorFilter};
use crate::weight_map::{DenseWeight, MtpWeights};

pub struct Qwen4ExpWeightLoader;

impl ModelWeightLoader for Qwen4ExpWeightLoader {
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
        if layer_kv_dtypes.iter().any(|d| *d != KvCacheDtype::Bf16) {
            bail!("qwen4_exp supports a BF16 KV cache only (--kv-cache-dtype bf16)");
        }
        let shared = Qwen4ExpShared::new(gpu, config)?;
        tracing::info!(
            "qwen4_exp: {:.1} MiB working memory (expanded expert scales included)",
            shared.scratch_bytes() as f64 / (1024.0 * 1024.0)
        );
        super::layers::load_layers(store, config, gpu, &shared, 0..config.num_hidden_layers)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        Ok(DenseWeight {
            weight: table(
                store,
                &format!("{LM}.embed_tokens.weight"),
                WeightDtype::BF16,
                config.vocab_size,
                config.hidden_size,
            )?,
        })
    }

    /// The model has no final norm: the last layer's mixer produces the
    /// LM-head input and [`Self::final_norm_kernel`] passes it through.
    fn load_final_norm(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        Ok(DenseWeight {
            weight: DevicePtr::NULL,
        })
    }

    fn load_lm_head(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        Ok(DenseWeight {
            weight: table(
                store,
                "lm_head.weight",
                WeightDtype::BF16,
                config.vocab_size,
                config.hidden_size,
            )?,
        })
    }

    /// The draft head is not implemented for this family; its tensors are
    /// not loaded ([`inventory::skip_tensor`]).
    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }

    /// Index data of the sparse attention: per KV block the raw key of
    /// every token and the index key of every `ratio` tokens, BF16.
    fn kv_aux_block_bytes(&self, config: &ModelConfig, kv_block_size: usize) -> usize {
        config
            .qwen4_exp
            .as_ref()
            .map_or(0, |q| q.index_block_bytes(kv_block_size))
    }

    fn tensor_filter(&self) -> Option<TensorFilter> {
        Some(inventory::FILTER)
    }

    fn token_serial(&self) -> bool {
        true
    }

    fn serial_prefill_rows(&self) -> usize {
        crate::layers::qwen4_exp::PREFILL_ROWS
    }

    fn final_norm_kernel(&self) -> (&'static str, &'static str) {
        ("gated_residual", "gr_passthrough")
    }
}

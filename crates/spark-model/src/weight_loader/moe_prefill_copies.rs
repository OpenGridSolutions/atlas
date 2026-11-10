// SPDX-License-Identifier: AGPL-3.0-only

//! Whether a loader builds the transposed copies of the MoE expert weights
//! that the fast prefill kernel reads. Without them prefill runs the
//! grouped GEMM on the original layout.
//!
//! The copies are permanent and come out of the memory the KV pool is
//! sized from afterwards, so MODEL.toml `[serving].moe_prefill_transpose`
//! decides per model. Without the key the loader builds them when they fit
//! into the memory that is free at load time — a reading that depends on
//! what the host and the driver hold at that moment.
//!
//! Used by the Qwen3 and Qwen3.5-family loaders; the Mistral and MiniMax
//! loaders keep their own fit checks and do not read the key.

use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;

/// Free memory the fit check leaves for the KV cache and buffers.
const HEADROOM_BYTES: usize = 2 * 1024 * 1024 * 1024;
/// NVFP4 scale group: one scale byte per 16 packed values.
const SCALE_GROUP: usize = 16;

/// Bytes the transposed copies take over all layers: 3 projections ×
/// `num_experts` × (packed + scale).
fn transposed_bytes(config: &ModelConfig) -> usize {
    let inter = config.moe_intermediate_size;
    let h = config.hidden_size;
    // gate/up: [inter, h/2] packed + [inter, h/group] scale
    let gu_bytes = inter * h / 2 + inter * h / SCALE_GROUP;
    // down:    [h, inter/2] packed + [h, inter/group] scale
    let d_bytes = h * inter / 2 + h * inter / SCALE_GROUP;
    config.num_experts * (2 * gu_bytes + d_bytes) * config.num_hidden_layers
}

/// `policy` is MODEL.toml `[serving].moe_prefill_transpose`; `available`
/// is only consulted without one.
fn skips(policy: Option<bool>, needed: usize, available: impl FnOnce() -> usize) -> bool {
    match policy {
        Some(build) => !build,
        None => needed > available().saturating_sub(HEADROOM_BYTES),
    }
}

/// True when the loader must not build the transposed MoE copies for
/// `config`. Logs the decision that deviates from building them.
pub(crate) fn skip_moe_transpose(config: &ModelConfig, gpu: &dyn GpuBackend) -> bool {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let needed = transposed_bytes(config);
    let mut seen = None;
    let skip = skips(config.moe_prefill_transpose, needed, || {
        *seen.insert(gpu.free_memory().unwrap_or(0))
    });
    match (skip, seen) {
        (true, Some(available)) => tracing::warn!(
            "Skipping MoE weight transposition ({:.1} GB needed, {:.1} GB available). \
             Prefill will use fallback grouped GEMM.",
            needed as f64 / GIB,
            available as f64 / GIB,
        ),
        (true, None) => tracing::info!(
            "MoE weight transposition off (MODEL.toml [serving].moe_prefill_transpose = false, \
             saves {:.1} GB for the KV pool). Prefill uses grouped GEMM.",
            needed as f64 / GIB,
        ),
        (false, _) => {}
    }
    skip
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: usize = 1024 * 1024 * 1024;

    #[test]
    fn model_toml_off_skips_however_much_memory_is_free() {
        assert!(skips(Some(false), 40 * GIB, || panic!(
            "free memory must not be read"
        )));
    }

    #[test]
    fn model_toml_on_builds_however_little_memory_is_free() {
        assert!(!skips(Some(true), 40 * GIB, || panic!(
            "free memory must not be read"
        )));
    }

    #[test]
    fn without_a_policy_the_copies_are_built_when_they_fit_with_headroom() {
        assert!(!skips(None, 40 * GIB, || 42 * GIB));
        assert!(skips(None, 40 * GIB, || 42 * GIB - 1));
        assert!(skips(None, 40 * GIB, || 0));
    }

    /// Qwen3-Next-80B: 48 layers × 512 experts of 2048 → 512 → 2048. The
    /// loader has logged "40.5 GB needed" for it on every GB10 load.
    #[test]
    fn transposed_bytes_match_the_logged_80b_figure() {
        let config: ModelConfig = serde_json::from_str(
            r#"{"hidden_size": 2048, "num_hidden_layers": 48, "num_experts": 512,
                "moe_intermediate_size": 512, "vocab_size": 151936,
                "num_attention_heads": 16, "num_key_value_heads": 2}"#,
        )
        .unwrap();
        let gib = transposed_bytes(&config) as f64 / GIB as f64;
        assert!((gib - 40.5).abs() < 0.05, "{gib:.2} GiB");
    }
}

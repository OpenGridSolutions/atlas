// SPDX-License-Identifier: AGPL-3.0-only

//! Per-model swap-target preparation (Task 6 Phase 1.3).
//!
//! Rebuilds, for an arbitrary (locally cached) model id, everything the
//! boot path derives from the checkpoint besides the weights themselves:
//! config, kernel target, tokenizer (+ chat template), sampling defaults,
//! parsers, grammar engine, per-model scheduler tokens, and the
//! [`ModelRuntime`] snapshot for the HTTP handlers. Also INSTALLS the
//! per-model process globals (numeric/boundary/mid-word masks, watchdog
//! params, loop-watchdog flag, TSCG) — call it only while the scheduler is
//! parked (state != Running), never while requests are decoding.
//!
//! Boot-fixed, args-scale values (max_batch_tokens, inference_reserve,
//! prefill budget, speculative flags) are deliberately NOT recomputed —
//! they are scheduler-level and stay at their boot values across swaps.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::snapshot::model_version_hash;

use crate::cli;

/// Everything a hot-swap load needs about its target model, computed
/// CPU-side before the snapshot payload is read into the arena.
pub(crate) struct PreparedSwap {
    pub model_id: String,
    pub model_hash: [u8; 32],
    pub model_dir: PathBuf,
    /// Post-processed config — weight_prefix is still unset; run
    /// `auto_detect_weight_prefix` after the store is rebuilt.
    pub config: ModelConfig,
    /// `arch/model/quant` key of the kernel target.
    pub target_key: String,
    pub modules: Vec<(&'static str, &'static [u8])>,
    pub model_runtime: crate::main_modules::ModelRuntime,
    pub scheduler_params: crate::scheduler::SchedulerModelParams,
    pub kv_dtype: spark_runtime::kv_cache::KvCacheDtype,
    pub layer_dtypes: Vec<spark_runtime::kv_cache::KvCacheDtype>,
    pub hss_cache_blocks_per_seq: Option<u32>,
    /// MODEL.toml `default_mtp_quantization` of the target (may be "").
    pub default_mtp_quantization: &'static str,
}

/// Stable registry key for a kernel target.
pub(crate) fn target_key(t: &atlas_core::target::KernelTarget) -> String {
    format!("{}/{}/{}", t.arch, t.model, t.quant)
}

/// Prepare a swap to `model_id`. Mirrors the boot path's config/tokenizer
/// phases in the same order; see module docs for what it installs.
pub(crate) fn prepare_swap_target(args: &cli::ServeArgs, model_id: &str) -> Result<PreparedSwap> {
    // 0/1. Resolve checkpoint + config (must already be cached locally —
    // preseed each swap target by booting it once).
    let model_dir = crate::model_resolver::resolve_model_dir(model_id, args.cache_dir.as_deref())
        .with_context(|| format!("resolving swap target '{model_id}'"))?;
    let revision = model_dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "local".to_string());
    let model_hash = model_version_hash(model_id, &revision);
    let (mut config, _config_json) = super::load_model_config(&model_dir)?;
    super::apply_lm_head_dtype_override(args, &mut config)?;
    super::merge_sidecar_quant_config(&model_dir, &mut config);

    // 2. Kernel target + quant compatibility (same gate as boot — a
    // mismatch would be silent garbage, not an error).
    let ptx_set = atlas_kernels::ptx_for_model_id(model_id, &config.model_type, config.hidden_size)
        .with_context(|| {
            format!(
                "no compiled kernel target matches swap target '{model_id}' \
                     (model_type '{}', hidden_size {})",
                config.model_type, config.hidden_size
            )
        })?;
    let model_quant = crate::main_modules::serve::canonicalize_model_quant(&config);
    if !crate::main_modules::serve::quant_pair_compatible(ptx_set.target.quant, &model_quant) {
        bail!(
            "kernel/model quant mismatch for swap target '{model_id}': kernel target {} \
             (quant={}) vs model quant={model_quant}",
            ptx_set.target,
            ptx_set.target.quant,
        );
    }
    if config.vision.is_some()
        && !ptx_set
            .modules
            .iter()
            .any(|(name, _)| *name == "vision_encoder")
    {
        tracing::warn!(
            "swap target '{model_id}' declares a vision tower but kernel target {} ships no \
             vision_encoder module — serving TEXT-ONLY",
            ptx_set.target,
        );
        config.vision = None;
    }

    // 2b. Topology onto the config (hot-swap is gated to world_size == 1).
    let topo = super::resolve_topology(args, &mut config)?;
    if topo.world_size != 1 {
        bail!(
            "hot-swap requires world_size == 1 (got {})",
            topo.world_size
        );
    }
    config.fp8_kv_calibration_tokens = if args.fp8_kv_calibration_tokens > 0 {
        args.fp8_kv_calibration_tokens
    } else {
        ptx_set.behavior.fp8_kv_calibration_tokens
    };

    // 6. Tokenizer + tokenizer-derived runtime. `resolve_tokenizer_runtime`
    // INSTALLS the numeric/boundary/mid-word masks (ArcSwap since 1.2, so
    // this re-install is exactly what makes model B's masks take effect).
    let mut eos_tokens = super::load_eos_tokens(&model_dir, &config);
    let sampling = super::load_sampling_defaults(&model_dir, args);
    let supports_thinking = config.capabilities().supports_thinking;
    let tokenizer = crate::tokenizer::ChatTokenizer::from_model_dir(
        &model_dir,
        eos_tokens[0],
        supports_thinking,
        &config.model_type,
        Some(std::path::Path::new(".")),
    )?;
    let tr = super::resolve_tokenizer_runtime(
        args,
        &mut config,
        &tokenizer,
        &mut eos_tokens,
        supports_thinking,
    );
    super::cap_vocab_size_to_tokenizer(&model_dir, &mut config);
    let kv = super::resolve_kv_cache_config(args, &config, ptx_set.behavior.default_kv_dtype)?;
    let tool_call_parser = super::resolve_tool_call_parser(args, &ptx_set, &config)?;

    // Install the remaining per-model globals (loop-watchdog flag,
    // watchdog params, TSCG) + emit the behavior audit log.
    super::log_behavior_audit(args, &ptx_set);

    let behavior = {
        let mut b = ptx_set.behavior.clone();
        if let Some(cli_budget) = args.max_thinking_budget {
            b.max_thinking_budget = cli_budget;
        }
        if let Some(cli_disable) = args.disable_tool_grammar {
            b.disable_tool_grammar = cli_disable;
        }
        b
    };
    let spontaneous_think_budget = args
        .max_thinking_budget
        .unwrap_or(ptx_set.behavior.max_thinking_budget);

    let model_runtime = crate::main_modules::ModelRuntime {
        tokenizer,
        // Clients address swap targets by the id they load them under.
        model_name: model_id.to_string(),
        vision_config: config.vision.clone(),
        default_temperature: sampling.temperature,
        default_top_k: sampling.top_k,
        default_top_p: sampling.top_p,
        default_top_n_sigma: sampling.top_n_sigma,
        default_min_p: sampling.min_p,
        tool_call_parser,
        reasoning_parser: tr.reasoning_parser_box,
        think_end_token_id: tr.think_end_token,
        think_start_token_id: tr.think_start_token,
        sampling_presets: ptx_set.sampling,
        tool_call_start_token_id: tr.tool_call_start_token,
        effective_context: 0,
        behavior,
    };
    let scheduler_params = crate::scheduler::SchedulerModelParams {
        eos_tokens,
        think_end_token: tr.think_end_token,
        think_start_token: tr.think_start_token,
        code_fence_token: tr.code_fence_token,
        tool_call_start_token: tr.tool_call_start_token,
        tool_call_end_token: tr.tool_call_end_token,
        grammar_engine: tr.grammar_engine,
        spontaneous_think_budget,
    };

    Ok(PreparedSwap {
        model_id: model_id.to_string(),
        model_hash,
        model_dir,
        config,
        target_key: target_key(&ptx_set.target),
        modules: ptx_set.modules.clone(),
        model_runtime,
        scheduler_params,
        kv_dtype: kv.kv_dtype,
        layer_dtypes: kv.layer_dtypes,
        hss_cache_blocks_per_seq: kv.hss_cache_blocks_per_seq,
        default_mtp_quantization: ptx_set.behavior.default_mtp_quantization,
    })
}

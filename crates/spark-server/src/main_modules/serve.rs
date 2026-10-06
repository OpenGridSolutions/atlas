// SPDX-License-Identifier: AGPL-3.0-only

//! Server initialization and runtime: phases 0-11 of the Atlas startup sequence.
//!
//! Refactor wave-4f extracted the bulk of each phase to `serve_phases.rs`
//! (resolve_topology, preflight_reserve, load_weight_store,
//! resolve_kv_cache_config, resolve_tokenizer_runtime, init_nccl_comm,
//! maybe_run_ep_worker, build_model, etc.) — `serve` now reads as a
//! straight call sequence rather than 1.8 KLOC of inline wiring.

use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::mpsc;

use crate::api::InferenceRequest;
use crate::main_modules::AppState;
use crate::main_modules::serve_phases;
use crate::tokenizer::ChatTokenizer;
use crate::{
    cli, conversation_store, rate_limiter, response_store, scheduler, scheduling_policy,
    session_manager,
};

pub(crate) async fn serve(mut args: cli::ServeArgs) -> Result<()> {
    tracing::info!("Atlas Spark starting...");
    tracing::info!("Licensed under AGPL-3.0-only — see /LICENSE in this container");

    // Categorized preseed lists (Plan.md Phase K) — parsed FIRST so a
    // malformed env var fails the boot before any GPU work starts. The
    // flat ATLAS_PRESEED_MODELS is GONE (no fallback): every preseeded
    // model now declares its serving category via the env var it is
    // listed in, and is only accepted at that category's endpoint.
    if std::env::var("ATLAS_PRESEED_MODELS").is_ok_and(|v| !v.trim().is_empty()) {
        anyhow::bail!(
            "ATLAS_PRESEED_MODELS has been removed — list each model under its category \
             instead: ATLAS_PRESEED_GENERATIVE (/v1/chat/completions), ATLAS_PRESEED_EMBED \
             (/v1/embeddings), ATLAS_PRESEED_RERANK (/v1/rerank), ATLAS_PRESEED_TRANSCRIBE \
             (/v1/audio/transcriptions), ATLAS_PRESEED_TRANSLATE (/v1/translate), \
             ATLAS_PRESEED_OCR (/v1/ocr), ATLAS_PRESEED_IMAGE (/v1/images/generations); \
             each is a JSON array of model ids"
        );
    }
    use crate::main_modules::hotswap::ModelCategory;
    let mut preseed_by_category: Vec<(ModelCategory, Vec<String>)> = Vec::new();
    for cat in [
        ModelCategory::Generative,
        ModelCategory::Embed,
        ModelCategory::Rerank,
        ModelCategory::Transcribe,
        ModelCategory::Translate,
        ModelCategory::Ocr,
        ModelCategory::Image,
    ] {
        let flags: &[String] = if cat == ModelCategory::Generative {
            &args.preseed_model
        } else {
            &[]
        };
        let ids = cli::resolve_category_models(
            cat.env_var(),
            std::env::var(cat.env_var()).ok().as_deref(),
            flags,
        )?;
        preseed_by_category.push((cat, ids));
    }
    // A model id in two categories would make the endpoint check
    // ambiguous — refuse the configuration outright.
    {
        let mut owner: std::collections::HashMap<String, ModelCategory> =
            std::collections::HashMap::new();
        for (cat, ids) in &preseed_by_category {
            for id in ids {
                if let Some(prev) = owner.insert(id.clone(), *cat) {
                    anyhow::bail!(
                        "model '{id}' is listed in both {} and {} — each model must have \
                         exactly one category",
                        prev.env_var(),
                        cat.env_var(),
                    );
                }
            }
        }
    }
    // The image sidecar loads on its first request, long after the boot
    // model sized its KV pool — reserve its memory in the budget now.
    if preseed_by_category
        .iter()
        .any(|(cat, ids)| *cat == ModelCategory::Image && !ids.is_empty())
    {
        spark_runtime::gpu::set_sidecar_pending_reserve_bytes(
            crate::api::images::pending_reserve_bytes(),
        );
    }
    for (cat, ids) in &preseed_by_category {
        if !ids.is_empty() {
            tracing::info!(
                "Preseed [{}] ({}): {} model(s) — {:?}",
                cat.as_str(),
                cat.endpoint(),
                ids.len(),
                ids,
            );
        }
    }

    // Boot-model fallback: with categorized preseed lists the arena serves
    // many models and requests lazy-load (Task 6 Phase 3), so naming one id
    // on the command line is redundant — default to the first
    // ATLAS_PRESEED_GENERATIVE entry. The boot preseed unloads it again
    // right after snapshotting, leaving the server idle either way.
    if args.model.is_none() && args.model_from_path.is_none() {
        let first_generative = preseed_by_category
            .iter()
            .find(|(cat, _)| *cat == ModelCategory::Generative)
            .and_then(|(_, ids)| ids.first().cloned());
        match first_generative {
            Some(id) => {
                tracing::info!(
                    "No MODEL argument — booting first ATLAS_PRESEED_GENERATIVE entry '{id}'"
                );
                args.model = Some(id);
            }
            None => anyhow::bail!(
                "no model to serve: pass MODEL, --model-from-path, or list at least one \
                 model in ATLAS_PRESEED_GENERATIVE"
            ),
        }
    }

    // 0. Resolve model directory from HF ID or path
    let model_dir = serve_phases::resolve_model_dir(&args)?;

    tracing::info!("Port: {}", args.port);

    tracing::info!("SSM decode dtype: f32 (full precision)");

    // 1. Load model config (supports HF config.json and Mistral params.json)
    let (mut config, config_json) = serve_phases::load_model_config(&model_dir)?;

    // CLI `--lm-head-dtype` override — shared with the hot-swap target
    // preparation (serve_phases::config).
    serve_phases::apply_lm_head_dtype_override(&args, &mut config)?;

    // ModelOpt-exported checkpoints drop a sibling `hf_quant_config.json`
    // whose TOP LEVEL is already the quantization block.
    serve_phases::merge_sidecar_quant_config(&model_dir, &mut config);

    if let Some(ref qc) = config.quantization_config {
        tracing::info!(
            "Quantization config: method={:?}, algo={:?}, format={:?}, {} module(s) in ignore list",
            qc.quant_method,
            qc.quant_algo,
            qc.format,
            qc.ignore_modules.len(),
        );
    }

    tracing::info!(
        "Model config: {} layers, {} attention, {} SSM, {} experts, rope_theta={}, head_dim={}, rotary_dim={}",
        config.num_hidden_layers,
        config.num_attention_layers(),
        config.num_ssm_layers(),
        config.num_experts,
        config.rope_theta,
        config.head_dim,
        config.rotary_dim(),
    );

    // 2. Select kernel target and initialize GPU backend
    //
    // Exact MODEL.toml `hf_id` match wins; otherwise each kernel target
    // declares which (model_type, hidden_size) pairs it supports via
    // [[model_types]] in MODEL.toml (exact hidden_size beats wildcards).
    let boot_model_id = args.model.clone().unwrap_or_default();
    let ptx_set =
        atlas_kernels::ptx_for_model_id(&boot_model_id, &config.model_type, config.hidden_size)
            .with_context(|| {
                format!(
                    "No compiled kernel target matches model_type '{}' / hidden_size={}. \
             Available targets: {:?}",
                    config.model_type,
                    config.hidden_size,
                    atlas_kernels::available_targets()
                        .iter()
                        .map(|t| &t.target.model)
                        .collect::<Vec<_>>(),
                )
            })?;
    let sampling_presets = ptx_set.sampling;

    // QV1 (2026-05-26): kernel ↔ model quant compatibility validation.
    //
    // `ptx_for_config` selects on (model_type, hidden_size) but not on
    // QUANT. With ATLAS_TARGET_QUANT=* the build emits one bundle per
    // model whose label happens to be the first variant compiled
    // ("nvfp4") even when the bundle contains native FP8 dispatch too.
    // For now we accept the historically-compatible pairs hardcoded in
    // `quant_pair_compatible` (and only those). Anything else hard
    // errors RIGHT HERE with an explicit "rebuild with X" message,
    // before any weight loading runs and any silent garbage path can
    // be entered. A future refinement moves the compat list into
    // MODEL.toml `[kernel].supported_quants`.
    let model_quant = canonicalize_model_quant(&config);
    let kernel_quant = ptx_set.target.quant;
    if !quant_pair_compatible(kernel_quant, &model_quant) {
        anyhow::bail!(
            "Kernel/model QUANT MISMATCH. Kernel target: {} (quant={kernel_quant}). \
             Model declares quant={model_quant} ({}). \
             The compiled kernel set has no known dispatch path for \
             quant '{model_quant}' — loading would produce silent garbage. \
             Rebuild with ATLAS_TARGET_QUANT={model_quant} (or =* to bundle multiple \
             variants) and restart.",
            ptx_set.target,
            describe_quant_source(&config),
        );
    }
    tracing::info!(
        "Selected kernel target: {} ({} modules) — quant compat: kernel={kernel_quant} \
         model={model_quant} OK",
        ptx_set.target,
        ptx_set.modules.len(),
    );

    // Text-only kernel target + a checkpoint that ships a vision tower: honor the
    // TARGET spec and serve text-only rather than failing the build at
    // `vision_encoder module not loaded`. Some VL checkpoints (e.g.
    // Kbenkhaled/Qwen3.5-27B-NVFP4) carry a `vision_config`, but their Atlas
    // kernel target (qwen3.5-27b) ships no `vision_encoder` PTX module. Drop the
    // vision tower to text-only; image inputs are unsupported until the target
    // is rebuilt with vision.
    if config.vision.is_some()
        && !ptx_set
            .modules
            .iter()
            .any(|(name, _)| *name == "vision_encoder")
    {
        tracing::warn!(
            "Checkpoint declares a vision tower but kernel target {} ships no \
             vision_encoder module — serving TEXT-ONLY (image inputs ignored). \
             Rebuild the target with vision to enable images.",
            ptx_set.target,
        );
        config.vision = None;
    }

    // Apply MODEL.toml [behavior].default_num_drafts unless user passed --num-drafts.
    serve_phases::apply_model_default_num_drafts(&mut args, &ptx_set);

    let (gpu, free_mem) = serve_phases::init_gpu_backend(&args, &ptx_set)?;

    // ── Pre-load reserve preflight ──
    let serve_phases::ReservePreflight {
        inference_reserve,
        buffer_arena_bytes,
        gdn_two_phase_bytes,
        ssm_prefill_chunk,
        max_batch_tokens_pre,
    } = serve_phases::preflight_reserve(&args, &config, free_mem)?;
    let total_reserve = inference_reserve + buffer_arena_bytes;

    // 2a-2. OOM watchdog: background async task that polls GPU memory every 2s.
    // On GB10 unified memory, GPU OOM = system freeze, so we exit(1) early.
    // Threshold: 2 GB (enough to detect runaway allocation before system locks up).
    //
    // CUDA-only: Apple Silicon UMA already exposes `currentAllocatedSize`
    // and the OS handles memory pressure via Metal's working-set policy,
    // so the dedicated watchdog isn't needed.
    #[cfg(feature = "cuda")]
    let _oom_watchdog = spark_runtime::cuda_backend::spawn_oom_watchdog(
        2048, // 2 GB threshold
        std::time::Duration::from_secs(2),
    );
    #[cfg(feature = "cuda")]
    tracing::info!("OOM watchdog started (threshold: 2 GB, interval: 2s)");

    // 2b. Resolve TP / EP topology and set on model config.
    let serve_phases::Topology {
        world_size,
        tp_size: _tp_size,
        ep_size,
        tp_rank: _tp_rank,
        ep_rank,
    } = serve_phases::resolve_topology(&args, &mut config)?;
    // FP8 KV calibration: CLI flag overrides MODEL.toml default.
    config.fp8_kv_calibration_tokens = if args.fp8_kv_calibration_tokens > 0 {
        args.fp8_kv_calibration_tokens
    } else {
        ptx_set.behavior.fp8_kv_calibration_tokens
    };

    // 3. Load model weights
    let oom_reserve_bytes = args.oom_guard_mb * 1024 * 1024;
    tracing::info!("OOM guard reserve: {} MB", args.oom_guard_mb);
    let mut store = serve_phases::load_weight_store(
        &args,
        &config,
        &model_dir,
        gpu.as_ref(),
        ep_rank,
        ep_size,
        oom_reserve_bytes,
    )?;

    // 3a. Migrate weights into the process-lifetime arena (hot-swap MVP).
    // Must run strictly before build_model: the model captures DevicePtr
    // values out of the store, so re-pointing afterwards changes nothing
    // (docs/hotswap/trace.md §4). Same-model unload/reload rebuilds views
    // as arena_base + offset from the manifest recorded here.
    let _hotswap_arena = serve_phases::migrate_weights_to_arena(&args, gpu.as_ref(), &mut store)?;

    // 3b. Auto-detect weight key prefix for nested models.
    serve_phases::auto_detect_weight_prefix(&store, &mut config);

    // Pre-flight weight-store / config consistency check. Runs before
    // NCCL init so a mis-matched checkpoint (wrong expert count, MiniMax
    // + MTP tensors + `--speculative`, missing embedding, etc.) aborts
    // this rank with a readable error BEFORE rank 1 ever connects or
    // `ncclCommInitRank` is called. Several community re-quants of
    // MiniMax M2.7 hang on NCCL init today because the actual mismatch
    // only surfaces later inside `build_model`; this check surfaces it
    // up-front.
    spark_model::preflight::preflight(&store, &config, args.speculative)
        .context("Checkpoint pre-flight check failed")?;

    // Resolve and log the QuantFormat dispatch decision now so a silent
    // fallback is visible in the server log (and not just in the
    // detection code path mid-load). The returned trait object is
    // currently only consulted via `detect_nvfp4_variant`; explicit
    // use at each load site is a follow-up migration.
    let quant_format = spark_model::quant_format::detect_quant_format(&config, &store);
    tracing::info!(
        "Quantization format: {} (base variant {:?}), ignored globs = {}",
        quant_format.name(),
        quant_format.base_variant(),
        match &config.quantization_config {
            Some(qc) => qc.ignore_modules.len(),
            None => 0,
        },
    );

    // MTP throughput-aware gate is applied at RUNTIME, not here. The earlier
    // static "FP8 ⇒ MTP off" weight-quant heuristic was removed: hardcoding the
    // decision against the weight format wrongly bars a future FP8 checkpoint
    // where MTP would help, and it conflated weight format (a proxy) with the
    // thing that actually decides MTP economics — the per-config verify-step
    // cost relative to a plain decode step. The scheduler now MEASURES that
    // ratio over the first decode steps of serving and auto-disables MTP only
    // when it is provably net-negative (verify multiplier ≥ 1 + num_drafts).
    // See `scheduler::mtp_gate`.

    // 4. Post-load OOM check + audit log.
    serve_phases::post_load_memory_audit(
        &args,
        &config,
        gpu.as_ref(),
        store.total_bytes(),
        free_mem,
        inference_reserve,
        total_reserve,
        gdn_two_phase_bytes,
        max_batch_tokens_pre,
    )?;

    // 5. Build model via factory.
    let serve_phases::PrefillBudget {
        prefill_budget,
        max_batch_tokens,
        spec_tokens: _spec_tokens,
    } = serve_phases::resolve_prefill_budget(&args, ssm_prefill_chunk);
    if args.dflash && args.enable_prefix_caching {
        tracing::warn!(
            "dflash: --enable-prefix-caching has a community-reported correctness regression on SM12.x with DFlash; outputs may be wrong on multi-turn cache hits. Run a greedy diff-test against a non-DFlash baseline before relying on outputs."
        );
    }
    // 2026-06-18: the previously-documented warm-Marconi-restore × MTP
    // corruption on hybrid SSM models is RESOLVED. Verified by a greedy
    // ground-truth A/B at batch=1 (the level MTP runs at — MTP is gated to
    // `active.len() == 1` in the scheduler): a real 4-turn agentic
    // conversation (incl. tool-call turns) produced byte-identical token
    // streams with Marconi ON vs OFF (full SSM recompute), 12/12 turns. The
    // #155 lineage (decode-era block-aligned snapshots, the
    // commit_verify_state_async live-state invariant, finish-leaf
    // sync_secondary) closed the interaction. Any residual divergence seen
    // only at batch>1 is FP8 low-margin argmax tie-breaking from
    // batch-size-dependent MoE-kernel rounding (a known FP8 quality-floor
    // property present for fresh non-cached sequences too), not a Marconi
    // state-management defect — so no warning is emitted here.
    let prefix_cache = serve_phases::build_prefix_cache(&args);
    let comm = serve_phases::init_nccl_comm(&args, gpu.as_ref(), world_size)?;
    if args.profile {
        // SAFETY: called before any threads are spawned.
        unsafe {
            std::env::set_var("ATLAS_PROFILE", "1");
        }
    }
    serve_phases::cap_vocab_size_to_tokenizer(&model_dir, &mut config);
    let serve_phases::KvCacheConfig {
        effective_kv_dtype_str: _,
        kv_dtype,
        layer_dtypes,
        hss_cache_blocks_per_seq,
    } = serve_phases::resolve_kv_cache_config(&args, &config, ptx_set.behavior.default_kv_dtype)?;

    // Fail-fast: every kernel handle the selected --kv-cache-dtype's dispatch
    // arms need must resolve NOW — not at first dispatch after a multi-minute
    // weight load (or, worse, via a silent wrong-kernel fall-through).
    // Validates each distinct per-layer dtype (high-precision / boundary
    // layers can differ from the base dtype).
    {
        let mut distinct: Vec<spark_runtime::kv_cache::KvCacheDtype> = vec![kv_dtype];
        for d in &layer_dtypes {
            if !distinct.contains(d) {
                distinct.push(*d);
            }
        }
        for d in distinct {
            spark_model::layers::qwen3_attention::validate_required_kv_kernels(
                gpu.as_ref(),
                d,
                config.head_dim,
            )
            .context("kv-cache kernel preflight failed")?;
        }
    }
    let dflash_drafter_state = serve_phases::load_dflash_drafter(&args, &ptx_set, gpu.as_ref())?;
    let dflash_args =
        dflash_drafter_state
            .as_ref()
            .map(|(s, c)| spark_model::factory::DflashBuildArgs {
                drafter_store: s,
                drafter_config: c.clone(),
                gamma: Some(args.dflash_gamma),
                window_size: if args.dflash_window_size > 0 {
                    Some(args.dflash_window_size)
                } else {
                    None
                },
            });
    // Hot-swap reload re-runs build_model on the scheduler thread; clone
    // the comm handle that is moved into this boot-time call. (KV dtypes
    // are recomputed per swap target since Task 6 / 1.3.)
    let reload_comm = comm.clone();

    // Watermark alloc registry (hot-swap unload): record every device
    // allocation build_model makes (~178: transposes, dequants, KV/SSM
    // pools, buffers) so unload can free exactly the derived buffers.
    // Frees always unrecord, so build-time self-frees never double-free.
    gpu.set_alloc_tracking(true);
    // NVMe KV tier (requirements §5): per-model store bound to the boot
    // model's identity + KV layout. Built before `layer_dtypes` moves.
    let boot_kv_identity = serve_phases::compute_kv_identity(
        &boot_model_id,
        args.block_size,
        &config,
        kv_dtype,
        &layer_dtypes,
    );
    // AUTH-4: trusting proxy headers is only safe when nothing but the
    // proxy can reach us.
    if args.trust_proxy_headers && !matches!(args.bind.as_str(), "127.0.0.1" | "localhost" | "::1")
    {
        anyhow::bail!(
            "--trust-proxy-headers requires a loopback bind (got --bind {}):              any direct client could claim X-Priority: high",
            args.bind
        );
    }
    crate::swap_costs::init(std::path::Path::new(&args.snapshot_dir), args.ewma_alpha);
    let boot_kv_tier = crate::kv_tier::KvTier::build_from_args(
        &args,
        &crate::kv_tier::KvTierIdentity {
            model_hash: boot_kv_identity.model_hash,
            layout_hash: boot_kv_identity.layout_hash,
        },
    )?;
    let avail_before_build = spark_runtime::meminfo::read().map(|m| m.mem_available);
    let model = serve_phases::build_model(
        &args,
        &boot_model_id,
        &config,
        &store,
        gpu,
        max_batch_tokens,
        kv_dtype,
        inference_reserve,
        layer_dtypes,
        hss_cache_blocks_per_seq,
        prefix_cache,
        comm,
        dflash_args,
        ptx_set.behavior.default_mtp_quantization,
    )?;
    // Stop recording: serving-time allocations come from pre-allocated
    // pools (runtime invariant); anything allocated after this line is not
    // owned by the model build and must not be freed at unload.
    if let Some(g) = model.gpu_backend() {
        g.set_alloc_tracking(false);
    }
    if let Some(tier) = &boot_kv_tier {
        model.set_kv_evict_guard(Some(tier.evict_guard()));
    }
    observe_runtime_reserve_delta(&boot_model_id, avail_before_build);

    // Kernel load audit: print the table of every kernel resolved during model
    // construction (grouped by module/operation family) + flag any MISSING
    // (handle 0 → silent slower-fallback dispatch). Catches build/codegen
    // regressions like a dropped pipelined GEMM at load time, not as a
    // mystery slowdown.
    tracing::info!(
        "{}",
        spark_runtime::kernel_audit::render_kernel_table(
            &ptx_set.modules,
            atlas_kernels::KERNEL_SET_HASH,
        )
    );

    // Phase 6.3 — HSS config built early so the EP worker can install it.
    let early_high_speed_swap_cfg = serve_phases::build_high_speed_swap_config(&args)?;

    // EP worker: rank > 0 enters command loop, returns when head exits.
    let mut model_opt = Some(model);
    if serve_phases::maybe_run_ep_worker(&args, &mut model_opt, &early_high_speed_swap_cfg)? {
        return Ok(());
    }
    let model = model_opt.expect("head retains model on rank 0");

    // TQ+ InnerQ: opt-in via `TURBO_INNERQ=N` (N = calibration token count).
    // Once enabled, the kernel-side apply pass starts accumulating K² stats
    // and the scheduler polls `maybe_finalize` per prefill chunk; once N
    // tokens have flowed through, scales activate and stay live for the
    // process lifetime. CUDA-only: the driver talks to the CUDA Driver API
    // directly via `atlas_core::registry`, which doesn't exist on metal.
    #[cfg(feature = "cuda")]
    if let Some(driver) = spark_model::layers::qwen3_attention::InnerQDriver::from_env() {
        match driver.start() {
            Ok(()) => {
                let _ = spark_model::layers::qwen3_attention::INNERQ.set(driver);
            }
            Err(e) => {
                tracing::warn!("InnerQ calibration disabled: start() failed: {e:#}");
            }
        }
    }

    // Build EOS token list from generation_config.json (authoritative) or config.json fallback
    let mut eos_tokens = serve_phases::load_eos_tokens(&model_dir, &config);

    // Read default sampling parameters from generation_config.json.
    let serve_phases::SamplingDefaults {
        temperature: default_temperature,
        top_k: default_top_k,
        top_p: default_top_p,
        top_n_sigma: default_top_n_sigma,
        min_p: default_min_p,
    } = serve_phases::load_sampling_defaults(&model_dir, &args);

    // 6. Load tokenizer
    // Thinking support is derived from model capabilities, not hardcoded model names.
    // Models with SSM layers or Qwen3.5-style architecture support <think> tokens.
    // The --enable-thinking flag controls OPEN-ENDED vs CLOSED thinking.
    let caps = config.capabilities();
    let supports_thinking = caps.supports_thinking;
    let tokenizer = ChatTokenizer::from_model_dir(
        &model_dir,
        eos_tokens[0],
        supports_thinking,
        &config.model_type,
        Some(std::path::Path::new(".")), // repo root for override templates
    )?;

    // (AM1 attractor-mask registration removed 2026-06-03 — see
    // decode_logits_seq.rs / compile_tools.rs; `lean` was an Atlas-only
    // decode artifact, now fixed at the grammar `first_char` rule.)

    // Tokenizer-derived runtime: vocab cap, reasoning parser, think tokens,
    // im_start hard-stop, tool-call open/close tokens, and the XGrammar
    // engine.
    let serve_phases::TokenizerRuntime {
        reasoning_parser_box,
        think_end_token,
        think_start_token,
        code_fence_token,
        tool_call_start_token,
        tool_call_end_token,
        grammar_engine,
    } = serve_phases::resolve_tokenizer_runtime(
        &args,
        &mut config,
        &tokenizer,
        &mut eos_tokens,
        supports_thinking,
    );

    // 7. Create scheduler channel + spawn scheduler
    let (request_tx, request_rx) = mpsc::channel::<InferenceRequest>(args.max_num_seqs);

    let model_name = serve_phases::resolve_model_name(&args, &config_json, &model_dir);

    // Tool call parser resolution: CLI > MODEL.toml > defaults table.
    let tool_call_parser = serve_phases::resolve_tool_call_parser(&args, &ptx_set, &config)?;

    // Per-model runtime snapshot (Task 6 Phase 1.1) — everything a
    // different model would need a different value for. Lives in a shared
    // ArcSwap cell so the hot-swap reloader (below) can replace it and the
    // HTTP handlers (AppState) read it, one snapshot per request.
    let model_runtime = Arc::new(crate::main_modules::ModelRuntime {
        tokenizer,
        model_name: model_name.clone(),
        vision_config: config.vision.clone(),
        default_temperature,
        default_top_k,
        default_top_p,
        default_top_n_sigma,
        default_min_p,
        tool_call_parser,
        reasoning_parser: reasoning_parser_box,
        think_end_token_id: think_end_token,
        think_start_token_id: think_start_token,
        sampling_presets,
        tool_call_start_token_id: tool_call_start_token,
        // Behavior and effective_context from MODEL.toml, embedded at build time.
        effective_context: 0, // TODO: embed effective_context in TargetPtxSet
        behavior: {
            let mut b = ptx_set.behavior.clone();
            if let Some(cli_budget) = args.max_thinking_budget {
                b.max_thinking_budget = cli_budget;
            }
            if let Some(cli_disable) = args.disable_tool_grammar {
                b.disable_tool_grammar = cli_disable;
            }
            b
        },
    });
    let runtime_cell = Arc::new(arc_swap::ArcSwap::new(model_runtime));

    // Readiness gate + scheduler handle are shared with the hot-swap
    // context, so both exist before the scheduler thread and AppState.
    let model_ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let scheduler_handle = scheduler::SchedulerHandle::new();

    // Hot-swap context: arena + manifest recorded by the boot migration.
    // The model revision is the HF snapshot hash (model_dir's last path
    // component), so snapshots stop resolving after a checkpoint update.
    // EP multi-rank and DFlash are out of hot-swap scope (worker ranks and
    // the drafter's second WeightStore are not arena-managed).
    let hotswap_enabled = ep_size == 1 && !args.dflash;
    if !hotswap_enabled && _hotswap_arena.is_some() {
        tracing::warn!("Hot-swap endpoints disabled: EP world_size > 1 or --dflash set");
    }
    // Model registry (Task 6 Phase 2): boot model + categorized preseed
    // lists, deduped. Lazy request-time swaps and the listing APIs only
    // see these. TRANSCRIBE ids are deliberately NOT registered here —
    // whisper is served by the resident sidecar, never by the arena — but
    // are collected for the /v1/audio/transcriptions model check.
    let boot_revision = model_dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "local".to_string());
    // The boot model's category: its entry in a preseed list if present,
    // else generative (the positional arg is a chat model by convention).
    let boot_category = preseed_by_category
        .iter()
        .find(|(_, ids)| ids.contains(&model_name))
        .map(|(cat, _)| *cat)
        .unwrap_or(ModelCategory::Generative);
    let mut model_registry = vec![crate::main_modules::hotswap::RegisteredModel {
        model_id: model_name.clone(),
        model_hash: spark_runtime::snapshot::model_version_hash(&model_name, &boot_revision),
        category: boot_category,
    }];
    #[allow(unused_mut)]
    let mut transcribe_models: Vec<String> = Vec::new();
    #[allow(unused_mut)]
    let mut image_models: Vec<String> = Vec::new();
    #[cfg(feature = "cuda")]
    for (cat, ids) in &preseed_by_category {
        for id in ids {
            if *cat == ModelCategory::Transcribe {
                tracing::info!(
                    "Preseed [transcribe]: '{id}' is served by the resident Whisper sidecar \
                     ({} loads it on first request) — not part of the arena swap registry",
                    cat.endpoint(),
                );
                transcribe_models.push(id.clone());
                continue;
            }
            if *cat == ModelCategory::Image {
                tracing::info!(
                    "Preseed [image]: '{id}' is served by the resident Qwen-Image sidecar \
                     ({} loads it on first request) — not part of the arena swap registry",
                    cat.endpoint(),
                );
                image_models.push(id.clone());
                continue;
            }
            if model_registry.iter().any(|m| m.model_id == *id) {
                continue;
            }
            // Servability-gated registration: a preseed entry whose model
            // class Atlas cannot serve yet (no loader / no kernel target /
            // quant mismatch / unparseable config) is SKIPPED with a
            // warning instead of aborting boot — the full .env list stays
            // in place and the server comes up with whatever is loadable.
            // Skipped ids are not registered, so requests naming them get
            // the regular 404 + available-models listing.
            match serve_phases::registered_model_checked(&args, id, *cat) {
                Ok(m) => model_registry.push(m),
                Err(e) => tracing::warn!(
                    "Preseed: skipping '{id}': {e:#} — model NOT registered (boot continues)"
                ),
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    if preseed_by_category.iter().any(|(_, ids)| !ids.is_empty()) {
        anyhow::bail!("--preseed-model / ATLAS_PRESEED_* requires the CUDA backend");
    }
    let lazy_enabled = preseed_by_category.iter().any(|(cat, ids)| {
        !matches!(cat, ModelCategory::Transcribe | ModelCategory::Image) && !ids.is_empty()
    });
    if lazy_enabled && !hotswap_enabled {
        anyhow::bail!(
            "preseed/lazy multi-model mode requires hot-swap (weights arena enabled, \
             no EP world_size > 1, no --dflash)"
        );
    }

    let hotswap_ctx = _hotswap_arena
        .filter(|_| hotswap_enabled)
        .map(|arena_state| {
            Arc::new(crate::main_modules::hotswap::HotswapCtx::new(
                arena_state.arena,
                arena_state.manifest,
                model_name.clone(),
                &boot_revision,
                model_registry.clone(),
                lazy_enabled,
                std::path::PathBuf::from(&args.snapshot_dir),
                scheduler_handle.clone(),
                model_ready.clone(),
            ))
        });

    // Memory heartbeat (Todo.md 1.3): one INFO line per minute on target
    // `mem`; threshold breaches WARN into critical.log (A1/A2/A8 early
    // warning).
    let _mem_heartbeat = crate::main_modules::heartbeat::spawn(hotswap_ctx.clone());

    // Model reloader for hot-swap load: runs on the scheduler thread while
    // parked (state NoModel, CUDA context bound). `None` reloads the model
    // that occupied the arena; `Some(id)` swaps to a DIFFERENT model (Task 6
    // Phase 1.3): per-model prep rebuilds config/tokenizer/masks/parsers,
    // the snapshot payload streams into the arena (io_uring + O_DIRECT),
    // and the model is rebuilt against its own kernel target. PTX modules
    // are never unloaded; a new target's modules load additively into the
    // AtlasRegistry singleton on first use.
    #[cfg(feature = "cuda")]
    let reload: scheduler::ModelReloader = {
        let ctx = hotswap_ctx.clone();
        let reload_args = args.clone();
        let boot_target_key = serve_phases::target_key(&ptx_set.target);
        let boot_modules = ptx_set.modules.clone();
        let reload_runtime_cell = runtime_cell.clone();
        Box::new(move |requested: Option<&str>| {
            let ctx = ctx
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("hot-swap context missing"))?;
            let t0 = std::time::Instant::now();
            let target_id = match requested {
                Some(id) => id.to_string(),
                None => ctx.active_model().model_id,
            };

            // Per-model CPU prep. Installs the target's scheduler masks /
            // watchdog params (safe: scheduler is parked, state != Running).
            let prep = serve_phases::prepare_swap_target(&reload_args, &target_id)?;

            let path = ctx.snapshot_path_for(&prep.model_id, &prep.model_hash);
            if !path.exists() {
                anyhow::bail!(
                    "snapshot not found: {} — boot '{}' once and POST /admin/models/snapshot",
                    path.display(),
                    prep.model_id,
                );
            }
            let (header, manifest) = spark_runtime::snapshot::read_snapshot_meta(&path)?;
            if header.model_version_hash != prep.model_hash {
                anyhow::bail!(
                    "snapshot {} was written for a different (model, revision, layout) — \
                     refusing to load",
                    path.display()
                );
            }
            ctx.arena.reset();
            let read_stats = spark_runtime::snapshot::read_payload_into_arena(
                &path,
                &ctx.arena,
                &header,
                256 * 1024 * 1024,
                reload_args.snapshot_read_qd,
            )?;
            let store = spark_runtime::weights_arena::store_from_manifest(&ctx.arena, &manifest)?;
            let mut config = prep.config;
            serve_phases::auto_detect_weight_prefix(&store, &mut config);

            // Backend handle bound to the target's kernel modules. The boot
            // target reuses the registry's default maps; any other target
            // loads its modules additively on first swap.
            let gpu: Box<dyn spark_runtime::gpu::GpuBackend> = if prep.target_key == boot_target_key
            {
                Box::new(spark_runtime::cuda_backend::AtlasCudaBackend::new(
                    reload_args.gpu_ordinal,
                    &boot_modules,
                )?)
            } else {
                Box::new(
                    spark_runtime::cuda_backend::AtlasCudaBackend::new_for_target(
                        &prep.target_key,
                        &prep.modules,
                    )?,
                )
            };
            // Same fail-fast as boot: every KV kernel the target's dtypes
            // dispatch to must resolve before the model is built.
            {
                let mut distinct: Vec<spark_runtime::kv_cache::KvCacheDtype> = vec![prep.kv_dtype];
                for d in &prep.layer_dtypes {
                    if !distinct.contains(d) {
                        distinct.push(*d);
                    }
                }
                for d in distinct {
                    spark_model::layers::qwen3_attention::validate_required_kv_kernels(
                        gpu.as_ref(),
                        d,
                        config.head_dim,
                    )
                    .context("kv-cache kernel preflight failed for swap target")?;
                }
            }
            // The inference reserve (SSM pools + snapshots + GDN prefill
            // buffers + CUDA headroom) is MODEL-dependent — state sizes and
            // GDN dims differ per checkpoint. Reusing the boot model's value
            // under-reserved the 2026-10-03 deployment OOM kill: the 80B
            // boot reserve (4.6 GB) stood in for the 27B swap target's real
            // ~8.7 GB, and the KV pool clamp handed the difference to KV.
            let inference_reserve = {
                // Only the reserve VALUE is wanted here — free_mem=MAX skips
                // preflight_reserve's fit bail, which reads free memory and
                // false-negatives right after an eviction while the driver
                // pool still holds the unloaded model's pages (it refused
                // the lazy reload after the 2026-10-03 image eviction at
                // "8.69 GB free" with ~40 GB pool-held). Whether things fit
                // is decided by the KV budget's pinnable clamp and, finally,
                // by the allocations themselves.
                let r = serve_phases::preflight_reserve(&reload_args, &config, usize::MAX)
                    .context("inference-reserve computation failed for swap target")?;
                if r.inference_reserve != inference_reserve {
                    tracing::info!(
                        "Swap target inference reserve: {:.2} GiB (boot model's was {:.2} GiB)",
                        r.inference_reserve as f64 / (1u64 << 30) as f64,
                        inference_reserve as f64 / (1u64 << 30) as f64,
                    );
                }
                crate::swap_costs::runtime_reserve_or(&prep.model_id, r.inference_reserve)
            };
            // Track the rebuild's derived-buffer allocations for the next
            // unload (same window as the boot build).
            gpu.set_alloc_tracking(true);
            crate::memdebug::on_build_phase("start", &prep.model_id);
            let avail_before_build = spark_runtime::meminfo::read().map(|m| m.mem_available);
            let prefix_cache = serve_phases::build_prefix_cache(&reload_args);
            let model = match serve_phases::build_model(
                &reload_args,
                &prep.model_id,
                &config,
                &store,
                gpu,
                max_batch_tokens,
                prep.kv_dtype,
                inference_reserve,
                prep.layer_dtypes.clone(),
                prep.hss_cache_blocks_per_seq,
                prefix_cache,
                reload_comm.clone(),
                None, // dflash out of hot-swap scope
                prep.default_mtp_quantization,
            ) {
                Ok(m) => {
                    crate::memdebug::on_build_phase("ok", &prep.model_id);
                    observe_runtime_reserve_delta(&prep.model_id, avail_before_build);
                    m
                }
                Err(e) => {
                    crate::memdebug::on_build_phase("fail", &prep.model_id);
                    // Todo.md 0.4: an aborted rebuild must not strand its
                    // device allocations — the 2026-09-11 retry loop leaked
                    // memory on every failed load until nothing fit anymore.
                    // build_model consumed the backend Box, so drain the
                    // process-global tracker handle-less, and DONTNEED the
                    // half-written arena payload right away instead of at
                    // the next attempt's reset.
                    let (freed, failed) =
                        spark_runtime::cuda_backend::drain_and_free_tracked_allocs();
                    ctx.arena.reset();
                    tracing::warn!(
                        target: "hotswap",
                        event = "load_fail_cleanup",
                        freed,
                        failed,
                        "freed device allocations from aborted model build"
                    );
                    return Err(e);
                }
            };
            if let Some(g) = model.gpu_backend() {
                g.set_alloc_tracking(false);
            }

            // Install the target's per-request CPU state and identity —
            // handlers snapshot the new ModelRuntime from here on.
            reload_runtime_cell.store(Arc::new(prep.model_runtime));
            *ctx.manifest.write() = Arc::new(manifest);
            *ctx.active.write() = crate::main_modules::hotswap::ActiveModel {
                model_id: prep.model_id.clone(),
                model_hash: prep.model_hash,
            };
            *ctx.active_since.lock() = std::time::Instant::now();

            let duration_ms = t0.elapsed().as_millis() as u64;
            crate::metrics::MODEL_SWAPS_TOTAL
                .with_label_values(&[prep.model_id.as_str()])
                .inc();
            crate::metrics::MODEL_SWAP_SECONDS.observe(duration_ms as f64 / 1000.0);
            crate::metrics::MODEL_SWAP_READ_BYTES_TOTAL.inc_by(read_stats.bytes_read);
            // MOD-5/8: refine this model's load + warmup costs per real swap.
            crate::swap_costs::observe_load(
                &prep.model_id,
                read_stats.duration_ms as f64,
                (duration_ms as f64 - read_stats.duration_ms as f64).max(0.0),
                read_stats.gib_per_s,
            );
            let stats = serde_json::json!({
                "model": prep.model_id,
                "kernel_target": prep.target_key,
                "bytes_read": read_stats.bytes_read,
                "read_ms": read_stats.duration_ms as u64,
                "gib_per_s": (read_stats.gib_per_s * 100.0).round() / 100.0,
                "duration_ms": duration_ms,
            });
            tracing::info!(
                "Hot-swap load '{}': {:.2} GiB payload in {} ms ({:.2} GiB/s), model rebuilt \
                 in {} ms total (kernel target {})",
                prep.model_id,
                read_stats.bytes_read as f64 / (1024.0 * 1024.0 * 1024.0),
                read_stats.duration_ms,
                read_stats.gib_per_s,
                duration_ms,
                prep.target_key,
            );
            // Per-target NVMe KV tier: new store directory (model hash)
            // and layout binding for the swapped-in model.
            let kv_identity = serve_phases::compute_kv_identity(
                &prep.model_id,
                reload_args.block_size,
                &config,
                prep.kv_dtype,
                &prep.layer_dtypes,
            );
            let kv_tier = crate::kv_tier::KvTier::build_from_args(
                &reload_args,
                &crate::kv_tier::KvTierIdentity {
                    model_hash: kv_identity.model_hash,
                    layout_hash: kv_identity.layout_hash,
                },
            )
            .unwrap_or_else(|e| {
                tracing::error!("kv-tier: reopen for swap target failed: {e:#}");
                None
            });
            if let Some(tier) = &kv_tier {
                model.set_kv_evict_guard(Some(tier.evict_guard()));
            }
            Ok(scheduler::SwapOutcome {
                model,
                params: prep.scheduler_params,
                stats,
                kv_tier,
            })
        })
    };
    #[cfg(not(feature = "cuda"))]
    let reload: scheduler::ModelReloader =
        Box::new(|_| anyhow::bail!("hot-swap reload requires the CUDA backend"));

    let scheduler_model = model;
    let scheduler_eos = eos_tokens;
    // EP gate. v1 single-sequence worker protocol required max_batch_size=1
    // because each cmd targeted one slot and the head's per-token broadcast
    // loop had no way to address slot N. v2 adds a per-cmd seq_id preamble
    // (set ATLAS_EP_PROTOCOL=v2) so the worker routes commands by slot_idx
    // and runs decode() per-seq. The head's decode_batch_dispatch EP branch
    // stages each seq's logits row to host between decode() calls so all N
    // rows survive into process_decode_logits — without that, the single-row
    // logits buffer overwrites and N>1 produces garbage.
    let max_batch_size = if world_size > 1 {
        if scheduler_model.ep_protocol_v2() {
            tracing::info!(
                "EP v2 active: honoring max_batch_size={}",
                args.max_batch_size,
            );
            args.max_batch_size
        } else {
            tracing::info!("EP v1 active: forcing max_batch_size=1");
            1
        }
    } else {
        args.max_batch_size
    };
    // `use_speculative` gates the scheduler's `step_mtp` path which already
    // dispatches both MTP and DFlash proposers via the shared `DraftProposer`
    // trait + the `drafts.len() ≥ 4` ladder route to `step_verify_dflash`
    // (scheduler.rs:3013). So `--dflash` enables `use_speculative` too.
    let use_speculative = (args.speculative || args.dflash) && scheduler_model.has_proposer();
    let use_self_spec = args.self_speculative && scheduler_model.has_self_speculative();
    let use_ngram_spec = args.ngram_speculative;
    // For DFlash, force `num_drafts = γ - 1` so the scheduler asks the
    // proposer for γ tokens (DraftProposer::propose semantics: "up to
    // num_drafts" → drafts.len() = γ → routes to step_verify_dflash).
    let num_drafts = if args.dflash {
        args.dflash_gamma.saturating_sub(1).max(1)
    } else {
        args.num_drafts
    };

    if args.dflash {
        tracing::info!(
            "DFlash speculative decoding: ENABLED (γ={}, window={}, drafter installed)",
            args.dflash_gamma,
            if args.dflash_window_size == 0 {
                "full".to_string()
            } else {
                args.dflash_window_size.to_string()
            }
        );
    } else if use_ngram_spec {
        tracing::info!("N-gram speculative decoding: ENABLED (K=2 verify, CPU proposer)");
    } else if use_self_spec {
        tracing::info!(
            "Self-speculative decoding: ENABLED ({num_drafts} drafts/step, layer-skipping)"
        );
    } else if use_speculative {
        tracing::info!("Speculative decoding: ENABLED ({num_drafts} drafts/step)");
    } else if scheduler_model.has_proposer() {
        tracing::info!(
            "MTP proposer available but speculative decoding disabled (use --speculative to enable)"
        );
    }

    let policy: Box<dyn scheduling_policy::SchedulingPolicy> = match args.scheduling_policy.as_str()
    {
        "fifo" => {
            tracing::info!("Scheduling policy: FIFO");
            Box::new(scheduling_policy::FifoPolicy)
        }
        "slai" => {
            tracing::info!(
                "Scheduling policy: SLAI (TBT deadline={}ms)",
                args.tbt_deadline_ms,
            );
            Box::new(scheduling_policy::SlaiPolicy::new(args.tbt_deadline_ms))
        }
        other => anyhow::bail!(
            "Unknown scheduling policy '{}'. Supported: fifo, slai",
            other,
        ),
    };

    // Use prefill_budget (which accounts for SSM no-chunking override) instead of raw CLI arg.
    let max_prefill_tokens = prefill_budget;
    let swap_space_gb = args.swap_space_gb;
    let block_size = args.block_size;

    // ── --high-speed-swap config validation (PCND: required-when-set) ──
    let high_speed_swap_cfg = serve_phases::validate_head_high_speed_swap(
        &args,
        &early_high_speed_swap_cfg,
        swap_space_gb,
    )?;

    let adaptive_sampling = args.adaptive_sampling;
    let session_manager = session_manager::SessionSsmManager::new(600); // 10 min TTL
    // Spontaneous-thinking budget: when the model emits `<think>` without
    // the request having explicitly enabled thinking, this caps how many
    // thinking tokens are allowed before `</think>` is force-emitted. CLI
    // override beats MODEL.toml. Used by the scheduler in place of a
    // previous hard-coded 512 fallback so MODEL.toml can right-size the
    // cap per architecture.
    let scheduler_spontaneous_think_budget = args
        .max_thinking_budget
        .unwrap_or(ptx_set.behavior.max_thinking_budget);
    // Per-model scheduler params — handed over as one unit so a hot-swap
    // load can install a different model's tokens/grammar (Task 6 / 1.3).
    let scheduler_params = scheduler::SchedulerModelParams {
        eos_tokens: scheduler_eos,
        think_end_token,
        think_start_token,
        code_fence_token,
        tool_call_start_token,
        tool_call_end_token,
        grammar_engine,
        spontaneous_think_budget: scheduler_spontaneous_think_budget,
    };
    let scheduler_run_handle = scheduler_handle.clone();
    let scheduler_hotswap = hotswap_ctx.clone();
    let scheduler_model_ready = model_ready.clone();
    std::thread::spawn(move || {
        scheduler::run(
            scheduler_model,
            request_rx,
            scheduler_run_handle,
            scheduler_hotswap,
            scheduler_model_ready,
            reload,
            scheduler_params,
            max_batch_size,
            use_speculative,
            num_drafts,
            policy,
            max_prefill_tokens,
            max_batch_tokens,
            use_self_spec,
            use_ngram_spec,
            swap_space_gb,
            high_speed_swap_cfg,
            block_size,
            adaptive_sampling,
            session_manager,
            boot_kv_tier,
            std::time::Duration::from_millis(args.drain_grace_ms.max(1)),
            args.growth_reserve_pct,
        );
    });

    // ── Task 6 Phase 2.2: boot preseed ──
    // Snapshot the boot model, unload it, snapshot every other registered
    // model (one at a time in the arena), and leave the server IDLE:
    // state `unloaded`, arena empty. Requests then lazy-load (Phase 3).
    #[cfg(feature = "cuda")]
    if lazy_enabled {
        let ctx = hotswap_ctx
            .as_ref()
            .context("preseed requires the weights arena (--weights-arena-gb > 0)")?;
        serve_phases::run_boot_preseed(&args, ctx, &ptx_set.modules).await?;
    }

    // Arena dispatcher (requirements §7): replaces the synchronous lazy
    // swap with queue-pressure-driven switching. Only with lazy
    // multi-model mode — single-model servers keep today's behaviour.
    let arena_dispatcher = match (&hotswap_ctx, lazy_enabled) {
        (Some(ctx), true) => Some(crate::arena_sched::ArenaDispatcher::spawn(
            std::sync::Arc::new(crate::arena_sched::ArenaQueues::new(
                args.max_queue_length,
                args.ewma_alpha,
            )),
            crate::arena_sched::DispatcherConfig {
                decision: crate::arena_sched::DecisionConfig {
                    kappa: args.kappa,
                    min_residency_factor: args.min_residency_factor,
                    max_wait_hard_secs: args.max_wait_hard_s as f64,
                    idle_epsilon: args.idle_epsilon,
                },
                interval: std::time::Duration::from_millis(args.scheduler_interval_ms.max(10)),
                priority_max_wait: std::time::Duration::from_secs(args.priority_max_wait_s),
                nonstream_max_wait: std::time::Duration::from_secs(args.nonstream_max_wait_s),
            },
            ctx.clone(),
        )),
        _ => None,
    };

    // 8. Build app state (model_ready created earlier, next to the
    // scheduler handle, so the hot-swap context shares both).
    let response_store = response_store::ResponseStore::from_env();
    let rate_limiter = rate_limiter::RateLimiter::from_env();
    let conversation_store = conversation_store::ConversationStore::from_env();
    serve_phases::log_response_store_audit(&response_store, &rate_limiter);
    let dump_writer = serve_phases::open_dump_writer(&args);
    let auth = build_auth_config(&args)?;
    let vision_max_pixels = resolve_vision_max_pixels(&args)?;
    if let Some(max_pixels) = vision_max_pixels {
        tracing::info!("Vision max_pixels cap enabled: {}", max_pixels);
    }
    let state = Arc::new(AppState {
        // Created next to the reload closure — the hot-swap handover
        // stores a new ModelRuntime into the same cell.
        runtime: runtime_cell,
        max_seq_len: args.max_seq_len,
        request_tx,
        vision_max_pixels,
        tool_max_tokens: args.tool_max_tokens,
        auto_compact_threshold: args.auto_compact,
        model_ready: model_ready.clone(),
        arena: arena_dispatcher,
        sse_keepalive: std::time::Duration::from_secs(args.sse_keepalive_interval_s.max(1)),
        trust_proxy_headers: args.trust_proxy_headers,
        serve_args: std::sync::Arc::new(args.clone()),
        nonstream_max_wait: std::time::Duration::from_secs(args.nonstream_max_wait_s.max(1)),
        request_timeout: args.request_timeout,
        disable_thinking: args.disable_thinking,
        default_chat_template_kwargs: args
            .default_chat_template_kwargs
            .as_ref()
            .and_then(|s| crate::openai::ChatTemplateKwargs::from_json(s)),
        response_store,
        rate_limiter,
        conversation_store,
        dump_writer,
        auth,
        hotswap: hotswap_ctx,
        transcribe_models,
        image_models,
    });

    serve_phases::log_behavior_audit(&args, &ptx_set);

    // 9-11. Build router + start HTTP server (extracted: serve_router.rs).
    crate::main_modules::serve_router::build_and_serve(state, model_ready, &args.bind, args.port)
        .await
}

/// Resolve `--require-auth` / `--auth-tokens-file` / `--auth-token` into an
/// optional `AuthConfig`. Validates at startup so misconfigurations fail
/// loudly instead of letting an unauthenticated server run silently.
fn build_auth_config(args: &cli::ServeArgs) -> Result<Option<Arc<crate::auth::AuthConfig>>> {
    if !args.require_auth {
        if args.auth_tokens_file.is_some() || args.auth_token.is_some() {
            tracing::warn!(
                "--auth-tokens-file / --auth-token supplied without --require-auth; \
                 tokens are loaded but the auth gate is OFF. Pass --require-auth to enforce."
            );
        }
        return Ok(None);
    }
    let cfg = match (&args.auth_tokens_file, &args.auth_token) {
        (Some(path), None) => crate::auth::AuthConfig::from_file(path)?,
        (None, Some(tok)) => {
            tracing::warn!(
                "--auth-token sets the bearer token via the command line; the value \
                 is visible to other local users via `ps`/`/proc/<pid>/cmdline`. \
                 Use --auth-tokens-file with permissions 0600 in production."
            );
            crate::auth::AuthConfig::from_inline(tok)?
        }
        (None, None) => {
            return Err(anyhow::anyhow!(
                "--require-auth was set but neither --auth-tokens-file nor \
                 --auth-token was supplied. Pick one (a tokens file is preferred)."
            ));
        }
        (Some(_), Some(_)) => unreachable!("clap conflicts_with should have rejected this"),
    };
    tracing::info!(
        "auth: require_auth=ON ({} bearer token{} loaded)",
        cfg.token_count(),
        if cfg.token_count() == 1 { "" } else { "s" },
    );
    Ok(Some(Arc::new(cfg)))
}

fn resolve_vision_max_pixels(args: &cli::ServeArgs) -> Result<Option<usize>> {
    if args.vision_max_pixels > 0 {
        return Ok(Some(args.vision_max_pixels));
    }
    let Some(raw) = std::env::var("ATLAS_VISION_MAX_PIXELS").ok() else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "0" {
        return Ok(None);
    }
    let parsed = trimmed.parse::<usize>().with_context(|| {
        format!("ATLAS_VISION_MAX_PIXELS must be a positive integer, got {raw:?}")
    })?;
    Ok((parsed > 0).then_some(parsed))
}

/// QV1 (2026-05-26): canonicalize the model's declared quantization to
/// one of `"fp8"`, `"nvfp4"`, `"bf16"`, or `"unknown"`. Reads
/// `quantization_config.quant_method`/`quant_algo`/`format` and applies
/// the heuristics needed across ModelOpt + compressed-tensors checkpoints.
/// Returns `"bf16"` when no quant config is present (the HF default for
/// unquantized BF16 weights).
pub(crate) fn canonicalize_model_quant(config: &atlas_core::config::ModelConfig) -> String {
    let Some(qc) = config.quantization_config.as_ref() else {
        return "bf16".to_string();
    };
    let method = qc.quant_method.to_ascii_lowercase();
    let algo = qc.quant_algo.to_ascii_lowercase();
    let fmt = qc.format.to_ascii_lowercase();
    // NVFP4 detection — explicit algo OR a format string containing "nvfp4"
    // (compressed-tensors: "nvfp4-pack-quantized" et al).
    //
    // ModelOpt "MIXED_PRECISION" (e.g. Nemotron-Super-120B-A12B-NVFP4,
    // Qwen3.6-35B-A3B-NVFP4) canonicalizes to "nvfp4": it is nvfp4-base
    // plus a few FP8 modules. Dispatch is per-MODULE and tensor-aware, NOT
    // by this string — the loader probes `*.weight_scale` presence and
    // dequants FP8→BF16 (weight_loader/nemotron.rs:78-108, quant_helpers.rs
    // dense_auto), and the lm_head MIXED_PRECISION path is already handled
    // (factory/build.rs:144). The nvfp4 kernel bundle also carries native
    // FP8/BF16 paths (see quant_pair_compatible: nvfp4↔fp8, nvfp4↔bf16).
    // So routing MIXED_PRECISION to the nvfp4 bundle is correct and cannot
    // silently mis-route an FP8 module (it would fault at load, not corrupt).
    if algo == "nvfp4" || algo == "mixed_precision" || fmt.contains("nvfp4") {
        return "nvfp4".into();
    }
    // FP8 detection — explicit algo OR method/format containing "fp8", OR
    // compressed-tensors' `float-quantized` block-FP8 (e.g.
    // Hcompany/Holo-3.1-*-FP8: `quant_method="compressed-tensors"`,
    // `format="float-quantized"`, num_bits=8). That format string contains no
    // literal "fp8", so match it explicitly. Canonicalizing to "fp8" lets the
    // nvfp4 kernel bundle accept it (quant_pair_compatible: nvfp4↔fp8) — the
    // loader detects the FP8E4M3 weight dtype as Fp8Dequanted and requants
    // FP8→BF16→NVFP4 from the 2D `.weight_scale` (nvfp4_detect.rs).
    if algo == "fp8" || method.contains("fp8") || fmt.contains("fp8") || fmt.contains("float-quant")
    {
        return "fp8".into();
    }
    // compressed-tensors with no FP8/NVFP4 marker is usually GPTQ/AWQ —
    // we don't currently dispatch those on Atlas; report verbatim so
    // the bail message is precise.
    if !algo.is_empty() {
        return algo;
    }
    if !method.is_empty() {
        return method;
    }
    "unknown".into()
}

/// QV1 helper: short debug string of where the quant declaration came
/// from, used in the bail message so the operator can locate the
/// mis-declared field quickly.
fn describe_quant_source(config: &atlas_core::config::ModelConfig) -> String {
    match config.quantization_config.as_ref() {
        Some(qc) => format!(
            "quant_method={:?}, quant_algo={:?}, format={:?}",
            qc.quant_method, qc.quant_algo, qc.format
        ),
        None => "no quantization_config in config.json".into(),
    }
}

/// QV1: returns `true` iff the kernel target's declared quant string is
/// known to handle the model's canonicalized quant.
///
/// The current Atlas build emits one bundle per (hw, model) regardless
/// of how many quant variants it dispatches at runtime: the bundle
/// label is whichever `ATLAS_TARGET_QUANT` value the build script
/// happened to record first (today: always `"nvfp4"`). Each bundle
/// nonetheless contains native FP8 / native NVFP4 / BF16-dequant code
/// paths for the same model. This compat table makes that explicit.
///
/// When new quants appear (e.g. FP4 E2M1 on a future SM), add the new
/// entry here AND the dispatch path in the weight loader. The
/// canonical home for this list will eventually be MODEL.toml
/// `[kernel].supported_quants` — until then, hardcode keeps the
/// fail-fast working without a build-time plumb-through.
pub(crate) fn quant_pair_compatible(kernel_quant: &str, model_quant: &str) -> bool {
    if kernel_quant == model_quant {
        return true;
    }
    matches!(
        (kernel_quant, model_quant),
        // The NVFP4-labeled bundle today carries native FP8 paths
        // (FP8 fused MoE batch1/2/3, w8a16_gemv decode, FP8 prefill).
        ("nvfp4", "fp8") |
        // The NVFP4 bundle also handles unquantized BF16 inputs via
        // runtime dequant → quantize. Slow but correct.
        ("nvfp4", "bf16") |
        // BF16 reference bundle handles any quant by dequant on load.
        ("bf16", "fp8") |
        ("bf16", "nvfp4")
    )
}

/// MEM-7 measurement: MemAvailable delta across `build_model` minus the
/// KV pool the budget reserved — the model's real runtime footprint
/// (derived buffers, SSM pools, CUDA workspaces). The plausibility clamp
/// in `swap_costs::runtime_reserve_or` guards against driver-pool noise.
fn observe_runtime_reserve_delta(model_id: &str, avail_before: Option<usize>) {
    let (Some(before), Some(after)) = (
        avail_before,
        spark_runtime::meminfo::read().map(|m| m.mem_available),
    ) else {
        return;
    };
    let delta = before.saturating_sub(after) as u64;
    let kv_reserved = spark_runtime::kv_stats::snapshot()
        .map(|s| s.reserved_bytes)
        .unwrap_or(0);
    crate::swap_costs::observe_runtime_reserve(model_id, delta.saturating_sub(kv_reserved as u64));
}

#[cfg(test)]
mod qv1_tests {
    use super::*;

    // canonicalize_model_quant is exercised via integration through
    // the server boot path; unit-testing it requires building
    // ModelConfig which has no `Default` impl (it's intentionally
    // bound to a loaded model). The pair-compatibility table is a
    // pure function and worth a unit test.

    #[test]
    fn compat_self_pair() {
        assert!(quant_pair_compatible("nvfp4", "nvfp4"));
        assert!(quant_pair_compatible("fp8", "fp8"));
        assert!(quant_pair_compatible("bf16", "bf16"));
    }

    #[test]
    fn compat_nvfp4_handles_fp8_and_bf16() {
        assert!(quant_pair_compatible("nvfp4", "fp8"));
        assert!(quant_pair_compatible("nvfp4", "bf16"));
    }

    #[test]
    fn incompat_unknown_rejected() {
        assert!(!quant_pair_compatible("nvfp4", "gptq-4bit"));
        assert!(!quant_pair_compatible("fp8", "nvfp4"));
    }
}

// SPDX-License-Identifier: AGPL-3.0-only

//! Small-text-model sidecars for embeddings and rerank (requirements E4,
//! MOD-3/4, WAIT-6, MEM-6).
//!
//! Embed/rerank models at or below `--sidecar-small-max-gib` load once on
//! first request and stay RESIDENT beside the arena model — a short
//! prefill-only request must never pay (or wait for) an arena swap.
//! Larger embed/rerank models keep the arena path through the pressure
//! scheduler.
//!
//! Each sidecar owns a dedicated worker thread (the `Model` contract is
//! single-thread exclusive access, same as the GPU scheduler) with its
//! own CUDA backend handle on the shared process context, a
//! `NoPrefixCaching` KV setup with a small fixed pool, and the model's
//! own tokenizer — prompts for sidecar models must never be encoded with
//! the arena model's tokenizer. The load runs under `UntrackedAllocs` so
//! a concurrent arena unload cannot free the sidecar's buffers (same
//! guard as Whisper/Qwen-Image), and the combined resident footprint is
//! published into the KV budget's sidecar reserve (MEM-6).

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::http::StatusCode;
use axum::response::Response;

use crate::main_modules::hotswap::ModelCategory;
use crate::main_modules::{AppState, serve_phases};

/// Hard cap on a sidecar prompt (single-shot prefill; buffers and the
/// fixed KV pool are sized for this).
pub const SIDECAR_MAX_TOKENS: usize = 8192;
/// Fixed KV pool per sidecar.
const SIDECAR_KV_POOL_BYTES: usize = 1 << 30;
/// Working margin per sidecar for activations/workspaces (reserve math).
const SIDECAR_WORKING_BYTES: usize = 1 << 30;

enum JobKind {
    /// Pooled last-token hidden state (embeddings).
    Embed,
    /// Raw first-token logits at the given ids (rerank yes/no).
    Probe(Vec<u32>),
}

struct Job {
    tokens: Vec<u32>,
    image_pixels: Vec<(Vec<f32>, usize, usize)>,
    kind: JobKind,
    reply: tokio::sync::oneshot::Sender<Result<Vec<f32>>>,
}

pub struct TextSidecar {
    pub model_id: String,
    pub tokenizer: Arc<crate::tokenizer::ChatTokenizer>,
    pub vision_config: Option<atlas_core::config::VisionConfig>,
    pub weights_bytes: u64,
    jobs: tokio::sync::mpsc::Sender<Job>,
}

impl TextSidecar {
    async fn run(
        &self,
        job_kind: JobKind,
        tokens: Vec<u32>,
        pixels: ImagePixels,
    ) -> Result<Vec<f32>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.jobs
            .send(Job {
                tokens,
                image_pixels: pixels,
                kind: job_kind,
                reply: tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("sidecar worker gone"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("sidecar reply dropped"))?
    }

    /// Embeddings: pooled last-token hidden state.
    pub async fn embed(&self, tokens: Vec<u32>, pixels: ImagePixels) -> Result<Vec<f32>> {
        self.run(JobKind::Embed, tokens, pixels).await
    }

    /// Rerank: raw logits at `probe_ids` after the prefill.
    pub async fn probe(
        &self,
        tokens: Vec<u32>,
        pixels: ImagePixels,
        probe_ids: Vec<u32>,
    ) -> Result<Vec<f32>> {
        self.run(JobKind::Probe(probe_ids), tokens, pixels).await
    }
}

type ImagePixels = Vec<(Vec<f32>, usize, usize)>;

struct Registry {
    loaded: HashMap<String, Arc<TextSidecar>>,
}

static REGISTRY: tokio::sync::Mutex<Option<Registry>> = tokio::sync::Mutex::const_new(None);

fn checkpoint_bytes(model_dir: &std::path::Path) -> u64 {
    std::fs::read_dir(model_dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| {
                    e.path()
                        .extension()
                        .and_then(|x| x.to_str())
                        .is_some_and(|x| x == "safetensors")
                })
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// MOD-3 eligibility + get-or-load. `None` = not a sidecar case (take the
/// arena path); `Some(Err)` = eligible but the load failed (503).
pub async fn maybe_for(
    state: &Arc<AppState>,
    model_id: &str,
    category: ModelCategory,
) -> Option<std::result::Result<Arc<TextSidecar>, Response>> {
    if !matches!(category, ModelCategory::Embed | ModelCategory::Rerank) {
        return None;
    }
    let ctx = state.hotswap.as_ref()?;
    let reg = ctx.registered(model_id)?;
    if reg.category != category {
        return None; // category 400 happens on the arena path
    }
    let limit = (state.serve_args.sidecar_small_max_gib * 1024.0 * 1024.0 * 1024.0) as u64;
    if limit == 0 {
        return None;
    }
    let model_dir =
        crate::model_resolver::resolve_model_dir(model_id, state.serve_args.cache_dir.as_deref())
            .ok()?;
    let bytes = checkpoint_bytes(&model_dir);
    if bytes == 0 || bytes > limit {
        return None; // MOD-4: larger models go through the arena queue
    }

    let mut guard = REGISTRY.lock().await;
    let registry = guard.get_or_insert_with(|| Registry {
        loaded: HashMap::new(),
    });
    if let Some(s) = registry.loaded.get(model_id) {
        return Some(Ok(Arc::clone(s)));
    }
    let args = Arc::clone(&state.serve_args);
    let id = model_id.to_string();
    let loaded = tokio::task::spawn_blocking(move || load_sidecar(&args, &id, &model_dir, bytes))
        .await
        .map_err(|e| anyhow::anyhow!("sidecar load task panicked: {e}"))
        .and_then(|r| r);
    match loaded {
        Ok(s) => {
            registry.loaded.insert(model_id.to_string(), Arc::clone(&s));
            publish_reserve(registry);
            Some(Ok(s))
        }
        Err(e) => {
            tracing::error!("text-sidecar '{model_id}': load failed: {e:#}");
            let body = serde_json::json!({
                "error": {
                    "message": format!("sidecar load for '{model_id}' failed: {e:#}"),
                    "type": "server_error",
                    "param": "model",
                    "code": "sidecar_load_failed",
                },
            });
            use axum::response::IntoResponse;
            Some(Err(
                (StatusCode::SERVICE_UNAVAILABLE, axum::Json(body)).into_response()
            ))
        }
    }
}

/// MEM-6: resident footprint of every loaded text sidecar, folded into
/// the KV budget's sidecar reserve (additive to the image pipeline's).
fn publish_reserve(registry: &Registry) {
    let bytes: u64 = registry
        .loaded
        .values()
        .map(|s| s.weights_bytes + (SIDECAR_KV_POOL_BYTES + SIDECAR_WORKING_BYTES) as u64)
        .sum();
    spark_runtime::gpu::set_text_sidecar_reserve_bytes(bytes as usize);
    tracing::info!(
        reserve_gib = bytes as f64 / (1u64 << 30) as f64,
        sidecars = registry.loaded.len(),
        "text-sidecar reserve published into the KV budget (MEM-6)"
    );
}

#[cfg(feature = "cuda")]
fn load_sidecar(
    args: &crate::cli::ServeArgs,
    model_id: &str,
    model_dir: &std::path::Path,
    checkpoint_bytes: u64,
) -> Result<Arc<TextSidecar>> {
    let t0 = std::time::Instant::now();
    // Keep the sidecar's allocations out of a concurrent hot-swap build's
    // watermark set (they would be freed at its unload).
    let _untracked = spark_runtime::cuda_backend::UntrackedAllocs::enter();

    // Memory preflight (no swap — pinned pages): weights + pool + margin.
    let need = checkpoint_bytes as usize + SIDECAR_KV_POOL_BYTES + SIDECAR_WORKING_BYTES;
    if let Some(mem) = spark_runtime::meminfo::read()
        && mem.mem_available < need + (4 << 30)
    {
        anyhow::bail!(
            "not enough free memory for the sidecar: need ~{:.1} GiB + 4 GiB margin, \
             {:.1} GiB available",
            need as f64 / (1u64 << 30) as f64,
            mem.mem_available as f64 / (1u64 << 30) as f64,
        );
    }

    let (mut config, _json) = serve_phases::load_model_config(model_dir)?;
    serve_phases::merge_sidecar_quant_config(model_dir, &mut config);
    let ptx_set = atlas_kernels::ptx_for_model_id(model_id, &config.model_type, config.hidden_size)
        .with_context(|| format!("no kernel target for sidecar '{model_id}'"))?;
    let model_quant = crate::main_modules::serve::canonicalize_model_quant(&config);
    if !crate::main_modules::serve::quant_pair_compatible(ptx_set.target.quant, &model_quant) {
        anyhow::bail!(
            "kernel/model quant mismatch for sidecar '{model_id}': {} vs {model_quant}",
            ptx_set.target.quant
        );
    }
    config.fp8_kv_calibration_tokens = ptx_set.behavior.fp8_kv_calibration_tokens;

    let key = serve_phases::target_key(&ptx_set.target);
    let backend =
        spark_runtime::cuda_backend::AtlasCudaBackend::new_for_target(&key, &ptx_set.modules)?;
    let gpu: &dyn spark_runtime::gpu::GpuBackend = &backend;
    gpu.bind_to_thread()?;

    let store = serve_phases::load_weight_store(args, &config, model_dir, gpu, 0, 1, 2 << 30)?;
    serve_phases::auto_detect_weight_prefix(&store, &mut config);
    serve_phases::cap_vocab_size_to_tokenizer(model_dir, &mut config);
    let kv =
        serve_phases::resolve_kv_cache_config(args, &config, ptx_set.behavior.default_kv_dtype)?;

    let eos_tokens = serve_phases::load_eos_tokens(model_dir, &config);
    let tokenizer = crate::tokenizer::ChatTokenizer::from_model_dir(
        model_dir,
        eos_tokens[0],
        false,
        &config.model_type,
        Some(std::path::Path::new(".")),
    )?;
    let vision_config = config.vision.clone();

    let model = spark_model::factory::build_model(
        config,
        &store,
        Box::new(backend),
        SIDECAR_MAX_TOKENS, // max_batch_tokens: single-shot prefill window
        args.block_size,
        SIDECAR_MAX_TOKENS,
        1, // one sequence at a time
        spark_model::layers::MtpQuantization::Bf16,
        false,
        Box::new(spark_runtime::prefix_cache::NoPrefixCaching),
        0,
        None,
        false,
        1,
        kv.kv_dtype,
        256 << 20, // inference reserve (unused on the override path)
        args.gpu_memory_utilization,
        0, // no Marconi slots
        kv.layer_dtypes,
        0,
        None,
        None,
        Some(SIDECAR_KV_POOL_BYTES),
    )?;

    let (jobs_tx, mut jobs_rx) = tokio::sync::mpsc::channel::<Job>(16);
    let worker_id = model_id.to_string();
    std::thread::Builder::new()
        .name(format!("atlas-sidecar-{}", worker_id.replace('/', "_")))
        .spawn(move || {
            if let Err(e) = model.bind_gpu_to_thread() {
                tracing::error!("text-sidecar '{worker_id}': bind failed: {e:#}");
                return;
            }
            while let Some(job) = jobs_rx.blocking_recv() {
                let result = run_job(model.as_ref(), &job);
                let _ = job.reply.send(result);
            }
            tracing::info!("text-sidecar '{worker_id}': worker stopped");
        })
        .context("spawn sidecar worker")?;

    tracing::info!(
        "text-sidecar '{model_id}': resident ({:.2} GiB checkpoint, {} ms load)",
        checkpoint_bytes as f64 / (1u64 << 30) as f64,
        t0.elapsed().as_millis(),
    );
    Ok(Arc::new(TextSidecar {
        model_id: model_id.to_string(),
        tokenizer: Arc::new(tokenizer),
        vision_config,
        weights_bytes: checkpoint_bytes,
        jobs: jobs_tx,
    }))
}

#[cfg(not(feature = "cuda"))]
fn load_sidecar(
    _args: &crate::cli::ServeArgs,
    _model_id: &str,
    _model_dir: &std::path::Path,
    _checkpoint_bytes: u64,
) -> Result<Arc<TextSidecar>> {
    anyhow::bail!("text sidecars require the CUDA backend")
}

#[cfg(feature = "cuda")]
fn run_job(model: &dyn spark_model::traits::Model, job: &Job) -> Result<Vec<f32>> {
    anyhow::ensure!(
        job.tokens.len() <= SIDECAR_MAX_TOKENS,
        "sidecar prompt of {} tokens exceeds the {SIDECAR_MAX_TOKENS}-token sidecar window",
        job.tokens.len(),
    );
    if !job.image_pixels.is_empty() {
        model.prepare_vision_embed(&job.image_pixels)?;
    }
    let mut seq = model.alloc_sequence()?;
    let stream = model.default_stream();
    let out = (|| -> Result<Vec<f32>> {
        let logits = model.prefill(&job.tokens, &mut seq, stream)?;
        model.synchronize(stream)?;
        match &job.kind {
            JobKind::Embed => model.pooled_hidden_f32(),
            JobKind::Probe(ids) => {
                let vocab = model.vocab_size();
                let fp32 = model.logits_ptr_is_fp32(logits);
                let elem = if fp32 { 4 } else { 2 };
                let mut buf = vec![0u8; vocab * elem];
                model.copy_logits_to_host(logits, &mut buf)?;
                Ok(ids
                    .iter()
                    .map(|&id| {
                        let i = id as usize;
                        if i >= vocab {
                            return f32::NAN;
                        }
                        if fp32 {
                            f32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap())
                        } else {
                            let bits =
                                u16::from_le_bytes(buf[i * 2..i * 2 + 2].try_into().unwrap());
                            f32::from_bits((bits as u32) << 16)
                        }
                    })
                    .collect())
            }
        }
    })();
    if let Err(e) = model.free_sequence(&mut seq) {
        tracing::error!("text-sidecar: free_sequence: {e:#}");
    }
    out
}

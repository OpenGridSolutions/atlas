// SPDX-License-Identifier: AGPL-3.0-only

//! `POST /v1/audio/transcriptions` — Whisper speech-to-text (multiclass
//! Plan.md W.3). Replaces the 501 stub.
//!
//! Whisper runs as a RESIDENT SIDECAR (gate finding W.0): loaded once on
//! first request via its own kernel target + backend, never entering the
//! scheduler / hot-swap arena — transcription does not evict the active
//! chat model. FP32 checkpoint → BF16; no quantization.

use std::sync::Arc;

use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::main_modules::app_state::AppState;
use spark_model::audio_preprocess;
use spark_model::whisper::Whisper;

use super::compact::openai_error_response;

const WHISPER_ID: &str = "openai/whisper-large-v3";

struct Sidecar {
    whisper: tokio::sync::Mutex<Whisper>,
    tokenizer: tokenizers::Tokenizer,
    sot: u32,
    eot: u32,
    task_transcribe: u32,
    no_timestamps: u32,
    suppress: Vec<u32>,
    begin_suppress: Vec<u32>,
    lang_to_id: std::collections::HashMap<String, u32>,
}

static SIDECAR: tokio::sync::OnceCell<Arc<Sidecar>> = tokio::sync::OnceCell::const_new();

/// Resident BF16 weight bytes of the sidecar, set once at load
/// (Σ tensor elements × 2 — the FP32 store is converted 1:1 to BF16
/// buffers). 0 = sidecar not loaded. Read by /v1/stats (Phase S).
static SIDECAR_WEIGHTS_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// /v1/stats residency probe: `Some(weights_bytes)` when the whisper
/// sidecar is resident, `None` before its first request.
pub fn sidecar_stats() -> Option<usize> {
    match SIDECAR_WEIGHTS_BYTES.load(std::sync::atomic::Ordering::Relaxed) {
        0 => None,
        b => Some(b),
    }
}

/// Build the loader-compatible view of the whisper snapshot: the HF repo
/// ships `model.safetensors` (F16 — unsupported dtype) plus FP32 shards
/// under a NON-standard index name (`model.safetensors.index.fp32.json`).
/// Shadow dir = symlinked FP32 shards + the fp32 index renamed to the
/// standard `model.safetensors.index.json`.
fn shadow_dir(model_dir: &std::path::Path) -> anyhow::Result<std::path::PathBuf> {
    let shadow = model_dir.join("atlas-fp32-view");
    std::fs::create_dir_all(&shadow)?;
    let idx = model_dir.join("model.safetensors.index.fp32.json");
    let idx_dst = shadow.join("model.safetensors.index.json");
    if !idx_dst.exists() {
        std::fs::copy(&idx, &idx_dst)?;
    }
    let map: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&idx)?)?;
    let files: std::collections::HashSet<String> = map["weight_map"]
        .as_object()
        .map(|m| m.values().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    for f in files {
        let dst = shadow.join(&f);
        if !dst.exists() {
            let src = std::fs::canonicalize(model_dir.join(&f))?;
            std::os::unix::fs::symlink(src, &dst)?;
        }
    }
    Ok(shadow)
}

async fn sidecar() -> anyhow::Result<Arc<Sidecar>> {
    SIDECAR
        .get_or_try_init(|| async {
            tokio::task::spawn_blocking(|| -> anyhow::Result<Arc<Sidecar>> {
                let t0 = std::time::Instant::now();
                let model_dir = crate::model_resolver::resolve_model_dir(WHISPER_ID, None)?;
                let set = atlas_kernels::ptx_for_config("whisper", 1280).ok_or_else(|| {
                    anyhow::anyhow!("no compiled kernel target for whisper (hidden 1280)")
                })?;
                let key = format!(
                    "{}/{}/{}",
                    set.target.arch, set.target.model, set.target.quant
                );
                let backend =
                    spark_runtime::cuda_backend::AtlasCudaBackend::new_for_target(&key, &set.modules)?;
                let gpu: &dyn spark_runtime::gpu::GpuBackend = &backend;
                gpu.bind_to_thread()?;

                let shadow = shadow_dir(&model_dir)?;
                use spark_runtime::weights::WeightLoader;
                let store = spark_runtime::fast_weights::FastSafetensorsLoader::new()
                    .load(&shadow, gpu, 2 * 1024 * 1024 * 1024)?;
                let whisper = Whisper::load(&store, Box::new(backend))?;
                // Resident footprint for /v1/stats: every checkpoint tensor
                // lives on as a BF16 buffer (2 bytes/element).
                let bf16_bytes: usize = store
                    .names()
                    .filter_map(|n| store.get(n).ok())
                    .map(|w| w.shape.iter().product::<usize>() * 2)
                    .sum();
                SIDECAR_WEIGHTS_BYTES
                    .store(bf16_bytes, std::sync::atomic::Ordering::Relaxed);
                // The FP32 store (~6.2 GiB device) is fully converted into
                // the sidecar's BF16 buffers — release it.
                let mut freed = 0usize;
                for name in store.names() {
                    if let Ok(w) = store.get(name)
                        && whisper_free_ok(&whisper, w.ptr)
                    {
                        freed += 1;
                    }
                }
                tracing::info!(
                    "Whisper sidecar ready in {:.1}s (fp32 store released: {freed} tensors)",
                    t0.elapsed().as_secs_f64()
                );

                let tokenizer =
                    tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
                        .map_err(|e| anyhow::anyhow!("whisper tokenizer: {e}"))?;
                let gc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
                    model_dir.join("generation_config.json"),
                )?)?;
                let u = |k: &str| gc[k].as_u64().map(|v| v as u32);
                let list = |k: &str| -> Vec<u32> {
                    gc[k].as_array()
                        .map(|a| a.iter().filter_map(|v| v.as_u64().map(|x| x as u32)).collect())
                        .unwrap_or_default()
                };
                let lang_to_id = gc["lang_to_id"]
                    .as_object()
                    .map(|m| {
                        m.iter()
                            .filter_map(|(k, v)| {
                                let code =
                                    k.trim_start_matches("<|").trim_end_matches("|>").to_string();
                                v.as_u64().map(|id| (code, id as u32))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                Ok(Arc::new(Sidecar {
                    whisper: tokio::sync::Mutex::new(whisper),
                    tokenizer,
                    sot: u("decoder_start_token_id").unwrap_or(50_258),
                    eot: u("eos_token_id").unwrap_or(50_257),
                    task_transcribe: gc["task_to_id"]["transcribe"]
                        .as_u64()
                        .map(|v| v as u32)
                        .unwrap_or(50_360),
                    no_timestamps: u("no_timestamps_token_id").unwrap_or(50_364),
                    suppress: list("suppress_tokens"),
                    begin_suppress: list("begin_suppress_tokens"),
                    lang_to_id,
                }))
            })
            .await?
        })
        .await
        .cloned()
}

/// Free a store tensor's device allocation via the sidecar's backend.
fn whisper_free_ok(w: &Whisper, ptr: spark_runtime::gpu::DevicePtr) -> bool {
    w.gpu_backend().free(ptr).is_ok()
}

pub async fn transcriptions(
    State(_state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    // ── multipart: file (required), model, language, response_format ──
    let mut file: Option<Vec<u8>> = None;
    let mut model = String::new();
    let mut language: Option<String> = None;
    let mut response_format = "json".to_string();
    loop {
        match multipart.next_field().await {
            Ok(Some(field)) => {
                let name = field.name().unwrap_or("").to_string();
                match name.as_str() {
                    "file" => match field.bytes().await {
                        Ok(b) => file = Some(b.to_vec()),
                        Err(e) => {
                            return openai_error_response(
                                StatusCode::BAD_REQUEST,
                                format!("reading file field: {e}"),
                            );
                        }
                    },
                    "model" => model = field.text().await.unwrap_or_default(),
                    "language" => language = field.text().await.ok(),
                    "response_format" => {
                        response_format = field.text().await.unwrap_or_else(|_| "json".into())
                    }
                    _ => {} // prompt/temperature etc.: accepted and ignored (v1)
                }
            }
            Ok(None) => break,
            Err(e) => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("multipart parse error: {e}"),
                );
            }
        }
    }
    let Some(bytes) = file else {
        return openai_error_response(StatusCode::BAD_REQUEST, "missing 'file' field".into());
    };
    // Category check (Plan.md Phase K): the REAL id of any model listed
    // in ATLAS_PRESEED_TRANSCRIBE is accepted, plus the OpenAI-habitual
    // "whisper-1" / short aliases (clients hardcode those).
    let alias_ok = matches!(model.as_str(), "" | "whisper-1" | "whisper-large-v3" | WHISPER_ID);
    let listed_ok = _state.transcribe_models.contains(&model);
    if !alias_ok && !listed_ok {
        // A model registered for another endpoint gets the explaining 400.
        if let Some(reg) = _state
            .hotswap
            .as_ref()
            .and_then(|ctx| ctx.registered(&model).cloned())
        {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "model '{model}' is categorized as {} and not available on \
                     /v1/audio/transcriptions; use {} instead",
                    reg.category.as_str(),
                    reg.category.endpoint(),
                ),
            );
        }
        return openai_error_response(
            StatusCode::NOT_FOUND,
            format!(
                "model '{model}' is not a transcription model; configured transcribe \
                 models: {:?} (alias 'whisper-1' also accepted)",
                _state.transcribe_models,
            ),
        );
    }
    if !matches!(response_format.as_str(), "json" | "text" | "verbose_json") {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("response_format '{response_format}' not supported (json|text|verbose_json)"),
        );
    }

    let sc = match sidecar().await {
        Ok(s) => s,
        Err(e) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("whisper sidecar init failed: {e:#}"),
            );
        }
    };
    let lang_code = language.unwrap_or_else(|| "en".into());
    let Some(&lang_id) = sc.lang_to_id.get(lang_code.as_str()) else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("unknown language '{lang_code}'"),
        );
    };
    let prompt = vec![sc.sot, lang_id, sc.task_transcribe, sc.no_timestamps];

    // ── decode + mel on a blocking thread, then chunk-serial GPU work ──
    let sc2 = sc.clone();
    let text = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
        let samples = audio_preprocess::decode_audio(bytes)?;
        let secs = samples.len() as f64 / audio_preprocess::SAMPLE_RATE as f64;
        let chunks = audio_preprocess::log_mel_chunks(&samples);
        tracing::info!(
            "Transcribing {secs:.1}s audio ({} chunk(s), lang {lang_code})",
            chunks.len()
        );
        let mut whisper = sc2.whisper.blocking_lock();
        whisper.gpu_backend().bind_to_thread()?;
        let mut text = String::new();
        for (i, mel) in chunks.iter().enumerate() {
            let tokens = whisper.transcribe_chunk(
                mel,
                &prompt,
                &sc2.suppress,
                &sc2.begin_suppress,
                sc2.eot,
                440,
            )?;
            let piece = sc2
                .tokenizer
                .decode(&tokens, true)
                .map_err(|e| anyhow::anyhow!("token decode: {e}"))?;
            tracing::info!("Whisper chunk {}/{}: {} tokens", i + 1, chunks.len(), tokens.len());
            if !text.is_empty() && !piece.is_empty() {
                text.push(' ');
            }
            text.push_str(piece.trim());
        }
        Ok(text)
    })
    .await;
    let text = match text {
        Ok(Ok(t)) => t,
        Ok(Err(e)) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("transcription failed: {e:#}"),
            );
        }
        Err(e) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("transcription task panicked: {e}"),
            );
        }
    };

    match response_format.as_str() {
        "text" => text.into_response(),
        _ => axum::Json(serde_json::json!({ "text": text })).into_response(),
    }
}

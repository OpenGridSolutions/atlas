// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::doc_overindented_list_items)]

//! Atlas Spark — pure Rust LLM inference server.
//!
//! Startup sequence:
//! 1. Parse CLI args
//! 2. Load model config
//! 3. Initialize GPU backend (AtlasCudaBackend)
//! 4. Load model weights (SafetensorsLoader)
//! 5. Build model via factory
//! 6. Load tokenizer
//! 7. Spawn scheduler thread
//! 8. Start axum HTTP server

mod adaptive_sampler;
mod anthropic;
mod api;
mod auth;
mod citation;
mod cli;
mod conversation_store;
pub mod grammar;
mod halluc_probe;
mod hint_injector;
mod llmlingua;
mod lookback_lens;
mod loop_detector;
mod loop_simhash;
mod lqer;
mod main_modules;
mod memdebug;
pub mod metrics;
mod model_resolver;
mod moe_quality;
mod ngram;
mod openai;
mod rate_limiter;
pub mod reasoning_parser;
mod refusal;
mod request_dumper;
mod response_store;
mod retrieval_heads;
mod scheduler;
mod scheduling_policy;
mod session_manager;
mod symbol_trie;
mod tokenizer;
mod tool_arg_dedup;
pub mod tool_parser;
mod tool_rag;
mod tscg;

use anyhow::Result;
use clap::Parser;

use crate::cli::{Cli, Command};
use crate::main_modules::serve;

pub(crate) use crate::main_modules::{AppState, ModelRuntime};

/// Re-export for convenience in api.rs / anthropic.rs.
pub type ModelBehavior = atlas_kernels::ModelBehavior;

/// Todo.md Phase 1.1: file-based logging next to stdout. Two per-process
/// sinks under `ATLAS_LOG_DIR` (default `logs/`, same convention as the
/// serve.sh tee): `atlas-<ts>.log` mirrors the stdout filter, and
/// `critical-<ts>.log` captures WARN+ERROR from EVERY target — when
/// something goes wrong, that one file is the place to look. Disable
/// with `ATLAS_FILE_LOG=0`. Retention is per-process files (rotate by
/// restart); prune old files externally if disk matters.
fn init_logging() {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::{EnvFilter, Layer, fmt};

    let env_filter =
        || EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let stdout = fmt::layer().with_filter(env_filter());

    let mut file_layers = Vec::new();
    if std::env::var("ATLAS_FILE_LOG").as_deref() != Ok("0") {
        let dir = spark_runtime::crash::log_dir();
        if std::fs::create_dir_all(&dir).is_ok() {
            let ts = spark_runtime::crash::timestamp_compact();
            let open = |name: String| {
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(dir.join(name))
                    .ok()
                    .map(std::sync::Mutex::new)
            };
            if let Some(w) = open(format!("atlas-{ts}.log")) {
                file_layers.push(
                    fmt::layer()
                        .with_ansi(false)
                        .with_writer(w)
                        .with_filter(env_filter())
                        .boxed(),
                );
            }
            if let Some(w) = open(format!("critical-{ts}.log")) {
                file_layers.push(
                    fmt::layer()
                        .with_ansi(false)
                        .with_writer(w)
                        .with_filter(tracing_subscriber::filter::LevelFilter::WARN)
                        .boxed(),
                );
            }
        }
    }

    tracing_subscriber::registry()
        .with(stdout)
        .with(file_layers)
        .init();
}

/// Todo.md Phase 1.6: the scheduler thread owns the model and the CUDA
/// context — a panic there is the whole server (case E1). Leave a last
/// memory snapshot in `logs/crash-<ts>.log` before the default hook
/// aborts/unwinds.
fn install_crash_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        spark_runtime::crash::write_crash_snapshot("panic", &info.to_string());
        default(info);
    }));
}

#[tokio::main]
async fn main() -> Result<()> {
    init_logging();
    install_crash_hook();

    let cli = Cli::parse();

    match cli.command {
        Command::Serve(args) => serve(args).await,
    }
}

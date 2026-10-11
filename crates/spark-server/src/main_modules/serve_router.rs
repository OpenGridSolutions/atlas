// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 9-11 of `serve()`: build the axum router with CORS +
//! middleware, mark ready, bind the listener, and start the HTTP
//! server. Extracted (refactor wave-4e) for the ≤500 LoC cap.

use std::sync::Arc;

use anyhow::Result;
use axum::Router;
use axum::routing::{get, post};

use crate::anthropic;
use crate::api;
use crate::main_modules::AppState;
use crate::main_modules::middleware::{
    model_state_middleware, openai_observability_middleware, priority_middleware,
    rate_limit_middleware, require_auth_middleware,
};

pub(crate) async fn build_and_serve(
    state: Arc<AppState>,
    model_ready: Arc<std::sync::atomic::AtomicBool>,
    bind: &str,
    port: u16,
) -> Result<()> {
    let cors = tower_http::cors::CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods([
            axum::http::Method::GET,
            axum::http::Method::POST,
            axum::http::Method::DELETE,
            axum::http::Method::OPTIONS,
        ])
        .allow_headers(tower_http::cors::Any);

    // Catch any panic in a handler and convert it to a 500 instead of
    // hanging the connection. With ~500 production unwraps still in the
    // codebase post-audit, this is cheap insurance — the panicking task
    // dies cleanly and the client sees a JSON error rather than a hung
    // socket. Default `tower_http::catch_panic` body is a plain text
    // "Service Internal Server Error"; we don't override the body so as
    // to avoid leaking backtrace contents to the client.
    let catch_panic = tower_http::catch_panic::CatchPanicLayer::new();

    // Captured before `state` moves into the router (ATLAS_MEM_DEBUG ready
    // line below). The boot model is always registered, hence the 1.
    let registered_models = state
        .hotswap
        .as_ref()
        .map(|h| h.registry.len())
        .unwrap_or(1);

    let app = Router::new()
        .route("/v1/chat/completions", post(api::chat_completions))
        .route("/v1/chat/completions/{id}", get(api::get_stored_completion))
        .route("/v1/completions", post(api::completions))
        .route("/v1/responses", post(api::responses_endpoint))
        .route(
            "/v1/responses/{id}",
            get(api::get_stored_response).delete(api::delete_stored_response),
        )
        .route(
            "/v1/responses/{id}/input_items",
            get(api::list_response_input_items),
        )
        .route("/v1/responses/{id}/cancel", post(api::cancel_response))
        .route("/v1/conversations", post(api::create_conversation))
        .route(
            "/v1/conversations/{id}",
            get(api::get_conversation)
                .post(api::update_conversation)
                .delete(api::delete_conversation),
        )
        .route(
            "/v1/conversations/{id}/items",
            post(api::add_conversation_items).get(api::list_conversation_items),
        )
        .route(
            "/v1/conversations/{id}/items/{item_id}",
            get(api::get_conversation_item).delete(api::delete_conversation_item),
        )
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .route("/v1/models", get(api::list_models))
        .route("/v1/models/{*model_id}", get(api::get_model))
        // Qwen3-Embedding serving (multiclass Plan.md 1.4): last-token
        // pooled + L2-normalized hidden states, OpenAI shape.
        .route("/v1/embeddings", post(api::embeddings::embeddings))
        // Qwen3-Reranker scoring (multiclass Plan.md 1.4): Jina/Cohere-
        // compatible rerank API over the yes/no logit probe.
        .route("/v1/rerank", post(api::rerank::rerank))
        // TranslateGemma text translation (multiclass Plan.md Phase T):
        // structured request, server-side forced prompt frame.
        .route("/v1/translate", post(api::translate::translate))
        // Chandra-2 document OCR (Todo 3.3): single-model category
        // endpoint, server-side Chandra prompt frame.
        .route("/v1/ocr", post(api::ocr::ocr))
        // Datalab-compatible conversion jobs (docs/design/deadline-
        // admission.md PR B): submit → poll → cancel. Not under /v1 —
        // the client is an unmodified Datalab driver pointed at Atlas.
        .route("/convert", post(api::convert::submit))
        .route(
            "/convert/{id}",
            get(api::convert::status).delete(api::convert::cancel),
        )
        .route("/rerank", post(api::rerank::rerank))
        // 501 stubs: return an OpenAI-shaped error body so auto-probe
        // clients (Helicone, LangChain, Vercel AI SDK) fall back instead
        // of hanging on a silent 404.
        .route(
            "/v1/batches",
            post(api::batches_stub).get(api::batch_list_stub),
        )
        .route(
            "/v1/batches/{id}",
            get(api::batch_get_stub).delete(api::batch_get_stub),
        )
        .route("/v1/batches/{id}/cancel", post(api::batch_get_stub))
        .route("/v1/files", post(api::files_stub).get(api::files_stub))
        .route(
            "/v1/files/{id}",
            get(api::files_stub).delete(api::files_stub),
        )
        .route("/v1/files/{id}/content", get(api::files_stub))
        // Whisper speech-to-text (multiclass Plan.md W.3) — resident
        // sidecar, independent of the hot-swap state (not GATED).
        .route("/v1/audio/transcriptions", post(api::audio::transcriptions))
        .route("/v1/audio/translations", post(api::audio_stub))
        .route("/v1/audio/speech", post(api::audio_stub))
        // Image generation (Todo 3.2) — served by the resident Qwen-Image
        // sidecar, independent of the hot-swap state (not GATED).
        .route("/v1/images/generations", post(api::images::generations))
        .route("/v1/images/edits", post(api::images_edit::edits))
        .route(
            "/v1/images/variations",
            post(api::images::edits_unsupported),
        )
        .route("/v1/moderations", post(api::moderations_stub))
        // Hot-swap admin surface (HOTSWAP_MVP_PLAN.md). Covered by the same
        // auth middleware as /v1/* — see the needs_auth predicate in
        // middleware.rs, which includes /admin explicitly.
        .route("/admin/models", get(api::admin::admin_models_state))
        .route("/admin/models/snapshot", post(api::admin::admin_snapshot))
        .route("/admin/models/unload", post(api::admin::admin_unload))
        .route("/admin/models/load", post(api::admin::admin_load))
        .route("/tokenize", post(api::tokenize))
        .route("/detokenize", post(api::detokenize))
        // Memory + residency diagnostics (Plan.md Phase S) — surfaces the
        // cudaMemGetInfo vs /proc/meminfo discrepancy as a metric.
        .route("/v1/stats", get(api::stats::stats))
        // Per-session KV tracking (E2): content-addressed probe across
        // RAM radix cache + NVMe store. Under /v1 → bearer-gated.
        .route(
            "/v1/atlas/session/state",
            post(api::session_state::session_state),
        )
        // TEL-3 introspection: arena, slots, queues, swap decision,
        // NVMe store. Bearer-gated via the /internal prefix (TEL-7).
        .route("/internal/state", get(api::internal_state::internal_state))
        .route("/health", get(api::health))
        .route("/health/live", get(api::health_live))
        .route("/metrics", get(api::metrics_handler))
        // Body size limit. Default 32 MB covers typical multi-image and
        // long-prompt requests; raise via `ATLAS_MAX_BODY_BYTES` (in
        // bytes) for unusual deployments. Lowering it protects against
        // DoS attempts that send oversized payloads to burn CPU on JSON
        // parsing + tokenization before the model even sees them.
        .layer(axum::extract::DefaultBodyLimit::max(
            std::env::var("ATLAS_MAX_BODY_BYTES")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(32 * 1024 * 1024),
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            priority_middleware,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            model_state_middleware,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            rate_limit_middleware,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            require_auth_middleware,
        ))
        .layer(axum::middleware::from_fn(openai_observability_middleware))
        .layer(cors)
        .layer(catch_panic)
        .with_state(state);

    // Model loaded, scheduler running — mark as ready.
    model_ready.store(true, std::sync::atomic::Ordering::Relaxed);

    let addr = format!("{bind}:{port}");
    if bind == "0.0.0.0" {
        tracing::warn!(
            "Atlas is listening on {addr} — reachable from any host on the network. \
             If this machine is on a shared LAN or has a public IP, pass \
             --bind 127.0.0.1 (or set --require-auth and a real firewall) before \
             accepting traffic."
        );
    } else if bind == "127.0.0.1" || bind == "localhost" || bind == "::1" {
        // m00ch13 (Discord 2026-05-07): combined `--network host` with `-p 8000`
        // expecting LAN reachability and got refused from another machine. The
        // default loopback bind is correct for security, but the failure mode
        // ("connection refused from $LAN_IP") is opaque without this hint.
        tracing::info!(
            "API reachable only from this machine (loopback). To expose on the \
             LAN pass --bind 0.0.0.0; combine with --require-auth and \
             --auth-tokens-file for non-trusted networks."
        );
    }
    tracing::info!("Listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    // ATLAS_MEM_DEBUG: readiness marker on the memdebug target — under
    // `RUST_LOG=off,memdebug=debug` every other boot line is silenced, so
    // this is the "start testing now" signal.
    crate::memdebug::on_server_ready(&addr, registered_models);
    // `into_make_service_with_connect_info` exposes the socket peer addr
    // to extractors — needed by `rate_limit_middleware` when the caller
    // didn't send X-Forwarded-For.
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .await?;

    Ok(())
}

// SPDX-License-Identifier: AGPL-3.0-only

//! /admin/models endpoints — model snapshot / unload / load without process
//! restart (HOTSWAP_MVP_PLAN.md Tasks 3–5). Mounted on the main router, so
//! the global auth middleware covers them (`--require-auth`).

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use super::compact::openai_error_response;
use crate::main_modules::AppState;
use crate::main_modules::hotswap::{HotswapCtx, SwapState};

/// Resolve the hot-swap context or produce the standard 503.
fn ctx_or_503(state: &AppState) -> Result<Arc<HotswapCtx>, Box<Response>> {
    state.hotswap.clone().ok_or_else(|| {
        Box::new(openai_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "hot-swap unavailable: weights arena disabled (--weights-arena-gb 0)".to_string(),
        ))
    })
}

/// GET /admin/models — current lifecycle state + arena stats.
pub async fn admin_models_state(State(state): State<Arc<AppState>>) -> Response {
    let ctx = match ctx_or_503(&state) {
        Ok(c) => c,
        Err(r) => return *r,
    };
    Json(serde_json::json!({
        "state": ctx.state().as_str(),
        "model": ctx.model_id,
        "arena": {
            "base": format!("{:#x}", ctx.arena.base().0),
            "capacity_bytes": ctx.arena.capacity(),
            "used_bytes": ctx.arena.used(),
            "host_registered": ctx.arena.host_registered(),
        },
        "tensor_count": ctx.manifest.read().len(),
        "snapshot_path": ctx.snapshot_path().display().to_string(),
    }))
    .into_response()
}

/// POST /admin/models/snapshot — dump the arena as an ATLASRAW file.
/// Returns `{path, bytes, payload_xxh3, duration_ms}`.
pub async fn admin_snapshot(State(state): State<Arc<AppState>>) -> Response {
    let ctx = match ctx_or_503(&state) {
        Ok(c) => c,
        Err(r) => return *r,
    };
    let _admin = ctx.admin_lock.lock().await;
    if ctx.state() != SwapState::Running {
        return openai_error_response(
            StatusCode::CONFLICT,
            format!(
                "cannot snapshot while model state is '{}' (need 'running')",
                ctx.state().as_str()
            ),
        );
    }
    let ctx2 = ctx.clone();
    match tokio::task::spawn_blocking(move || ctx2.snapshot_blocking()).await {
        Ok(Ok((path, stats))) => Json(serde_json::json!({
            "path": path.display().to_string(),
            "bytes": stats.bytes,
            "payload_xxh3": format!("{:#018x}", stats.payload_xxh3),
            "duration_ms": stats.duration_ms as u64,
            "o_direct": stats.o_direct,
        }))
        .into_response(),
        Ok(Err(e)) => openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("snapshot failed: {e:#}"),
        ),
        Err(join) => openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("snapshot task panicked: {join}"),
        ),
    }
}

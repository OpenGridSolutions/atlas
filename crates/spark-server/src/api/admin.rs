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

/// POST /admin/models/unload — drain (≤2000 ms), free the model's device
/// allocations, drop the model index, reset the arena. The arena mapping
/// itself is never returned to the kernel.
pub async fn admin_unload(State(state): State<Arc<AppState>>) -> Response {
    let ctx = match ctx_or_503(&state) {
        Ok(c) => c,
        Err(r) => return *r,
    };
    let _admin = ctx.admin_lock.lock().await;
    if ctx.state() != SwapState::Running {
        return openai_error_response(
            StatusCode::CONFLICT,
            format!(
                "cannot unload while model state is '{}' (need 'running')",
                ctx.state().as_str()
            ),
        );
    }
    // Gate inference immediately — the scheduler confirms Draining when it
    // picks the command up, but new requests must bounce right away.
    ctx.set_state(SwapState::Draining);
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    ctx.scheduler
        .push_control(crate::scheduler::ControlMsg::Unload { reply: reply_tx });
    match tokio::time::timeout(std::time::Duration::from_secs(60), reply_rx).await {
        Ok(Ok(Ok(stats))) => Json(stats).into_response(),
        Ok(Ok(Err(e))) => {
            openai_error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("unload failed: {e:#}"))
        }
        Ok(Err(_)) => openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "unload reply channel dropped (scheduler died?)".to_string(),
        ),
        Err(_) => openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "unload timed out after 60 s".to_string(),
        ),
    }
}

/// POST /admin/models/load {"model": "..."} — reload the snapshot into the
/// arena and rebuild the model (same-model scope). Body optional; when
/// present, `model` must match the model that occupied the arena.
pub async fn admin_load(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    let ctx = match ctx_or_503(&state) {
        Ok(c) => c,
        Err(r) => return *r,
    };
    // Optional JSON body {"model": "..."}.
    if !body.is_empty() {
        let requested = match serde_json::from_slice::<serde_json::Value>(&body) {
            Ok(v) => v.get("model").and_then(|m| m.as_str()).map(String::from),
            Err(e) => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("invalid JSON body: {e}"),
                );
            }
        };
        if let Some(requested) = requested
            && requested != ctx.model_id
        {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "same-model reload only (MVP): arena holds '{}', requested '{requested}'",
                    ctx.model_id
                ),
            );
        }
    }
    let _admin = ctx.admin_lock.lock().await;
    if ctx.state() != SwapState::NoModel {
        return openai_error_response(
            StatusCode::CONFLICT,
            format!(
                "cannot load while model state is '{}' (need 'unloaded')",
                ctx.state().as_str()
            ),
        );
    }
    // Resolve the snapshot before waking the scheduler: 404 beats a
    // scheduler-side error for a missing file.
    let path = ctx.snapshot_path();
    if !path.exists() {
        return openai_error_response(
            StatusCode::NOT_FOUND,
            format!("no snapshot for '{}' at {}", ctx.model_id, path.display()),
        );
    }
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    ctx.scheduler
        .push_control(crate::scheduler::ControlMsg::Load { reply: reply_tx });
    // Payload read (~50 GiB) + model rebuild (transposes) can take minutes.
    match tokio::time::timeout(std::time::Duration::from_secs(900), reply_rx).await {
        Ok(Ok(Ok(stats))) => Json(stats).into_response(),
        Ok(Ok(Err(e))) => {
            openai_error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("load failed: {e:#}"))
        }
        Ok(Err(_)) => openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "load reply channel dropped (scheduler died?)".to_string(),
        ),
        Err(_) => openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "load timed out after 900 s".to_string(),
        ),
    }
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

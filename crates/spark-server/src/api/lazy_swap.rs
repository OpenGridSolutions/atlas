// SPDX-License-Identifier: AGPL-3.0-only

//! Request-time lazy model swap (Task 6 Phase 3).
//!
//! Called by the `/v1/chat/completions` and `/v1/completions` handlers
//! BEFORE the per-request [`ModelRuntime`] snapshot is taken and before
//! anything is enqueued to the scheduler:
//!
//! - requested model == active and Running → pass through (no cost);
//! - requested model registered but not active (or nothing loaded, e.g.
//!   right after a preseed boot) → synchronously unload the current model
//!   (existing 2 s drain) and load the requested one from its snapshot,
//!   then proceed — the triggering request pays the ~6 s swap and gets an
//!   `X-Atlas-Model-Swap: <ms>` response header;
//! - requested model not registered → 404 with a JSON body listing the
//!   available models. Never an implicit safetensors init.
//!
//! Only active in lazy multi-model mode (`HotswapCtx.lazy_enabled`, i.e. a
//! preseed list was configured). Without it this is a pass-through and
//! single-model behaviour is unchanged.

use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use super::compact::openai_error_response;
use crate::main_modules::AppState;
use crate::main_modules::hotswap::{HotswapCtx, LazyPhase, ModelCategory, SwapState};

/// Response-header name carrying the swap latency a request paid.
pub const SWAP_HEADER: &str = "x-atlas-model-swap";

/// Gate verdict of [`prepare_model_gate`]: either proceed now, or hold a
/// queue ticket whose grant the caller must await (streaming handlers
/// open their SSE response first — WAIT-1).
pub enum GatePass {
    Ready(Option<u64>),
    Wait(crate::arena_sched::Ticket),
}

/// Ensure `requested` is the active model AND belongs to the endpoint's
/// category, swapping/queueing as needed. Returns `Ok(None)`
/// (pass-through), `Ok(Some(ms))` (this request waited `ms` for the
/// model), or `Err(response)` to short-circuit.
pub async fn ensure_model_active(
    state: &Arc<AppState>,
    requested: &str,
    required: ModelCategory,
    priority: bool,
    streaming: bool,
) -> Result<Option<u64>, Response> {
    match prepare_model_gate(state, requested, required, priority, streaming).await? {
        GatePass::Ready(ms) => Ok(ms),
        GatePass::Wait(ticket) => match ticket.wait().await {
            crate::arena_sched::TicketOutcome::Granted => {
                Ok(Some(ticket.waited().as_millis() as u64))
            }
            crate::arena_sched::TicketOutcome::Expired => Err(wait_expired(requested)),
            crate::arena_sched::TicketOutcome::Rejected => Err(swap_error(format!(
                "model '{requested}' failed to load; the previous model was restored"
            ))),
        },
    }
}

/// Validation + fast path + (with the dispatcher) ticket registration —
/// everything that answers INSTANTLY. The potentially long wait is the
/// returned ticket; without a dispatcher the legacy synchronous swap
/// runs here and the result is `Ready`.
pub async fn prepare_model_gate(
    state: &Arc<AppState>,
    requested: &str,
    required: ModelCategory,
    priority: bool,
    streaming: bool,
) -> Result<GatePass, Response> {
    let Some(ctx) = state.hotswap.as_ref() else {
        return Ok(GatePass::Ready(None));
    };
    if !ctx.lazy_enabled || requested.is_empty() {
        return Ok(GatePass::Ready(None));
    }
    // Category gate BEFORE the active-model fast path — a mis-categorized
    // request must 400 even when its model happens to be resident.
    let Some(reg) = ctx.registered(requested) else {
        // Sidecar models (transcribe, image) are configured but never in
        // the swap registry; give them the categorized 400 rather than a
        // generic 404.
        if state.transcribe_models.iter().any(|m| m == requested) {
            return Err(category_mismatch(
                requested,
                ModelCategory::Transcribe,
                required,
            ));
        }
        if state.image_models.iter().any(|m| m == requested) {
            return Err(category_mismatch(requested, ModelCategory::Image, required));
        }
        let available: Vec<&str> = ctx.registry.iter().map(|m| m.model_id.as_str()).collect();
        let body = serde_json::json!({
            "error": {
                "message": format!(
                    "model '{requested}' is not registered on this server; available: {}",
                    available.join(", ")
                ),
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            },
            "available_models": available,
        });
        return Err((StatusCode::NOT_FOUND, axum::Json(body)).into_response());
    };
    if reg.category != required {
        return Err(category_mismatch(requested, reg.category, required));
    }
    // ── Pressure-scheduler gate (requirements §7) ──
    // With the arena dispatcher active, requests never swap synchronously:
    // the resident model fast-paths (SCH-7), everything else queues a
    // ticket and waits for the dispatcher's grant. The 503-for-third-model
    // and "model is unloading" rejections are gone (DRN-4).
    if let Some(dispatcher) = state.arena.as_ref() {
        if ctx.state() == SwapState::Running
            && ctx.active_model().model_id == requested
            && !dispatcher.swap_in_progress()
        {
            dispatcher.queues.note_arrival(requested);
            return Ok(GatePass::Ready(None));
        }
        // WAIT-3: a non-streaming client whose estimated wait already
        // exceeds its hard bound gets the 503 BEFORE enqueueing.
        if !streaming {
            let est = dispatcher.estimate_wait_secs(requested);
            if est > state.nonstream_max_wait.as_secs_f64() {
                return Err(wait_estimate_exceeded(requested, est));
            }
        }
        let ticket = match dispatcher.queues.submit(requested, priority, streaming) {
            Ok(t) => t,
            Err(crate::arena_sched::SubmitError::QueueFull { retry_after_secs }) => {
                let body = serde_json::json!({
                    "error": {
                        "message": format!(
                            "queue for model '{requested}' is full; retry shortly"
                        ),
                        "type": "server_error",
                        "param": "model",
                        "code": "model_queue_full",
                    },
                });
                return Err((
                    StatusCode::TOO_MANY_REQUESTS,
                    [(
                        axum::http::header::RETRY_AFTER,
                        retry_after_secs.to_string(),
                    )],
                    axum::Json(body),
                )
                    .into_response());
            }
        };
        dispatcher.poke();
        return Ok(GatePass::Wait(ticket));
    }

    // Fast path: right model, ready to serve.
    if ctx.state() == SwapState::Running && ctx.active_model().model_id == requested {
        return Ok(GatePass::Ready(None));
    }

    // ── Legacy synchronous swap (no dispatcher): Idle | Swapping{to} ──
    // Exactly one request claims the swap; same-target requests wait on it
    // (and pass through once it settles, paying no header); a request for
    // a THIRD model while a swap is in flight gets 503 + Retry-After.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(950);
    loop {
        if std::time::Instant::now() > deadline {
            return Err(swap_error(format!(
                "gave up waiting for '{requested}' to become active"
            )));
        }
        // Re-check the fast path each round — a waited-on swap may have
        // just installed our target.
        if ctx.state() == SwapState::Running && ctx.active_model().model_id == requested {
            return Ok(GatePass::Ready(None));
        }
        enum Action {
            Claim(Arc<tokio::sync::Notify>),
            Wait(Arc<tokio::sync::Notify>),
            Reject(String),
        }
        let action = {
            let mut phase = ctx.lazy_phase.lock();
            match &*phase {
                LazyPhase::Idle => {
                    let done = Arc::new(tokio::sync::Notify::new());
                    *phase = LazyPhase::Swapping {
                        to: requested.to_string(),
                        done: done.clone(),
                    };
                    Action::Claim(done)
                }
                LazyPhase::Swapping { to, done } if to == requested => Action::Wait(done.clone()),
                LazyPhase::Swapping { to, .. } => Action::Reject(to.clone()),
            }
        };
        match action {
            Action::Claim(done) => {
                let t0 = std::time::Instant::now();
                let from = ctx.active_model().model_id;
                let result = swap_to(ctx, requested).await;
                *ctx.lazy_phase.lock() = LazyPhase::Idle;
                done.notify_waiters();
                result?;
                let ms = t0.elapsed().as_millis() as u64;
                tracing::info!(
                    "Lazy swap: request for '{requested}' triggered a model swap (from \
                     '{from}') — {ms} ms paid by this request"
                );
                return Ok(GatePass::Ready(Some(ms)));
            }
            Action::Wait(done) => {
                // Notify is only an accelerator: a bounded timeout guards
                // the register-after-notify race, and the loop re-checks.
                let _ =
                    tokio::time::timeout(std::time::Duration::from_secs(2), done.notified()).await;
                continue;
            }
            Action::Reject(to) => {
                let body = serde_json::json!({
                    "error": {
                        "message": format!(
                            "a model swap to '{to}' is in flight; '{requested}' cannot be \
                             served right now — retry shortly",
                        ),
                        "type": "server_error",
                        "param": "model",
                        "code": "model_swap_in_flight",
                    },
                    "swapping_to": to,
                });
                return Err((
                    StatusCode::SERVICE_UNAVAILABLE,
                    [(axum::http::header::RETRY_AFTER, "10")],
                    axum::Json(body),
                )
                    .into_response());
            }
        }
    }
}

/// Perform (or piggyback on) the unload→load sequence. Serialized against
/// admin snapshot/unload/load operations by the same `admin_lock`.
pub(crate) async fn swap_to(ctx: &Arc<HotswapCtx>, target: &str) -> Result<(), Response> {
    let _admin = ctx.admin_lock.lock().await;
    // Re-check under the lock: a concurrent request may have completed
    // this exact swap while we waited.
    if ctx.state() == SwapState::Running && ctx.active_model().model_id == target {
        return Ok(());
    }

    // Unload whatever is running (empty/short drain; in-flight decode gets
    // the existing 2 s grace, then finish_reason "model_swap").
    if ctx.state() == SwapState::Running {
        ctx.set_state(SwapState::Draining);
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        ctx.scheduler
            .push_control(crate::scheduler::ControlMsg::Unload { reply: reply_tx });
        match tokio::time::timeout(std::time::Duration::from_secs(60), reply_rx).await {
            Ok(Ok(Ok(_))) => {}
            Ok(Ok(Err(e))) => return Err(swap_error(format!("unload failed: {e:#}"))),
            Ok(Err(_)) => {
                return Err(swap_error("unload reply channel dropped (scheduler died?)"));
            }
            Err(_) => return Err(swap_error("unload timed out after 60 s")),
        }
        // The scheduler flips to NoModel just after replying.
        for _ in 0..100 {
            if ctx.state() == SwapState::NoModel {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    }
    if ctx.state() != SwapState::NoModel {
        return Err(swap_error(format!(
            "cannot load '{target}': state is '{}' (want 'unloaded')",
            ctx.state().as_str()
        )));
    }

    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    ctx.scheduler
        .push_control(crate::scheduler::ControlMsg::Load {
            model: Some(target.to_string()),
            reply: reply_tx,
        });
    match tokio::time::timeout(std::time::Duration::from_secs(900), reply_rx).await {
        Ok(Ok(Ok(_stats))) => Ok(()),
        Ok(Ok(Err(e))) => Err(swap_error(format!("load '{target}' failed: {e:#}"))),
        Ok(Err(_)) => Err(swap_error("load reply channel dropped (scheduler died?)")),
        Err(_) => Err(swap_error(format!("load '{target}' timed out after 900 s"))),
    }
}

/// Drain and unload the active arena model; the lazy-swap machinery
/// reloads it on its next request. Serialized against swaps and admin
/// operations by `admin_lock`. Returns `true` when a model was actually
/// unloaded, `false` when the arena was already empty. Used by the image
/// sidecar's memory-pressure eviction.
pub async fn unload_active(ctx: &Arc<HotswapCtx>) -> anyhow::Result<bool> {
    let _admin = ctx.admin_lock.lock().await;
    match ctx.state() {
        SwapState::NoModel => return Ok(false),
        SwapState::Running => {}
        other => anyhow::bail!(
            "cannot evict the arena model: state is '{}' (a swap is in flight)",
            other.as_str()
        ),
    }
    ctx.set_state(SwapState::Draining);
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    ctx.scheduler
        .push_control(crate::scheduler::ControlMsg::Unload { reply: reply_tx });
    match tokio::time::timeout(std::time::Duration::from_secs(60), reply_rx).await {
        Ok(Ok(Ok(_stats))) => {}
        Ok(Ok(Err(e))) => anyhow::bail!("unload failed: {e:#}"),
        Ok(Err(_)) => anyhow::bail!("unload reply channel dropped (scheduler died?)"),
        Err(_) => anyhow::bail!("unload timed out after 60 s"),
    }
    // The scheduler flips to NoModel just after replying.
    for _ in 0..100 {
        if ctx.state() == SwapState::NoModel {
            return Ok(true);
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    anyhow::bail!("arena did not reach 'unloaded' after the unload reply")
}

/// The Phase-K category 400: names the model's actual category, the
/// endpoint that refused it, and the endpoint that serves it. Never a
/// silent fallback.
fn category_mismatch(requested: &str, actual: ModelCategory, required: ModelCategory) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": format!(
                "model '{requested}' is categorized as {} and not available on {}; \
                 use {} instead",
                actual.as_str(),
                required.endpoint(),
                actual.endpoint(),
            ),
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_wrong_category",
        },
        "model_category": actual.as_str(),
        "model_endpoint": actual.endpoint(),
    });
    (StatusCode::BAD_REQUEST, axum::Json(body)).into_response()
}

/// Map a non-granted ticket outcome onto its HTTP error response.
pub(crate) fn wait_outcome_response(
    outcome: crate::arena_sched::TicketOutcome,
    requested: &str,
) -> Response {
    match outcome {
        crate::arena_sched::TicketOutcome::Expired => wait_expired(requested),
        _ => swap_error(format!(
            "model '{requested}' failed to load; the previous model was restored"
        )),
    }
}

/// WAIT-3: predicted wait already exceeds the non-streaming bound.
fn wait_estimate_exceeded(requested: &str, est_secs: f64) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": format!(
                "model '{requested}' is not resident and the estimated wait \
                 ({est_secs:.0}s) exceeds the non-streaming bound; use streaming \
                 or retry later"
            ),
            "type": "server_error",
            "param": "model",
            "code": "model_wait_too_long",
        },
    });
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(
            axum::http::header::RETRY_AFTER,
            format!("{:.0}", est_secs.clamp(5.0, 600.0)),
        )],
        axum::Json(body),
    )
        .into_response()
}

/// PRI-4 / WAIT-4 expiry: the queue wait bound was exceeded.
fn wait_expired(requested: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": format!(
                "request for model '{requested}' exceeded its queue wait bound; retry later"
            ),
            "type": "server_error",
            "param": "model",
            "code": "model_wait_timeout",
        },
    });
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(axum::http::header::RETRY_AFTER, "30")],
        axum::Json(body),
    )
        .into_response()
}

fn swap_error(msg: impl std::fmt::Display) -> Response {
    openai_error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("lazy model swap failed: {msg}"),
    )
}

/// Attach the swap-latency header to a response (no-op when `ms` is None).
pub fn attach_swap_header(mut resp: Response, ms: Option<u64>) -> Response {
    if let Some(ms) = ms
        && let Ok(v) = axum::http::HeaderValue::from_str(&ms.to_string())
    {
        resp.headers_mut()
            .insert(axum::http::HeaderName::from_static(SWAP_HEADER), v);
    }
    resp
}

// SPDX-License-Identifier: AGPL-3.0-only

//! One model in memory at a time, sidecars included.
//!
//! A sidecar model (Whisper, the image pipeline) is swapped in like an
//! arena model: its request queues a ticket with the arena dispatcher,
//! and the dispatcher's swap ends in [`load_exclusive`] — unload the arena
//! model, unload any other sidecar, then load the requested one. The way
//! back needs no code here: the scheduler unloads every sidecar before it
//! loads an arena model.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use std::time::Instant;

use axum::http::StatusCode;
use axum::response::Response;

use super::compact::openai_error_response;
use crate::arena_sched::{SubmitError, TicketOutcome};
use crate::main_modules::AppState;
use crate::main_modules::hotswap::{HotswapCtx, SwapState};
use crate::sidecars::{Lease, Residency, SidecarSlot};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SidecarKind {
    Whisper,
    Image,
}

/// How often a request queues again when a swap took its sidecar out
/// between the grant and the lease.
const LEASE_ATTEMPTS: usize = 3;

/// Sidecar model ids the endpoints have accepted so far. The endpoints
/// validate a request's model before they get here, so this is what tells
/// a sidecar swap target from an arena one.
static KINDS: LazyLock<parking_lot::Mutex<HashMap<String, SidecarKind>>> =
    LazyLock::new(|| parking_lot::Mutex::new(HashMap::new()));

/// The sidecar kind of `model_id`; `None` for arena models.
pub(crate) fn kind_of(model_id: &str) -> Option<SidecarKind> {
    KINDS.lock().get(model_id).copied()
}

/// Get sidecar `model_id` into memory — as the ONLY model — and lease it.
/// The returned lease keeps it there; the handlers clone it into their
/// blocking GPU work.
pub(super) async fn acquire<T: Send + Sync + 'static>(
    state: &AppState,
    kind: SidecarKind,
    model_id: &str,
    slot: &SidecarSlot<T>,
) -> Result<Arc<Lease<T>>, Response> {
    if slot.resident_model().as_deref() != Some(model_id)
        && let Err(e) = load_source(kind, model_id).await
    {
        // Refused before it queues: a swap would unload the resident model
        // for a load that cannot succeed.
        return Err(openai_error_response(
            StatusCode::NOT_FOUND,
            format!("model '{model_id}' cannot be loaded on this server: {e:#}"),
        ));
    }
    KINDS.lock().insert(model_id.to_string(), kind);
    for _ in 0..LEASE_ATTEMPTS {
        ensure_resident(state, model_id).await?;
        if let Some(lease) = slot.lease().await
            && slot.resident_model().as_deref() == Some(model_id)
        {
            return Ok(Arc::new(lease));
        }
    }
    Err(openai_error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        format!(
            "model '{model_id}' was swapped out {LEASE_ATTEMPTS} times before this request \
             could start; retry later"
        ),
    ))
}

/// Whether `model_id` has what its load needs (checkpoint on disk, compiled
/// kernel target). Reads the model cache, hence off the async workers.
async fn load_source(kind: SidecarKind, model_id: &str) -> anyhow::Result<()> {
    let id = model_id.to_string();
    tokio::task::spawn_blocking(move || match kind {
        SidecarKind::Whisper => super::audio::load_source().map(drop),
        SidecarKind::Image => super::images::load_source(&id).map(drop),
    })
    .await?
}

async fn ensure_resident(state: &AppState, model_id: &str) -> Result<(), Response> {
    let resident = || crate::sidecars::resident_model().as_deref() == Some(model_id);
    if let Some(dispatcher) = state.arena.as_ref() {
        if resident() && !dispatcher.swap_in_progress() {
            dispatcher.queues.note_arrival(model_id);
            return Ok(());
        }
        // Queued on the tier that can wait: transcriptions and images are
        // long jobs by nature, the non-streaming wait bound is not for them.
        let ticket = match dispatcher.queues.submit(model_id, false, true) {
            Ok(t) => t,
            Err(SubmitError::QueueFull { retry_after_secs }) => {
                return Err(super::lazy_swap::queue_full(model_id, retry_after_secs));
            }
        };
        dispatcher.poke();
        return match ticket.wait().await {
            TicketOutcome::Granted => Ok(()),
            other => Err(super::lazy_swap::wait_outcome_response(other, model_id)),
        };
    }
    if resident() {
        return Ok(());
    }
    // No dispatcher: nothing swaps models on demand, so nothing would
    // bring the arena model back either. Load only into an empty server.
    let refuse = |how: &str| {
        Err(openai_error_response(
            StatusCode::CONFLICT,
            format!(
                "model '{model_id}' cannot be loaded: only one model is in memory at a time \
                 and this server does not swap models on demand; {how}"
            ),
        ))
    };
    match state.hotswap.as_ref() {
        Some(ctx) if ctx.state() == SwapState::NoModel => {
            super::lazy_swap::swap_to(ctx, model_id).await
        }
        Some(_) => refuse(
            "unload the active model first (POST /admin/models/unload) or configure a \
             preseed list",
        ),
        None => refuse("hot-swap is disabled, so the active model cannot be unloaded"),
    }
}

/// Make sidecar `target` the one model in memory: the arena model and any
/// other sidecar are unloaded first. The caller holds `admin_lock`.
pub(super) async fn load_exclusive(
    ctx: &Arc<HotswapCtx>,
    kind: SidecarKind,
    target: &str,
) -> anyhow::Result<()> {
    if crate::sidecars::resident_model().as_deref() == Some(target) {
        return Ok(());
    }
    let arena_unloaded = super::lazy_swap::unload_arena(ctx).await?;
    // Waits for the other sidecar's in-flight requests.
    let sidecars_unloaded =
        tokio::task::spawn_blocking(crate::sidecars::unload_all_blocking).await?;
    let t0 = Instant::now();
    match kind {
        SidecarKind::Whisper => super::audio::load().await?,
        SidecarKind::Image => {
            super::images::load(target, arena_unloaded || sidecars_unloaded > 0).await?
        }
    }
    *ctx.active_since.lock() = Instant::now();
    crate::swap_costs::observe_load(target, t0.elapsed().as_secs_f64() * 1000.0, 0.0, 0.0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_model_that_is_not_on_disk_has_no_load_source() {
        // What `acquire` answers 404 with, before any model is evicted.
        let missing = load_source(SidecarKind::Image, "atlas-test/not-downloaded").await;
        assert!(missing.is_err());
    }
}

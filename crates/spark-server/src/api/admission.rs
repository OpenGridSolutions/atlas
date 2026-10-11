// SPDX-License-Identifier: AGPL-3.0-only

//! Request admission for `accepted_processing_time` (design doc §3).
//!
//! Every inference handler calls [`admit`] BEFORE it enqueues anything:
//! the ledger computes the deterministic estimate under its lock, rejects
//! with 503 `capacity_exceeded` (+ `estimated_seconds`, `Retry-After`)
//! when `estimate × safety > budget`, and otherwise hands back an
//! [`Admission`] holding the reservation guard. Handlers attach
//! `X-Atlas-Estimated-Seconds` to their response and clamp the request's
//! runtime deadline to the budget so an accepted request that overruns
//! ends as an error, never as a late success.
//!
//! No budget given ⇒ never rejected; the estimate is still computed (and
//! reserved) when the model has a profile, so the header is informative
//! and the ledger stays accurate for budgeted neighbours.

use std::sync::Arc;
use std::time::Instant;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::arena_sched::{Estimate, EstimateError, Reservation, Snapshot, Verdict, WorkProfile};
use crate::main_modules::AppState;
use crate::main_modules::hotswap::SwapState;

pub const ESTIMATE_HEADER: &str = "x-atlas-estimated-seconds";
pub const BUDGET_HEADER: &str = "x-atlas-accepted-processing-time";

/// Outcome of an admitted request. Keep it alive for the request's
/// duration — dropping it releases the reservation.
pub struct Admission {
    pub estimate: Option<Estimate>,
    pub deadline: Option<Instant>,
    #[allow(dead_code)] // held for its Drop
    reservation: Option<Reservation>,
}

impl Admission {
    /// Safety-adjusted prediction in seconds, if an estimate exists.
    pub fn estimated_secs(&self, state: &AppState) -> Option<f64> {
        self.estimate
            .map(|e| e.total_secs() * state.ledger.safety_factor())
    }

    /// Attach `X-Atlas-Estimated-Seconds` (no-op without an estimate).
    pub fn attach_header(&self, state: &AppState, mut resp: Response) -> Response {
        if let Some(secs) = self.estimated_secs(state)
            && let Ok(v) = axum::http::HeaderValue::from_str(&format!("{secs:.1}"))
        {
            resp.headers_mut()
                .insert(axum::http::HeaderName::from_static(ESTIMATE_HEADER), v);
        }
        resp
    }

    /// Header value for responses built before the admission is moved
    /// elsewhere (queued SSE).
    pub fn header_secs(&self, state: &AppState) -> Option<f64> {
        self.estimated_secs(state)
    }

    /// The earlier of the request's own timeout and the budget deadline.
    pub fn clamp_timeout(&self, timeout_at: Option<Instant>) -> Option<Instant> {
        match (timeout_at, self.deadline) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }
}

/// Attach `X-Atlas-Estimated-Seconds` from a precomputed value.
pub fn attach_estimate_header(mut resp: Response, secs: Option<f64>) -> Response {
    if let Some(secs) = secs
        && let Ok(v) = axum::http::HeaderValue::from_str(&format!("{secs:.1}"))
    {
        resp.headers_mut()
            .insert(axum::http::HeaderName::from_static(ESTIMATE_HEADER), v);
    }
    resp
}

/// Why a budgeted request was not admitted.
pub enum Refusal {
    /// `estimate × safety > budget`.
    Capacity {
        estimate_secs: f64,
        budget_secs: f64,
    },
    /// No profile / uncalibrated for this work; only raised when a
    /// budget was given.
    Unavailable(EstimateError),
}

/// Server snapshot for `model_id` (everything but `reserved_secs`, which
/// the ledger fills under its lock).
pub fn snapshot_for(state: &AppState, model_id: &str) -> Snapshot {
    let max_batch = state.serve_args.max_batch_size.max(1);
    let n_active = crate::scheduler::scheduler_load();
    let (target_resident, model_wait_secs, queued_ahead) = match (&state.hotswap, &state.arena) {
        (Some(ctx), Some(dispatcher)) if ctx.lazy_enabled => {
            let resident = ctx.state() == SwapState::Running
                && ctx.active_model().model_id == model_id
                && !dispatcher.swap_in_progress();
            let wait = if resident {
                0.0
            } else {
                dispatcher.estimate_wait_secs(model_id)
            };
            (resident, wait, dispatcher.queues.queued_for(model_id))
        }
        // Single-model / legacy mode: whatever is loaded serves.
        _ => (true, 0.0, 0),
    };
    Snapshot {
        target_resident,
        model_wait_secs,
        n_active,
        max_batch,
        reserved_secs: 0.0,
        queued_ahead,
    }
}

/// Byte-based prompt size for a request whose tokenizer may not be
/// resident yet. `Debug` output over-estimates content length, which is
/// the safe direction.
pub fn prompt_tokens_guess<T: std::fmt::Debug>(payload: &T) -> usize {
    WorkProfile::tokens_from_bytes(format!("{payload:?}").len())
}

/// Estimate + reserve. `Ok` always when no budget is given.
pub fn admit(
    state: &Arc<AppState>,
    model_id: &str,
    work: WorkProfile,
    budget_secs: Option<f64>,
) -> Result<Admission, Refusal> {
    let now = Instant::now();
    let profile = atlas_kernels::estimate_for_hf_id(model_id);
    let snap = snapshot_for(state, model_id);
    let (verdict, reservation) =
        state
            .ledger
            .admit(model_id, profile.as_ref(), &work, snap, budget_secs, now);
    let deadline = budget_secs.map(|b| now + std::time::Duration::from_secs_f64(b));
    match verdict {
        Verdict::Admitted(est) => {
            tracing::debug!(
                model = model_id,
                est_s = est.total_secs(),
                budget_s = budget_secs,
                "admission: accepted"
            );
            Ok(Admission {
                estimate: Some(est),
                deadline,
                reservation,
            })
        }
        Verdict::Rejected {
            estimate,
            budget_secs,
        } => {
            let predicted = estimate.total_secs() * state.ledger.safety_factor();
            tracing::info!(
                model = model_id,
                est_s = predicted,
                ahead_s = estimate.ahead_secs,
                model_s = estimate.model_secs,
                work_s = estimate.work_secs,
                budget_s = budget_secs,
                "admission: rejected (capacity_exceeded)"
            );
            crate::metrics::ADMISSION_REJECTED_TOTAL
                .with_label_values(&[model_id, "capacity_exceeded"])
                .inc();
            Err(Refusal::Capacity {
                estimate_secs: predicted,
                budget_secs,
            })
        }
        Verdict::NoEstimate(e) => {
            if budget_secs.is_some() {
                crate::metrics::ADMISSION_REJECTED_TOTAL
                    .with_label_values(&[model_id, "estimate_unavailable"])
                    .inc();
                Err(Refusal::Unavailable(e))
            } else {
                Ok(Admission {
                    estimate: None,
                    deadline,
                    reservation: None,
                })
            }
        }
    }
}

/// OpenAI-shaped refusal (chat, completions, embeddings, rerank,
/// translate, ocr, messages, responses).
pub fn openai_refusal(model_id: &str, r: Refusal) -> Response {
    match r {
        Refusal::Capacity {
            estimate_secs,
            budget_secs,
        } => {
            let body = serde_json::json!({
                "error": {
                    "message": format!(
                        "estimated completion time {estimate_secs:.1}s exceeds \
                         accepted_processing_time {budget_secs:.1}s for model '{model_id}'"
                    ),
                    "type": "server_error",
                    "param": "accepted_processing_time",
                    "code": "capacity_exceeded",
                },
                "estimated_seconds": estimate_secs.ceil(),
            });
            (
                StatusCode::SERVICE_UNAVAILABLE,
                [(
                    axum::http::header::RETRY_AFTER,
                    format!("{:.0}", (estimate_secs - budget_secs).clamp(1.0, 600.0)),
                )],
                axum::Json(body),
            )
                .into_response()
        }
        Refusal::Unavailable(e) => {
            let body = serde_json::json!({
                "error": {
                    "message": unavailable_message(model_id, e),
                    "type": "server_error",
                    "param": "accepted_processing_time",
                    "code": "estimate_unavailable",
                },
            });
            (StatusCode::SERVICE_UNAVAILABLE, axum::Json(body)).into_response()
        }
    }
}

pub fn unavailable_message(model_id: &str, e: EstimateError) -> String {
    match e {
        EstimateError::NoProfile => format!(
            "model '{model_id}' has no [estimate] profile in its MODEL.toml; \
             accepted_processing_time cannot be honoured (run tests/estimate_calibration.py)"
        ),
        EstimateError::Uncalibrated => format!(
            "model '{model_id}' [estimate] profile lacks the constants for this request type"
        ),
    }
}

/// Budget from a header (multipart/binary endpoints). Invalid values
/// are a 400 at the call site; this returns `Err(raw)` for them.
pub fn budget_from_headers(headers: &axum::http::HeaderMap) -> Result<Option<f64>, String> {
    let Some(v) = headers.get(BUDGET_HEADER) else {
        return Ok(None);
    };
    let raw = v
        .to_str()
        .map_err(|_| "non-ASCII header value".to_string())?;
    let secs: f64 = raw
        .trim()
        .parse()
        .map_err(|_| format!("'{raw}' is not a number of seconds"))?;
    if !secs.is_finite() || secs <= 0.0 {
        return Err(format!("'{raw}' must be a positive number of seconds"));
    }
    Ok(Some(secs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_budget_parsing() {
        let mut h = axum::http::HeaderMap::new();
        assert_eq!(budget_from_headers(&h).unwrap(), None);
        h.insert(BUDGET_HEADER, "42.5".parse().unwrap());
        assert_eq!(budget_from_headers(&h).unwrap(), Some(42.5));
        h.insert(BUDGET_HEADER, "-1".parse().unwrap());
        assert!(budget_from_headers(&h).is_err());
        h.insert(BUDGET_HEADER, "soon".parse().unwrap());
        assert!(budget_from_headers(&h).is_err());
    }

    #[test]
    fn prompt_guess_grows_with_payload() {
        let small = prompt_tokens_guess(&vec!["hi"]);
        let big = prompt_tokens_guess(&vec!["hello world, this is a longer message"; 20]);
        assert!(big > small);
    }

    #[test]
    fn clamp_timeout_takes_the_earlier() {
        let now = Instant::now();
        let a = Admission {
            estimate: None,
            deadline: Some(now + std::time::Duration::from_secs(5)),
            reservation: None,
        };
        let t = a.clamp_timeout(Some(now + std::time::Duration::from_secs(300)));
        assert_eq!(t, a.deadline);
        let none = Admission {
            estimate: None,
            deadline: None,
            reservation: None,
        };
        assert_eq!(none.clamp_timeout(None), None);
    }
}

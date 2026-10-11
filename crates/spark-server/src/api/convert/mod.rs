// SPDX-License-Identifier: AGPL-3.0-only

//! Datalab-compatible document conversion surface (design doc
//! `docs/design/deadline-admission.md`, PR B).
//!
//! - `POST /convert` (multipart) → `{success, request_id,
//!   request_check_url, estimated_seconds}`; the job runs in the
//!   background ([`worker`]).
//! - `GET /convert/{id}` → `pending | processing | complete | failed`;
//!   `complete` carries the Marker-style document ([`result_tree`]).
//! - `DELETE /convert/{id}` → cancel, idempotent.
//!
//! Every error body is `{success: false, error, error_code}`;
//! `unprocessable_document` is the only code that blames the document
//! (clients must not fall back on it), everything else is Atlas-side.
//! The Atlas extension `accepted_processing_time` (seconds) becomes the
//! job deadline here; the admission estimate that rejects up front is
//! PR A (`estimated_seconds` is `null` until then).

pub mod cache;
pub mod job_store;
pub mod params;
pub mod result_tree;
pub mod worker;

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::multipart::MultipartRejection;
use axum::extract::{Multipart, Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use serde_json::json;

use crate::main_modules::AppState;
use cache::ResultCache;
use job_store::{CreateError, JobConfig, JobState, JobStore};

#[derive(Clone, Debug)]
pub struct ConvertConfig {
    pub page_parallelism: usize,
    pub max_pages: usize,
    pub max_failed_pages_pct: u8,
}

pub struct ConvertService {
    pub jobs: JobStore,
    pub cache: Option<Arc<dyn ResultCache>>,
    pub cfg: ConvertConfig,
}

impl ConvertService {
    /// Build from the serve flags; opens the result cache directory when
    /// `--ocr-cache-dir` is set (fails at boot if it is not writable).
    pub fn from_args(args: &crate::cli::ServeArgs) -> anyhow::Result<Arc<Self>> {
        let cache: Option<Arc<dyn ResultCache>> = match args.ocr_cache_dir.as_deref() {
            Some(dir) => {
                let c = cache::FsResultCache::open(
                    dir,
                    Duration::from_secs(args.ocr_cache_ttl_days.saturating_mul(86_400)),
                    (args.ocr_cache_max_gb * (1u64 << 30) as f64) as u64,
                )
                .map_err(|e| anyhow::anyhow!("--ocr-cache-dir {}: {e}", dir.display()))?;
                tracing::info!(
                    "OCR result cache at {} (ttl {} d, max {:.1} GiB)",
                    dir.display(),
                    args.ocr_cache_ttl_days,
                    args.ocr_cache_max_gb
                );
                Some(Arc::new(c))
            }
            None => None,
        };
        Ok(Arc::new(Self {
            jobs: JobStore::new(JobConfig {
                result_ttl: Duration::from_secs(args.ocr_result_ttl_s.max(1)),
                poll_idle: Duration::from_secs(args.ocr_poll_idle_s.max(1)),
                store_max_bytes: args.ocr_store_max_mb.saturating_mul(1 << 20),
                max_jobs: args.ocr_max_jobs.max(1),
            }),
            cache,
            cfg: ConvertConfig {
                page_parallelism: args.ocr_page_parallelism.max(1),
                max_pages: args.ocr_max_pages.max(1),
                max_failed_pages_pct: args.ocr_max_failed_pages_pct.min(100),
            },
        }))
    }

    /// Periodic sweep (expiry, abandonment, deadlines) every 10 s.
    pub fn spawn_gc(self: &Arc<Self>) {
        let svc = Arc::clone(self);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(10));
            loop {
                tick.tick().await;
                let r = svc.jobs.sweep(Instant::now());
                for id in &r.abandoned {
                    tracing::warn!(
                        "convert {id}: abandoned (no poll for {}s)",
                        svc.jobs.config().poll_idle.as_secs()
                    );
                }
                for id in &r.deadline_exceeded {
                    tracing::warn!("convert {id}: deadline exceeded while running");
                }
            }
        });
    }

    /// For `/internal/state`.
    pub fn state_json(&self) -> serde_json::Value {
        let (pending, processing, complete, failed) = self.jobs.counts();
        json!({
            "jobs": {"pending": pending, "processing": processing, "complete": complete, "failed": failed},
            "stored_bytes": self.jobs.stored_bytes(),
            "cache_enabled": self.cache.is_some(),
            "page_parallelism": self.cfg.page_parallelism,
        })
    }
}

/// Datalab-shaped error body.
pub fn datalab_error(status: StatusCode, error_code: &str, message: impl Into<String>) -> Response {
    let body = json!({
        "success": false,
        "error": message.into(),
        "error_code": error_code,
    });
    (status, Json(body)).into_response()
}

/// Absolute `request_check_url`. Scheme comes from `X-Forwarded-Proto`
/// only when proxy headers are trusted; host from `Host`. Without a
/// host header the URL is path-only.
pub fn check_url(headers: &HeaderMap, trust_proxy: bool, id: &str) -> String {
    let host = headers
        .get(axum::http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .filter(|h| !h.is_empty() && !h.contains(['/', ' ']));
    let scheme = if trust_proxy {
        headers
            .get("x-forwarded-proto")
            .and_then(|v| v.to_str().ok())
            .filter(|s| *s == "https" || *s == "http")
            .unwrap_or("http")
    } else {
        "http"
    };
    match host {
        Some(h) => format!("{scheme}://{h}/convert/{id}"),
        None => format!("/convert/{id}"),
    }
}

/// `POST /convert`
pub async fn submit(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    multipart: Result<Multipart, MultipartRejection>,
) -> Response {
    let multipart = match multipart {
        Ok(m) => m,
        Err(e) => {
            return datalab_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                format!("expected multipart/form-data: {e}"),
            );
        }
    };
    let params = match params::parse_multipart(multipart).await {
        Ok(p) => p,
        Err(params::ParamError::Multipart(m)) if m.contains("length limit") => {
            return datalab_error(StatusCode::PAYLOAD_TOO_LARGE, "payload_too_large", m);
        }
        Err(e) => return datalab_error(StatusCode::BAD_REQUEST, "invalid_request", e.to_string()),
    };
    // Synchronous decisions (design doc Teil 2 §7): document kind,
    // model availability, store capacity. Page count and the estimate
    // (PR A) follow here too once the latter exists.
    if !crate::ocr::input::is_pdf(&params.file) && image::guess_format(&params.file).is_err() {
        return datalab_error(
            StatusCode::BAD_REQUEST,
            "unprocessable_document",
            "file is neither a PDF nor a supported image",
        );
    }
    let Some(model_id) = crate::api::ocr::resolve_ocr_model(&state) else {
        return datalab_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "unavailable",
            "no OCR model registered (ATLAS_PRESEED_OCR)",
        );
    };
    let svc = Arc::clone(&state.convert);
    // Page count + selection synchronously (design doc Teil 2 §7): the
    // estimate needs it, and a bad document is refused right here.
    let pages = match worker::count_and_select(&params).await {
        Ok(p) => p,
        Err((code, msg)) => {
            let status = if code == "unprocessable_document" {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::SERVICE_UNAVAILABLE
            };
            return datalab_error(status, code, msg);
        }
    };
    if pages.len() > svc.cfg.max_pages {
        return datalab_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "document_too_large",
            format!(
                "{} pages selected, server limit is {} (--ocr-max-pages)",
                pages.len(),
                svc.cfg.max_pages
            ),
        );
    }
    let admission = match crate::api::admission::admit(
        &state,
        &model_id,
        crate::arena_sched::WorkProfile {
            prompt_tokens: 0,
            output_tokens: 0,
            pages: pages.len(),
        },
        params.accepted_processing_time,
    ) {
        Ok(a) => a,
        Err(crate::api::admission::Refusal::Capacity {
            estimate_secs,
            budget_secs,
        }) => {
            let body = json!({
                "success": false,
                "error": format!(
                    "estimated completion time {estimate_secs:.1}s exceeds \
                     accepted_processing_time {budget_secs:.1}s"
                ),
                "error_code": "capacity_exceeded",
                "estimated_seconds": estimate_secs.ceil(),
            });
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                [(
                    axum::http::header::RETRY_AFTER,
                    format!("{:.0}", (estimate_secs - budget_secs).clamp(1.0, 600.0)),
                )],
                Json(body),
            )
                .into_response();
        }
        Err(crate::api::admission::Refusal::Unavailable(e)) => {
            return datalab_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "unavailable",
                crate::api::admission::unavailable_message(&model_id, e),
            );
        }
    };
    let estimated_seconds = admission.estimated_secs(&state);
    let id = match job_store::new_job_id() {
        Ok(id) => id,
        Err(e) => {
            return datalab_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                format!("cannot generate job id: {e}"),
            );
        }
    };
    let now = Instant::now();
    let deadline = params
        .accepted_processing_time
        .map(|s| now + Duration::from_secs_f64(s));
    let cancel = match svc.jobs.create(id.clone(), now, deadline) {
        Ok(c) => c,
        Err(CreateError::TooManyJobs) => {
            return datalab_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "capacity_exceeded",
                "too many conversion jobs in flight; retry later",
            );
        }
        Err(CreateError::StoreFull) => {
            return datalab_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "capacity_exceeded",
                "result store full; results expire shortly",
            );
        }
    };
    let url = check_url(&headers, state.trust_proxy_headers, &id);
    tracing::info!(
        "convert {id}: accepted '{}' ({} KB, max_pages={:?}, deadline={:?}s)",
        params.filename,
        params.file.len() / 1024,
        params.max_pages,
        params.accepted_processing_time
    );
    tokio::spawn(worker::run_job(
        Arc::clone(&state),
        svc,
        worker::JobSpec {
            id: id.clone(),
            params,
            pages,
            model_id,
            cancel,
            deadline,
            submitted: now,
            admission: Some(admission),
        },
    ));
    Json(json!({
        "success": true,
        "request_id": id,
        "request_check_url": url,
        "estimated_seconds": estimated_seconds.map(f64::ceil),
    }))
    .into_response()
}

/// `GET /convert/{id}`
pub async fn status(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let Some(view) = state.convert.jobs.poll(&id, Instant::now()) else {
        return datalab_error(
            StatusCode::NOT_FOUND,
            "unknown_request",
            format!("no conversion job '{id}' (expired, cancelled, or server restarted)"),
        );
    };
    match view.state {
        JobState::Pending | JobState::Processing => Json(json!({
            "status": view.state.status_str(),
            "success": null,
            "error": "",
            "error_code": null,
            "request_id": id,
        }))
        .into_response(),
        JobState::Complete(doc) => {
            let mut v = (*doc).clone();
            v["request_id"] = json!(id);
            Json(v).into_response()
        }
        JobState::Failed { error, error_code } => Json(json!({
            "status": "failed",
            "success": false,
            "error": error,
            "error_code": error_code,
            "request_id": id,
        }))
        .into_response(),
    }
}

/// `DELETE /convert/{id}` — idempotent cancel.
pub async fn cancel(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let known = state.convert.jobs.cancel(&id, Instant::now());
    Json(json!({
        "success": true,
        "request_id": id,
        "status": if known { "cancelled" } else { "unknown" },
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(
                axum::http::HeaderName::from_bytes(k.as_bytes()).unwrap(),
                axum::http::HeaderValue::from_str(v).unwrap(),
            );
        }
        h
    }

    #[test]
    fn check_url_scheme_and_host() {
        assert_eq!(
            check_url(&headers(&[("host", "atlas.example")]), false, "abc"),
            "http://atlas.example/convert/abc"
        );
        // Proto header ignored unless trusted.
        assert_eq!(
            check_url(
                &headers(&[("host", "atlas.example"), ("x-forwarded-proto", "https")]),
                false,
                "abc"
            ),
            "http://atlas.example/convert/abc"
        );
        assert_eq!(
            check_url(
                &headers(&[("host", "atlas.example"), ("x-forwarded-proto", "https")]),
                true,
                "abc"
            ),
            "https://atlas.example/convert/abc"
        );
        assert_eq!(check_url(&HeaderMap::new(), true, "abc"), "/convert/abc");
        // A host header with a path separator is not trusted.
        assert_eq!(
            check_url(&headers(&[("host", "evil/x")]), false, "abc"),
            "/convert/abc"
        );
    }
}

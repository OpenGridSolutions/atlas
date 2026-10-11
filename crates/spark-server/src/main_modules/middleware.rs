// SPDX-License-Identifier: AGPL-3.0-only

//! Axum middleware: observability, auth, rate-limiting.

use std::sync::Arc;

use crate::main_modules::AppState;
use crate::{openai, rate_limiter};

/// OpenAI-compatible observability headers. Injects on every `/v1/*`
/// response:
/// - `x-request-id`: UUID v4 generated per-request (re-used if client
///   supplied one, letting callers correlate logs end-to-end).
/// - `openai-processing-ms`: server wall-clock time in milliseconds.
/// - `x-ratelimit-*`: static "unlimited" stubs so clients that
///   parse them for backoff don't treat missing headers as unlimited
///   (some SDKs assume 0 = exhausted).
/// - `openai-organization`, `openai-version`: static stubs for
///   parity with `api.openai.com` — several wrappers log these.
pub(crate) async fn openai_observability_middleware(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::http::{HeaderName, HeaderValue};
    let start = std::time::Instant::now();
    let path = req.uri().path().to_string();
    let is_v1 = path.starts_with("/v1/");
    let incoming_req_id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let mut resp = next.run(req).await;
    // Every inference-ish path gets a request id and a row in the event
    // store (PR C) so an app-side fallback can be found in the server's
    // own records; the OpenAI stub headers stay /v1-only.
    let observed = is_v1 || path.starts_with("/convert");
    if !observed {
        return resp;
    }
    let status = resp.status();
    let elapsed_ms = start.elapsed().as_millis();
    let rid = incoming_req_id.unwrap_or_else(|| format!("req_{}", openai::uuid_v4()));
    let endpoint = request_endpoint(&path);
    // 503 is Atlas back-pressure by design (capacity_exceeded, queue
    // full, model wait) — recorded as a request row, not as a failure.
    if status.is_server_error() && status != axum::http::StatusCode::SERVICE_UNAVAILABLE {
        crate::metrics::REQUEST_FAILURES_TOTAL
            .with_label_values(&[endpoint.as_str(), status.as_str()])
            .inc();
        crate::observe::emit(
            crate::observe::Event::new(
                2,
                "request_failed",
                format!("{} {} → {}", endpoint, status.as_u16(), rid),
            )
            .request_id(rid.clone())
            .detail(serde_json::json!({"endpoint": endpoint, "status": status.as_u16(), "elapsed_ms": elapsed_ms as u64})),
        );
    }
    crate::observe::record_request(crate::observe::RequestRecord {
        ts_ms: crate::observe::now_ms(),
        request_id: rid.clone(),
        endpoint: endpoint.clone(),
        status: status.as_u16(),
        elapsed_ms: elapsed_ms as u64,
        model: None,
    });
    let headers = resp.headers_mut();
    if let Ok(v) = HeaderValue::from_str(&rid) {
        headers.insert(HeaderName::from_static("x-request-id"), v);
    }
    if !is_v1 {
        return resp;
    }
    if let Ok(v) = HeaderValue::from_str(&elapsed_ms.to_string()) {
        headers.insert(HeaderName::from_static("openai-processing-ms"), v);
    }
    // Static "effectively unlimited" stubs (Atlas does not enforce rate limits).
    for (k, v) in [
        ("x-ratelimit-limit-requests", "1000000"),
        ("x-ratelimit-remaining-requests", "999999"),
        ("x-ratelimit-reset-requests", "0s"),
        ("x-ratelimit-limit-tokens", "1000000000"),
        ("x-ratelimit-remaining-tokens", "999999999"),
        ("x-ratelimit-reset-tokens", "0s"),
        ("openai-organization", "atlas-local"),
        ("openai-version", "2026-01-01"),
    ] {
        if let Ok(val) = HeaderValue::from_str(v) {
            headers.insert(HeaderName::from_static(k), val);
        }
    }
    resp
}

/// Bearer-token gate. Active when the operator passed `--require-auth`
/// (which lands in `AppState.auth` as `Some(...)`); otherwise this is a
/// pass-through. When active, requests to `/v1/*`, `/convert*`,
/// `/tokenize`, and `/detokenize` must carry `Authorization: Bearer
/// <token>` (or `X-API-Key: <token>`) matching one of the loaded tokens. Health / metrics / liveness paths stay open
/// (they're scrape targets / discovery endpoints, expected to be
/// firewall-protected at the network layer).
///
/// Token comparison is constant-time per candidate (see `auth::AuthConfig`)
/// so an attacker can't recover the secret by measuring early-exit
/// timings. The error body matches the OpenAI shape so client SDKs
/// surface "missing_api_key" / "invalid_api_key" the same way they do
/// against `api.openai.com`.
/// Endpoint label for metrics/records: path with ids collapsed
/// (`/convert/abc` → `/convert/{id}`).
pub(crate) fn request_endpoint(path: &str) -> String {
    let mut parts: Vec<&str> = path.split('/').collect();
    for p in parts.iter_mut().skip(1) {
        if p.len() >= 16
            && p.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            *p = "{id}";
        }
    }
    parts.join("/")
}

/// Inference paths bounced with 503 while the arena is not `Running`.
/// In lazy multi-model mode (Task 6 Phase 3) the lazy-load endpoints
/// handle that state themselves — they queue a ticket instead — so they
/// pass through; introspection, health and admin are never gated.
pub(crate) fn swap_gated(path: &str, lazy_enabled: bool) -> bool {
    const GATED: [&str; 7] = [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/messages",
        "/v1/embeddings",
        "/v1/rerank",
        "/rerank",
    ];
    const LAZY_HANDLED: [&str; 5] = [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/rerank",
        "/rerank",
    ];
    GATED.contains(&path) && !(lazy_enabled && LAZY_HANDLED.contains(&path))
}

/// Inference-endpoint gate for hot-swap (plan Task 4): while a model swap
/// is in progress or no model is loaded, inference requests get an
/// immediate 503 + `Retry-After: 1` with a JSON body naming the state,
/// instead of queueing into a scheduler that will fail them anyway.
pub(crate) async fn model_state_middleware(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(ctx) = state.hotswap.as_ref() {
        let swap_state = ctx.state();
        if swap_state != crate::main_modules::hotswap::SwapState::Running
            && swap_gated(req.uri().path(), ctx.lazy_enabled)
        {
            let body = serde_json::json!({
                "error": {
                    "message": format!(
                        "no model available (state: {}); retry shortly",
                        swap_state.as_str()
                    ),
                    "type": "server_error",
                    "param": null,
                    "code": "model_unavailable",
                },
                "state": swap_state.as_str(),
            });
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                [(axum::http::header::RETRY_AFTER, "1")],
                axum::Json(body),
            )
                .into_response();
        }
    }
    next.run(req).await
}

/// AUTH-3/PRI: proxy-asserted priority, stamped on every request by
/// [`priority_middleware`]. `true` only when `--trust-proxy-headers` is
/// set AND the proxy sent `X-Priority: high`.
#[derive(Clone, Copy, Debug, Default)]
pub struct RequestPriority(pub bool);

/// AUTH-4: `--trust-proxy-headers` is only safe when nothing but the
/// proxy can reach the listener — a loopback bind.
pub(crate) fn is_loopback_bind(bind: &str) -> bool {
    matches!(bind, "127.0.0.1" | "localhost" | "::1")
}

/// AUTH-3/PRI-1: the proxy-asserted priority of one request. `true`
/// only when the operator trusts proxy headers AND the proxy sent
/// `X-Priority: high` (case-insensitive value). Any other value, a
/// missing header, or an untrusting server yields the normal tier.
pub(crate) fn proxy_priority(trust_proxy_headers: bool, headers: &axum::http::HeaderMap) -> bool {
    trust_proxy_headers
        && headers
            .get("x-priority")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.eq_ignore_ascii_case("high"))
}

pub(crate) async fn priority_middleware(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    mut req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let high = proxy_priority(state.trust_proxy_headers, req.headers());
    req.extensions_mut().insert(RequestPriority(high));
    next.run(req).await
}

/// Paths behind `--require-auth`: every inference and management
/// surface. `/admin` (hot-swap) and `/internal` (TEL-7 introspection)
/// are in the set deliberately; health, liveness and metrics stay open
/// (scrape targets, expected to be firewalled at the network layer).
pub(crate) fn needs_auth(path: &str) -> bool {
    path.starts_with("/v1/")
        || path.starts_with("/admin")
        || path.starts_with("/internal")
        || path.starts_with("/convert")
        || path == "/tokenize"
        || path == "/detokenize"
}

/// AUTH-1: the bearer verdict for one request — `None` lets it through;
/// `Some((status, code, message))` is the OpenAI-shaped rejection.
pub(crate) fn bearer_rejection(
    auth_cfg: &crate::auth::AuthConfig,
    headers: &axum::http::HeaderMap,
) -> Option<(axum::http::StatusCode, &'static str, &'static str)> {
    // Datalab clients (the /convert surface) send the token as
    // `X-API-Key`; both carriers validate against the same token set.
    let presented = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()))
        .map(str::trim);
    match presented {
        None => Some((
            axum::http::StatusCode::UNAUTHORIZED,
            "missing_api_key",
            "Missing Authorization: Bearer header",
        )),
        Some(t) if !auth_cfg.validate(t.as_bytes()) => Some((
            axum::http::StatusCode::UNAUTHORIZED,
            "invalid_api_key",
            "Invalid bearer token",
        )),
        Some(_) => None,
    }
}

pub(crate) async fn require_auth_middleware(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let Some(auth_cfg) = state.auth.as_ref() else {
        return next.run(req).await;
    };
    if !needs_auth(req.uri().path()) {
        return next.run(req).await;
    }
    let Some((status, code, message)) = bearer_rejection(auth_cfg, req.headers()) else {
        return next.run(req).await;
    };
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": null,
            "code": code,
        }
    });
    (status, axum::Json(body)).into_response()
}

/// Per-identity rate-limit middleware. When the limiter is enabled via
/// `ATLAS_RATE_LIMIT_RPM` / `ATLAS_RATE_LIMIT_TPM`, admission is checked
/// before dispatching to the handler; denied requests short-circuit with
/// 429 + OpenAI error body + `retry-after` header.
///
/// Identity precedence (from `rate_limiter::extract_identity`):
/// Bearer token → X-Forwarded-For → socket peer addr.
///
/// Headers emitted in both allow and deny paths:
///   x-ratelimit-limit-{requests,tokens}
///   x-ratelimit-remaining-{requests,tokens}
///   x-ratelimit-reset-{requests,tokens}
///
/// These overwrite the static "unlimited" stubs from
/// `openai_observability_middleware`. When the limiter is disabled, this
/// middleware is a pass-through (the observability stubs stand).
pub(crate) async fn rate_limit_middleware(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::http::{HeaderName, HeaderValue, StatusCode};
    use axum::response::IntoResponse;

    // Only apply to /v1/* routes — health/metrics/tokenize stay open.
    let is_v1 = req.uri().path().starts_with("/v1/");
    if !is_v1 || !state.rate_limiter.config().is_enabled() {
        return next.run(req).await;
    }

    // Peer addr comes from the ConnectInfo extension injected by
    // `into_make_service_with_connect_info`. Falls through to None when
    // running under a unit-test harness that doesn't set it.
    let peer = req
        .extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .map(|ci| ci.0);
    let identity = rate_limiter::extract_identity(req.headers(), peer);

    // Estimated token cost = this request's conservative upper bound.
    // We use max_seq_len as the ceiling; true-up is applied on completion
    // for streaming paths (handlers call `refund_tokens` with the actual
    // usage). This over-counts for small requests but prevents a single
    // client from consuming the whole TPM budget in one burst.
    let estimated = state.max_seq_len as u64;

    let decision = state.rate_limiter.admit(&identity, estimated);
    if !decision.allowed {
        let (param, code) = match decision.denied_by {
            Some(rate_limiter::DenialReason::Requests) => ("requests", "rate_limit_exceeded"),
            Some(rate_limiter::DenialReason::Tokens) => ("tokens", "rate_limit_exceeded"),
            None => ("", "rate_limit_exceeded"),
        };
        let body = serde_json::json!({
            "error": {
                "message": format!("Rate limit exceeded for {param}. Retry after {}s.", decision.retry_after_secs),
                "type": "rate_limit_exceeded",
                "param": param,
                "code": code,
            }
        });
        let mut resp = (StatusCode::TOO_MANY_REQUESTS, axum::Json(body)).into_response();
        apply_rate_headers(resp.headers_mut(), &decision);
        if let Ok(v) = HeaderValue::from_str(&decision.retry_after_secs.to_string()) {
            resp.headers_mut()
                .insert(HeaderName::from_static("retry-after"), v);
        }
        return resp;
    }

    // Stash the identity + reservation on the request so handlers can
    // refund the over-estimated portion once the true usage is known.
    let mut req = req;
    req.extensions_mut().insert(rate_limiter::RequestContext {
        identity: identity.clone(),
        reserved_tokens: estimated,
    });
    let mut resp = next.run(req).await;
    apply_rate_headers(resp.headers_mut(), &decision);
    resp
}

pub(crate) fn apply_rate_headers(
    headers: &mut axum::http::HeaderMap,
    d: &rate_limiter::RateDecision,
) {
    use axum::http::{HeaderName, HeaderValue};
    let set = |h: &mut axum::http::HeaderMap, k: &'static str, v: String| {
        if let Ok(val) = HeaderValue::from_str(&v) {
            h.insert(HeaderName::from_static(k), val);
        }
    };
    set(
        headers,
        "x-ratelimit-limit-requests",
        d.requests.limit.to_string(),
    );
    set(
        headers,
        "x-ratelimit-remaining-requests",
        d.requests.remaining.to_string(),
    );
    set(
        headers,
        "x-ratelimit-reset-requests",
        format!("{}s", d.requests.reset_secs),
    );
    set(
        headers,
        "x-ratelimit-limit-tokens",
        d.tokens.limit.to_string(),
    );
    set(
        headers,
        "x-ratelimit-remaining-tokens",
        d.tokens.remaining.to_string(),
    );
    set(
        headers,
        "x-ratelimit-reset-tokens",
        format!("{}s", d.tokens.reset_secs),
    );
}

// SPDX-License-Identifier: AGPL-3.0-only

//! Pure-function tests for the request middleware: the bearer gate
//! (AUTH-1), the proxy-priority rule (AUTH-3/PRI-1), the loopback rule
//! behind `--trust-proxy-headers` (AUTH-4) and the swap gate.
//! Catalogue: T-SEC-05, T-SEC-06, T-SEC-09.

use axum::http::{HeaderMap, HeaderValue, StatusCode};

use super::middleware::{
    bearer_rejection, is_loopback_bind, needs_auth, proxy_priority, swap_gated,
};
use crate::auth::AuthConfig;

fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
    let mut h = HeaderMap::new();
    for (k, v) in pairs {
        h.insert(*k, HeaderValue::from_str(v).unwrap());
    }
    h
}

/// T-SEC-05: `--trust-proxy-headers` + non-loopback bind must refuse to
/// start — this is the predicate `serve()` bails on.
#[test]
fn trust_proxy_headers_requires_a_loopback_bind() {
    for ok in ["127.0.0.1", "localhost", "::1"] {
        assert!(is_loopback_bind(ok), "{ok}");
    }
    for bad in ["0.0.0.0", "10.0.0.5", "::", "192.168.1.2", ""] {
        assert!(!is_loopback_bind(bad), "{bad}");
    }
}

/// T-SEC-06: without `--trust-proxy-headers` the header is inert.
#[test]
fn x_priority_is_ignored_without_trust() {
    assert!(!proxy_priority(false, &headers(&[("x-priority", "high")])));
    assert!(!proxy_priority(false, &headers(&[("x-priority", "HIGH")])));
}

#[test]
fn x_priority_high_is_honoured_only_when_trusted() {
    assert!(proxy_priority(true, &headers(&[("x-priority", "high")])));
    assert!(proxy_priority(true, &headers(&[("x-priority", "HIGH")])));
    assert!(!proxy_priority(true, &headers(&[("x-priority", "low")])));
    assert!(!proxy_priority(true, &headers(&[("x-priority", "")])));
    assert!(!proxy_priority(true, &HeaderMap::new()));
}

/// T-SEC-09: introspection and admin surfaces sit behind the bearer gate;
/// the scrape/liveness endpoints stay open.
#[test]
fn introspection_and_admin_paths_need_auth() {
    for p in [
        "/v1/chat/completions",
        "/v1/atlas/session/state",
        "/internal/state",
        "/admin/models",
        "/admin/models/unload",
        "/tokenize",
        "/detokenize",
        "/convert",
        "/convert/abc123",
    ] {
        assert!(needs_auth(p), "{p}");
    }
    for p in ["/health", "/health/live", "/metrics"] {
        assert!(!needs_auth(p), "{p}");
    }
}

#[test]
fn bearer_gate_distinguishes_missing_and_invalid() {
    let cfg = AuthConfig::from_inline("sk-atlas-test-token-0123456789").unwrap();
    let (s, code, _) = bearer_rejection(&cfg, &HeaderMap::new()).unwrap();
    assert_eq!((s, code), (StatusCode::UNAUTHORIZED, "missing_api_key"));
    let (s, code, _) =
        bearer_rejection(&cfg, &headers(&[("authorization", "Bearer wrong")])).unwrap();
    assert_eq!((s, code), (StatusCode::UNAUTHORIZED, "invalid_api_key"));
    assert!(
        bearer_rejection(
            &cfg,
            &headers(&[("authorization", "Bearer sk-atlas-test-token-0123456789")])
        )
        .is_none()
    );
    // Whitespace around the token is tolerated.
    assert!(
        bearer_rejection(
            &cfg,
            &headers(&[("authorization", "Bearer  sk-atlas-test-token-0123456789 ")])
        )
        .is_none()
    );
    // Any other scheme counts as absent, not as invalid.
    let (_, code, _) = bearer_rejection(&cfg, &headers(&[("authorization", "Basic abc")])).unwrap();
    assert_eq!(code, "missing_api_key");
    // Datalab clients (/convert) carry the token as X-API-Key.
    assert!(
        bearer_rejection(
            &cfg,
            &headers(&[("x-api-key", "sk-atlas-test-token-0123456789")])
        )
        .is_none()
    );
    let (_, code, _) = bearer_rejection(&cfg, &headers(&[("x-api-key", "wrong")])).unwrap();
    assert_eq!(code, "invalid_api_key");
}

/// During a swap only inference paths bounce; introspection (T-DRN-08
/// polls `/health` and `/internal/state` through a swap) and admin never do.
#[test]
fn swap_gate_spares_introspection_and_lazy_endpoints() {
    assert!(swap_gated("/v1/chat/completions", false));
    assert!(!swap_gated("/v1/chat/completions", true));
    // No lazy handler on the Anthropic path: gated even in lazy mode.
    assert!(swap_gated("/v1/messages", true));
    for p in [
        "/internal/state",
        "/v1/atlas/session/state",
        "/health",
        "/health/live",
        "/metrics",
        "/admin/models",
    ] {
        assert!(!swap_gated(p, false), "{p}");
        assert!(!swap_gated(p, true), "{p}");
    }
}

// SPDX-License-Identifier: AGPL-3.0-only

//! CTX-1: the per-request context policy. `X-Atlas-Context-Policy`
//! selects `summarize` (default), `truncate` or `reject`; without the
//! header the operator default applies. The middleware stamps the
//! resolved policy on every request as an extension, like the proxy
//! priority, so every chat surface (OpenAI, Anthropic, Responses) reads
//! the same value.

use std::sync::Arc;

use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json};

use crate::AppState;

pub const HEADER: &str = "x-atlas-context-policy";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
#[clap(rename_all = "lowercase")]
pub enum ContextPolicy {
    /// Replace older turns by a model-written summary (CMP-7).
    #[default]
    Summarize,
    /// Drop older turns without a summary (CTX-3).
    Truncate,
    /// HTTP 400 `context_length_exceeded` on overflow (CTX-2).
    Reject,
}

impl ContextPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            ContextPolicy::Summarize => "summarize",
            ContextPolicy::Truncate => "truncate",
            ContextPolicy::Reject => "reject",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "summarize" | "summarise" => Some(ContextPolicy::Summarize),
            "truncate" => Some(ContextPolicy::Truncate),
            "reject" => Some(ContextPolicy::Reject),
            _ => None,
        }
    }
}

/// Resolve the policy of a request: header value, else `default`.
/// `Err` carries the rejected header value.
pub fn from_headers(headers: &HeaderMap, default: ContextPolicy) -> Result<ContextPolicy, String> {
    match headers.get(HEADER).map(|v| v.to_str().unwrap_or("")) {
        None => Ok(default),
        Some(v) => ContextPolicy::parse(v).ok_or_else(|| v.to_string()),
    }
}

pub(crate) async fn context_policy_middleware(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    mut req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    match from_headers(req.headers(), state.compaction.default_policy) {
        Ok(p) => {
            req.extensions_mut().insert(p);
            next.run(req).await
        }
        Err(bad) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": format!(
                        "invalid X-Atlas-Context-Policy {bad:?}: expected summarize, truncate or reject"
                    ),
                    "type": "invalid_request_error",
                    "param": null,
                    "code": "invalid_context_policy",
                }
            })),
        )
            .into_response(),
    }
}

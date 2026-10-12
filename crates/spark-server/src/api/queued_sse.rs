// SPDX-License-Identifier: AGPL-3.0-only

//! WAIT-1/2/5: streaming requests that have to wait for an arena swap
//! answer IMMEDIATELY with HTTP 200 + SSE headers and a `: queued`
//! comment, tick `: waiting` keep-alives while queued (comments are
//! ignored by OpenAI-compatible clients and keep the reverse proxy's
//! read timeout from firing), and then splice the real handler's SSE
//! body into the same connection. A failure after the stream has started
//! (queue timeout, load failure, or a non-SSE error response from the
//! inner handler) is delivered as an SSE `error` event followed by
//! `[DONE]` (WAIT-5) — the HTTP status is already committed.

use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::http::{StatusCode, header};
use axum::response::Response;
use futures::StreamExt;

use crate::arena_sched::{Ticket, TicketOutcome};

type Inner = std::pin::Pin<Box<dyn Future<Output = Response> + Send>>;

/// SSE error payload in the shape streaming clients already parse.
fn sse_error_chunk(code: &str, message: &str) -> Bytes {
    let payload = serde_json::json!({
        "error": { "message": message, "type": "server_error", "code": code }
    });
    Bytes::from(format!("event: error\ndata: {payload}\n\ndata: [DONE]\n\n"))
}

/// Build the deferred-SSE response: `inner` runs only after the ticket
/// is granted and receives the queue wait in ms (for the usage headers
/// it can no longer set — the response head is already gone).
pub fn respond(ticket: Ticket, keepalive: Duration, inner: Inner) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::convert::Infallible>>(8);
    tokio::spawn(async move {
        // WAIT-1: the client sees bytes immediately.
        if tx
            .send(Ok(Bytes::from_static(b": queued\n\n")))
            .await
            .is_err()
        {
            return;
        }
        let outcome = loop {
            tokio::select! {
                o = ticket.wait() => break o,
                _ = tokio::time::sleep(keepalive) => {
                    // WAIT-2 keep-alive; a send failure means the client
                    // hung up — the ticket drops with us (DRN-5).
                    if tx.send(Ok(Bytes::from_static(b": waiting\n\n"))).await.is_err() {
                        return;
                    }
                }
            }
        };
        match outcome {
            TicketOutcome::Granted => {
                let resp = inner.await;
                forward_or_wrap(resp, &tx).await;
            }
            TicketOutcome::Expired => {
                let _ = tx
                    .send(Ok(sse_error_chunk(
                        "model_wait_timeout",
                        "queue wait bound exceeded; retry later",
                    )))
                    .await;
            }
            TicketOutcome::Rejected => {
                let _ = tx
                    .send(Ok(sse_error_chunk(
                        "model_load_failed",
                        "the requested model failed to load; the previous model was restored",
                    )))
                    .await;
            }
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        ))
        .expect("static SSE response head")
}

/// CMP-10 / WAIT-2: a streaming request whose history must be compacted
/// first answers with the SSE head immediately and ticks `: compacting`
/// while the summary is generated; the real stream follows. The
/// compaction id travels as an SSE comment because the response head
/// (and with it `X-Atlas-Compaction-Id`) is already on the wire.
pub fn respond_while_compacting(keepalive: Duration, mut inner: Inner) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::convert::Infallible>>(8);
    tokio::spawn(async move {
        if tx
            .send(Ok(Bytes::from_static(b": compacting\n\n")))
            .await
            .is_err()
        {
            return;
        }
        let resp = loop {
            tokio::select! {
                r = &mut inner => break r,
                _ = tokio::time::sleep(keepalive) => {
                    if tx.send(Ok(Bytes::from_static(b": compacting\n\n"))).await.is_err() {
                        return;
                    }
                }
            }
        };
        if let Some(id) = resp
            .headers()
            .get(crate::compaction::request::COMPACTION_ID_HEADER)
            .and_then(|v| v.to_str().ok())
        {
            let note = format!(": compacted id={id}\n\n");
            if tx.send(Ok(Bytes::from(note))).await.is_err() {
                return;
            }
        }
        forward_or_wrap(resp, &tx).await;
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        ))
        .expect("static SSE response head")
}

/// Splice the inner handler's response into the open SSE connection:
/// an SSE body is forwarded verbatim; anything else (a JSON error from
/// validation/tokenization) becomes an SSE `error` event (WAIT-5).
async fn forward_or_wrap(
    resp: Response,
    tx: &tokio::sync::mpsc::Sender<Result<Bytes, std::convert::Infallible>>,
) {
    let is_sse = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("text/event-stream"));
    if is_sse {
        let mut body = resp.into_body().into_data_stream();
        while let Some(chunk) = body.next().await {
            match chunk {
                Ok(bytes) => {
                    if tx.send(Ok(bytes)).await.is_err() {
                        return;
                    }
                }
                Err(e) => {
                    let _ = tx
                        .send(Ok(sse_error_chunk("stream_error", &format!("{e}"))))
                        .await;
                    return;
                }
            }
        }
    } else {
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap_or_default();
        let msg = String::from_utf8_lossy(&bytes);
        let _ = tx
            .send(Ok(sse_error_chunk(
                &format!("http_{}", status.as_u16()),
                msg.trim(),
            )))
            .await;
    }
}

#[cfg(test)]
#[path = "queued_sse_tests.rs"]
mod tests;

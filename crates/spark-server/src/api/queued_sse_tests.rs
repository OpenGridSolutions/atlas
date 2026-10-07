// SPDX-License-Identifier: AGPL-3.0-only

//! WAIT-1/2/5 tests for the deferred SSE response: immediate `: queued`,
//! `: waiting` keep-alives, and in-stream `error` events once the HTTP
//! status is committed. Catalogue: T-WAIT-01, T-WAIT-06 and the
//! server half of T-WAIT-02.

use std::time::Duration;

use axum::body::{Body, BodyDataStream, Bytes};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;

use super::{Inner, respond};
use crate::arena_sched::ArenaQueues;

fn sse_inner(chunks: &'static [&'static str]) -> Inner {
    Box::pin(async move {
        let stream = futures::stream::iter(
            chunks
                .iter()
                .map(|c| Ok::<_, std::convert::Infallible>(Bytes::from_static(c.as_bytes()))),
        );
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(Body::from_stream(stream))
            .unwrap()
    })
}

fn json_error_inner(status: StatusCode, body: &'static str) -> Inner {
    Box::pin(
        async move { (status, [(header::CONTENT_TYPE, "application/json")], body).into_response() },
    )
}

fn open(resp: Response) -> BodyDataStream {
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/event-stream"
    );
    resp.into_body().into_data_stream()
}

async fn next_chunk(stream: &mut BodyDataStream) -> Option<String> {
    let chunk = tokio::time::timeout(Duration::from_secs(2), stream.next())
        .await
        .expect("a chunk within 2 s")?;
    Some(String::from_utf8(chunk.expect("body error").to_vec()).unwrap())
}

async fn drain(stream: &mut BodyDataStream) -> String {
    let mut out = String::new();
    while let Some(c) = next_chunk(stream).await {
        out.push_str(&c);
    }
    out
}

/// The JSON payload of the SSE `error` event in `text`.
fn error_payload(text: &str) -> serde_json::Value {
    let line = text
        .lines()
        .skip_while(|l| *l != "event: error")
        .find(|l| l.starts_with("data: {"))
        .unwrap_or_else(|| panic!("no error event in {text:?}"));
    serde_json::from_str(&line["data: ".len()..]).unwrap()
}

/// T-WAIT-01: HTTP 200 + SSE head + `: queued` arrive before the grant;
/// the inner SSE body is spliced in verbatim afterwards.
#[tokio::test]
async fn queued_comment_arrives_before_the_grant() {
    let q = ArenaQueues::new(8, 0.2);
    let ticket = q.submit("m", false, true).unwrap();
    let inner = sse_inner(&["data: {\"a\":1}\n\n", "data: [DONE]\n\n"]);
    let mut body = open(respond(ticket, Duration::from_secs(30), inner));
    assert_eq!(next_chunk(&mut body).await.as_deref(), Some(": queued\n\n"));
    q.grant_all("m");
    assert_eq!(
        drain(&mut body).await,
        "data: {\"a\":1}\n\ndata: [DONE]\n\n"
    );
}

/// T-WAIT-02 (server side): keep-alive comments tick at the configured
/// cadence while the ticket waits, and stop once it is granted.
#[tokio::test]
async fn keepalive_comments_tick_while_queued() {
    let q = ArenaQueues::new(8, 0.2);
    let ticket = q.submit("m", false, true).unwrap();
    let inner = sse_inner(&["data: [DONE]\n\n"]);
    let mut body = open(respond(ticket, Duration::from_millis(20), inner));
    assert_eq!(next_chunk(&mut body).await.as_deref(), Some(": queued\n\n"));
    assert_eq!(
        next_chunk(&mut body).await.as_deref(),
        Some(": waiting\n\n")
    );
    assert_eq!(
        next_chunk(&mut body).await.as_deref(),
        Some(": waiting\n\n")
    );
    q.grant_all("m");
    let rest = drain(&mut body).await;
    assert!(rest.ends_with("data: [DONE]\n\n"), "{rest:?}");
    assert!(!rest.contains("event: error"), "{rest:?}");
}

/// T-WAIT-06 (PRI-4 arm): an expired ticket ends the stream with an SSE
/// `error` event and `[DONE]`; the inner handler never runs.
#[tokio::test]
async fn expired_ticket_ends_the_stream_with_an_sse_error() {
    let q = ArenaQueues::new(8, 0.2);
    // Priority ticket for a model that is not resident → PRI-4 expiry.
    let ticket = q.submit("other", true, true).unwrap();
    let inner = sse_inner(&["data: never\n\n"]);
    let mut body = open(respond(ticket, Duration::from_secs(30), inner));
    assert_eq!(next_chunk(&mut body).await.as_deref(), Some(": queued\n\n"));
    tokio::time::sleep(Duration::from_millis(5)).await;
    let expired = q.expire_sweep(Some("m"), Duration::from_millis(1), Duration::from_secs(60));
    assert_eq!(expired, 1);
    let rest = drain(&mut body).await;
    assert_eq!(error_payload(&rest)["error"]["code"], "model_wait_timeout");
    assert!(rest.ends_with("data: [DONE]\n\n"), "{rest:?}");
    assert!(!rest.contains("never"));
}

/// T-WAIT-06 (MOD-10 arm): a failed model load rejects the ticket; the
/// client sees `model_load_failed` in-stream.
#[tokio::test]
async fn rejected_ticket_reports_the_failed_load() {
    let q = ArenaQueues::new(8, 0.2);
    let ticket = q.submit("m", false, true).unwrap();
    let inner = sse_inner(&["data: never\n\n"]);
    let mut body = open(respond(ticket, Duration::from_secs(30), inner));
    assert_eq!(next_chunk(&mut body).await.as_deref(), Some(": queued\n\n"));
    assert_eq!(q.reject_all("m"), 1);
    let rest = drain(&mut body).await;
    assert_eq!(error_payload(&rest)["error"]["code"], "model_load_failed");
    assert!(rest.ends_with("data: [DONE]\n\n"), "{rest:?}");
}

/// WAIT-5: a non-SSE inner response (validation error after the grant)
/// cannot change the committed status — it becomes an `error` event
/// carrying the original status and body.
#[tokio::test]
async fn non_sse_inner_response_is_wrapped_as_an_error_event() {
    let q = ArenaQueues::new(8, 0.2);
    let ticket = q.submit("m", false, true).unwrap();
    let inner = json_error_inner(
        StatusCode::BAD_REQUEST,
        r#"{"error":{"message":"prompt too long"}}"#,
    );
    let mut body = open(respond(ticket, Duration::from_secs(30), inner));
    assert_eq!(next_chunk(&mut body).await.as_deref(), Some(": queued\n\n"));
    q.grant_all("m");
    let rest = drain(&mut body).await;
    let payload = error_payload(&rest);
    assert_eq!(payload["error"]["code"], "http_400");
    assert!(
        payload["error"]["message"]
            .as_str()
            .unwrap()
            .contains("prompt too long")
    );
    assert!(rest.ends_with("data: [DONE]\n\n"), "{rest:?}");
}

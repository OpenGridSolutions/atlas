// SPDX-License-Identifier: AGPL-3.0-only

//! Canonical JSON of one chat message (CMP-1, decision 4 of
//! 2026-10-06). Two clients that send the same conversation must derive
//! the same message keys, whatever their JSON serialiser does, so the
//! bytes that feed the hash are normalised here:
//!
//! - object keys sorted, no whitespace, `serde_json` string escaping;
//! - every text NFC-normalised;
//! - `content` is always `{"images":[...],"text":"..."}` — a plain
//!   string body and a one-part array body are the same message; an
//!   image is represented by the BLAKE3 hex of its raw reference (data
//!   URI or URL) so the key never depends on decoding the image;
//! - `tool_calls` entries are `{"arguments":<canonical JSON when the
//!   argument string parses, else the string>,"id":...,"name":...}`;
//! - `name` and `tool_call_id` appear only when present;
//! - `reasoning_content` is NOT part of the key: it is a rendering aid
//!   for historical turns, not content the conversation depends on.

use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

use crate::openai::IncomingMessage;

/// Canonical bytes of `m` (as a `String`; always valid JSON).
pub fn canonical_message(m: &IncomingMessage) -> String {
    let mut obj = serde_json::Map::new();
    obj.insert("role".into(), Value::String(nfc(&m.role)));
    let images: Vec<Value> = m
        .content
        .images
        .iter()
        .map(|i| Value::String(blake3::hash(i.as_bytes()).to_hex().to_string()))
        .collect();
    let mut content = serde_json::Map::new();
    content.insert("images".into(), Value::Array(images));
    content.insert("text".into(), Value::String(nfc(&m.content.text)));
    obj.insert("content".into(), Value::Object(content));
    if let Some(tcs) = &m.tool_calls {
        let calls: Vec<Value> = tcs
            .iter()
            .map(|tc| {
                let mut c = serde_json::Map::new();
                let args = match serde_json::from_str::<Value>(&tc.function.arguments) {
                    Ok(v) => nfc_value(v),
                    Err(_) => Value::String(nfc(&tc.function.arguments)),
                };
                c.insert("arguments".into(), args);
                c.insert(
                    "id".into(),
                    tc.id
                        .as_deref()
                        .map(|s| Value::String(nfc(s)))
                        .unwrap_or(Value::Null),
                );
                c.insert("name".into(), Value::String(nfc(&tc.function.name)));
                Value::Object(c)
            })
            .collect();
        obj.insert("tool_calls".into(), Value::Array(calls));
    }
    if let Some(id) = &m.tool_call_id {
        obj.insert("tool_call_id".into(), Value::String(nfc(id)));
    }
    if let Some(name) = &m.name {
        obj.insert("name".into(), Value::String(nfc(name)));
    }
    crate::tool_arg_dedup::canonical_value_to_string(&Value::Object(obj))
}

fn nfc(s: &str) -> String {
    s.nfc().collect()
}

/// NFC-normalise every string inside a JSON value (keys included).
fn nfc_value(v: Value) -> Value {
    match v {
        Value::String(s) => Value::String(nfc(&s)),
        Value::Array(a) => Value::Array(a.into_iter().map(nfc_value).collect()),
        Value::Object(o) => Value::Object(
            o.into_iter()
                .map(|(k, v)| (nfc(&k), nfc_value(v)))
                .collect(),
        ),
        other => other,
    }
}

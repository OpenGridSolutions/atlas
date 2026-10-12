// SPDX-License-Identifier: AGPL-3.0-only

use super::canonical::canonical_message;
use super::chain::message_chain;
use super::effective::{SUMMARY_OPEN, build_effective};
use super::entry::{CompactionEntry, FORMAT_VERSION};
use super::head::head_len;
use super::index::CompactionIndex;
use super::plan::{choose_cut, cut_allowed};
use super::policy::{ContextPolicy, from_headers};
use super::summarize::strip_think;
use crate::openai::{IncomingMessage, ParsedContent};
use crate::tool_parser::{IncomingFunction, IncomingToolCall};

fn msg(role: &str, text: &str) -> IncomingMessage {
    IncomingMessage {
        role: role.to_string(),
        content: ParsedContent {
            text: text.to_string(),
            images: Vec::new(),
        },
        tool_calls: None,
        tool_call_id: None,
        name: None,
        reasoning_content: None,
    }
}

fn tool_call(args: &str) -> IncomingMessage {
    let mut m = msg("assistant", "");
    m.tool_calls = Some(vec![IncomingToolCall {
        id: Some("call_1".into()),
        function: IncomingFunction {
            name: "read".into(),
            arguments: args.into(),
        },
    }]);
    m
}

fn tool_result(text: &str) -> IncomingMessage {
    let mut m = msg("tool", text);
    m.tool_call_id = Some("call_1".into());
    m.name = Some("read".into());
    m
}

// ── CMP-1 / decision 4: canonical serialisation ──

#[test]
fn canonical_test_vectors() {
    assert_eq!(
        canonical_message(&msg("user", "hi")),
        r#"{"content":{"images":[],"text":"hi"},"role":"user"}"#
    );
    // Tool-call arguments are re-serialised with sorted keys; id/name
    // carried; tool_call_id and name only when present.
    assert_eq!(
        canonical_message(&tool_call(r#"{"path": "b", "a": 1}"#)),
        r#"{"content":{"images":[],"text":""},"role":"assistant","tool_calls":[{"arguments":{"a":1,"path":"b"},"id":"call_1","name":"read"}]}"#
    );
    assert_eq!(
        canonical_message(&tool_result("ok")),
        r#"{"content":{"images":[],"text":"ok"},"name":"read","role":"tool","tool_call_id":"call_1"}"#
    );
    // Arguments that are not JSON stay a string.
    assert!(canonical_message(&tool_call("not json")).contains(r#""arguments":"not json""#));
}

#[test]
fn canonical_normalises_nfc_and_ignores_reasoning() {
    let composed = msg("user", "r\u{e9}sum\u{e9}");
    let decomposed = msg("user", "re\u{301}sume\u{301}");
    assert_eq!(canonical_message(&composed), canonical_message(&decomposed));
    let mut with_reasoning = msg("assistant", "answer");
    with_reasoning.reasoning_content = Some("thoughts".into());
    assert_eq!(
        canonical_message(&with_reasoning),
        canonical_message(&msg("assistant", "answer"))
    );
}

#[test]
fn canonical_images_are_digests_of_the_raw_reference() {
    let mut a = msg("user", "look");
    a.content.images.push("data:image/png;base64,AAAA".into());
    let c = canonical_message(&a);
    let digest = blake3::hash(b"data:image/png;base64,AAAA")
        .to_hex()
        .to_string();
    assert!(c.contains(&digest));
    assert!(!c.contains("AAAA\""));
}

// ── CMP-1 / CMP-12: the chain ──

#[test]
fn chain_vectors_are_stable() {
    let keys = message_chain("test-model", &[msg("system", "s"), msg("user", "u")]);
    assert_eq!(keys.len(), 2);
    // Pinned so that a serialisation change is noticed (entries on NVMe
    // would silently stop matching).
    assert_eq!(keys[0].to_hex(), "b83adb31fb739444628cf92af5be93df");
    assert_eq!(keys[1].to_hex(), "f856f95f1e185dde0a04f4e689b02502");
}

#[test]
fn chain_edit_changes_only_later_keys_and_depends_on_model() {
    let a = message_chain(
        "m",
        &[msg("user", "1"), msg("assistant", "2"), msg("user", "3")],
    );
    let b = message_chain(
        "m",
        &[msg("user", "1"), msg("assistant", "2x"), msg("user", "3")],
    );
    assert_eq!(a[0], b[0]);
    assert_ne!(a[1], b[1]);
    assert_ne!(a[2], b[2]);
    let c = message_chain("other", &[msg("user", "1")]);
    assert_ne!(a[0], c[0]);
}

// ── CMP-4: head ──

#[test]
fn head_detection_cases() {
    let sys = |n: usize| {
        (0..n)
            .map(|i| msg("system", &format!("s{i}")))
            .collect::<Vec<_>>()
    };
    let mut two_sys = sys(2);
    two_sys.push(msg("user", "u"));
    two_sys.push(msg("assistant", "a"));
    assert_eq!(head_len(&two_sys, true), 3);
    assert_eq!(head_len(&two_sys, false), 2);
    let dev = [msg("developer", "d"), msg("user", "u"), msg("user", "u2")];
    assert_eq!(head_len(&dev, true), 2);
    let none = [msg("user", "u"), msg("assistant", "a")];
    assert_eq!(head_len(&none, true), 1);
    assert_eq!(head_len(&none, false), 0);
    let sys_only = sys(1);
    assert_eq!(head_len(&sys_only, true), 1);
}

// ── CMP-5: cut selection ──

#[test]
fn cut_never_separates_a_tool_call_from_its_result() {
    let roles = [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
        "assistant",
    ];
    assert!(
        !cut_allowed(&roles, 2),
        "after the call, before its results"
    );
    assert!(!cut_allowed(&roles, 3), "between two results");
    assert!(cut_allowed(&roles, 4));
    assert!(cut_allowed(&roles, 1));
    assert!(!cut_allowed(&roles, 6), "nothing may remain after the cut");
}

#[test]
fn cut_picks_minimal_tail_reaching_target_then_shrinks_to_budget() {
    let roles = ["system", "user", "assistant", "user", "assistant", "user"];
    let sizes = [10, 100, 100, 100, 100, 50];
    // Target 200: tail {4,5} = 150 < 200, {3,4,5} = 250 ≥ 200 → cut 2.
    let p = choose_cut(&roles, &sizes, 1, 200, 10_000).unwrap();
    assert_eq!(p.cut, 2);
    assert_eq!(p.tail_tokens, 250);
    assert_eq!(p.replaced_tokens, 200);
    // Budget 160: {3,4,5} too big → shrink to {4,5} = 150 → cut 3.
    let p = choose_cut(&roles, &sizes, 1, 200, 160).unwrap();
    assert_eq!(p.cut, 3);
    // Budget below every tail: the newest unit survives (cut 4).
    let p = choose_cut(&roles, &sizes, 1, 200, 10).unwrap();
    assert_eq!(p.cut, 4);
    assert_eq!(p.tail_tokens, 50);
    // Cumulative: first = 4 leaves nothing to fold.
    assert!(choose_cut(&roles, &sizes, 5, 200, 10_000).is_none());
    assert!(choose_cut(&roles, &sizes, 4, 200, 10_000).is_some());
}

#[test]
fn cut_respects_tool_pairs_when_shrinking() {
    let roles = ["user", "assistant", "tool", "user"];
    let sizes = [10, 10, 1000, 10];
    // Cut 1 (between call and result) is forbidden; the only allowed
    // cuts are 0 and 2.
    let p = choose_cut(&roles, &sizes, 0, 5, 20).unwrap();
    assert_eq!(p.cut, 2);
}

// ── CMP-2 / CMP-6 / CTX-3: effective context ──

#[test]
fn effective_appends_summary_to_head_system_message() {
    let raw = [
        msg("system", "rules"),
        msg("user", "u1"),
        msg("assistant", "a1"),
        msg("user", "u2"),
    ];
    let eff = build_effective(&raw, 1, 2, "the gist");
    assert_eq!(eff.len(), 2);
    assert_eq!(eff[0].role, "system");
    assert!(eff[0].content.text.starts_with("rules\n\n"));
    assert!(eff[0].content.text.contains(SUMMARY_OPEN));
    assert!(eff[0].content.text.contains("the gist"));
    assert_eq!(eff[1].content.text, "u2");
}

#[test]
fn effective_creates_system_message_when_head_has_none() {
    let raw = [msg("user", "u1"), msg("assistant", "a1"), msg("user", "u2")];
    let eff = build_effective(&raw, 1, 1, "gist");
    assert_eq!(eff.len(), 3);
    assert_eq!(eff[0].role, "system");
    assert_eq!(eff[1].content.text, "u1");
    assert_eq!(eff[2].content.text, "u2");
}

#[test]
fn effective_truncate_adds_no_section() {
    let raw = [
        msg("system", "rules"),
        msg("user", "u1"),
        msg("assistant", "a1"),
        msg("user", "u2"),
    ];
    let eff = build_effective(&raw, 1, 2, "");
    assert_eq!(eff[0].content.text, "rules");
    assert_eq!(eff.len(), 2);
}

// ── CMP-11: entry ──

#[test]
fn entry_roundtrip_and_version_gate() {
    let e = CompactionEntry {
        version: FORMAT_VERSION,
        key: "00112233445566778899aabbccddeeff".into(),
        model: "m".into(),
        policy: "summarize".into(),
        head_len: 1,
        cut: 4,
        replaced_messages: 4,
        replaced_tokens: 1234,
        summary: "s".into(),
        pred: None,
        created_at: 7,
    };
    let back = CompactionEntry::from_json_bytes(&e.to_json_bytes()).unwrap();
    assert_eq!(back, e);
    assert_eq!(back.pred_bytes(), spark_runtime::block_keys::BlockKey::ZERO);
    let mut newer = e.clone();
    newer.version = FORMAT_VERSION + 1;
    assert!(CompactionEntry::from_json_bytes(&newer.to_json_bytes()).is_none());
}

#[test]
fn index_finds_latest_entry_and_bounds_ram() {
    let idx = CompactionIndex::new(2);
    let keys = message_chain(
        "m",
        &[
            msg("system", "s"),
            msg("user", "1"),
            msg("assistant", "2"),
            msg("user", "3"),
        ],
    );
    let mk = |k: &spark_runtime::block_keys::BlockKey, cut| {
        std::sync::Arc::new(CompactionEntry {
            version: FORMAT_VERSION,
            key: k.to_hex(),
            model: "m".into(),
            policy: "summarize".into(),
            head_len: 1,
            cut,
            replaced_messages: cut,
            replaced_tokens: 0,
            summary: "s".into(),
            pred: None,
            created_at: 0,
        })
    };
    assert!(idx.find_latest(&keys, 1).is_none());
    idx.insert(mk(&keys[1], 1));
    idx.insert(mk(&keys[2], 2));
    let (c, e) = idx.find_latest(&keys, 1).unwrap();
    assert_eq!(c, 2);
    assert_eq!(e.cut, 2);
    // Capacity 2: a third entry evicts the oldest.
    idx.insert(mk(&keys[3], 3));
    assert_eq!(idx.ram_entries(), 2);
    assert!(idx.get(&keys[1]).is_none());
    // An edited history (different key at index 2) misses — CMP-12.
    let edited = message_chain(
        "m",
        &[
            msg("system", "s"),
            msg("user", "1"),
            msg("assistant", "2!"),
            msg("user", "3"),
        ],
    );
    assert!(idx.find_latest(&edited, 1).is_none());
}

// ── CTX-1: policy ──

#[test]
fn policy_parsing_and_header_resolution() {
    assert_eq!(
        ContextPolicy::parse("Summarize"),
        Some(ContextPolicy::Summarize)
    );
    assert_eq!(
        ContextPolicy::parse("summarise"),
        Some(ContextPolicy::Summarize)
    );
    assert_eq!(
        ContextPolicy::parse("truncate"),
        Some(ContextPolicy::Truncate)
    );
    assert_eq!(ContextPolicy::parse("reject"), Some(ContextPolicy::Reject));
    assert_eq!(ContextPolicy::parse("drop"), None);
    let mut h = axum::http::HeaderMap::new();
    assert_eq!(
        from_headers(&h, ContextPolicy::Truncate),
        Ok(ContextPolicy::Truncate)
    );
    h.insert("x-atlas-context-policy", "reject".parse().unwrap());
    assert_eq!(
        from_headers(&h, ContextPolicy::Summarize),
        Ok(ContextPolicy::Reject)
    );
    h.insert("x-atlas-context-policy", "bogus".parse().unwrap());
    assert_eq!(
        from_headers(&h, ContextPolicy::Summarize),
        Err("bogus".to_string())
    );
}

#[test]
fn strip_think_removes_forced_reasoning() {
    assert_eq!(strip_think("<think>hmm</think>\n\nSummary."), "Summary.");
    assert_eq!(strip_think("  Summary. "), "Summary.");
}

#[test]
fn fit_rule_uses_bounded_reserve() {
    let cfg = super::CompactionConfig {
        default_policy: ContextPolicy::Summarize,
        trigger_pct: 70,
        tail_pct: 25,
        summary_max_tokens: 768,
        output_reserve: 2048,
        keep_first_user: true,
        prompt: "p".into(),
    };
    assert!(
        cfg.fits(30_000, 4096, 32_768),
        "reserve is min(max_tokens, 2048)"
    );
    assert!(!cfg.fits(30_721, 4096, 32_768));
    assert!(cfg.fits(32_000, 100, 32_768));
    assert_eq!(cfg.tail_target(32_768), 8192);
    assert_eq!(
        cfg.tail_budget(32_768, 4096, 1000),
        32_768 - 2048 - 1000 - 768
    );
}

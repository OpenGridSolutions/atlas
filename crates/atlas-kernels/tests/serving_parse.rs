// SPDX-License-Identifier: AGPL-3.0-only

//! `[serving].sessions` in MODEL.toml: the number of sessions the server
//! keeps warm in the KV pool for a model (the context window per session
//! is derived from it at load time). Exercises the build-script parser
//! directly, so it runs without compiled kernel targets.

#[path = "../build_serving.rs"]
mod build_serving;

use build_serving::{parse_serving, serving_moe_prefill_transpose, serving_sessions};

fn kernel_dir(model: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10")
        .join(model)
}

#[test]
fn sessions_are_read_from_the_serving_section() {
    let toml = "[model]\nname = \"x\"\n\n[serving]\nsessions = 11\n";
    assert_eq!(serving_sessions(toml, "test"), Some(11));
}

#[test]
fn absent_section_means_no_session_count() {
    assert_eq!(serving_sessions("[model]\nname = \"x\"\n", "test"), None);
}

#[test]
#[should_panic(expected = "[serving].sessions must be an integer")]
fn section_without_sessions_is_a_build_error() {
    serving_sessions("[serving]\n", "test");
}

#[test]
#[should_panic(expected = "[serving].sessions must be an integer")]
fn non_integer_sessions_is_a_build_error() {
    serving_sessions("[serving]\nsessions = \"many\"\n", "test");
}

#[test]
#[should_panic(expected = "[serving].sessions must be at least 1")]
fn zero_sessions_is_a_build_error() {
    serving_sessions("[serving]\nsessions = 0\n", "test");
}

#[test]
#[should_panic(expected = "unknown key [serving].session ")]
fn misspelled_serving_key_is_a_build_error() {
    serving_sessions("[serving]\nsession = 4\n", "test");
}

#[test]
fn moe_prefill_transpose_is_read_from_the_serving_section() {
    let toml = "[serving]\nsessions = 11\nmoe_prefill_transpose = false\n";
    assert_eq!(serving_moe_prefill_transpose(toml, "test"), Some(false));
    assert_eq!(serving_sessions(toml, "test"), Some(11));
}

#[test]
fn absent_moe_prefill_transpose_leaves_the_decision_to_the_loader() {
    let toml = "[serving]\nsessions = 11\n";
    assert_eq!(serving_moe_prefill_transpose(toml, "test"), None);
    assert_eq!(
        serving_moe_prefill_transpose("[model]\nname = \"x\"\n", "test"),
        None
    );
}

#[test]
#[should_panic(expected = "[serving].moe_prefill_transpose must be true or false")]
fn non_boolean_moe_prefill_transpose_is_a_build_error() {
    serving_moe_prefill_transpose(
        "[serving]\nsessions = 4\nmoe_prefill_transpose = 0\n",
        "test",
    );
}

/// Every token-serving target of the GB10 deployment declares its session
/// count; the non-token models (speech, image) have none.
#[test]
fn deployment_targets_declare_sessions() {
    for model in [
        "qwen3-next-80b-a3b",
        "qwen3-vl-30b-a3b",
        "qwen3.8-27b",
        "chandra-ocr-2",
        "qwen3-dense-8b",
        "qwen3-vl-dense-8b",
        "translategemma-12b",
    ] {
        assert!(
            parse_serving(&kernel_dir(model)).sessions.is_some(),
            "kernels/gb10/{model}/MODEL.toml has no [serving].sessions"
        );
    }
    for model in ["whisper-large-v3", "qwen-image-2.1"] {
        assert_eq!(
            parse_serving(&kernel_dir(model)).sessions,
            None,
            "kernels/gb10/{model}/MODEL.toml: a non-token model has no sessions"
        );
    }
}

/// The 80B's transposed MoE copies (40.5 GiB) would take most of its KV
/// pool, so its MODEL.toml pins the decision instead of leaving it to the
/// free memory at load time.
#[test]
fn qwen3_next_80b_pins_moe_prefill_transpose_off() {
    assert_eq!(
        parse_serving(&kernel_dir("qwen3-next-80b-a3b")).moe_prefill_transpose,
        Some(false)
    );
}

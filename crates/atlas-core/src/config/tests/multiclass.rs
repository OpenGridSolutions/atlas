// SPDX-License-Identifier: AGPL-3.0-only

//! Config-parse tests for the multiclass models (Plan.md 1.1, config
//! half): dense `qwen3` (Embedding/Reranker-8B) and dense `qwen3_vl`
//! (VL-Embedding/Reranker-8B). Fixtures are verbatim copies of the HF
//! checkpoints' config.json (test_data/), so these tests reproduce the
//! exact "missing field hidden_size at line 65" boot crash the flat
//! fallback arm produced for qwen3_vl before the dedicated parsers.
//!
//! Logical child of `config::tests` via `#[path]`; `use super::*`
//! resolves to config.rs items.

use super::*;

fn fixture(name: &str) -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data")
            .join(name),
    )
    .unwrap_or_else(|e| panic!("fixture {name}: {e}"))
}

/// Shared assertions for the dense 8B text tower — all four checkpoints
/// carry the identical geometry.
fn assert_dense_8b_text(cfg: &ModelConfig) {
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.num_hidden_layers, 36);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.intermediate_size, 12288);
    // Standard Qwen3 dense attention has q/k norms but NO output gate —
    // only Qwen3-Next gates. The generic flat arm would set true.
    assert!(!cfg.attn_gated);
    // Pure attention: every layer FullAttention, no SSM, no MoE.
    assert_eq!(cfg.layer_types.len(), cfg.num_hidden_layers);
    assert_eq!(cfg.num_attention_layers(), 36);
    assert_eq!(cfg.num_ssm_layers(), 0);
    assert_eq!(cfg.num_experts, 0);
    // Stock BF16 checkpoints — no quantization_config.
    assert!(cfg.quantization_config.is_none());
}

#[test]
fn parse_qwen3_embedding_8b() {
    let cfg = parse_config(&fixture("qwen3_embedding_8b_config.json")).unwrap();
    assert_eq!(cfg.model_type, "qwen3");
    assert_dense_8b_text(&cfg);
    assert_eq!(cfg.vocab_size, 151665);
    assert!(cfg.vision.is_none());
}

#[test]
fn parse_qwen3_reranker_8b() {
    let cfg = parse_config(&fixture("qwen3_reranker_8b_config.json")).unwrap();
    assert_eq!(cfg.model_type, "qwen3");
    assert_dense_8b_text(&cfg);
    assert_eq!(cfg.vocab_size, 151669);
    assert!(cfg.vision.is_none());
}

#[test]
fn parse_qwen3_vl_embedding_8b() {
    // THE boot crasher: no top-level hidden_size (nested in text_config),
    // 65-line config.json → flat arm died with "missing field hidden_size
    // at line 65".
    let cfg = parse_config(&fixture("qwen3_vl_embedding_8b_config.json")).unwrap();
    assert_eq!(cfg.model_type, "qwen3_vl");
    assert_dense_8b_text(&cfg);
    assert!(cfg.nested_config);
    assert_eq!(cfg.eos_token_id, 151645);
    // MRoPE from text_config.rope_scaling (NOT rope_parameters).
    assert_eq!(cfg.mrope_section, [24, 20, 20]);
    assert!(cfg.mrope_interleaved);
    // ViT: 27 blocks, hidden 1152, deepstack [8,16,24], pad token 151655.
    let v = cfg.vision.as_ref().expect("vision tower must parse");
    assert_eq!(v.depth, 27);
    assert_eq!(v.hidden_size, 1152);
    assert_eq!(v.deepstack_visual_indexes, vec![8, 16, 24]);
    assert_eq!(v.out_hidden_size, 4096);
    assert_eq!(v.image_pad_token_id, 151655);
}

#[test]
fn parse_qwen3_vl_reranker_8b() {
    // Quirk: this checkpoint ships a stray TOP-LEVEL hidden_size next to
    // the nested one — the flat arm would have "succeeded" into a garbage
    // config (0 layers). The qwen3_vl arm must read text_config.
    let cfg = parse_config(&fixture("qwen3_vl_reranker_8b_config.json")).unwrap();
    assert_eq!(cfg.model_type, "qwen3_vl");
    assert_dense_8b_text(&cfg);
    assert!(cfg.nested_config);
    assert_eq!(cfg.mrope_section, [24, 20, 20]);
    let v = cfg.vision.as_ref().expect("vision tower must parse");
    assert_eq!(v.depth, 27);
    assert_eq!(v.deepstack_visual_indexes, vec![8, 16, 24]);
}

/// Parse the LIVE config.json of every cached model (the full 6-model
/// preseed list) straight from the HF cache. Skips models that are not
/// cached (CI without the cache stays green); on the box this is the
/// end-to-end acceptance check for the boot crash.
#[test]
fn parse_all_cached_model_configs() {
    let hub = std::env::var("HF_HOME")
        .map(|h| std::path::PathBuf::from(h).join("hub"))
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join(".cache/huggingface/hub")
        });
    // (repo dir name, expected model_type, expected hidden_size)
    let expected = [
        ("models--nvidia--Qwen3-Next-80B-A3B-Thinking-NVFP4", "qwen3_next", 2048),
        ("models--vincentzed-hf--Qwen3-Coder-Next-NVFP4", "qwen3_next", 2048),
        ("models--Qwen--Qwen3-Embedding-8B", "qwen3", 4096),
        ("models--Qwen--Qwen3-Reranker-8B", "qwen3", 4096),
        ("models--Qwen--Qwen3-VL-Embedding-8B", "qwen3_vl", 4096),
        ("models--Qwen--Qwen3-VL-Reranker-8B", "qwen3_vl", 4096),
    ];
    let mut parsed = 0;
    for (repo, model_type, hidden) in expected {
        let snap_root = hub.join(repo).join("snapshots");
        let Ok(mut snaps) = std::fs::read_dir(&snap_root) else {
            eprintln!("skip (not cached): {repo}");
            continue;
        };
        let Some(Ok(snap)) = snaps.next() else {
            eprintln!("skip (no snapshot): {repo}");
            continue;
        };
        let json = std::fs::read_to_string(snap.path().join("config.json"))
            .unwrap_or_else(|e| panic!("{repo}: reading config.json: {e}"));
        let cfg = parse_config(&json).unwrap_or_else(|e| panic!("{repo}: parse failed: {e:#}"));
        assert_eq!(cfg.model_type, model_type, "{repo}: model_type");
        assert_eq!(cfg.hidden_size, hidden, "{repo}: hidden_size");
        assert!(cfg.num_hidden_layers > 0, "{repo}: num_hidden_layers");
        parsed += 1;
    }
    eprintln!("parsed {parsed}/{} cached model configs", expected.len());
}

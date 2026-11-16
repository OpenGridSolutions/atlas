// SPDX-License-Identifier: AGPL-3.0-only

//! `qwen4_exp` (Qwen3.8-Flash-Next) config parsing. The fixture is the
//! `text_config` of nvidia/Qwen3.8-Flash-Next-FP4-Local (revision b9ce18a6)
//! with `layer_types` generated and the vision/quantization sections dropped.

use super::super::*;

fn fixture(edit: impl Fn(&mut serde_json::Value)) -> String {
    let layer_types: Vec<&str> = (0..48)
        .map(|i| {
            if i % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    let mut v: serde_json::Value = serde_json::from_str(
        r#"{
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "image_token_id": 248056,
        "model_type": "qwen4_exp",
        "tie_word_embeddings": false,
        "text_config": {
            "attention_bias": false, "bos_token_id": 248044, "dtype": "bfloat16",
            "eos_token_id": 248044, "full_attention_interval": 4, "hc_count": 4,
            "hc_lowrank": 320, "head_dim": 256, "heads_per_ngram": 8, "hidden_act": "silu",
            "hidden_size": 2560, "indexer_budget": 2048, "indexer_compress_ratio": 4,
            "indexer_head_dim": 128, "indexer_kv_heads": 1, "indexer_n_heads": 4,
            "layer_types": [], "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128, "linear_num_key_heads": 16,
            "linear_num_value_heads": 48, "linear_value_head_dim": 128,
            "make_ngram_vocab_size_divisible_by": 128, "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144, "model_type": "qwen4_exp_text",
            "moe_intermediate_size": 640,
            "mtp": {"hybrid": true, "layer_types": ["full_attention"],
                    "mtp_use_hidden_state_from_layer": null, "num_hidden_layers": 1,
                    "rope_theta": 10000000},
            "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
            "ngram_size": 3, "ngram_vocab_size_base": 20000000, "num_attention_heads": 24,
            "num_experts": 512, "num_experts_per_tok": 10, "num_hidden_layers": 48,
            "num_key_value_heads": 2, "output_gate_type": "sigmoid", "pad_token_id": null,
            "partial_rotary_factor": 0.25, "ple_conv_kernel_size": 4, "ple_embed_dim": 2560,
            "ple_layer_ids": [2], "rms_norm_eps": 1e-06,
            "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
                                "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                                "rope_type": "default"},
            "shared_expert_intermediate_size": 640, "split_ngram_parts": 128,
            "tie_word_embeddings": false, "vocab_size": 248320,
            "ple_embedding_dtype": "nvfp4"
        }
    }"#,
    )
    .expect("fixture is valid JSON");
    v["text_config"]["layer_types"] = layer_types.into();
    edit(&mut v);
    v.to_string()
}

#[test]
fn parses_the_flash_next_text_config() {
    let cfg = parse_config(&fixture(|_| {})).expect("qwen4_exp config parses");
    assert_eq!(cfg.model_type, "qwen4_exp");
    assert_eq!(cfg.hidden_size, 2560);
    assert_eq!(cfg.num_hidden_layers, 48);
    assert_eq!(cfg.num_ssm_layers(), 36);
    assert_eq!(cfg.num_attention_layers(), 12);
    assert_eq!(cfg.layer_type(2), LayerType::LinearAttention);
    assert_eq!(cfg.layer_type(3), LayerType::FullAttention);
    assert_eq!(
        (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim
        ),
        (24, 2, 256)
    );
    assert_eq!(
        (cfg.linear_num_key_heads, cfg.linear_num_value_heads),
        (16, 48)
    );
    assert_eq!((cfg.num_experts, cfg.num_experts_per_tok), (512, 10));
    assert_eq!(cfg.moe_intermediate_size, 640);
    assert_eq!(cfg.shared_expert_intermediate_size, 640);
    assert_eq!(cfg.vocab_size, 248320);
    assert_eq!(cfg.eos_token_id, 248044);
    assert_eq!(cfg.max_position_embeddings, 262144);
    // rope values come from the nested rope_parameters
    assert_eq!(cfg.rope_theta, 1e7);
    assert_eq!(cfg.rotary_dim(), 64);
    assert!(cfg.mrope_interleaved);
    assert_eq!(cfg.mrope_section, [11, 11, 10]);
    assert!(cfg.attn_gated);
    assert!(
        cfg.norm_topk_prob,
        "reference default when the key is absent"
    );
    assert_eq!(cfg.hc_mult, 4, "stream buffers are sized by hc_mult");
    assert!(
        cfg.skip_lm_head_quantization(),
        "the LM head stays in its checkpoint format"
    );

    let q = cfg.qwen4_exp.clone().expect("family parameters present");
    assert_eq!(q.hc_lowrank, 320);
    assert!(q.gdn_output_gate_sigmoid);
    assert_eq!(q.ple_layers, vec![1], "ple_layer_ids is one-based");
    assert_eq!((q.ngram_heads(), q.ngram_head_dim()), (16, 160));
    assert_eq!((q.ple_conv_dilation(), q.ple_conv_state_len()), (3, 9));
    assert_eq!(q.indexer_top_blocks(), 512);
    assert_eq!(q.index_block_bytes(16), (16 + 4) * 128 * 2);
    // GDN history [10240, 4] plus the n-gram layer's [4 * 2560, 9], FP32.
    assert_eq!(cfg.gdn_conv_state_bytes(), 10240 * 4 * 4);
    assert_eq!(q.ple_conv_state_bytes(4 * 2560), 10240 * 9 * 4);
    assert_eq!(cfg.ssm_conv_state_bytes(), 10240 * 4 * 4 + 10240 * 9 * 4);
    assert_eq!(cfg.ssm_h_state_bytes(), 48 * 128 * 128 * 4);
    assert_eq!((q.indexer_n_heads, q.indexer_head_dim), (4, 128));
    assert_eq!((q.split_ngram_parts, q.ngram_vocab_divisor), (128, 128));
    assert_eq!(q.ngram_vocab_size_base, 20_000_000);
}

#[test]
fn other_families_have_no_qwen4_parameters() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    assert!(cfg.qwen4_exp.is_none());
    assert_eq!(cfg.ssm_conv_state_bytes(), cfg.gdn_conv_state_bytes());
}

#[test]
fn explicit_norm_topk_prob_wins() {
    let cfg = parse_config(&fixture(|v| {
        v["text_config"]["norm_topk_prob"] = false.into()
    }))
    .expect("parses");
    assert!(!cfg.norm_topk_prob);
}

#[test]
fn missing_family_parameters_fail_fast() {
    for key in [
        "hc_count",
        "hc_lowrank",
        "indexer_budget",
        "indexer_compress_ratio",
        "indexer_head_dim",
        "indexer_n_heads",
        "indexer_kv_heads",
        "ngram_size",
        "heads_per_ngram",
        "ngram_vocab_size_base",
        "split_ngram_parts",
        "make_ngram_vocab_size_divisible_by",
        "ple_layer_ids",
        "ple_embed_dim",
        "ple_conv_kernel_size",
        "output_gate_type",
    ] {
        let json = fixture(|v| {
            v["text_config"]
                .as_object_mut()
                .expect("object")
                .remove(key);
        });
        assert!(
            parse_config(&json).is_err(),
            "config without `{key}` must be rejected"
        );
    }
}

#[test]
fn inconsistent_geometry_is_rejected() {
    let bad = [
        ("ple_layer_ids", serde_json::json!([0])),
        ("ple_layer_ids", serde_json::json!([49])),
        ("ple_embed_dim", serde_json::json!(2500)),
        ("indexer_budget", serde_json::json!(2047)),
        ("indexer_kv_heads", serde_json::json!(2)),
        ("output_gate_type", serde_json::json!("tanh")),
        ("ngram_size", serde_json::json!(1)),
    ];
    for (key, value) in bad {
        let json = fixture(|v| v["text_config"][key] = value.clone());
        assert!(
            parse_config(&json).is_err(),
            "`{key}` = {value} must be rejected"
        );
    }
}

#[test]
fn text_config_is_required() {
    let json = fixture(|v| {
        v.as_object_mut().expect("object").remove("text_config");
    });
    assert!(parse_config(&json).is_err());
}

/// Parses the real `config.json`, including its vision and quantization
/// sections. Run with `ATLAS_QWEN4_SNAPSHOT=<snapshot dir> cargo test -p
/// atlas-core -- --ignored downloaded`.
#[test]
#[ignore = "requires a downloaded checkpoint (ATLAS_QWEN4_SNAPSHOT)"]
fn parses_the_downloaded_checkpoint_config() {
    let dir = std::env::var("ATLAS_QWEN4_SNAPSHOT").expect("ATLAS_QWEN4_SNAPSHOT is set");
    let json = std::fs::read_to_string(format!("{dir}/config.json")).expect("config.json reads");
    let cfg = parse_config(&json).expect("checkpoint config parses");
    assert_eq!(cfg.model_type, "qwen4_exp");
    assert_eq!(cfg.layer_types.len(), cfg.num_hidden_layers);
    assert!(cfg.qwen4_exp.is_some());
    let quant = cfg
        .quantization_config
        .expect("quantization_config is parsed");
    println!("quantization_config: {quant:?}");
}

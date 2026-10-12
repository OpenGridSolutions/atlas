// SPDX-License-Identifier: AGPL-3.0-only

//! A checkpoint that is served through the `(model_type, hidden_size)`
//! fallback (no exact `hf_id` target) must inherit the `[estimate]`
//! block of the target that actually serves it. The admission path reads
//! the profile from the model registry, which resolves it this way; an
//! hf_id-only lookup cannot see it. Needs the compiled target set
//! (on-box build; the `ATLAS_SKIP_BUILD` stub has no targets).

#[test]
fn sibling_checkpoint_inherits_target_estimate() {
    if atlas_kernels::available_targets().is_empty() {
        eprintln!("skipped: no compiled kernel targets (ATLAS_SKIP_BUILD stub)");
        return;
    }
    let sibling = "nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4";
    let set = atlas_kernels::ptx_for_model_id(sibling, "qwen3_next", 2048)
        .expect("qwen3-next-80b-a3b target is compiled for the wildcard build");
    assert_eq!(set.target.model, "qwen3-next-80b-a3b");
    let profile = set
        .estimate
        .expect("qwen3-next-80b-a3b/MODEL.toml carries an [estimate] block");
    assert!(profile.prefill_tok_s > 0.0);
    assert!(profile.decode_step_ms.iter().all(|&ms| ms > 0.0));
    assert!(
        atlas_kernels::estimate_for_hf_id(sibling).is_none(),
        "hf_id-only lookup must not resolve a sibling checkpoint — the registry path does"
    );
}

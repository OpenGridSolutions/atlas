// SPDX-License-Identifier: AGPL-3.0-only

//! Which tensors of a `qwen4_exp` checkpoint are loaded into memory.

use crate::weight_loader::TensorFilter;

/// Prefix of the text decoder's tensors.
pub const LM: &str = "model.language_model";

/// True for the row tensors of the n-gram embedding table
/// (`...ple.ple_embedding.ngram_embedding.shard_<n>.weight[_scale]`). They
/// stay on disk; a token reads its rows through `spark_runtime::row_table`.
pub fn is_ngram_table_row_tensor(name: &str) -> bool {
    name.contains(".ple_embedding.ngram_embedding.shard_")
}

/// Tensors the weight loaders must not put on the device:
///
/// * the n-gram table rows (read from disk on demand);
/// * `*.input_scale` — activation scales for 4-bit activations. Atlas runs
///   the experts with BF16 activations, as the model card's recipes do
///   (`FORCE_A16`), so they are never read;
/// * `mtp.*` and `model.visual.*` — the draft head and the vision tower are
///   not implemented for this family yet.
pub fn skip_tensor(name: &str) -> bool {
    is_ngram_table_row_tensor(name)
        || name.ends_with(".input_scale")
        || name.starts_with("mtp.")
        || name.starts_with("model.visual.")
}

/// The family's filter for the weight loaders.
pub const FILTER: TensorFilter = TensorFilter {
    skip: skip_tensor,
    on_disk: is_ngram_table_row_tensor,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_rows_stay_on_disk_but_their_metadata_is_loaded() {
        let p = "model.language_model.layers.1.ple.ple_embedding";
        assert!(skip_tensor(&format!("{p}.ngram_embedding.shard_0.weight")));
        assert!((FILTER.on_disk)(&format!(
            "{p}.ngram_embedding.shard_3.weight_scale"
        )));
        assert!((FILTER.skip)("mtp.fc_hidden.weight") && !(FILTER.on_disk)("mtp.fc_hidden.weight"));
        assert!(!skip_tensor(&format!("{p}.ngram_embedding.weight_scale_2")));
        assert!(!skip_tensor(&format!("{p}.layer_multipliers")));
        assert!(!skip_tensor(&format!("{p}.ngram_heads_offsets")));
        assert!(!skip_tensor(
            "model.language_model.layers.1.ple.key_proj.weight"
        ));
    }

    #[test]
    fn unused_and_unimplemented_parts_are_skipped() {
        let e = "model.language_model.layers.0.mlp.experts.3.gate_proj";
        assert!(skip_tensor(&format!("{e}.input_scale")));
        assert!(!skip_tensor(&format!("{e}.weight")));
        assert!(!skip_tensor(&format!("{e}.weight_scale_2")));
        assert!(!skip_tensor(&format!("{e}.weight_scale.nvfp4_csf_fixed")));
        assert!(!skip_tensor(&format!(
            "{e}.weight_scale.nvfp4_csf_exceptions"
        )));
        assert!(skip_tensor("mtp.layers.0.mlp.experts.0.gate_proj.weight"));
        assert!(skip_tensor("model.visual.blocks.0.attn.qkv.weight"));
        assert!(!skip_tensor("lm_head.weight"));
        assert!(!skip_tensor("model.language_model.embed_tokens.weight"));
        assert!(!skip_tensor(
            "model.language_model.hyper_connection_mixer.hc_norm.weight"
        ));
    }
}

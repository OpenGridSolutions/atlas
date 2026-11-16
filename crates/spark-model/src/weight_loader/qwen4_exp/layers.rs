// SPDX-License-Identifier: AGPL-3.0-only

//! Assembles the decoder layers of a `qwen4_exp` checkpoint.

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use atlas_core::config::{LayerType, ModelConfig, Qwen4ExpConfig};
use parking_lot::Mutex;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::row_table::{DirectRows, RowPlane};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::inventory::LM;
use super::ngram::NgramHasher;
use super::tensors::{host_bytes, mixer, mxfp8, tensor};
use crate::layer::TransformerLayer;
use crate::layers::qwen4_exp::weights::{
    AttnWeights, ExpertProj, GdnWeights, MoeWeights, PleWeights, Sublayer,
};
use crate::layers::qwen4_exp::{Ple, Qwen4ExpLayer, Qwen4ExpShared};

/// Device tables of one routed-expert projection of one layer, with the
/// compressed scales validated.
#[allow(clippy::too_many_arguments)]
fn expert_proj(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    shared: &Qwen4ExpShared,
    mlp: &str,
    proj: &str,
    num_experts: usize,
    rows: usize,
    in_dim: usize,
) -> Result<ExpertProj> {
    let cols = in_dim / 16; // one scale per 16 weights
    let fixed_shape = super::csf::fixed_shape(rows, cols);
    let (mut packed, mut fixed, mut exc) = (Vec::new(), Vec::new(), Vec::new());
    let (mut counts, mut scale2_src) = (Vec::new(), Vec::new());
    for e in 0..num_experts {
        let p = format!("{mlp}.experts.{e}.{proj}");
        packed.push(
            tensor(
                store,
                &format!("{p}.weight"),
                WeightDtype::UInt8,
                &[rows, in_dim / 2],
            )?
            .0,
        );
        let f = format!("{p}.weight_scale.nvfp4_csf_fixed");
        fixed.push(tensor(store, &f, WeightDtype::UInt8, &fixed_shape)?.0);
        let x = store.get(&format!("{p}.weight_scale.nvfp4_csf_exceptions"))?;
        if x.dtype != WeightDtype::UInt32 || x.shape.len() != 1 {
            bail!(
                "{p}: csf exceptions must be a U32 vector, found {:?} {:?}",
                x.dtype,
                x.shape
            );
        }
        exc.push(x.ptr.0);
        counts.push(x.shape[0] as u32);
        scale2_src.push(tensor(
            store,
            &format!("{p}.weight_scale_2"),
            WeightDtype::FP32,
            &[],
        )?);
    }
    let bytes = |v: &[u64]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let upload = |b: &[u8]| -> Result<_> {
        let p = gpu.alloc(b.len())?;
        gpu.copy_h2d(b, p)?;
        Ok(p)
    };
    let scale2 = gpu.alloc(num_experts * 4)?;
    let stream = gpu.default_stream();
    for (e, src) in scale2_src.iter().enumerate() {
        gpu.copy_d2d_async(*src, scale2.offset(e * 4), 4, stream)?;
    }
    gpu.synchronize(stream)?;
    let out = ExpertProj {
        packed_ptrs: upload(&bytes(&packed))?,
        scale2,
        fixed_ptrs: upload(&bytes(&fixed))?,
        exc_ptrs: upload(&bytes(&exc))?,
        exc_counts: upload(
            &counts
                .iter()
                .flat_map(|c| c.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?,
        max_exc: counts.iter().copied().max().unwrap_or(0),
    };
    let errors = shared.csf_errors(gpu, &out, num_experts, rows, cols)?;
    if errors != 0 {
        bail!("{mlp}.experts.*.{proj}: {errors} malformed entries in the compressed block scales");
    }
    Ok(out)
}

fn moe(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    shared: &Qwen4ExpShared,
    config: &ModelConfig,
    mlp: &str,
) -> Result<MoeWeights> {
    let (h, e) = (config.hidden_size, config.num_experts);
    let (inter, sh) = (
        config.moe_intermediate_size,
        config.shared_expert_intermediate_size,
    );
    Ok(MoeWeights {
        router: tensor(
            store,
            &format!("{mlp}.gate.weight"),
            WeightDtype::BF16,
            &[e, h],
        )?,
        shared_gate: tensor(
            store,
            &format!("{mlp}.shared_expert_gate.weight"),
            WeightDtype::BF16,
            &[1, h],
        )?,
        sh_gate: mxfp8(store, &format!("{mlp}.shared_expert.gate_proj"), sh, h)?,
        sh_up: mxfp8(store, &format!("{mlp}.shared_expert.up_proj"), sh, h)?,
        sh_down: mxfp8(store, &format!("{mlp}.shared_expert.down_proj"), h, sh)?,
        gate: expert_proj(store, gpu, shared, mlp, "gate_proj", e, inter, h)?,
        up: expert_proj(store, gpu, shared, mlp, "up_proj", e, inter, h)?,
        down: expert_proj(store, gpu, shared, mlp, "down_proj", e, h, inter)?,
    })
}

fn gdn(store: &WeightStore, config: &ModelConfig, p: &str) -> Result<GdnWeights> {
    let h = config.hidden_size;
    let nv = config.linear_num_value_heads;
    let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let value_dim = nv * config.linear_value_head_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let d_conv = config.linear_conv_kernel_dim;
    let bf16 = WeightDtype::BF16;
    Ok(GdnWeights {
        qkv: mxfp8(store, &format!("{p}.in_proj_qkv"), conv_dim, h)?,
        z: mxfp8(store, &format!("{p}.in_proj_z"), value_dim, h)?,
        a: mxfp8(store, &format!("{p}.in_proj_a"), nv, h)?,
        b: mxfp8(store, &format!("{p}.in_proj_b"), nv, h)?,
        out: mxfp8(store, &format!("{p}.out_proj"), h, value_dim)?,
        conv: tensor(
            store,
            &format!("{p}.conv1d.weight"),
            bf16,
            &[conv_dim, 1, d_conv],
        )?,
        a_log: tensor(store, &format!("{p}.A_log"), bf16, &[nv])?,
        dt_bias: tensor(store, &format!("{p}.dt_bias"), bf16, &[nv])?,
        norm: tensor(
            store,
            &format!("{p}.norm.weight"),
            bf16,
            &[config.linear_value_head_dim],
        )?,
    })
}

fn attention(
    store: &WeightStore,
    config: &ModelConfig,
    p: &str,
    attn_idx: usize,
    q: &Qwen4ExpConfig,
) -> Result<AttnWeights> {
    let (h, hd) = (config.hidden_size, config.head_dim);
    let (ih, id) = (q.indexer_n_heads + q.indexer_kv_heads, q.indexer_head_dim);
    let (q_dim, kv_dim) = (
        config.num_attention_heads * hd,
        config.num_key_value_heads * hd,
    );
    Ok(AttnWeights {
        q: mxfp8(store, &format!("{p}.q_proj"), 2 * q_dim, h)?,
        k: mxfp8(store, &format!("{p}.k_proj"), kv_dim, h)?,
        v: mxfp8(store, &format!("{p}.v_proj"), kv_dim, h)?,
        o: mxfp8(store, &format!("{p}.o_proj"), h, q_dim)?,
        q_norm: tensor(
            store,
            &format!("{p}.q_norm.weight"),
            WeightDtype::BF16,
            &[hd],
        )?,
        k_norm: tensor(
            store,
            &format!("{p}.k_norm.weight"),
            WeightDtype::BF16,
            &[hd],
        )?,
        index_qk: mxfp8(store, &format!("{p}.indexer.index_qk_proj"), ih * id, h)?,
        index_q_norm: tensor(
            store,
            &format!("{p}.indexer.q_layernorm.weight"),
            WeightDtype::BF16,
            &[id],
        )?,
        index_k_norm: tensor(
            store,
            &format!("{p}.indexer.k_layernorm.weight"),
            WeightDtype::BF16,
            &[id],
        )?,
        attn_idx,
    })
}

fn i64s(b: &[u8]) -> Vec<i64> {
    b.chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

/// N-gram layer: weights on the device, table rows on disk.
fn ple(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    q: &Qwen4ExpConfig,
    p: &str,
) -> Result<Ple> {
    let (h, wide) = (config.hidden_size, config.hc_mult * config.hidden_size);
    let (heads, head_dim, emb) = (q.ngram_heads(), q.ngram_head_dim(), q.ple_embed_dim);
    let taps = q.ple_conv_kernel_size;
    let e = format!("{p}.ple_embedding");
    let i64v = |name: &str, n: usize| -> Result<Vec<i64>> {
        Ok(i64s(&host_bytes(
            store,
            gpu,
            &format!("{e}.{name}"),
            WeightDtype::Int64,
            &[n],
        )?))
    };
    let scale2 = host_bytes(
        store,
        gpu,
        &format!("{e}.ngram_embedding.weight_scale_2"),
        WeightDtype::FP32,
        &[1],
    )?;

    // The table: `split_ngram_parts` shards of packed rows and row scales.
    let mut planes = Vec::with_capacity(2 * q.split_ngram_parts);
    let mut shard_rows = None;
    for shard in 0..q.split_ngram_parts {
        let base = format!("{e}.ngram_embedding.shard_{shard}");
        let w = store
            .on_disk(&format!("{base}.weight"))
            .context("n-gram table rows")?;
        let s = store
            .on_disk(&format!("{base}.weight_scale"))
            .context("n-gram table scales")?;
        let rows = *shard_rows.get_or_insert(w.shape[0]);
        if w.dtype != WeightDtype::UInt8
            || w.shape != [rows, head_dim / 2]
            || s.dtype != WeightDtype::FP8E4M3
            || s.shape != [rows, head_dim / 16]
        {
            bail!("{base}: unexpected table shard {:?}/{:?}", w.shape, s.shape);
        }
        for (t, row_bytes) in [(w, head_dim / 2), (s, head_dim / 16)] {
            planes.push(RowPlane {
                path: t.path.clone(),
                data_offset: t.data_offset,
                row_bytes,
                rows: rows as u64,
            });
        }
    }
    let shard_rows = shard_rows.context("n-gram table has no shards")?;
    let hasher = NgramHasher::new(
        i64v("layer_multipliers", q.ngram_size)?,
        i64v("ngram_heads_vocab_sizes", heads)?,
        i64v("ngram_heads_offsets", heads)?,
        q.heads_per_ngram,
        config.eos_token_id,
        shard_rows,
        q.split_ngram_parts,
    )?;
    let rows = DirectRows::open(planes)?;
    tracing::info!(
        "n-gram table: {} shards x {shard_rows} rows on disk, read engine {:?}",
        q.split_ngram_parts,
        rows.engine()
    );

    let f32w = |name: &str| {
        tensor(
            store,
            &format!("{p}.{name}.weight"),
            WeightDtype::FP32,
            &[wide],
        )
    };
    Ok(Ple {
        weights: PleWeights {
            key_proj: tensor(
                store,
                &format!("{p}.key_proj.weight"),
                WeightDtype::BF16,
                &[wide, emb],
            )?,
            value_proj: tensor(
                store,
                &format!("{p}.value_proj.weight"),
                WeightDtype::BF16,
                &[h, emb],
            )?,
            conv: tensor(
                store,
                &format!("{p}.conv1d.weight"),
                WeightDtype::BF16,
                &[wide, 1, taps],
            )?,
            w_key: f32w("norm_key")?,
            w_query: f32w("norm_query")?,
            w_conv: f32w("norm_conv")?,
            scale2: f32::from_le_bytes([scale2[0], scale2[1], scale2[2], scale2[3]]),
        },
        hasher,
        rows: Mutex::new(Box::new(rows)),
        head_dim: head_dim as u32,
        taps: taps as u32,
        dilation: q.ple_conv_dilation() as u32,
    })
}

/// All decoder layers. `layer_range` restricts the build to a sub-range
/// (kernel tests load a few layers); the model build passes all layers.
pub fn load_layers(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    shared: &Arc<Qwen4ExpShared>,
    layer_range: std::ops::Range<usize>,
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    let Some(q) = config.qwen4_exp.as_ref() else {
        bail!("qwen4_exp loader needs the family parameters of the config");
    };
    let (h, hc, rank) = (config.hidden_size, config.hc_mult, q.hc_lowrank);
    let last = config.num_hidden_layers - 1;
    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(layer_range.len());
    for i in layer_range {
        let lp = format!("{LM}.layers.{i}");
        // Index among the attention layers = position in the KV cache.
        let attn_idx = (0..i)
            .filter(|l| config.layer_type(*l) == LayerType::FullAttention)
            .count();
        let sublayer = match config.layer_type(i) {
            LayerType::LinearAttention => {
                Sublayer::Gdn(gdn(store, config, &format!("{lp}.linear_attn"))?)
            }
            LayerType::FullAttention => Sublayer::Attn(attention(
                store,
                config,
                &format!("{lp}.self_attn"),
                attn_idx,
                q,
            )?),
            other => bail!("layer {i}: qwen4_exp has no {other:?} layers"),
        };
        layers.push(Box::new(Qwen4ExpLayer {
            shared: shared.clone(),
            layer_idx: i,
            is_first: i == 0,
            ple: if q.ple_layers.contains(&i) {
                // The n-gram layer keeps its convolution history in the
                // layer's recurrent-state slot, which only GDN layers have.
                if !matches!(sublayer, Sublayer::Gdn(_)) {
                    bail!("layer {i}: an n-gram injection on an attention layer is not supported");
                }
                Some(ple(store, gpu, config, q, &format!("{lp}.ple"))?)
            } else {
                None
            },
            attn_mixer: mixer(
                store,
                &format!("{lp}.attn_hyper_connection"),
                h,
                hc,
                rank,
                true,
            )?,
            sublayer,
            mlp_mixer: mixer(
                store,
                &format!("{lp}.mlp_hyper_connection"),
                h,
                hc,
                rank,
                true,
            )?,
            moe: moe(store, gpu, shared, config, &format!("{lp}.mlp"))
                .with_context(|| format!("layer {i} MoE"))?,
            final_mixer: if i == last {
                Some(mixer(
                    store,
                    &format!("{LM}.hyper_connection_mixer"),
                    h,
                    hc,
                    rank,
                    false,
                )?)
            } else {
                None
            },
        }));
    }
    Ok(layers)
}

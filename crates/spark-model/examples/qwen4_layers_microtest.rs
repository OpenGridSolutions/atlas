// SPDX-License-Identifier: AGPL-3.0-only
//! End-to-end check of the Qwen3.8-Flash-Next decoder layers on real weights:
//! the production loader and `Qwen4ExpLayer::decode` (GPU) against the
//! independent CPU reference in `qwen4_support/reference.rs`.
//!
//! Loads the first `layers` decoder layers of the checkpoint (default 4:
//! GDN, GDN with the n-gram layer, GDN, attention), decodes a short token
//! sequence and compares the residual streams after every layer and token.
//!
//! The chunk prefill (`Qwen4ExpLayer::prefill`) rounds intermediate values
//! to BF16 where decode keeps FP32, and that can swap the weakest routed
//! expert of a token. It is therefore checked in three ways:
//!   * one layer at a time on identical inputs: a one-token prefill, and
//!     decode steps continuing from prefilled state, against decode;
//!   * chunking: one chunk, two chunks and token-by-token give the same
//!     streams exactly;
//!   * end to end against the CPU reference, with a loose bound.
//! A last pass shrinks the attention budget so the short sequence outgrows
//! full attention, and repeats the comparison with decode there.
//!
//! Usage:
//!   cargo run --release -p spark-model --features cuda,gpu-examples \
//!       --example qwen4_layers_microtest -- <snapshot_dir> [layers] [tokens]
use anyhow::{Context, Result, bail};
use atlas_core::config::LayerType;
use spark_model::layers::qwen4_exp::Qwen4ExpShared;
use spark_model::weight_loader::qwen4_exp::{inventory, layers::load_layers};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use spark_runtime::weights::WeightLoader;

#[path = "qwen4_support/layer_seq.rs"]
mod layer_seq;
#[path = "qwen4_support/mod.rs"]
mod support;
use layer_seq::{Quiet, Seq, new_states};
use support::reference::{self, H, HC};

const EOS: u32 = 248044;
/// Tokens decoded; includes an EOS so the n-gram boundary rule is exercised.
const TOKENS: [u32; 12] = [
    9707, 1917, 11, 419, 374, EOS, 264, 1273, 13, 2585, 1558, 419,
];
const BLOCK: usize = 16;

/// Loader filter: everything outside the first decoder layers is skipped.
/// The layer count is fixed at 4 here; `layers` on the command line may be
/// smaller.
fn skip(name: &str) -> bool {
    let keep = (0..4).any(|i| name.starts_with(&format!("{}.layers.{i}.", inventory::LM)));
    !keep || inventory::skip_tensor(name)
}

fn check(name: &str, got: &[f32], want: &[f64]) -> (bool, f64, f64) {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (g, w)| m.max((*g as f64 - w).abs()));
    let (mut dot, mut ng, mut nw) = (0.0f64, 0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        dot += *g as f64 * w;
        ng += (*g as f64).powi(2);
        nw += w * w;
    }
    let cos = dot / (ng.sqrt() * nw.sqrt());
    let nan = got.iter().any(|g| !g.is_finite());
    let ok = !nan && cos > 0.9999 && err / scale < 2e-2;
    println!(
        "  {name}: max|ref|={scale:.3e} max_err={err:.2e} rel={:.2e} cosine={cos:.7} {}",
        err / scale,
        if ok { "ok" } else { "FAIL" }
    );
    (ok, err / scale, cos)
}

/// Cosine bound for one layer on identical inputs (BF16 rounding, at most
/// a swapped low-weight expert), and across layers (several of them).
const NEAR: f64 = 0.999;
const LOOSE: f64 = 0.98;

/// Worst relative error and cosine seen, and whether every check passed.
struct Tally(bool, f64, f64);
impl Tally {
    fn add(&mut self, name: &str, got: &[f32], want: &[f64]) {
        let (ok, rel, cos) = check(name, got, want);
        self.0 &= ok;
        self.1 = self.1.max(rel);
        self.2 = self.2.min(cos);
    }
}

/// Decode against prefill for one layer at a time: the layers below decode,
/// layer `l` prefills tokens before `split` one by one and decodes the rest.
/// Its streams must stay near `dec[pos][l]`, the all-decode run.
fn one_layer_at_a_time(
    base: &Seq<'_>,
    kv: &mut PagedKvCache,
    dec: &[Vec<Vec<f32>>],
    split: usize,
) -> Result<bool> {
    let (gpu, config) = (base.gpu, base.config);
    let mut q = Quiet::new();
    for l in 0..base.layers.len() {
        let mut lo = base.with(&base.layers[..l], new_states(gpu, config, l)?);
        let own = new_states(gpu, config, l + 1)?.split_off(l);
        let mut hi = base.with(&base.layers[l..l + 1], own);
        for (pos, want) in dec.iter().enumerate() {
            lo.decode(kv, pos, |_, _| {})?;
            let name = format!("layer {l} token {pos}");
            if pos >= split {
                hi.decode(kv, pos, |_, got| q.add(&name, &got, &want[l], NEAR))?;
                continue;
            }
            if l > 0 {
                // Hand the decoded streams to the prefill path.
                let bytes = support::download(gpu, base.buffers.hc_streams(), HC * H * 4)?;
                gpu.copy_h2d(&bytes, base.shared.prefill_streams())?;
            }
            hi.prefill(kv, pos, 1, |_, _, got| q.add(&name, got, &want[l], NEAR))?;
        }
    }
    Ok(q.done("one layer at a time, prefill and prefill-then-decode against decode"))
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let dir = a
        .get(1)
        .context("usage: qwen4_layers_microtest <snapshot_dir> [layers] [tokens]")?;
    let n_layers: usize = a.get(2).map_or(4, |s| s.parse().unwrap());
    let n: usize = a.get(3).map_or(TOKENS.len(), |s| s.parse().unwrap());
    if n_layers == 0 || n_layers > 4 || n == 0 || n > TOKENS.len() {
        bail!("layers must be 1..=4 and tokens 1..={}", TOKENS.len());
    }
    let config =
        atlas_core::config::parse_config(&std::fs::read_to_string(format!("{dir}/config.json"))?)?;
    let q_cfg = config
        .qwen4_exp
        .clone()
        .context("not a qwen4_exp checkpoint")?;
    println!("=== qwen4 layers microtest: layers 0..{n_layers}, {n} tokens ===");

    let be = support::backend()?;
    let gpu: &dyn GpuBackend = &be;

    // ── the server's default loader, restricted to the first layers ──
    let mut loader = spark_runtime::fast_weights::FastSafetensorsLoader::new();
    loader.skip_tensor = Some(skip);
    loader.peak_memory_multiplier = Some(1.02);
    let mut store = loader.load(std::path::Path::new(dir), gpu, 1 << 30)?;
    store.register_on_disk(spark_runtime::fast_weights::locate_tensors(
        std::path::Path::new(dir),
        inventory::is_ngram_table_row_tensor,
    )?)?;
    println!(
        "  loaded {} tensors, {:.2} GiB",
        store.len(),
        store.total_bytes() as f64 / (1u64 << 30) as f64
    );
    let shared = Qwen4ExpShared::new(gpu, &config)?;
    let layers = load_layers(&store, &config, gpu, &shared, 0..n_layers)?;

    // ── runtime objects ──
    let buffers = BufferArena::new(&config, TOKENS.len(), 64, BLOCK, gpu)?;
    let mut kv = PagedKvCache::new(
        KvCacheConfig {
            block_size: BLOCK,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            num_layers: config.num_attention_layers(),
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: Vec::new(),
            layer_dims: Vec::new(),
            cache_blocks_per_seq: None,
        },
        16,
        gpu,
    )?;
    // Index plane of the sparse attention, next to the KV blocks.
    kv.alloc_aux(q_cfg.index_block_bytes(BLOCK), gpu)?;
    let meta = gpu.alloc(8192)?;
    let new_seq = |first_block: u32| -> Result<Seq<'_>> {
        Ok(Seq {
            gpu,
            config: &config,
            buffers: &buffers,
            shared: &shared,
            layers: &layers,
            dir,
            meta,
            states: new_states(gpu, &config, n_layers)?,
            block_table: vec![first_block],
        })
    };

    // ── decode against the CPU reference; its streams are kept ──
    let mut refs = Vec::new();
    for i in 0..n_layers {
        let is_attn = config.layer_type(i) == LayerType::FullAttention;
        refs.push(reference::Layer::load(
            dir,
            i,
            is_attn,
            q_cfg.ple_layers.contains(&i),
        )?);
    }
    let mut tally = Tally(true, 0.0, 1.0);
    let mut want: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut dec: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut seq = new_seq(1)?;
    for pos in 0..n {
        let token = TOKENS[pos];
        let emb = reference::embedding(dir, token)?;
        let mut streams: Vec<f64> = (0..HC * H).map(|e| emb[e % H]).collect();
        let prev = |d: usize| {
            if pos >= d {
                TOKENS[pos - d] as i64
            } else {
                EOS as i64
            }
        };
        let mut per_layer = Vec::new();
        for r in refs.iter_mut() {
            r.forward(&mut streams, pos, (token as i64, prev(1), prev(2)))?;
            per_layer.push(streams.clone());
        }
        println!("decode, token {pos} (id {token})");
        let mut got_layers = Vec::new();
        seq.decode(&mut kv, pos, |i, got| {
            tally.add(&format!("layer {i} streams"), &got, &per_layer[i]);
            got_layers.push(got);
        })?;
        want.push(per_layer);
        dec.push(got_layers);
    }
    println!(
        "  decode against the reference: worst rel={:.2e} cosine={:.7}",
        tally.1, tally.2
    );
    let mut ok = tally.0;

    // ── prefill ──
    println!("prefill");
    let split = (n / 2).clamp(1, 5);
    ok &= one_layer_at_a_time(&new_seq(2)?, &mut kv, &dec, split)?;
    let mut one: Vec<Vec<f32>> = vec![Vec::new(); n * n_layers];
    let mut q = Quiet::new();
    new_seq(3)?.prefill(&mut kv, 0, n, |i, pos, got| {
        let want32: Vec<f32> = want[pos][i].iter().map(|v| *v as f32).collect();
        q.add(&format!("token {pos} layer {i}"), got, &want32, LOOSE);
        one[pos * n_layers + i] = got.to_vec();
    })?;
    ok &= q.done("one chunk against the reference");
    let mut q = Quiet::new();
    let mut same = |i: usize, pos: usize, got: &[f32]| {
        q.add(
            &format!("token {pos} layer {i}"),
            got,
            &one[pos * n_layers + i],
            1.0,
        )
    };
    let mut seq = new_seq(4)?;
    for (start, len) in [(0, split), (split, n - split)] {
        if len > 0 {
            seq.prefill(&mut kv, start, len, &mut same)?;
        }
    }
    let mut seq = new_seq(5)?;
    for pos in 0..n {
        seq.prefill(&mut kv, pos, 1, &mut same)?;
    }
    ok &= q.done("two chunks and token by token against one chunk");

    // ── beyond full attention: prefill against decode ──
    let ratio = q_cfg.indexer_compress_ratio;
    if n_layers == 4 && n > 2 * ratio {
        println!("attention budget of one index block");
        let mut small = config.clone();
        small.qwen4_exp.as_mut().unwrap().indexer_budget = ratio;
        let shared = Qwen4ExpShared::new(gpu, &small)?;
        let layers = load_layers(&store, &small, gpu, &shared, 0..n_layers)?;
        let mut seq = Seq {
            gpu,
            config: &small,
            buffers: &buffers,
            shared: &shared,
            layers: &layers,
            dir,
            meta,
            states: new_states(gpu, &small, n_layers)?,
            block_table: vec![6],
        };
        let mut dec: Vec<Vec<Vec<f32>>> = Vec::new();
        for pos in 0..n {
            let mut got_layers = Vec::new();
            seq.decode(&mut kv, pos, |_, got| got_layers.push(got))?;
            dec.push(got_layers);
        }
        seq.block_table = vec![7];
        ok &= one_layer_at_a_time(&seq, &mut kv, &dec, n)?;
        let mut q = Quiet::new();
        seq.states = new_states(gpu, &small, n_layers)?;
        seq.block_table = vec![8];
        seq.prefill(&mut kv, 0, n, |i, pos, got| {
            q.add(&format!("token {pos} layer {i}"), got, &dec[pos][i], LOOSE)
        })?;
        ok &= q.done("one chunk against decode");
    }
    support::finish(ok)
}

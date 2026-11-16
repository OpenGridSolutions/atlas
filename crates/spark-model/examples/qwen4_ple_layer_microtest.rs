// SPDX-License-Identifier: AGPL-3.0-only
//! Oracle for the Qwen3.8-Flash-Next n-gram layer kernels (module
//! `ple_layer`): ple_decode_rows, ple_gate, ple_conv_add, ple_state_update
//! against an f64 CPU reference of the published formulas.
//!
//! The GPU runs the token sequence in several calls (chunk sizes 5, 1, 1, 12,
//! rest), carrying the convolution state between them; the reference computes
//! the whole sequence in one pass from a zero history. The sequence's state
//! slot starts dirty and a second slot must stay untouched.
//!
//! Usage:
//!   cargo run --release -p spark-model --features cuda,gpu-examples \
//!       --example qwen4_ple_layer_microtest -- [tokens] [seed] [snapshot_dir]
//!
//! With `snapshot_dir` the layer weights and the table rows (addressed by the
//! n-gram hash of a token sequence) come from the checkpoint.
use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

#[path = "qwen4_support/mod.rs"]
mod support;
use support::{Rng, bf16_to_f32, e2m1, e4m3, f32_to_bf16, f32s, le16, le32, sigmoid, u16s, upload};

const H: usize = 2560;
const HC: usize = 4;
const C: usize = HC * H;
const HEADS: usize = 16;
const HEAD_DIM: usize = 160;
const E: usize = HEADS * HEAD_DIM;
const TAPS: usize = 4;
const DILATION: usize = 3;
const STATE_LEN: usize = (TAPS - 1) * DILATION;
const EPS: f64 = 1e-6;
const EOS: i64 = 248044;
const PLE: &str = "model.language_model.layers.1.ple";

struct Layer {
    key_proj: Vec<u16>,   // [C, E]
    value_proj: Vec<u16>, // [H, E]
    conv: Vec<u16>,       // [C, TAPS]
    w_key: Vec<f32>,
    w_query: Vec<f32>,
    w_conv: Vec<f32>,
    scale2: f32,
    rows: Vec<u8>,   // [T, HEADS, HEAD_DIM / 2]
    scales: Vec<u8>, // [T, HEADS, HEAD_DIM / 16]
}

/// Table row ids of one token: heads 0..8 hash (tok, prev), heads 8..16 hash
/// (tok, prev, prev2); a predecessor behind an EOS reads as EOS.
fn ngram_ids(
    tok: i64,
    prev: i64,
    prev2: i64,
    mult: &[i64],
    sizes: &[i64],
    offs: &[i64],
) -> Vec<i64> {
    let prev2 = if prev == EOS { EOS } else { prev2 };
    let bi = tok.wrapping_mul(mult[0]) ^ prev.wrapping_mul(mult[1]);
    let tri = bi ^ prev2.wrapping_mul(mult[2]);
    (0..HEADS)
        .map(|h| (if h < HEADS / 2 { bi } else { tri }).rem_euclid(sizes[h]) + offs[h])
        .collect()
}

fn i64s(b: &[u8]) -> Vec<i64> {
    b.chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8])))
        .collect()
}

fn layer(dir: Option<&str>, t: usize, r: &mut Rng) -> Result<Layer> {
    let Some(dir) = dir else {
        return Ok(Layer {
            key_proj: r.bf16s(C * E, -0.02, 0.02),
            value_proj: r.bf16s(H * E, -0.02, 0.02),
            conv: r.bf16s(C * TAPS, -0.6, 0.6),
            w_key: (0..C).map(|_| r.u(-0.2, 0.2)).collect(),
            w_query: (0..C).map(|_| r.u(-0.2, 0.2)).collect(),
            w_conv: (0..C).map(|_| r.u(-0.2, 0.2)).collect(),
            scale2: 3.3e-5,
            rows: (0..t * E / 2).map(|_| (r.n() >> 56) as u8).collect(),
            // E4M3 bytes 0x40..0x78: scales 2 .. 256
            scales: (0..t * E / 16)
                .map(|_| 0x40 + (r.n() % 0x38) as u8)
                .collect(),
        });
    };
    let get = |n: &str, d: &str| support::read_tensor(dir, &format!("{PLE}.{n}"), d).map(|x| x.1);
    let emb = "ple_embedding";
    let mult = i64s(&get(&format!("{emb}.layer_multipliers"), "I64")?);
    let sizes = i64s(&get(&format!("{emb}.ngram_heads_vocab_sizes"), "I64")?);
    let offs = i64s(&get(&format!("{emb}.ngram_heads_offsets"), "I64")?);
    let shard0 = support::locate(
        dir,
        &format!("{PLE}.{emb}.ngram_embedding.shard_0.weight"),
        "U8",
    )?;
    let shard_rows = shard0.shape[0] as i64;
    if shard0.shape != [2500012, HEAD_DIM / 2] {
        bail!("unexpected table shard shape {:?}", shard0.shape);
    }
    // A token sequence with an EOS inside, so the boundary rule is exercised.
    let toks: Vec<i64> = (0..t)
        .map(|i| {
            if i == 7 {
                EOS
            } else {
                (r.n() % 248_000) as i64
            }
        })
        .collect();
    let (mut rows, mut scales) = (Vec::new(), Vec::new());
    for i in 0..t {
        let prev = if i >= 1 { toks[i - 1] } else { EOS };
        let prev2 = if i >= 2 { toks[i - 2] } else { EOS };
        for id in ngram_ids(toks[i], prev, prev2, &mult, &sizes, &offs) {
            let (shard, row) = (id / shard_rows, (id % shard_rows) as u64);
            let base = format!("{PLE}.{emb}.ngram_embedding.shard_{shard}");
            let w = support::locate(dir, &format!("{base}.weight"), "U8")?;
            let s = support::locate(dir, &format!("{base}.weight_scale"), "F8_E4M3")?;
            rows.extend(support::read_at(
                &w,
                row * (HEAD_DIM as u64 / 2),
                HEAD_DIM / 2,
            )?);
            scales.extend(support::read_at(
                &s,
                row * (HEAD_DIM as u64 / 16),
                HEAD_DIM / 16,
            )?);
        }
    }
    Ok(Layer {
        key_proj: u16s(&get("key_proj.weight", "BF16")?),
        value_proj: u16s(&get("value_proj.weight", "BF16")?),
        conv: u16s(&get("conv1d.weight", "BF16")?),
        w_key: f32s(&get("norm_key.weight", "F32")?),
        w_query: f32s(&get("norm_query.weight", "F32")?),
        w_conv: f32s(&get("norm_conv.weight", "F32")?),
        scale2: f32s(&get(
            &format!("{emb}.ngram_embedding.weight_scale_2"),
            "F32",
        )?)[0],
        rows,
        scales,
    })
}

/// Group RMS norm of one stream with weight (1 + w).
fn rms(x: &[f64], w: &[f32]) -> Vec<f64> {
    let rs = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + EPS).sqrt();
    x.iter()
        .zip(w)
        .map(|(v, w)| v * rs * (1.0 + *w as f64))
        .collect()
}

fn check(name: &str, got: &[f32], want: &[f64], gate: f64) -> bool {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (g, w)| m.max((*g as f64 - w).abs()));
    let nan = got.iter().filter(|g| !g.is_finite()).count();
    let ok = nan == 0 && scale > 0.0 && err / scale <= gate;
    println!(
        "  {name:<9} max|ref|={scale:.4e} max_err={err:.3e} rel={:.3e} nan={nan} gate={gate:.0e} {}",
        err / scale,
        if ok { "ok" } else { "FAIL" }
    );
    ok
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let t: usize = a.get(1).map_or(24, |s| s.parse().unwrap());
    let seed: u64 = a.get(2).map_or(0x91E, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x91E)
    });
    let dir = a.get(3).map(String::as_str);
    println!(
        "=== ple_layer microtest: tokens={t} seed=0x{seed:X} weights={} ===",
        dir.map_or("random", |_| "checkpoint layers.1.ple")
    );
    let mut r = Rng(seed);
    let l = layer(dir, t, &mut r)?;
    let streams_in: Vec<f32> = (0..t * C).map(|_| r.u(-0.08, 0.08)).collect();

    // ── CPU reference (f64), whole sequence from a zero history ──
    let c_emb: Vec<f64> = (0..t * E)
        .map(|j| {
            let nib = (l.rows[j / 2] >> (4 * (j & 1))) & 0xF;
            e2m1(nib) * e4m3(l.scales[j / 16]) * l.scale2 as f64
        })
        .collect();
    // The kernels consume the embedding and both projections as BF16.
    let emb_bf: Vec<u16> = c_emb.iter().map(|v| f32_to_bf16(*v as f32)).collect();
    let proj = |w: &[u16], rows: usize| -> Vec<u16> {
        let mut out = Vec::with_capacity(t * rows);
        for tok in 0..t {
            let e: Vec<f64> = emb_bf[tok * E..(tok + 1) * E]
                .iter()
                .map(|b| bf16_to_f32(*b) as f64)
                .collect();
            for row in 0..rows {
                let s: f64 = w[row * E..(row + 1) * E]
                    .iter()
                    .zip(&e)
                    .map(|(a, b)| bf16_to_f32(*a) as f64 * b)
                    .sum();
                out.push(f32_to_bf16(s as f32));
            }
        }
        out
    };
    let key = proj(&l.key_proj, C);
    let value = proj(&l.value_proj, H);
    let (mut c_gated, mut c_conv_in) = (vec![0.0f64; t * C], vec![0.0f64; t * C]);
    let mut gates = Vec::new();
    for tok in 0..t {
        let v: Vec<f64> = value[tok * H..(tok + 1) * H]
            .iter()
            .map(|b| bf16_to_f32(*b) as f64)
            .collect();
        for s in 0..HC {
            let o = tok * C + s * H;
            let k: Vec<f64> = key[o..o + H]
                .iter()
                .map(|b| bf16_to_f32(*b) as f64)
                .collect();
            let x: Vec<f64> = streams_in[o..o + H].iter().map(|v| *v as f64).collect();
            let kn = rms(&k, &l.w_key[s * H..(s + 1) * H]);
            let qn = rms(&x, &l.w_query[s * H..(s + 1) * H]);
            let d = kn.iter().zip(&qn).map(|(a, b)| a * b).sum::<f64>() / (H as f64).sqrt();
            let g = sigmoid(d.abs().max(1e-6).sqrt() * if d == 0.0 { 0.0 } else { d.signum() });
            gates.push(g);
            let gv: Vec<f64> = v.iter().map(|v| g * v).collect();
            c_conv_in[o..o + H].copy_from_slice(&rms(&gv, &l.w_conv[s * H..(s + 1) * H]));
            c_gated[o..o + H].copy_from_slice(&gv);
        }
    }
    let mut c_streams: Vec<f64> = streams_in.iter().map(|v| *v as f64).collect();
    for tok in 0..t {
        for c in 0..C {
            let mut y = 0.0f64;
            for k in 0..TAPS {
                // hist index tok + D*k over [zeros(STATE_LEN) ++ conv_in]
                let j = tok + DILATION * k;
                if j >= STATE_LEN {
                    y += bf16_to_f32(l.conv[c * TAPS + k]) as f64
                        * c_conv_in[(j - STATE_LEN) * C + c];
                }
            }
            c_streams[tok * C + c] += c_gated[tok * C + c] + y * sigmoid(y);
        }
    }

    // ── GPU ──
    let be = support::backend()?;
    let gpu: &dyn GpuBackend = &be;
    let st = gpu.create_stream()?;
    let k = |name: &str| gpu.kernel("ple_layer", name);
    let d_rows = upload(gpu, &l.rows)?;
    let d_scales = upload(gpu, &l.scales)?;
    let d_emb = gpu.alloc(t * E * 2)?;
    KernelLaunch::new(gpu, k("ple_decode_rows")?)
        .grid([t as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(d_rows)
        .arg_ptr(d_scales)
        .arg_ptr(d_emb)
        .arg_u32(HEADS as u32)
        .arg_u32(HEAD_DIM as u32)
        .arg_f32(l.scale2)
        .launch(st)?;
    gpu.synchronize(st)?;
    let g_emb = u16s(&support::download(gpu, d_emb, t * E * 2)?);
    let emb_wrong = g_emb.iter().zip(&emb_bf).filter(|(a, b)| a != b).count();
    let emb_peak = c_emb.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    println!(
        "  rows      BF16 values differing from the reference: {emb_wrong} of {} (max|ref|={emb_peak:.4e})",
        t * E
    );
    let mut ok = emb_wrong == 0 && emb_peak > 0.0;

    let d_key = upload(gpu, &le16(&key))?;
    let d_value = upload(gpu, &le16(&value))?;
    let d_streams = upload(gpu, &le32(&streams_in))?;
    let d_wk = upload(gpu, &le32(&l.w_key))?;
    let d_wq = upload(gpu, &le32(&l.w_query))?;
    let d_wc = upload(gpu, &le32(&l.w_conv))?;
    let d_conv = upload(gpu, &le16(&l.conv))?;
    let d_gated = gpu.alloc(t * C * 4)?;
    let d_conv_in = gpu.alloc(t * C * 4)?;
    // Two state slots: slot 1 is the sequence, slot 0 a bystander with a marker.
    // Slot 1 starts dirty (left over from an earlier sequence): history before
    // position 0 must read as zero without the slot being cleared.
    let stride = (C * STATE_LEN * 4) as u64;
    let mut state0 = vec![7.5f32; C * STATE_LEN];
    state0.extend(vec![-3.25f32; C * STATE_LEN]);
    let d_state = upload(gpu, &le32(&state0))?;
    let d_slot = upload(gpu, &1u32.to_le_bytes())?;
    let d_pos = upload(gpu, &0u32.to_le_bytes())?;

    let (mut done, mut chunks) = (0usize, Vec::new());
    for want in [5usize, 1, 1, 12, usize::MAX] {
        let n = want.min(t - done);
        if n == 0 {
            break;
        }
        chunks.push(n);
        let off = |p: DevicePtr, per_tok: usize| p.offset(done * per_tok);
        // The previous call's kernels read the position buffer.
        gpu.synchronize(st)?;
        gpu.copy_h2d(&(done as u32).to_le_bytes(), d_pos)?;
        KernelLaunch::new(gpu, k("ple_gate")?)
            .grid([n as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(off(d_key, C * 2))
            .arg_ptr(off(d_value, H * 2))
            .arg_ptr(off(d_streams, C * 4))
            .arg_ptr(d_wk)
            .arg_ptr(d_wq)
            .arg_ptr(d_wc)
            .arg_ptr(off(d_gated, C * 4))
            .arg_ptr(off(d_conv_in, C * 4))
            .arg_u32(H as u32)
            .arg_u32(HC as u32)
            .arg_f32(EPS as f32)
            .launch(st)?;
        KernelLaunch::new(gpu, k("ple_conv_add")?)
            .grid([(C as u32).div_ceil(256), n as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(off(d_gated, C * 4))
            .arg_ptr(off(d_conv_in, C * 4))
            .arg_ptr(d_conv)
            .arg_ptr(d_state)
            .arg_ptr(d_slot)
            .arg_ptr(d_pos)
            .arg_ptr(off(d_streams, C * 4))
            .arg_u64(stride)
            .arg_u32(C as u32)
            .arg_u32(n as u32)
            .arg_u32(TAPS as u32)
            .arg_u32(DILATION as u32)
            .launch(st)?;
        KernelLaunch::new(gpu, k("ple_state_update")?)
            .grid([(C as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(off(d_conv_in, C * 4))
            .arg_ptr(d_state)
            .arg_ptr(d_slot)
            .arg_u64(stride)
            .arg_u32(C as u32)
            .arg_u32(n as u32)
            .arg_u32(STATE_LEN as u32)
            .launch(st)?;
        done += n;
    }
    gpu.synchronize(st)?;
    println!("  GPU calls of {chunks:?} tokens");
    let dl = |p: DevicePtr, n: usize| -> Result<Vec<f32>> {
        Ok(f32s(&support::download(gpu, p, n * 4)?))
    };
    ok &= check("gated", &dl(d_gated, t * C)?, &c_gated, 1e-5);
    ok &= check("conv_in", &dl(d_conv_in, t * C)?, &c_conv_in, 1e-5);
    ok &= check("streams", &dl(d_streams, t * C)?, &c_streams, 1e-5);
    // Final state: the last STATE_LEN conv inputs per channel, oldest first.
    let state = dl(d_state, 2 * C * STATE_LEN)?;
    let want_state: Vec<f64> = (0..C * STATE_LEN)
        .map(|i| {
            let (c, j) = (i / STATE_LEN, i % STATE_LEN);
            let pos = t as isize - STATE_LEN as isize + j as isize;
            if pos < 0 {
                0.0
            } else {
                c_conv_in[pos as usize * C + c]
            }
        })
        .collect();
    ok &= check("state", &state[C * STATE_LEN..], &want_state, 1e-5);
    let bystander_ok = state[..C * STATE_LEN].iter().all(|v| *v == 7.5);
    println!("  other sequence slot untouched: {bystander_ok}");
    ok &= bystander_ok;
    let (gmin, gmax) = gates
        .iter()
        .fold((1.0f64, 0.0f64), |(a, b), g| (a.min(*g), b.max(*g)));
    println!("  gate values span {gmin:.3} .. {gmax:.3} (a constant 0.5 would hide a wrong gate)");

    for p in [
        d_rows, d_scales, d_emb, d_key, d_value, d_streams, d_wk, d_wq, d_wc, d_conv, d_gated,
        d_conv_in, d_state, d_slot, d_pos,
    ] {
        gpu.free(p).ok();
    }
    support::finish(ok)
}

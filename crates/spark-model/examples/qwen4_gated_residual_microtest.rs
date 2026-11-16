// SPDX-License-Identifier: AGPL-3.0-only
//! Oracle for the Qwen3.8-Flash-Next gated-residual kernels (module
//! `gated_residual`): gr_expand, gr_norm_inject, gr_down_silu, gr_up_mix,
//! gr_post against an f64 CPU reference of the published formulas.
//!
//! Usage:
//!   cargo run --release -p spark-model --features cuda,gpu-examples \
//!       --example qwen4_gated_residual_microtest -- [tokens] [seed] [snapshot_dir]
//!
//! With `snapshot_dir` (an HF snapshot of the checkpoint) the weights of
//! `layers.0.attn_hyper_connection` are read from disk; without it they are
//! random at a comparable scale.
use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

#[path = "qwen4_support/mod.rs"]
mod support;
use support::{Rng, bf16_to_f32, f32_to_bf16, f32s, le16, le32, sigmoid, u16s, upload};

const H: usize = 2560;
const HC: usize = 4;
const RANK: usize = 320;
const EPS: f32 = 1e-6;
const SITE: &str = "model.language_model.layers.0.attn_hyper_connection";

struct Weights {
    norm: Vec<f32>,   // [HC*H]
    inject: Vec<u16>, // [HC, HC*H]
    down: Vec<u16>,   // [RANK, HC*H]
    up: Vec<u16>,     // [HC*H, RANK]
}
/// All elements of one tensor, with the element count checked.
fn tensor(dir: &str, name: &str, dtype: &str, elems: usize) -> Result<Vec<u8>> {
    let (shape, bytes) = support::read_tensor(dir, name, dtype)?;
    if shape.iter().product::<usize>() != elems {
        bail!("{name}: expected {elems} elements, found shape {shape:?}");
    }
    Ok(bytes)
}
fn download_f32(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    Ok(f32s(&support::download(gpu, p, n * 4)?))
}
fn weights(dir: Option<&str>, r: &mut Rng) -> Result<Weights> {
    let w = HC * H;
    if let Some(dir) = dir {
        return Ok(Weights {
            norm: f32s(&tensor(dir, &format!("{SITE}.hc_norm.weight"), "F32", w)?),
            inject: u16s(&tensor(
                dir,
                &format!("{SITE}.block_inject_weight.weight"),
                "BF16",
                HC * w,
            )?),
            down: u16s(&tensor(
                dir,
                &format!("{SITE}.input_mix_weight_down.weight"),
                "BF16",
                RANK * w,
            )?),
            up: u16s(&tensor(
                dir,
                &format!("{SITE}.input_mix_weight_up.weight"),
                "BF16",
                w * RANK,
            )?),
        });
    }
    let mut bf =
        |n: usize, s: f32| -> Vec<u16> { (0..n).map(|_| f32_to_bf16(r.u(-s, s))).collect() };
    let inject = bf(HC * w, 0.02);
    let down = bf(RANK * w, 0.02);
    let up = bf(w * RANK, 0.1);
    Ok(Weights {
        norm: (0..w).map(|_| r.u(-0.2, 0.2)).collect(),
        inject,
        down,
        up,
    })
}

/// Largest absolute error relative to the largest reference magnitude.
fn check(name: &str, got: &[f32], want: &[f64], gate: f64) -> bool {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (g, w)| m.max((*g as f64 - w).abs()));
    let nan = got.iter().filter(|g| !g.is_finite()).count();
    let ok = nan == 0 && scale > 0.0 && err / scale <= gate;
    println!(
        "  {name:<8} max|ref|={scale:.4e} max_err={err:.3e} rel={:.3e} nan={nan} gate={gate:.0e} {}",
        err / scale,
        if ok { "ok" } else { "FAIL" }
    );
    ok
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let t: usize = a.get(1).map_or(3, |s| s.parse().unwrap());
    let seed: u64 = a.get(2).map_or(0x94E, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x94E)
    });
    let dir = a.get(3).map(String::as_str);
    let w = HC * H;
    println!(
        "=== gated_residual microtest: tokens={t} seed=0x{seed:X} weights={} ===",
        dir.map_or("random", |_| "checkpoint layers.0.attn_hyper_connection")
    );
    let mut r = Rng(seed);
    let wt = weights(dir, &mut r)?;
    // Token 0 starts as four copies of an embedding (gr_expand); the streams are
    // then perturbed per stream so the mix sees four different inputs.
    let hidden: Vec<u16> = (0..t * H).map(|_| f32_to_bf16(r.u(-0.05, 0.05))).collect();
    let drift: Vec<f32> = (0..t * w).map(|_| r.u(-0.03, 0.03)).collect();
    let block_out: Vec<u16> = (0..t * H).map(|_| f32_to_bf16(r.u(-0.5, 0.5))).collect();

    // ── CPU reference (f64) ──
    let mut streams: Vec<f64> = (0..t * w)
        .map(|e| bf16_to_f32(hidden[(e / w) * H + e % H]) as f64 + drift[e] as f64)
        .collect();
    let (mut c_norm, mut c_inj, mut c_act, mut c_mix) = (vec![], vec![], vec![], vec![]);
    for tok in 0..t {
        let x = &streams[tok * w..(tok + 1) * w];
        let mut n = vec![0.0f64; w];
        for i in 0..HC {
            let ms = x[i * H..(i + 1) * H].iter().map(|v| v * v).sum::<f64>() / H as f64;
            let rs = 1.0 / (ms + EPS as f64).sqrt();
            for d in 0..H {
                n[i * H + d] = x[i * H + d] * rs * (1.0 + wt.norm[i * H + d] as f64);
            }
        }
        let dot = |row: &[u16], v: &[f64]| -> f64 {
            row.iter()
                .zip(v)
                .map(|(a, b)| bf16_to_f32(*a) as f64 * b)
                .sum()
        };
        for j in 0..HC {
            c_inj.push(2.0 * sigmoid(dot(&wt.inject[j * w..(j + 1) * w], &n) / HC as f64));
        }
        let act: Vec<f64> = (0..RANK)
            .map(|k| {
                let v = dot(&wt.down[k * w..(k + 1) * w], &n) / HC as f64;
                v * sigmoid(v)
            })
            .collect();
        for d in 0..H {
            let s: f64 = (0..HC)
                .map(|i| {
                    let row = &wt.up[(i * H + d) * RANK..(i * H + d + 1) * RANK];
                    sigmoid(dot(row, &act)) * n[i * H + d]
                })
                .sum();
            c_mix.push(s / HC as f64);
        }
        c_norm.extend(n);
        c_act.extend(act);
    }
    let c_streams_in = streams.clone();
    for tok in 0..t {
        for i in 0..HC {
            for d in 0..H {
                streams[tok * w + i * H + d] +=
                    bf16_to_f32(block_out[tok * H + d]) as f64 * c_inj[tok * HC + i];
            }
        }
    }

    // ── GPU ──
    let be = support::backend()?;
    let gpu: &dyn GpuBackend = &be;
    let st = gpu.create_stream()?;
    let k = |name: &str| gpu.kernel("gated_residual", name);
    let (tu, hu, hcu, ru) = (t as u32, H as u32, HC as u32, RANK as u32);

    let d_hidden = upload(gpu, &le16(&hidden))?;
    let d_block = upload(gpu, &le16(&block_out))?;
    let d_norm_w = upload(gpu, &le32(&wt.norm))?;
    let d_inject = upload(gpu, &le16(&wt.inject))?;
    let d_down = upload(gpu, &le16(&wt.down))?;
    let d_up = upload(gpu, &le16(&wt.up))?;
    let d_streams = gpu.alloc(t * w * 4)?;
    let d_normed = gpu.alloc(t * w * 4)?;
    let d_inj = gpu.alloc(t * HC * 4)?;
    let d_act = gpu.alloc(t * RANK * 4)?;
    let d_mixed = gpu.alloc(t * H * 2)?;

    KernelLaunch::new(gpu, k("gr_expand")?)
        .grid([tu, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(d_hidden)
        .arg_ptr(d_streams)
        .arg_u32(hu)
        .arg_u32(hcu)
        .launch(st)?;
    gpu.synchronize(st)?;
    let expanded = download_f32(gpu, d_streams, t * w)?;
    let expand_ok = expanded
        .iter()
        .enumerate()
        .all(|(e, v)| *v == bf16_to_f32(hidden[(e / w) * H + e % H]));
    println!("  expand   exact={expand_ok}");
    // Continue from the perturbed streams the reference used.
    let s_in: Vec<f32> = c_streams_in.iter().map(|v| *v as f32).collect();
    gpu.copy_h2d(&le32(&s_in), d_streams)?;

    KernelLaunch::new(gpu, k("gr_norm_inject")?)
        .grid([tu, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(d_streams)
        .arg_ptr(d_norm_w)
        .arg_ptr(d_inject)
        .arg_ptr(d_normed)
        .arg_ptr(d_inj)
        .arg_u32(hu)
        .arg_u32(hcu)
        .arg_f32(EPS)
        .launch(st)?;
    KernelLaunch::new(gpu, k("gr_down_silu")?)
        .grid([ru, tu, 1])
        .block([256, 1, 1])
        .arg_ptr(d_normed)
        .arg_ptr(d_down)
        .arg_ptr(d_act)
        .arg_u32(hu)
        .arg_u32(hcu)
        .arg_u32(ru)
        .launch(st)?;
    KernelLaunch::new(gpu, k("gr_up_mix")?)
        .grid([hu.div_ceil(256), tu, 1])
        .block([256, 1, 1])
        .arg_ptr(d_normed)
        .arg_ptr(d_act)
        .arg_ptr(d_up)
        .arg_ptr(d_mixed)
        .arg_u32(hu)
        .arg_u32(hcu)
        .arg_u32(ru)
        .launch(st)?;
    KernelLaunch::new(gpu, k("gr_post")?)
        .grid([tu, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(d_block)
        .arg_ptr(d_inj)
        .arg_ptr(d_streams)
        .arg_u32(hu)
        .arg_u32(hcu)
        .launch(st)?;
    gpu.synchronize(st)?;

    let g_mixed: Vec<f32> = {
        let mut b = vec![0u8; t * H * 2];
        gpu.copy_d2h(d_mixed, &mut b)?;
        u16s(&b).into_iter().map(bf16_to_f32).collect()
    };
    let mut ok = expand_ok;
    ok &= check(
        "normed",
        &download_f32(gpu, d_normed, t * w)?,
        &c_norm,
        1e-5,
    );
    ok &= check("inject", &download_f32(gpu, d_inj, t * HC)?, &c_inj, 1e-5);
    ok &= check("act", &download_f32(gpu, d_act, t * RANK)?, &c_act, 1e-4);
    // BF16 output: every element within one BF16 rounding step (2^-8 relative).
    let mixed_bad = g_mixed
        .iter()
        .zip(&c_mix)
        .filter(|(g, w)| !((**g as f64 - **w).abs() <= w.abs() / 256.0 + 1e-6))
        .count();
    println!(
        "  mixed    elements beyond one BF16 step: {mixed_bad} of {}",
        c_mix.len()
    );
    ok &= mixed_bad == 0 && c_mix.iter().any(|v| v.abs() > 1e-3);
    ok &= check(
        "streams",
        &download_f32(gpu, d_streams, t * w)?,
        &streams,
        1e-5,
    );
    let spread = c_inj.iter().fold(0.0f64, |m, v| m.max((v - 1.0).abs()));
    println!("  inject weights deviate from 1.0 by up to {spread:.3} (0 would hide a wrong gate)");

    for p in [
        d_hidden, d_block, d_norm_w, d_inject, d_down, d_up, d_streams, d_normed, d_inj, d_act,
        d_mixed,
    ] {
        gpu.free(p).ok();
    }
    support::finish(ok)
}

// SPDX-License-Identifier: AGPL-3.0-only
//! Oracle for `mxfp8_gemv` (Qwen3.8-Flash-Next): FP8-E4M3 weights with one
//! power-of-two scale byte per 32 columns, against an f64 CPU reference.
//!
//! Usage:
//!   cargo run --release -p spark-model --features cuda,gpu-examples \
//!       --example qwen4_mxfp8_gemv_microtest -- [tokens] [seed] [snapshot_dir] [tensor]
//!
//! With `snapshot_dir` the weight and its `weight_scale` are read from the
//! checkpoint (default tensor: layers.0.linear_attn.in_proj_qkv); without it
//! they are random and cover every E4M3 byte.
use anyhow::{Result, bail};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::KernelLaunch;

#[path = "qwen4_support/mod.rs"]
mod support;
use support::{Rng, bf16_to_f32, e4m3, f32_to_bf16, read_tensor, upload};

const DEFAULT_TENSOR: &str = "model.language_model.layers.0.linear_attn.in_proj_qkv";

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let t: usize = a.get(1).map_or(2, |s| s.parse().unwrap());
    let seed: u64 = a.get(2).map_or(0x8F8, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x8F8)
    });
    let dir = a.get(3).map(String::as_str);
    let tensor = a.get(4).map_or(DEFAULT_TENSOR, String::as_str);
    let mut r = Rng(seed);

    let (n, k, weight, scale) = if let Some(dir) = dir {
        let (ws, w) = read_tensor(dir, &format!("{tensor}.weight"), "F8_E4M3")?;
        let (ss, s) = read_tensor(dir, &format!("{tensor}.weight_scale"), "U8")?;
        if ws.len() != 2 || ss != [ws[0], ws[1] / 32] || ws[1] % 32 != 0 {
            bail!("{tensor}: weight {ws:?} and weight_scale {ss:?} are not MXFP8-shaped");
        }
        (ws[0], ws[1], w, s)
    } else {
        // N not a multiple of 4 exercises the partial last block.
        let (n, k) = (1030usize, 2560usize);
        let w: Vec<u8> = (0..n * k).map(|_| (r.n() >> 56) as u8).collect();
        let s: Vec<u8> = (0..n * k / 32).map(|_| 108 + (r.n() % 12) as u8).collect();
        (n, k, w, s)
    };
    println!(
        "=== mxfp8_gemv microtest: tokens={t} N={n} K={k} seed=0x{seed:X} weights={} ===",
        dir.map_or("random", |_| tensor)
    );
    let act: Vec<u16> = (0..t * k).map(|_| f32_to_bf16(r.u(-2.0, 2.0))).collect();

    // ── CPU reference (f64) ──
    let mut want = vec![0.0f64; t * n];
    let mut mag = vec![0.0f64; t * n]; // sum of |terms|: the scale rounding noise lives on
    for row in 0..n {
        let w: Vec<f64> = (0..k)
            .map(|c| {
                e4m3(weight[row * k + c]) * 2f64.powi(scale[row * (k / 32) + c / 32] as i32 - 127)
            })
            .collect();
        for tok in 0..t {
            let (mut s, mut m) = (0.0f64, 0.0f64);
            for c in 0..k {
                let term = bf16_to_f32(act[tok * k + c]) as f64 * w[c];
                s += term;
                m += term.abs();
            }
            want[tok * n + row] = s;
            mag[tok * n + row] = m;
        }
    }

    // ── GPU ──
    let be = support::backend()?;
    let gpu: &dyn GpuBackend = &be;
    let st = gpu.create_stream()?;
    let d_a = upload(
        gpu,
        &act.iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    let d_b = upload(gpu, &weight)?;
    let d_s = upload(gpu, &scale)?;
    let d_c = gpu.alloc(t * n * 2)?;
    KernelLaunch::new(gpu, gpu.kernel("mxfp8_gemv", "mxfp8_gemv")?)
        .grid([(n as u32).div_ceil(4), t as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(d_a)
        .arg_ptr(d_b)
        .arg_ptr(d_s)
        .arg_ptr(d_c)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(st)?;
    gpu.synchronize(st)?;
    let mut cb = vec![0u8; t * n * 2];
    gpu.copy_d2h(d_c, &mut cb)?;
    let got: Vec<f32> = cb
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();

    // Per element: one BF16 rounding step of the result plus FP32 summation
    // noise proportional to the sum of the term magnitudes.
    let mut bad = 0usize;
    let (mut worst, mut dot, mut ng, mut nw) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..t * n {
        let (g, w) = (got[i] as f64, want[i]);
        let tol = w.abs() / 256.0 + mag[i] * 1e-6 + 1e-30;
        if !((g - w).abs() <= tol) {
            bad += 1;
        }
        worst = worst.max((g - w).abs() / tol);
        dot += g * w;
        ng += g * g;
        nw += w * w;
    }
    let cosine = dot / (ng.sqrt() * nw.sqrt());
    let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    println!(
        "  elements beyond tolerance: {bad} of {}  worst={worst:.3} of tolerance  cosine={cosine:.7}  max|ref|={peak:.4e}",
        t * n
    );
    for p in [d_a, d_b, d_s, d_c] {
        gpu.free(p).ok();
    }
    support::finish(bad == 0 && cosine > 0.99999 && peak > 0.0)
}

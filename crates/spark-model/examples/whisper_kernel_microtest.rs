// SPDX-License-Identifier: AGPL-3.0-only

//! Correctness oracle for the whisper.cu kernels (multiclass Plan.md W.2
//! risk gate: validate the NEW attention kernel against a CPU reference
//! BEFORE any end-to-end encoder/decoder code).
//!
//! Covers: whisper_attention (non-causal = encoder/cross, causal =
//! decoder self-attn, including the growing-cache alignment nq < nk)
//! and whisper_layernorm (bias LayerNorm). Gates: zero NaN, cosine >=
//! 0.999 vs CPU.
//!
//! Usage: cargo run --release -p spark-model --features cuda,gpu-examples \
//!          --example whisper_kernel_microtest -- [seed]

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const GATE: f64 = 0.999;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32)
    }
}
fn b2f(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn f2b(f: f32) -> u16 {
    let bits = f.to_bits();
    let rb = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rb) >> 16) as u16
}
fn up(gpu: &dyn GpuBackend, v: &[u16]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}
fn down(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut raw = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| b2f(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}
fn verdict(name: &str, gpu_out: &[f32], cpu_out: &[f32]) -> Result<()> {
    let nan = gpu_out.iter().filter(|v| v.is_nan()).count();
    let (mut dot, mut ng, mut nc) = (0f64, 0f64, 0f64);
    for (g, c) in gpu_out.iter().zip(cpu_out) {
        dot += (*g as f64) * (*c as f64);
        ng += (*g as f64) * (*g as f64);
        nc += (*c as f64) * (*c as f64);
    }
    let cos = dot / (ng.sqrt() * nc.sqrt());
    println!("  {name}: cosine={cos:.6} nan={nan}");
    if nan > 0 || cos < GATE {
        bail!("{name} FAILED (cosine {cos:.6}, nan {nan})");
    }
    Ok(())
}

/// CPU reference: softmax(QK^T*scale)V over [n, H, D] bf16 values.
#[allow(clippy::too_many_arguments)]
fn cpu_attention(
    q: &[u16],
    k: &[u16],
    v: &[u16],
    nq: usize,
    nk: usize,
    h: usize,
    d: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut out = vec![0f32; nq * h * d];
    for qi in 0..nq {
        for hi in 0..h {
            let limit = if causal { nk - nq + qi + 1 } else { nk };
            let qv: Vec<f32> = (0..d).map(|x| b2f(q[(qi * h + hi) * d + x])).collect();
            let mut scores: Vec<f32> = (0..limit)
                .map(|j| {
                    (0..d)
                        .map(|x| qv[x] * b2f(k[(j * h + hi) * d + x]))
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let m = scores.iter().cloned().fold(f32::MIN, f32::max);
            let mut sum = 0f32;
            for s in &mut scores {
                *s = (*s - m).exp();
                sum += *s;
            }
            for x in 0..d {
                let o: f32 = (0..limit)
                    .map(|j| scores[j] * b2f(v[(j * h + hi) * d + x]))
                    .sum();
                out[(qi * h + hi) * d + x] = o / sum;
            }
        }
    }
    out
}

fn main() -> Result<()> {
    let seed: u64 = std::env::args().nth(1).map_or(0xA0D10, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0xA0D10)
    });
    let set = atlas_kernels::available_targets()
        .into_iter()
        .find(|t| t.target.model == "whisper-large-v3")
        .ok_or_else(|| anyhow::anyhow!("whisper-large-v3 target not compiled — build with ATLAS_TARGET_MODEL='*'"))?;
    println!("=== whisper kernel microtest: {} (seed 0x{seed:X}) ===", set.target);
    let backend = AtlasCudaBackend::new(0, &set.modules)?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let mut rng = Rng(seed);

    // ── attention: encoder-ish non-causal, decoder-ish causal ──
    let (h, d) = (20usize, 64usize);
    for (name, nq, nk, causal) in [
        ("attention non-causal [nq=nk=97]", 97usize, 97usize, false),
        ("attention cross [nq=13, nk=750]", 13, 750, false),
        ("attention causal growing [nq=5, nk=41]", 5, 41, true),
        ("attention decode step [nq=1, nk=448]", 1, 448, true),
    ] {
        let q: Vec<u16> = (0..nq * h * d).map(|_| f2b(rng.uniform(-1.0, 1.0))).collect();
        let k: Vec<u16> = (0..nk * h * d).map(|_| f2b(rng.uniform(-1.0, 1.0))).collect();
        let v: Vec<u16> = (0..nk * h * d).map(|_| f2b(rng.uniform(-1.0, 1.0))).collect();
        let (qp, kp, vp) = (up(gpu, &q)?, up(gpu, &k)?, up(gpu, &v)?);
        let op = gpu.alloc(nq * h * d * 2)?;
        let scale = 1.0 / (d as f32).sqrt();
        let kern = gpu.kernel("whisper", "whisper_attention")?;
        KernelLaunch::new(gpu, kern)
            .grid([nq as u32, h as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(qp)
            .arg_ptr(kp)
            .arg_ptr(vp)
            .arg_ptr(op)
            .arg_u32(nq as u32)
            .arg_u32(nk as u32)
            .arg_u32(h as u32)
            .arg_u32(d as u32)
            .arg_f32(scale)
            .arg_u32(causal as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let got = down(gpu, op, nq * h * d)?;
        let want = cpu_attention(&q, &k, &v, nq, nk, h, d, scale, causal);
        verdict(name, &got, &want)?;
    }

    // ── layernorm with bias ──
    let (t, hid) = (7usize, 1280usize);
    let x: Vec<u16> = (0..t * hid).map(|_| f2b(rng.uniform(-2.0, 2.0))).collect();
    let g: Vec<u16> = (0..hid).map(|_| f2b(rng.uniform(0.5, 1.5))).collect();
    let b: Vec<u16> = (0..hid).map(|_| f2b(rng.uniform(-0.5, 0.5))).collect();
    let (xp, gp, bp) = (up(gpu, &x)?, up(gpu, &g)?, up(gpu, &b)?);
    let op = gpu.alloc(t * hid * 2)?;
    let kern = gpu.kernel("whisper", "whisper_layernorm")?;
    KernelLaunch::new(gpu, kern)
        .grid([t as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(xp)
        .arg_ptr(gp)
        .arg_ptr(bp)
        .arg_ptr(op)
        .arg_u32(hid as u32)
        .arg_f32(1e-5)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let got = down(gpu, op, t * hid)?;
    let mut want = vec![0f32; t * hid];
    for ti in 0..t {
        let row: Vec<f32> = (0..hid).map(|i| b2f(x[ti * hid + i])).collect();
        let mean = row.iter().sum::<f32>() / hid as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hid as f32;
        let rstd = 1.0 / (var + 1e-5).sqrt();
        for i in 0..hid {
            want[ti * hid + i] = (row[i] - mean) * rstd * b2f(g[i]) + b2f(b[i]);
        }
    }
    verdict("layernorm [7x1280]", &got, &want)?;

    println!("=== whisper kernel microtest: PASS ===");
    Ok(())
}

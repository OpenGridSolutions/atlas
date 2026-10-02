// SPDX-License-Identifier: AGPL-3.0-only

//! bf16-target smoke test — Plan.md (multiclass) step 1.0, the HARD GATE
//! before any dense-loader work.
//!
//! `quant=bf16` is supported by the runtime but the `qwen3-dense-8b/bf16`
//! target is the FIRST bf16-labeled target ever built. "Supported but never
//! exercised" is exactly the latent-bug category this gate exists for. This
//! harness:
//!
//!   1. resolves the `qwen3-dense-8b` / `bf16` target from the compiled
//!      target set (bails with a rebuild hint if absent),
//!   2. initializes the production `AtlasCudaBackend`/`AtlasRegistry` path
//!      with THAT target's modules (not target 0),
//!   3. runs the bf16 compute kernels a dense bf16 model's forward actually
//!      hits with bf16 WEIGHTS — `dense_gemm_bf16` (prefill projections),
//!      `dense_gemv_bf16` (decode projections), `rms_norm` — against
//!      independent CPU references.
//!
//! Pass gates (per kernel): zero NaNs, non-degenerate output (non-zero
//! variance), cosine vs CPU ≥ gate. Exit 0 = PASS, 1 = FAIL — scriptable
//! (scripts/bf16_smoke_test.sh).
//!
//! Deliberately NOT covered here: the paged attention / prefill-attention
//! kernels. Under `-DHDIM=128` + the common flag set, this target's
//! attention PTX is byte-identical (build-cache-deduped) to the modules the
//! six existing head_dim=128 nvfp4 targets ship — those binaries are not new
//! surface. The bf16 GEMM/GEMV weight path is: no production model runs
//! bf16 weights today.
//!
//! Usage:
//!   cargo run --release -p spark-model --features cuda,gpu-examples \
//!       --example bf16_target_smoke -- [seed]

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const GEMM_COSINE_GATE: f64 = 0.999; // BF16 MMA-reorder noise (dense_gemm_microtest SSOT)
const GEMV_COSINE_GATE: f64 = 0.999;
const NORM_COSINE_GATE: f64 = 0.9999; // elementwise op — much tighter

// ── deterministic PRNG (splitmix64, per microtest convention) ──
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

fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}
fn u16s_to_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}
fn download_bf16(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut raw = vec![0u8; n * 2];
    gpu.copy_d2h(ptr, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Cosine + NaN + degeneracy verdict for one kernel. Returns Err on FAIL so
/// the first failing kernel aborts the smoke with a precise message.
fn verdict(name: &str, gpu_out: &[u16], cpu_out: &[u16], gate: f64) -> Result<()> {
    let mut nan = 0usize;
    let (mut dot, mut ng, mut nc) = (0f64, 0f64, 0f64);
    let (mut sum, mut sum_sq) = (0f64, 0f64);
    for i in 0..gpu_out.len() {
        let g = bf16_bits_to_f32(gpu_out[i]);
        if g.is_nan() {
            nan += 1;
            continue;
        }
        let c = bf16_bits_to_f32(cpu_out[i]) as f64;
        let g = g as f64;
        dot += g * c;
        ng += g * g;
        nc += c * c;
        sum += g;
        sum_sq += g * g;
    }
    let n = gpu_out.len() as f64;
    let variance = (sum_sq / n) - (sum / n) * (sum / n);
    let cosine = if ng > 0.0 && nc > 0.0 {
        dot / (ng.sqrt() * nc.sqrt())
    } else {
        0.0
    };
    println!("  {name}: cosine={cosine:.6} nan={nan} variance={variance:.3e} (gate {gate})");
    if nan > 0 {
        bail!("{name}: {nan} NaN outputs — bf16 path is broken");
    }
    if variance <= 0.0 {
        bail!("{name}: degenerate (zero-variance) output — kernel wrote nothing real");
    }
    if cosine < gate {
        bail!("{name}: cosine {cosine:.6} below gate {gate} vs CPU reference");
    }
    Ok(())
}

fn main() -> Result<()> {
    let seed: u64 = std::env::args().nth(1).map_or(0xBF16, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0xBF16)
    });

    // ── 1. resolve the bf16 target (the thing under test) ──
    let set = atlas_kernels::available_targets()
        .into_iter()
        .find(|t| t.target.model == "qwen3-dense-8b" && t.target.quant == "bf16")
        .ok_or_else(|| {
            anyhow::anyhow!(
                "target qwen3-dense-8b/bf16 not in the compiled set — rebuild with \
                 ATLAS_TARGET_MODEL='*' ATLAS_TARGET_QUANT='*' (or =qwen3-dense-8b / =bf16). \
                 Compiled targets: {:?}",
                atlas_kernels::available_targets()
                    .iter()
                    .map(|t| format!("{}/{}", t.target.model, t.target.quant))
                    .collect::<Vec<_>>()
            )
        })?;
    println!(
        "=== bf16 target smoke: {} ({} modules, seed 0x{seed:X}) ===",
        set.target,
        set.modules.len()
    );
    // Selection sanity: this target must be what ptx_for_config resolves for
    // a dense qwen3/4096 checkpoint (the runtime path Embedding/Reranker use).
    match atlas_kernels::ptx_for_config("qwen3", 4096) {
        Some(t) if t.target.model == "qwen3-dense-8b" => {
            println!("  ptx_for_config(\"qwen3\", 4096) -> {} ✓", t.target)
        }
        other => bail!(
            "ptx_for_config(\"qwen3\", 4096) resolved {:?} instead of qwen3-dense-8b — \
             [[model_types]] wiring is wrong",
            other.map(|t| t.target.to_string())
        ),
    }

    let backend = AtlasCudaBackend::new(0, &set.modules)?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let mut rng = Rng(seed);

    // ── 2. dense_gemm_bf16 (module "gemm") — prefill projection shape ──
    // C[M,N] = A[M,K] · B[N,K]^T, fp32 accum. M=64 tokens, K=N=4096 (o_proj).
    {
        let (m, n, k) = (64usize, 4096usize, 4096usize);
        let a: Vec<u16> = (0..m * k)
            .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
            .collect();
        let b: Vec<u16> = (0..n * k)
            .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
            .collect();
        let a_ptr = upload(gpu, &u16s_to_le(&a))?;
        let b_ptr = upload(gpu, &u16s_to_le(&b))?;
        let c_ptr = gpu.alloc(m * n * 2)?;
        let handle = gpu.kernel("gemm", "dense_gemm_bf16")?;
        KernelLaunch::new(gpu, handle)
            .grid([(n as u32).div_ceil(16), (m as u32).div_ceil(16), 1])
            .block([16, 16, 1])
            .arg_ptr(a_ptr)
            .arg_ptr(b_ptr)
            .arg_ptr(c_ptr)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let c_gpu = download_bf16(gpu, c_ptr, m * n)?;
        let mut c_cpu = vec![0u16; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += bf16_bits_to_f32(a[row * k + kk]) * bf16_bits_to_f32(b[col * k + kk]);
                }
                c_cpu[row * n + col] = f32_to_bf16_bits(acc);
            }
        }
        verdict(
            "dense_gemm_bf16 [64x4096x4096]",
            &c_gpu,
            &c_cpu,
            GEMM_COSINE_GATE,
        )?;
    }

    // ── 3. dense_gemv_bf16 (module "gemv") — decode projection shape ──
    // C[1,N] = A[1,K] · B[N,K]^T. N=12288 (MLP up), K=4096.
    {
        let (n, k) = (12288usize, 4096usize);
        let a: Vec<u16> = (0..k)
            .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
            .collect();
        let b: Vec<u16> = (0..n * k)
            .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
            .collect();
        let a_ptr = upload(gpu, &u16s_to_le(&a))?;
        let b_ptr = upload(gpu, &u16s_to_le(&b))?;
        let c_ptr = gpu.alloc(n * 2)?;
        let handle = gpu.kernel("gemv", "dense_gemv_bf16")?;
        // Grid: (ceil(N / N_PER_BLOCK=4), 1, 1), Block: (256,1,1) — kernel header.
        KernelLaunch::new(gpu, handle)
            .grid([(n as u32).div_ceil(4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a_ptr)
            .arg_ptr(b_ptr)
            .arg_ptr(c_ptr)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let c_gpu = download_bf16(gpu, c_ptr, n)?;
        let mut c_cpu = vec![0u16; n];
        for (col, slot) in c_cpu.iter_mut().enumerate() {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += bf16_bits_to_f32(a[kk]) * bf16_bits_to_f32(b[col * k + kk]);
            }
            *slot = f32_to_bf16_bits(acc);
        }
        verdict(
            "dense_gemv_bf16 [1x12288x4096]",
            &c_gpu,
            &c_cpu,
            GEMV_COSINE_GATE,
        )?;
    }

    // ── 4. rms_norm (module "norm") — (1 + weight) scaling per kernel doc ──
    {
        let (t, h) = (64usize, 4096usize);
        let eps = 1e-6f32;
        let x: Vec<u16> = (0..t * h)
            .map(|_| f32_to_bf16_bits(rng.uniform(-2.0, 2.0)))
            .collect();
        let w: Vec<u16> = (0..h)
            .map(|_| f32_to_bf16_bits(rng.uniform(-0.5, 0.5)))
            .collect();
        let x_ptr = upload(gpu, &u16s_to_le(&x))?;
        let w_ptr = upload(gpu, &u16s_to_le(&w))?;
        let o_ptr = gpu.alloc(t * h * 2)?;
        let handle = gpu.kernel("norm", "rms_norm")?;
        // Grid: (num_tokens,1,1), Block: (min(hidden,1024),1,1) — kernel header.
        KernelLaunch::new(gpu, handle)
            .grid([t as u32, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(x_ptr)
            .arg_ptr(w_ptr)
            .arg_ptr(o_ptr)
            .arg_u32(h as u32)
            .arg_f32(eps)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let o_gpu = download_bf16(gpu, o_ptr, t * h)?;
        let mut o_cpu = vec![0u16; t * h];
        for tok in 0..t {
            let row = &x[tok * h..(tok + 1) * h];
            let mut sum_sq = 0f32;
            for &b in row {
                let v = bf16_bits_to_f32(b);
                sum_sq += v * v;
            }
            let inv_rms = 1.0 / (sum_sq / h as f32 + eps).sqrt();
            for i in 0..h {
                let v = bf16_bits_to_f32(row[i]);
                let wt = 1.0 + bf16_bits_to_f32(w[i]);
                o_cpu[tok * h + i] = f32_to_bf16_bits(v * inv_rms * wt);
            }
        }
        verdict("rms_norm [64x4096]", &o_gpu, &o_cpu, NORM_COSINE_GATE)?;
    }

    println!("=== bf16 target smoke: PASS (gate Plan.md 1.0 satisfied) ===");
    Ok(())
}

// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen-Image-2.1 golden test (Todo 3.2) — compares every pipeline stage of
//! the native sidecar against tensors dumped from the diffusers reference
//! (`scripts/qwen_image_ref_dump.py`, CPU fp32):
//!
//!   text : text tower hidden states for the reference token ids
//!   dit  : sigma schedule, prefix/target block activations, per-step
//!          noise predictions and latents (reference prompt embeddings and
//!          reference noise as inputs, so stages are isolated)
//!   vae  : decoder block outputs and the final RGBA image
//!
//! The reference is fp32, the sidecar BF16 with fp32 accumulation, so the
//! gates are cosine similarities, not exact matches. Exit 0 = all gates
//! pass, 1 = any failure.
//!
//! Usage:
//!   cargo run --release -p spark-model --features cuda,gpu-examples \
//!       --example qwen_image_golden -- <golden_dir> [model_dir]
//!
//! With `QIMG_GENERATE=<out.png>` (optional `QIMG_SIZE=1024`,
//! `QIMG_STEPS=40`, `QIMG_SEED=42`) it additionally runs a full generation
//! for the reference prompt and reports the timing — the quick performance
//! probe before going through the server.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use spark_model::qwen_image::st::StDir;
use spark_model::qwen_image::{QwenImage, Trace, sched};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

struct Report {
    failures: usize,
}

impl Report {
    /// Cosine / max-abs / relative-RMS of `got` vs `want`; gate on cosine.
    fn check(&mut self, name: &str, got: &[f32], want: &[f32], gate: f64) {
        if got.len() != want.len() {
            println!(
                "  [FAIL] {name:<24} length {} vs reference {}",
                got.len(),
                want.len()
            );
            self.failures += 1;
            return;
        }
        let (mut dot, mut ng, mut nw, mut err, mut max_abs) = (0f64, 0f64, 0f64, 0f64, 0f64);
        let mut nan = 0usize;
        for (g, w) in got.iter().zip(want) {
            if !g.is_finite() {
                nan += 1;
                continue;
            }
            let (g, w) = (*g as f64, *w as f64);
            dot += g * w;
            ng += g * g;
            nw += w * w;
            err += (g - w) * (g - w);
            max_abs = max_abs.max((g - w).abs());
        }
        let cos = dot / (ng.sqrt() * nw.sqrt()).max(1e-30);
        let rel = (err / nw.max(1e-30)).sqrt();
        let ok = nan == 0 && cos >= gate;
        println!(
            "  [{}] {name:<24} cos {cos:.6}  rel-rms {rel:.4}  max-abs {max_abs:.4}  (gate {gate}){}",
            if ok { "PASS" } else { "FAIL" },
            if nan > 0 {
                format!("  {nan} non-finite!")
            } else {
                String::new()
            }
        );
        if !ok {
            self.failures += 1;
        }
    }
}

/// Reference `[1, C, (1,) H, W]` → NHWC `[H*W, C]`.
fn nchw_to_nhwc(v: &[f32], c: usize) -> Vec<f32> {
    let hw = v.len() / c;
    let mut out = vec![0f32; v.len()];
    for ch in 0..c {
        for p in 0..hw {
            out[p * c + ch] = v[ch * hw + p];
        }
    }
    out
}

fn find<'a>(trace: &'a Trace, name: &str) -> Result<&'a [f32]> {
    trace
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| v.as_slice())
        .with_context(|| format!("trace entry '{name}' missing"))
}

fn default_model_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    let snaps =
        Path::new(&home).join(".cache/huggingface/hub/models--Qwen--Qwen-Image-2.1/snapshots");
    let mut dirs: Vec<_> = std::fs::read_dir(&snaps)
        .with_context(|| format!("no snapshot under {}", snaps.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    dirs.sort();
    dirs.pop()
        .context("Qwen-Image-2.1 snapshot not found in the HF cache")
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let golden = PathBuf::from(
        args.next()
            .context("usage: qwen_image_golden <golden_dir> [model_dir]")?,
    );
    let model_dir = match args.next() {
        Some(d) => PathBuf::from(d),
        None => default_model_dir()?,
    };

    let set = atlas_kernels::ptx_for_model_id("Qwen/Qwen-Image-2.1", "qwen_image", 4096)
        .filter(|t| t.target.model == "qwen-image-2.1")
        .context(
            "target qwen-image-2.1 not in the compiled set — rebuild with \
             ATLAS_TARGET_MODEL='*' ATLAS_TARGET_QUANT='*'",
        )?;
    println!(
        "=== qwen-image golden: {} ({} modules) ===",
        set.target,
        set.modules.len()
    );
    let backend = AtlasCudaBackend::new(0, &set.modules)?;
    backend.bind_to_thread()?;
    let t0 = std::time::Instant::now();
    let model = QwenImage::load(&model_dir, Box::new(backend))?;
    println!(
        "loaded {:.1} GiB in {:.1}s",
        model.weights_bytes() as f64 / (1u64 << 30) as f64,
        t0.elapsed().as_secs_f64()
    );
    let mut rep = Report { failures: 0 };

    // ── text encoder ──
    println!("text encoder:");
    let text = StDir::open_files(&[golden.join("text.safetensors")])?;
    let ids: Vec<u32> = text
        .read_i64("input_ids")?
        .iter()
        .map(|v| *v as u32)
        .collect();
    let drop = text.read_i64("drop_idx")?[0] as usize;
    let t1 = std::time::Instant::now();
    let hidden = model.text_hidden(&ids)?;
    println!(
        "  {} tokens in {:.2}s",
        ids.len(),
        t1.elapsed().as_secs_f64()
    );
    rep.check(
        "hidden (all tokens)",
        &hidden,
        &text.read_f32("hidden_all")?,
        0.995,
    );
    let embeds = text.read_f32("prompt_embeds")?;
    rep.check("prompt_embeds", &hidden[drop * 4096..], &embeds, 0.995);

    // ── DiT ──
    println!("DiT:");
    let dit = StDir::open_files(&[golden.join("dit.safetensors")])?;
    let hw = dit.read_i64("lat_hw")?;
    let (lat_h, lat_w) = (hw[0] as usize, hw[1] as usize);
    let n = lat_h * lat_w;
    let tp = embeds.len() / 4096;
    let want_sigmas = dit.read_f32("sigmas")?;
    let steps = want_sigmas.len() - 1;
    rep.check("sigmas", &sched::sigmas(steps, n), &want_sigmas, 0.999_999);

    let mut latents = dit.read_f32("noise_packed")?;
    let mut trace = Trace::new();
    let t2 = std::time::Instant::now();
    model.denoise_host(&embeds, lat_h, lat_w, &mut latents, steps, Some(&mut trace))?;
    println!(
        "  {steps} steps, {tp} prompt + {n} target tokens in {:.2}s",
        t2.elapsed().as_secs_f64()
    );
    rep.check(
        "txt_in",
        find(&trace, "txt_in")?,
        &dit.read_f32("step0.txt_in")?,
        0.999,
    );
    for l in [0usize, 1, 15, 31] {
        // The reference's first step ran the joint sequence: prefix rows
        // first, then the target rows.
        let joint = dit.read_f32(&format!("step0.block{l}"))?;
        rep.check(
            &format!("prefix.block{l}"),
            find(&trace, &format!("prefix.block{l}"))?,
            &joint[..tp * 4096],
            if l == 31 { 0.9 } else { 0.99 },
        );
        rep.check(
            &format!("step0.block{l}"),
            find(&trace, &format!("step0.block{l}"))?,
            &joint[tp * 4096..],
            0.99,
        );
        rep.check(
            &format!("step1.block{l}"),
            find(&trace, &format!("step1.block{l}"))?,
            &dit.read_f32(&format!("step1.block{l}"))?,
            0.99,
        );
    }
    for i in 0..steps {
        rep.check(
            &format!("noise_pred.{i}"),
            find(&trace, &format!("noise_pred.{i}"))?,
            &dit.read_f32(&format!("noise_pred.{i}"))?,
            0.99,
        );
    }
    let want_final = dit.read_f32("final_latents")?;
    rep.check("final_latents", &latents, &want_final, 0.99);

    // ── VAE decoder (reference latents in, to isolate the stage) ──
    println!("VAE decoder:");
    let vae = StDir::open_files(&[golden.join("vae.safetensors")])?;
    let mut vtrace = Trace::new();
    let t3 = std::time::Instant::now();
    let img = model.decode_host(&want_final, lat_h, lat_w, Some(&mut vtrace))?;
    println!(
        "  {}x{} px in {:.2}s",
        lat_w * 16,
        lat_h * 16,
        t3.elapsed().as_secs_f64()
    );
    for (name, c) in [
        ("post_quant_conv", 64usize),
        ("conv_in", 1152),
        ("mid.res0", 1152),
        ("mid.attn0", 1152),
        ("mid", 1152),
        ("up0", 1152),
        ("up1", 1152),
        ("up2", 576),
        ("up3", 288),
        ("up4", 144),
    ] {
        rep.check(
            name,
            find(&vtrace, name)?,
            &nchw_to_nhwc(&vae.read_f32(name)?, c),
            0.99,
        );
    }
    rep.check(
        "image",
        &img,
        &nchw_to_nhwc(&vae.read_f32("image")?, 4),
        0.99,
    );

    if let Ok(out) = std::env::var("QIMG_GENERATE") {
        let env = |k: &str, d: u64| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let size = env("QIMG_SIZE", 1024) as usize;
        let params = spark_model::qwen_image::GenParams {
            width: size,
            height: size,
            steps: env("QIMG_STEPS", 40) as usize,
            seed: env("QIMG_SEED", 42),
        };
        println!("generate: {size}x{size}, {} steps", params.steps);
        let t4 = std::time::Instant::now();
        let mut last = std::time::Instant::now();
        let image = model.generate(&ids, drop, &params, |done, total| {
            if done <= 2 || done == total {
                println!(
                    "  step {done}/{total}: {:.2}s",
                    last.elapsed().as_secs_f64()
                );
            }
            last = std::time::Instant::now();
        })?;
        println!(
            "  total {:.1}s (incl. text encoder + VAE)",
            t4.elapsed().as_secs_f64()
        );
        image::save_buffer(
            &out,
            &image.pixels,
            image.width as u32,
            image.height as u32,
            image::ColorType::Rgba8,
        )?;
        println!("  wrote {out}");
    }

    if rep.failures > 0 {
        bail!("{} golden check(s) FAILED", rep.failures);
    }
    println!("ALL GOLDEN CHECKS PASSED");
    Ok(())
}

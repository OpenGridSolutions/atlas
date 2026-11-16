// SPDX-License-Identifier: AGPL-3.0-only
//! Oracle for the Qwen3.8-Flash-Next sparse-attention indexer kernels
//! (module `qsa`): qsa_index_raw, qsa_index_pool, qsa_score, qsa_select, qsa_gather against
//! an f64 CPU reference of the published selection rule.
//!
//! A sequence is decoded token by token with a small budget (3 blocks of 4
//! tokens), so the top-k selection is active from token 16 on. KV blocks are
//! mapped to shuffled physical blocks. Checked per token: the block scores,
//! the selected block set, the number of gathered tokens and the gathered
//! K/V rows.
//!
//! Usage:
//!   cargo run --release -p spark-model --features cuda,gpu-examples \
//!       --example qwen4_qsa_microtest -- [tokens] [seed]
use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::KernelLaunch;

#[path = "qwen4_support/mod.rs"]
mod support;
use support::{Rng, bf16_to_f32, f32_to_bf16, f32s, le16, u16s, upload};

const D: usize = 128;
const HEADS: usize = 4;
const RATIO: usize = 4;
const TOPK: usize = 3;
const BS: usize = 16;
const ROT: usize = 64;
const THETA: f64 = 1e7;
const EPS: f64 = 1e-6;
const ROW: usize = 16; // K/V elements per token
const MAX_BLOCKS: usize = 64; // index blocks the score buffer holds
const AUX_BLOCK: usize = (BS + BS / RATIO) * D * 2; // bytes

fn r16(v: f64) -> f64 {
    bf16_to_f32(f32_to_bf16(v as f32)) as f64
}
fn rms1p(x: &[f64], w: &[f64]) -> Vec<f64> {
    let rs = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + EPS).sqrt();
    x.iter()
        .zip(w)
        .map(|(v, w)| r16(v * rs * (1.0 + w)))
        .collect()
}
fn rope(x: &mut [f64], pos: usize) {
    for i in 0..ROT / 2 {
        let ang = pos as f64 * THETA.powf(-2.0 * i as f64 / ROT as f64);
        let (a, b) = (x[i], x[i + ROT / 2]);
        x[i] = r16(a * ang.cos() - b * ang.sin());
        x[i + ROT / 2] = r16(b * ang.cos() + a * ang.sin());
    }
}

/// Geometry arguments of qsa_score.
fn common(l: KernelLaunch<'_>) -> KernelLaunch<'_> {
    l.arg_u64(AUX_BLOCK as u64)
        .arg_u32(BS as u32)
        .arg_u32(RATIO as u32)
        .arg_u32(D as u32)
        .arg_u32(HEADS as u32)
        .arg_u32(ROT as u32)
        .arg_f32(THETA as f32)
        .arg_f32(EPS as f32)
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let t_total: usize = a.get(1).map_or(70, |s| s.parse().unwrap());
    let seed: u64 = a.get(2).map_or(0x05A, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x05A)
    });
    println!(
        "=== qsa microtest: tokens={t_total} seed=0x{seed:X} budget={} ===",
        TOPK * RATIO
    );
    let mut r = Rng(seed);
    let q_norm: Vec<u16> = r.bf16s(D, -0.3, 0.3);
    let k_norm: Vec<u16> = r.bf16s(D, -0.3, 0.3);
    let w64 = |w: &[u16]| -> Vec<f64> { w.iter().map(|x| bf16_to_f32(*x) as f64).collect() };
    let (q_norm64, k_norm64) = (w64(&q_norm), w64(&k_norm));

    // Shuffled physical blocks; block 0 stays unused.
    let logical_blocks = t_total.div_ceil(BS);
    let block_table: Vec<i32> = (0..logical_blocks)
        .map(|b| (1 + (b * 5 + 3) % (logical_blocks + 1)) as i32)
        .collect();
    let phys_blocks = logical_blocks + 2;
    let phys = |p: usize| block_table[p / BS] as usize * BS + p % BS;

    let be = support::backend()?;
    let gpu: &dyn GpuBackend = &be;
    let st = gpu.create_stream()?;
    let k = |name: &str| gpu.kernel("qsa", name);
    let d_qnorm = upload(gpu, &le16(&q_norm))?;
    let d_knorm = upload(gpu, &le16(&k_norm))?;
    let d_bt = upload(
        gpu,
        &block_table
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    let d_aux = upload(gpu, &vec![0x7Fu8; phys_blocks * AUX_BLOCK])?; // dirty
    let d_kc = upload(gpu, &vec![0u8; phys_blocks * BS * ROW * 2])?;
    let d_vc = upload(gpu, &vec![0u8; phys_blocks * BS * ROW * 2])?;
    let out_rows = TOPK * RATIO + RATIO - 1;
    let d_ko = gpu.alloc(out_rows * ROW * 2)?;
    let d_vo = gpu.alloc(out_rows * ROW * 2)?;
    let d_qk = gpu.alloc((HEADS + 1) * D * 2)?;
    let d_pos = gpu.alloc(4)?;
    let d_scores = gpu.alloc(MAX_BLOCKS * 4)?;
    let d_sel = gpu.alloc(TOPK * 4)?;
    let d_count = gpu.alloc(4)?;
    let d_tokens = gpu.alloc(4)?;

    let (mut raw_keys, mut index_keys): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (Vec::new(), Vec::new());
    let (mut k_rows, mut v_rows): (Vec<Vec<u16>>, Vec<Vec<u16>>) = (Vec::new(), Vec::new());
    let (mut ok, mut sparse_steps, mut worst_score) = (true, 0usize, 0.0f64);
    for pos in 0..t_total {
        let qk: Vec<u16> = r.bf16s((HEADS + 1) * D, -1.0, 1.0);
        // K/V row of the token: element 0 identifies the position.
        let mut kr = r.bf16s(ROW, -1.0, 1.0);
        let mut vr = r.bf16s(ROW, -1.0, 1.0);
        kr[0] = f32_to_bf16(pos as f32);
        vr[0] = f32_to_bf16(-(pos as f32));
        gpu.copy_h2d(&le16(&qk), d_qk)?;
        gpu.copy_h2d(&(pos as u32).to_le_bytes(), d_pos)?;
        gpu.copy_h2d(&le16(&kr), d_kc.offset(phys(pos) * ROW * 2))?;
        gpu.copy_h2d(&le16(&vr), d_vc.offset(phys(pos) * ROW * 2))?;
        k_rows.push(kr);
        v_rows.push(vr);

        // ── reference ──
        let qk64: Vec<f64> = qk.iter().map(|x| bf16_to_f32(*x) as f64).collect();
        raw_keys.push(qk64[HEADS * D..].to_vec());
        if (pos + 1) % RATIO == 0 {
            let g0 = pos + 1 - RATIO;
            let mean: Vec<f64> = (0..D)
                .map(|d| r16(raw_keys[g0..=pos].iter().map(|k| k[d]).sum::<f64>() / RATIO as f64))
                .collect();
            let mut key = rms1p(&mean, &k_norm64);
            rope(&mut key, g0);
            index_keys.push(key);
        }
        let visible = (pos + 1) / RATIO;
        let queries: Vec<Vec<f64>> = (0..HEADS)
            .map(|h| {
                let mut q = rms1p(&qk64[h * D..(h + 1) * D], &q_norm64);
                rope(&mut q, pos);
                q
            })
            .collect();
        let scores: Vec<f64> = index_keys[..visible]
            .iter()
            .map(|key| {
                queries
                    .iter()
                    .map(|q| q.iter().zip(key).map(|(a, b)| a * b).sum::<f64>().max(0.0))
                    .sum()
            })
            .collect();
        let mut want: Vec<usize> = (0..visible).collect();
        if visible > TOPK {
            want.sort_by(|a, b| scores[*b].partial_cmp(&scores[*a]).unwrap().then(a.cmp(b)));
            want.truncate(TOPK);
            want.sort_unstable();
            sparse_steps += 1;
        }

        // ── GPU ──
        KernelLaunch::new(gpu, k("qsa_index_raw")?)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(d_qk)
            .arg_ptr(d_aux)
            .arg_ptr(d_bt)
            .arg_ptr(d_pos)
            .arg_u64(AUX_BLOCK as u64)
            .arg_u32(BS as u32)
            .arg_u32(D as u32)
            .arg_u32(HEADS as u32)
            .launch(st)?;
        KernelLaunch::new(gpu, k("qsa_index_pool")?)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(d_knorm)
            .arg_ptr(d_aux)
            .arg_ptr(d_bt)
            .arg_ptr(d_pos)
            .arg_u64(AUX_BLOCK as u64)
            .arg_u32(BS as u32)
            .arg_u32(RATIO as u32)
            .arg_u32(D as u32)
            .arg_u32(ROT as u32)
            .arg_f32(THETA as f32)
            .arg_f32(EPS as f32)
            .launch(st)?;
        common(
            KernelLaunch::new(gpu, k("qsa_score")?)
                .grid([(MAX_BLOCKS as u32).div_ceil(256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(d_qk)
                .arg_ptr(d_qnorm)
                .arg_ptr(d_aux)
                .arg_ptr(d_bt)
                .arg_ptr(d_pos)
                .arg_ptr(d_scores),
        )
        .launch(st)?;
        KernelLaunch::new(gpu, k("qsa_select")?)
            .grid([1, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(d_scores)
            .arg_ptr(d_pos)
            .arg_ptr(d_sel)
            .arg_ptr(d_count)
            .arg_u32(RATIO as u32)
            .arg_u32(TOPK as u32)
            .launch(st)?;
        KernelLaunch::new(gpu, k("qsa_gather")?)
            .grid([TOPK as u32 + 1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(d_sel)
            .arg_ptr(d_count)
            .arg_ptr(d_pos)
            .arg_ptr(d_bt)
            .arg_ptr(d_kc)
            .arg_ptr(d_vc)
            .arg_ptr(d_ko)
            .arg_ptr(d_vo)
            .arg_ptr(d_tokens)
            .arg_u32(BS as u32)
            .arg_u32(RATIO as u32)
            .arg_u32(ROW as u32)
            .arg_u32(TOPK as u32)
            .launch(st)?;
        gpu.synchronize(st)?;

        let dl = |p, n: usize| support::download(gpu, p, n);
        let count = i32::from_le_bytes(dl(d_count, 4)?.try_into().unwrap_or([0; 4])) as usize;
        let tokens = i32::from_le_bytes(dl(d_tokens, 4)?.try_into().unwrap_or([0; 4])) as usize;
        let sel: Vec<usize> = dl(d_sel, TOPK * 4)?
            .chunks_exact(4)
            .take(count.min(TOPK))
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
            .collect();
        let g_scores = f32s(&dl(d_scores, visible.max(1) * 4)?);
        let mut step_ok = count == want.len();
        let mut sorted = sel.clone();
        sorted.sort_unstable();
        step_ok &= sorted == want;
        for (b, s) in scores.iter().enumerate() {
            let err = (g_scores[b] as f64 - s).abs() / s.abs().max(1.0);
            worst_score = worst_score.max(err);
            // A BF16 rounding step of one rotated element moves a score by ~1e-3.
            step_ok &= err < 5e-3;
        }
        // Gathered rows: the selected blocks in the kernel's order, then the tail.
        let tail_start = visible * RATIO;
        let mut expect: Vec<usize> = sel
            .iter()
            .flat_map(|b| b * RATIO..(b + 1) * RATIO)
            .collect();
        expect.extend(tail_start..=pos);
        step_ok &= tokens == expect.len();
        let (gk, gv) = (
            u16s(&dl(d_ko, out_rows * ROW * 2)?),
            u16s(&dl(d_vo, out_rows * ROW * 2)?),
        );
        for (i, p) in expect.iter().enumerate().take(out_rows) {
            step_ok &= gk[i * ROW..(i + 1) * ROW] == k_rows[*p][..];
            step_ok &= gv[i * ROW..(i + 1) * ROW] == v_rows[*p][..];
        }
        if !step_ok {
            println!(
                "  token {pos}: FAIL count={count} selected={sel:?} expected={want:?} tokens={tokens} expected_tokens={}",
                expect.len()
            );
        }
        ok &= step_ok;
    }
    println!(
        "  {t_total} tokens, top-k selection active in {sparse_steps} steps, worst score error {worst_score:.2e}"
    );
    support::finish(ok && sparse_steps > 0)
}

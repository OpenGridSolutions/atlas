// SPDX-License-Identifier: AGPL-3.0-only
//! Oracle for `csf_expand_fixed` + `csf_expand_exceptions` (Qwen3.8-Flash-Next):
//! csf-compressed NVFP4 block scales expanded into per-expert scratch slots.
//!
//! Without arguments: random scale matrices are compressed on the CPU and the
//! GPU expansion must give back the original bytes (round trip), in both
//! selection modes (id list with a duplicate and a skip; route counts).
//!
//! With `snapshot_dir [layer] [proj]`: the compressed scales of a few experts
//! of one decoder layer are read from the checkpoint and the GPU result must
//! equal the CPU decoder byte for byte.
//!
//! Usage:
//!   cargo run --release -p spark-model --features cuda,gpu-examples \
//!       --example qwen4_csf_expand_microtest -- [snapshot_dir] [layer] [proj]
use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

#[path = "qwen4_support/mod.rs"]
mod support;
use support::{Rng, read_tensor, u32s, upload};

/// One expert's compressed scales.
struct Csf {
    fixed: Vec<u8>,
    exceptions: Vec<u32>,
}

/// CPU decoder: base + code per slab, then the exceptions.
fn decode(csf: &Csf, rows: usize, cols: usize) -> Vec<u8> {
    let slab_len = 16 * (1 + cols / 2);
    let mut out = vec![0u8; rows * cols];
    for r in 0..rows {
        let slab = &csf.fixed[(r / 16) * slab_len..(r / 16 + 1) * slab_len];
        let codes = &slab[16 + (r % 16) * (cols / 2)..];
        for c in 0..cols {
            out[r * cols + c] = slab[r % 16] + ((codes[c / 2] >> (4 * (c & 1))) & 0xF);
        }
    }
    for rec in &csf.exceptions {
        out[(rec & 0xFF_FFFF) as usize] = (rec >> 24) as u8;
    }
    out
}

/// CPU encoder for the round trip: the row base is its smallest byte (capped
/// at 240); bytes outside base..=base+15 become exceptions.
fn encode(scales: &[u8], rows: usize, cols: usize) -> Csf {
    let slab_len = 16 * (1 + cols / 2);
    let mut fixed = vec![0u8; rows / 16 * slab_len];
    let mut exceptions = Vec::new();
    for r in 0..rows {
        let row = &scales[r * cols..(r + 1) * cols];
        let base = (*row.iter().min().unwrap_or(&0)).min(240);
        let slab = &mut fixed[(r / 16) * slab_len..(r / 16 + 1) * slab_len];
        slab[r % 16] = base;
        for (c, &b) in row.iter().enumerate() {
            let code = if (base..=base + 15).contains(&b) {
                b - base
            } else {
                exceptions.push(((b as u32) << 24) | (r * cols + c) as u32);
                (c % 16) as u8 // arbitrary code under an exception
            };
            slab[16 + (r % 16) * (cols / 2) + c / 2] |= code << (4 * (c & 1));
        }
    }
    Csf { fixed, exceptions }
}

fn le<T: Copy, const N: usize>(v: &[T], f: fn(T) -> [u8; N]) -> Vec<u8> {
    v.iter().flat_map(|x| f(*x)).collect()
}

/// Expands `experts` (slot index -> compressed scales) on the GPU for one
/// selection and returns the whole scratch area, pre-filled with 0xEE.
fn run(
    gpu: &dyn GpuBackend,
    experts: &[Csf],
    rows: usize,
    cols: usize,
    ids: Option<&[i32]>,
    route_counts: Option<&[u32]>,
) -> Result<Vec<u8>> {
    let st = gpu.create_stream()?;
    let e = experts.len();
    let mut owned = Vec::new();
    let (mut fixed_ptrs, mut exc_ptrs, mut exc_counts) = (vec![], vec![], vec![]);
    for x in experts {
        let f = upload(gpu, &x.fixed)?;
        let ex = upload(gpu, &le(&x.exceptions, u32::to_le_bytes))?;
        fixed_ptrs.push(f.0);
        exc_ptrs.push(ex.0);
        exc_counts.push(x.exceptions.len() as u32);
        owned.extend([f, ex]);
    }
    let d_fixed = upload(gpu, &le(&fixed_ptrs, u64::to_le_bytes))?;
    let d_exc = upload(gpu, &le(&exc_ptrs, u64::to_le_bytes))?;
    let d_cnt = upload(gpu, &le(&exc_counts, u32::to_le_bytes))?;
    let d_ids = match ids {
        Some(v) => upload(gpu, &le(v, i32::to_le_bytes))?,
        None => DevicePtr(0),
    };
    // The kernels take the counts as prefix sums, i32 [E + 1].
    let d_rc = match route_counts {
        Some(v) => {
            let mut offsets = vec![0i32];
            for c in v {
                offsets.push(offsets[offsets.len() - 1] + *c as i32);
            }
            upload(gpu, &le(&offsets, i32::to_le_bytes))?
        }
        None => DevicePtr(0),
    };
    let d_out = upload(gpu, &vec![0xEEu8; e * rows * cols])?;
    let n = ids.map_or(e, <[i32]>::len) as u32;
    let max_exc = exc_counts.iter().copied().max().unwrap_or(0);
    KernelLaunch::new(gpu, gpu.kernel("csf_expand", "csf_expand_fixed")?)
        .grid([(rows / 16) as u32, n, 1])
        .block([256, 1, 1])
        .arg_ptr(d_fixed)
        .arg_ptr(d_ids)
        .arg_ptr(d_rc)
        .arg_ptr(d_out)
        .arg_u32(rows as u32)
        .arg_u32(cols as u32)
        .launch(st)?;
    if max_exc > 0 {
        KernelLaunch::new(gpu, gpu.kernel("csf_expand", "csf_expand_exceptions")?)
            .grid([max_exc.div_ceil(256), n, 1])
            .block([256, 1, 1])
            .arg_ptr(d_exc)
            .arg_ptr(d_cnt)
            .arg_ptr(d_ids)
            .arg_ptr(d_rc)
            .arg_ptr(d_out)
            .arg_u32(rows as u32)
            .arg_u32(cols as u32)
            .launch(st)?;
    }
    gpu.synchronize(st)?;
    let mut out = vec![0u8; e * rows * cols];
    gpu.copy_d2h(d_out, &mut out)?;
    owned.extend([d_fixed, d_exc, d_cnt, d_out]);
    for p in owned.into_iter().chain([d_ids, d_rc]) {
        if p.0 != 0 {
            gpu.free(p).ok();
        }
    }
    Ok(out)
}

/// Selected slots must equal `want`, every other slot must be untouched.
fn verify(name: &str, got: &[u8], want: &[Vec<u8>], selected: &[bool], size: usize) -> bool {
    let (mut wrong, mut touched) = (0usize, 0usize);
    for (e, sel) in selected.iter().enumerate() {
        let slot = &got[e * size..(e + 1) * size];
        if *sel {
            wrong += slot.iter().zip(&want[e]).filter(|(a, b)| a != b).count();
        } else {
            touched += slot.iter().filter(|b| **b != 0xEE).count();
        }
    }
    let n = selected.iter().filter(|s| **s).count();
    println!(
        "  {name}: {n} expert(s) expanded, wrong bytes={wrong}, bytes written to unselected slots={touched}"
    );
    wrong == 0 && touched == 0 && n > 0
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let be = support::backend()?;
    let gpu: &dyn GpuBackend = &be;
    let mut ok = true;

    if let Some(dir) = a.get(1) {
        let layer: usize = a.get(2).map_or(0, |s| s.parse().unwrap());
        let proj = a.get(3).map_or("gate_proj", String::as_str);
        let picks = [0usize, 7, 255, 511];
        println!(
            "=== csf_expand microtest: checkpoint layer {layer} {proj}, experts {picks:?} ==="
        );
        let (mut experts, mut want, mut dims) = (vec![], vec![], (0usize, 0usize));
        for e in picks {
            let p = format!("model.language_model.layers.{layer}.mlp.experts.{e}.{proj}");
            let (ws, _) = read_tensor(dir, &format!("{p}.weight"), "U8")?;
            let (fs, fixed) = read_tensor(dir, &format!("{p}.weight_scale.nvfp4_csf_fixed"), "U8")?;
            let (_, exc) = read_tensor(
                dir,
                &format!("{p}.weight_scale.nvfp4_csf_exceptions"),
                "U32",
            )?;
            let (rows, cols) = (ws[0], ws[1] / 8); // packed nibbles: 2 per byte, 16 per scale
            if fs != [rows / 16, 16 * (1 + cols / 2)] {
                bail!("{p}: fixed shape {fs:?} does not match weight {ws:?}");
            }
            dims = (rows, cols);
            let csf = Csf {
                fixed,
                exceptions: u32s(&exc),
            };
            want.push(decode(&csf, rows, cols));
            experts.push(csf);
        }
        let (rows, cols) = dims;
        let exc_total: usize = experts.iter().map(|x| x.exceptions.len()).sum();
        println!(
            "  rows={rows} cols={cols} exceptions in the {} experts: {exc_total}",
            picks.len()
        );
        let ids = [2i32, 0, -1, 2];
        let got = run(gpu, &experts, rows, cols, Some(&ids), None)?;
        ok &= verify(
            "id list [2,0,-1,2]",
            &got,
            &want,
            &[true, false, true, false],
            rows * cols,
        );
        let got = run(gpu, &experts, rows, cols, None, Some(&[0, 3, 0, 1]))?;
        ok &= verify(
            "route counts [0,3,0,1]",
            &got,
            &want,
            &[false, true, false, true],
            rows * cols,
        );
    } else {
        println!("=== csf_expand microtest: CPU-compressed random scales (round trip) ===");
        let mut r = Rng(0xC5F);
        for (rows, cols) in [(640usize, 160usize), (2560, 40), (32, 2)] {
            let e = 6;
            let (mut experts, mut want) = (vec![], vec![]);
            for x in 0..e {
                // A narrow band per row plus a share of outliers (none for expert 0).
                let scales: Vec<u8> = (0..rows * cols)
                    .map(|i| {
                        let band = 90 + ((i / cols) % 24) as u8;
                        if x > 0 && r.n() % 97 == 0 {
                            (r.n() % 127) as u8
                        } else {
                            band + (r.n() % 14) as u8
                        }
                    })
                    .collect();
                let csf = encode(&scales, rows, cols);
                if decode(&csf, rows, cols) != scales {
                    bail!("CPU encode/decode round trip failed — the oracle itself is wrong");
                }
                experts.push(csf);
                want.push(scales);
            }
            println!("  rows={rows} cols={cols}");
            let ids = [5i32, 1, -1, 1, 0];
            let got = run(gpu, &experts, rows, cols, Some(&ids), None)?;
            ok &= verify(
                "id list [5,1,-1,1,0]",
                &got,
                &want,
                &[true, true, false, false, false, true],
                rows * cols,
            );
            let got = run(gpu, &experts, rows, cols, None, Some(&[0, 0, 4, 1, 0, 0]))?;
            ok &= verify(
                "route counts",
                &got,
                &want,
                &[false, false, true, true, false, false],
                rows * cols,
            );
        }
    }
    support::finish(ok)
}

// SPDX-License-Identifier: AGPL-3.0-only

//! Arena ATS probe — HOTSWAP_MVP_PLAN.md Task 1.5, the load-bearing spike.
//!
//! Tests the single claim the weights arena rests on: **the GPU can read and
//! write plain anonymous `mmap` memory coherently via NVLink-C2C ATS, with the
//! host VA passed directly as a kernel argument.** Nothing else in the repo
//! exercises this today.
//!
//! Phases:
//!   1. mmap 1 GiB anonymous (page-aligned), split into dst/src halves,
//!      write a known bf16 pattern from the host.
//!   2. `cuPointerGetAttribute` on the raw VA (informational — expected to be
//!      unknown to CUDA before registration).
//!   3. Correctness, unregistered: launch the production `bf16_add_inplace`
//!      kernel with the host VAs as arguments, then verify the result both
//!      ways — direct host reads of the mmap AND `copy_d2h` through CUDA.
//!   4. Bandwidth, unregistered: `cuMemcpyDtoD` of the full 1 GiB into a
//!      device buffer, plus kernel effective bandwidth vs a device-memory
//!      baseline (same kernel on `cuMemAlloc`'d copies).
//!   5. Repeat correctness + bandwidth after `cuMemHostRegister_v2`, then
//!      unregister. Notes any difference.
//!
//! Every phase is best-effort: a failure is recorded and later phases still
//! run (caveat: an ILLEGAL_ADDRESS fault can poison the CUDA context, in
//! which case everything after it fails too — that in itself is the answer).
//! Exit code is 0 only if at least one path proved correct both directions.
//!
//! Run inside the GB10 container (needs the GPU):
//!   cargo run --release -p spark-runtime --example arena_ats_probe

use std::ffi::c_void;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use spark_runtime::cuda_backend::{
    AtlasCudaBackend, cuda_host_register, cuda_host_unregister, cuda_pointer_attribute_u32,
    cuda_pointer_attribute_u64, host_register_flags, pointer_attr,
};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const ARENA_BYTES: usize = 1 << 30; // 1 GiB
const HALF: usize = ARENA_BYTES / 2; // dst region | src region
const N: usize = HALF / 2; // bf16 elements per region (256 Mi)
const BLOCK: usize = 256;
const TIMED_ITERS: usize = 5;

/// bf16 bit pattern of a small integer (exact for v ≤ 255).
fn bf16_bits(v: u32) -> u16 {
    ((v as f32).to_bits() >> 16) as u16
}

fn dst_val(i: usize) -> u32 {
    1 + (i % 97) as u32
}

fn src_val(i: usize) -> u32 {
    1 + (i % 89) as u32
}

fn gib_per_s(bytes: usize, secs: f64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0) / secs
}

struct Outcome {
    label: &'static str,
    passed: bool,
    detail: String,
}

fn write_pattern(dst: *mut u16, src: *mut u16) {
    // Safety: both point into the live 1 GiB mapping, N elements each.
    let d = unsafe { std::slice::from_raw_parts_mut(dst, N) };
    let s = unsafe { std::slice::from_raw_parts_mut(src, N) };
    for i in 0..N {
        d[i] = bf16_bits(dst_val(i));
        s[i] = bf16_bits(src_val(i));
    }
}

/// Launch `bf16_add_inplace(dst, src, n)` and synchronize.
fn run_add(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    stream: u64,
    dst: u64,
    src: u64,
    iters: usize,
) -> Result<()> {
    let mut dst_p = dst;
    let mut src_p = src;
    let mut n_i = N as i32;
    let mut params = [
        &mut dst_p as *mut u64 as *mut c_void,
        &mut src_p as *mut u64 as *mut c_void,
        &mut n_i as *mut i32 as *mut c_void,
    ];
    let grid_x = N.div_ceil(BLOCK) as u32;
    for _ in 0..iters {
        gpu.launch(
            kernel,
            [grid_x, 1, 1],
            [BLOCK as u32, 1, 1],
            0,
            stream,
            &mut params,
        )?;
    }
    gpu.synchronize(stream)
}

/// Verify dst[i] == bf16(dst_val + src_val) by reading the mapping from the
/// host — proves GPU writes are visible to the CPU.
fn verify_host(dst: *const u16) -> Result<()> {
    let d = unsafe { std::slice::from_raw_parts(dst, N) };
    let mut mismatches = 0usize;
    for (i, &got) in d.iter().enumerate() {
        let want = bf16_bits(dst_val(i) + src_val(i));
        if got != want {
            if mismatches < 10 {
                eprintln!("    mismatch at [{i}]: got {got:#06x}, want {want:#06x}");
            }
            mismatches += 1;
        }
    }
    if mismatches > 0 {
        bail!("{mismatches}/{N} elements wrong on host read-back");
    }
    Ok(())
}

/// Verify the same result arrives through `cuMemcpyDtoH` with the host VA
/// treated as a device pointer — proves the memcpy path accepts the VA.
fn verify_d2h(gpu: &dyn GpuBackend, dst_va: u64) -> Result<()> {
    let mut buf = vec![0u8; HALF];
    gpu.copy_d2h(DevicePtr(dst_va), &mut buf)?;
    let direct = unsafe { std::slice::from_raw_parts(dst_va as *const u8, HALF) };
    if buf != direct {
        bail!("copy_d2h bytes differ from direct host reads of the same VA");
    }
    Ok(())
}

/// Correctness pass (pattern → kernel → verify both directions).
fn correctness(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    stream: u64,
    dst_va: u64,
    src_va: u64,
) -> Result<()> {
    write_pattern(dst_va as *mut u16, src_va as *mut u16);
    run_add(gpu, kernel, stream, dst_va, src_va, 1).context("kernel launch on mmap VA")?;
    verify_host(dst_va as *const u16).context("host visibility of kernel writes")?;
    verify_d2h(gpu, dst_va).context("copy_d2h of the mmap VA")?;
    Ok(())
}

/// Kernel effective bandwidth: bf16_add moves 3 × N × 2 bytes per pass
/// (read dst, read src, write dst). Warmup pass first (first-touch / TLB).
fn kernel_bandwidth(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    stream: u64,
    dst: u64,
    src: u64,
) -> Result<(f64, f64)> {
    let t0 = Instant::now();
    run_add(gpu, kernel, stream, dst, src, 1)?;
    let cold = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    run_add(gpu, kernel, stream, dst, src, TIMED_ITERS)?;
    let warm = t1.elapsed().as_secs_f64() / TIMED_ITERS as f64;
    let bytes = 3 * N * 2;
    Ok((gib_per_s(bytes, cold), gib_per_s(bytes, warm)))
}

fn report_pointer_attrs(when: &str, va: u64) {
    println!("  pointer attributes {when} (VA {va:#x}):");
    match cuda_pointer_attribute_u32(pointer_attr::MEMORY_TYPE, va) {
        Ok(t) => println!("    MEMORY_TYPE   = {t} (1=host 2=device 3=array 4=unified)"),
        Err(e) => println!("    MEMORY_TYPE   → {e}"),
    }
    match cuda_pointer_attribute_u64(pointer_attr::DEVICE_POINTER, va) {
        Ok(p) => println!(
            "    DEVICE_POINTER= {p:#x} ({})",
            if p == va { "== host VA" } else { "!= host VA" }
        ),
        Err(e) => println!("    DEVICE_POINTER→ {e}"),
    }
    match cuda_pointer_attribute_u32(pointer_attr::IS_MANAGED, va) {
        Ok(m) => println!("    IS_MANAGED    = {m}"),
        Err(e) => println!("    IS_MANAGED    → {e}"),
    }
}

fn main() -> Result<()> {
    println!("== arena_ats_probe: GPU reads/writes of anonymous mmap (Task 1.5) ==");

    // ── Backend init: same PTX-module path as the server. ──
    let sets = atlas_kernels::available_targets();
    let has_add = |t: &atlas_kernels::TargetPtxSet| t.modules.iter().any(|(n, _)| *n == "bf16_add");
    let set = sets
        .iter()
        .find(|t| t.target.model.contains("qwen3-next") && has_add(t))
        .or_else(|| sets.iter().find(|t| has_add(t)))
        .context("no compiled kernel target contains module 'bf16_add'")?;
    println!(
        "target: {} ({} modules), kernel bf16_add::bf16_add_inplace",
        set.target.model,
        set.modules.len()
    );
    let backend = AtlasCudaBackend::new(0, &set.modules)?;
    let gpu: &dyn GpuBackend = &backend;
    let kernel = gpu.kernel("bf16_add", "bf16_add_inplace")?;
    let stream = gpu.default_stream();

    // ── Phase 1: the arena candidate — 1 GiB anonymous mmap. ──
    let base = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            ARENA_BYTES,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if base == libc::MAP_FAILED {
        bail!(
            "mmap(1 GiB anonymous) failed: {}",
            std::io::Error::last_os_error()
        );
    }
    let base_va = base as u64;
    assert!(
        base_va.is_multiple_of(4096),
        "mmap not 4096-aligned: {base_va:#x} (should be impossible)"
    );
    let huge = unsafe { libc::madvise(base, ARENA_BYTES, libc::MADV_HUGEPAGE) };
    let dst_va = base_va;
    let src_va = base_va + HALF as u64;
    println!(
        "mmap: base {base_va:#x}, 1 GiB, MADV_HUGEPAGE {}",
        if huge == 0 { "accepted" } else { "rejected" }
    );

    let t = Instant::now();
    write_pattern(dst_va as *mut u16, src_va as *mut u16);
    println!(
        "pattern write (host, faults 1 GiB in): {:.2} GiB/s",
        gib_per_s(ARENA_BYTES, t.elapsed().as_secs_f64())
    );

    // ── Phase 2: what does CUDA think of the raw VA? ──
    report_pointer_attrs("before register", dst_va);

    let mut outcomes: Vec<Outcome> = Vec::new();

    // ── Phase 3: correctness, unregistered. THE go/no-go signal. ──
    println!("\n-- phase 3: correctness, unregistered mmap --");
    match correctness(gpu, kernel, stream, dst_va, src_va) {
        Ok(()) => {
            println!("  PASS: kernel + copy_d2h + host visibility all correct");
            outcomes.push(Outcome {
                label: "unregistered correctness",
                passed: true,
                detail: "kernel reads+writes host VA, both directions verified".into(),
            });
        }
        Err(e) => {
            println!("  FAIL: {e:#}");
            println!("  (an ILLEGAL_ADDRESS here may poison the context; later phases may");
            println!("   fail as a consequence — the registered phase is the fallback rung)");
            outcomes.push(Outcome {
                label: "unregistered correctness",
                passed: false,
                detail: format!("{e:#}"),
            });
        }
    }

    // ── Phase 4: bandwidth, unregistered. ──
    println!("\n-- phase 4: bandwidth, unregistered --");
    let dev_full = gpu.alloc(ARENA_BYTES)?;
    // Large sequential read of the arena through the copy engine.
    {
        let t0 = Instant::now();
        gpu.copy_d2d(DevicePtr(base_va), dev_full, ARENA_BYTES)?;
        let cold = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        for _ in 0..TIMED_ITERS {
            gpu.copy_d2d(DevicePtr(base_va), dev_full, ARENA_BYTES)?;
        }
        let warm = t1.elapsed().as_secs_f64() / TIMED_ITERS as f64;
        println!(
            "  memcpy DtoD, mmap → device, 1 GiB: cold {:.2} GiB/s, warm {:.2} GiB/s",
            gib_per_s(ARENA_BYTES, cold),
            gib_per_s(ARENA_BYTES, warm)
        );
    }
    // Device-memory baseline for the same kernel.
    let dev_dst = gpu.alloc(HALF)?;
    let dev_src = gpu.alloc(HALF)?;
    gpu.copy_d2d(DevicePtr(dst_va), dev_dst, HALF)?;
    gpu.copy_d2d(DevicePtr(src_va), dev_src, HALF)?;
    match kernel_bandwidth(gpu, kernel, stream, dev_dst.0, dev_src.0) {
        Ok((cold, warm)) => {
            println!("  kernel on cuMemAlloc buffers (baseline): cold {cold:.2}, warm {warm:.2} GiB/s")
        }
        Err(e) => println!("  kernel baseline FAILED: {e:#}"),
    }
    match kernel_bandwidth(gpu, kernel, stream, dst_va, src_va) {
        Ok((cold, warm)) => {
            println!("  kernel on mmap VAs:                      cold {cold:.2}, warm {warm:.2} GiB/s")
        }
        Err(e) => println!("  kernel on mmap FAILED: {e:#}"),
    }

    // ── Phase 5: registered with cuMemHostRegister_v2. ──
    println!("\n-- phase 5: cuMemHostRegister_v2 --");
    // Safety: `base` is the live 1 GiB mapping created above; it outlives the
    // registration (unregistered at the end of this block).
    let registered = match unsafe { cuda_host_register(base, ARENA_BYTES, 0) } {
        Ok(()) => {
            println!("  registered with flags=0");
            true
        }
        Err(e0) => {
            println!("  flags=0 failed ({e0}); retrying with DEVICEMAP");
            match unsafe { cuda_host_register(base, ARENA_BYTES, host_register_flags::DEVICEMAP) } {
                Ok(()) => {
                    println!("  registered with flags=DEVICEMAP");
                    true
                }
                Err(e1) => {
                    println!("  FAIL: {e1}");
                    outcomes.push(Outcome {
                        label: "registered correctness",
                        passed: false,
                        detail: format!("cuMemHostRegister_v2 refused the mapping: {e1}"),
                    });
                    false
                }
            }
        }
    };
    if registered {
        report_pointer_attrs("after register", dst_va);
        // On ATS the device pointer should equal the host VA; if the driver
        // hands back a different one, use it — that is the finding.
        let dev_view = cuda_pointer_attribute_u64(pointer_attr::DEVICE_POINTER, dst_va)
            .map(|p| (p, p.wrapping_sub(dst_va)))
            .unwrap_or((dst_va, 0));
        let (reg_dst, delta) = dev_view;
        let reg_src = src_va.wrapping_add(delta);
        if delta != 0 {
            println!("  NOTE: device view differs from host VA by {delta:#x} — using it");
        }
        match correctness(gpu, kernel, stream, reg_dst, reg_src) {
            Ok(()) => {
                println!("  PASS: registered correctness, both directions");
                outcomes.push(Outcome {
                    label: "registered correctness",
                    passed: true,
                    detail: format!(
                        "device view {} host VA",
                        if delta == 0 { "==" } else { "!=" }
                    ),
                });
            }
            Err(e) => {
                println!("  FAIL: {e:#}");
                outcomes.push(Outcome {
                    label: "registered correctness",
                    passed: false,
                    detail: format!("{e:#}"),
                });
            }
        }
        match kernel_bandwidth(gpu, kernel, stream, reg_dst, reg_src) {
            Ok((cold, warm)) => {
                println!("  kernel on registered mmap: cold {cold:.2}, warm {warm:.2} GiB/s")
            }
            Err(e) => println!("  kernel on registered mmap FAILED: {e:#}"),
        }
        {
            let t1 = Instant::now();
            for _ in 0..TIMED_ITERS {
                gpu.copy_d2d(DevicePtr(base_va.wrapping_add(delta)), dev_full, ARENA_BYTES)?;
            }
            let warm = t1.elapsed().as_secs_f64() / TIMED_ITERS as f64;
            println!(
                "  memcpy DtoD, registered mmap → device, warm {:.2} GiB/s",
                gib_per_s(ARENA_BYTES, warm)
            );
        }
        // Safety: same base that was registered above, still mapped.
        if let Err(e) = unsafe { cuda_host_unregister(base) } {
            println!("  cuMemHostUnregister failed: {e}");
        }
    }

    // ── Summary. ──
    println!("\n== summary ==");
    for o in &outcomes {
        println!(
            "  {} — {}: {}",
            if o.passed { "PASS" } else { "FAIL" },
            o.label,
            o.detail
        );
    }
    let any_pass = outcomes.iter().any(|o| o.passed);
    if any_pass {
        println!(
            "verdict: GPU access to anonymous mmap PROVEN on at least one path — \
             record details in docs/hotswap/findings.md and proceed to Task 2"
        );
        Ok(())
    } else {
        println!(
            "verdict: ASSUMPTION FAILED — no path let the GPU touch the mmap. \
             Per HOTSWAP_MVP_PLAN.md §Working Agreements: STOP, write \
             docs/hotswap/findings.md, evaluate cuMemAllocManaged → VMM → cuFile"
        );
        bail!("arena ATS assumption not proven")
    }
}

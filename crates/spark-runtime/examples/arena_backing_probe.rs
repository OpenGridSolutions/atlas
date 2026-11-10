// SPDX-License-Identifier: AGPL-3.0-only

//! Arena backing probe — which host-memory backing can carry a fully
//! self-managed, kernel-independent GPU memory block on GB10?
//!
//! `arena_ats_probe` proved the GPU can use a plain anonymous mmap. This
//! probe compares the candidate *backings* for one big pinned block, and
//! adds the write-heavy paths (KV cache, activations) that the weights
//! arena never exercised:
//!
//!   thp / 4k            anonymous, MADV_HUGEPAGE (today's weights arena) or
//!                       MADV_NOHUGEPAGE; the GPU takes the first-touch faults
//!   thp- / 4k-prefault  same, host-prefaulted with MADV_POPULATE_WRITE
//!   thp- / 4k-reg       same, cuMemHostRegister_v2 instead of a prefault
//!   mlock               thp-prefault + mlock (CAP_IPC_LOCK / RLIMIT_MEMLOCK)
//!   hugetlb             MAP_HUGETLB 2 MiB (needs a reserved host pool)
//!
//! Per backing: GPU first-touch write speed, correctness (kernel read+write,
//! both directions verified), kernel bandwidth, pure GPU writes (memset and
//! device→block copies), block→device copies, and what the kernel reports
//! for the range (Rss / AnonHugePages / Locked, process VmLck / VmPin).
//! A `cuMemAlloc` buffer is the baseline. A backing that cannot be set up
//! is reported with its errno and skipped — that is a finding, not an error.
//!
//!   cargo run --release -p spark-runtime --example arena_backing_probe [GiB]

use std::ffi::c_void;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use spark_runtime::cuda_backend::{AtlasCudaBackend, cuda_host_register, cuda_host_unregister};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const BLOCK: usize = 256;
const ITERS: usize = 3;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

struct Mode {
    name: &'static str,
    map_flags: i32,
    /// `Some(true)` = MADV_HUGEPAGE, `Some(false)` = MADV_NOHUGEPAGE.
    thp: Option<bool>,
    /// Host prefault (MADV_POPULATE_WRITE) after the THP advice, so the GPU
    /// never takes a first-touch fault.
    prefault: bool,
    mlock: bool,
    register: bool,
}

const fn mode(name: &'static str, thp: bool, prefault: bool, register: bool) -> Mode {
    Mode {
        name,
        map_flags: 0,
        thp: Some(thp),
        prefault,
        mlock: false,
        register,
    }
}

const MODES: &[Mode] = &[
    mode("thp", true, false, false),
    mode("4k", false, false, false),
    mode("thp-prefault", true, true, false),
    mode("4k-prefault", false, true, false),
    mode("thp-reg", true, false, true),
    mode("4k-reg", false, false, true),
    Mode {
        mlock: true,
        ..mode("mlock", true, true, false)
    },
    Mode {
        map_flags: libc::MAP_HUGETLB | libc::MAP_HUGE_2MB,
        thp: None,
        ..mode("hugetlb", false, false, false)
    },
];

/// Throughput numbers in GiB/s for one buffer (mapping or device baseline).
#[derive(Default)]
struct Speeds {
    first_touch: f64,
    kernel_cold: f64,
    kernel_warm: f64,
    memset: f64,
    copy_out: f64,
    copy_in: f64,
}

struct Ctx<'a> {
    gpu: &'a dyn GpuBackend,
    kernel: KernelHandle,
    stream: u64,
    size: usize,
    /// Device scratch of `size` bytes: copy peer and kernel baseline.
    dev: DevicePtr,
}

impl Ctx<'_> {
    fn half(&self) -> usize {
        self.size / 2
    }
    /// bf16 elements per half.
    fn n(&self) -> usize {
        self.half() / 2
    }
}

fn bf16_bits(v: u32) -> u16 {
    ((v as f32).to_bits() >> 16) as u16
}
fn dst_val(i: usize) -> u32 {
    1 + (i % 97) as u32
}
fn src_val(i: usize) -> u32 {
    1 + (i % 89) as u32
}
fn rate(bytes: usize, secs: f64) -> f64 {
    bytes as f64 / GIB / secs
}

fn run_add(c: &Ctx, dst: u64, src: u64, iters: usize) -> Result<()> {
    let (mut d, mut s, mut n) = (dst, src, c.n() as i32);
    let mut params = [
        &mut d as *mut u64 as *mut c_void,
        &mut s as *mut u64 as *mut c_void,
        &mut n as *mut i32 as *mut c_void,
    ];
    let grid = c.n().div_ceil(BLOCK) as u32;
    for _ in 0..iters {
        c.gpu.launch(
            c.kernel,
            [grid, 1, 1],
            [BLOCK as u32, 1, 1],
            0,
            c.stream,
            &mut params,
        )?;
    }
    c.gpu.synchronize(c.stream)
}

/// Host writes the pattern, the GPU kernel adds in place, the host and
/// `copy_d2h` must both see the sum.
fn correctness(c: &Ctx, base: u64) -> Result<()> {
    let (n, half) = (c.n(), c.half());
    let d = unsafe { std::slice::from_raw_parts_mut(base as *mut u16, n) };
    let s = unsafe { std::slice::from_raw_parts_mut((base + half as u64) as *mut u16, n) };
    for i in 0..n {
        d[i] = bf16_bits(dst_val(i));
        s[i] = bf16_bits(src_val(i));
    }
    run_add(c, base, base + half as u64, 1).context("kernel launch on the mapping")?;
    let wrong = (0..n)
        .filter(|&i| d[i] != bf16_bits(dst_val(i) + src_val(i)))
        .count();
    if wrong > 0 {
        bail!("{wrong}/{n} elements wrong on host read-back");
    }
    let mut buf = vec![0u8; half];
    c.gpu.copy_d2h(DevicePtr(base), &mut buf)?;
    if buf != unsafe { std::slice::from_raw_parts(base as *const u8, half) } {
        bail!("copy_d2h bytes differ from direct host reads");
    }
    Ok(())
}

fn timed(iters: usize, bytes: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    let t = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    Ok(rate(bytes * iters, t.elapsed().as_secs_f64()))
}

/// Measure every path on the buffer at `ptr`; `peer` is the copy partner.
fn measure(c: &Ctx, ptr: u64, peer: DevicePtr, check: bool) -> Result<Speeds> {
    let (gpu, size) = (c.gpu, c.size);
    let memset = || -> Result<()> {
        gpu.memset(DevicePtr(ptr), 0x5A, size)?;
        gpu.device_synchronize()
    };
    let mut s = Speeds {
        first_touch: timed(1, size, memset)?,
        ..Speeds::default()
    };
    if check {
        correctness(c, ptr)?;
    }
    let src = ptr + c.half() as u64;
    // bf16_add moves 3 × N × 2 bytes per pass (read dst, read src, write dst).
    let kbytes = 3 * c.n() * 2;
    s.kernel_cold = timed(1, kbytes, || run_add(c, ptr, src, 1))?;
    s.kernel_warm = timed(1, kbytes * ITERS, || run_add(c, ptr, src, ITERS))?;
    s.memset = timed(ITERS, size, memset)?;
    s.copy_out = timed(ITERS, size, || gpu.copy_d2d(DevicePtr(ptr), peer, size))?;
    s.copy_in = timed(ITERS, size, || gpu.copy_d2d(peer, DevicePtr(ptr), size))?;
    Ok(s)
}

/// Sum a `/proc/self/smaps` field (kB) over every VMA inside the range.
fn smaps_kb(base: u64, size: usize, field: &str) -> u64 {
    let Ok(text) = std::fs::read_to_string("/proc/self/smaps") else {
        return 0;
    };
    let (mut inside, mut total) = (false, 0u64);
    for line in text.lines() {
        let first = line.split_whitespace().next().unwrap_or("");
        if let Some((a, _)) = first.split_once('-')
            && let Ok(start) = u64::from_str_radix(a, 16)
        {
            inside = start >= base && start < base + size as u64;
        } else if inside && first == field {
            total += line
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0);
        }
    }
    total
}

fn status_kb(field: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with(field))
                .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        })
        .unwrap_or(0)
}

fn os_err() -> String {
    std::io::Error::last_os_error().to_string()
}

/// Set up one backing, measure it, tear it down. `Err` = backing unavailable
/// or a GPU path failed; the message is the finding.
fn probe_mode(c: &Ctx, m: &Mode) -> Result<(Speeds, String)> {
    let size = c.size;
    let base = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | m.map_flags,
            -1,
            0,
        )
    };
    if base == libc::MAP_FAILED {
        bail!("mmap failed: {}", os_err());
    }
    let va = base as u64;
    let result = (|| -> Result<(Speeds, String)> {
        if let Some(thp) = m.thp {
            let advice = if thp {
                libc::MADV_HUGEPAGE
            } else {
                libc::MADV_NOHUGEPAGE
            };
            if unsafe { libc::madvise(base, size, advice) } != 0 {
                bail!("madvise(thp advice) failed: {}", os_err());
            }
        }
        if m.prefault && unsafe { libc::madvise(base, size, libc::MADV_POPULATE_WRITE) } != 0 {
            bail!("MADV_POPULATE_WRITE failed: {}", os_err());
        }
        if m.mlock && unsafe { libc::mlock(base, size) } != 0 {
            bail!("mlock failed: {}", os_err());
        }
        if m.register {
            // Safety: live page-aligned mapping, unregistered before munmap.
            unsafe { cuda_host_register(base, size, 0) }
                .map_err(|e| anyhow::anyhow!("cuMemHostRegister_v2 failed: {e}"))?;
        }
        let speeds = measure(c, va, c.dev, true);
        let kernel_view = format!(
            "rss {:.2} GiB, huge {:.2} GiB, locked {:.2} GiB, VmLck {:.2} GiB, VmPin {:.2} GiB",
            smaps_kb(va, size, "Rss:") as f64 / 1048576.0,
            (smaps_kb(va, size, "AnonHugePages:") + smaps_kb(va, size, "Private_Hugetlb:")) as f64
                / 1048576.0,
            smaps_kb(va, size, "Locked:") as f64 / 1048576.0,
            status_kb("VmLck:") as f64 / 1048576.0,
            status_kb("VmPin:") as f64 / 1048576.0,
        );
        if m.register
            && let Err(e) = unsafe { cuda_host_unregister(base) }
        {
            println!("  cuMemHostUnregister failed: {e}");
        }
        Ok((speeds?, kernel_view))
    })();
    unsafe { libc::munmap(base, size) };
    result
}

fn print_row(name: &str, s: &Speeds) {
    println!(
        "  {name:<13} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
        s.first_touch, s.kernel_cold, s.kernel_warm, s.memset, s.copy_out, s.copy_in
    );
}

fn main() -> Result<()> {
    let gib: usize = match std::env::args().nth(1) {
        Some(a) => a.parse().context("size argument must be whole GiB")?,
        None => 4,
    };
    // The kernel takes the element count as i32: half/2 elements per region.
    if gib == 0 || gib > 4 {
        bail!("size must be 1..=4 GiB (bf16_add_inplace takes n as i32)");
    }
    let size = gib << 30;
    println!("== arena_backing_probe: {gib} GiB per backing ==");

    let sets = atlas_kernels::available_targets();
    let set = sets
        .iter()
        .find(|t| t.modules.iter().any(|(n, _)| *n == "bf16_add"))
        .context("no compiled kernel target contains module 'bf16_add'")?;
    let backend = AtlasCudaBackend::new(0, &set.modules)?;
    let gpu: &dyn GpuBackend = &backend;
    let c = Ctx {
        gpu,
        kernel: gpu.kernel("bf16_add", "bf16_add_inplace")?,
        stream: gpu.default_stream(),
        size,
        dev: gpu.alloc(size)?,
    };

    // Baseline: the same paths on driver-owned device memory.
    let peer = gpu.alloc(size)?;
    let baseline = measure(&c, peer.0, c.dev, false).context("cuMemAlloc baseline")?;
    gpu.free(peer)?;

    let mut rows: Vec<(&str, Result<(Speeds, String)>)> = Vec::new();
    for m in MODES {
        println!("-- {} --", m.name);
        let r = probe_mode(&c, m);
        match &r {
            Ok((_, view)) => println!("  PASS  {view}"),
            Err(e) => println!("  UNAVAILABLE/FAIL  {e:#}"),
        }
        rows.push((m.name, r));
    }

    println!("\n== GiB/s ==");
    println!(
        "  {:<13} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "backing", "1st-touch", "k-cold", "k-warm", "memset", "copy-out", "copy-in"
    );
    print_row("cuMemAlloc", &baseline);
    for (name, r) in &rows {
        match r {
            Ok((s, _)) => print_row(name, s),
            Err(e) => println!("  {name:<13} — {e:#}"),
        }
    }
    println!("\n== kernel view ==");
    for (name, r) in &rows {
        if let Ok((_, view)) = r {
            println!("  {name:<13} {view}");
        }
    }
    gpu.free(c.dev)?;
    Ok(())
}

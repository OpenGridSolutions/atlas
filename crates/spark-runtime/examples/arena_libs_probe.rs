// SPDX-License-Identifier: AGPL-3.0-only

//! Arena libs probe — does everything *besides* our own kernels also work on
//! a self-managed host block, and what does the driver still take for itself?
//!
//! `arena_backing_probe` covered kernel launches, memset and device copies on
//! each candidate backing. This probe takes one block (hugetlb, falling back
//! to prefaulted THP) and checks the remaining consumers of device pointers:
//!
//!   cuBLASLt   bf16 GEMM with operands/output and/or the workspace in the
//!              block — each compared bit-for-bit against an all-`cuMemAlloc`
//!              run of the same call, and timed
//!   driver     pageable→block and pinned→block uploads, async block→block
//!              copy, async memset, CUDA graph capture + replay of a kernel
//!              that reads and writes the block
//!
//! It also snapshots MemAvailable / free huge pages / VmRSS / raw
//! `cuMemGetInfo` at each stage, so the memory the driver holds outside the
//! block (context, modules, library handles, freed `cuMemAlloc` buffers) is
//! visible. Not covered: the cuBLASLt FP8 wrappers (no in-tree caller), and
//! CUTLASS / FlashInfer (only linked when their `*_HOME` is set at build time).
//!
//!   cargo run --release -p spark-runtime --example arena_libs_probe [GiB]

use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use spark_runtime::cublaslt;
use spark_runtime::cuda_backend::{AtlasCudaBackend, cuda_mem_get_info_raw};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

// The SSM-qkvz projection shape cublaslt.rs quotes its GB10 numbers for.
const M: usize = 3537;
const N: usize = 12288;
const K: usize = 2048;
const WS_BYTES: usize = 64 << 20;
const ITERS: usize = 20;
/// Bytes per driver-path test buffer.
const PATH_BYTES: usize = 16 << 20;
const KERNEL_BLOCK: usize = 256;

#[derive(Clone, Copy, PartialEq)]
enum Place {
    Device,
    Block,
}

impl Place {
    fn name(self) -> &'static str {
        match self {
            Place::Device => "device",
            Place::Block => "block",
        }
    }
}

fn os_err() -> String {
    std::io::Error::last_os_error().to_string()
}

/// The self-managed block: one mapping, bump-allocated, never returned.
struct Block {
    base: *mut c_void,
    size: usize,
    used: Cell<usize>,
    backing: &'static str,
}

impl Block {
    fn map(size: usize) -> Result<Self> {
        let map = |flags: i32| unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | flags,
                -1,
                0,
            )
        };
        let mut base = map(libc::MAP_HUGETLB | libc::MAP_HUGE_2MB);
        let mut backing = "hugetlb";
        if base == libc::MAP_FAILED {
            println!("  hugetlb unavailable ({}), using prefaulted THP", os_err());
            base = map(0);
            backing = "thp-prefault";
            if base == libc::MAP_FAILED {
                bail!("mmap failed: {}", os_err());
            }
            if unsafe { libc::madvise(base, size, libc::MADV_HUGEPAGE) } != 0 {
                bail!("madvise(MADV_HUGEPAGE) failed: {}", os_err());
            }
        }
        let block = Block {
            base,
            size,
            used: Cell::new(0),
            backing,
        };
        if unsafe { libc::madvise(base, size, libc::MADV_POPULATE_WRITE) } != 0 {
            bail!("MADV_POPULATE_WRITE failed: {}", os_err());
        }
        Ok(block)
    }

    fn take(&self, bytes: usize) -> Result<u64> {
        let start = self.used.get();
        let end = start + bytes.next_multiple_of(4096);
        if end > self.size {
            bail!("block exhausted: {end} > {} bytes", self.size);
        }
        self.used.set(end);
        Ok(self.base as u64 + start as u64)
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.base, self.size) };
    }
}

/// Hands out buffers from the block or from `cuMemAlloc`, and moves bytes in
/// and out. Block buffers are read and written through the host mapping, so
/// the comparison does not depend on the driver copy paths under test.
struct Bufs<'a> {
    gpu: &'a dyn GpuBackend,
    block: &'a Block,
    device: RefCell<Vec<DevicePtr>>,
    device_bytes: Cell<usize>,
}

impl Bufs<'_> {
    fn buf(&self, place: Place, bytes: usize) -> Result<u64> {
        match place {
            Place::Block => self.block.take(bytes),
            Place::Device => {
                let p = self.gpu.alloc(bytes)?;
                self.device.borrow_mut().push(p);
                self.device_bytes.set(self.device_bytes.get() + bytes);
                Ok(p.0)
            }
        }
    }

    fn filled(&self, place: Place, data: &[u8]) -> Result<u64> {
        let p = self.buf(place, data.len())?;
        match place {
            Place::Block => host_mut(p, data.len()).copy_from_slice(data),
            Place::Device => self.gpu.copy_h2d(data, DevicePtr(p))?,
        }
        Ok(p)
    }

    fn read(&self, place: Place, ptr: u64, len: usize) -> Result<Vec<u8>> {
        let mut out = vec![0u8; len];
        match place {
            Place::Block => out.copy_from_slice(host_mut(ptr, len)),
            Place::Device => self.gpu.copy_d2h(DevicePtr(ptr), &mut out)?,
        }
        Ok(out)
    }

    fn free_device(&self) -> Result<()> {
        for p in self.device.borrow_mut().drain(..) {
            self.gpu.free(p)?;
        }
        Ok(())
    }
}

fn host_mut<'a>(ptr: u64, len: usize) -> &'a mut [u8] {
    // Safety: every caller passes a live range inside the block mapping.
    unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, len) }
}

fn bytes_of<T>(data: &[T]) -> &[u8] {
    // Safety: plain-old-data element types only (u16).
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

fn hash(i: u64) -> u64 {
    let mut x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    x ^= x >> 31;
    x.wrapping_mul(0xBF58_476D_1CE4_E5B9)
}

fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// Deterministic bf16 GEMM inputs.
struct Data {
    act: Vec<u16>,
    weight: Vec<u16>,
}

impl Data {
    fn new() -> Self {
        let unit = |i: u64| (hash(i) >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        Data {
            act: (0..M * K).map(|i| bf16(unit(i as u64) * 0.5)).collect(),
            weight: (0..N * K)
                .map(|i| bf16(unit(i as u64 ^ 0x1111) * 0.5))
                .collect(),
        }
    }
}

/// One warm-up call, then `ITERS` timed calls; milliseconds per call.
fn time_calls(gpu: &dyn GpuBackend, f: impl Fn() -> Result<()>) -> Result<f64> {
    f()?;
    gpu.device_synchronize()?;
    let t = Instant::now();
    for _ in 0..ITERS {
        f()?;
    }
    gpu.device_synchronize()?;
    Ok(t.elapsed().as_secs_f64() * 1e3 / ITERS as f64)
}

type CaseResult = Result<(f64, Vec<u8>)>;

fn bf16_case(b: &Bufs, d: &Data, io: Place, ws: Place) -> CaseResult {
    let act = b.filled(io, bytes_of(&d.act))?;
    let weight = b.filled(io, bytes_of(&d.weight))?;
    let out = b.buf(io, M * N * 2)?;
    let workspace = b.buf(ws, WS_BYTES)?;
    let stream = b.gpu.default_stream();
    let (m, n, k) = (M as u32, N as u32, K as u32);
    let ms = time_calls(b.gpu, || {
        cublaslt::bf16_gemm_act_weight_t_ws(act, weight, out, m, n, k, stream, workspace, WS_BYTES)
    })?;
    Ok((ms, b.read(io, out, M * N * 2)?))
}

/// Print one case; returns whether it matched the reference bit-for-bit.
fn report(name: &str, r: &CaseResult, reference: Option<&[u8]>) -> bool {
    let (ms, out) = match r {
        Ok(v) => v,
        Err(e) => {
            println!("  {name:<28} FAIL  {e:#}");
            return false;
        }
    };
    let verdict = match reference {
        None => "reference".to_string(),
        Some(want) if want == out.as_slice() => "bit-identical".to_string(),
        Some(want) => {
            let val = |b: &[u8]| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16);
            let diffs = want.chunks_exact(2).zip(out.chunks_exact(2));
            let (count, max) = diffs.fold((0usize, 0f32), |(c, mx), (w, o)| {
                if w == o {
                    (c, mx)
                } else {
                    (c + 1, mx.max((val(w) - val(o)).abs()))
                }
            });
            format!("DIFFERS in {count} elements, max abs {max:e}")
        }
    };
    println!("  {name:<28} {ms:>7.3} ms  {verdict}");
    reference.is_none() || verdict == "bit-identical"
}

fn expect(ptr: u64, want: &[u8], what: &str) -> Result<()> {
    let wrong = host_mut(ptr, want.len())
        .iter()
        .zip(want)
        .filter(|(a, b)| a != b)
        .count();
    if wrong > 0 {
        bail!("{what}: {wrong}/{} bytes wrong", want.len());
    }
    Ok(())
}

/// Driver entry points that take a device pointer, pointed at the block.
fn driver_paths(b: &Bufs, kernel: KernelHandle) -> Result<Vec<(&'static str, Result<()>)>> {
    let gpu = b.gpu;
    let stream = gpu.create_stream()?;
    let p = b.block.take(PATH_BYTES)?;
    let q = b.block.take(PATH_BYTES)?;
    let pat_a: Vec<u8> = (0..PATH_BYTES).map(|i| hash(i as u64) as u8).collect();
    let pat_b: Vec<u8> = (0..PATH_BYTES)
        .map(|i| hash(i as u64 ^ 0x77) as u8)
        .collect();
    let mut rows: Vec<(&'static str, Result<()>)> = Vec::new();

    rows.push(("pageable -> block (copy_h2d)", {
        gpu.copy_h2d(&pat_a, DevicePtr(p))
            .and_then(|()| expect(p, &pat_a, "upload"))
    }));
    rows.push(("pinned -> block (async)", {
        (|| {
            let pin = gpu.alloc_host_pinned(PATH_BYTES)?;
            let src = host_mut(pin as u64, PATH_BYTES);
            src.copy_from_slice(&pat_b);
            gpu.copy_h2d_async(src, DevicePtr(p), stream)?;
            gpu.synchronize(stream)?;
            gpu.free_host_pinned(pin, PATH_BYTES)?;
            expect(p, &pat_b, "pinned upload")
        })()
    }));
    rows.push(("block -> block (async d2d)", {
        gpu.copy_d2d_async(DevicePtr(p), DevicePtr(q), PATH_BYTES, stream)
            .and_then(|()| gpu.synchronize(stream))
            .and_then(|()| expect(q, &pat_b, "d2d"))
    }));
    rows.push(("memset_async on block", {
        gpu.memset_async(DevicePtr(q), 0xA5, PATH_BYTES, stream)
            .and_then(|()| gpu.synchronize(stream))
            .and_then(|()| expect(q, &vec![0xA5u8; PATH_BYTES], "memset"))
    }));
    rows.push(("graph capture + replay", {
        (|| {
            // bf16 integers: dst = 1 + i%97, src = 1 + i%89, sum stays exact.
            let n = PATH_BYTES / 2;
            let int = |v: usize| ((v as f32).to_bits() >> 16) as u16;
            let dst = unsafe { std::slice::from_raw_parts_mut(p as *mut u16, n) };
            let src = unsafe { std::slice::from_raw_parts_mut(q as *mut u16, n) };
            for i in 0..n {
                dst[i] = int(1 + i % 97);
                src[i] = int(1 + i % 89);
            }
            let (mut d, mut s, mut len) = (p, q, n as i32);
            let mut params = [
                &mut d as *mut u64 as *mut c_void,
                &mut s as *mut u64 as *mut c_void,
                &mut len as *mut i32 as *mut c_void,
            ];
            let grid = [n.div_ceil(KERNEL_BLOCK) as u32, 1, 1];
            gpu.begin_capture(stream)?;
            gpu.launch(
                kernel,
                grid,
                [KERNEL_BLOCK as u32, 1, 1],
                0,
                stream,
                &mut params,
            )?;
            let graph = gpu.end_capture(stream)?;
            gpu.launch_graph(graph, stream)?;
            gpu.synchronize(stream)?;
            gpu.destroy_graph(graph)?;
            let wrong = (0..n)
                .filter(|&i| dst[i] != int(2 + i % 97 + i % 89))
                .count();
            if wrong > 0 {
                bail!("{wrong}/{n} elements wrong after replay");
            }
            Ok(())
        })()
    }));
    Ok(rows)
}

fn proc_kb(path: &str, field: &str) -> u64 {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with(field))
                .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        })
        .unwrap_or(0)
}

struct Snap {
    label: String,
    avail_kb: u64,
    huge_free_kb: u64,
    rss_kb: u64,
    cuda_free: Option<usize>,
}

fn snap(label: impl Into<String>) -> Snap {
    let huge_kb = proc_kb("/proc/meminfo", "Hugepagesize:");
    Snap {
        label: label.into(),
        avail_kb: proc_kb("/proc/meminfo", "MemAvailable:"),
        huge_free_kb: proc_kb("/proc/meminfo", "HugePages_Free:") * huge_kb,
        rss_kb: proc_kb("/proc/self/status", "VmRSS:"),
        cuda_free: cuda_mem_get_info_raw().map(|(free, _)| free),
    }
}

fn print_snaps(snaps: &[Snap]) {
    println!("\n== memory per stage (MiB; MemAvailable is host-wide) ==");
    println!(
        "  {:<44} {:>9} {:>8} {:>9} {:>7} {:>9}",
        "stage", "MemAvail", "delta", "HugeFree", "VmRSS", "cudaFree"
    );
    let mut prev = snaps[0].avail_kb as i64;
    for s in snaps {
        let cuda = s
            .cuda_free
            .map_or("-".to_string(), |b| (b >> 20).to_string());
        println!(
            "  {:<44} {:>9} {:>+8} {:>9} {:>7} {:>9}",
            s.label,
            s.avail_kb / 1024,
            (s.avail_kb as i64 - prev) / 1024,
            s.huge_free_kb / 1024,
            s.rss_kb / 1024,
            cuda
        );
        prev = s.avail_kb as i64;
    }
}

fn main() -> Result<()> {
    let gib: usize = match std::env::args().nth(1) {
        Some(a) => a.parse().context("size argument must be whole GiB")?,
        None => 1,
    };
    if gib == 0 {
        bail!("block size must be at least 1 GiB");
    }
    println!("== arena_libs_probe: {gib} GiB block, GEMM {M}x{N}x{K} ==");
    let mut snaps = vec![snap("process start")];

    let sets = atlas_kernels::available_targets();
    let set = sets
        .iter()
        .find(|t| t.modules.iter().any(|(n, _)| *n == "bf16_add"))
        .context("no compiled kernel target contains module 'bf16_add'")?;
    let backend = AtlasCudaBackend::new(0, &set.modules)?;
    let gpu: &dyn GpuBackend = &backend;
    let kernel = gpu.kernel("bf16_add", "bf16_add_inplace")?;
    snaps.push(snap("CUDA context + kernel modules"));

    let block = Block::map(gib << 30)?;
    println!("  block backing: {}", block.backing);
    snaps.push(snap(format!(
        "block mapped + prefaulted ({})",
        block.backing
    )));

    let data = Data::new();
    let bufs = Bufs {
        gpu,
        block: &block,
        device: RefCell::new(Vec::new()),
        device_bytes: Cell::new(0),
    };
    let mut all_ok = true;

    println!("\n== cuBLASLt bf16 GEMM ==");
    let reference = bf16_case(&bufs, &data, Place::Device, Place::Device);
    report("io=device ws=device", &reference, None);
    snaps.push(snap("device reference + cuBLASLt handle"));
    let reference = reference.context("bf16 device reference")?.1;
    for (io, ws) in [
        (Place::Block, Place::Device),
        (Place::Device, Place::Block),
        (Place::Block, Place::Block),
    ] {
        let name = format!("io={} ws={}", io.name(), ws.name());
        all_ok &= report(&name, &bf16_case(&bufs, &data, io, ws), Some(&reference));
    }

    println!("\n== driver paths on the block ==");
    for (name, r) in driver_paths(&bufs, kernel)? {
        match r {
            Ok(()) => println!("  {name:<28} PASS"),
            Err(e) => {
                all_ok = false;
                println!("  {name:<28} FAIL  {e:#}");
            }
        }
    }

    let device_mib = bufs.device_bytes.get() >> 20;
    snaps.push(snap(format!(
        "all cases done ({device_mib} MiB cuMemAlloc live)"
    )));
    bufs.free_device()?;
    gpu.device_synchronize()?;
    snaps.push(snap("probe's cuMemAlloc buffers freed"));
    print_snaps(&snaps);
    println!(
        "\n  block used: {} of {} MiB; cuBLASLt keeps its own {} MiB device workspace",
        block.used.get() >> 20,
        block.size >> 20,
        WS_BYTES >> 20
    );
    if !all_ok {
        bail!("at least one case failed or differed from the device reference");
    }
    println!("\nALL CASES PASS");
    Ok(())
}

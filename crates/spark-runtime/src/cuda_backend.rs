// SPDX-License-Identifier: AGPL-3.0-only

//! Real CUDA GPU backend using AtlasRegistry.
//!
//! SBIO IORouter: all CUDA operations flow through `GpuBackend`.
//! Uses `AtlasRegistry` for kernel loading/launching and raw CUDA
//! driver API for memory management.

use std::ffi::c_void;

use anyhow::{Result, bail};
use atlas_core::registry::AtlasRegistry;

mod gpu_impl;

// ── Raw CUDA driver API for memory operations ──

unsafe extern "C" {
    pub(super) fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    pub(super) fn cuMemFree_v2(dptr: u64) -> i32;
    pub(super) fn cuMemcpyHtoDAsync_v2(
        dst: u64,
        src: *const c_void,
        bytes: usize,
        stream: u64,
    ) -> i32;
    pub(super) fn cuMemcpyDtoHAsync_v2(
        dst: *mut c_void,
        src: u64,
        bytes: usize,
        stream: u64,
    ) -> i32;
    pub(super) fn cuMemcpyDtoDAsync_v2(dst: u64, src: u64, bytes: usize, stream: u64) -> i32;
    pub(super) fn cuStreamSynchronize(stream: u64) -> i32;
    pub(super) fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> i32;
    pub(super) fn cuMemsetD8Async(dst: u64, value: u8, n: usize, stream: u64) -> i32;
    // CUDA graph capture/replay
    pub(super) fn cuStreamBeginCapture(hStream: u64, mode: u32) -> i32;
    pub(super) fn cuStreamEndCapture(hStream: u64, phGraph: *mut u64) -> i32;
    // CUDA-graph instantiate. NVIDIA's libcuda exports the 3-arg
    // `cuGraphInstantiateWithFlags`; SCALE's libcuda (gfx1151) exports only
    // `cuGraphInstantiate` — same ABI `(CUgraphExec*, CUgraph, u64)`, no
    // `WithFlags` alias. `atlas_scale` (set by build.rs from ATLAS_TARGET_HW)
    // picks the symbol that exists so the binary links on both targets.
    #[cfg(not(atlas_scale))]
    pub(super) fn cuGraphInstantiateWithFlags(
        phGraphExec: *mut u64,
        hGraph: u64,
        flags: u64,
    ) -> i32;
    #[cfg(atlas_scale)]
    pub(super) fn cuGraphInstantiate(phGraphExec: *mut u64, hGraph: u64, flags: u64) -> i32;
    pub(super) fn cuGraphLaunch(hGraphExec: u64, hStream: u64) -> i32;
    pub(super) fn cuGraphExecDestroy(hGraphExec: u64) -> i32;
    pub(super) fn cuGraphDestroy(hGraph: u64) -> i32;
    fn cuCtxGetCurrent(pctx: *mut u64) -> i32;
    pub(super) fn cuCtxSetCurrent(ctx: u64) -> i32;
    pub(super) fn cuStreamCreate(phStream: *mut u64, flags: u32) -> i32;
    // Page-locked host memory for efficient async transfers
    pub(super) fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) -> i32;
    pub(super) fn cuMemFreeHost(p: *mut c_void) -> i32;
    // Managed (unified) memory — allows over-subscription with Linux swap paging
    pub(super) fn cuMemAllocManaged(dptr: *mut u64, bytesize: usize, flags: u32) -> i32;
    // CUDA events for inter-stream synchronization
    pub(super) fn cuEventCreate(phEvent: *mut u64, flags: u32) -> i32;
    pub(super) fn cuEventRecord(hEvent: u64, hStream: u64) -> i32;
    pub(super) fn cuStreamWaitEvent(hStream: u64, hEvent: u64, flags: u32) -> i32;
    pub(super) fn cuEventDestroy_v2(hEvent: u64) -> i32;
    // Host-memory registration + pointer introspection (weights-arena work,
    // HOTSWAP_MVP_PLAN.md Task 1.5). Additive only.
    pub(super) fn cuMemHostRegister_v2(p: *mut c_void, bytesize: usize, flags: u32) -> i32;
    pub(super) fn cuMemHostUnregister(p: *mut c_void) -> i32;
    pub(super) fn cuPointerGetAttribute(data: *mut c_void, attribute: u32, ptr: u64) -> i32;
    // Device-wide barrier for hot-swap (migration, unload).
    pub(super) fn cuCtxSynchronize() -> i32;
}

// ── Host-memory registration + pointer introspection ──
//
// Public wrappers (same pattern as `cuda_free_memory_bytes`) so examples and
// the weights-arena code can reach these without going through `GpuBackend`.
// Errors carry the raw CUDA status: probes treat "CUDA does not know this
// pointer" as data, not failure.

/// `cuPointerGetAttribute` attribute ids (cuda.h `CUpointer_attribute`).
pub mod pointer_attr {
    /// `CU_POINTER_ATTRIBUTE_MEMORY_TYPE` — u32: 1=host, 2=device, 3=array, 4=unified.
    pub const MEMORY_TYPE: u32 = 2;
    /// `CU_POINTER_ATTRIBUTE_DEVICE_POINTER` — u64 device pointer for the allocation.
    pub const DEVICE_POINTER: u32 = 3;
    /// `CU_POINTER_ATTRIBUTE_IS_MANAGED` — u32 boolean.
    pub const IS_MANAGED: u32 = 8;
}

/// `cuMemHostRegister` flags (cuda.h `CU_MEMHOSTREGISTER_*`).
pub mod host_register_flags {
    pub const PORTABLE: u32 = 1;
    pub const DEVICEMAP: u32 = 2;
    pub const READ_ONLY: u32 = 8;
}

/// Page-lock and register an existing host range (e.g. an anonymous mmap) with
/// the CUDA driver via `cuMemHostRegister_v2`.
///
/// # Safety
/// `ptr..ptr+bytes` must be a live, page-aligned host mapping that stays
/// mapped until [`cuda_host_unregister`] is called on the same `ptr`.
pub unsafe fn cuda_host_register(ptr: *mut c_void, bytes: usize, flags: u32) -> Result<()> {
    let status = unsafe { cuMemHostRegister_v2(ptr, bytes, flags) };
    if status != 0 {
        bail!("cuMemHostRegister_v2 failed: status {status}, ptr {ptr:p}, {bytes} bytes, flags {flags:#x}");
    }
    Ok(())
}

/// Undo [`cuda_host_register`].
///
/// # Safety
/// `ptr` must be the exact base previously registered and still mapped.
pub unsafe fn cuda_host_unregister(ptr: *mut c_void) -> Result<()> {
    let status = unsafe { cuMemHostUnregister(ptr) };
    if status != 0 {
        bail!("cuMemHostUnregister failed: status {status}, ptr {ptr:p}");
    }
    Ok(())
}

/// Query a 32-bit `cuPointerGetAttribute` value (see [`pointer_attr`]).
pub fn cuda_pointer_attribute_u32(attribute: u32, ptr: u64) -> Result<u32> {
    let mut data: u32 = 0;
    let status =
        unsafe { cuPointerGetAttribute(&mut data as *mut u32 as *mut c_void, attribute, ptr) };
    if status != 0 {
        bail!("cuPointerGetAttribute({attribute}) failed: status {status}, ptr {ptr:#x}");
    }
    Ok(data)
}

/// Query a 64-bit `cuPointerGetAttribute` value (see [`pointer_attr`]).
pub fn cuda_pointer_attribute_u64(attribute: u32, ptr: u64) -> Result<u64> {
    let mut data: u64 = 0;
    let status =
        unsafe { cuPointerGetAttribute(&mut data as *mut u64 as *mut c_void, attribute, ptr) };
    if status != 0 {
        bail!("cuPointerGetAttribute({attribute}) failed: status {status}, ptr {ptr:#x}");
    }
    Ok(data)
}

/// Production GPU backend wrapping AtlasRegistry + raw CUDA driver API.
///
/// Initialized once via [`AtlasCudaBackend::new`], which loads all PTX
/// modules from `atlas-kernels` into the global AtlasRegistry singleton.
pub struct AtlasCudaBackend {
    /// Default CUDA stream handle (from AtlasRegistry).
    default_stream: u64,
    /// CUDA context handle for cross-thread binding.
    cuda_ctx: u64,
    /// Kernel target this handle resolves modules from: `None` = the boot
    /// target loaded at registry init; `Some(key)` = a target loaded later
    /// via `AtlasRegistry::ensure_target` (different-model hot-swap,
    /// Task 6 Phase 1.3). Kernel lookups happen at model-init time only;
    /// layers store the resolved handles.
    pub(crate) target_key: Option<String>,
}

/// Arena-owned `[base, base+len)` ranges: `free()` on an interior pointer is
/// a debug-logged no-op (see `GpuBackend::protect_range`).
///
/// Process-global rather than per-backend-instance: `AtlasCudaBackend` is a
/// thin handle onto the singleton `AtlasRegistry`/CUDA context, and hot-swap
/// legitimately creates more than one handle (the model owns one, the admin
/// path another). All handles must agree on what is arena-owned.
static PROTECTED_RANGES: std::sync::RwLock<Vec<(u64, u64)>> = std::sync::RwLock::new(Vec::new());

/// Watermark allocation registry (see `GpuBackend::set_alloc_tracking`).
/// Process-global for the same multiple-handles reason as
/// `PROTECTED_RANGES`: allocations recorded through the model's backend
/// handle must be drainable through any other handle at unload.
struct AllocTracker {
    recording: bool,
    live: std::collections::HashSet<u64>,
}

static ALLOC_TRACKER: std::sync::LazyLock<std::sync::Mutex<AllocTracker>> =
    std::sync::LazyLock::new(|| {
        std::sync::Mutex::new(AllocTracker {
            recording: false,
            live: std::collections::HashSet::new(),
        })
    });

pub(super) fn tracker_record(ptr: u64) {
    let mut t = ALLOC_TRACKER.lock().expect("ALLOC_TRACKER poisoned");
    if t.recording {
        t.live.insert(ptr);
    }
}

/// Unrecord on free — unconditionally, so tracked build-time allocations
/// freed later (e.g. MoE transpose originals) never double-free at unload.
pub(super) fn tracker_unrecord(ptr: u64) {
    ALLOC_TRACKER
        .lock()
        .expect("ALLOC_TRACKER poisoned")
        .live
        .remove(&ptr);
}

pub(super) fn tracker_set_recording(on: bool) {
    ALLOC_TRACKER
        .lock()
        .expect("ALLOC_TRACKER poisoned")
        .recording = on;
}

pub(super) fn tracker_drain() -> Vec<u64> {
    ALLOC_TRACKER
        .lock()
        .expect("ALLOC_TRACKER poisoned")
        .live
        .drain()
        .collect()
}

impl AtlasCudaBackend {
    /// Initialize the CUDA backend on the given GPU ordinal.
    ///
    /// Loads the provided PTX modules into AtlasRegistry.
    /// Use `atlas_kernels::ptx_for_model()` or `ptx_modules()` to
    /// obtain the correct module set for the target model.
    /// Subsequent calls reuse the cached singleton.
    pub fn new(ordinal: usize, ptx_modules: &[(&'static str, &'static [u8])]) -> Result<Self> {
        let registry = AtlasRegistry::get_or_init(ordinal, ptx_modules)
            .map_err(|e| anyhow::anyhow!("AtlasRegistry init failed: {e}"))?;
        let default_stream = registry.raw_stream();

        // Capture current CUDA context for cross-thread binding.
        let mut cuda_ctx: u64 = 0;
        let status = unsafe { cuCtxGetCurrent(&mut cuda_ctx) };
        if status != 0 || cuda_ctx == 0 {
            bail!("cuCtxGetCurrent failed: status {status}, ctx {cuda_ctx:#x}");
        }
        STATS_CTX.store(cuda_ctx, std::sync::atomic::Ordering::Relaxed);

        tracing::info!(
            "AtlasCudaBackend initialized on GPU {ordinal} with {} PTX modules",
            ptx_modules.len()
        );

        Ok(Self {
            default_stream,
            cuda_ctx,
            target_key: None,
        })
    }

    /// Backend handle bound to a DIFFERENT kernel target than the boot one
    /// (model hot-swap across model families, Task 6 Phase 1.3). Loads the
    /// target's PTX modules into the registry on first use (never
    /// unloaded); kernel lookups through this handle resolve that target's
    /// modules. Same CUDA context and default stream as every other handle.
    ///
    /// The calling thread must have the CUDA context current (the swap
    /// runs on the scheduler thread, which binds it).
    pub fn new_for_target(
        target_key: &str,
        ptx_modules: &[(&'static str, &'static [u8])],
    ) -> Result<Self> {
        let registry = AtlasRegistry::try_get()
            .ok_or_else(|| anyhow::anyhow!("AtlasRegistry not initialized — boot first"))?;
        let newly = registry
            .ensure_target(target_key, ptx_modules)
            .map_err(|e| anyhow::anyhow!("loading kernel target '{target_key}': {e}"))?;
        if newly {
            tracing::info!(
                "Kernel target '{target_key}' loaded: {} PTX modules (kept for process lifetime)",
                ptx_modules.len()
            );
        }
        let default_stream = registry.raw_stream();
        let mut cuda_ctx: u64 = 0;
        let status = unsafe { cuCtxGetCurrent(&mut cuda_ctx) };
        if status != 0 || cuda_ctx == 0 {
            bail!("cuCtxGetCurrent failed: status {status}, ctx {cuda_ctx:#x}");
        }
        Ok(Self {
            default_stream,
            cuda_ctx,
            target_key: Some(target_key.to_string()),
        })
    }

    /// True if `ptr` lies inside a protected (arena-owned) range.
    pub(super) fn is_protected(&self, ptr: u64) -> bool {
        PROTECTED_RANGES
            .read()
            .expect("PROTECTED_RANGES lock poisoned")
            .iter()
            .any(|&(base, len)| ptr >= base && ptr < base + len)
    }

    pub(super) fn add_protected_range(&self, base: u64, len: u64) {
        PROTECTED_RANGES
            .write()
            .expect("PROTECTED_RANGES lock poisoned")
            .push((base, len));
    }
}

// ── OOM Watchdog ────────────────────────────────────────────────────
//
// Background task that polls GPU free memory every `interval` and calls
// `std::process::exit(1)` if it drops below `threshold_bytes`.
// On GB10 unified memory, GPU OOM = system OOM = kernel freeze, so
// killing the process early prevents unrecoverable system hangs.

/// Query GPU free memory without requiring a GpuBackend reference.
/// Safe to call from any thread that shares the CUDA context.
///
/// Process-wide CUDA context handle for read-only diagnostics from
/// non-CUDA threads (/v1/stats, Plan.md Phase S). Contexts may be
/// current on any number of threads simultaneously; binding it on an
/// HTTP worker never disturbs the scheduler thread's binding. Stored
/// once at backend init (same global pattern as
/// `gpu::baseline_free_bytes`).
static STATS_CTX: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// RAW `cuMemGetInfo` — deliberately WITHOUT the `/proc/meminfo`
/// MemAvailable max that `free_memory()` applies. /v1/stats exposes
/// this uncorrected number next to the meminfo reading so the
/// discrepancy (the DGX Spark unified-memory known issue) is visible
/// as a metric. Callable from any thread: binds the stored context on
/// demand when the caller has none. Returns `(free, total)` bytes.
pub fn cuda_mem_get_info_raw() -> Option<(usize, usize)> {
    let mut cur: u64 = 0;
    unsafe { cuCtxGetCurrent(&mut cur) };
    if cur == 0 {
        let ctx = STATS_CTX.load(std::sync::atomic::Ordering::Relaxed);
        if ctx == 0 {
            return None;
        }
        if unsafe { cuCtxSetCurrent(ctx) } != 0 {
            return None;
        }
    }
    let mut free: usize = 0;
    let mut total: usize = 0;
    if unsafe { cuMemGetInfo_v2(&mut free, &mut total) } != 0 {
        return None;
    }
    Some((free, total))
}

/// On unified memory systems (GB10), `cuMemGetInfo` reports Linux's "free" memory
/// which excludes reclaimable buff/cache. This under-reports available memory by
/// 30-50%. We take the max of CUDA's report and `/proc/meminfo` MemAvailable
/// to get the true available memory.
pub fn cuda_free_memory_bytes() -> Option<usize> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    let status = unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
    if status != 0 {
        return None;
    }

    // On unified memory, also check MemAvailable from /proc/meminfo.
    // This includes reclaimable buff/cache that CUDA doesn't account for.
    if let Some(mem_available) = system_available_memory_bytes() {
        free = free.max(mem_available);
    }
    Some(free)
}

/// Read MemAvailable from /proc/meminfo (Linux only).
/// Returns None on non-Linux or if parsing fails.
fn system_available_memory_bytes() -> Option<usize> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in contents.lines() {
        if line.starts_with("MemAvailable:") {
            let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// Start a background OOM watchdog that polls GPU memory every `interval`.
/// If free memory drops below `threshold_mb` MB, the process exits immediately.
///
/// Returns a `tokio::task::JoinHandle` — drop it to stop the watchdog (on shutdown).
pub fn spawn_oom_watchdog(
    threshold_mb: usize,
    interval: std::time::Duration,
) -> tokio::task::JoinHandle<()> {
    let threshold_bytes = threshold_mb * 1024 * 1024;
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        // Track consecutive low-memory readings to avoid false positives
        // during transient allocation spikes.
        let mut consecutive_low = 0u32;
        loop {
            tick.tick().await;
            if let Some(free) = cuda_free_memory_bytes() {
                if free < threshold_bytes {
                    consecutive_low += 1;
                    let free_mb = free / (1024 * 1024);
                    tracing::error!(
                        "OOM watchdog: GPU free memory critically low: {} MB (threshold: {} MB) [{}/3]",
                        free_mb,
                        threshold_mb,
                        consecutive_low,
                    );
                    if consecutive_low >= 3 {
                        tracing::error!(
                            "OOM watchdog: 3 consecutive readings below threshold. \
                             Terminating to prevent system freeze."
                        );
                        // Flush logs before exit
                        std::process::exit(1);
                    }
                } else {
                    consecutive_low = 0;
                }
            }
        }
    })
}

#[cfg(test)]
mod tests;

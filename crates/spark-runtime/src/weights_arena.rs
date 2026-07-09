// SPDX-License-Identifier: AGPL-3.0-only

//! Process-lifetime weights arena (HOTSWAP_MVP_PLAN.md Task 2).
//!
//! One anonymous `mmap` holds every weight tensor as `base + offset` views,
//! so model unload/reload never returns weight memory to the kernel and
//! `arena_base` stays constant across load/unload cycles. On GB10 the GPU
//! reads the mapping coherently via NVLink-C2C ATS with the raw host VA as
//! a kernel argument — proven by `examples/arena_ats_probe.rs` (Task 1.5:
//! correctness PASS unregistered; `cudaHostRegister` optional, ~2% gain).
//!
//! Invariants (plan §2):
//! - The arena is never unmapped or shrunk during process lifetime. There
//!   is deliberately no `Drop`. `reset()` only rewinds the bump cursor.
//! - Allocations are 256-byte aligned minimum: kernels do vectorised loads
//!   and nothing asserts alignment at use sites, so misalignment would be
//!   silent wrong results (trace §4.2).
//! - Arena count is a parameter of the design, not a singleton: every
//!   instance is self-contained (dual-residency / drafter arenas later).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, bail};

use crate::gpu::{DevicePtr, GpuBackend};
use crate::weights::{WeightDtype, WeightStore};

/// One tensor's placement inside the arena. Also the snapshot manifest
/// entry (ATLASRAW): field meanings must stay stable within a
/// `format_version`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaTensorEntry {
    pub name: String,
    /// Byte offset from `arena_base`.
    pub offset: u64,
    /// Payload length in bytes.
    pub len: u64,
    /// `WeightDtype::to_code()`.
    pub dtype: u8,
    pub shape: Vec<u64>,
}

/// Stats returned by [`migrate_weight_store`], for the boot log.
#[derive(Debug)]
pub struct MigrationStats {
    pub tensors: usize,
    pub bytes: usize,
    pub arena_used: usize,
    pub duration_ms: u128,
    pub freed_ptrs: usize,
}

/// A single mmap-backed bump arena.
pub struct WeightsArena {
    base: *mut u8,
    capacity: usize,
    used: AtomicUsize,
    host_registered: bool,
}

// Safety: `base` points at a process-lifetime anonymous mapping that is never
// unmapped or moved; the bump cursor is atomic. Concurrent readers of arena
// bytes are the same hazard class as sharing any `DevicePtr`, which the
// codebase already does — writers are confined to migration/reload, which run
// single-threaded while the scheduler holds no model.
unsafe impl Send for WeightsArena {}
unsafe impl Sync for WeightsArena {}

/// Minimum allocation alignment (matches `cuMemAlloc_v2`'s guarantee, which
/// is what every kernel implicitly assumed before the arena).
pub const ARENA_ALIGN: usize = 256;

impl WeightsArena {
    /// Map `size_bytes` (rounded up to 4 KiB) of anonymous memory.
    ///
    /// `host_register`: optionally page-lock + register the mapping with the
    /// CUDA driver (`--arena-host-register`). Registration failure is a
    /// warning, not an error — the ATS probe proved unregistered access
    /// works; registration was ~2% on reads.
    pub fn new(size_bytes: usize, host_register: bool) -> Result<Self> {
        if size_bytes == 0 {
            bail!("weights arena size must be > 0 (use the CLI flag to disable the arena instead)");
        }
        let capacity = size_bytes.div_ceil(4096) * 4096;
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                capacity,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            bail!(
                "weights arena mmap({:.1} GiB) failed: {}",
                capacity as f64 / (1024.0 * 1024.0 * 1024.0),
                std::io::Error::last_os_error()
            );
        }
        let huge = unsafe { libc::madvise(base, capacity, libc::MADV_HUGEPAGE) };
        if huge != 0 {
            tracing::debug!("weights arena: MADV_HUGEPAGE rejected (non-fatal)");
        }

        let mut host_registered = false;
        if host_register {
            #[cfg(feature = "cuda")]
            {
                // Safety: the mapping above is live and page-aligned, and by
                // the arena invariant it is never unmapped.
                match unsafe { crate::cuda_backend::cuda_host_register(base, capacity, 0) } {
                    Ok(()) => host_registered = true,
                    Err(e) => tracing::warn!(
                        "weights arena: cuMemHostRegister_v2 failed ({e}); continuing \
                         unregistered (ATS access works either way on GB10)"
                    ),
                }
            }
            #[cfg(not(feature = "cuda"))]
            tracing::warn!("weights arena: arena_host_register ignored (no CUDA backend)");
        }

        Ok(Self {
            base: base as *mut u8,
            capacity,
            used: AtomicUsize::new(0),
            host_registered,
        })
    }

    /// Bump-allocate `len` bytes at `align` (power of two, >= [`ARENA_ALIGN`]
    /// for weight tensors). Returns the byte offset from `base`.
    pub fn alloc(&self, len: usize, align: usize) -> Result<u64> {
        debug_assert!(align.is_power_of_two(), "arena align must be a power of two");
        let offset = self
            .used
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                let aligned = cur.div_ceil(align) * align;
                let end = aligned.checked_add(len)?;
                (end <= self.capacity).then_some(end)
            })
            .map_err(|cur| {
                anyhow::anyhow!(
                    "weights arena exhausted: need {len} B at align {align}, \
                     used {cur} of {} B — increase --weights-arena-gb",
                    self.capacity
                )
            })?
            .div_ceil(align)
            * align;
        debug_assert!(offset.is_multiple_of(align), "arena offset misaligned");
        Ok(offset as u64)
    }

    /// Mark the whole arena reusable. Never unmaps — `base` and the mapping
    /// survive for the process lifetime (plan §2 invariant).
    pub fn reset(&self) {
        self.used.store(0, Ordering::Relaxed);
    }

    /// True if `ptr` points anywhere inside the arena mapping (capacity, not
    /// just the used prefix: stale views after `reset()` must still be
    /// recognised as arena-owned by the free-guard).
    pub fn owns(&self, ptr: DevicePtr) -> bool {
        let base = self.base as u64;
        ptr.0 >= base && ptr.0 < base + self.capacity as u64
    }

    pub fn base(&self) -> DevicePtr {
        DevicePtr(self.base as u64)
    }

    pub fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn host_registered(&self) -> bool {
        self.host_registered
    }

    /// Register the arena range with the backend's free-guard so
    /// `gpu.free(interior_ptr)` becomes a no-op (trace §4.1: 36 call sites
    /// free store-owned pointers).
    pub fn protect_on(&self, gpu: &dyn GpuBackend) {
        gpu.protect_range(self.base as u64, self.capacity);
    }

    /// The used prefix of the arena as a byte slice (snapshot payload).
    ///
    /// # Safety
    /// The caller must ensure no concurrent writer (migration or reload) is
    /// active. Weights are read-only during inference, so snapshotting a
    /// serving model is safe.
    pub unsafe fn used_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.base, self.used()) }
    }

    /// Boot-time functional ATS assertion (plan Task 2 acceptance).
    ///
    /// A *functional* mini-test — GPU read of the arena (`cuMemcpyDtoD` out)
    /// and GPU write into it (`cuMemsetD8`), both verified from the host.
    /// Deliberately NOT `cuPointerGetAttribute`: the Task 1.5 probe showed
    /// it fails with status 1 on an unregistered mapping even though access
    /// works.
    ///
    /// Uses the first 4 KiB, so it must run while the arena is empty
    /// (before migration or right after `reset()`).
    pub fn ats_self_test(&self, gpu: &dyn GpuBackend) -> Result<()> {
        const PROBE: usize = 4096;
        assert!(self.used() == 0, "ats_self_test needs an empty arena");

        // Host writes a pattern the GPU must be able to read back out.
        let region = unsafe { std::slice::from_raw_parts_mut(self.base, PROBE) };
        for (i, b) in region.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        let dev = gpu.alloc(PROBE)?;
        let readback = (|| -> Result<()> {
            gpu.copy_d2d(self.base(), dev, PROBE)
                .context("GPU read of the arena (cuMemcpyDtoD from host VA)")?;
            let mut host = vec![0u8; PROBE];
            gpu.copy_d2h(dev, &mut host)?;
            for (i, (&got, &want)) in host.iter().zip(region.iter()).enumerate() {
                if got != want {
                    bail!("GPU read of arena wrong at byte {i}: got {got:#04x}, want {want:#04x}");
                }
            }
            // GPU write into the arena, host-visible.
            gpu.memset(self.base(), 0xA5, PROBE)
                .context("GPU write into the arena (cuMemsetD8 on host VA)")?;
            gpu.device_synchronize()?;
            let region = unsafe { std::slice::from_raw_parts(self.base, PROBE) };
            if let Some(i) = region.iter().position(|&b| b != 0xA5) {
                bail!(
                    "GPU write to arena not visible on host at byte {i}: {:#04x}",
                    region[i]
                );
            }
            Ok(())
        })();
        let _ = gpu.free(dev);
        // Leave the probed region zeroed either way.
        unsafe { std::slice::from_raw_parts_mut(self.base, PROBE) }.fill(0);
        readback.context(
            "weights-arena ATS self-test FAILED — the GPU cannot access the anonymous \
             mmap on this platform; see docs/hotswap/findings.md fallback ladder",
        )
    }
}

/// Migrate every tensor of `store` into `arena` in **name-sorted order**
/// (deterministic layout — HashMap iteration is not reproducible), re-point
/// the store's `DevicePtr`s at `arena_base + offset` views, then free the
/// loader's original per-tensor allocations.
///
/// Returns the manifest (one entry per tensor, in layout order) and stats.
pub fn migrate_weight_store(
    gpu: &dyn GpuBackend,
    arena: &WeightsArena,
    store: &mut WeightStore,
) -> Result<(Vec<ArenaTensorEntry>, MigrationStats)> {
    let t0 = Instant::now();
    let names = store.sorted_names();
    let stream = gpu.default_stream();

    let mut entries = Vec::with_capacity(names.len());
    let mut old_ptrs = Vec::with_capacity(names.len());
    let mut bytes = 0usize;

    for (i, name) in names.iter().enumerate() {
        let tensor = store.get(name)?;
        let len = tensor.byte_size();
        let offset = arena
            .alloc(len, ARENA_ALIGN)
            .with_context(|| format!("migrating '{name}' ({len} B)"))?;
        let dst = arena.base().offset(offset as usize);
        gpu.copy_d2d_async(tensor.ptr, dst, len, stream)
            .with_context(|| format!("copying '{name}' into the arena"))?;
        entries.push(ArenaTensorEntry {
            name: name.clone(),
            offset,
            len: len as u64,
            dtype: tensor.dtype.to_code(),
            shape: tensor.shape.iter().map(|&d| d as u64).collect(),
        });
        old_ptrs.push(tensor.ptr);
        bytes += len;

        // Re-point the store at the arena view.
        let t = store
            .get_mut(name)
            .expect("tensor vanished during migration");
        t.ptr = dst;

        // Bound the async queue depth; ~300k outstanding memcpys would
        // otherwise stall inside random driver calls.
        if i % 8192 == 8191 {
            gpu.synchronize(stream)?;
        }
    }

    // All copies must complete before the sources are freed.
    gpu.device_synchronize()?;
    for ptr in &old_ptrs {
        gpu.free(*ptr)
            .with_context(|| format!("freeing pre-migration allocation {ptr}"))?;
    }
    gpu.device_synchronize()?;

    Ok((
        entries,
        MigrationStats {
            tensors: names.len(),
            bytes,
            arena_used: arena.used(),
            duration_ms: t0.elapsed().as_millis(),
            freed_ptrs: old_ptrs.len(),
        },
    ))
}

/// Rebuild a [`WeightStore`] from a manifest against the (already filled)
/// arena — the reload path: views only, no allocation, no copies.
pub fn store_from_manifest(
    arena: &WeightsArena,
    entries: &[ArenaTensorEntry],
) -> Result<WeightStore> {
    let mut map = std::collections::HashMap::with_capacity(entries.len());
    for e in entries {
        let end = e.offset + e.len;
        if end > arena.capacity() as u64 {
            bail!(
                "manifest entry '{}' [{}, {end}) exceeds arena capacity {}",
                e.name,
                e.offset,
                arena.capacity()
            );
        }
        map.insert(
            e.name.clone(),
            crate::weights::WeightTensor {
                ptr: arena.base().offset(e.offset as usize),
                shape: e.shape.iter().map(|&d| d as usize).collect(),
                dtype: WeightDtype::from_code(e.dtype)?,
            },
        );
    }
    Ok(WeightStore::from_map(map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::mock::MockGpuBackend;
    use crate::weights::WeightTensor;

    fn small_arena() -> WeightsArena {
        WeightsArena::new(1 << 20, false).unwrap()
    }

    #[test]
    fn alloc_is_aligned_and_bumps() {
        let a = small_arena();
        let o1 = a.alloc(100, 256).unwrap();
        let o2 = a.alloc(1, 256).unwrap();
        let o3 = a.alloc(300, 256).unwrap();
        assert_eq!(o1, 0);
        assert_eq!(o2, 256);
        assert_eq!(o3, 512);
        assert_eq!(a.used(), 512 + 300);
    }

    #[test]
    fn alloc_fails_when_exhausted_and_reset_recovers() {
        let a = small_arena();
        assert!(a.alloc((1 << 20) - 100, 256).is_ok());
        assert!(a.alloc(4096, 256).is_err());
        a.reset();
        assert_eq!(a.used(), 0);
        assert!(a.alloc(4096, 256).is_ok());
    }

    #[test]
    fn owns_covers_capacity_not_used() {
        let a = small_arena();
        assert!(a.owns(a.base()));
        assert!(a.owns(a.base().offset((1 << 20) - 1)));
        assert!(!a.owns(a.base().offset(1 << 20)));
        assert!(!a.owns(DevicePtr(a.base().0 - 1)));
    }

    #[test]
    fn migration_repoints_sorted_frees_and_manifests() {
        let gpu = MockGpuBackend::new();
        let mut map = std::collections::HashMap::new();
        for (name, len) in [("b.w", 300usize), ("a.w", 100), ("c.w", 512)] {
            let ptr = gpu.alloc(len).unwrap();
            map.insert(
                name.to_string(),
                WeightTensor {
                    ptr,
                    shape: vec![len / 2],
                    dtype: WeightDtype::BF16,
                },
            );
        }
        let mut store = WeightStore::from_map(map);
        assert_eq!(gpu.alloc_count(), 3);

        let arena = small_arena();
        let (entries, stats) = migrate_weight_store(&gpu, &arena, &mut store).unwrap();

        // Name-sorted layout, 256-aligned offsets.
        assert_eq!(
            entries.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(),
            vec!["a.w", "b.w", "c.w"]
        );
        assert_eq!(entries[0].offset, 0);
        assert_eq!(entries[1].offset, 256);
        assert_eq!(entries[2].offset, 256 + 512);
        // Store re-pointed at arena views.
        for e in &entries {
            let t = store.get(&e.name).unwrap();
            assert_eq!(t.ptr.0, arena.base().0 + e.offset);
            assert!(arena.owns(t.ptr));
        }
        // Originals freed.
        assert_eq!(gpu.alloc_count(), 0);
        assert_eq!(stats.tensors, 3);
        assert_eq!(stats.bytes, 100 + 300 + 512);

        // Rebuild from manifest matches.
        let rebuilt = store_from_manifest(&arena, &entries).unwrap();
        for e in &entries {
            assert_eq!(rebuilt.get(&e.name).unwrap().ptr.0, arena.base().0 + e.offset);
        }
    }

    #[test]
    fn free_guard_suppresses_arena_interior_frees() {
        let gpu = MockGpuBackend::new();
        let arena = small_arena();
        arena.protect_on(&gpu);
        let real = gpu.alloc(64).unwrap();
        // Freeing an arena interior pointer is a no-op, real allocs still free.
        gpu.free(arena.base().offset(300)).unwrap();
        gpu.free(real).unwrap();
        assert_eq!(gpu.alloc_count(), 0);
    }
}

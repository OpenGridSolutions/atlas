// SPDX-License-Identifier: AGPL-3.0-only
//! Mock GPU backend for unit tests (no GPU required).

use super::*;
use parking_lot::Mutex;
use std::collections::HashMap;

#[derive(Debug)]
pub struct MockAlloc {
    pub bytes: usize,
    pub data: Vec<u8>,
}

/// Records kernel launches and memory operations for test assertions.
pub struct MockGpuBackend {
    allocs: Mutex<HashMap<u64, MockAlloc>>,
    next_ptr: Mutex<u64>,
    launches: Mutex<Vec<MockLaunch>>,
    /// Per-instance (unlike the CUDA backend's process-global list) so
    /// parallel tests stay isolated.
    protected: Mutex<Vec<(u64, u64)>>,
    /// Watermark alloc registry, mirroring the CUDA backend (per-instance).
    tracking: Mutex<(bool, std::collections::HashSet<u64>)>,
    /// Every `copy_d2d`/`copy_d2d_async` as `(src, dst, bytes, live_allocs
    /// at issue time)` — the live count lets tests prove that streaming
    /// migration frees sources between shards, not only at the end.
    d2d_copies: Mutex<Vec<(u64, u64, usize, usize)>>,
}

#[derive(Debug, Clone)]
pub struct MockLaunch {
    pub func: u64,
    pub grid: [u32; 3],
    pub block: [u32; 3],
}

impl Default for MockGpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MockGpuBackend {
    pub fn new() -> Self {
        Self {
            allocs: Mutex::new(HashMap::new()),
            next_ptr: Mutex::new(0x1000_0000),
            launches: Mutex::new(Vec::new()),
            protected: Mutex::new(Vec::new()),
            tracking: Mutex::new((false, std::collections::HashSet::new())),
            d2d_copies: Mutex::new(Vec::new()),
        }
    }

    pub fn alloc_count(&self) -> usize {
        self.allocs.lock().len()
    }

    pub fn launch_count(&self) -> usize {
        self.launches.lock().len()
    }

    pub fn read_alloc(&self, ptr: DevicePtr) -> Option<Vec<u8>> {
        self.allocs.lock().get(&ptr.0).map(|a| a.data.clone())
    }

    /// `(src, dst, bytes, live_allocs)` per device-to-device copy issued.
    pub fn d2d_copy_log(&self) -> Vec<(u64, u64, usize, usize)> {
        self.d2d_copies.lock().clone()
    }
}

/// Find the allocation containing `ptr` (supports offset pointers).
fn find_alloc(allocs: &HashMap<u64, MockAlloc>, ptr: DevicePtr) -> Option<(usize, &MockAlloc)> {
    for (&base, alloc) in allocs.iter() {
        if ptr.0 >= base && ptr.0 < base + alloc.bytes as u64 {
            return Some(((ptr.0 - base) as usize, alloc));
        }
    }
    None
}

/// Mutable version of find_alloc.
fn find_alloc_mut(
    allocs: &mut HashMap<u64, MockAlloc>,
    ptr: DevicePtr,
) -> Option<(usize, &mut MockAlloc)> {
    for (&base, alloc) in allocs.iter_mut() {
        if ptr.0 >= base && ptr.0 < base + alloc.bytes as u64 {
            return Some(((ptr.0 - base) as usize, alloc));
        }
    }
    None
}

impl GpuBackend for MockGpuBackend {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        let mut next = self.next_ptr.lock();
        let ptr = *next;
        *next += bytes as u64;
        // Align to 256 bytes
        *next = (*next + 255) & !255;
        self.allocs.lock().insert(
            ptr,
            MockAlloc {
                bytes,
                data: vec![0u8; bytes],
            },
        );
        {
            let mut t = self.tracking.lock();
            if t.0 {
                t.1.insert(ptr);
            }
        }
        Ok(DevicePtr(ptr))
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.alloc(bytes) // Mock: same as regular alloc
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        // Mirror the CUDA backend's arena free-guard semantics.
        if self
            .protected
            .lock()
            .iter()
            .any(|&(base, len)| ptr.0 >= base && ptr.0 < base + len)
        {
            return Ok(());
        }
        self.tracking.lock().1.remove(&ptr.0);
        self.allocs.lock().remove(&ptr.0);
        Ok(())
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        let mut allocs = self.allocs.lock();
        // Support offset pointers: find the allocation containing dst
        let (offset, alloc) = find_alloc_mut(&mut allocs, dst)
            .ok_or_else(|| anyhow::anyhow!("copy_h2d: ptr {dst} not allocated"))?;
        alloc.data[offset..offset + src.len()].copy_from_slice(src);
        Ok(())
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        let allocs = self.allocs.lock();
        // Support offset pointers: find the allocation containing src
        let (offset, alloc) = find_alloc(&allocs, src)
            .ok_or_else(|| anyhow::anyhow!("copy_d2h: ptr {src} not allocated"))?;
        dst.copy_from_slice(&alloc.data[offset..offset + dst.len()]);
        Ok(())
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        let live = self.allocs.lock().len();
        self.d2d_copies.lock().push((src.0, dst.0, bytes, live));
        Ok(())
    }

    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        _shared_mem: u32,
        _stream: u64,
        _params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        self.launches.lock().push(MockLaunch {
            func: func.0,
            grid,
            block,
        });
        Ok(())
    }

    fn synchronize(&self, _stream: u64) -> Result<()> {
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        0
    }

    fn kernel(&self, _module: &str, _func_name: &str) -> Result<KernelHandle> {
        Ok(KernelHandle(0xDEAD))
    }

    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        let mut allocs = self.allocs.lock();
        let (offset, alloc) = find_alloc_mut(&mut allocs, ptr)
            .ok_or_else(|| anyhow::anyhow!("memset: ptr {ptr} not allocated"))?;
        alloc.data[offset..offset + bytes].fill(value);
        Ok(())
    }

    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, _stream: u64) -> Result<()> {
        self.memset(ptr, value, bytes)
    }

    fn total_memory(&self) -> Result<usize> {
        Ok(128 * 1024 * 1024 * 1024) // 128 GB
    }

    fn free_memory(&self) -> Result<usize> {
        Ok(120 * 1024 * 1024 * 1024) // 120 GB
    }

    fn protect_range(&self, base: u64, len: usize) {
        self.protected.lock().push((base, len as u64));
    }

    fn set_alloc_tracking(&self, on: bool) {
        self.tracking.lock().0 = on;
    }

    fn take_tracked_allocs(&self) -> Vec<DevicePtr> {
        self.tracking.lock().1.drain().map(DevicePtr).collect()
    }
}

// SPDX-License-Identifier: AGPL-3.0-only

//! Watermark allocation registry (see `GpuBackend::set_alloc_tracking`)
//! and the byte accounting behind `GpuBackend::alloc_stats`.
//!
//! Every device allocation of the process falls into one of three bins:
//!
//! * **build** — made while recording is on (a model build). Unload frees
//!   these through [`tracker_drain`].
//! * **sidecar** — made inside a sidecar scope; owned by `sidecar_ledger`.
//! * **late** — everything else: made while no build is recording. The KV
//!   pool is sized during the build from what is allocated up to then, so
//!   a late allocation is memory the budget never saw, and nothing frees it
//!   at unload. The ledger makes that visible instead of leaving it to a
//!   free-memory reading taken at the right moment.
//!
//! Process-global for the same multiple-handles reason as
//! `PROTECTED_RANGES`: allocations recorded through the model's backend
//! handle must be drainable through any other handle at unload.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use super::{UNTRACKED_ALLOCS, sidecar_ledger};
use crate::gpu::AllocStats;

#[derive(Default)]
struct AllocTracker {
    recording: bool,
    /// Live build allocations: device pointer → bytes.
    build: HashMap<u64, usize>,
    build_bytes: usize,
    /// Live late allocations: device pointer → bytes.
    late: HashMap<u64, usize>,
    late_bytes: usize,
    /// High-water mark of `late_bytes` since the last build finished.
    late_peak_bytes: usize,
}

impl AllocTracker {
    fn record(&mut self, ptr: u64, bytes: usize) {
        if self.recording {
            if let Some(prev) = self.build.insert(ptr, bytes) {
                self.build_bytes -= prev;
            }
            self.build_bytes += bytes;
        } else {
            self.record_late(ptr, bytes);
        }
    }

    fn record_late(&mut self, ptr: u64, bytes: usize) {
        if let Some(prev) = self.late.insert(ptr, bytes) {
            self.late_bytes -= prev;
        }
        self.late_bytes += bytes;
        self.late_peak_bytes = self.late_peak_bytes.max(self.late_bytes);
    }

    fn unrecord(&mut self, ptr: u64) {
        if let Some(bytes) = self.build.remove(&ptr) {
            self.build_bytes -= bytes;
        }
        if let Some(bytes) = self.late.remove(&ptr) {
            self.late_bytes -= bytes;
        }
    }

    /// Turning recording off ends a build and opens a new serving window:
    /// the peak restarts at what is live right now, so late allocations a
    /// previous model left behind stay visible as the new model's floor.
    fn set_recording(&mut self, on: bool) {
        self.recording = on;
        if !on {
            self.late_peak_bytes = self.late_bytes;
        }
    }

    fn drain(&mut self) -> Vec<u64> {
        self.build_bytes = 0;
        self.build.drain().map(|(ptr, _)| ptr).collect()
    }

    fn stats(&self) -> AllocStats {
        AllocStats {
            build_live_bytes: self.build_bytes,
            late_live_bytes: self.late_bytes,
            late_live_count: self.late.len(),
            late_peak_bytes: self.late_peak_bytes,
        }
    }
}

static ALLOC_TRACKER: LazyLock<Mutex<AllocTracker>> =
    LazyLock::new(|| Mutex::new(AllocTracker::default()));

fn tracker() -> std::sync::MutexGuard<'static, AllocTracker> {
    ALLOC_TRACKER.lock().expect("ALLOC_TRACKER poisoned")
}

pub(super) fn tracker_record(ptr: u64, bytes: usize) {
    if UNTRACKED_ALLOCS.with(|c| c.get()) {
        sidecar_ledger::record(ptr);
        return;
    }
    tracker().record(ptr, bytes);
}

/// Account a process-lifetime device allocation made outside
/// `GpuBackend::alloc` (the library workspaces: cuBLASLt, CUTLASS,
/// FlashInfer). Always late — its owner is a process-global, so it must
/// never land in a build's drain list even when first used during one.
pub(crate) fn record_process_alloc(ptr: u64, bytes: usize) {
    tracker().record_late(ptr, bytes);
}

/// Unrecord on free — unconditionally, so tracked build-time allocations
/// freed later (e.g. MoE transpose originals) never double-free at unload.
pub(super) fn tracker_unrecord(ptr: u64) {
    sidecar_ledger::unrecord(ptr);
    tracker().unrecord(ptr);
}

pub(super) fn tracker_set_recording(on: bool) {
    tracker().set_recording(on);
}

pub(super) fn tracker_drain() -> Vec<u64> {
    tracker().drain()
}

/// Byte accounting of the process's device allocations (handle-less twin
/// of `GpuBackend::alloc_stats`, for callers that hold no backend).
pub fn alloc_stats() -> AllocStats {
    tracker().stats()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1024 * 1024;

    fn building() -> AllocTracker {
        let mut t = AllocTracker::default();
        t.set_recording(true);
        t
    }

    #[test]
    fn allocations_during_a_build_count_as_build_bytes() {
        let mut t = building();
        t.record(0x1000, 3 * MIB);
        t.record(0x2000, 5 * MIB);
        let s = t.stats();
        assert_eq!(s.build_live_bytes, 8 * MIB);
        assert_eq!(s.late_live_bytes, 0);
        assert_eq!(s.late_peak_bytes, 0);
    }

    #[test]
    fn allocations_after_the_build_are_late() {
        let mut t = building();
        t.record(0x1000, 3 * MIB);
        t.set_recording(false);
        t.record(0x2000, 160 * MIB);
        t.record(0x3000, 40 * MIB);
        let s = t.stats();
        assert_eq!(s.build_live_bytes, 3 * MIB);
        assert_eq!(s.late_live_bytes, 200 * MIB);
        assert_eq!(s.late_live_count, 2);
        assert_eq!(s.late_peak_bytes, 200 * MIB);
    }

    #[test]
    fn a_freed_scratch_buffer_leaves_only_its_peak() {
        let mut t = AllocTracker::default();
        t.record(0x2000, 64 * MIB);
        t.unrecord(0x2000);
        t.record(0x2000, 16 * MIB); // the driver hands the address out again
        let s = t.stats();
        assert_eq!(s.late_live_bytes, 16 * MIB);
        assert_eq!(s.late_live_count, 1);
        assert_eq!(s.late_peak_bytes, 64 * MIB);
    }

    #[test]
    fn a_new_build_restarts_the_peak_at_what_is_still_live() {
        let mut t = AllocTracker::default();
        t.record(0x2000, 100 * MIB); // kept by the first model, never freed
        t.record(0x3000, 50 * MIB);
        t.unrecord(0x3000);
        assert_eq!(t.stats().late_peak_bytes, 150 * MIB);
        // Unload + next build.
        t.drain();
        t.set_recording(true);
        t.record(0x9000, 7 * MIB);
        t.set_recording(false);
        let s = t.stats();
        assert_eq!(s.late_live_bytes, 100 * MIB, "the leftover is still live");
        assert_eq!(s.late_peak_bytes, 100 * MIB, "peak restarts at the floor");
        assert_eq!(s.build_live_bytes, 7 * MIB);
    }

    #[test]
    fn drain_hands_out_build_pointers_only() {
        let mut t = building();
        t.record(0x1000, 3 * MIB);
        t.record(0x2000, 5 * MIB);
        t.unrecord(0x1000); // freed by the build itself
        t.set_recording(false);
        t.record(0x3000, 9 * MIB);
        assert_eq!(t.drain(), vec![0x2000]);
        let s = t.stats();
        assert_eq!(s.build_live_bytes, 0);
        assert_eq!(s.late_live_bytes, 9 * MIB);
        assert!(t.drain().is_empty());
    }

    #[test]
    fn a_process_lifetime_allocation_is_late_even_during_a_build() {
        let mut t = building();
        t.record(0x1000, 3 * MIB);
        t.record_late(0x5000, 64 * MIB); // library workspace, first used now
        assert_eq!(t.drain(), vec![0x1000]);
        let s = t.stats();
        assert_eq!(s.late_live_bytes, 64 * MIB);
        assert_eq!(s.late_live_count, 1);
    }

    #[test]
    fn freeing_an_unknown_pointer_changes_nothing() {
        let mut t = building();
        t.record(0x1000, 3 * MIB);
        t.unrecord(0xDEAD);
        assert_eq!(t.stats().build_live_bytes, 3 * MIB);
    }
}

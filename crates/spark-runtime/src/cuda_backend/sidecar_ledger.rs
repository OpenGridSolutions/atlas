// SPDX-License-Identifier: AGPL-3.0-only

//! Per-owner ledger of sidecar device allocations.
//!
//! A sidecar (Whisper, Qwen-Image) lives outside the hot-swap arena, and
//! its model structs free at most part of their buffers on drop. Only one
//! model is in memory at a time, so unloading a sidecar has to hand back
//! ALL of its memory: every allocation made inside a [`SidecarAllocs`]
//! scope is recorded under the scope's owner, and [`free_sidecar_allocs`]
//! releases whatever is still live. `free()` unrecords unconditionally, so
//! a buffer the sidecar already freed itself is never freed twice.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use super::{UntrackedAllocs, cuMemFree_v2};

/// Live sidecar allocations: device pointer → owner.
static LEDGER: LazyLock<Mutex<HashMap<u64, &'static str>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

thread_local! {
    /// Owner of the sidecar scope this thread is in, if any.
    static OWNER: Cell<Option<&'static str>> = const { Cell::new(None) };
}

/// RAII scope: device allocations made on this thread belong to the
/// sidecar `owner`. They stay out of the hot-swap watermark tracker (the
/// scope is an [`UntrackedAllocs`] scope too) and are released by
/// [`free_sidecar_allocs`] when the sidecar is unloaded.
pub struct SidecarAllocs {
    prev: Option<&'static str>,
    _untracked: UntrackedAllocs,
}

impl SidecarAllocs {
    pub fn enter(owner: &'static str) -> Self {
        Self {
            prev: OWNER.with(|c| c.replace(Some(owner))),
            _untracked: UntrackedAllocs::enter(),
        }
    }
}

impl Drop for SidecarAllocs {
    fn drop(&mut self) {
        OWNER.with(|c| c.set(self.prev));
    }
}

/// Record `ptr` for the current thread's sidecar scope (no-op outside one).
pub(super) fn record(ptr: u64) {
    if let Some(owner) = OWNER.with(|c| c.get()) {
        LEDGER
            .lock()
            .expect("sidecar ledger poisoned")
            .insert(ptr, owner);
    }
}

pub(super) fn unrecord(ptr: u64) {
    LEDGER.lock().expect("sidecar ledger poisoned").remove(&ptr);
}

/// Remove and return every live allocation of `owner`.
fn take(owner: &str) -> Vec<u64> {
    let mut ledger = LEDGER.lock().expect("sidecar ledger poisoned");
    let ptrs: Vec<u64> = ledger
        .iter()
        .filter(|(_, o)| **o == owner)
        .map(|(ptr, _)| *ptr)
        .collect();
    for ptr in &ptrs {
        ledger.remove(ptr);
    }
    ptrs
}

/// `cuMemFree` every allocation `owner` still holds. Returns
/// `(freed, failed)`. Call after the sidecar's model structs are dropped,
/// on a thread with the CUDA context bound.
pub fn free_sidecar_allocs(owner: &str) -> (usize, usize) {
    let mut freed = 0usize;
    let mut failed = 0usize;
    for ptr in take(owner) {
        let status = unsafe { cuMemFree_v2(ptr) };
        if status == 0 {
            freed += 1;
        } else {
            failed += 1;
            tracing::error!(
                "free_sidecar_allocs({owner}): cuMemFree_v2({ptr:#x}) failed: status {status}"
            );
        }
    }
    (freed, failed)
}

#[cfg(test)]
mod tests {
    use super::super::{tracker_record, tracker_unrecord};
    use super::*;

    fn sorted(mut ptrs: Vec<u64>) -> Vec<u64> {
        ptrs.sort_unstable();
        ptrs
    }

    #[test]
    fn allocations_inside_a_scope_belong_to_its_owner() {
        tracker_record(0xA000); // before the scope: nobody's
        {
            let _scope = SidecarAllocs::enter("test-owner-a");
            tracker_record(0xA100);
            tracker_record(0xA200);
        }
        tracker_record(0xA300); // after the scope: nobody's
        assert_eq!(sorted(take("test-owner-a")), vec![0xA100, 0xA200]);
        assert!(take("test-owner-a").is_empty(), "take drains the owner");
    }

    #[test]
    fn a_freed_allocation_is_never_handed_out_again() {
        {
            let _scope = SidecarAllocs::enter("test-owner-b");
            tracker_record(0xB100);
            tracker_record(0xB200);
        }
        // The sidecar frees one buffer itself, outside any scope.
        tracker_unrecord(0xB100);
        assert_eq!(take("test-owner-b"), vec![0xB200]);
    }

    #[test]
    fn owners_are_kept_apart_and_nested_scopes_restore() {
        let _outer = SidecarAllocs::enter("test-owner-c");
        tracker_record(0xC100);
        {
            let _inner = SidecarAllocs::enter("test-owner-d");
            tracker_record(0xD100);
        }
        tracker_record(0xC200);
        assert_eq!(take("test-owner-d"), vec![0xD100]);
        assert_eq!(sorted(take("test-owner-c")), vec![0xC100, 0xC200]);
    }

    #[test]
    fn a_scope_on_another_thread_does_not_leak_into_this_one() {
        let _scope = SidecarAllocs::enter("test-owner-e");
        std::thread::spawn(|| tracker_record(0xE100))
            .join()
            .unwrap();
        assert!(take("test-owner-e").is_empty());
    }
}

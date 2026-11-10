// SPDX-License-Identifier: AGPL-3.0-only

//! Residency of the sidecar models (Whisper, Qwen-Image).
//!
//! Only one model is in memory at a time. A sidecar is loaded after the
//! arena model was unloaded (`api::sidecar_gate`) and is unloaded again
//! before the next arena model loads (the scheduler's `Load` arm). A
//! [`SidecarSlot`] is the resettable home of one sidecar: a request holds
//! a [`Lease`] for as long as it uses the model, and an unload waits for
//! every lease to be returned before it frees the model.

use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use spark_runtime::cuda_backend::{SidecarAllocs, free_sidecar_allocs};
use tokio::sync::{OwnedRwLockReadGuard, RwLock};

pub(crate) struct SidecarSlot<T> {
    cell: Arc<RwLock<Option<T>>>,
    /// Model id while resident — readable without the async lock.
    resident: parking_lot::Mutex<Option<String>>,
    in_flight: Arc<AtomicUsize>,
    /// Frees an unloaded sidecar's memory; runs on a blocking thread.
    release: fn(T),
}

/// A request's hold on the resident sidecar. Dereferences to it.
pub(crate) struct Lease<T> {
    guard: OwnedRwLockReadGuard<Option<T>, T>,
    in_flight: Arc<AtomicUsize>,
}

impl<T> Deref for Lease<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T> Drop for Lease<T> {
    fn drop(&mut self) {
        self.in_flight.fetch_sub(1, Ordering::AcqRel);
    }
}

impl<T: Send + Sync + 'static> SidecarSlot<T> {
    pub(crate) fn new(release: fn(T)) -> Self {
        Self {
            cell: Arc::new(RwLock::new(None)),
            resident: parking_lot::Mutex::new(None),
            in_flight: Arc::new(AtomicUsize::new(0)),
            release,
        }
    }

    /// Lease the resident sidecar; `None` when the slot is empty (never
    /// loaded, or unloaded for another model in the meantime).
    pub(crate) async fn lease(&self) -> Option<Lease<T>> {
        let guard = Arc::clone(&self.cell).read_owned().await;
        let guard = OwnedRwLockReadGuard::try_map(guard, Option::as_ref).ok()?;
        self.in_flight.fetch_add(1, Ordering::AcqRel);
        Some(Lease {
            guard,
            in_flight: Arc::clone(&self.in_flight),
        })
    }

    /// Fill an empty slot with what `load` returns; a filled slot is left
    /// alone. `load` runs under the slot's lock, so the sidecar loads once.
    pub(crate) async fn install<F, Fut>(&self, model_id: &str, load: F) -> anyhow::Result<()>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = anyhow::Result<T>>,
    {
        let mut cell = self.cell.write().await;
        if cell.is_none() {
            *cell = Some(load().await?);
            *self.resident.lock() = Some(model_id.to_string());
        }
        Ok(())
    }
}

/// What the one-model-at-a-time bookkeeping needs from a slot.
pub(crate) trait Residency: Sync {
    /// Model id of the resident sidecar.
    fn resident_model(&self) -> Option<String>;
    /// Requests holding a lease right now.
    fn in_flight(&self) -> usize;
    /// Wait for every lease, then free the sidecar. `true` when one was
    /// resident. Blocks: scheduler thread or `spawn_blocking` only.
    fn unload_blocking(&self) -> bool;
}

impl<T: Send + Sync + 'static> Residency for SidecarSlot<T> {
    fn resident_model(&self) -> Option<String> {
        self.resident.lock().clone()
    }

    fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Acquire)
    }

    fn unload_blocking(&self) -> bool {
        let taken = {
            let mut cell = self.cell.blocking_write();
            *self.resident.lock() = None;
            cell.take()
        };
        match taken {
            Some(sidecar) => {
                (self.release)(sidecar);
                true
            }
            None => false,
        }
    }
}

fn slots() -> [&'static dyn Residency; 2] {
    [
        crate::api::audio::residency(),
        crate::api::images::residency(),
    ]
}

/// Run a sidecar's blocking `load` with its device allocations recorded
/// for `owner`. A failed load frees what it allocated — nothing may stay
/// behind beside the next model.
pub(crate) async fn load_blocking<T: Send + 'static>(
    owner: &'static str,
    load: impl FnOnce() -> anyhow::Result<T> + Send + 'static,
) -> anyhow::Result<T> {
    tokio::task::spawn_blocking(move || {
        let _allocs = SidecarAllocs::enter(owner);
        let loaded = load();
        if loaded.is_err() {
            let (freed, failed) = free_sidecar_allocs(owner);
            tracing::warn!(
                "{owner} sidecar load failed: {freed} device buffers freed, {failed} failed"
            );
        }
        loaded
    })
    .await?
}

/// Free every device buffer `owner` still holds, after its model structs
/// were dropped. Needs the CUDA context bound on this thread.
pub(crate) fn free_allocs(owner: &'static str) {
    let (freed, failed) = free_sidecar_allocs(owner);
    tracing::info!("{owner} sidecar unloaded: {freed} device buffers freed, {failed} failed");
}

/// Model id of the resident sidecar, if one is in memory.
pub(crate) fn resident_model() -> Option<String> {
    slots().iter().find_map(|s| s.resident_model())
}

/// Requests a resident sidecar is serving right now.
pub(crate) fn in_flight() -> usize {
    slots().iter().map(|s| s.in_flight()).sum()
}

/// Unload every resident sidecar; returns how many were. Blocks until
/// their in-flight requests are done: scheduler thread or `spawn_blocking`
/// only.
pub(crate) fn unload_all_blocking() -> usize {
    slots().iter().filter(|s| s.unload_blocking()).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    static RELEASED: AtomicUsize = AtomicUsize::new(0);

    fn count_release(_: u32) {
        RELEASED.fetch_add(1, Ordering::AcqRel);
    }

    fn drop_release(_: u32) {}

    async fn install(slot: &SidecarSlot<u32>, id: &str, value: u32) {
        slot.install(id, || async move { Ok(value) }).await.unwrap();
    }

    async fn unload(slot: &Arc<SidecarSlot<u32>>) -> bool {
        let slot = Arc::clone(slot);
        tokio::task::spawn_blocking(move || slot.unload_blocking())
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn an_empty_slot_has_nothing_to_lease_or_unload() {
        let slot = Arc::new(SidecarSlot::<u32>::new(drop_release));
        assert!(slot.lease().await.is_none());
        assert_eq!(slot.resident_model(), None);
        assert!(!unload(&slot).await);
    }

    #[tokio::test]
    async fn a_loaded_sidecar_is_leased_and_counted() {
        let slot = SidecarSlot::<u32>::new(drop_release);
        install(&slot, "model-a", 7).await;
        assert_eq!(slot.resident_model().as_deref(), Some("model-a"));
        let first = slot.lease().await.unwrap();
        let second = slot.lease().await.unwrap();
        assert_eq!((*first, *second), (7, 7));
        assert_eq!(slot.in_flight(), 2);
        drop(first);
        drop(second);
        assert_eq!(slot.in_flight(), 0);
    }

    #[tokio::test]
    async fn install_leaves_a_resident_sidecar_alone() {
        let slot = SidecarSlot::<u32>::new(drop_release);
        install(&slot, "model-a", 1).await;
        slot.install("model-b", || async { panic!("must not load twice") })
            .await
            .unwrap();
        assert_eq!(slot.resident_model().as_deref(), Some("model-a"));
        assert_eq!(*slot.lease().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn a_failed_load_leaves_the_slot_empty() {
        let slot = SidecarSlot::<u32>::new(drop_release);
        let err = slot
            .install("model-a", || async { anyhow::bail!("no memory") })
            .await
            .unwrap_err();
        assert_eq!(err.to_string(), "no memory");
        assert_eq!(slot.resident_model(), None);
        assert!(slot.lease().await.is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn unload_waits_for_the_last_lease_then_releases_once() {
        let slot = Arc::new(SidecarSlot::<u32>::new(count_release));
        install(&slot, "model-a", 1).await;
        let lease = slot.lease().await.unwrap();

        let unloading = tokio::spawn({
            let slot = Arc::clone(&slot);
            async move { unload(&slot).await }
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!unloading.is_finished(), "unload must wait for the lease");
        assert_eq!(RELEASED.load(Ordering::Acquire), 0);

        drop(lease);
        assert!(unloading.await.unwrap());
        assert_eq!(RELEASED.load(Ordering::Acquire), 1);
        assert_eq!(slot.resident_model(), None);
        assert!(slot.lease().await.is_none());

        // The slot is reusable: the next request loads the model again.
        install(&slot, "model-a", 2).await;
        assert_eq!(*slot.lease().await.unwrap(), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_lease_survives_its_request_being_dropped() {
        // The handler moves the lease into its blocking GPU work, so a
        // client disconnect cannot free the model under a running kernel.
        let slot = Arc::new(SidecarSlot::<u32>::new(drop_release));
        install(&slot, "model-a", 1).await;
        let lease = Arc::new(slot.lease().await.unwrap());
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (finish_tx, finish_rx) = std::sync::mpsc::channel::<()>();
        let work = tokio::task::spawn_blocking({
            let lease = Arc::clone(&lease);
            move || {
                started_tx.send(()).unwrap();
                finish_rx.recv().unwrap();
                **lease
            }
        });
        started_rx.await.unwrap();
        drop(lease); // the request future is gone

        let unloading = tokio::spawn({
            let slot = Arc::clone(&slot);
            async move { unload(&slot).await }
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            !unloading.is_finished(),
            "the GPU work still holds the lease"
        );
        finish_tx.send(()).unwrap();
        assert_eq!(work.await.unwrap(), 1);
        assert!(unloading.await.unwrap());
    }
}

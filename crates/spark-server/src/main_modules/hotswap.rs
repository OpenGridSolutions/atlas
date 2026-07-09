// SPDX-License-Identifier: AGPL-3.0-only

//! Hot-swap context shared between the admin endpoints and the scheduler
//! (HOTSWAP_MVP_PLAN.md Tasks 3–5). Holds the process-lifetime weights
//! arena, its manifest, and the swap state machine.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use anyhow::{Context, Result};
use spark_runtime::snapshot::{
    SnapshotWriteStats, model_version_hash, snapshot_file_name, write_snapshot,
};
use spark_runtime::weights_arena::{ArenaTensorEntry, WeightsArena};

/// Server-side model lifecycle state. Reported by `GET /admin/models` and
/// gates the inference endpoints (503 while not `Running`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SwapState {
    Running = 0,
    Draining = 1,
    NoModel = 2,
    Loading = 3,
}

impl SwapState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Draining => "draining",
            Self::NoModel => "unloaded",
            Self::Loading => "loading",
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Draining,
            2 => Self::NoModel,
            3 => Self::Loading,
            _ => Self::Running,
        }
    }
}

/// Identity of the model currently occupying the arena. Swapped as one
/// unit by the reloader when a different model is installed (Task 6 / 1.3).
#[derive(Clone)]
pub struct ActiveModel {
    /// HF id of the model occupying the arena.
    pub model_id: String,
    /// Identity of (model id, revision, arena layout version).
    pub model_hash: [u8; 32],
}

/// Shared hot-swap state. Constructed at boot when the weights arena is
/// enabled; `None` in `AppState` means `--weights-arena-gb 0` and every
/// /admin/models endpoint answers 503.
pub struct HotswapCtx {
    pub arena: Arc<WeightsArena>,
    /// Arena layout recorded by the boot migration or the last snapshot
    /// load (name-sorted order).
    pub manifest: parking_lot::RwLock<Arc<Vec<ArenaTensorEntry>>>,
    /// Model currently occupying the arena; replaced on a different-model
    /// swap together with `manifest`.
    pub active: parking_lot::RwLock<ActiveModel>,
    pub snapshot_dir: PathBuf,
    state: AtomicU8,
    /// Serializes admin operations (snapshot/unload/load) against each other.
    pub admin_lock: tokio::sync::Mutex<()>,
    /// Pushes control commands into (and wakes) the scheduler loop.
    pub scheduler: crate::scheduler::SchedulerHandle,
    /// The router-level readiness gate (also read by /health).
    pub model_ready: Arc<std::sync::atomic::AtomicBool>,
}

impl HotswapCtx {
    pub fn new(
        arena: Arc<WeightsArena>,
        manifest: Vec<ArenaTensorEntry>,
        model_id: String,
        model_revision: &str,
        snapshot_dir: PathBuf,
        scheduler: crate::scheduler::SchedulerHandle,
        model_ready: Arc<std::sync::atomic::AtomicBool>,
    ) -> Self {
        let model_hash = model_version_hash(&model_id, model_revision);
        Self {
            arena,
            manifest: parking_lot::RwLock::new(Arc::new(manifest)),
            active: parking_lot::RwLock::new(ActiveModel {
                model_id,
                model_hash,
            }),
            snapshot_dir,
            state: AtomicU8::new(SwapState::Running as u8),
            admin_lock: tokio::sync::Mutex::new(()),
            scheduler,
            model_ready,
        }
    }

    pub fn state(&self) -> SwapState {
        SwapState::from_u8(self.state.load(Ordering::Acquire))
    }

    pub fn set_state(&self, s: SwapState) {
        self.state.store(s as u8, Ordering::Release);
    }

    /// Snapshot of the active model's identity (one consistent read).
    pub fn active_model(&self) -> ActiveModel {
        self.active.read().clone()
    }

    /// Default on-disk location for the ACTIVE model's snapshot.
    pub fn snapshot_path(&self) -> PathBuf {
        let active = self.active_model();
        self.snapshot_path_for(&active.model_id, &active.model_hash)
    }

    /// On-disk snapshot location for an arbitrary (model, hash).
    pub fn snapshot_path_for(&self, model_id: &str, model_hash: &[u8; 32]) -> PathBuf {
        self.snapshot_dir
            .join(snapshot_file_name(model_id, model_hash))
    }

    /// Blocking snapshot write (call from `spawn_blocking`). Requires the
    /// admin lock to be held by the caller; safe while serving (weights are
    /// read-only during inference).
    pub fn snapshot_blocking(&self) -> Result<(PathBuf, SnapshotWriteStats)> {
        let active = self.active_model();
        let path = self.snapshot_path_for(&active.model_id, &active.model_hash);
        let manifest = self.manifest.read().clone();
        let stats = write_snapshot(&path, &self.arena, &manifest, active.model_hash)
            .with_context(|| format!("writing snapshot {}", path.display()))?;
        tracing::info!(
            "Snapshot written: {} ({:.2} GiB in {} ms, {:.2} GiB/s, xxh3 {:#018x}, O_DIRECT {})",
            path.display(),
            stats.bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            stats.duration_ms,
            stats.bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                / (stats.duration_ms.max(1) as f64 / 1000.0),
            stats.payload_xxh3,
            stats.o_direct,
        );
        Ok((path, stats))
    }
}

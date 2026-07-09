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

/// Shared hot-swap state. Constructed at boot when the weights arena is
/// enabled; `None` in `AppState` means `--weights-arena-gb 0` and every
/// /admin/models endpoint answers 503.
pub struct HotswapCtx {
    pub arena: Arc<WeightsArena>,
    /// Arena layout recorded by the boot migration (name-sorted order).
    /// Same-model scope: fixed for the process lifetime.
    pub manifest: parking_lot::RwLock<Arc<Vec<ArenaTensorEntry>>>,
    /// HF id of the model occupying the arena.
    pub model_id: String,
    /// Identity of (model id, revision, arena layout version).
    pub model_hash: [u8; 32],
    pub snapshot_dir: PathBuf,
    state: AtomicU8,
    /// Serializes admin operations (snapshot/unload/load) against each other.
    pub admin_lock: tokio::sync::Mutex<()>,
}

impl HotswapCtx {
    pub fn new(
        arena: Arc<WeightsArena>,
        manifest: Vec<ArenaTensorEntry>,
        model_id: String,
        model_revision: &str,
        snapshot_dir: PathBuf,
    ) -> Self {
        let model_hash = model_version_hash(&model_id, model_revision);
        Self {
            arena,
            manifest: parking_lot::RwLock::new(Arc::new(manifest)),
            model_id,
            model_hash,
            snapshot_dir,
            state: AtomicU8::new(SwapState::Running as u8),
            admin_lock: tokio::sync::Mutex::new(()),
        }
    }

    pub fn state(&self) -> SwapState {
        SwapState::from_u8(self.state.load(Ordering::Acquire))
    }

    pub fn set_state(&self, s: SwapState) {
        self.state.store(s as u8, Ordering::Release);
    }

    /// Default on-disk location for this model's snapshot.
    pub fn snapshot_path(&self) -> PathBuf {
        self.snapshot_dir
            .join(snapshot_file_name(&self.model_id, &self.model_hash))
    }

    /// Blocking snapshot write (call from `spawn_blocking`). Requires the
    /// admin lock to be held by the caller; safe while serving (weights are
    /// read-only during inference).
    pub fn snapshot_blocking(&self) -> Result<(PathBuf, SnapshotWriteStats)> {
        let path = self.snapshot_path();
        let manifest = self.manifest.read().clone();
        let stats = write_snapshot(&path, &self.arena, &manifest, self.model_hash)
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

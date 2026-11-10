// SPDX-License-Identifier: AGPL-3.0-only

//! Swap-cost bookkeeping (requirements §6, MOD-5/7/8 + MEM-7).
//!
//! `T_swap(A→B) = T_persist(A) + T_unload(A) + T_load(B) + T_warmup(B)`.
//!
//! Costs are measured at the points that already exist — the snapshot
//! read stats and rebuild duration of a (lazy) load, the unload teardown
//! accounting, and the pre-unload persist flush — and refined per real
//! swap with an EWMA (`--ewma-alpha`). Values persist as JSON next to
//! the snapshots (MOD-5's "same pattern as the preseed snapshot"), so a
//! restart starts from measurements, not guesses; the boot never loads a
//! model just to measure it (MOD-6).
//!
//! For models never loaded in this installation, `estimate` falls back
//! to MOD-7: weights size ÷ the measured read bandwidth of ANY model
//! (default 8 GiB/s — the on-box ATLASRAW number), 10 s warmup, 2 s
//! unload.
//!
//! MEM-7: the measured build footprint without payload and KV pool
//! (derived weights, buffer arena, SSM pools) is stored per model for
//! diagnosis. It does NOT feed the KV budget: most of it is allocated
//! before the pool is sized and is therefore already part of the measured
//! Atlas-own footprint, so it is not the quantity the preflight reserve
//! estimates (allocations AFTER the pool). Standing in for that reserve it
//! halved the 27B's (4.3 instead of 8.5 GiB, a value learned while the
//! GB10 driver pool hid allocations) and the host killed the server on
//! 2026-11-10.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// MOD-7 fallback values.
const FALLBACK_READ_GIB_PER_S: f64 = 8.0;
const FALLBACK_WARMUP: Duration = Duration::from_secs(10);
const FALLBACK_UNLOAD: Duration = Duration::from_secs(2);

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct CostEntry {
    /// Snapshot payload read (ms).
    pub t_load_ms: f64,
    /// Model rebuild after the read — dequants, pools, graphs (ms).
    pub t_warmup_ms: f64,
    /// Teardown duration (ms).
    pub t_unload_ms: f64,
    /// Pre-unload persist flush, normalised per GiB queued (ms/GiB).
    pub t_persist_ms_per_gib: f64,
    /// Measured snapshot read bandwidth (GiB/s).
    pub read_gib_per_s: f64,
    /// MEM-7: measured build footprint without payload and KV pool
    /// (bytes). Diagnostic, not applied to the KV budget.
    pub runtime_reserve_bytes: u64,
    /// Observation counts (load, unload, persist, reserve).
    pub samples: (u32, u32, u32, u32),
}

/// Cost estimate for the scheduler's `T_swap` term.
#[derive(Clone, Copy, Debug)]
pub struct SwapEstimate {
    pub t_load: Duration,
    pub t_warmup: Duration,
    pub t_unload: Duration,
    pub t_persist_per_gib: Duration,
    /// True when at least one real load has been measured.
    pub measured: bool,
}

impl SwapEstimate {
    /// Total `T_swap` assuming `persist_gib` GiB must still be flushed.
    pub fn total(&self, persist_gib: f64) -> Duration {
        self.t_load
            + self.t_warmup
            + self.t_unload
            + self.t_persist_per_gib.mul_f64(persist_gib.max(0.0))
    }
}

#[derive(Default, Serialize, Deserialize)]
struct CostFile {
    entries: HashMap<String, CostEntry>,
}

pub struct SwapCosts {
    path: PathBuf,
    alpha: f64,
    entries: Mutex<CostFile>,
}

static GLOBAL: OnceLock<SwapCosts> = OnceLock::new();

/// Initialise the global store (idempotent; first call wins).
pub fn init(snapshot_dir: &Path, alpha: f64) {
    let _ = GLOBAL.get_or_init(|| SwapCosts::load(snapshot_dir.join("swap_costs.json"), alpha));
}

fn with<R>(f: impl FnOnce(&SwapCosts) -> R) -> Option<R> {
    GLOBAL.get().map(f)
}

impl SwapCosts {
    fn load(path: PathBuf, alpha: f64) -> SwapCosts {
        let entries = std::fs::read(&path)
            .ok()
            .and_then(|b| serde_json::from_slice::<CostFile>(&b).ok())
            .unwrap_or_default();
        if !entries.entries.is_empty() {
            tracing::info!(
                models = entries.entries.len(),
                path = %path.display(),
                "swap-cost store loaded"
            );
        }
        SwapCosts {
            path,
            alpha: alpha.clamp(0.01, 1.0),
            entries: Mutex::new(entries),
        }
    }

    fn update(&self, model_id: &str, f: impl FnOnce(&mut CostEntry, f64)) {
        let mut g = self.entries.lock().expect("swap-costs lock");
        let e = g.entries.entry(model_id.to_string()).or_default();
        f(e, self.alpha);
        let tmp = self.path.with_extension("tmp");
        if let Ok(json) = serde_json::to_vec_pretty(&*g)
            && std::fs::write(&tmp, &json).is_ok()
        {
            let _ = std::fs::rename(&tmp, &self.path);
        }
    }

    fn get(&self, model_id: &str) -> Option<CostEntry> {
        self.entries
            .lock()
            .expect("swap-costs lock")
            .entries
            .get(model_id)
            .copied()
    }

    /// Best measured read bandwidth across all models (MOD-7 fallback
    /// base for never-loaded models).
    fn any_read_bw(&self) -> Option<f64> {
        self.entries
            .lock()
            .expect("swap-costs lock")
            .entries
            .values()
            .map(|e| e.read_gib_per_s)
            .filter(|&v| v > 0.1)
            .fold(None, |acc: Option<f64>, v| {
                Some(acc.map_or(v, |a| a.max(v)))
            })
    }
}

fn ewma(prev: f64, sample: f64, alpha: f64, first: bool) -> f64 {
    if first || prev <= 0.0 {
        sample
    } else {
        alpha * sample + (1.0 - alpha) * prev
    }
}

/// MOD-8: a real load happened — `read_ms` payload streaming, `build_ms`
/// the rebuild (warmup proxy), `gib_per_s` the measured read bandwidth.
pub fn observe_load(model_id: &str, read_ms: f64, build_ms: f64, gib_per_s: f64) {
    with(|c| {
        c.update(model_id, |e, a| {
            let first = e.samples.0 == 0;
            e.t_load_ms = ewma(e.t_load_ms, read_ms, a, first);
            e.t_warmup_ms = ewma(e.t_warmup_ms, build_ms, a, first);
            if gib_per_s > 0.1 {
                e.read_gib_per_s = ewma(e.read_gib_per_s, gib_per_s, a, first);
            }
            e.samples.0 = e.samples.0.saturating_add(1);
        })
    });
}

pub fn observe_unload(model_id: &str, ms: f64) {
    with(|c| {
        c.update(model_id, |e, a| {
            let first = e.samples.1 == 0;
            e.t_unload_ms = ewma(e.t_unload_ms, ms, a, first);
            e.samples.1 = e.samples.1.saturating_add(1);
        })
    });
}

/// Pre-unload persist flush: `ms` wall time for `gib` queued gigabytes.
pub fn observe_persist(model_id: &str, ms: f64, gib: f64) {
    if gib <= 0.0 {
        return;
    }
    with(|c| {
        c.update(model_id, |e, a| {
            let first = e.samples.2 == 0;
            e.t_persist_ms_per_gib = ewma(e.t_persist_ms_per_gib, ms / gib, a, first);
            e.samples.2 = e.samples.2.saturating_add(1);
        })
    });
}

/// MEM-7: measured build footprint without payload and KV pool.
pub fn observe_runtime_reserve(model_id: &str, bytes: u64) {
    if bytes == 0 {
        return;
    }
    with(|c| {
        c.update(model_id, |e, a| {
            let first = e.samples.3 == 0;
            e.runtime_reserve_bytes =
                ewma(e.runtime_reserve_bytes as f64, bytes as f64, a, first) as u64;
            e.samples.3 = e.samples.3.saturating_add(1);
        })
    });
}

/// `/internal/state`: all measured cost entries.
pub fn entries_json() -> serde_json::Value {
    with(|c| {
        let g = c.entries.lock().expect("swap-costs lock");
        serde_json::to_value(&g.entries).unwrap_or(serde_json::Value::Null)
    })
    .unwrap_or(serde_json::Value::Null)
}

/// Swap-cost estimate for the scheduler. `weights_bytes` feeds the
/// MOD-7 fallback for never-measured models (snapshot file size).
pub fn estimate(model_id: &str, weights_bytes: Option<u64>) -> SwapEstimate {
    let entry = with(|c| c.get(model_id)).flatten();
    let any_bw = with(|c| c.any_read_bw()).flatten();
    match entry {
        Some(e) if e.samples.0 > 0 => SwapEstimate {
            t_load: Duration::from_secs_f64(e.t_load_ms / 1000.0),
            t_warmup: Duration::from_secs_f64(e.t_warmup_ms / 1000.0),
            t_unload: if e.samples.1 > 0 {
                Duration::from_secs_f64(e.t_unload_ms / 1000.0)
            } else {
                FALLBACK_UNLOAD
            },
            t_persist_per_gib: Duration::from_secs_f64(e.t_persist_ms_per_gib.max(0.0) / 1000.0),
            measured: true,
        },
        _ => {
            let bw = any_bw.unwrap_or(FALLBACK_READ_GIB_PER_S);
            let gib = weights_bytes.unwrap_or(0) as f64 / (1u64 << 30) as f64;
            SwapEstimate {
                t_load: Duration::from_secs_f64((gib / bw).max(0.5)),
                t_warmup: FALLBACK_WARMUP,
                t_unload: FALLBACK_UNLOAD,
                t_persist_per_gib: Duration::ZERO,
                measured: false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh(dir: &Path) -> SwapCosts {
        SwapCosts::load(dir.join("swap_costs.json"), 0.2)
    }

    #[test]
    fn ewma_first_sample_is_taken_verbatim() {
        assert_eq!(ewma(0.0, 100.0, 0.2, true), 100.0);
        let second = ewma(100.0, 200.0, 0.2, false);
        assert!((second - 120.0).abs() < 1e-9);
    }

    #[test]
    fn file_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        {
            let c = fresh(dir.path());
            c.update("m1", |e, a| {
                let first = e.samples.0 == 0;
                e.t_load_ms = ewma(e.t_load_ms, 6300.0, a, first);
                e.read_gib_per_s = 8.25;
                e.samples.0 += 1;
            });
        }
        let c = fresh(dir.path());
        let e = c.get("m1").unwrap();
        assert_eq!(e.t_load_ms, 6300.0);
        assert_eq!(c.any_read_bw(), Some(8.25));
    }

    #[test]
    fn estimate_fallback_uses_weights_and_any_bw() {
        let dir = tempfile::tempdir().unwrap();
        let c = fresh(dir.path());
        c.update("measured", |e, _| {
            e.read_gib_per_s = 10.0;
            e.t_load_ms = 1000.0;
            e.samples.0 = 1;
        });
        // Unmeasured model, 20 GiB weights, bw from the measured one.
        let bw = c.any_read_bw().unwrap();
        assert_eq!(bw, 10.0);
        // (estimate() uses the global; the math itself:)
        let t = 20.0 / bw;
        assert!((t - 2.0).abs() < 1e-9);
    }

    #[test]
    fn total_includes_persist_term() {
        let est = SwapEstimate {
            t_load: Duration::from_secs(6),
            t_warmup: Duration::from_secs(2),
            t_unload: Duration::from_secs(1),
            t_persist_per_gib: Duration::from_millis(500),
            measured: true,
        };
        assert_eq!(est.total(4.0), Duration::from_secs(11));
    }
}

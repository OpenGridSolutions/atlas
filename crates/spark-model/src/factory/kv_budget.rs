// SPDX-License-Identifier: AGPL-3.0-only

//! Pure KV-budget arithmetic (Plan.md Phase M), extracted from
//! `factory::build` so the 2026-07-13 TranslateGemma load failure is
//! unit-replayable without a GPU. `build.rs` feeds measured values in
//! and logs from the returned struct; the math here must stay
//! semantically identical to what it replaced:
//!
//!   used_so_far = total − free_now
//!   manual override (ATLAS_KV_EXTERNAL_RESERVE_GB): used −= reserve
//!   else auto: atlas_own = baseline − free_now, sanity-gated
//!   total_budget = total × utilization
//!   kv = pinned override OR (total_budget − used − inference_reserve),
//!        both clamped to (free_now − inference_reserve)

/// Where `used_so_far` came from — drives build.rs's log line choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UsedSource {
    /// Raw `total − free` (no baseline captured, e.g. mock backend).
    Raw,
    /// Manual ATLAS_KV_EXTERNAL_RESERVE_GB discount.
    Manual,
    /// Auto self-relative: `baseline − free_now` passed the sanity gate.
    AutoBaseline,
    /// Auto measure implausible → fell back to raw `total − free`.
    AutoImplausible,
}

pub(crate) struct BudgetInputs {
    /// `gpu.total_memory()`.
    pub total_mem: usize,
    /// `gpu.free_memory()` right now — since Phase M this is
    /// `max(cudaMemGetInfo, MemAvailable + SwapFree)`.
    pub actual_free: usize,
    /// `gpu::baseline_free_bytes()` captured at context init.
    pub baseline_free: Option<usize>,
    /// ATLAS_KV_EXTERNAL_RESERVE_GB in BYTES (already parsed, > 0).
    pub manual_external_reserve: Option<usize>,
    /// --gpu-memory-utilization.
    pub gpu_memory_utilization: f64,
    /// SSM pools + CUDA headroom.
    pub inference_reserve: usize,
    /// --kv-pool-gb in bytes (deterministic hot-swap sizing).
    pub kv_pool_override: Option<usize>,
}

pub(crate) struct BudgetResult {
    pub used_so_far: usize,
    pub total_budget: usize,
    pub kv_budget: usize,
    pub source: UsedSource,
    /// Auto path only: co-tenant bytes excluded (`raw_used − atlas_own`).
    pub cotenants: usize,
}

pub(crate) fn compute(i: &BudgetInputs) -> BudgetResult {
    let raw_used = i.total_mem.saturating_sub(i.actual_free);
    let (used_so_far, source, cotenants) = if let Some(ext) = i.manual_external_reserve {
        (raw_used.saturating_sub(ext), UsedSource::Manual, 0)
    } else if let Some(baseline) = i.baseline_free {
        // Sanity gate identical to the original build.rs block: atlas_own
        // positive and no larger than total used (co-tenants can't be
        // negative); implausible numbers fall back to raw.
        let atlas_own = baseline.saturating_sub(i.actual_free);
        if atlas_own > 0 && atlas_own <= raw_used {
            (atlas_own, UsedSource::AutoBaseline, raw_used - atlas_own)
        } else {
            (raw_used, UsedSource::AutoImplausible, 0)
        }
    } else {
        (raw_used, UsedSource::Raw, 0)
    };

    let total_budget = (i.total_mem as f64 * i.gpu_memory_utilization) as usize;
    let physical_cap = i.actual_free.saturating_sub(i.inference_reserve);
    let kv_budget = match i.kv_pool_override {
        Some(pinned) => pinned.min(physical_cap),
        None => total_budget
            .saturating_sub(used_so_far)
            .saturating_sub(i.inference_reserve)
            .min(physical_cap),
    };
    BudgetResult {
        used_so_far,
        total_budget,
        kv_budget,
        source,
        cotenants,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    fn gib(v: f64) -> usize {
        (v * GIB) as usize
    }
    fn as_gib(v: usize) -> f64 {
        v as f64 / GIB
    }

    /// The measured 2026-07-13 10:51 failure, verbatim from the user's
    /// server log: baseline 116.5 GB, free-now 20.6 GB (the flawed
    /// cudaMemGetInfo/MemAvailable reading), 119.7 GB total × 90 % util,
    /// 2.6 GB reserve → "Atlas-own 95.9 GB" and a 9.2 GB KV pool, below
    /// the 14.6 GB that --max-seq-len=104416 needs on TranslateGemma.
    fn failure_scene(free_now: usize) -> BudgetInputs {
        BudgetInputs {
            total_mem: gib(119.7),
            actual_free: free_now,
            baseline_free: Some(gib(116.5)),
            manual_external_reserve: None,
            gpu_memory_utilization: 0.90,
            inference_reserve: gib(2.6),
            kv_pool_override: None,
        }
    }

    /// TranslateGemma's requirement in the failing config: 6526 blocks
    /// × ~2.29 MB/block (48-layer selective-boundary cache) ≈ 14.6 GiB.
    const NEEDED_GIB: f64 = 14.6;

    #[test]
    fn failure_replay_old_reading_starves_kv() {
        // Pre-fix reading: 20.6 GB (swap-reclaimable memory invisible).
        let r = compute(&failure_scene(gib(20.6)));
        assert_eq!(r.source, UsedSource::AutoBaseline);
        assert!((as_gib(r.used_so_far) - 95.9).abs() < 0.15, "{}", as_gib(r.used_so_far));
        assert!((as_gib(r.kv_budget) - 9.2).abs() < 0.15, "{}", as_gib(r.kv_budget));
        // …which is exactly the observed failure:
        assert!(as_gib(r.kv_budget) < NEEDED_GIB);
    }

    #[test]
    fn fix_replay_meminfo_reading_restores_budget() {
        // Post-fix reading in the same system state: the ~70 GB of
        // swap-reclaimable memory is visible again (20.6 cuda vs ~91
        // MemAvailable+SwapFree).
        let r = compute(&failure_scene(gib(91.0)));
        assert_eq!(r.source, UsedSource::AutoBaseline);
        // Atlas-own back to reality: TranslateGemma weights + arena ≈ 26 GB.
        assert!((as_gib(r.used_so_far) - 25.5).abs() < 0.15, "{}", as_gib(r.used_so_far));
        assert!(as_gib(r.kv_budget) > 65.0, "{}", as_gib(r.kv_budget));
        assert!(as_gib(r.kv_budget) > NEEDED_GIB);
        // Overcommit guard: weights + KV stay under the util ceiling.
        assert!(r.used_so_far + r.kv_budget + gib(2.6) <= r.total_budget + gib(0.01));
    }

    #[test]
    fn implausible_baseline_falls_back_to_raw() {
        // Baseline below free-now (co-tenant freed more than we ever
        // allocated / stale baseline) → atlas_own = 0 → raw fallback.
        let mut i = failure_scene(gib(91.0));
        i.baseline_free = Some(gib(50.0));
        let r = compute(&i);
        assert_eq!(r.source, UsedSource::AutoImplausible);
        assert_eq!(r.used_so_far, gib(119.7) - gib(91.0));
    }

    #[test]
    fn manual_reserve_wins_over_auto() {
        let mut i = failure_scene(gib(91.0));
        i.manual_external_reserve = Some(gib(3.0));
        let r = compute(&i);
        assert_eq!(r.source, UsedSource::Manual);
        // raw used (28.7) minus the manual co-tenant discount (3.0)
        assert!((as_gib(r.used_so_far) - 25.7).abs() < 0.15);
    }

    #[test]
    fn kv_pool_override_is_clamped_to_physical() {
        let mut i = failure_scene(gib(20.6));
        i.kv_pool_override = Some(gib(200.0));
        let r = compute(&i);
        // min(200, 20.6 − 2.6) = 18.0
        assert!((as_gib(r.kv_budget) - 18.0).abs() < 0.05);
    }

    #[test]
    fn no_baseline_uses_raw_used() {
        let mut i = failure_scene(gib(60.0));
        i.baseline_free = None;
        let r = compute(&i);
        assert_eq!(r.source, UsedSource::Raw);
        assert_eq!(r.used_so_far, gib(119.7) - gib(60.0));
    }
}

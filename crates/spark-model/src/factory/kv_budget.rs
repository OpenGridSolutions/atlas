// SPDX-License-Identifier: AGPL-3.0-only

//! Pure KV-budget arithmetic (Plan.md Phase M), extracted from
//! `factory::build` so the 2026-07-13 TranslateGemma load failure is
//! unit-replayable without a GPU. `build.rs` feeds measured values in
//! and logs from the returned struct; the math here must stay
//! semantically identical to what it replaced:
//!
//!   used_so_far = total − free_now
//!   manual override (ATLAS_KV_EXTERNAL_RESERVE_GB): used −= reserve
//!   else auto: atlas_own = baseline − free_now (both swap-inclusive,
//!   neither clamped), sanity-gated
//!   total_budget = total × utilization
//!   kv = pinned override OR (total_budget − used − inference_reserve),
//!        both clamped to (pinnable_free − inference_reserve) — swap-free
//!        space is excluded from the clamp, KV pages are pinned
//!   a pending sidecar reserve (Todo 3.2b) comes off both the budget and
//!   the physical clamp — memory a configured sidecar has yet to allocate —
//!   but never takes more than half of the pool the model would get
//!   without it: the generative model is the primary workload, the sidecar
//!   refuses work at request time when its memory is not there

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
    /// `max(cudaMemGetInfo, MemAvailable + SwapFree)`. May exceed
    /// `total_mem` (swap); `compute` clamps it to `total_mem` for the raw
    /// used figure and the physical cap, not for the baseline delta.
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
    /// `MemAvailable` WITHOUT SwapFree, when /proc/meminfo is readable.
    /// The physical clamp uses this instead of `actual_free`: KV pool
    /// pages are GPU-touched and pinned on UMA, so swap space is no room
    /// for them — counting it handed the 2026-10-03 deployment ~7.7 GB
    /// of phantom headroom and the host OOM-killed the server at the
    /// first prefill. `actual_free` keeps the swap-inclusive basis for
    /// the baseline delta (both readings share it).
    pub pinnable_free: Option<usize>,
    /// `gpu::sidecar_pending_reserve_bytes()` — memory the small text
    /// sidecars (embed/rerank) hold outside the arena and the KV pool.
    pub sidecar_reserve: usize,
}

pub(crate) struct BudgetResult {
    pub used_so_far: usize,
    pub total_budget: usize,
    pub kv_budget: usize,
    pub source: UsedSource,
    /// Auto path only: co-tenant bytes excluded (`raw_used − atlas_own`).
    pub cotenants: usize,
    /// How much of `sidecar_reserve` was actually kept out of the pool.
    pub sidecar_reserved: usize,
}

pub(crate) fn compute(i: &BudgetInputs) -> BudgetResult {
    // UMA basis: free readings are `MemAvailable + SwapFree` and can exceed
    // the CUDA total (swap is not part of `total_mem`). `raw_used` needs
    // the reading clamped to `total_mem`. The baseline delta must NOT be
    // clamped: both readings carry the same swap term, so only their plain
    // difference is this process's footprint. Clamping the baseline first
    // dropped `baseline − total` (the swap size on an idle box) from
    // Atlas-own and handed it to the pool — 2.8 to 3.5 GB too much on every
    // load, 2026-11-10.
    let actual_free = i.actual_free.min(i.total_mem);
    let raw_used = i.total_mem - actual_free;
    let (used_so_far, source, cotenants) = if let Some(ext) = i.manual_external_reserve {
        (raw_used.saturating_sub(ext), UsedSource::Manual, 0)
    } else if let Some(baseline) = i.baseline_free {
        // Sanity gate: atlas_own positive and no larger than the device.
        // The gate used to be `<= raw_used`, which a baseline above the
        // total fails on every load (124.0 GB baseline vs 119.7 GB total,
        // Memory_errors.log 2026-09-11) — the raw fallback then bills
        // co-tenants and page cache to Atlas.
        let atlas_own = baseline.saturating_sub(i.actual_free);
        if atlas_own > 0 && atlas_own <= i.total_mem {
            (
                atlas_own,
                UsedSource::AutoBaseline,
                raw_used.saturating_sub(atlas_own),
            )
        } else {
            (raw_used, UsedSource::AutoImplausible, 0)
        }
    } else {
        (raw_used, UsedSource::Raw, 0)
    };

    let total_budget = (i.total_mem as f64 * i.gpu_memory_utilization) as usize;
    let pinnable_free = i.pinnable_free.unwrap_or(actual_free).min(actual_free);
    let pool = |sidecar: usize| {
        let physical_cap = pinnable_free
            .saturating_sub(i.inference_reserve)
            .saturating_sub(sidecar);
        match i.kv_pool_override {
            Some(pinned) => pinned.min(physical_cap),
            None => total_budget
                .saturating_sub(used_so_far)
                .saturating_sub(i.inference_reserve)
                .saturating_sub(sidecar)
                .min(physical_cap),
        }
    };
    let unreserved = pool(0);
    let kv_budget = pool(i.sidecar_reserve).max(unreserved / 2);
    BudgetResult {
        used_so_far,
        total_budget,
        kv_budget,
        source,
        cotenants,
        sidecar_reserved: unreserved - kv_budget,
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
            sidecar_reserve: 0,
            pinnable_free: None,
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
        assert!(
            (as_gib(r.used_so_far) - 95.9).abs() < 0.15,
            "{}",
            as_gib(r.used_so_far)
        );
        assert!(
            (as_gib(r.kv_budget) - 9.2).abs() < 0.15,
            "{}",
            as_gib(r.kv_budget)
        );
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
        assert!(
            (as_gib(r.used_so_far) - 25.5).abs() < 0.15,
            "{}",
            as_gib(r.used_so_far)
        );
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
    fn pending_sidecar_reserve_comes_off_the_kv_pool() {
        // A 30 GB image pipeline is configured but not resident yet: the
        // generative model's pool must leave that much unclaimed, on the
        // auto path and under a pinned --kv-pool-gb alike.
        let base = compute(&failure_scene(gib(91.0)));
        let mut i = failure_scene(gib(91.0));
        i.sidecar_reserve = gib(30.0);
        let r = compute(&i);
        assert!((as_gib(base.kv_budget) - as_gib(r.kv_budget) - 30.0).abs() < 0.01);
        assert_eq!(r.sidecar_reserved, gib(30.0));
        i.kv_pool_override = Some(gib(200.0));
        // min(200, 91 − 2.6 − 30) = 58.4
        assert!((as_gib(compute(&i).kv_budget) - 58.4).abs() < 0.05);
    }

    #[test]
    fn sidecar_reserve_yields_instead_of_starving_the_model() {
        // Tight box: 45 GB free, the full 34 GB reserve would leave no KV
        // pool at all (the 2026-10-02 "No memory left for KV cache" load
        // failure). The reserve gives way at half the unreserved pool.
        let mut i = failure_scene(gib(45.0));
        let unreserved = compute(&i).kv_budget;
        i.sidecar_reserve = gib(34.0);
        let r = compute(&i);
        assert!(unreserved > 0);
        assert_eq!(r.kv_budget, unreserved / 2);
        assert_eq!(r.sidecar_reserved, unreserved - unreserved / 2);
        assert!(r.sidecar_reserved < gib(34.0));
    }

    #[test]
    fn baseline_above_total_stays_on_the_auto_path() {
        // The 2026-09-11 hot-swap failure: the boot baseline was captured
        // as MemAvailable + SwapFree (124.0 GB) and exceeds the CUDA total
        // (119.7 GB). Gated against `total − free`, atlas_own > raw_used on
        // EVERY later load, so the auto path was structurally dead and the
        // raw fallback billed all resident memory to Atlas.
        let mut i = failure_scene(gib(12.3));
        i.baseline_free = Some(gib(124.0));
        let r = compute(&i);
        assert_eq!(r.source, UsedSource::AutoBaseline);
        // Both readings carry the same swap term, so their difference is
        // the footprint — not `total − free-now`, which would hide the
        // part of the baseline that lies above the CUDA total.
        assert_eq!(r.used_so_far, gib(124.0) - gib(12.3));
        assert_eq!(r.cotenants, 0);
    }

    /// 2026-11-10, first day on a driver that reports its memory pools as
    /// free (620.70.06): free readings are honest, the baseline sits above
    /// the CUDA total by the swap size, and the pool fills the box.
    fn honest_driver_scene(baseline: f64, free_now: f64, reserve: f64) -> BudgetInputs {
        BudgetInputs {
            total_mem: gib(119.69),
            actual_free: gib(free_now),
            baseline_free: Some(gib(baseline)),
            manual_external_reserve: None,
            gpu_memory_utilization: 0.90,
            inference_reserve: gib(reserve),
            kv_pool_override: None,
            sidecar_reserve: 0,
            // 8 GiB swap, all of it free at load time.
            pinnable_free: Some(gib(free_now - 8.0)),
        }
    }

    #[test]
    fn swap_above_total_is_not_dropped_from_atlas_own() {
        // 80B, serve log 17:50: baseline 122.5, free-now 72.4, reserve 4.64.
        // The baseline was clamped to the total before the subtraction, so
        // the log said "Atlas-own 47.3 GB" for a 50.1 GB footprint and the
        // pool got the missing 2.8 GB (55.8 GB; MemAvailable 3.9 GiB left).
        let i = honest_driver_scene(122.5, 72.4, 4.64);
        let r = compute(&i);
        assert_eq!(r.source, UsedSource::AutoBaseline);
        assert!(
            (as_gib(r.used_so_far) - 50.1).abs() < 0.05,
            "{}",
            as_gib(r.used_so_far)
        );
        assert!(
            (as_gib(r.kv_budget) - 52.98).abs() < 0.05,
            "{}",
            as_gib(r.kv_budget)
        );
        assert!(r.used_so_far + r.kv_budget + i.inference_reserve <= r.total_budget);
    }

    #[test]
    fn crash_replay_27b_stays_under_the_ceiling() {
        // 27B, serve log 18:01: baseline 123.2, free-now 80.4. Logged
        // "Atlas-own 39.3 GB" (real: 42.8) and a 64.1 GB pool next to
        // 8.2 GB of SSM pools: 115 GB of a 107.7 GB ceiling. RAM and swap
        // ran out during the first request and the host killed the server.
        // With the full footprint and the model's own reserve (8.54):
        let i = honest_driver_scene(123.2, 80.4, 8.54);
        let r = compute(&i);
        assert!(
            (as_gib(r.used_so_far) - 42.8).abs() < 0.05,
            "{}",
            as_gib(r.used_so_far)
        );
        assert!(
            (as_gib(r.kv_budget) - 56.38).abs() < 0.05,
            "{}",
            as_gib(r.kv_budget)
        );
        assert!(r.used_so_far + r.kv_budget + i.inference_reserve <= r.total_budget);
    }

    #[test]
    fn small_footprint_below_the_swap_size_is_still_measured() {
        // A 3 GB model on an idle box: both readings exceed the CUDA total.
        // Clamped, the difference was 0 and the gate called it implausible.
        let r = compute(&honest_driver_scene(123.7, 120.7, 0.5));
        assert_eq!(r.source, UsedSource::AutoBaseline);
        assert!((as_gib(r.used_so_far) - 3.0).abs() < 0.01);
    }

    #[test]
    fn free_above_total_clamps_used_to_zero() {
        // Fresh boot with empty swap: MemAvailable + SwapFree > CUDA total.
        let mut i = failure_scene(gib(125.0));
        i.baseline_free = None;
        let r = compute(&i);
        assert_eq!(r.source, UsedSource::Raw);
        assert_eq!(r.used_so_far, 0);
        // Budget: total×0.9 − 0 − reserve; the physical cap (clamped free
        // − reserve = 117.1) does not bind.
        assert!((as_gib(r.kv_budget) - (119.7 * 0.9 - 2.6)).abs() < 0.05);
    }

    #[test]
    fn no_baseline_uses_raw_used() {
        let mut i = failure_scene(gib(60.0));
        i.baseline_free = None;
        let r = compute(&i);
        assert_eq!(r.source, UsedSource::Raw);
        assert_eq!(r.used_so_far, gib(119.7) - gib(60.0));
    }

    /// The 2026-10-03 deployment OOM kill, verbatim from the serve log:
    /// lazy swap to the 27B next to the resident 26.7 GB image sidecar.
    /// baseline 74.8 (re-armed post-unload, BEFORE the sidecar loaded),
    /// free-now 52.7 = MemAvailable 45.0 + SwapFree 7.7, boot model's
    /// inference reserve 4.6 GB standing in for the target's real 8.7 GB,
    /// sidecar post-load reserve 7.0 GB.
    fn oom_scene() -> BudgetInputs {
        BudgetInputs {
            total_mem: gib(119.7),
            actual_free: gib(52.7),
            baseline_free: Some(gib(74.8)),
            manual_external_reserve: None,
            gpu_memory_utilization: 0.90,
            inference_reserve: gib(4.6),
            kv_pool_override: None,
            sidecar_reserve: gib(7.0),
            pinnable_free: None,
        }
    }

    #[test]
    fn oom_replay_swap_inclusive_clamp_oversubscribes() {
        // Pre-fix: the clamp counts SwapFree → 52.7 − 4.6 − 7.0 = 41.1 GB
        // KV. With the real sibling allocations (~11.9 GB vs the 4.6
        // reserve) that exceeded MemAvailable and the host OOM-killed the
        // server at the first prefill.
        let r = compute(&oom_scene());
        assert!(
            (as_gib(r.kv_budget) - 41.1).abs() < 0.05,
            "{}",
            as_gib(r.kv_budget)
        );
        let real_siblings = gib(11.9);
        assert!(
            r.kv_budget + real_siblings > gib(45.0),
            "did not oversubscribe"
        );
    }

    #[test]
    fn oom_fix_pinnable_clamp_and_target_reserve_fit() {
        // Post-fix: the clamp uses MemAvailable only (45.0) and the swap
        // path recomputes the TARGET model's reserve (8.7 GB) → KV
        // 45.0 − 8.7 − 7.0 = 29.3 GB; KV + real siblings fit MemAvailable
        // with headroom.
        let mut i = oom_scene();
        i.pinnable_free = Some(gib(45.0));
        i.inference_reserve = gib(8.7);
        let r = compute(&i);
        assert!(
            (as_gib(r.kv_budget) - 29.3).abs() < 0.05,
            "{}",
            as_gib(r.kv_budget)
        );
        let real_siblings = gib(11.9);
        assert!(r.kv_budget + real_siblings < gib(45.0));
    }

    #[test]
    fn pinnable_free_never_raises_the_cap() {
        // meminfo hiccup reporting more than the swap-inclusive reading:
        // the clamp must not widen beyond actual_free.
        let mut i = oom_scene();
        i.pinnable_free = Some(gib(200.0));
        let r = compute(&i);
        assert!((as_gib(r.kv_budget) - 41.1).abs() < 0.05);
    }
}

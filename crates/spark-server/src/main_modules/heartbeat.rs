// SPDX-License-Identifier: AGPL-3.0-only

//! Memory heartbeat (Todo.md Phase 1.3): one compact INFO line per
//! minute on target `mem` — the grep-able time profile that replaces a
//! dashboard (`grep 'mem:' logs/atlas-*.log`). Threshold breaches log
//! WARN instead, which routes them into critical.log as the early
//! warning for cases A1/A2/A8 (OOM, watchdog kill, swap thrashing).
//!
//! Interval: `ATLAS_MEM_HEARTBEAT_SECS` (default 60, 0 = off).
//! Thresholds: `ATLAS_MEM_WARN_AVAIL_GB` (default 8),
//! `ATLAS_MEM_WARN_SWAP_GB` (default 2).

use std::sync::Arc;

use crate::main_modules::hotswap::HotswapCtx;

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub(crate) fn spawn(hotswap: Option<Arc<HotswapCtx>>) -> Option<tokio::task::JoinHandle<()>> {
    let secs = env_f64("ATLAS_MEM_HEARTBEAT_SECS", 60.0);
    if secs <= 0.0 {
        tracing::info!("Memory heartbeat disabled (ATLAS_MEM_HEARTBEAT_SECS=0)");
        return None;
    }
    let warn_avail_gb = env_f64("ATLAS_MEM_WARN_AVAIL_GB", 8.0);
    let warn_swap_gb = env_f64("ATLAS_MEM_WARN_SWAP_GB", 2.0);
    tracing::info!(
        "Memory heartbeat: every {secs:.0}s on target 'mem' \
         (WARN below {warn_avail_gb} GB available / {warn_swap_gb} GB swap-free)"
    );
    Some(tokio::spawn(async move {
        let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
        let mut tick = tokio::time::interval(std::time::Duration::from_secs_f64(secs));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            let mem = spark_runtime::meminfo::read();
            let cuda = spark_runtime::cuda_backend::cuda_mem_get_info_raw();
            let pool = spark_runtime::kv_stats::snapshot();
            let (avail_gb, swap_gb) = mem
                .map(|m| (gib(m.mem_available), gib(m.swap_free)))
                .unwrap_or((-1.0, -1.0));
            let cuda_free_gb = cuda.map(|(f, _)| gib(f)).unwrap_or(-1.0);
            let (pool_used, pool_total) = pool
                .map(|p| (p.used_blocks as i64, p.total_blocks as i64))
                .unwrap_or((-1, -1));
            let (arena_used_gb, arena_cap_gb) = hotswap
                .as_ref()
                .map(|c| (gib(c.arena.used()), gib(c.arena.capacity())))
                .unwrap_or((-1.0, -1.0));
            let rss_mib = std::fs::read_to_string("/proc/self/status")
                .ok()
                .and_then(|s| {
                    s.lines()
                        .find_map(|l| l.strip_prefix("VmRSS:"))
                        .and_then(|r| r.split_whitespace().next())
                        .and_then(|kb| kb.parse::<i64>().ok())
                })
                .map_or(-1, |kb| kb / 1024);
            let low = (avail_gb >= 0.0 && avail_gb < warn_avail_gb)
                || (swap_gb >= 0.0 && swap_gb < warn_swap_gb);
            if low {
                tracing::warn!(
                    target: "mem",
                    avail_gb, swap_gb, cuda_free_gb, pool_used, pool_total,
                    arena_used_gb, arena_cap_gb, rss_mib,
                    "memory LOW — below warn threshold (early A1/A2/A8 signal)"
                );
            } else {
                tracing::info!(
                    target: "mem",
                    avail_gb, swap_gb, cuda_free_gb, pool_used, pool_total,
                    arena_used_gb, arena_cap_gb, rss_mib,
                    "heartbeat"
                );
            }
        }
    }))
}

// SPDX-License-Identifier: AGPL-3.0-only

//! `GET /v1/stats` — memory + residency diagnostics (Plan.md Phase S).
//!
//! Purpose-built to make the DGX Spark unified-memory measurement bug
//! VISIBLE: `cuda_mem_get_info_free_gb` is the raw (flawed, fluctuating)
//! `cuMemGetInfo` number the KV-budget path currently sees;
//! `proc_meminfo_available_gb` is the truly allocatable amount
//! (`MemAvailable + SwapFree`, HugePages-aware). Their difference,
//! `discrepancy_gb`, is the SUCCESS METRIC for the separate follow-up
//! KV-budget fix: large (~40+ GB) and unstable until the fix lands,
//! small/stable after. This endpoint is strictly read-only — the budget
//! computation is NOT changed here.
//!
//! Per-model numbers reuse what Atlas already knows: the weights arena
//! fill (`HotswapCtx.arena.used()`), the paged-KV pool counters
//! (`spark_runtime::kv_stats`, published by `PagedKvCache`), and the
//! whisper sidecar's BF16 footprint (`audio::sidecar_stats`).

use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Json, Response};

use crate::main_modules::AppState;
use crate::main_modules::hotswap::{ModelCategory, SwapState};

fn gb(bytes: usize) -> f64 {
    (bytes as f64 / (1024.0 * 1024.0 * 1024.0) * 100.0).round() / 100.0
}

pub async fn stats(State(state): State<Arc<AppState>>) -> Response {
    // ── Memory diagnosis (the success metric) ──
    #[cfg(feature = "cuda")]
    let cuda_info = spark_runtime::cuda_backend::cuda_mem_get_info_raw();
    #[cfg(not(feature = "cuda"))]
    let cuda_info: Option<(usize, usize)> = None;
    let meminfo = spark_runtime::meminfo::read();

    let cuda_free_gb = cuda_info.map(|(free, _)| gb(free));
    let proc_available_gb = meminfo.map(|m| gb(m.available_for_alloc));
    let discrepancy_gb = match (proc_available_gb, cuda_free_gb) {
        (Some(p), Some(c)) => Some(((p - c) * 100.0).round() / 100.0),
        _ => None,
    };
    let total_gb = cuda_info
        .map(|(_, total)| gb(total))
        .or_else(|| meminfo.map(|m| gb(m.mem_total)));

    let memory = serde_json::json!({
        // Raw cuMemGetInfo — the flawed number the KV budget currently
        // sees (DGX Spark known issue: swap-reclaimable DRAM invisible).
        "cuda_mem_get_info_free_gb": cuda_free_gb,
        // MemAvailable + SwapFree (HugePages-aware) — the real
        // allocatable amount. Display-only until the budget fix.
        "proc_meminfo_available_gb": proc_available_gb,
        // THE metric: large & fluctuating before the KV-budget fix,
        // small & stable after.
        "discrepancy_gb": discrepancy_gb,
        "total_unified_memory_gb": total_gb,
        "meminfo_detail": meminfo.map(|m| serde_json::json!({
            "mem_available_gb": gb(m.mem_available),
            "swap_free_gb": gb(m.swap_free),
            "hugepages": m.hugepages,
        })),
        // What the LAST KV-budget computation actually used as its
        // free-now input, and the budget it derived. The follow-up
        // /proc/meminfo budget fix moves THIS pair; before/after
        // comparison happens at the exact decision point.
        "kv_budget_last": spark_runtime::kv_stats::budget_inputs().map(
            |(free_now, budget)| serde_json::json!({
                "free_now_seen_gb": gb(free_now),
                "kv_budget_gb": gb(budget),
            }),
        ),
    });

    // ── Per-model residency ──
    let kv = spark_runtime::kv_stats::snapshot();
    let mut models: Vec<serde_json::Value> = Vec::new();
    if let Some(ctx) = state.hotswap.as_ref() {
        let active = ctx.active_model();
        let running = ctx.state() == SwapState::Running;
        for m in &ctx.registry {
            let resident = running && active.model_id == m.model_id;
            let kv_here = if resident { kv } else { None };
            models.push(serde_json::json!({
                "id": m.model_id,
                "category": m.category.as_str(),
                "resident": resident,
                "residency": if resident { Some("arena") } else { None },
                "weights_bytes": if resident { ctx.arena.used() } else { 0 },
                "kv_reserved_bytes": kv_here.map_or(0, |k| k.reserved_bytes),
                "kv_reserved_blocks": kv_here.map_or(0, |k| k.total_blocks),
                "kv_used_blocks": kv_here.map_or(0, |k| k.used_blocks),
            }));
        }
    }
    // Whisper sidecar: outside the arena registry (Phase K), resident
    // from its first transcription until process exit. No paged KV pool
    // — it manages its own cross-attention buffers.
    let sidecar_bytes = super::audio::sidecar_stats();
    for id in &state.transcribe_models {
        models.push(serde_json::json!({
            "id": id,
            "category": ModelCategory::Transcribe.as_str(),
            "resident": sidecar_bytes.is_some(),
            "residency": sidecar_bytes.map(|_| "sidecar"),
            "weights_bytes": sidecar_bytes.unwrap_or(0),
            "kv_reserved_bytes": 0,
            "kv_reserved_blocks": 0,
            "kv_used_blocks": 0,
        }));
    }

    let arena = state.hotswap.as_ref().map(|ctx| {
        serde_json::json!({
            "capacity_bytes": ctx.arena.capacity(),
            "used_bytes": ctx.arena.used(),
            "state": ctx.state().as_str(),
        })
    });

    Json(serde_json::json!({
        "object": "atlas.stats",
        "memory": memory,
        "models": models,
        "arena": arena,
    }))
    .into_response()
}

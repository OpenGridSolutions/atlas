// SPDX-License-Identifier: AGPL-3.0-only

//! `GET /internal/state` (TEL-3): the full scheduler/tier introspection
//! surface — arena model + residency, sidecar models, the MEM-9 capacity
//! profile, KV pool occupancy, batch size, per-model/per-tier queues
//! with D/S/λ, the last swap decision with its inputs, measured swap
//! costs, and the NVMe store + write queue. Bearer-gated like `/admin`
//! (TEL-7). The lightweight liveness/readiness surface stays on
//! `/health` + `/health/live` (TEL-1/2).

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::response::{IntoResponse, Response};

use crate::main_modules::AppState;

/// MEM-9: capacity, not reservation — how many sequences fit at 8k, 32k
/// and the configured maximum context, from the live pool geometry.
fn capacity_profile(max_seq_len: usize) -> serde_json::Value {
    let (block_size, bytes_per_block, ssm_bytes) = spark_runtime::kv_stats::pool_geometry();
    let pool = spark_runtime::kv_stats::snapshot();
    if block_size == 0 || bytes_per_block == 0 {
        return serde_json::Value::Null;
    }
    let bytes_per_token = bytes_per_block as f64 / block_size as f64;
    let pool_tokens = pool
        .as_ref()
        .map(|p| p.total_blocks * block_size)
        .unwrap_or(0);
    let seqs_at = |ctx: usize| {
        if ctx == 0 { 0 } else { pool_tokens / ctx }
    };
    serde_json::json!({
        "block_size_tokens": block_size,
        "bytes_per_token": bytes_per_token,
        "ssm_bytes_per_seq": ssm_bytes,
        "pool_tokens_total": pool_tokens,
        "max_context_tokens": max_seq_len,
        "parallel_seqs": {
            "at_8k": seqs_at(8192),
            "at_32k": seqs_at(32768),
            "at_max_context": seqs_at(max_seq_len),
        },
    })
}

pub async fn internal_state(State(state): State<Arc<AppState>>) -> Response {
    let arena = state.hotswap.as_ref().map(|ctx| {
        let active = ctx.active_model();
        serde_json::json!({
            "state": ctx.state().as_str(),
            "model": active.model_id,
            "resident_s": ctx.active_since.lock().elapsed().as_secs_f64(),
            "registered": ctx
                .registry
                .iter()
                .map(|m| m.model_id.clone())
                .collect::<Vec<_>>(),
        })
    });

    let pool = spark_runtime::kv_stats::snapshot().map(|p| {
        serde_json::json!({
            "total_blocks": p.total_blocks,
            "free_blocks": p.total_blocks.saturating_sub(p.used_blocks),
            "used_blocks": p.used_blocks,
            "reserved_bytes": p.reserved_bytes,
        })
    });

    let nvme = crate::kv_tier::with_active(|t| {
        let (stats, queued_bytes, dropped_full) = t.stats();
        serde_json::json!({
            "entries": stats.entries,
            "total_bytes": stats.total_bytes,
            "written_today_bytes": stats.written_today,
            "fs_free_bytes": stats.free_bytes,
            "fs_total_bytes": stats.fs_total_bytes,
            "write_queue_bytes": queued_bytes,
            "write_queue_rejections": dropped_full,
        })
    });

    let body = serde_json::json!({
        "arena": arena,
        "sidecars": {
            "transcribe": state.transcribe_models,
            "image": state.image_models,
        },
        "capacity_profile": capacity_profile(state.max_seq_len),
        "kv_pool": pool,
        "batch": {
            "scheduler_load": crate::scheduler::scheduler_load(),
        },
        "scheduler": state.arena.as_ref().map(|d| d.state_json()),
        "swap_costs": crate::swap_costs::entries_json(),
        "nvme_store": nvme,
    });
    Json(body).into_response()
}

#[cfg(test)]
mod tests {
    use super::capacity_profile;
    use spark_runtime::kv_stats;

    /// T-TEL-01 / MEM-9: the capacity profile is derived from the live
    /// pool geometry (capacity, not reservation).
    #[test]
    fn capacity_profile_derives_from_pool_geometry() {
        kv_stats::publish_pool_geometry(16, 4096);
        kv_stats::publish_ssm_bytes_per_seq(2048);
        kv_stats::publish_pool(1000, 1000 * 4096);
        let p = capacity_profile(4096);
        assert_eq!(p["block_size_tokens"], 16);
        assert_eq!(p["bytes_per_token"], 256.0);
        assert_eq!(p["ssm_bytes_per_seq"], 2048);
        assert_eq!(p["pool_tokens_total"], 16_000);
        assert_eq!(p["max_context_tokens"], 4096);
        assert_eq!(p["parallel_seqs"]["at_8k"], 1);
        assert_eq!(p["parallel_seqs"]["at_32k"], 0);
        assert_eq!(p["parallel_seqs"]["at_max_context"], 3);
    }
}

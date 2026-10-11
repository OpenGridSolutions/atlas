// SPDX-License-Identifier: AGPL-3.0-only

//! Prometheus metrics for Atlas Spark.

use lazy_static::lazy_static;
use prometheus::{
    Histogram, IntCounter, IntCounterVec, IntGauge, register_histogram, register_int_counter,
    register_int_counter_vec, register_int_gauge,
};

lazy_static! {
    pub static ref REQUESTS_TOTAL: IntCounter =
        register_int_counter!("atlas_requests_total", "Total requests processed").unwrap();
    pub static ref REQUESTS_ACTIVE: IntGauge =
        register_int_gauge!("atlas_requests_active", "Currently active requests").unwrap();
    pub static ref TTFT_SECONDS: Histogram = register_histogram!(
        "atlas_time_to_first_token_seconds",
        "Time to first token",
        vec![0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
    )
    .unwrap();
    pub static ref GENERATION_TOKENS_TOTAL: IntCounter =
        register_int_counter!("atlas_generation_tokens_total", "Total tokens generated").unwrap();
    pub static ref PROMPT_TOKENS_TOTAL: IntCounter =
        register_int_counter!("atlas_prompt_tokens_total", "Total prompt tokens processed")
            .unwrap();

    // ── Loop-detector telemetry (P5.2, 2026-04-25) ──
    //
    // Track the verdict distribution emitted by `loop_detector::detect`
    // so we can tune thresholds against production traffic instead of
    // single dump fixtures. Labels:
    //   - verdict ∈ {none, hint, suppress}
    //   - channel ∈ {text, tools, combined, n/a (None verdict)}
    //   - spinning ∈ {0, 1} — was Layer-2 spinning detection also active
    pub static ref LOOP_DETECTOR_VERDICTS: IntCounterVec =
        register_int_counter_vec!(
            "atlas_loop_detector_verdicts_total",
            "Loop detector verdicts emitted, by verdict + channel + spinning flag",
            &["verdict", "channel", "spinning"]
        ).unwrap();

    // ── Anthropic translation-drift counter (P5.1) ──
    //
    // Increments whenever the Anthropic→OpenAI translator produces a
    // round-trip diff against the original Anthropic shape. Diffs
    // indicate translation bugs that compound across long agentic
    // sessions. Logging the actual diff is gated behind the
    // ATLAS_DEBUG_TRANSLATION_DRIFT env var (anthropic.rs).
    pub static ref ANTHROPIC_TRANSLATION_DRIFTS: IntCounter =
        register_int_counter!(
            "atlas_anthropic_translation_drifts_total",
            "Anthropic ↔ OpenAI translator round-trip mismatches detected"
        ).unwrap();

    // ── Speculative-decode telemetry (A.2 EASD scaffolding) ──
    //
    // Per-K acceptance counters. Enables measuring baseline accept
    // rates across MTP K-paths so we can decide whether EASD
    // activation (per-step D2H of verify logits + entropy gating,
    // arXiv:2512.23765) is worth its cost. EASD itself is gated
    // behind future activation once these baselines are measured.
    pub static ref SPEC_DECODE_VERIFY: IntCounterVec =
        register_int_counter_vec!(
            "atlas_spec_decode_verify_total",
            "MTP draft verify outcomes by K and result",
            &["k", "outcome"]
        ).unwrap();

    // ── Tool-call telemetry ──
    //
    // Total successful tool calls emitted by the API layer (sum across
    // streaming + blocking). Paired with the "Tool call: name(args)"
    // info log so operators can both grep logs and graph rates.
    // Unlabeled (no `name` label) — high-cardinality tool names would
    // blow up Prometheus cardinality.
    pub static ref TOOL_CALLS_TOTAL: IntCounter =
        register_int_counter!(
            "atlas_tool_calls_total",
            "Total successful tool calls emitted by the server"
        ).unwrap();

    // ── Model hot-swap telemetry (Task 6 §5) ──
    //
    // A swap costs ~6+ s and the triggering request pays it; these make
    // that observable in dashboards, not just in per-request headers.
    // `model` label cardinality is bounded by the boot-time registry
    // (a handful of preseeded models), so labeled counters are safe.
    pub static ref MODEL_SWAPS_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "atlas_model_swaps_total",
            "Completed model hot-swap loads, by target model",
            &["model"]
        ).unwrap();
    pub static ref MODEL_SWAP_SECONDS: Histogram = register_histogram!(
        "atlas_model_swap_seconds",
        "Wall time of a model hot-swap load (snapshot read + rebuild)",
        vec![1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0]
    )
    .unwrap();
    pub static ref MODEL_SWAP_READ_BYTES_TOTAL: IntCounter =
        register_int_counter!(
            "atlas_model_swap_read_bytes_total",
            "Snapshot payload bytes read across all hot-swap loads"
        ).unwrap();
    /// Seconds each model spent as the ACTIVE (resident) model. Attributed
    /// to the outgoing model when a swap/unload replaces it.
    pub static ref MODEL_RESIDENCY_SECONDS_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "atlas_model_residency_seconds_total",
            "Cumulative seconds each model was the resident (active) model",
            &["model"]
        ).unwrap();

    // ── Loop-watchdog stop counter (Plan.md 1.3) ──
    //
    // Every loop-watchdog kill surfaces to the client as
    // `finish_reason: "length"` — indistinguishable from a genuine
    // max_tokens truncation. This counter makes watchdog-caused
    // truncations diagnosable from /metrics without grepping logs.
    // detector ∈ {fuzzy, content_loop, simhash, line_loop}.
    pub static ref WATCHDOG_STOPS_TOTAL: IntCounterVec =
        register_int_counter_vec!(
            "atlas_watchdog_stops_total",
            "Generations hard-stopped by a loop watchdog, by detector",
            &["detector"]
        ).unwrap();
}

// ── NVMe tier + pressure scheduler metrics (TEL-6, PRI-6) ──
lazy_static::lazy_static! {
    /// Decode throughput per finished request (tokens/s).
    pub static ref TOKENS_PER_SECOND: prometheus::Histogram = prometheus::register_histogram!(
        "atlas_tokens_per_second",
        "Decode tokens per second per finished request",
        vec![1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
    )
    .unwrap();

    /// Queue wait before a model grant, per model and tier (PRI-6).
    pub static ref QUEUE_WAIT_SECONDS: prometheus::HistogramVec =
        prometheus::register_histogram_vec!(
            "atlas_queue_wait_seconds",
            "Arena queue wait until grant",
            &["model", "tier"],
            vec![0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0, 120.0, 300.0]
        )
        .unwrap();

    /// Live queue lengths per model and tier (PRI-6).
    pub static ref QUEUE_LEN: prometheus::IntGaugeVec = prometheus::register_int_gauge_vec!(
        "atlas_queue_len",
        "Arena queue length",
        &["model", "tier"]
    )
    .unwrap();

    /// HTTP 5xx responses by endpoint and status (feeds the 5xx-rate
    /// detector and the event store).
    pub static ref REQUEST_FAILURES_TOTAL: IntCounterVec = register_int_counter_vec!(
        "atlas_request_failures_total",
        "HTTP responses with status >= 500",
        &["endpoint", "status"]
    )
    .unwrap();

    /// Requests admitted with an estimate (budgeted or not), by model.
    pub static ref ADMISSION_ACCEPTED_TOTAL: IntCounterVec = register_int_counter_vec!(
        "atlas_admission_accepted_total",
        "Requests admitted by the completion-time estimate",
        &["model"]
    )
    .unwrap();

    /// Budgeted requests refused at admission (`accepted_processing_time`),
    /// by model and reason ∈ {capacity_exceeded, estimate_unavailable}.
    pub static ref ADMISSION_REJECTED_TOTAL: IntCounterVec = register_int_counter_vec!(
        "atlas_admission_rejected_total",
        "Requests refused by the completion-time admission estimate",
        &["model", "reason"]
    )
    .unwrap();

    /// Bytes handed to the NVMe block-store write queue (TEL-6).
    pub static ref NVME_WRITE_BYTES_TOTAL: IntCounter = register_int_counter!(
        "atlas_nvme_write_bytes_total",
        "Payload bytes enqueued to the NVMe block store"
    )
    .unwrap();

    /// Bytes read back from the NVMe block store (TEL-6).
    pub static ref NVME_READ_BYTES_TOTAL: IntCounter = register_int_counter!(
        "atlas_nvme_read_bytes_total",
        "Payload bytes restored from the NVMe block store"
    )
    .unwrap();

    /// Tokens whose KV entered the RAM cache from the NVMe store — the
    /// numerator of the restore hit rate (÷ atlas_prompt_tokens_total).
    pub static ref NVME_RESTORED_TOKENS_TOTAL: IntCounter = register_int_counter!(
        "atlas_nvme_restored_tokens_total",
        "Prompt tokens restored from the NVMe tier into the RAM cache"
    )
    .unwrap();
}

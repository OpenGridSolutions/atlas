// SPDX-License-Identifier: AGPL-3.0-only

//! `tracing` layer that turns ERROR events (all targets) and WARN events
//! from an allow-list of targets into observer [`Event`]s. Catches the
//! anomalies the codebase already logs — CUDA errors, crash snapshots,
//! KV leaks, kv-tier/NVMe failures, hot-swap load failures, host memory
//! warnings — without touching their call sites. Non-blocking: it only
//! formats the fields and calls [`super::emit`].

use tracing::field::{Field, Visit};
use tracing::{Event as TracingEvent, Level, Subscriber};
use tracing_subscriber::layer::{Context, Layer};

pub struct ObserveLayer;

/// WARN-level targets worth an event (prefix match).
const WARN_TARGETS: &[&str] = &[
    "spark::kv_tier",
    "spark_storage",
    "mem",
    "hotswap",
    "spark::api::convert",
    "spark::scheduler::hotswap_ctl",
    "spark::arena_sched",
];

/// Map a log target onto an event kind and the push priority.
pub fn classify(level: &Level, target: &str, message: &str) -> Option<(&'static str, u8)> {
    if *level == Level::ERROR {
        return Some(match target {
            t if t.starts_with("cuda_err") => ("cuda_error", 5),
            "proc" => ("crash_snapshot", 5),
            t if t.starts_with("memdebug") => ("kv_leak", 4),
            t if t.contains("hotswap") => ("hotswap_error", 5),
            t if t.contains("kv_tier") || t.contains("spark_storage") => ("nvme_tier_error", 4),
            _ if message.contains("OOM watchdog") => ("oom_watchdog", 5),
            _ => ("log_error", 4),
        });
    }
    if *level == Level::WARN && WARN_TARGETS.iter().any(|t| target.starts_with(t)) {
        return Some(match target {
            t if t.contains("kv_tier") || t.contains("spark_storage") => ("nvme_tier_warn", 3),
            "mem" => ("host_memory", 3),
            t if t.contains("hotswap") => ("hotswap_warn", 3),
            t if t.contains("convert") => ("convert_warn", 2),
            _ => ("log_warn", 2),
        });
    }
    None
}

#[derive(Default)]
struct FieldCollector {
    message: String,
    fields: serde_json::Map<String, serde_json::Value>,
}

impl Visit for FieldCollector {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let s = format!("{value:?}");
        if field.name() == "message" {
            self.message = s;
        } else {
            self.fields
                .insert(field.name().to_string(), serde_json::Value::String(s));
        }
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        } else {
            self.fields.insert(
                field.name().to_string(),
                serde_json::Value::String(value.to_string()),
            );
        }
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        self.fields
            .insert(field.name().to_string(), serde_json::json!(value));
    }
    fn record_u64(&mut self, field: &Field, value: u64) {
        self.fields
            .insert(field.name().to_string(), serde_json::json!(value));
    }
    fn record_f64(&mut self, field: &Field, value: f64) {
        self.fields
            .insert(field.name().to_string(), serde_json::json!(value));
    }
    fn record_bool(&mut self, field: &Field, value: bool) {
        self.fields
            .insert(field.name().to_string(), serde_json::json!(value));
    }
}

impl<S: Subscriber> Layer<S> for ObserveLayer {
    fn on_event(&self, event: &TracingEvent<'_>, _ctx: Context<'_, S>) {
        let meta = event.metadata();
        let level = meta.level();
        if *level != Level::ERROR && *level != Level::WARN {
            return;
        }
        // Our own warnings must not feed back into the queue.
        if meta.target().starts_with("observe") {
            return;
        }
        let mut fc = FieldCollector::default();
        event.record(&mut fc);
        let Some((kind, priority)) = classify(level, meta.target(), &fc.message) else {
            return;
        };
        let model = fc
            .fields
            .get("model")
            .or_else(|| fc.fields.get("to"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let mut detail = fc.fields;
        detail.insert(
            "target".into(),
            serde_json::Value::String(meta.target().to_string()),
        );
        let mut ev =
            super::Event::new(priority, kind, fc.message).detail(serde_json::Value::Object(detail));
        ev.model = model;
        super::emit(ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification() {
        assert_eq!(
            classify(&Level::ERROR, "cuda_err", "illegal address"),
            Some(("cuda_error", 5))
        );
        assert_eq!(
            classify(&Level::ERROR, "proc", "crash"),
            Some(("crash_snapshot", 5))
        );
        assert_eq!(
            classify(&Level::ERROR, "memdebug", "KV LEAK"),
            Some(("kv_leak", 4))
        );
        assert_eq!(
            classify(
                &Level::ERROR,
                "spark_runtime::cuda_backend",
                "OOM watchdog: ..."
            ),
            Some(("oom_watchdog", 5))
        );
        assert_eq!(
            classify(&Level::ERROR, "anything", "x"),
            Some(("log_error", 4))
        );
        assert_eq!(
            classify(&Level::WARN, "spark::kv_tier", "export failed"),
            Some(("nvme_tier_warn", 3))
        );
        assert_eq!(
            classify(&Level::WARN, "mem", "low"),
            Some(("host_memory", 3))
        );
        assert_eq!(
            classify(&Level::WARN, "spark::api::chat", "x"),
            None,
            "not allow-listed"
        );
        assert_eq!(classify(&Level::INFO, "cuda_err", "x"), None);
    }
}

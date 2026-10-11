// SPDX-License-Identifier: AGPL-3.0-only

//! Crash forensics (Todo.md Phase 1.6): a last memory snapshot written to
//! a plain file the moment the process is about to die — panic hook and
//! OOM-watchdog exit both call in here. File-based by design: when the
//! process is gone, the log file is all that's left to diagnose cases
//! A1/A2/E1 of the instability catalog.

use std::io::Write;
use std::path::PathBuf;

/// Log directory shared with the file-logging sinks: `ATLAS_LOG_DIR`,
/// default `logs` (relative to the working directory, matching the
/// existing `logs/serve-*.log` convention).
pub fn log_dir() -> PathBuf {
    PathBuf::from(std::env::var("ATLAS_LOG_DIR").unwrap_or_else(|_| "logs".into()))
}

/// Compact UTC timestamp `yyyymmdd-hhmmss` without a chrono dependency
/// (civil-from-days, Howard Hinnant's algorithm).
pub fn timestamp_compact() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let (h, m, s) = (secs / 3600 % 24, secs / 60 % 60, secs % 60);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}{mo:02}{d:02}-{h:02}{m:02}{s:02}")
}

fn rss_mib() -> i64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("VmRSS:"))
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|kb| kb.parse::<i64>().ok())
        })
        .map_or(-1, |kb| kb / 1024)
}

type FatalHook = Box<dyn Fn(&str, &str) + Send + Sync>;
static FATAL_HOOK: std::sync::OnceLock<FatalHook> = std::sync::OnceLock::new();

/// Register a synchronous last-words hook, called from
/// [`write_crash_snapshot`] before the file is written — the server uses
/// it to push a notification while the process still exists (the panic
/// hook and the OOM watchdog both end in an exit/abort, so nothing async
/// would ever flush). Only the first registration takes effect.
pub fn set_fatal_hook(hook: FatalHook) {
    let _ = FATAL_HOOK.set(hook);
}

/// Write `logs/crash-<ts>.log` with the last meminfo/CUDA/KV-pool state.
/// Infallible by contract (best effort — we are already dying); returns
/// the path when the file was written. Also mirrors the snapshot as a
/// `tracing::error!` on target `proc` so it lands in critical.log when
/// the subscriber is still alive.
pub fn write_crash_snapshot(kind: &str, detail: &str) -> Option<PathBuf> {
    let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
    let mem = crate::meminfo::read();
    let cuda = crate::cuda_backend::cuda_mem_get_info_raw();
    let pool = crate::kv_stats::snapshot();
    let budget = crate::kv_stats::budget_decision();
    let rss = rss_mib();

    let mut body = format!(
        "kind: {kind}\ntime_utc: {}\ndetail: {detail}\n",
        timestamp_compact()
    );
    if let Some(m) = mem {
        body += &format!(
            "meminfo: available={:.2} GiB, swap_free={:.2} GiB, total={:.2} GiB, hugepages={}\n",
            gib(m.mem_available),
            gib(m.swap_free),
            gib(m.mem_total),
            m.hugepages
        );
    }
    if let Some((free, total)) = cuda {
        body += &format!(
            "cuda: free={:.2} GiB, total={:.2} GiB (raw cuMemGetInfo)\n",
            gib(free),
            gib(total)
        );
    }
    if let Some(p) = pool {
        body += &format!(
            "kv_pool: used={}/{} blocks, reserved={:.2} GiB\n",
            p.used_blocks,
            p.total_blocks,
            gib(p.reserved_bytes)
        );
    }
    if let Some(d) = budget {
        body += &format!(
            "kv_budget_last: source={}, used_so_far={:.2} GiB, total_budget={:.2} GiB, \
             baseline={:.2} GiB, cotenants={:.2} GiB\n",
            d.source,
            gib(d.used_so_far),
            gib(d.total_budget),
            gib(d.baseline_free),
            gib(d.cotenants)
        );
    }
    body += &format!("rss: {rss} MiB\n");

    tracing::error!(target: "proc", kind, detail, "crash snapshot:\n{body}");
    if let Some(hook) = FATAL_HOOK.get() {
        hook(kind, &format!("{detail}\n{body}"));
    }

    let dir = log_dir();
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join(format!("crash-{}.log", timestamp_compact()));
    let mut f = std::fs::File::create(&path).ok()?;
    f.write_all(body.as_bytes()).ok()?;
    Some(path)
}

#[cfg(test)]
mod tests {
    #[test]
    fn timestamp_is_sane() {
        let ts = super::timestamp_compact();
        assert_eq!(ts.len(), 15, "{ts}");
        // Year 2026+ and a '-' separator at the fixed position.
        assert!(ts[..4].parse::<u32>().unwrap() >= 2026, "{ts}");
        assert_eq!(&ts[8..9], "-");
    }
}

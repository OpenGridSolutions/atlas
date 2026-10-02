// SPDX-License-Identifier: AGPL-3.0-only

//! Central CUDA-error reporting (Todo.md Phase 1.4, catalog case D1/D2).
//!
//! Every kernel launch flows through `GpuBackend::launch` and every async
//! kernel error surfaces at a synchronize — instrumenting those two choke
//! points covers all model kernels without touching call sites. Errors
//! log on the dedicated target `cuda_err` (routed into critical.log by
//! the WARN+ file sink) with the kernel NAME resolved from the handle
//! registry that `GpuBackend::kernel` fills at model-init time.
//!
//! Sticky detection: device-side asserts, illegal addresses and friends
//! poison the CUDA context — every later call fails until restart. The
//! flag is exposed to /health so orchestrators see 503 instead of an
//! endless stream of per-request 500s. Cleared on hot-swap unload? NO —
//! a poisoned primary context does not heal by unloading a model; only
//! a process restart clears it.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

static KERNEL_NAMES: Mutex<Option<HashMap<u64, String>>> = Mutex::new(None);
static STICKY: AtomicBool = AtomicBool::new(false);
static STICKY_DETAIL: Mutex<Option<String>> = Mutex::new(None);

/// Called from `GpuBackend::kernel` for every resolved kernel so launch
/// errors can name the kernel instead of printing a raw handle.
pub fn register_kernel_name(handle: u64, module: &str, func: &str) {
    let mut g = KERNEL_NAMES.lock().expect("KERNEL_NAMES poisoned");
    g.get_or_insert_with(HashMap::new)
        .insert(handle, format!("{module}::{func}"));
}

pub fn kernel_name(handle: u64) -> String {
    KERNEL_NAMES
        .lock()
        .expect("KERNEL_NAMES poisoned")
        .as_ref()
        .and_then(|m| m.get(&handle).cloned())
        .unwrap_or_else(|| format!("<unregistered {handle:#x}>"))
}

/// CUDA driver statuses that poison the context (sticky errors): later
/// calls on the same context keep failing until process restart.
/// 700 ILLEGAL_ADDRESS, 710 ASSERT, 713 HARDWARE_STACK_ERROR,
/// 714 ILLEGAL_INSTRUCTION, 715 MISALIGNED_ADDRESS,
/// 716 INVALID_ADDRESS_SPACE, 719 LAUNCH_FAILED, 999 UNKNOWN.
fn is_sticky_status(detail: &str) -> bool {
    ["700", "710", "713", "714", "715", "716", "719", "999"]
        .iter()
        .any(|code| {
            detail.contains(&format!("status {code}")) || detail.contains(&format!("error {code}"))
        })
        || detail.contains("ILLEGAL")
        || detail.contains("device-side assert")
        || detail.contains("LAUNCH_FAILED")
}

/// Report a CUDA failure on target `cuda_err`. `op` names the choke
/// point ("launch", "stream_sync", "ctx_sync", "graph_launch"),
/// `what` the kernel/graph, `detail` the driver error text.
pub fn report(op: &str, what: &str, detail: &str) {
    let sticky = is_sticky_status(detail);
    if sticky && !STICKY.swap(true, Ordering::Relaxed) {
        *STICKY_DETAIL.lock().expect("STICKY_DETAIL poisoned") =
            Some(format!("{op} {what}: {detail}"));
        tracing::error!(
            target: "cuda_err",
            op,
            what,
            detail,
            "STICKY CUDA error — context is poisoned, every later call will fail; \
             /health now reports 503; restart the server"
        );
    } else {
        tracing::error!(target: "cuda_err", op, what, detail, sticky, "CUDA error");
    }
}

/// `Some(first sticky error)` once the context is poisoned; /health
/// turns this into a 503.
pub fn sticky() -> Option<String> {
    if STICKY.load(Ordering::Relaxed) {
        STICKY_DETAIL
            .lock()
            .expect("STICKY_DETAIL poisoned")
            .clone()
            .or_else(|| Some("sticky CUDA error".into()))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::is_sticky_status;

    #[test]
    fn sticky_classification() {
        assert!(is_sticky_status("cuStreamSynchronize failed: status 700"));
        assert!(is_sticky_status(
            "Kernel launch failed: CUDA_ERROR_ILLEGAL_ADDRESS"
        ));
        assert!(!is_sticky_status("cuMemAlloc failed: status 2")); // OOM: not sticky
        assert!(!is_sticky_status("status 1 invalid value"));
    }
}

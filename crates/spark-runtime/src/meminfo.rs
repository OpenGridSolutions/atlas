// SPDX-License-Identifier: AGPL-3.0-only

//! `/proc/meminfo` reader (Plan.md Phase S).
//!
//! On DGX Spark's unified memory, `cudaMemGetInfo` under-reports free
//! memory because swap-reclaimable DRAM is invisible to it (NVIDIA DGX
//! Spark known issue). The truly allocatable amount is
//! `MemAvailable + SwapFree` — with a HugePages special case: when
//! HugePages are configured (`HugePages_Total != 0`), allocatable is
//! `HugePages_Free × Hugepagesize` and swap does not apply.
//!
//! Phase S uses this READ-ONLY for the /v1/stats diagnosis (the
//! `discrepancy_gb` success metric); the SEPARATE follow-up KV-budget
//! fix will call the same reader from `spark_model::factory::build`.

/// Parsed snapshot of the /proc/meminfo fields Phase S cares about.
/// All values in BYTES.
#[derive(Debug, Clone, Copy)]
pub struct MemInfo {
    pub mem_total: usize,
    pub mem_available: usize,
    pub swap_free: usize,
    /// True when the HugePages special case applied.
    pub hugepages: bool,
    /// The allocatable amount under the Phase-S rule:
    /// `MemAvailable + SwapFree`, or `HugePages_Free × Hugepagesize`
    /// when HugePages are configured.
    pub available_for_alloc: usize,
}

/// Read and parse /proc/meminfo. `None` on non-Linux or parse failure.
pub fn read() -> Option<MemInfo> {
    parse(&std::fs::read_to_string("/proc/meminfo").ok()?)
}

fn parse(contents: &str) -> Option<MemInfo> {
    // /proc/meminfo values are in kB except the HugePages_* counts.
    let mut mem_total = None;
    let mut mem_available = None;
    let mut swap_free = None;
    let mut hugepages_total = 0usize;
    let mut hugepages_free = 0usize;
    let mut hugepagesize_bytes = 0usize;
    for line in contents.lines() {
        let mut it = line.split_whitespace();
        let (Some(key), Some(val)) = (it.next(), it.next()) else {
            continue;
        };
        let Ok(n) = val.parse::<usize>() else { continue };
        match key {
            "MemTotal:" => mem_total = Some(n * 1024),
            "MemAvailable:" => mem_available = Some(n * 1024),
            "SwapFree:" => swap_free = Some(n * 1024),
            "HugePages_Total:" => hugepages_total = n, // plain count
            "HugePages_Free:" => hugepages_free = n,   // plain count
            "Hugepagesize:" => hugepagesize_bytes = n * 1024,
            _ => {}
        }
    }
    let mem_total = mem_total?;
    let mem_available = mem_available?;
    let swap_free = swap_free.unwrap_or(0);
    let hugepages = hugepages_total != 0;
    let available_for_alloc = if hugepages {
        hugepages_free * hugepagesize_bytes
    } else {
        mem_available + swap_free
    };
    Some(MemInfo {
        mem_total,
        mem_available,
        swap_free: if hugepages { 0 } else { swap_free },
        hugepages,
        available_for_alloc,
    })
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn standard_meminfo_sums_available_and_swap() {
        let m = parse(
            "MemTotal:       125000000 kB\n\
             MemFree:         2000000 kB\n\
             MemAvailable:   60000000 kB\n\
             SwapFree:       30000000 kB\n\
             HugePages_Total:       0\n\
             HugePages_Free:        0\n\
             Hugepagesize:       2048 kB\n",
        )
        .unwrap();
        assert!(!m.hugepages);
        assert_eq!(m.available_for_alloc, (60_000_000 + 30_000_000) * 1024);
        assert_eq!(m.mem_total, 125_000_000 * 1024);
    }

    #[test]
    fn hugepages_override_ignores_swap() {
        let m = parse(
            "MemTotal:       125000000 kB\n\
             MemAvailable:   60000000 kB\n\
             SwapFree:       30000000 kB\n\
             HugePages_Total:    1000\n\
             HugePages_Free:      600\n\
             Hugepagesize:       2048 kB\n",
        )
        .unwrap();
        assert!(m.hugepages);
        assert_eq!(m.swap_free, 0);
        assert_eq!(m.available_for_alloc, 600 * 2048 * 1024);
    }
}

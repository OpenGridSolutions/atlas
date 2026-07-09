# Task 6 — Decisions (Lazy-Loading Multi-Model Engine)

> Continuation of HOTSWAP_MVP_PLAN.md; see TASK6_MULTIMODEL_PLAN.md for the
> task list and docs/hotswap/trace.md for the load-path analysis.

## D1: Swap model = synchronous SSD lazy-load (chosen)

**Decision:** A model change is served by a **synchronous** unload → reload
from the ATLASRAW snapshot on NVMe. The request that triggers the swap
blocks until the new model is ready.

**Measured cost (on-box, GB10):** same-model reload of the 47 GiB
Qwen3-Next-80B snapshot took **6.3 s @ 8.25 GiB/s** (sequential O_DIRECT
fallback; io_uring may improve this on the host — check the reload log
line for which path ran). So a swap-triggering request sees **TTFT > 6 s**
plus model rebuild time. That is the price of this mode, and it is paid on
*every* model change, in both directions.

**Why accepted anyway:**

- **N models, one memory budget.** The arena holds exactly one model
  (~50 GiB); snapshots on SSD are cheap (~47 GiB each, NVMe). The design
  scales to any number of preseeded models without further RAM.
- **Correctness surface stays small.** One model in the scheduler, one set
  of masks, one tokenizer — the Phase 1 seams swap them atomically. No
  double bookkeeping of two live models.
- **Observability over surprise:** the swap is surfaced, not hidden — an
  `X-Atlas-Model-Swap: <ms>` response header plus a log line whenever a
  request triggered a swap (plan §5).

**Rejected alternative — dual-residency pointer-flip:** if only *two*
models are ever hot, keeping both resident (two arenas, swap = pointer
flip) would cut the switch to < 100 ms with no I/O at all. Rejected here
because it is explicitly out of scope for Task 6 (plan §5), doubles the
weight footprint (2 × ~50 GiB leaves too little for KV/SSM pools on a
128 GB box for the 80B-class pair), and the seams built in Phase 1 are a
prerequisite for it anyway — it can be layered on later without undoing
this work.

**Consequences to keep in mind:**

- Interleaved A/B traffic thrashes: every alternation costs ~6 s. Clients
  that mix models should batch by model. Phase 3.2 defines what concurrent
  mixed traffic does (wait on in-flight load; third model → 503).
- The 900 s admin-load timeout and the 2 s drain grace both sit in front of
  the 6 s payload read; worst-case swap latency is drain + read + rebuild.
- Unregistered models are never lazily initialized from safetensors at
  request time — 404 with the available-model list (plan Phase 3.1).

## D2: Migration peak fix = streaming per-shard copy+free (0.1)

Boot migration streams in shards of ≤ 1 GiB / ≤ 8192 ops: enqueue the
shard's copies (round-robin over 4 CUDA streams), sync, free that shard's
source allocations immediately. Peak device footprint ≈ weights + one
shard instead of 2× weights (~94 GiB), which is what made dirty-memory
boots fail and blocked preseed (N back-to-back loads).

**On-box correction:** source ranges are **never coalesced** across
tensors, even when VA-adjacent — every tensor is its own
`cuMemAlloc`/`cuMemAllocManaged` allocation and `cuMemcpyDtoDAsync`
rejects a range spanning an allocation boundary (`CUDA_ERROR_INVALID_VALUE`,
observed as a boot crash on the first merged copy). If migration time
still exceeds the 5 s acceptance bound with per-tensor copies, the next
lever is `cuMemcpyBatchAsync` (CUDA ≥ 12.8), not range merging.

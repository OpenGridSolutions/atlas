# HOTSWAP_MVP_PLAN.md — Working Brief for Claude Code (v1.1, post-trace)

> v1.1 incorporates the findings of `docs/hotswap/trace.md` (Task 1). Changes vs v1.0
> are tagged **[v1.1]**. Work through tasks in order; acceptance criteria gate each.

## 1. Context & Goal

Target: NVIDIA DGX Spark (GB10, 128 GB coherent UMA, aarch64, single GPU, NVMe).
CPU and GPU are cache-coherent via NVLink-C2C + ATS.

**MVP goal:** snapshot → unload → reload of the **same model**
(`nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4`) over admin API, no process restart.

**[v1.1] Scope guard:** the MVP is **same-model reload only**. Loading a *different*
model is a separate follow-up: `AppState` (tokenizer, chat template, sampling
defaults) is immutable by design, the scheduler owns the model exclusively with no
unload path, and three `OnceLock` sampling masks in `scheduler/helpers.rs:378-439`
are write-once with a **silent no-op** on second set — a different-model load today
would run with the wrong masks (silent wrong sampling). Follow-up requires: ArcSwap
seams for AppState + masks, and a model-handover channel to the scheduler thread.

Dual-residency remains the follow-up after that; keep arena count parameterized
(also needed for the `--dflash` drafter's second `WeightStore`).

## 2. Hard Invariants

- Arena never freed/shrunk during process lifetime; `arena_base` constant.
- Per-tensor pointers are `arena_base + offset` views. **[v1.1]** Verified safe:
  kernels receive raw `u64` (`gpu.rs:155`), the KV cache already uses interior
  pointers (`paged_impl.rs`), and nothing in the workspace introspects pointer
  attributes. Zero changes needed in spark-model's ~70–80 read sites.
- **[v1.1] Alignment is a correctness requirement:** no code asserts weight-pointer
  alignment; kernels do vectorised loads; misalignment = silent wrong results.
  `arena.alloc(len, align=256)` MUST `debug_assert!(offset % 256 == 0)`.
- **[v1.1] Deterministic layout:** migrate in tensor-**name-sorted** order (HashMap
  iteration is nondeterministic; the snapshot's `model_version_hash` covers layout).
- No `cuModuleUnload`; PTX modules persist.
- Payload I/O via io_uring + O_DIRECT (4096-aligned); footer/manifest via buffered I/O.
- **[v1.1] Metrics caveat:** `cuMemGetInfo` / `nvidia-smi memory.used` do NOT count
  `cuMemAllocManaged` pages until migration and are unreliable on this box.
  Acceptance metrics = uvm slab canary (`uvm_va_range_external_t` count),
  tokens/s + temperature-0 output equality — never device-memory accounting.

## 3. Platform Facts (verified)

- Pre-flight false-negative after restart (NVIDIA known issue) — Task 0 escape landed.
- Loader allocates one `cuMemAlloc_v2` **per tensor** (~43,900 uvm ranges for 297,728
  tensors); on first alloc failure it flips permanently to `cuMemAllocManaged`
  (`fast_weights/mod.rs:359-373`). Both read strategies (O_DIRECT vs buffered,
  >5000-tensor cap) share this single allocation site.
- `WeightTensor { ptr: DevicePtr(u64), shape, dtype }` — no Drop anywhere; today's
  weight allocations are never freed (leak by design). Task 2's migration must ADD
  the explicit free loop; `cuMemFree_v2` deallocates both alloc kinds.
- spark-runtime does NOT use cudarc for memory/streams/events — it has its own
  `extern "C"` block (`cuda_backend.rs:18-69`) with streams, events,
  `cuStreamWaitEvent`, async memcpys, memset, graph capture already declared.
  **Extend that block; no `vendor/cudarc` edits needed.**

## 4. Known Code Anchors (from trace)

| What | Where |
|---|---|
| Migration seam | between `serve.rs:210` (`load_weight_store`) and `serve.rs:351` (`build_model`) — model copies ptrs **by value**, so migrate strictly before build |
| Chokepoint accessor | `weight_map/model_a.rs:26` `ptr(store, name)` |
| Loader alloc site (both paths) | `fast_weights/mod.rs:354-389` |
| UVM fallback log line (one-shot) | `fast_weights/mod.rs:366` — "switching to managed" |
| OOM-guard skip bug | `fast_weights/mod.rs:230` — guard disabled once `offload_logged` |
| Interior-pointer precedent | `kv_cache/paged_impl.rs` (slabs + offsets + free list + memset reset) |
| BufferArena — imitate API shape only (NOT a bump allocator) | `buffers.rs:116-171` |
| 36 `gpu.free` sites on store ptrs | `nvfp4_detect.rs:276` (hard-fail `?`, Bf16Raw arm), `ssm_qwen35.rs:286-289`, `layers/moe/helpers_a.rs:208-265`, `weight_loader/nemotron.rs:169-278`, `qwen35/.../linear_attn_arms.rs:230-231` — **none fire for Qwen3-Next-NVFP4** (gates verified in trace §4.1) |
| Derived-buffer alloc/free counts in build | 178 `gpu.alloc` / 104 `gpu.free` in spark-model |
| AppState / scheduler ownership | `app_state.rs:17-97`, `serve.rs:564-607`, `scheduler/mod.rs:110-134` |
| Router (no /admin today; auth middleware is global) | `serve_router.rs:44-130` |
| cufile-sys (fallback only) | `spark-storage` bench/probe path |

## 5. Tasks

### Task 0 — DONE. Pre-flight escape landed and verified on-box.

### **[v1.1] Task 0b — Fix the OOM-guard skip bug (small, do before next dirty boot)**
`fast_weights/mod.rs:230`: `check_oom_guard` runs only `if !offload_logged`, so the
per-shard guard is disabled for all remaining shards after the first managed
fallback — precisely when it matters. Fix: run the guard unconditionally per shard
(or invert the condition). Also correct the too-strong comments added in Task 0
(`weights.rs`, `serve.sh`) claiming the guard "remains active".
**Accept:** guard log/check visibly fires per shard in a constrained-memory run.

### **[v1.1] Task 1.5 — The load-bearing spike (BEFORE any weights_arena.rs code)**
Nothing in the repo exercises GPU reads of anonymous mmap. Test the single claim the
arena rests on:
```
mmap 1 GiB anonymous (4096-aligned) → write pattern from host
→ pass host VA as kernel arg to an existing elementwise kernel
→ copy_d2h and verify; verify kernel writes visible to host
→ repeat with cuMemHostRegister_v2; note any difference
→ boot-time assertion candidate: cuPointerGetAttribute on the VA
```
Fallback ladder if it fails: `cuMemAllocManaged` arena → VMM
(`cuMemCreate/cuMemMap`) → cuFile. Record results in `docs/hotswap/findings.md`.
**Accept:** one path proven correct both directions + bandwidth number; decision
recorded.

### Task 2 — Arena + post-load migration
New `crates/spark-runtime/src/weights_arena.rs`:
- General **multi-instance bump allocator** (count parameterized): backing chosen by
  Task 1.5 result; `alloc(len, 256)` with the debug-assert; `reset()`; `owns(ptr)`.
- **[v1.1] Backend additions to `cuda_backend.rs` extern block + `GpuBackend` trait
  (incl. `gpu/mock.rs`):** `cuCtxSynchronize` → `device_synchronize()` (blocking gap);
  `cuMemHostRegister_v2`/`cuMemHostUnregister` behind `arena_host_register` flag;
  `cuPointerGetAttribute` for a boot-time ATS assertion.
- **[v1.1] Free-guard:** route `GpuBackend::free` through `arena.owns(ptr)` → no-op +
  `tracing::debug!` for arena-backed pointers. (MVP model verified unaffected by the
  36 sites, but the guard is mandatory before ANY other model runs through the arena.
  Follow-up issue: debug assertion + explicit call-site fixes.)
- `WeightStore::iter_mut_sorted()` accessor (field is private today).
- Migration at the seam: name-sorted copy_d2d into arena, record manifest entries,
  re-point store ptrs, `device_synchronize()`, then explicit `gpu.free` loop over
  old pointers (works for both alloc kinds).
- Flag `--weights-arena-gb` (default 50).

**Accept:** identical temperature-0 output vs baseline; uvm slab canary collapses
from ~43,900 to a handful; tokens/s within 5%; boot-time ATS assertion passes.

### Task 3 — Snapshot format + `POST /admin/models/snapshot`
As v1.0 (4096-byte header, O_DIRECT payload write, name-sorted layout, manifest,
12-byte footer trailer `manifest_len u32 · manifest_xxh3_lo u32 · magic "AMFT"`,
`format_version = 1` hard-rejected if unknown). **[v1.1]** Mount an `/admin` router
on the existing router so global auth middleware covers it automatically
(`serve_router.rs:124`).

### Task 4 — `POST /admin/models/unload`
As v1.0 (drain 2000 ms → 503 + Retry-After:1 → finish_reason "model_swap" → KV/SSM
pool reset, prefix-cache invalidation → drop model + store index → `arena.reset()`
→ `device_synchronize()`), **plus:**
- **[v1.1] CRITICAL — derived-buffer registry.** `build_model` makes **178
  `gpu.alloc`** calls (transposes, dequant, predequant, re-quant, LM-head, …) ≈
  **14 GB per cycle** (the 1.3× overhead), never freed. Without handling, the
  100-cycle test OOMs by ~cycle 7. MVP decision: **watermark registry** in
  `GpuBackend` — record every alloc after a watermark set post-migration; `unload`
  walks and frees the registry (guard skips arena-backed ptrs). Second-arena routing
  is the dual-residency-era follow-up (arena is already multi-instance-ready).
- `model_ready: Arc<AtomicBool>` (`app_state.rs:63`) is the existing not-ready gate —
  reuse it; inference handlers already read it (`api/misc_handlers.rs:132`).

**Accept:** post-unload inference → 503; arena mapping unchanged; uvm canary + RSS
stable; registry frees leave `MemFree` roughly restored.

### Task 5 — `POST /admin/models/load` (same model)
As v1.0 (resolve snapshot, footer→manifest buffered, io_uring O_DIRECT payload read
into `arena_base`, xxh3 verify, rebuild WeightStore views from manifest, rebuild
model without PTX reload, KV pool from static `--kv-pool-gb`, open scheduler),
**plus [v1.1]:** rebuilding the model re-runs `build_model` → reset the watermark
registry before, so the fresh derived buffers are tracked for the next unload.
Tokenizer/template/masks are untouched (same model).
**Accept:** boot-load → snapshot → unload → load; temperature-0 identity; 100-cycle
loop: constant `arena_base`, constant uvm canary, `MemFree` drift < 1 GB, no FD leaks.

### Task 6 — Test & bench harness
As v1.0 (cycle script, `snapshot-verify` CLI, `examples/arena_io_probe.rs` io_uring
bandwidth probe QD 8/16/32). **[v1.1]** Add the Task 1.5 ATS probe as a permanent
example; assert acceptance via canary/tokens-s, not `cuMemGetInfo`.

## 6. Follow-ups Logged (out of MVP scope)
1. Different-model load: ArcSwap seams for AppState + OnceLock masks (silent-no-op
   → wrong-sampling hazard), scheduler model-handover channel.
2. Free-site hardening (trace §4.1 option 2): debug assertion + explicit site fixes —
   prerequisite for MiniMax-M2 / step3p7 / Nemotron / Qwen3.5 through the arena.
3. Dual-residency budget correction: per-model cost is ~47 GB store **+ ~14 GB
   derived** ≈ 61 GB → two resident models exceed budget unless (a) dead store
   originals (superseded by transposed/predequant copies) are freed, or (b) the
   snapshot stores the **derived** forms instead of raw. Investigate (b) — it may
   also cut reload work. VMM (`cuMemCreate/cuMemMap`) is the right primitive for
   pointer-stable rebacking in this era.
4. `serve.sh` logging via `tee` to `/workspace/logs/` (done ad hoc; keep).
5. Confirm on next boot via `grep -m1 "switching to managed"` whether the last run
   was managed-fallback or pure-device (decides how seriously to treat managed-path
   performance as the arena fallback).

## 7. Working Agreements
Unchanged from v1.0: release builds via `cargo build --release -p spark-server`;
small commits prefixed `hotswap:` on `modifications`; STOP + `findings.md` on any
platform-assumption failure; additive-only cudarc/extern changes.

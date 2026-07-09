# TASK6_MULTIMODEL_PLAN.md — Lazy-Loading Multi-Model Engine

> Working brief for Claude Code, continuing HOTSWAP_MVP_PLAN.md (v1.1/v1.2).
> Place in repo root. Work phase by phase, in order; each phase has an acceptance
> gate that MUST pass before the next phase starts. One commit per numbered step,
> `hotswap:` prefix, branch `modifications`.

## 0. Status this builds on (verified on-box)

- MVP Tasks 2–5 done: arena (mmap-backed, ATS self-test OK), snapshot (ATLASRAW,
  format_version=1), unload (drain + watermark registry), **same-model** load.
- Manual API test PASSED: snapshot→unload→load, reload **6.3 s @ 8.25 GiB/s**,
  temperature-0 output bit-identical incl. CoT.
- **Two open items that block this task — see Phase 0.**

## 1. The correctness blocker you MUST NOT ignore

This task requires **different-model** swaps (Thinking ↔ Coding ↔ …). The MVP was
accepted for **same-model reload only**. Trace §7 established, and it is still true:

- `AppState` (tokenizer, chat template, sampling defaults) is **immutable** — `Arc`,
  no `RwLock`/`ArcSwap`. A different model needs a different tokenizer/template.
- The three sampling masks in `scheduler/helpers.rs:378-439` are **`OnceLock` with a
  silent no-op on second `set()`**. Loading model B keeps model A's masks →
  **silently wrong sampling**, no error, no crash.
- The scheduler **owns the model exclusively** in its thread; there is no handover
  channel for a new `Box<dyn Model>`.

**Therefore different-model load is UNSAFE until the three seams in Phase 1 exist.**
Do not wire lazy-loading (Phase 3) before Phase 1's A→B→A gate passes. A green
same-model test does NOT prove different-model correctness.

## Phase 0 — Prerequisites (blockers; do first)

### 0.1 Streaming arena migration (open regression)
First real boot showed: migration copies ALL tensors, frees originals only after →
transient peak ≈ 2× weights (~94 GiB). On a dirty-memory start this leaves too little
for SSM pools/scratch and the boot fails ("only 2.53 GB remains but 4.65 GB needed").
Also migration took **73 s @ 0.65 GiB/s** vs the probe's 55 GiB/s — per-tensor
`cuMemcpyDtoD` launch overhead across 297,728 tiny copies (~166 KB avg), worsened by
managed-memory sources on dirty start.
**Fix:** interleave copy+free per shard (bound peak at weights + one shard); batch
copies (coalesce adjacent source ranges per shard) and/or queue async on one stream
with a single synchronize.
**Accept:** clean boot succeeds **without** a host `drop_caches` from a dirty start;
migration < 5 s; uvm canary collapses to a handful post-migration.
**This blocks preseed** — preseed loads N models back to back; the 2× peak makes
model 2 fail otherwise.

### 0.2 Decide and record the swap model
Lazy-load-from-SSD (this plan) costs ~6.3 s **synchronously** per model change: a
request that triggers a swap sees TTFT > 6 s. Document this in
`docs/hotswap/task6-decisions.md`. If only two models are ever hot, dual-residency
(both resident, pointer-flip < 100 ms) is better — but out of scope here; this plan
implements SSD lazy-load for N models. Note the tradeoff so it is a conscious choice.

## Phase 1 — Different-model safety seams (the real work)

Implement the three seams from trace §7. Nothing else in this task is safe without
them.

### 1.1 Swappable AppState
Wrap the per-model CPU state so it can be replaced atomically at swap time:
tokenizer (`ChatTokenizer`, incl. its `minijinja` chat template), sampling defaults,
think/EOS token ids, `sampling_presets` IF they are per-model (verify:
`sampling_presets` is per-kernel-target today, not per-model — confirm before moving).
Use `arc_swap::ArcSwap<ModelRuntime>` (add `arc-swap` to the workspace) or an
`Arc<RwLock<...>>` if the read path can tolerate it (it is hot — prefer ArcSwap).
Handlers read a consistent snapshot per request.

### 1.2 Fix the OnceLock mask hazard
The three masks (numeric/boundary/mid-word) and watchdog params in
`scheduler/helpers.rs` are write-once. Convert the **model-dependent** ones to
`ArcSwap<Arc<[bool]>>` (or per-scheduler state) so a swap installs new masks. Leave
genuinely static ones as-is. **A second install must take effect, not be dropped.**

### 1.3 Scheduler model handover
Add a scheduler control command (through the existing mutex+condvar queue used by
unload) that hands the thread a new `Box<dyn Model>` + its masks + its `ModelRuntime`,
replacing the current one, without tearing down the thread, the CUDA context, or the
PTX modules. Unload → parked state already exists; extend it with "install model X".

### 1.4 ACCEPTANCE GATE (do not proceed past this until green)
Two preseeded models A and B (pick two with **different tokenizers/templates**, e.g.
the Thinking model and a second NVFP4 model). Test:
1. Boot → both snapshotted → `state: unloaded`.
2. Load A → temperature-0 completion `P` → must equal a **fresh standalone A process**.
3. Swap to B → temperature-0 completion `Q` → must equal a **fresh standalone B
   process** (this is the test that catches the mask/tokenizer bug — comparing B to
   itself is NOT sufficient).
4. Swap back to A → must again equal `P`.
Deliver `scripts/multimodel_ab_test.sh` driving this. **If Q differs from fresh-B, the
seams are incomplete — stop and report, do not paper over it.**

## Phase 2 — Preseed orchestration

### 2.1 Config (`cli/serve_args.rs`)
Add `ATLAS_PRESEED_MODELS`. Accept a JSON array string
(`ATLAS_PRESEED_MODELS='["model-1","model-2"]'`) parsed with `serde_json::from_str`
into `Vec<String>`; on parse error, fail fast with a message naming the bad value.
(Also accept the `--preseed-model` repeatable flag as an alternative.)

### 2.2 Boot loop (`main_modules/serve.rs`)
Before the HTTP server starts, for each model in order:
- Compute its `model_version_hash`; look for a matching snapshot in `--snapshot-dir`.
- **Snapshot present AND hash matches** → skip init (leave on SSD).
- **Missing OR hash mismatch** → load from safetensors into the arena → write `.raw`
  snapshot → unload. (Hash mismatch = stale snapshot after a kernel/format change;
  regenerate. Do NOT trust filename alone.)
- Peak-safety: only one model in the arena at a time (unload before next) — depends
  on Phase 0.1.
**Post-condition (assert it):** after the loop, `state: unloaded`, arena `used_bytes:
0`, no model in the scheduler. The server waits idle.

## Phase 3 — Lazy-load interceptor

### 3.1 Request-time swap (`scheduler/hotswap_ctl.rs` or a tower layer)
On `/v1/chat/completions` (and `/v1/completions`): read `model`.
- Requested model == active → straight through.
- Requested is registered but not active → **synchronously** `unload_current()` (if
  any) → `load_requested()` from snapshot → then enqueue.
- Requested not in the preseed registry → **404** with a JSON body listing available
  models. Never implicitly init an unregistered model from safetensors at request time.

### 3.2 Concurrency (define explicitly — this is where it goes wrong)
Guard the active model + swap with a state machine (`Idle(model) | Swapping(to)`).
- Concurrent requests for the **same** target that is loading → wait on the in-flight
  load, then proceed (don't double-swap).
- A request for a **third** model while a swap is in flight → 503 + `Retry-After`, or
  serialize behind a queue (pick one, document it).
- Reuse the Phase-1 drain: in-flight decode on the old model finishes (2 s cap) before
  the swap frees it.
**Accept:** hammer the endpoint with interleaved A/B/A/B requests under concurrency;
no wrong-model responses, no double-swaps, no deadlock; each response's content
matches its requested model.

## Phase 4 — Model listing API

### 4.1 `GET /admin/models` (extend existing) + `GET /v1/models` (OpenAI-shaped)
List every preseeded model with `{ id, state: "running"|"unloaded", used_bytes,
snapshot_present, model_version_hash }`. Exactly one may be `running`; after preseed,
none. `/v1/models` mirrors the ids in OpenAI format for client compatibility.
**Accept:** reflects reality across a boot → idle → load A → swap B → unload cycle.

## 5. Cross-cutting requirements

- **Latency honesty:** the swap is ~6 s; the triggering request blocks for it. Add a
  `X-Atlas-Model-Swap: <ms>` response header and a log line when a request caused a
  swap, so this is observable, not surprising.
- **Metrics:** swap count, swap duration, bytes/s, per-model residency time
  (Prometheus already a dep).
- **Acceptance metrics stay:** uvm slab canary, temperature-0 identity vs fresh
  process, tokens/s — never `cuMemGetInfo`/`nvidia-smi`.
- **io_uring:** the dev container's seccomp forces sequential O_DIRECT fallback; the
  GB10 host may allow io_uring. The reload log line says which ran — check it; if
  perf matters, add `security_opt: [seccomp=unconfined]` (or an io_uring-allowing
  profile) to the compose service and re-measure.
- **Out of scope (do NOT attempt here):** dual-residency pointer-flip; EP/world_size>1
  and `--dflash` (hot-swap disabled there); on-demand init of unregistered models.
- **CUDA graph leak (flagged earlier):** captured graphs are not `cuGraphExecDestroy`d
  on unload. With N preseed cycles + repeated swaps this leaks per swap. Fix it as
  part of 1.3's handover teardown, and verify with a many-swap MemFree-drift loop.

## 6. Suggested session split
- **Session 1:** Phase 0 + Phase 1, stop at the 1.4 A→B→A gate. This is the risky,
  correctness-critical half.
- **Session 2:** Phase 2–4 (orchestration), once the seams are proven safe.

## 7. Build/verify
`cargo build --release -p spark-server` clean under clippy; non-GPU tests pass.
Claude does NOT run GPU workloads — deliver scripts (`multimodel_ab_test.sh`,
concurrency hammer) for the human to run. STOP + `docs/hotswap/findings.md` on any
platform-assumption failure.

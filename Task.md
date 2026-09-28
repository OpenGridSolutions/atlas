# Task: Per-phase KV-cache observability for atlas (prefill / thinking / generation)

## Context

This repo is **atlas**, a Rust inference server (workspace under `crates/`,
scheduler in the `spark` crate). Logging uses `tracing` with an `EnvFilter`
driven by `RUST_LOG`. Relevant log sites you will find in the code (they appear
in production logs):

- `spark::scheduler::prefill_a_step` — chunked prefill start, "Prefill first token"
- `spark_model::model::trait_impl::prefill_b::prefix_lookup` — "Prefix cache hit: N tokens (B blocks) reused"
- `spark_model::model::trait_impl::decode_a` — CUDA graph capture per slot
- `spark::scheduler::decode_logits_step` — decode loop / "process_decode_logits"
- `spark::scheduler::lifecycle` — `finish_sequence`, including the
  "blocking response send failed (receiver dropped)" path

The server currently becomes unstable under client timeouts: a request whose
HTTP receiver was dropped keeps decoding to `max_tokens`. I need observability
to (a) watch how the model consumes KV cache in each phase and (b) prove or
rule out a KV block leak in the finish paths.

There may be a draft module `memdebug.rs` in the repo root or provided
alongside this task; if present, use it as a starting point and adapt it to
the actual codebase APIs. If not, create it fresh.

## Goal

Add a debug mechanism, enabled by env var **`ATLAS_MEM_DEBUG=1`**, that makes
KV-cache usage transparent **per request and per phase** — prefill, thinking,
and generation. Note: thinking is not a separate compute phase; it is the
leading segment of decode, tracked by the existing `thinking_tokens` counter.
The instrumentation must attribute decode-time KV consumption to thinking vs.
generation accordingly. All output goes to a dedicated tracing target
**`memdebug`**, so that

```
RUST_LOG="off,memdebug=debug" ATLAS_MEM_DEBUG=1 ./serve.sh
```

shows ONLY these debug lines and silences every other module.

## Requirements

1. **Explore first.** Locate: the KV block manager (allocation, free list,
   refcounts, prefix-cache/evictable bookkeeping), the sequence struct and its
   lifecycle (admission → prefill → decode → finish), and every finish path
   including receiver-dropped / cancel / error. Summarize findings before
   editing.

2. **Prefill phase instrumentation.** When a sequence's prefill completes,
   log for that request:
   - prompt tokens, prefix-cache hit tokens and reused blocks,
   - newly allocated blocks for the prompt (net of reuse),
   - blocks held by this sequence after prefill.

3. **Decode phase instrumentation, split into thinking vs. generation.**
   The scheduler already distinguishes these (see `decode_logits_step`:
   "output_tokens=N, thinking_tokens=M"). Attribute KV block consumption to
   the phase that caused it:
   - every time a NEW block is allocated for the growing sequence, log block
     index/count, tokens so far, and `phase` = `"thinking"` or `"generation"`
     depending on whether the sequence is currently emitting thinking tokens,
   - a periodic summary every 128 output tokens (blocks held, pool free,
     thinking vs. generation token counts so far),
   - on the thinking→generation transition, log a one-line phase summary:
     blocks and tokens consumed by the thinking phase.
   For models without a thinking phase (e.g. TranslateGemma), everything is
   attributed to `"generation"` and no transition line appears.
   This is the "watch the model's fingers" part: it must be visible how each
   phase eats blocks as output grows toward `max_tokens`.

4. **Finish instrumentation — all paths.** On every finish (normal stop,
   length, cancel, receiver dropped, error): log blocks released and blocks
   the sequence still holds afterwards (must be 0), plus a per-phase breakdown
   of what this request consumed: prefill (prompt tokens, blocks net of
   prefix-cache reuse), thinking (tokens, blocks), generation (tokens,
   blocks). Prefer implementing the release logging via RAII/Drop on the
   sequence's block holding so no path can skip it.

5. **Pool accounting + leak invariant.** Maintain/log pool stats
   `{total, free, active_used, cached_evictable}` at request admission and
   finish, plus `unaccounted = total - free - active_used - cached_evictable`.
   Whenever the scheduler goes idle (zero live sequences):
   - `active_used > 0`  → `tracing::error!(target: "memdebug", ...)` with
     "KV LEAK: blocks held by sequences while scheduler is idle"
   - otherwise a debug line "idle invariant ok".

6. **GPU + host memory context.** At request start and end, log
   `cudaMemGetInfo` free/total (MiB) and process RSS, with deltas per request
   and a cumulative drift vs. the first instrumented request (CUDA-graph leak
   detection — note: a new graph is captured per request; if `drift` sinks
   monotonically, graphs or other GPU allocations are leaking).

7. **Zero cost when off.** With `ATLAS_MEM_DEBUG` unset: no logging, no
   snapshots, no measurable overhead (e.g. `Option<Guard>` fields, early
   returns). Observability only — do NOT change scheduling, caching, or
   eviction behavior in any way.

8. **Log hygiene.** All output on target `memdebug` as structured tracing
   fields (no free-form prose), levels: `debug` for accounting, `error` only
   for invariant violations. Verify the subscriber uses `EnvFilter` from
   `RUST_LOG` so the `off,memdebug=debug` directive works; if a hard-coded
   level filter exists, switch it to `EnvFilter::from_default_env()` with the
   current default as fallback.

## Acceptance criteria

- `cargo build` (and `cargo clippy`) pass.
- With `RUST_LOG=off,memdebug=debug ATLAS_MEM_DEBUG=1`: only `memdebug` lines
  appear; without `ATLAS_MEM_DEBUG`: no memdebug output at all.
- A translate request shows: request start → prefill summary (with prefix-hit
  reuse) → periodic decode summaries with growing block usage (all attributed
  to `generation`, `thinking` tokens/blocks reported as 0, no transition
  line) → finish with full release and per-phase breakdown → pool stats.
- A request against a thinking-capable generative model (e.g. the
  Qwen3-…-Thinking preseed) shows: decode allocations labeled `thinking`
  first, one thinking→generation transition line with the thinking phase's
  token/block totals, then `generation`-labeled allocations, and a finish
  breakdown where thinking + generation tokens sum to `output_tokens`.
- Simulated client disconnect mid-generation (drop the HTTP connection while
  decoding): the finish path still logs a complete block release, and the
  idle invariant afterwards reports ok — or, if it reports KV LEAK, that is
  the bug I'm hunting: fix the release in that finish path as part of this
  task and note what was wrong.
- Add a short usage section (env vars, filter directive, how to read the
  fields, what a leak looks like) to the existing docs (`Doku.md` or
  `docs/`), and the two env lines as a comment in `.devpod/compose.yml`.

## Non-goals

- No changes to scheduling policy, prefix-cache eviction, `max_tokens`
  handling, or the HTTP API.
- No new dependencies unless unavoidable (prefer std + existing `tracing`).

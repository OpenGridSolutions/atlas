# Plan — Mid-chat output truncation after model switching (Task 6 follow-up)

Status: analysis complete; **Phase 1 (1.1–1.3) and 2.1 IMPLEMENTED** on branch
`modifications` — see §0.1. Phases 2.2, 3, 4 remain open.
Evidence base: `logs/serve-20260710-184943.log` (the failing AnythingLLM session, 2026-07-10
18:49–19:04 local) + full code audit of the KV sizing, scheduler, request-limit, and
watchdog paths.

---

## 0. What actually happened (root-cause correction)

The suspected cause was inconsistent KV-cache memory management across model swaps.
**The log disproves that for the observed cuts.** All five lazy swaps in the session were
clean (~7 s each, arena base constant, 0 sequences force-finished). Every truncated
response maps 1:1 to a **content-loop watchdog hard stop**, reported to the client as
`finish_reason: "length"`:

| Chat response (UI) | Log event | Tokens |
|---|---|---|
| 19:01, Coder, 28.0 s @ 55.0 tok/s (~1540) | `17:01:57 Fuzzy repetition: 36-tok pattern x3 … stopping at 1542 tokens (rollback declined: StreamUnsafe)` → `Done: 1542 tokens (length)` | 1542 |
| 19:02, Coder, 14.8 s @ 39.8 tok/s (~588) | `17:02:46 … stopping at 590 tokens (rollback declined: StreamUnsafe)` | 590 |
| 19:03, Coder, 11.0 s @ 50.9 tok/s (~560) | `17:03:29 … stopping at 560 tokens (rollback declined: StreamUnsafe)` | 560 |
| 19:04, Thinking (after swap), 27.9 s @ 44.7 tok/s (~1245) | `17:04:37 … stopping at 1245 tokens (rollback declined: StreamUnsafe)` | 1245 |

The generated content was a Vue/Tailwind CRUD answer containing ~20 near-identical
`<th scope="col" class="px-6 py-3 text-left …">` lines inside a markdown code fence — a
legitimate 36-token repeating pattern. The fuzzy-repetition detector treats 3 fuzzy
repeats as a generation loop and, because rollback is **always declined on streaming
requests** (`StreamUnsafe`), hard-stops the sequence. It fired for **both** models,
before and after swaps — it is model- and swap-independent.

However, the audit **did** find real memory-management inconsistencies on the swap path
(the thing the user suspected). They did not cause these cuts, but they make KV capacity
nondeterministic across swaps and can cause genuine mid-decode KV exhaustion in longer
chats. Those are Phases 3–4 below.

---

## 0.1 Implemented (2026-07-10, branch `modifications`)

| Commit | Step | Change |
|---|---|---|
| `6d1d283` | 1.1 | ``` fence parity tracked in the CONTENT phase on both decode paths (`handle_content_token` + `emit_token`; `ActiveSeq`/`SwappedSeq` gained `code_fence_token`); fuzzy-repetition + content-loop detectors skip inside fences |
| `4fcc579` | 1.2 | API-side SimHash + line-loop guards fence-gated via `StreamState.in_code_fence` (`stream_guards::update_fence_state`, split-delta safe); both now honor `ATLAS_DISABLE_WATCHDOGS=1`; fence content kept out of guard scan buffers |
| `a74ac0b` | 1.3 | fuzzy detector respects `ATLAS_DISABLE_WATCHDOGS=1` (kill-switch asymmetry closed); new `atlas_watchdog_stops_total{detector}` counter (fuzzy \| content_loop \| simhash \| line_loop) |
| `fccee05` | 2.1 | `max_tokens` clamped to `max_seq_len - prompt_len` at admission in /v1/chat/completions + /v1/completions (logged when it bites) |

**Second test round (2026-07-10 evening, curl):** run 1 of the CRUD prompt completed
(`stop`, 3968 tokens — first time ever), confirming the fence fix works when the fence
is recognized. Two residual cuts were root-caused from `serve-20260710-205302.log` and
fixed in `ff7bbfd` (step 1.4):

- **"count from 1 to 500" cut at 49 tokens** — the digit-normalized content-loop
  detector (`matched_period=0`, first check after the 48-token warmup) cannot tell
  counting from a stuck number loop: after digit normalization both are period-2
  (digit, newline). Fix: numeric-token VALUES table + counting exemption — a strictly
  monotonic number sequence in the scan window vetoes the normalized-path fire; stuck
  (`7,7,7`) or oscillating (`46,7,47,9`) values still fire.
- **CRUD run 3 cut at 1596 tokens** (fuzzy, `rollback=1,2`, then `CapReached`) — the
  fence toggle listened only for the atomic ``` token (73594), but the vocab has two
  merged variants (`54275 " ```"`, `41233 " ```\n"`). When sampling emitted one of
  those, `in_code_fence` went stale and the fuzzy detector ran inside the code block —
  hence the intermittency (same prompt, temp 0.6). Fix: per-model fence-token MASK
  (every id whose text carries an odd number of ``` markers) drives the toggle on all
  scheduler paths; boot log now prints `Fence-token mask: N/vocab ids`.

Deviations from the original plan text (deliberate, small scope for a first test round):
- 1.2: per-model MODEL.toml `[behavior]` switches for the API-side guards NOT added yet
  (fence gating + the env kill switch cover the observed failure; add the toml keys if a
  model needs the guards fully off).
- 1.3: the `"atlas": {"stop": …}` extension field on the final SSE chunk NOT added yet —
  watchdog stops are still reported as plain `finish_reason: "length"`; use the WARN log
  line + the new counter to tell them apart. 1.4 (fuzzy re-tune) not needed so far.

**How to verify on the box:** rebuild + restart serve.sh, replay the failing chat
(Laravel/Inertia/Vue CRUD prompt, then "please finish the rest") against both models.
Expect: the `<th …>` table renders to the end, no `Fuzzy repetition`/`SimHash` WARN in
the log, `finish_reason: "stop"`, and `curl -s localhost:11434/metrics | grep
atlas_watchdog_stops_total` stays flat. `X-Atlas-Model-Swap` still appears on
swap-triggering requests.

---

## 0.2 Other hypotheses for "cut with finish_reason: length" (if it still reproduces)

The four cuts in the analyzed session were all watchdog kills. If a truncation still
shows up (curl or UI) after the fixes above, check these — in order of likelihood:

1. **Default `max_tokens` = 4096.** If the request omits `max_tokens` (typical for bare
   curl), the server defaults to a hard-coded 4096 (`openai/chat_request.rs:441`);
   AnythingLLM sends its own value (16384 in the analyzed session). A long answer
   legitimately ends at 4096 with `finish_reason: "length"`. Diagnose: `Done: N tokens
   (length)` with N ≈ 4096 in the log, `atlas_watchdog_stops_total` flat. Fix: send
   `max_tokens` explicitly (or implement a context-derived default, Phase 4.1).
2. **Thinking tokens do NOT consume the budget, but a forced think-end can still eat the
   turn.** For the Thinking model, `<think>` tokens are "free" (`remaining` untouched),
   so this is usually NOT the cause — but check `thinking_tokens=` in the
   `process_decode_logits: remaining=0` log line to see how the budget was spent.
3. **Request timeout.** `--request-timeout` / `req.timeout` ends the stream mid-output
   and reports `"length"` (`decode_logits_step.rs` timeout check; log: `Request timeout
   after …`). More likely as chats grow (slower decode at depth).
4. **Tool-path caps** (only when the client sends tools — AnythingLLM agents do):
   `tool_max_tokens` boot cap, `max_post_think_content_tokens` (MODEL.toml), inter-tool
   prose budget, tool-loop cap — all end the stream as `"length"`. Log lines name them.
5. **KV exhaustion** (now less likely with the 2.1 clamp, until Phase 2.2/3 land):
   signature is `decode_batch error: … KV cache exhausted` in the log and an SSE
   `{"error": …}` frame instead of a `finish_reason` chunk — and it kills ALL concurrent
   generations at once.
6. **Client-side rendering** (AnythingLLM only): a `data:` frame carrying an error object
   is silently dropped by some UIs; always cross-check with curl (`-N` streaming) and the
   server log's `Done: … (reason)` line, which is authoritative.

---

## 1. Findings inventory

### F1 — Fuzzy-repetition detector misfires on legitimate repetitive code (ROOT CAUSE)
- Detector: `crates/spark-server/src/scheduler/repetition.rs:17` — pattern lengths 15–40,
  3 consecutive near-copies, per-pair mismatch tolerance `pattern_len / fuzzy_repeat_tolerance_div`
  (default div 12 ≈ 8%). Repetitive HTML/SQL/enum code trivially satisfies this.
- Call site `crates/spark-server/src/scheduler/decode_logits_step.rs:652`: gated only on
  the per-model `enable_loop_watchdog` flag; exempts thinking and tool-call XML but has
  **no code-fence awareness**.
- On fire it tries `rollback_to_boundary` (`scheduler/rollback.rs:203`), but every
  streaming request has `cancel_flag = Some(..)` → `RollbackFallback::StreamUnsafe`
  (`rollback.rs:213-216`) → **hard stop, always, for streaming chat**. "Rollback+re-steer
  ENABLED" in the boot log is dead code for streams.
- `ActiveSeq.in_code_fence` exists (`scheduler/types.rs:203`) and `code_fence_token` is
  threaded through `SchedulerModelParams`, but `toggle_code_fence` is only called inside
  the `<think>` branch (`decode_logits_step.rs:258`). In the content phase the flag is
  never maintained and no detector consults it.

### F2 — Two more always-on API-side guards with the same misfire class
- SimHash semantic-loop guard (`crates/spark-server/src/loop_simhash.rs`, driver
  `api/chat_stream/handle_token.rs:530-571`): Jaccard ≥55% over word bigrams vs the last
  16 sentences → sets `cancel_flag`, truncates. **Unconditionally on** for every streaming
  chat; no MODEL.toml flag, no env var, no request param. Identical `<th …>` rows are
  >55% Jaccard duplicates. It fired twice in the failing session (warn-only there because
  the fuzzy stop won the race).
- Line-level guard (`api/stream_guards.rs:44`): same non-blank line ≥4× → cancel. Skips
  lines *starting with* ``` but has no interior-fence awareness.

### F3 — Watchdog stops are invisible/mislabeled (observability)
- Every watchdog kill finishes with the last token = ordinary content, so
  `lifecycle.rs:23-29` reports `finish_reason: "length"` — indistinguishable from a real
  `max_tokens` truncation for the client and for anyone reading `/metrics`.
- `ATLAS_DISABLE_WATCHDOGS=1` does **not** cover the fuzzy detector (gate at
  `decode_logits_step.rs:652` omits the `disable_watchdogs()` check) nor the two API-side
  guards. The per-request `repetition_detection` override only tunes the token-level
  content-loop detector, not fuzzy/SimHash/line guards.

### F4 — KV pool sizing on the swap path uses boot-model inputs (the real memory inconsistency)
- `inference_reserve` is computed once at boot from the **boot model's** SSM/GDN geometry
  (`serve_phases/preflight.rs:116`) and reused verbatim by the reload closure
  (`serve.rs:672`). Model B's KV budget is therefore reduced (or under-reserved) by model
  A's reserve. `max_batch_tokens` likewise boot-fixed (`kv_cache.rs:114` → `serve.rs:670`).
- `baseline_free_bytes` is a process-global captured once at boot
  (`preflight.rs:220`, `spark-runtime/src/gpu.rs:19-33`), never refreshed on swap. On the
  GB10 UMA box the auto-measure is already implausible at boot (`baseline 23.1 GB` vs
  `free-now 32.8 GB`) so `factory/build.rs:307-335` falls back to raw `used_so_far` on
  **every** build. Observed effect: KV pool drifted 52062 → 50569 → 54016 → 59370 →
  59323 → 59312 blocks across six builds of the *same two models* in one session —
  nondeterministic capacity depending on ambient free memory at build time.
- Neither `preflight_reserve` (pre-load bail) nor `post_load_memory_audit` runs on the
  swap path (`preflight.rs:119,271` are boot-only); a mis-sized swap fails at raw
  allocation time with no guard.
- Perf side effect on this box: `Skipping MoE weight transposition (40.5 GB needed,
  ~33 GB available)` on every build → fallback grouped-GEMM prefill, always.

### F5 — KV exhaustion mid-decode kills the whole batch with an untyped SSE error
- Decode-time block alloc failure (`spark-model/src/model/block_mgmt.rs:270-289`, after a
  single prefix-cache evict retry) propagates out of `decode_batch`, and
  `scheduler/decode_step.rs:57-66` then drains **every** active sequence with
  `send_error` — one starved sequence cuts all concurrent chats.
- The client gets a bare `{"error": …, "code": 500}` SSE data frame with **no
  `finish_reason` chunk and no `[DONE]`** (`api/chat_stream/handle_error.rs:11-24`).
  AnythingLLM-class UIs render nothing → silent truncation.
- No preemption/spill is attempted for a *running* decode; spill only runs proactively
  for incoming prompts and only with `--swap-space-gb > 0` (`scheduler/mod.rs:376-407`).

### F6 — No per-model context budget anywhere in the request path
- `max_tokens` defaults to hard-coded 4096 (`openai/chat_request.rs:441`), never derived
  from context; nothing clamps `prompt_len + max_tokens` (admission checks prompt alone
  vs boot `max_seq_len`, `api/chat/mod.rs:226`, `api/completions.rs:131`).
- `ModelRuntime.effective_context` is **dead**: written `0` at boot (`serve.rs:497`) and
  at swap (`swap_target.rs:171`), read nowhere.
- Auto-compact (when enabled) triggers at a hard-coded `0.70 × boot max_seq_len`
  (`api/chat/template.rs:110`) — the configured `--auto-compact` fraction is only used as
  an on/off flag, and the boundary never adapts to the swapped-in model.

### F7 — Smaller swap-staleness items
- `use_mtp`/`num_drafts`/`mtp_gate` computed once from the **boot** model
  (`scheduler/mod.rs:212-256`), never recomputed on swap — wrong speculative dispatch if
  proposer capability differs between models.
- `block_size` in the scheduler's spill heuristic is the boot value (`mod.rs:379`), not
  `model.block_size()` of the current model.
- KV spill files carry no model identity; cross-model restore is prevented only by the
  drain invariant (fragile — no defense in depth in `restore_sequence_state`).
- Cosmetic: first lazy load logs `swap … (from 'X')` where X == target (post-preseed
  state is `unloaded` but `active` still names the boot model).

Verified NOT broken (for the record): the prefix/radix cache, paged KV pool, and SSM
session snapshots are all model-owned or explicitly cleared on swap — **no stale KV bytes
cross the swap boundary**; the scheduler reads block counts live from the current model.

---

## 2. Fix plan

One commit per numbered step, `hotswap:` prefix (watchdog steps may use `watchdog:`),
branch `modifications`. No GPU workloads run by Claude; GPU validation scripts are
delivered for the user to run. Ordered by user impact.

### Phase 1 — Stop cutting legitimate code output (fixes the reported bug)

**1.1 Maintain `in_code_fence` in the content phase and gate the scheduler detectors on it.**
- Call `toggle_code_fence(a.in_code_fence, tok, code_fence_token)` in
  `handle_content_token` (`decode_logits_content.rs:66`), mirroring
  `decode_logits_step.rs:258`; keep the reset at `</think>`.
- Gate the fuzzy-repetition check (`decode_logits_step.rs:652`) and the content-loop
  check (`decode_logits_content.rs:123`) on `!a.in_code_fence`.
- Fallback when the tokenizer has no atomic ``` token (`code_fence_token == None`):
  detectors stay active (current behavior), documented.
- Tests: unit test that a sequence of repeated `<th …>`-like token patterns inside a
  fence is not stopped, and the same pattern outside a fence still is; fence state
  correctly closes on the second ``` .

**1.2 Track code fences in the streaming layer and suppress the API-side guards inside fences.**
- Add fence tracking to `StreamState` (count ``` occurrences in sanitized content);
  skip `SimHashLoopGuard::check` and `stream_guards::check_loop_watchdog` while inside a
  fence.
- Make both API-side guards configurable: honor `ATLAS_DISABLE_WATCHDOGS` and add
  MODEL.toml `[behavior]` keys (`enable_simhash_guard`, `enable_line_loop_guard`,
  defaults = current behavior) plumbed through `ModelBehavior` →
  `log_behavior_audit` — same ArcSwap pattern as `WatchdogParams` so they swap per model.

**1.3 Make watchdog stops honest and controllable.**
- Fix the `ATLAS_DISABLE_WATCHDOGS` asymmetry: the fuzzy detector must respect
  `disable_watchdogs()` like the content-loop detector does.
- Distinguishable finish: when any watchdog ends a sequence, record it on the sequence
  and (a) log at WARN with request id (exists), (b) emit `finish_reason: "length"` but
  add a `"atlas": {"stop": "loop_watchdog", "detector": "fuzzy|simhash|line|content"}`
  extension field on the final chunk, and (c) add a Prometheus counter
  `atlas_watchdog_stops_total{detector, model}`. (Keeping `"length"` preserves OpenAI
  client compatibility; the extension + metric make it diagnosable.)
- Extend the per-request `repetition_detection` override so clients can relax/disable
  the fuzzy detector too (e.g. `fuzzy: false` or a tolerance field).

**1.4 Re-tune the fuzzy detector for code-shaped output (after 1.1 lands, may become optional).**
- Raise the repeat requirement from 3 to a `[behavior]`-configurable
  `fuzzy_repeat_min_repeats` (default 3 outside fences); consider requiring ≥4 repeats
  when mismatches > 0 in both pairs. Only if false positives persist outside fences
  (e.g. markdown tables outside ``` blocks).

Acceptance gate Phase 1: replay the failing prompt ("Laravel + Inertia + Vue CRUD, finish
the table") against both preseeded models on the GB10 box — full table renders, no
`Fuzzy repetition` / SimHash stop, response ends with a real EOS `finish_reason: "stop"`.
Deliver `scripts/watchdog_codefence_test.sh` (boots serve.sh, sends the repro prompt,
fails if the answer stops mid-`<th` or the new watchdog metric increments).

### Phase 2 — Honest failure when KV genuinely runs out

**2.1 Clamp the generation budget at admission.**
- In the chat/completions handlers: `max_tokens = min(req.max_tokens, max_seq_len -
  prompt_len)` (reject with 400 only when the prompt alone doesn't fit, as today). This
  turns the "runs until the pool dies" case into a clean `finish_reason: "length"`.

**2.2 Isolate per-sequence KV exhaustion instead of killing the batch.**
- In `ensure_blocks_through_decode`, surface a typed `KvExhausted { seq }` error.
- In `decode_step.rs` (and the MTP/verify twins): on `KvExhausted`, finish **only** the
  offending sequence with `finish_reason: "length"` (plus the `atlas.stop =
  "kv_exhausted"` extension + a metric), keep the rest of the batch running. Retry the
  step without the finished sequence.
- On the streaming error path that remains (real device errors), emit a terminal chunk +
  `[DONE]` after the error frame so clients always see a closed stream.

Acceptance gate Phase 2: `scripts/kv_exhaustion_test.sh` — boot with a tiny
`--kv-pool-gb`, run two concurrent long generations; exactly one may finish with
`length`, the other must complete, and both streams must terminate with `[DONE]`.

### Phase 3 — Deterministic, per-model KV sizing across swaps

**3.1 Recompute `inference_reserve` and `max_batch_tokens` per swap target.**
- Extract the model-dependent parts of `preflight_reserve` and `resolve_prefill_budget`
  into functions of `(args, config)`; call them in `prepare_swap_target` and carry the
  values in `PreparedSwap`; reload closure passes the per-model values to `build_model`
  instead of the boot bindings. (Scheduler consumes `max_batch_tokens` via
  `SchedulerModelParams` — add the field and rebind on swap, fixing the F7 `block_size`
  staleness in the same step by using `model.block_size()` in the spill heuristic.)

**3.2 Make the KV budget deterministic instead of ambient-free-memory-derived.**
- The "auto-measure implausible" fallback fires on every UMA build; capacity drift of
  ~15% between swaps of the same model is unacceptable for reproducibility. Compute the
  pre-KV footprint deterministically: `weights_arena_capacity + buffer_arena_bytes +
  reserve(config)` (all known exactly at build time) and derive
  `kv_budget = total × util − footprint`; keep the free-memory probe only as a safety
  clamp (`min(kv_budget, actual_free − reserve)`). Same-model reload must yield an
  identical block count every time; assert that in the snapshot-reload test.

**3.3 Run a memory audit on the swap path.**
- Port `post_load_memory_audit` into the reload closure (after `build_model`), bailing
  the swap (state returns to `NoModel`, load error to the caller) instead of letting the
  next request hit a raw CUDA OOM.

**3.4 Defense in depth for spill + speculative state.**
- Stamp `model_version_hash` into KV spill file headers; verify in
  `restore_sequence_state`.
- Recompute `use_mtp`/`num_drafts` from the swapped-in model (add
  `has_proposer`-derived fields to `SwapOutcome`), reset `mtp_gate` measurements on swap.

Acceptance gate Phase 3: extend `scripts/lazy_swap_concurrency_test.sh` (or new
`scripts/kv_consistency_test.sh`) — swap A→B→A→B, scrape the logged block counts (or a
new `atlas_kv_blocks{model}` gauge): same model ⇒ identical block count on every load;
`KV budget auto-measure implausible` no longer appears.

### Phase 4 — Per-model context plumbing (removes the last boot-static limits)

**4.1 Wire `effective_context` for real.**
- Populate it in both `ModelRuntime` constructions from
  `min(config.max_position_embeddings, args.max_seq_len)` (boot flag becomes an upper
  clamp, not the truth); use it in the prompt-too-long checks, the 2.1 clamp, and
  `/v1/models` metadata.

**4.2 Fix auto-compact.**
- Use the configured `--auto-compact` fraction as the actual trigger ratio (today the
  0.70 is hard-coded and the flag value is only an on/off switch) and compact against
  the per-model `effective_context` from 4.1.

Acceptance gate Phase 4: unit tests for the clamp formulas; on-box check that a prompt
sized between the two models' contexts is accepted by the larger and 400'd by the
smaller after a swap.

### Out of scope (unchanged from Task 6)
Dual-residency, EP/world_size > 1, `--dflash`, on-demand init of unregistered models.
MoE-transposition-at-preseed (fixes the permanent grouped-GEMM fallback, F4 last bullet)
is a worthwhile separate task: transpose once during preseed and persist into the `.raw`
snapshot — noted here, not scheduled.

---

## 3. Suggested order & effort

| Step | Fixes | Size | Risk |
|---|---|---|---|
| 1.1 | the reported truncations (scheduler side) | S | low |
| 1.2 | same misfire class in the API layer | M | low |
| 1.3 | observability + kill-switch asymmetry | S | low |
| 2.1 | context-blind budget | S | low |
| 2.2 | whole-batch kill on KV exhaustion | M | medium (touches decode step twins) |
| 3.1 | per-model reserve/budgets on swap | M | medium |
| 3.2 | nondeterministic KV capacity | M | medium (UMA measurement semantics) |
| 3.3 | unguarded swap OOM | S | low |
| 3.4 | latent spill/MTP staleness | S | low |
| 4.1–4.2 | per-model context limits | M | low |
| 1.4 | only if needed after 1.1 | S | low |

Phase 1 alone should make the AnythingLLM session complete its answers; Phases 2–3 fix
the genuine (latent) KV memory inconsistencies that the switching test surfaced.

# Bug Hunt Brief — `max_tokens` ignored / `finish_reason` mislabeled

> Self-directed debugging task for Claude Code. A fixed test oracle defines
> success. Form hypotheses, test each against the oracle, keep only what moves the
> oracle, revert everything that doesn't. Branch `modifications`.

## 1. The bug (observed, verified by direct curl — no client involved)

Generation stops far short of the requested `max_tokens`, and `finish_reason` is
reported as `"length"` even when a length limit was not actually reached.

Measured facts:
- `{"max_tokens":16384}`, CRUD prompt → `finish_reason:"length"`, completion_tokens **1793**, prompt_tokens 36.
- `{"max_tokens":100}`, "count to 500" prompt → `finish_reason:"length"`, completion_tokens **49**.
- `{"max_tokens":8192}`, same "count to 500" prompt → `finish_reason:"length"`, completion_tokens **49** (identical to the 100 case).
- The generated text itself is **correct** (valid Vue/Laravel, correct counting) — just cut off early.

Interpretation constraints (do not violate these when forming hypotheses):
- Correct-but-truncated text ⇒ this is a **control-flow / limit bug on the Rust side**
  (request → effective max_new_tokens → sampling stop condition → finish_reason label).
  It is NOT a kernel bug and NOT a KV-cache bug: corrupt KV/kernels produce *wrong*
  tokens (garbage/repetition), not *correct* tokens that stop early.
- The effective limit varies by prompt (49 vs 1793), so it is not a single hardcoded
  constant; it is more likely a mis-derived limit (e.g. a wrong `max_new_tokens`
  default, an off-by-something on `max_seq_len - prompt`, or a stop-token/finish
  labeling error where `stop` is reported as `length`).
- **Do NOT touch kernel code, KV-cache allocation, the arena, or CUDA plumbing** to
  fix this. If a hypothesis points there, it is almost certainly wrong — re-derive.

## 2. The oracle (success criterion)

`scripts/repro_maxtokens.sh` — server must already be running (`./serve.sh`).

```bash
#!/usr/bin/env bash
set -euo pipefail
M="${1:-vincentzed-hf/Qwen3-Coder-Next-NVFP4}"
MT="${2:-8192}"
resp=$(curl -s http://localhost:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"count from 1 to 500, one number per line, nothing else\"}],\"max_tokens\":$MT,\"stream\":false}")
echo "$resp" | python3 -c '
import sys,json
d=json.load(sys.stdin); c=d["choices"][0]; u=d["usage"]
fr=c["finish_reason"]; ct=u["completion_tokens"]; pt=u["prompt_tokens"]
# Counting 1..500 one per line needs well over 600 tokens. Reaching far with a
# clean stop = the limit is honored. Stopping early with "length" = the bug.
ok = ct > 600
print(f"finish_reason={fr} completion_tokens={ct} prompt_tokens={pt} -> {\"PASS\" if ok else \"FAIL\"}")
'
```

**PASS** = `completion_tokens > 600` (ideally with `finish_reason` other than a false
`length`). **FAIL** = stops early (the current behavior: ~49).

Rationale for the oracle: "count to 500, one per line" has a deterministic minimum
length (>600 tokens). Any early stop is unambiguous. It needs no subjective judgement
of "looks truncated". The human runs this after each build and reports PASS/FAIL plus
the printed line.

Secondary probe (run once at the start to characterize): call the oracle with
`max_tokens` = 100, 2000, 8192 and record completion_tokens for each. If all three
stop at the *same* count → the request value is fully ignored. If the count tracks the
request up to a ceiling → there is a cap. This distinguishes "ignored" from "clamped".

## 3. Method — bisect first, hypotheses second

### Step A: locate the break with git bisect (preferred — deterministic)
The bug appeared during/after the Task 6 ArcSwap `ModelRuntime` refactor. There is a
checkpoint commit from just before the autonomous Task 6 session.
1. Confirm current HEAD is FAIL with the oracle.
2. `git bisect start` → `git bisect bad` (HEAD) → `git bisect good <checkpoint-hash>`.
3. For each step: the human builds (`cargo build --release -p spark-server`), starts
   the server, runs the oracle, and reports good/bad. (Claude does NOT run GPU
   workloads; Claude drives the bisect logic and interprets results.)
4. git converges to the first bad commit. Read ONLY that commit's diff; the root cause
   is almost always visible in it.

If the checkpoint commit is **also FAIL**, the bug predates Task 6 → skip bisect, go to
Step B.

### Step B: hypothesis-driven search (if bisect is inconclusive or bug predates Task 6)
Instrument the path with debug logging, then let the oracle discriminate. Log the
effective `max_new_tokens` and its source, and the exact reason `finish_reason` is set,
at each of these stations (trace the real names in code):
1. **Request parse** — what is `max_tokens` right after deserializing the OpenAI body?
   (Is the field even populated? default? Option unwrapped to a wrong fallback?)
2. **Runtime/defaults merge** — where request params meet `generation_config.json` /
   `ModelRuntime` defaults (the ArcSwap refactor touched this). Does a default
   `max_new_tokens` win over the request? Is the ArcSwap snapshot read once and
   carrying stale/other-model defaults?
3. **Scheduler / sampling loop** — the actual stop check: is it comparing against the
   request limit or something else (e.g. a per-batch cap, `max_seq_len` misused, a
   remaining-budget miscalculation)?
4. **finish_reason labeling** — where length vs stop vs eos is decided. Is a real stop
   token being mislabeled as `length`? Log the deciding token id and whether it was in
   the stop set.

Form each hypothesis as: "if X, then changing Y flips the oracle to PASS". Make the
minimal change, have the human re-run the oracle. If it does not flip → **revert the
change** (`git checkout -- <files>` or `git stash drop`) before trying the next. Never
stack unverified changes.

## 4. Discipline / guardrails
- One hypothesis = one minimal, reverted-if-failed change. No stacking.
- Never edit kernels, KV-cache, arena, or CUDA code for this bug.
- Every "fix" must be proven by the human flipping the oracle FAIL→PASS, and must NOT
  regress the earlier working same-model behavior (re-run `multimodel_ab_test.sh`
  afterward: A→B→A must still be bit-identical).
- Also verify the fix on BOTH models (Thinking and Coder) and at two `max_tokens`
  values, so the fix generalizes rather than accidentally matching one number.
- Claude does not run GPU workloads. Claude proposes commands; the human runs the
  server and the oracle and reports the printed PASS/FAIL line.
- When found: show the root-cause commit/diff, explain the mechanism in one paragraph,
  commit as `hotswap: fix max_tokens/finish_reason (<root cause>)`.

## 5. First actions for Claude
1. Write `scripts/repro_maxtokens.sh` exactly as in §2.
2. Ask the human for the checkpoint commit hash (`git log --oneline | grep -i checkpoint`).
3. Have the human run the secondary probe (100/2000/8192) once and report the three
   completion_tokens, to classify ignored-vs-clamped before bisecting.
4. Drive the bisect (Step A). Report the first-bad commit and its diff.

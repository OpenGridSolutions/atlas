# Plan — Multi-class model integration (embeddings, reranking, VL)

Status: read-only architecture analysis complete (2026-07-10); step 1.0 implemented.
Scope reduction 2026-07-11: **Whisper / audio transcription is descoped entirely**
to focus on the embedding and reranker classes first.
Scope: integrate the four cached models into Atlas alongside the running
Qwen3-Next-80B-Thinking / Qwen3-Coder-Next (both `qwen3_next`, NVFP4, hot-swap +
arena + snapshot infrastructure from Task 6):

| Model (cached) | config `model_type` | Geometry | Class |
|---|---|---|---|
| Qwen/Qwen3-Embedding-8B | `qwen3` (dense) | hidden 4096, 36 L, GQA 32/8, head_dim 128, BF16 | text embedding |
| Qwen/Qwen3-Reranker-8B | `qwen3` (dense) | **identical** to Embedding-8B | text reranking |
| Qwen/Qwen3-VL-Embedding-8B | `qwen3_vl` | same 8B text tower + ViT (27 blocks, hidden 1152, deepstack [8,16,24]) | VL embedding |
| Qwen/Qwen3-VL-Reranker-8B | `qwen3_vl` | same as VL-Embedding | VL reranking |

---

## 1. Current state (traced, with citations)

### 1.1 Input path — text is tokens, images are REAL and wired

`/v1/chat/completions` flow: route (`main_modules/serve_router.rs:44`) →
`api/chat/mod.rs` → `msg_entry::build_msg_entries` (`api/chat/msg_entry.rs:55`)
→ Jinja render → `prompt_tokens: Vec<u32>` → `InferenceRequest::{Blocking,Streaming}`
(`api/inference_types.rs:54`) → scheduler mpsc → prefill
(`scheduler/prefill_a_step.rs:76`) → `model.prefill_chunk(...)`.

**Vision is NOT dead code.** Verdict: real, end-to-end wired, exercised by tests —
but live only when three gates pass:
1. Client sends `image_url` content parts (base64 data URIs; parsed at
   `openai/chat_message.rs:197,235`).
2. Server-side decode: `spark-model/src/vision_preprocess.rs:22-130` — base64 →
   `image` crate PNG/JPEG decode → bicubic resize snapped to patch grid → SigLIP
   normalize → packed `[patches, 1536]` f32. This runs inside `build_msg_entries`
   (`msg_entry.rs:239-265`), gated on `rt.vision_config.is_some()`.
3. `<|image_pad|>` tokens are replicated per patch (`template.rs:135`,
   `tokenizer/chat_impl.rs:292`); after token-embedding lookup the ViT output rows
   are `copy_d2d_async`'d over the pad positions
   (`model/trait_impl/prefill_c.rs:130-153`).

The GPU ViT (`layers/vision_encoder/enc_impl/*`) is a real 27-block Qwen3-VL
tower with DeepStack mergers — **exactly the geometry of the cached
Qwen3-VL-Embedding/Reranker-8B ViT** (depth 27, hidden 1152, deepstack [8,16,24]).
It is loaded only by the Qwen3-VL-family loaders (`weight_loader/qwen3_vl.rs:193`,
`qwen35.rs:139`, `qwen35_dense.rs:774`); all other archs return `None`. Critical
gate: if the selected **kernel target ships no vision module**, `serve.rs:143-155`
forces `config.vision = None` and serves text-only.

**The missing primitive for embeddings** is small and precisely located: every
forward entry returns **logits** (`traits/model.rs:44,50,64`), and the only host
readout is `copy_logits_to_host` (`model.rs:232`). But:
- `buffers.hidden_states()` holds **all-position** post-last-layer hidden states
  (`spark-runtime/src/buffers.rs:220`);
- `hidden_after_norm()` (`model.rs:255`, impl `trait_impl/meta.rs:287`) exposes the
  post-final-norm **last-token** hidden state on device — with no D2H copy today.
- Final-norm + lm_head are applied to the last token only, as a separate finalize
  step (`prefill_b/finalize_last.rs`) — prefill itself is sampling-free.

### 1.2 Kernel target system

A target = `(hw, model, quant)` resolved from the `kernels/` tree
(`atlas-kernels/build.rs:652-811`): `kernels/gb10/common/` holds the ~146-kernel
base set; `kernels/gb10/<model>/MODEL.toml` declares `[[model_types]]` (runtime
selection key), `[sampling.*]`, `[behavior]`; `kernels/gb10/<model>/<quant>/`
holds override `.cu` files that **shadow common kernels by file stem**
(`build.rs:840-864`) — that is the "model-specific overrides" count in build logs.

20 gb10 targets exist today, all `nvfp4` (deepseek-v4-flash, gemma-4-26b/31b,
holo-3.1-0.8b/4b/35b, minimax-m2, mistral-small-4, nemotron-3-nano/super,
ornith-1.0-9b, qwen3-vl-30b-a3b, **qwen3-next-80b-a3b** (default), qwen3.5-27b/
35b/122b/397b, qwen3.6-27b/35b, step3p7-flash).

**Shape specialization — the decisive fact:** kernel dims (hidden, heads, vocab,
experts) are **runtime launch parameters** (`rms_norm.cu:49`, `rope.cu:36-48`);
the only compile-time bake is `head_dim` via `-DHDIM` (`paged_decode_attn.cu:22`,
default 256; six targets already build the `-DHDIM=128` variant, and the build
dedups identical `.cu`+flags across targets — `build.rs:201-225,362-417`).
Therefore: **a new model is a new target** (it needs its own `[[model_types]]`
entry for `ptx_for_config(model_type, hidden_size)` selection,
`atlas-kernels/src/lib.rs:378`), **but a target can be nearly free** — an empty
quant dir inherits the full common set, and a `-DHDIM=128` build is already
cached. Runtime selection failure is a clean boot/swap error listing available
targets (`serve.rs:87-98`, `swap_target.rs:71-79`); hot-swap loads extra targets
append-only via `AtlasRegistry::ensure_target` (`atlas-core/src/registry.rs:256`).

No `qwen3` (dense attention-only) model_type exists in any MODEL.toml, and **no
bf16 quant target has ever been built** (the machinery supports `quant=bf16`;
base kernels include bf16 GEMM/GEMV/attention paths).

### 1.3 Model layer

- Arch dispatch is a single match on `model_type` (`spark-model/src/factory.rs:45-94`);
  everything downstream is model-agnostic (`factory/build.rs` calls loader
  methods; `WeightStore`/`WeightsArena` are name-keyed and arch-agnostic —
  `spark-runtime/src/weights.rs:131`, `weights_arena.rs:33` — so **arena,
  snapshots, and hot-swap work unchanged for any new arch**).
- A **pure dense attention-only transformer already exists and serves today:
  Gemma-4-31B** (`weight_loader/gemma4/loader_a.rs:363-504`,
  `Qwen3AttentionLayer::new_ungated` + `FfnComponent::Dense`). This is the
  template for dense Qwen3. (`qwen35_dense` is NOT pure — it keeps GDN layers.)
- **Non-autoregressive serving already exists**: `max_tokens <= 1` triggers
  immediate-finish after prefill — sequence never enters decode, KV freed at once
  (`prefill_b_step.rs:192,267`, `phase_promote_prefills.rs:56-75`,
  `lifecycle.rs:84`).
- **Batched single-forwards already exist**: `prefill_batch_chunk`
  (`traits/model.rs:121`, real batched dispatch `prefill_b/batch.rs:74`, driver
  `run_batched_prefill.rs:20`) — N prompts, one forward, N last-position outputs.
- Reranker scoring: blocking requests already read full-vocab logits D2H
  (`sample_step.rs:238-273` via `copy_logits_to_host`); indexing 2 entries
  (yes/no token ids) is free.
- `crates/atlas-embed/` exists as an explicit scaffolding crate (`lib.rs:5`,
  `token.rs:19 todo!`) — the intended home for pooling was already reserved.

### 1.4 API surface

Present: chat/completions, completions, responses, conversations, messages
(Anthropic), models, admin/models(+snapshot/unload/load), tokenize/detokenize,
health, metrics. Missing for this plan:
- `POST /v1/embeddings` — exists as **501 stub** (`serve_router.rs:78`) — replace.
- `POST /v1/rerank` — **does not exist** — add (Jina/Cohere-compatible shape:
  `{model, query, documents[], top_n?}` → `results[{index, relevance_score}]`).
All routes register in the `Router::new()` chain at `serve_router.rs:44-107`.
(`/v1/audio/*` stays an untouched 501 stub — audio is out of scope.)

**ENV caveat:** listing the embedding/reranker models in `ATLAS_PRESEED_MODELS`
does NOT make them runnable. `model_type = "qwen3"` (dense) has no loader match
in `factory.rs::loader_for_config`, and before step 1.0 had no kernel target
either — a preseed/boot of these models fails at target selection
(`serve.rs:87-98`) or at loader dispatch. The ENV setting presupposes phases
1.0–1.4; it does not replace them.

---

## 2. THE architectural call — kernel-sharing strategy

Grounded in §1.2 (dims are runtime; HDIM=128 variant already cached; a target
with an empty quant dir inherits the common set; selection requires a
`[[model_types]]` entry):

**D1 — Embedding-8B and Reranker-8B share ONE new target, with each other, and
it contains zero new kernels.**
New target `kernels/gb10/qwen3-dense-8b/` with `[[model_types]] model_type =
"qwen3", hidden_size = 4096`, `KERNEL.toml extra_nvcc_flags = ["-DHDIM=128"]`,
and an **empty kernel dir** — 100% common-set reuse (attention prefill/decode,
GEMM/GEMV, rms_norm, rope are all runtime-dimensioned; no SSM/MoE kernels are
even loaded for a dense model). The two models are byte-identical in geometry
(§0 table), differ only in head usage — and the "head kernel" is **not a kernel**:
reranking reads 2 logit entries through the existing `copy_logits_to_host`, and
embedding pooling is a D2H copy of `hidden_after_norm()` (last-token pooling, which
is what Qwen3-Embedding uses) — host-side normalize. They can NOT reuse an
*existing* target because no target declares model_type `qwen3` — but the new
target costs one MODEL.toml and ~0 nvcc invocations (all cache hits).

**D2 — Quant: bf16, not NVFP4.** The checkpoints are stock BF16; embedding/rerank
quality is cosine-similarity-sensitive and there is no pre-quantized NVFP4
checkpoint to reuse. The base kernel set already contains the full bf16 path
(`dense_gemm_bf16`, `dense_gemv_bf16`, bf16 paged attention, bf16 norms), and
`KernelTarget.quant = "bf16"` is supported by the runtime gates — it has simply
never been built. This makes `qwen3-dense-8b/bf16` the first bf16 target; the
serve/swap quant-compat gate (`serve.rs:113`, `swap_target.rs:82`) must accept it
(one condition, not new machinery). 8B bf16 ≈ 15.2 GiB weights — fits the arena
and the 128 GB UMA comfortably.

**D3 — VL-Embedding/Reranker share the SAME text-tower kernels; the ViT comes as
override modules, not a new family.** The implemented `VisionEncoder` is exactly
this ViT (27 blocks, hidden 1152, deepstack [8,16,24] — §1.1). Vision kernels are
not in `common/`; they ship as model-dir overrides (as in `qwen3-vl-30b-a3b`).
Decision: one additional target `kernels/gb10/qwen3-vl-dense-8b/`
(`model_type = "qwen3_vl"`, hidden 4096) whose bf16 dir carries **only the vision
kernel overrides copied from qwen3-vl-30b-a3b** — text kernels stay 100% common.
(Two targets rather than one multi-`model_types` target so the VL target can ship
vision modules and per-class `[behavior]` without burdening the text-only one.)
Both VL models share this one target with each other — same justification as D1.

Summary: **2 new thin targets (text-dense, VL-dense) sharing everything. Zero
new kernels except the vision overrides copied for the VL target.**

---

## 3. Multimodal preprocessing plug points

- **Images**: already solved — `vision_preprocess.rs` (base64 → decode → patchify)
  is called from the message-assembly seam before templating. The new
  `/v1/embeddings` (VL) and `/v1/rerank` (VL) handlers call the same helper and
  produce the same `image_pixels` + pad-expanded `prompt_tokens`; the ViT splice
  is downstream and unchanged.

---

## 4. What does NOT fit the single-active-model swap — flagged explicitly

The Task 6 model is **one resident model, swap on demand**. That is fine for
"switch chat model", and it technically *works* for embeddings (preseed
Embedding-8B, lazy-swap to it on an /v1/embeddings call, swap back). But **RAG
traffic interleaves embeddings and generation per user request** — with ~7 s
swaps each way this is unusable beyond a demo.

Decision: plan an **auxiliary resident model slot** (Phase 3) rather than bending
the swap machinery:
- A second, small `WeightsArena` (own capacity flag, e.g. `--aux-arena-gb 20`)
  holding one auxiliary model (embedding/reranker class) **concurrently**
  with the primary chat model. UMA budget: 47 GiB chat + 15–18 GiB aux + KV
  still fits 128 GB (KV budget must subtract the aux arena — feeds the existing
  Phase-3 KV-sizing work from the truncation plan).
- Aux models are non-autoregressive (single forward, KV freed immediately —
  §1.3), so they need no persistent KV pool, no decode loop, no drain machinery:
  a lightweight request loop on its own thread + CUDA stream, using
  `prefill_batch_chunk` for batching. The primary scheduler is untouched.
- Registry/kernels: aux target loads via the existing `ensure_target` (append-only,
  already hot-swap-proof). Snapshots/preseed reuse the same `.raw` format —
  arch-agnostic (§1.3).

Until Phase 3 lands, Phases 1–2 ship on the existing swap model (correct, slow
for interleaved traffic — documented limitation).

---

## 5. Dependency-ordered implementation plan

One commit per numbered step, `hotswap:`/`multiclass:` prefix, branch
`modifications`. GPU validation is run by the user; each phase ships a script.

### Phase 1 — Text reranker + embeddings (cheapest win; no new kernels)

**1.0 bf16 target smoke test — GATE for everything below.**
*Status: **PASSED on the box 2026-07-11** (`2648317`) — target compiles as
`(gb10, qwen3-dense-8b, bf16)`, `ptx_for_config("qwen3", 4096)` selects it,
dense_gemm_bf16 / dense_gemv_bf16 / rms_norm all cosine=1.000000, 0 NaN,
non-degenerate. 1.1+ unblocked. Config half of 1.1 already landed early
(`caf0af5`: dense qwen3 + qwen3_vl parsers, 6/6 cached configs parse) plus
servability-gated preseed registration (`5ad90c2`: full 6-model .env list
boots, unservable classes skip with a warning) — both verified on the box.*
`quant=bf16` is supported by the runtime per D2 but has **never been built or
executed** — "supported but never exercised" is exactly the latent-bug category
that bit us with the watchdog stack. Before any dense-loader work: create the
bf16 target skeleton `kernels/gb10/qwen3-dense-8b/bf16` (empty kernel dir +
`KERNEL.toml -DHDIM=128`), extend the two quant-compat gates (`serve.rs:113`,
`swap_target.rs:82`) to accept bf16, and run the smallest feasible bf16 smoke on
the box — route an already-runnable model through the bf16 kernel path (bf16
GEMM/GEMV, bf16 paged attention, norms from the common set) or, if no bf16
checkpoint boots yet, a minimal harness that drives those kernels with real
weights and compares against a reference.
*Acceptance (hard):* a bf16 forward produces correct, non-NaN, non-degenerate
outputs against a reference (greedy tokens or logits within tolerance of a known
CPU/BF16 reference for a fixed prompt); build logs show the bf16 target compiling
with ~0 new nvcc invocations. **If bf16 surprises, we learn it here — at step
1.0, not after half of Phase 1 is built.** No 1.1+ work starts before this
passes.

**1.1 Dense-Qwen3 model support.**
*Status: implemented (`b24b41d`, config half `caf0af5`) — awaiting on-box boot
smoke of Reranker-8B. Deviations captured in the commit: all-BF16 via the
Gemma-4 pattern (NVFP4 placeholders shadowed), dense_minus_one for the
standard-convention norms, per-store weight-name prefix (the Embedding
checkpoint is an unprefixed sentence-transformers export).*
`Qwen3DenseWeightLoader` modeled on `gemma4/loader_a.rs`
(`Qwen3AttentionLayer::new_ungated` + `FfnComponent::Dense`, standard qwen3
weight names); match arm `"qwen3"` in `factory.rs::loader_for_config`;
`load_lm_head` must tolerate the Embedding checkpoint having **no lm_head**
(Reranker has one; Embedding pools hidden states). ModelConfig parse for plain
`qwen3` (trivial — subset of qwen3_next).
*Acceptance:* `cargo test` green; boot Reranker-8B on the box, prompt-too-long
and basic chat-completion smoke works (it is a causal LM — it can literally chat).

**1.2 Finalize target `gb10/qwen3-dense-8b/bf16`** (per D1/D2; skeleton and
quant gates already landed + smoke-verified in 1.0): complete the MODEL.toml
(`[[model_types]] qwen3/4096`, minimal `[behavior]` — watchdogs off, no
thinking, sampling presets).
*Acceptance:* boot of a `qwen3`/4096 checkpoint selects the target via
`ptx_for_config`; behavior audit logs show watchdogs off.

**1.3 Non-generative request path.**
New `InferenceRequest::Pooled { prompt_tokens, pooling: LastToken|MeanNorm|
LogitPair(u32,u32), reply: oneshot }` (or a flag on Blocking); scheduler branch in
the two immediate-finish sites returns after prefill; new Model methods
`copy_hidden_to_host` (over `hidden_after_norm()` — last-token pooled) and the
LogitPair read via existing `copy_logits_to_host`. Batching: route through
`run_batched_prefill_step` untouched (outputs per-stream). Home for pooling/
normalize helpers: the reserved `atlas-embed` crate.
*Acceptance:* unit tests with mock GPU; embedding vector for a fixed prompt is
identical across two runs and across batch-of-1 vs batch-of-8 (bitwise, temp-0
path has no sampling).

**1.4 Endpoints.**
Replace the `/v1/embeddings` stub (OpenAI shape: `input: str|[str]`, returns
float arrays, L2-normalized; instruction-prefix support per Qwen3-Embedding
usage); add `/v1/rerank` (renders the Qwen3-Reranker chat template per
(query, document) pair, scores = softmax over yes/no token logits). Both go
through the lazy-swap interceptor (`ensure_model_active`) so preseeded
embedding/reranker models swap in on demand — registry/preseed/snapshot reuse
is automatic (name-keyed arena).
*Acceptance:* `scripts/embed_rerank_test.sh` — preseed chat+embedding+reranker;
/v1/embeddings returns 4096-dim normalized vectors, self-similarity ≈ 1.0,
related > unrelated similarity; /v1/rerank orders an obvious (query, docs) set
correctly; swap back to chat still bit-identical (existing A/B methodology).

### Phase 2 — VL embedding + reranker (reuses the wired vision path)

**2.1 Target `gb10/qwen3-vl-dense-8b/bf16`** (per D3): vision override kernels
copied from `qwen3-vl-30b-a3b`, text from common.
**2.2 Loader wiring**: `qwen3_vl` (non-MoE, dense FFN) arm — reuse
`Qwen3VLWeightLoader::load_vision_encoder` for the tower + the Phase-1 dense text
loader; verify `config.vision` parses from these checkpoints (deepstack indexes
match the implemented encoder).
**2.3 Endpoint input**: accept `image_url` content parts in /v1/embeddings and
/v1/rerank documents; call the existing `vision_preprocess` + pad-expansion seam;
`Pooled` requests carry `image_pixels` (field already exists on requests).
*Acceptance:* `scripts/vl_embed_test.sh` — image and its caption embed closer
than image vs unrelated text; text-only inputs on the VL model match Phase-1
behavior; serve.rs must NOT log "serving TEXT-ONLY" (vision module present).

### Phase 3 — Auxiliary resident model slot (the RAG enabler; see §4)

**3.0 BLOCKING PRECONDITION (hard gate, not a parallel wish):** the
deterministic per-model KV-pool sizing from the truncation plan (its Phase 3:
per-model `inference_reserve`/budgets on the swap path + deterministic KV budget
instead of the ambient-free-memory fallback) **MUST be completed and verified on
the box BEFORE 3.1 starts.** Rationale: the aux slot only works correctly if the
primary KV budget provably subtracts the aux arena — with today's
"auto-measure implausible → raw used_so_far" fallback, adding a 15–18 GiB aux
arena would silently shrink (or OOM) the primary model's KV in a
non-deterministic way. Verification = the truncation-plan Phase-3 acceptance
(same model ⇒ identical block count on every load; no "auto-measure implausible"
in logs).

**3.1** Second arena + aux model lifecycle (load/unload admin API,
`--aux-model`/`ATLAS_AUX_MODEL`, preseed integration).
**3.2** Aux request loop (own thread + stream, `prefill_batch_chunk`, no decode);
route /v1/embeddings and /v1/rerank to the aux slot when the requested model is
aux-resident, else fall back to lazy-swap.
**3.3** KV budget integration: primary KV sizing subtracts the aux arena
(builds directly on the 3.0-gated deterministic-KV work).
*Acceptance:* `scripts/rag_concurrency_test.sh` — interleaved chat + embedding
requests with ZERO model swaps (swap counter flat), chat latency unaffected
within noise, aux embeddings identical to Phase-1 values; **plus:** the primary
KV budget demonstrably subtracts the aux arena (logged block count with aux
loaded = block count without aux minus exactly the aux-arena share), and a boot
with chat + aux shows neither OOM nor silent KV shrinkage (block count matches
the deterministic prediction).

### Explicit infra reuse (all phases)
Arena/snapshot/preseed/lazy-swap apply unchanged to every model here: weights are
name-keyed (`WeightStore`/`WeightsArena`/`.raw` snapshots are arch-agnostic), the
registry accepts any (id, hash), and `ensure_target` loads new kernel targets on
first swap. The only touched invariants: quant gate learns bf16 (1.2), KV budget
learns the aux arena (3.3).

### Deliberately out of scope
Audio entirely (whisper-large-v3 transcription descoped 2026-07-11 to focus on
the embedding/reranker classes; `/v1/audio/*` stays a 501 stub), image
generation, EP/world_size>1 for aux models, quantizing the embedding models to
NVFP4 (revisit only if bf16 latency disappoints).

---

## 6. Risk register

| Risk | Phase | Mitigation |
|---|---|---|
| First-ever bf16 target exposes untested kernel paths | **1.0 (gate)** | front-loaded as step 1.0: bf16 skeleton + quant gates + on-box smoke with hard non-NaN/reference acceptance BEFORE any loader work |
| Embedding checkpoint lacks lm_head → loader assumptions | 1.1 | make `load_lm_head` optional; Pooled path never calls lm_head |
| Aux slot memory pressure silently shrinks / OOMs primary KV | **3.0 (blocking gate)** | deterministic KV sizing (truncation-plan Phase 3) MUST be completed and verified before 3.1 starts; 3.3 acceptance proves the aux-arena subtraction explicitly |
| Qwen3-VL config variance (deepstack/dtype differs between the two VL checkpoints: bf16 vs f32 vision dtype) | 2.2 | config-parse test against both cached config.json before loader work |

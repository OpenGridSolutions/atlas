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
*Status: implemented (`1e556e9` logit probe, `2f71b89` pooled hidden) —
as flags on Blocking requests (`logit_probe`, `capture_hidden`) rather than a
new variant; wired at all seven first-token sampling sites; batched-prefill
paths refuse hidden capture (shared norm buffer).*
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
*Status: implemented (`dd8ba4d` /v1/rerank + /rerank, `1e6002b`
/v1/embeddings replacing the 501 stub) — awaiting on-box
`scripts/embed_rerank_test.sh` against the running server.*
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
*Status: implemented (`5b43425` target + loader, `47ac302` multimodal inputs)
— awaiting on-box `scripts/vl_embed_test.sh`. Note: the ViT's deepstack taps
are computed but LLM-unused (pre-existing, applies to VL-MoE serving too).*

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

## Phase W — Whisper speech-to-text (re-scoped 2026-07-11, after Phase 2)

### W.0 GATE FINDING — encoder-decoder feasibility (checked FIRST, per mandate)

**Verdict: ADDITIVE — no fundamental rebuild of Atlas' abstractions. Proceeding
autonomously.** Grounds:

1. **The decoder-only Model/scheduler stack cannot carry Whisper cheaply**: the
   `Model` trait has 30 REQUIRED methods (of 74), shaped around paged KV, SSM
   checkpoints, MTP/speculative and spill — a `WhisperModel: Model` would be
   ~16 real methods + 14 landmine stubs, and the scheduler's prefill/decode
   phases assume token prompts, not audio. Forcing whisper through it touches
   the paths we must not destabilize.
2. **Atlas already ships the escape hatch**: the `VisionEncoder`
   (`layers/vision_encoder/enc_impl/*`) is a self-contained GPU transformer
   encoder OUTSIDE the Model trait — bias-ful GEMMs (`vision_gemm_bias`),
   GELU, NON-causal attention at an odd head_dim (72), own kernel module,
   driven synchronously. Whisper's audio encoder (conv×2 + 32 bidirectional
   layers, head_dim 64, sinusoidal pos-emb) is architecturally the same
   animal; the decoder (32 layers, causal self-attn ≤448 pos + cross-attn
   over a STATIC 1500-frame encoder output + GELU MLP) is small enough
   (per-seq KV = [448, 20, 64] × 2 × 32 L ≈ 88 MB bf16 dense buffers) to run
   as a plain greedy loop in the same module — no paged KV, no scheduler.
3. **Consequence — Whisper as a resident sidecar module**, invoked directly by
   the /v1/audio/transcriptions handler behind its own async mutex: it never
   enters the scheduler, the hot-swap arena, or the registry. ~3.2 GiB BF16
   weights stay resident next to the chat model (128 GB UMA absorbs it), so
   transcription NEVER forces a 7 s model swap and the active chat model
   stays untouched. **Deliberate deviation from "Lazy-Swap-Verhalten wie
   gehabt"** — a swap-based whisper would evict the chat model on every
   transcription; resident is strictly better and additive.
4. **Kernels**: one genuinely new kernel is required — `whisper_attention`
   (bf16, runtime `causal` flag, head_dim 64, Q/K/V as dense buffers; naive
   tiled implementation suffices: encoder worst case ≈ 0.4 TFLOP per 30 s
   chunk, milliseconds on GB10). Everything else composes from existing
   kernels: `dense_gemm_bf16`/`gemm` (+ host-side bias add fused via the
   vision module's `vision_gemm_bias` pattern), `gelu` (gemma override),
   conv frontend via host im2col (3000×128 mel is 1.5 MB) + GEMM.
5. **Precision (deviation from "F16 nativ")**: the weight pipeline has NO
   FP16 dtype (BF16/FP32/FP8 only) — and the cached checkpoint ships
   `model.fp32-*.safetensors` (FP32, not F16). Loading converts FP32→BF16
   (lossless vs F16 in exponent, −3 mantissa bits); compute is BF16. The
   requirement's core intent — do NOT route through runtime BF16→NVFP4
   quantization — is honored: no quantization anywhere in the whisper path.

### W.1 Audio frontend (CPU)
`spark-model/src/audio_preprocess.rs`: multipart bytes → decode (`symphonia`:
wav/mp3/flac/m4a/ogg) → resample to 16 kHz mono (linear) → log-mel spectrogram
(**128 mel bins — verified against HF config `num_mel_bins: 128`**; n_fft 400,
hop 160, HF Whisper mel filterbank formula, log10 + max-8dB clamp + /4 scaling
per reference) → `[n_frames, 128]` f32, padded/split to 3000-frame (30 s)
windows. Pure CPU, unit-tested against golden values (synthetic sine).

### W.2 Whisper module (GPU, VisionEncoder pattern)
`spark-model/src/whisper/`: weight loader (name-keyed from the store —
`model.encoder.*` / `model.decoder.*`, FP32→BF16 via `dense_f32_safe`),
encoder forward (conv1(s1)+GELU → conv2(s2)+GELU → +sinusoid pos →
32×[LN, self-attn(non-causal), LN, MLP-GELU] → LN), decoder greedy loop
(SOT/lang/task/notimestamps protocol from `generation_config.json`, causal
self-attn with dense KV append, cross-attn over the per-chunk encoder output,
suppress/begin-suppress token sets, EOT stop, max 448). Kernel target
`gb10/whisper-large-v3/bf16`: `whisper_attention.cu` (new) + `gelu.cu`
(gemma copy) + w4a16_gemm.cu (eager-symbol standard copy) on top of common.
Chunking **stage 1: fixed 30 s slicing**, transcripts concatenated in order —
the module API takes `Vec<MelChunk>` so overlapping/merging is a later
additive change (documented).

### W.3 Serving
Resident sidecar: loaded on first /v1/audio/transcriptions request (guarded by
an async `OnceCell`-style mutex; weights via the existing model_resolver +
WeightStore, NOT the hot-swap arena), then kept for process lifetime.
Endpoint: OpenAI multipart (`file`, `model`, optional `language`,
`response_format` json|text|verbose_json subset). Whisper is NOT added to the
preseed registry (it can't be scheduler-served); the resolver accepts
`openai/whisper-large-v3` and the endpoint 404s other model ids with a clear
message.

### W.4 Acceptance (scripts/whisper_test.sh; temp-file response reads)
Fixtures: deterministic synthetic speech is unreliable — fixture strategy is
espeak-ng if present on the box, else a checked-in generated WAV
(`test_data/whisper_fixture.wav`, spoken digits via espeak at build time of
the fixture, committed once). A1 short audio → non-empty text; A2 keyword
check (case-insensitive, no WER obsession); A3 >30 s audio (fixture repeated
+ distinct second-half keyword) → late keyword present (chunking works); A4
the 6 existing models still answer afterwards (S1 pattern).

### W status — **ACCEPTED on the box 2026-07-12**
*W.1 `9af625d` · W.2a `881d5a9` (kernel microtest PASS on box) · W.2b
`13043b4` · W.3 `b7d4894` · W.4 `01eceea` · config-parser fix `e3da13c`
(whisper arm — no top-level hidden_size; preseed now skips with the honest
"sidecar-served" message instead of a parse error; decoder-only parses
untouched, 7/7 scheduler models register).*
*Acceptance run: `whisper_test.sh` 4/4 PASS — A1/A2 word-perfect JFK
transcript ("And so my fellow Americans, ask not what your country can do for
you…"), A3 both 30 s chunks transcribed (4× keyword), A4 chat unaffected.
Sidecar init 3.7 s (fp32→BF16, store released), ~3.7 s per 30 s chunk.*

### W risks
| Risk | Mitigation |
|---|---|
| New whisper_attention kernel correctness | CPU-reference microtest (bf16_target_smoke pattern) BEFORE end-to-end |
| Mel filterbank mismatch vs HF | golden-value unit test from the HF formula; keywords-not-WER acceptance |
| FP32 checkpoint is 6.2 GiB on disk | loads once, converts to ~3.2 GiB BF16 resident |
| Sidecar memory vs KV budget | 3.2 GiB; boot order note — start Atlas after, or accept the self-relative KV measure absorbing it |

## Phase K — categorized preseed (2026-07-12) + Phase T — TranslateGemma

### T.0 GATE FINDING (checked FIRST) — Gemma-3 text backbone loadability

**Verdict: ADDITIVE — proceeding autonomously.** Grounds, from the cached
`google/translategemma-12b-it` checkpoint:
1. Config is `model_type: gemma3` with NESTED `text_config` and **no top-level
   `hidden_size`** — the exact whisper-class parser failure. Fix is the known
   recipe: a small schema-specific `gemma3` dispatch arm (text_config: hidden
   3840, 48 L, 16 Q/8 KV, head_dim 256, GeGLU, `_sliding_window_pattern: 6`
   → 5 sliding : 1 full layer_types, dual rope theta via
   `rope_local_base_freq` 10k / `rope_theta` 1M, vocab 262208). No fundamental
   parser rebuild.
2. Weight layout is EXACTLY the Gemma-4 loader's structure: prefix
   `language_model.model.*` (auto_detect_weight_prefix's first branch), the
   same 4 norms (input/post_attention/pre_feedforward/post_feedforward), q/k
   norms, GeGLU mlp gate/up/down. Vision tower present on disk but ignored
   (name-keyed loads; text-only per requirement). `Gemma4WeightLoader` is
   reusable with three conditional divergences: (a) attention scale — gemma4
   hardcodes 1.0, gemma3 needs the default 1/√head_dim (query_pre_attn_scalar
   256 == head_dim); (b) no `v_norm` tensors (guard on presence); (c) no
   logit softcapping (`final_logit_softcapping: null`).
3. Kernel target: new thin `gb10/translategemma-12b/bf16` (model_type
   gemma3/3840). head_dim 256 uniform = the common HDIM default — NO flag
   variant needed. Overrides copied from gemma-4-31b where eagerly resolved
   (w4a16_gemm, gelu, embed_scale; NOT rms_norm.cu — gemma3 stores
   zero-centered norm weights, matching common's (1+w) kernel as-is).
4. BF16 end to end (checkpoint is stock BF16); NO NVFP4 runtime quantization.
   ~24 GiB in the arena — served via arena + lazy swap like the Qwen models,
   under the TRANSLATE category.

### K — five category ENVs replace ATLAS_PRESEED_MODELS (REMOVED, no fallback)
`ATLAS_PRESEED_GENERATIVE|EMBED|RERANK|TRANSCRIBE|TRANSLATE`, each a JSON
list (same syntax as before). `RegisteredModel` gains `category`;
`ensure_model_active` + the embed/rerank/translate handlers enforce
endpoint↔category with an explaining 400 ("model X is categorized as embed
and not available on /v1/chat/completions") — no silent fallback, no
role-guessing. GENERATIVE/EMBED/RERANK/TRANSLATE stay arena+lazy-swap;
TRANSCRIBE stays the Whisper sidecar (never registered in the swap registry —
the endpoint validates the id against the TRANSCRIBE list; real name AND the
whisper-1 alias accepted). `.devpod/compose.yml` switches to the five vars.
CLI `--preseed-model` maps to GENERATIVE.

### T — /v1/translate (forced TranslateGemma template)
Request `{source_lang_code, target_lang_code, text}` → server-side FORCED
prompt frame extracted from the checkpoint's own chat_template.jinja
(`<start_of_turn>user\nProduce only the {Target} translation, … Please
translate the following {Source} text into {Target}:\n\n\n{text}
<end_of_turn>\n<start_of_turn>model\n`, language NAMES resolved from the
template's embedded code→name map, parsed at init — the generic chat
template path is bypassed entirely, avoiding the known "content as an
iterable with exactly one item" bug). Blocking request, temp 0, eos [1,106].
**2K limit: REJECT with 400** when the rendered prompt exceeds 1800 tokens
(chunked translation splices sentences and silently degrades quality; overlap
chunking stays a later additive feature). Response `{"text": …}`.

### K/T acceptance — scripts/categories_translate_test.sh
C1 real names on the right endpoints; C2 wrong endpoint → explaining 400;
C3 openai/whisper-large-v3 (real name) transcribes; T1 de→en keyword; T2
de→fr French keyword; T3 template enforcement evidence (translate refuses on
/v1/chat/completions per category AND the forced frame is logged/inspectable);
R1 all existing classes still answer (S1/A4 pattern). Boot log must show
per-category registration counts and the whisper sidecar note.

### K/T status (2026-07-12)
DONE end to end. Boot log: five per-category preseed lines, whisper
sidecar note, translategemma snapshot written first-boot (22.70 GiB,
1065 tensors, 7.5 s), 8 models registered, server idle.
`categories_translate_test.sh`: **15 PASS, 0 FAIL** — C1a/b real names,
C2a–d categorized 400s on all four wrong-endpoint combinations, C3 real
whisper id, T1 "The cat is sitting on the table and drinking milk.",
T2 "Le chien dort dans le jardin, sous le grand arbre.", T3a/b forced
template, R1a–d all categories. ATLAS_PRESEED_MODELS gone from
compose.yml (serve.sh bridges stale container envs with unset +
category defaults until the container is recreated).

### K/T risks
| Risk | Mitigation |
|---|---|
| gemma4-loader divergences (scale/v_norm/softcap) subtly wrong for gemma3 | boot chat-smoke of translategemma (causal LM) before endpoint work; wrong scale ⇒ immediate gibberish |
| eager kernel modules missing from thin target (w4a16 lesson) | copy gemma-4-31b overrides preemptively; boot smoke catches the rest |
| category refactor breaks the 7 existing registrations | R1 regression + full test-script battery after restart |

## Phase S — /v1/stats memory diagnostics (2026-07-12)

Read-only GET endpoint that makes the KNOWN GB10 measurement bug visible
as a number: on unified memory, `cudaMemGetInfo` under-reports free
memory (swap-reclaimable DRAM is invisible — NVIDIA DGX Spark known
issue), so the self-relative KV budget in `spark_model::factory::build`
gets a fluctuating ~19–20 GB "free-now" instead of ~65 GB, sizes the KV
pool at 960–1348 blocks, and translategemma-12b currently fails to load
("KV cache can hold at most 0 concurrent sequence(s)"). This phase does
NOT fix the budget — it ships the metric (`discrepancy_gb`) the later
/proc/meminfo budget fix will be verified against. TranslateGemma
continuing to fail to load is EXPECTED and correct here.

### S.1 Design
- **`GET /v1/stats`** (api/stats.rs, routed next to /health; no auth
  beyond the global middleware). All numbers come from what Atlas
  already computed at load time — no new measurement machinery.
- **Memory block** (the success metric):
  - `cuda_mem_get_info_free_gb` — live `cuMemGetInfo` from the HTTP
    thread. CUDA contexts are shareable across threads; the process-wide
    context is stored once at backend init
    (`spark_runtime::cuda_backend` global, BASELINE_FREE_BYTES
    precedent) and bound on demand (`cuCtxGetCurrent` → if none,
    `cuCtxSetCurrent(stored)`), then `cuMemGetInfo_v2`. No scheduler
    round-trip, works while the server idles parked.
  - `proc_meminfo_available_gb` — NEW `spark_runtime::meminfo::read()`:
    `MemAvailable + SwapFree`; HugePages special case (HugePages_Total
    != 0 → `HugePages_Free × Hugepagesize`, SwapFree = 0). Deliberately
    placed in spark-runtime: the SEPARATE budget fix will call the same
    reader from factory::build. Display-only here — budget math
    untouched.
  - `discrepancy_gb` = proc − cuda (large ~40+ GB and fluctuating until
    the fix; small/stable after — THE metric), `total_unified_memory_gb`
    from cuMemGetInfo total (fallback MemTotal).
- **Per-model block**: hotswap registry (id, category, Phase K) ×
  live residency:
  - arena model: `weights_bytes = ctx.arena.used()` (the boot-log
    "Weights arena" number), `residency: "arena"`.
  - KV numbers via NEW `spark_runtime::kv_stats` process-global atomics
    (one PagedKvCache exists at a time): `new()` publishes
    total_blocks + exact allocated pool bytes; every free-list
    mutation (alloc_block / try_alloc_block / dec_ref-free /
    return_evicted_block) publishes the free count (single mutator
    thread, Relaxed stores — negligible); `Drop` clears. Handler reads
    reserved_blocks/bytes + `kv_used_blocks = total − free` (live per
    call).
  - whisper sidecar: separate entry (`residency: "sidecar"`), resident
    iff the OnceCell is initialized; `weights_bytes` = Σ tensor elems ×
    2 (BF16) computed once at sidecar load in audio.rs; KV fields 0
    (sidecar manages its own buffers, no paged pool).
  - registered non-resident models: `resident: false`, zero usage.

### S.2 Acceptance — scripts/stats_test.sh (temp-file convention)
S1 valid JSON with both memory numbers + discrepancy_gb; S2
`discrepancy_gb > 20` (documents the UNFIXED state; the later budget fix
flips this same check to small/stable); S3 after a chat request the
generative model shows resident with plausible weights_bytes and
kv_reserved_blocks > 0; S4 after a transcribe the sidecar entry shows
residency=sidecar. Regression: embed_rerank_test.sh + whisper_test.sh
stay green.

### S.3 Risks
| Risk | Mitigation |
|---|---|
| binding the CUDA ctx on HTTP worker threads disturbs the scheduler | contexts are per-thread CURRENT settings; setting it on extra threads never unbinds the scheduler thread's binding; read-only cuMemGetInfo only |
| kv_stats atomics drift from the real free list | single choke: every `free_blocks` mutation site sets the count from `free_blocks.len()` (not incremental deltas) |
| stale stats after unload/swap | PagedKvCache::drop clears; arena bytes read live from HotswapCtx |

## Phase V — ig1/Qwen3-VL-30B Instruct: forced-thinking misconfig (2026-07-12)

Pre-existing UPSTREAM bug (commit 743f946 sweep, 2026-06-09), not a
session regression — the session diff on the chat/template path is the
Phase-K category parameter only. Symptoms (empty content, full
reasoning_content, invented "Human:/Assistant:" turns, runaway past the
turn end at temperature 0) were NOT template/EOS problems: the log shows
the checkpoint's own ChatML template rendering correctly and EOS
[151645, 151643] registered. Root cause: the qwen3-vl-30b-a3b target's
MODEL.toml set `thinking_default = true` although its hf_id is the
INSTRUCT variant with no thinking mode. With thinking enabled the
scheduler starts `inside_thinking` (prefill_b_step), every token lands
in reasoning_content, the never-emitted </think> runs the budget out,
the injected </think> puts the model out of distribution and it drifts
into base-model dialog hallucination. Fix: `thinking_default = false`
(one line + comment); a genuine Thinking-VL checkpoint on this target
still gets thinking via the template-forced detection (unclosed <think>
in the prompt tail — the qwen3-next-80b pattern). Verified at temp 0:
text-only → content 'OK' (2 tokens, reasoning empty); image (red field,
white square) → correct one-sentence answer in content; 80B + Coder
regression → 'pong' each.

## Phase Q — Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 replaces ig1 (planned 2026-07-12)

ig1's NVFP4 quant has a defective vision encoder; the official Qwen FP8
release replaces it. Checkpoint is ALREADY CACHED (31 GB blobs).

### Q.0 Feasibility findings (verified against the cached checkpoint)
What carries over UNCHANGED:
- config `model_type: qwen3_vl_moe`, text hidden 2048, 128 experts —
  same config parser arm, same `Qwen3VLWeightLoader` dispatch, same
  kernel target `qwen3-vl-30b-a3b` (`quant_pair_compatible(nvfp4, fp8)`
  is true; `canonicalize_model_quant` maps `quant_method: "fp8"` →
  "fp8"). NO new kernel target, NO parser change, NO endpoint change.
- Vision tower ships BF16 (every `visual.*` module is in
  `ignored_layers`; 351 vision tensors, 0 scale tensors) — the existing
  ViT loader path reads it as-is. lm_head ignored → BF16 too.
- Phase-V thinking fix lives on the TARGET (MODEL.toml), so it applies
  to the new checkpoint automatically.
- Arena/snapshot/lazy-swap/preseed: name-keyed, quant-agnostic.

What does NOT load today (the actual work):
1. **Fused-stacked FP8 experts** — the checkpoint uses transformers'
   new fused serialization: `experts.gate_up_proj` [E=128, 2048, 1536]
   FP8-E4M3 + `gate_up_proj_scale_inv` [128, 16, 12] F32 (128×128
   blocks), `experts.down_proj` [128, 768, 2048] + [128, 6, 16].
   Atlas main-layer MoE loading (`load_moe_no_shared`) reads ONLY the
   per-expert split names (`experts.{e}.gate_proj.weight`, …); the
   stacked branch exists solely in the MTP path (loaders_moe.rs (B))
   and only for BF16 without scales. Axis orientation must be verified
   against transformers' packed layout before slicing (the [E, 2048,
   1536] vs MTP's assumed [E, 2*I, H] disagree — 2048 = H here, 1536 =
   2×768 = 2I, i.e. stored transposed).
2. **Block-scaled FP8 attention** — q/k/v/o are FP8-E4M3 +
   `weight_scale_inv` F32 [32, 16] (DeepSeek-V3 convention). Building
   blocks exist (`loaders_fp8.rs` handles `weight_scale_inv`,
   `dequant_fp8_blockscaled_to_bf16` kernel, MiniMax expert path) but
   the qwen3_vl attention load path currently assumes BF16/NVFP4.

### Q.1 Implementation steps
1. **Loader, experts**: add a fused-stacked FP8 branch to the VL MoE
   path. Preferred route (memory): dequant each expert slice
   FP8→BF16 (block scales) → runtime-quantize NVFP4 → free transients,
   exactly the MiniMax M2 pattern (ssm_qwen35_more.rs `quant` closure)
   — lands at ~17 GB experts, proven kernels. Native-FP8 fused-MoE
   serving (qwen3.rs `native_fp8`) is the perf upgrade but needs the
   per-expert-split names + kernel-scale-format repack — defer unless
   the dequant route misbehaves. Gate/up split per verified axis
   orientation; transpose into the [out, in] convention during dequant.
2. **Loader, attention**: dequant q/k/v/o FP8→BF16 via the block-scaled
   helper at load; keep the BF16 attention path (dense_gemv), matching
   how gemma3/qwen3_dense serve attention. Small memory cost, uniform
   precision behaviour.
3. **Ops swap**: replace ig1 with Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 in
   ATLAS_PRESEED_GENERATIVE (compose.yml + serve.sh bridge defaults),
   update the target MODEL.toml `hf_id` doc line. Boot preseed writes
   the new snapshot; delete the stale ig1 snapshot file to reclaim
   ~17 GB (optional; ig1 stays lazily loadable until removed).
4. **Acceptance** (the vision fix is the point of the swap):
   - Q1 text-only temp 0 → content "OK", no invented turns (Phase-V
     check on the new checkpoint).
   - Q2 image color/shape (red field, white square) → correct answer.
   - Q3 VISION QUALITY vs the defective ig1 encoder: OCR-style read of
     structured fixture images (scripts/gen_test_images.py shapes) —
     description must name rectangle/ellipse/diagonal correctly.
   - Q4 regression: 80B, Coder, embed/rerank/whisper/translate suites.

### Q.V2 — vision token budgets (follow-up 2026-07-12)
Control calc against HF smart_resize proved the first min-pixels fix
faithful to the HF floor (264×86 → 448×160 → 70 vision tokens + 28 text
= 98, measured exactly) — but the HF 65536 floor itself is too coarse
for OCR (formula read as one glyph), and the 1280-px long-side cap
(rooted in the ViT's fixed p_max=6400 scratch) broke wide documents
(2500×1200 → 760 tokens vs 2964 per HF reference). Fix: OCR floor
raised to 262144 px (≥256 tokens; 264×86 → 290), long-side cap replaced
by a pixel budget of 4.19M px = 16384 patches = 4096 tokens with the
ViT scratch (p_max, incl. the [seq,seq] SDPA matrix → ~1.6 GB) sized
from the same shared constant, HF-faithful rounding (ceil up / floor
down), explicit --vision-max-pixels still wins below the hard bound.
Expected: formula ~290+28 tokens and structurally readable; 2500-px
document pages native ≈ 3k tokens; token counts for previously
side-capped large images INCREASE by design.

### Q.2 Risks
| Risk | Mitigation |
|---|---|
| fused-tensor axis orientation guessed wrong | verify against transformers Qwen3VLMoe packed layout + a one-expert dequant checksum vs HF before wiring the splitter |
| 128×128 block-scale grid misapplied on non-multiple dims (1536/128=12 exact here — fine, but 768/128=6 too) | shape-assert scale grid == ceil(dim/128) per axis at load |
| dequant transients spike arena/UMA during load | free FP8 source + BF16 transient per expert (MiniMax pattern), stream-serial |
| ig1 snapshot/registry leftovers confuse ops | explicit compose/serve.sh swap + optional snapshot delete, boot log shows the new id under [generative] |

## 6. Risk register

| Risk | Phase | Mitigation |
|---|---|---|
| First-ever bf16 target exposes untested kernel paths | **1.0 (gate)** | front-loaded as step 1.0: bf16 skeleton + quant gates + on-box smoke with hard non-NaN/reference acceptance BEFORE any loader work |
| Embedding checkpoint lacks lm_head → loader assumptions | 1.1 | make `load_lm_head` optional; Pooled path never calls lm_head |
| Aux slot memory pressure silently shrinks / OOMs primary KV | **3.0 (blocking gate)** | deterministic KV sizing (truncation-plan Phase 3) MUST be completed and verified before 3.1 starts; 3.3 acceptance proves the aux-arena subtraction explicitly |
| Qwen3-VL config variance (deepstack/dtype differs between the two VL checkpoints: bf16 vs f32 vision dtype) | 2.2 | config-parse test against both cached config.json before loader work |

# Atlas Architecture

A 5-minute tour of the crate graph, the build pipeline, and the request lifecycle.

## Crate graph

Atlas is a single Cargo workspace with 15 crates organized into four layers:

```
                         ┌─────────────────────────┐
   HTTP / scheduling     │      spark-server       │
                         │  (OpenAI + Anthropic    │
                         │  API, scheduler, tool   │
                         │  parsing, grammar)      │
                         └──────────┬──────────────┘
                                    │
                         ┌──────────┴──────────────┐
   model + layers        │       spark-model       │
                         │  (TransformerModel,     │
                         │  per-arch layers, MoE,  │
                         │  SSM, attention, MTP,   │
                         │  weight loading)        │
                         └──────────┬──────────────┘
                                    │
            ┌───────────────────────┼─────────────────────┐
            │                       │                     │
   GPU runtime          inter-rank comm           NVMe-backed swap
   ┌────────┴────────┐ ┌────────────┴───────┐  ┌──────────┴────────┐
   │  spark-runtime  │ │    spark-comm      │  │  spark-storage    │
   │  (PagedKvCache, │ │  (NCCL backend +   │  │  (HSS, io_uring)  │
   │  buffers,       │ │  RDMA, all_reduce, │  └───────────────────┘
   │  cuda_backend,  │ │  broadcast)        │
   │  sampler,       │ └────────────────────┘
   │  weights)       │
   └────────┬────────┘
            │
   ┌────────┴────────┐
   │  atlas-kernels  │   PTX bytes embedded at compile time
   │  (build.rs:     │   per (hardware, model, quant) tuple
   │  compile *.cu   │   from kernels/<hw>/<model>/<quant>/
   │  → PTX → embed) │
   └────────┬────────┘
            │
   ┌────────┴────────┐    Stable types + traits shared across the whole tree
   │  atlas-core     │    — ModelConfig, registry, hardware capability
   │                 │    enums, error types.
   └─────────────────┘

   Placeholder crates (not yet wired into the hot path):
   atlas-activation · atlas-norm · atlas-quant · atlas-embed · atlas-reduce
   These exist as future homes for host-side primitives and are
   currently thin scaffolding so the dependency graph is stable.

   Test/bench crates:
   atlas-spark-bench  (Criterion microbenchmarks per kernel category)
   cufile-sys         (FFI for NVIDIA cuFile / GDS — currently dormant on GB10)
```

## Layer-by-layer

### `atlas-core`
The shared "vocabulary" crate. Defines:
- `ModelConfig` — every supported model's HuggingFace `config.json` parsed into a normalized struct, plus per-family overrides (Qwen3.5 / Qwen3.6 / Nemotron / Gemma-4 / Mistral / MiniMax).
- `LayerType` — `FullAttention | LinearAttention | Mlp | Moe | Dense`.
- `KernelTarget` / `TargetPtxSet` — what kernels the binary was compiled with.
- Public error / result types.

Every other crate depends on this. **Keep it stable.** Breaking changes here ripple everywhere.

### `atlas-kernels`
A pure-build-script crate whose `build.rs` (~915 LoC) does the heavy lifting:

1. Reads `kernels/<hw>/HARDWARE.toml` for arch, sm, fp32-residual flag.
2. Reads `kernels/<hw>/<model>/MODEL.toml` for model-specific kernel-target metadata, sampling presets, behavior knobs.
3. For each `(hardware, model, quant)` tuple selected via `ATLAS_TARGET_*` env vars (or `*` wildcard), compiles every `*.cu` under `kernels/<hw>/<model>/<quant>/` and `kernels/<hw>/<quant>/` (shared) to PTX via `nvcc`, deduplicating model-specific overrides over the shared pool.
4. Emits a generated `target_ptx.rs` containing `static PTX_BYTES: &[(&str, &[u8])]` arrays — the kernel modules as PTX strings linked into the binary.
5. Marks `cargo:rerun-if-changed=` on every `.cu` and `.toml` it touched so incremental builds work.

Skip mode (`ATLAS_SKIP_BUILD=1`): emits a stub registry. Used by CI for GPU-free `cargo check` / `clippy` runs.

### `spark-runtime`
GPU-side runtime primitives:
- `cuda_backend.rs` — `GpuBackend` trait + `AtlasCudaBackend` (cudarc-based) impl. Wraps device pointers, kernel handles, streams, allocator. `cuda_backend/alloc_tracker.rs` counts every device allocation as build, sidecar or late (`docs/design/memory-model.md`).
- `buffers.rs` — pinned-host + device scratch arenas sized per-batch.
- `kv_cache.rs` — paged KV cache with FP8 / NVFP4 / BF16 dtype variants.
- `sampler.rs` — host-side sampling (top-k, top-p, temperature, repetition penalty, DRY, Lloyd–Max). The GPU produces logits; host samples.
- `weights.rs` — safetensors loader with shard streaming + OOM guard.

Hardware-agnostic: the `GpuBackend` trait could be implemented by another vendor backend. Today only `AtlasCudaBackend` exists.

### `spark-comm`
NCCL-backed multi-rank communication:
- `nccl_backend.rs` — collective ops (`all_reduce`, `broadcast`, `all_gather`) with RoCEv2 wiring.
- Used by EP/TP layers to exchange expert outputs / sharded attention outputs.

### `spark-storage`
NVMe-backed KV cache swap (high-speed swap). When the on-device KV cache fills, blocks evict to disk via `io_uring` and stream back on demand. GB10 lacks GDS support, so the path uses pinned-host bounce buffers.

### `spark-model`
The bulk of the inference logic:
- `model.rs` (split into `model/`) — `TransformerModel` orchestrator: per-layer forward dispatch, prefill/decode/verify path selection, SSM state management, KV cache binding, EP/TP composition.
- `layers/` — per-architecture layer impls:
  - `qwen3_attention/` — full-attention with KV cache (paged or MLA); decode + prefill + verify trait impls.
  - `qwen3_ssm/` — gated delta net (Qwen3-Next/3.5/3.6 hybrid models).
  - `nemotron_mamba2/` — Mamba2 SSM variant.
  - `moe/` — sparse mixture-of-experts (routed + shared expert).
  - `dense_ffn/` — non-MoE FFN.
  - `vision_encoder/` — Qwen3-VL ViT.
  - `mtp_head.rs` / `dflash_head/` — speculative drafter heads.
- `weight_loader/` — per-architecture weight name → typed-struct mapping; one variant per family (`qwen3.rs`, `qwen35.rs`, `gemma4.rs`, `nemotron.rs`, `mistral_loader.rs`, `minimax.rs`, `qwen3_vl.rs`, `dflash_loader.rs`).
- `weight_map/` — quant-format detection + dequant utilities (NVFP4 E2M1 + FP8 group scales, FP8 block-scaled, BF16 raw).
- `traits.rs` — `Model` trait that the scheduler talks to. The trait is split across multiple `impl Model for TransformerModel { ... }` blocks under `model/trait_impl/` (see `docs/adr/0006-multi-file-module-idiom.md`).
- `factory.rs` — given a `ModelConfig`, picks the right `WeightLoader` and constructs a `TransformerModel`.

### `spark-server`
HTTP + scheduling:
- `main_modules/serve.rs` — boot sequence (phases 0–11, helpers under `serve_phases/`).
- `main_modules/serve_router.rs` — route table and middleware stack; `main_modules/middleware.rs` holds the middleware (observability headers, bearer auth, rate limit, model-state gate, priority).
- `main_modules/hotswap.rs` — hot-swap context shared by the admin endpoints and the scheduler (weights arena, manifest, swap state machine).
- `main_modules/text_sidecar.rs` — small text-model sidecars for embeddings and rerank.
- `scheduler/` — main scheduler loop (`run()`), per-step phases (decode, verify K=2/3/γ, MTP, NGram, prefill, sample, emit, lifecycle).
- `prefill_pacer.rs` — prefill pacing: time-sized prefill chunks and the decode share while a prompt is prefilled next to running sessions (`docs/design/prefill-pacing.md`).
- `arena_sched.rs` — pressure-based arena scheduler: per-model queues, swap decision, completion estimate + reservation ledger. `swap_costs.rs` keeps the swap-cost bookkeeping.
- `api/` — HTTP handlers:
  - text generation: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/conversations`;
  - other modalities: `/v1/embeddings`, `/v1/rerank`, `/v1/translate`, `/v1/ocr`, `/convert` (async document-conversion jobs), `/v1/audio/transcriptions`, `/v1/images/generations`, `/v1/images/edits`;
  - operations: `/admin/models*` (snapshot / load / unload), `/tokenize`, `/detokenize`, `/v1/stats`, `/internal/state`, `/health`, `/metrics`;
  - cross-cutting: `admission.rs` (`accepted_processing_time`), `lazy_swap.rs` (request-time model swap), `queued_sse.rs` (SSE keep-alives while a request waits).
- `anthropic/` — Anthropic API translation layer (`/v1/messages`).
- `openai/` — OpenAI request/response types.
- `compaction/` — context compaction when a conversation outgrows the window (`docs/design/context-compaction.md`).
- `kv_tier.rs` — NVMe KV tier glue: per-model block store, background writer, scheduler-side persist hooks.
- `session_manager.rs` — session-scoped SSM state.
- `response_store.rs`, `conversation_store.rs` — stores behind stateful Responses resume, opt-in chat-completion storage and the Conversations API.
- `ocr/` — native OCR pipeline behind `/v1/ocr` and `/convert`.
- `tool_parser/` — per-model tool-call format parsers (Hermes JSON, Qwen3-coder XML, Gemma-4 native, MiniMax XML, Mistral native, bare JSON, etc.).
- `reasoning_parser/` — per-model-family thinking/reasoning block detection.
- `grammar/` — XGrammar-backed constrained decoding (tool schemas, JSON mode).
- `tokenizer.rs` — chat-template rendering via Jinja, streaming decode.
- `auth.rs`, `rate_limiter.rs` — bearer-token auth and per-identity token-bucket rate limiting.
- `observe/`, `metrics.rs` — anomaly events (ntfy push + SQLite store), state snapshots, per-request records; Prometheus metrics.

## Request lifecycle (decode path)

1. **HTTP + middleware** — Axum receives `POST /v1/chat/completions` (`main_modules/serve_router.rs`). The request passes the middleware stack in `main_modules/middleware.rs`, outermost first: panic catch, CORS, observability headers, bearer auth (`--require-auth`), per-identity rate limit (429), model-state gate (503 while no model is `Running`; lazy-load endpoints pass through), context policy, proxy-asserted priority, body-size limit.
2. **Admission** — `api/chat/mod.rs::chat_completions` parses the body, then calls `api/admission::admit` before anything is enqueued: the estimate is reserved in the ledger, and a request whose `accepted_processing_time` budget cannot be met is rejected synchronously with 503 `capacity_exceeded`. See `docs/design/deadline-admission.md`.
3. **Model gate** — `api/lazy_swap::prepare_model_gate` makes the requested model the active one (lazy multi-model mode only; otherwise a pass-through). An unregistered model is a 404. If the request has to wait for a swap, a streaming client gets HTTP 200 + SSE keep-alives immediately (`api/queued_sse.rs`), a blocking client waits on its ticket.
4. **Pre-process** — `chat_completions_inner` takes one per-request `ModelRuntime` snapshot so a hot-swap cannot mix two models' state. `chat/prepare::prepare_prompt` then injects the tool system prompt, runs `chat/msg_entry::build_msg_entries` (messages, tools, image preprocessing), resolves thinking, runs loop detection, and renders + tokenizes the chat template via `chat/template::render_template`.
5. **Context fit** — if the prompt does not fit the window, the context policy decides: `reject` returns 400, otherwise a compaction entry is created synchronously and the prompt is rendered again (streaming clients receive `: compacting` keep-alives meanwhile). See `docs/design/context-compaction.md`.
6. **Sampling setup + submit** — `chat/finish::finish_request` resolves sampling preset, stops and grammar via `chat/sampling_setup::build_sampling`, clamps `max_tokens` to the remaining context, and sends an `InferenceRequest` over `state.request_tx` from the streaming (`api/chat_stream/`) or blocking (`api/chat_blocking.rs`) path.
7. **Scheduler tick** (`scheduler/mod.rs::run` loop, single GPU thread; a receiver thread feeds the pending queue):
   - Apply hot-swap control commands; import completed NVMe restores.
   - Drain pending requests; a request whose prefix is available on the NVMe tier is staged for restore first (`kv_tier.rs`).
   - Swap out active sequences to disk when KV blocks run low; swap them back in when blocks free up.
   - Start and continue prefills; promote prefill-completed sequences to active.
   - Sample drafts (MTP / NGram / DFlash if enabled).
   - Verify drafts via the model's verify path (`decode_verify_k2/k3/k4/kgamma`).
   - Emit accepted tokens as `StreamEvent`s on the request's channel.
8. **Per-token forward**:
   - For each layer: `TransformerLayer::decode` — `qwen3_attention`, `qwen3_ssm`, `moe`, etc.
   - GPU kernels launched via `gpu.kernel(module, fn)` lookups against the embedded `TargetPtxSet`.
   - KV cache append, paged decode attention, SSM state update, MoE routing + expert GEMV.
9. **Sample** — host pulls logits, applies sampling, emits token.
10. **Response** — the streaming handler turns `StreamEvent`s into SSE chunks (reasoning / tool-call parsing, stop strings); the blocking path collects them into one JSON response.

## Build pipeline

```
$ ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.6-35b-a3b ATLAS_TARGET_QUANT=nvfp4 \
    cargo build --release -p spark-server
```

What happens:
1. Cargo invokes `atlas-kernels/build.rs`.
2. `build.rs` walks `kernels/gb10/qwen3.6-35b-a3b/nvfp4/*.cu` + `kernels/gb10/common/*.cu` (shared), invokes `nvcc -arch=sm_121 --ptx` on each, and emits `OUT_DIR/target_ptx.rs` with the resulting PTX bytes.
3. The Rust crates compile, linking the generated module.
4. At runtime, `spark-runtime::gpu` loads the PTX into the CUDA driver and exposes kernel handles.

For a multi-target binary (sweep mode): `ATLAS_TARGET_MODEL='*' ATLAS_TARGET_QUANT='*'` compiles every kernel target. The binary picks the right target at startup based on the loaded model's `model_type` and `hidden_size`.

## See also

- `docs/HARDWARE.md` — adding a new SM target / model family.
- `docs/DEPLOYMENT.md` — Docker, multi-rank, NVMe swap.
- `docs/adr/` — Architecture Decision Records explaining major design choices.

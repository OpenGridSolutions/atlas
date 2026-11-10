# Changelog

All notable changes to Atlas are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For per-release deep dives — kernel-level wins, the engineering history
behind specific subsystems — see the
[Atlas Spark Journey](docs/ATLAS_SPARK_JOURNEY.md).

## [Unreleased]

### Changed

- **One model in memory at a time, sidecars included.** Whisper and
  Qwen-Image no longer stay resident beside the arena model: a request for
  one queues with the arena dispatcher like any other model switch, the
  arena model is unloaded first, and the next arena load unloads the sidecar
  again (its device buffers are freed through a per-owner allocation
  ledger). Without a dispatcher (no preseed list) a sidecar only loads into
  an empty server; otherwise the request gets a 409.
- **Context window from `MODEL.toml`**: `[serving] sessions = N` names how
  many sessions stay warm for a model; the server splits the KV pool evenly
  across them (`window = min(native max, pool / N)`, rounded down to
  blocks). `--max-seq-len` is now optional and pins a fixed window for
  every model; targets without `[serving]` and `--high-speed-swap` keep the
  former 32768 default. `sessions < --max-batch-size` fails the load.
- **`[serving] moe_prefill_transpose`** in `MODEL.toml` pins whether the
  Qwen3 / Qwen3.5-family loaders build the transposed MoE expert copies for
  prefill; without the key they keep deciding by the free memory at load
  time. Set to `false` for `qwen3-next-80b-a3b`, whose copies (40.5 GiB)
  never fit on GB10 so far and would otherwise appear as soon as the free
  reading is accurate.
- `.devpod/install.sh` writes the NVIDIA module option
  `NVreg_EnableSystemMemoryPools` from `ATLAS_NVIDIA_SYSTEM_MEMORY_POOLS`
  (required; `0` in the checked-in env). With the driver default the freed
  load staging stays cached in the driver, invisible to `MemAvailable` and
  `cuMemGetInfo`, and the KV pool is sized too small (80B: 11 GiB). Needs a
  host reboot.
- `--sidecar-small-max-gib` defaults to 0: embed/rerank models go through
  the arena queue instead of staying resident beside the arena model.
- `serve.sh` requires `ATLAS_NVME_CACHE_DIR` and always passes
  `--nvme-cache-dir` — with one model in memory, the NVMe tier is what
  carries sessions across a model switch. It no longer passes a
  `--max-seq-len` unless `ATLAS_MAX_SEQ_LEN` is set.

### Removed

- `--request-timeout`: there is no server-side time limit per request any
  more. A scheduler that makes no step for `--stall-exit-s` (default 300)
  while sequences are open ends the process with exit code 1 instead.
- `ATLAS_IMAGE_RESERVE_GB` and the KV-budget reserve for a configured but
  not yet loaded image model: the image model never shares memory with an
  arena model's KV pool any more.
- **Session slots (E6)**: `--slot-tokens`, `--priority-reserved-slots`,
  `[behavior].slot_tokens`, the `slots_total` admission cap and the
  `X-Atlas-Context-Remaining` header are gone (32fb1126 reverted). Capacity
  and concurrency are block-based: `--max-batch-size` bounds the batch, the
  KV pool bounds residency, LRU reclaim of the prefix cache (persisted
  blocks first) and the disk spill handle overcommit. The session probe's
  `slot` fragment is now `context` (`max_tokens` = `--max-seq-len`).

### Fixed

- Fused prefill+decode step (`mixed_forward`): chunks inside a cached
  prefix now take the standard prefill path — the fused step has no cache
  skip / SSM replay and recomputed them (re-advancing SSM state). It also
  extends `kv_valid_tokens` now, so the prefix-cache insert of a prompt
  that finishes on the standard path no longer stops at the first fused
  chunk.
- `/internal/state` and `POST /v1/atlas/session/state` were never routed
  (the handler file was not even compiled).
- **KV-8**: the block-key chain only ever appended; rejected speculative
  drafts and watchdog rollbacks left stale keys that were sealed and
  persisted for different content. `BlockKeyChain::truncate_to` +
  `SequenceState::{pop_tokens, truncate_tokens}` keep it in step.
- Scheduler hangs under pool pressure: the swap-in test and both pool
  guards now reclaim LRU prefix-cache blocks instead of spilling live
  sequences; swapped sequences resume from the idle branch; the drain
  polls while NVMe restores are staged (they complete on a channel).
- KV-7 evict guard is advisory: eviction is strict LRU. Skipping
  unpersisted blocks failed prefill with `KV cache exhausted` when the
  write queue lagged, and — worse — evicted a freshly NVMe-restored chain
  before older sessions whose tails were still queued, so restores never
  hit. An unpersisted block is only cache; its persist job holds a copy.
- NVMe restore uploads run on their own stream with a GPU-side fence
  (import 1.2 s → 35 ms of scheduler time per 2.8k-token session).
- Hybrid (SSM) restores never resumed: the persist path exported only the
  deepest snapshot and rejected it when unaligned (the prompt-end and
  finish-leaf snapshots always are), and a cold single-shot prefill saved
  no block-aligned prompt-tail checkpoint at all. Every aligned snapshot
  is persisted now, and the tail-checkpoint split is unconditional while
  the tier is active. Qwen3-Next-80B, 7.7k-token sessions: restored TTFT
  8.5 s → 0.31 s, warm-stream stall under churn 8.5 s → 0.38 s.
- NVMe restore imports reclaim cache blocks too — a cache-full pool no
  longer shrinks every restore to one block.

### Added

- **Prefill pacing** (`--prefill-max-pause-ms`, `--prefill-decode-share`;
  off by default): while a prompt is prefilled next to running sessions,
  the prefill chunk is sized by time instead of tokens — the scheduler
  measures fixed and per-token cost of a step per model and keeps one step
  near the configured pause — and the running sessions keep the configured
  share of GPU time through decode-only steps between chunks (100 = the
  prefill waits until nobody generates). Chunk 0 is part of the
  interleaving; middle chunks ride fused with the decode step. Without a
  running session a prefill keeps full chunks. Not applied to EP, MLA
  models and prompts with images. `serve.sh` reads
  `ATLAS_PREFILL_MAX_PAUSE_MS` / `ATLAS_PREFILL_DECODE_SHARE`.
- **NVMe KV tier** (`--nvme-cache-dir`): content-addressed block store for
  KV blocks, SSM snapshots and tail records, keyed by chained BLAKE3 block
  keys (`k_i = H(k_{i-1} ‖ tokens_i ‖ media_digests_i)`) shared with the RAM
  radix cache. `persist_mode = on_release` writes at request end and before
  a model switch (atomic tmp+fdatasync+rename, xxh3 checksums, header
  binding to model + KV layout); restore stages contiguous prefixes
  asynchronously into the RAM cache before admission (`--nvme-restore-min-
  tokens`), with SSM resume points for hybrid models. GC by TTL, orphan
  cascade and free-space pressure; daily write-budget monitoring; the KV-7
  evict guard keeps unpersisted blocks in RAM.
- **Media digests in the prefix-cache keys**: vision/OCR prompts are now
  cacheable — the BLAKE3 chain folds in per-image digests, so identical
  token streams with different images never share blocks (closes the
  issue-#58 all-or-nothing vision bypass).
- **Pressure-based arena scheduler**: per-model two-tier queues (priority /
  normal) + a pure, replay-tested swap decision (`D_B ≥ S_B = κ·T_swap·(1 +
  N_active + N_queued + λ·T_swap)`; candidate by highest D/S, starvation
  over hysteresis, idle fast-path) replace the synchronous first-request-
  pays lazy swap. Swap costs (T_load/T_warmup/T_unload/T_persist) are
  measured per real swap, EWMA-refined and persisted next to the snapshots;
  the measured runtime reserve feeds the KV budget (MEM-7). Queued
  streaming requests answer immediately with SSE keep-alives; non-streaming
  requests get estimated-wait 503s below the proxy timeout.
- **Priority tier** behind `--trust-proxy-headers` (`X-Priority: high`,
  loopback-bind enforced): priority requests are admitted first but never
  trigger a model switch.
- **Resident embed/rerank sidecars** (`--sidecar-small-max-gib`): small
  embed/rerank models load once beside the arena model with their own
  worker, tokenizer and fixed KV pool — no arena waits; their footprint
  joins the KV-budget sidecar reserve.
- `GET /internal/state` (Bearer-gated): queues per model/tier with D/S/λ,
  last swap decision + inputs, capacity profile (bytes/token, S_ssm,
  parallel sequences at 8k/32k/max), KV pool, swap costs, NVMe store +
  write queue. `/health` gains `draining`/`swapping`. New metrics:
  tokens/s, queue wait + length per tier, NVMe read/write/restored bytes.

- Holo-3.1-35B-A3B / Holo-3.1-0.8B / Ornith-1.0-9B model support on GB10
  (sm_121): hybrid Gated-DeltaNet + full-attention + (256-expert MoE | dense
  FFN) + Qwen3-VL vision tower. Brings CUTLASS Sm120 NVFP4 grouped MoE, FLA
  chunked-scan GDN prefill + wmma DV-block decode, cuBLASLt/CUTLASS attention
  projections, kernel-batched co-dispatch prefill, radix-KV + Marconi
  SSM-snapshot prefix caching, and self-relative auto KV budget. (#203)
- GEMM-based Qwen3-VL ViT attention kernel (tensor-core SDPA replacing the
  warp-per-query kernel) + tensor-core ViT block GEMMs + batched multi-image
  forward — ~2× image-request TTFT on GB10. (#202)

### Fixed

- Weight-only NVFP4 (W4A16) checkpoints now load. llm-compressor
  `nvfp4-pack-quantized` with `input_activations: None` ships no static
  activation scale; the loader previously required `input_global_scale` and
  failed (e.g. `AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4`). The
  field is loaded-but-unused (activations are quantized dynamically), so it is
  now optional. W4A4/W4A8 checkpoints are unaffected. (#203)
- `--gpu-memory-utilization` now enforces a hard ceiling on total GPU
  memory (weights + buffers + KV cache + reserves), matching the vLLM /
  sparkrun convention.  Previously the fraction was applied only to
  post-weight free memory, causing the KV cache to over-allocate by
  20-27 GB when values below the ~0.88 default were used.  This blocked
  multi-service co-residency on shared-memory systems (e.g. DGX Spark
  GB10).  The flag now behaves as documented: `0.50` on a 120 GB device
  caps Atlas at ~60 GB total.  (#170)

## [0.1.0] — 2026-05-06

Initial public release. Atlas is a pure-Rust LLM inference engine
targeting NVIDIA GB10 (DGX Spark, SM121) with twelve hand-tuned
(Hardware × Model × Quantization) targets.

### Added

- Pure-Rust runtime — no Python, no PyTorch — for hybrid Attention +
  SSM/GDN/Mamba-2 architectures with NVFP4 / FP8 / BF16 quantization.
- 35 hyperoptimized CUDA kernels per target, compiled to PTX and
  embedded in the binary at build time. Multi-model image dispatches
  the right kernel set at startup from `config.json`.
- OpenAI- and Anthropic-compatible HTTP API (`/v1/chat/completions`,
  `/v1/responses`, `/v1/messages`, `/v1/models`, `/v1/conversations`,
  `/tokenize`, `/detokenize`, `/health`, `/metrics`).
- Tool calling with grammar-constrained decoding (Hermes,
  Qwen3-Coder, Mistral, MiniMax-XML formats).
- MTP speculative decoding (K=2 pipelined verify), self-speculative
  layer-skipping, and N-gram speculative decoding.
- Prefix caching: radix-tree (RadixAttention) + SSM snapshot cache
  (Marconi-style). 10× warm-cache TTFT reduction.
- KV cache dtypes: BF16, FP8, NVFP4, turbo3, turbo4. Optional
  per-layer high-precision overlay (`--kv-high-precision-layers`).
- Multi-GPU expert parallelism (EP=2 over RoCEv2) for models that
  exceed a single GB10's weight budget (122B-class, MiniMax M2.7).
- Vision encoder (Qwen3-VL, Qwen3.6 ViT).
- High-speed NVMe KV swap (sliding-window, io_uring) for
  long-context decoding past the HBM cap.
- Bearer-token authentication (`--require-auth` +
  `--auth-tokens-file`), constant-time validated. Default bind is
  `127.0.0.1`; `--bind 0.0.0.0` warns when used.
- Twelve supported (GB10, model, quant) targets across Qwen3.5 /
  Qwen3.6 / Qwen3-Next / Qwen3-VL / Gemma-4 / Mistral-Small-4 /
  MiniMax-M2.7 / Nemotron-H families.
- mdBook documentation at `book/src/`, rustdoc at `target/doc/`,
  Docker image `avarok/atlas-gb10:latest`.

### Engineering notes

For the kernel-level perf history — long-context regression sweeps,
the parking_lot migration, the libcuda + libnccl CI stubs, the
multi-stage scheduler refactor — see
[`docs/ATLAS_SPARK_JOURNEY.md`](docs/ATLAS_SPARK_JOURNEY.md) and the
[`book/`](book/) chapters under `deep-dives/`.

[Unreleased]: https://github.com/Avarok-Cybersecurity/atlas/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Avarok-Cybersecurity/atlas/releases/tag/v0.1.0

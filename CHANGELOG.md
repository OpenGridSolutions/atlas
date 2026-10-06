# Changelog

All notable changes to Atlas are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For per-release deep dives — kernel-level wins, the engineering history
behind specific subsystems — see the
[Atlas Spark Journey](docs/ATLAS_SPARK_JOURNEY.md).

## [Unreleased]

### Added

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

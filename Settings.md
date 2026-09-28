# Settings-Übersicht — was wird wo konfiguriert?

Stand: 2026-09-28 (nach Todo.md 0.7). Grundprinzip seit 0.7:

> **MODEL.toml ist die Quelle der Wahrheit für modellspezifisches Verhalten.**
> CLI-Flags sind entweder (a) zum Starten des Servers nötig, (b) maschinen-/
> deployment-bezogen, oder (c) explizite per-Run-Overrides für Experimente.
> Env-Variablen sind Betriebs-/Debug-Schalter.

Ziel dieser Übersicht: pro Modell die optimalen Werte über die Test-Bench
(Todo.md Phase 2) finden und im jeweiligen `kernels/gb10/<modell>/MODEL.toml`
festschreiben — nicht in serve.sh/compose.

---

## 1. Per-Modell: `MODEL.toml` des Kernel-Targets

### `[behavior]` — Laufzeitverhalten (zur Compile-Zeit in atlas-kernels gebacken)

| Key | Default | Wirkung / Tuning-Hinweis |
|---|---|---|
| `default_kv_dtype` | "" (= fp8) | KV-Cache-Präzision (`bf16`/`fp8`/`nvfp4`/Turbo-Varianten). **Der wichtigste Qualitätsknopf**: nvfp4 spart Speicher, kollabiert aber auf manchen Modellen (Gemma `<unused>`, Mistral-Loops). Explizites `--kv-cache-dtype` überschreibt. |
| `default_mtp_quantization` | "" (= bf16) | **Neu (0.7).** MTP-Head-Präzision (`bf16`/`fp8`/`nvfp4`). bf16 = höchste Acceptance-Rate = meist bester End-to-End-Durchsatz. Explizites `--mtp-quantization` überschreibt. |
| `default_num_drafts` | 0 (= CLI) | Spekulations-Tiefe K−1. Optimal variiert pro Modell (K=2 schlägt teils K=3). |
| `fp8_kv_calibration_tokens` | 0 | Online-Kalibrierung der FP8-KV-Scales über die ersten N Token — nötig für FP8-Checkpoints ohne k/v_scale-Tensoren (siehe qwen3.6-35b-a3b, #211). |
| `max_thinking_budget` | 256 | Obergrenze Thinking-Tokens pro Request. |
| `thinking_default` | false | Thinking an, wenn der Request nichts angibt. |
| `thinking_in_tools` | true | Thinking auch innerhalb von Tool-Loops erlauben. |
| `tool_call_parser` | "" | Parser-Override (`hermes`, `qwen3_coder`, …); leer = Mapping aus tool_defaults.toml per model_type. |
| `disable_tool_steering` | false | `<tool_call>`-Steering-Präfix unterdrücken (Nemotron-Super-Fall). |
| `disable_tool_grammar` | false | XGrammar-Erzwingung bei `tool_choice=auto` aus (vLLM-Verhalten). |
| `tool_retry` | true | Fehlgeschlagene Tool-Call-Parses re-steern. |
| `enable_loop_watchdog` | false | Content-Loop-Watchdog für dieses Modell. |
| `think_loop_min_repeats` / `think_loop_scan_window` | 3 / 160 | Thinking-Loop-Erkennung. |
| `confidence_early_stop` / `confidence_run_length` | true / 30 | F2-Confidence-Early-Stop. |
| `fuzzy_repeat_tolerance_div` | 12 | Toleranz der Fuzzy-Repeat-Erkennung. |
| `max_inter_tool_prose` | 384 | Prosa-Budget zwischen Tool-Calls. |
| `max_post_think_content_tokens` | 100000 | Content-Cap nach `</think>`. |
| `min_p_floor` / `temperature_max` | 0.0 / 0.0 (aus) | Server-seitige Sampling-Leitplanken gegen Quantisierungsrauschen. |
| `rollback_resteer` | true | Watchdog-Rollback+Re-Steer (Cap 2/Sequenz). |
| `rom_head` | "" | ROM-Head-Variante. |
| `tscg` | false | Tool-Sequence-Constrained-Generation. |

Außerdem pro Modell in MODEL.toml: `[sampling.*]`-Presets (thinking_text,
thinking_coding, non_thinking, tools), `[dflash]` (draft_model etc.),
`[[model_types]]` sowie die Architektur-/Geometrie-Keys (`layers_*`, `moe_*`,
`mtp_layers`, …) und die Quant-Variante als Unterverzeichnis (`nvfp4/`, `bf16/`).

**Änderungen an MODEL.toml erfordern einen Rebuild** (atlas-kernels backt
`[behavior]` zur Compile-Zeit ein — `parse_behavior()` liest NUR diese Tabelle).

### Heutige `default_kv_dtype`-Belegung (gb10)

fp8: deepseek-v4-flash · bf16: gemma-4-26b-a4b, gemma-4-31b, mistral-small-4,
step3p7-flash, qwen3.6-27b · **unbelegt (→ fp8-Fallback): alle übrigen ~15
Targets** — Kandidaten für den Test-Bench-Sweep (Todo 0.7d).

---

## 2. CLI-Flags (`spark serve`)

### Zum Starten nötig / Deployment-bezogen (global ok)

| Flag | Default | Zweck |
|---|---|---|
| `MODEL` (positional) / `--model-from-path` | — | Boot-Modell |
| `--bind` / `--port` | 127.0.0.1 / 8888 | Listener (0.0.0.0 nur mit `--require-auth` sinnvoll) |
| `--require-auth` / `--auth-tokens-file` / `--auth-token` | aus | Bearer-Auth |
| `--gpu-ordinal` / `--cache-dir` / `--snapshot-dir` / `--snapshot-read-qd` | 0 / HF / snapshots | Maschine/Pfade |
| `--weights-arena-gb` / `--arena-host-register` | an / aus | Hot-Swap-Arena (0 = Hot-Swap aus) |
| `--preseed-model` (+ `ATLAS_PRESEED_*`-Envs) | — | Modell-Registry & Kategorien |
| `--rank`/`--world-size`/`--tp-size`/`--ep-size`/`--master-*` | 1 Node | Multi-GPU/EP |
| `--request-timeout` | 300 s | HTTP-Timeout |
| `--oom-guard-mb` | 4096 | Lade-Abbruchschwelle |
| `--dump` | aus | Request/Response-Mitschnitt (Debug) |

### Kapazität & Speicher (deployment-abhängig, gemeinsam für alle Modelle)

| Flag | Default | Tuning |
|---|---|---|
| `--gpu-memory-utilization` | 0.9 | Gesamt-Cap (vLLM-Semantik) |
| `--max-seq-len` | 32768 | Worst-Case-Kontext; treibt KV-Boot-Check (mit `ATLAS_KV_OVERCOMMIT=1` nur Warnung) |
| `--max-batch-size` | 8 | Parallele Decode-Sequenzen (Prod: 4, 8 nach Todo 0.6) |
| `--block-size` | 16 | KV-Blockgröße |
| `--kv-pool-gb` | aus | KV-Pool pinnen (deterministische Hot-Swaps) |
| `--swap-space-gb` | 3 | Disk-Spill für KV-Überlauf (Prod: 32–64) |
| `--high-speed-swap*` (7 Flags) | aus | Block-Level-KV-Streaming auf NVMe |
| `--max-prefill-tokens` | 8192 | Chunked-Prefill-Chunkgröße (0 = aus) |
| `--enable-prefix-caching` | aus | Radix-Prefix-Cache |
| `--ssm-cache-slots` / `--ssm-checkpoint-interval` | 0 / 0 | Marconi-SSM-Snapshots (16 / 256 für Multi-Turn) |
| `--scheduling-policy` / `--tbt-deadline-ms` | fifo | SLAI = SLO-aware |

### Per-Modell-relevant — CLI ist hier nur noch der EXPLIZITE Override

| Flag | Default ohne Flag | MODEL.toml-Pendant |
|---|---|---|
| `--kv-cache-dtype` | MODEL.toml → fp8 | `default_kv_dtype` ✓ |
| `--mtp-quantization` | MODEL.toml → bf16 | `default_mtp_quantization` ✓ (neu) |
| `--num-drafts` | 1 | `default_num_drafts` ✓ |
| `--max-thinking-budget` / `--disable-thinking` | MODEL.toml | `max_thinking_budget` / `thinking_default` ✓ |
| `--tool-call-parser` | model_type-Mapping | `tool_call_parser` ✓ |
| `--disable-tool-grammar` | MODEL.toml | `disable_tool_grammar` ✓ |
| `--default-chat-template-kwargs` | MODEL.toml | (thinking-Keys) ✓ |
| `--kv-high-precision-layers` | 0 / auto je Dtype | ✗ **noch CLI-only** — Kandidat für `[behavior]` |
| `--lm-head-dtype` | default | ✗ CLI-only — Kandidat |
| `--mtp-vocab` | 100352 | ✗ CLI-only — Kandidat |
| `--speculative` / `--self-speculative` / `--ngram-speculative` / `--dflash*` | aus | teils (`[dflash]`) — Aktivierung ist CLI |
| `--default-top-n-sigma` / `--default-min-p` | 1.0 / 0.08 | `[sampling.*]`-Presets + `min_p_floor` (teilweise) |
| `--vision-max-pixels` | Modell-Cap | ✗ CLI/Env |
| `--adaptive-sampling` / `--profile` / `--warmup-prompt` / `--auto-compact` | aus | Experimente |

## 3. Env-Variablen (Betrieb/Debug)

| Variable | Wirkung |
|---|---|
| `ATLAS_PRESEED_{GENERATIVE,EMBED,RERANK,TRANSCRIBE,TRANSLATE}` | Kategorisierte Modell-Registry (JSON-Arrays) |
| `ATLAS_KV_OVERCOMMIT=1` | Boot-Worst-Case-Check → Warnung (paged on-demand) |
| `ATLAS_KV_EXTERNAL_RESERVE_GB` | Manueller Co-Tenant-Abzug im KV-Budget |
| `ATLAS_SKIP_OOM_PREFLIGHT=1` | Loader-Preflight aus (UMA-False-Negatives) |
| `ATLAS_MEM_DEBUG=1` (+ `RUST_LOG=off,memdebug=debug`) | Per-Request-KV-Tracing (Doku.md §7) |
| `ATLAS_DISABLE_WATCHDOGS=1` | ALLE Auto-Watchdogs aus — **nie in Produktion** |
| `ATLAS_CUBLAS_GEMM=1` | cuBLASLt-Route für BF16-Dense-GEMMs (Plan.md H.2) |
| `ATLAS_FAST_LOAD=0` / `ATLAS_FAST_LOAD_PREFETCH_SHARDS=1` | Loader-Varianten |
| `ATLAS_VISION_MAX_PIXELS` | wie `--vision-max-pixels` |
| `RUST_LOG` | Tracing-Filter (Default info) |

## 4. Test-Bench-Sweep pro Modell (Empfehlung)

Für jedes Target in dieser Reihenfolge sweepen und das Ergebnis ins MODEL.toml
schreiben — danach startet der Server ohne modellbezogene CLI-Flags:

1. `default_kv_dtype` (bf16 → fp8 → nvfp4) × `--kv-high-precision-layers`
   (0/2/3/auto), Gate: Coherence + Long-Context-Needle + Agentic-Suite.
2. `default_mtp_quantization` (bf16/fp8/nvfp4) × `default_num_drafts` (1/2),
   Gate: tok/s End-to-End (Acceptance-Rate schlägt Draft-Speed).
3. `fp8_kv_calibration_tokens` (nur FP8-Checkpoints ohne Scales): 0/256.
4. Watchdog-/Thinking-Keys nur bei auffälligem Verhalten (Loops, Prosa-Flut).

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

### `[serving]` — warme Sessions und Kontextfenster

| Key | Default | Wirkung |
|---|---|---|
| `sessions` | – (fehlt = `--max-seq-len`, sonst 32768) | Anzahl der Sessions, die für dieses Modell parallel warm im Speicher liegen. Der Server teilt den KV-Pool gleichmäßig auf: Fenster je Session = `min(natives Maximum des Modells, KV-Pool / sessions)`, auf 16-Token-Blöcke abgerundet. Muss ≥ `--max-batch-size` sein (sonst Abbruch beim Laden). |
| `moe_prefill_transpose` | – (fehlt = der Loader entscheidet nach dem beim Laden freien Speicher) | `false`: keine transponierten Kopien der MoE-Expertengewichte für den Prefill; der Prefill läuft über das Grouped-GEMM auf dem Originallayout, der Speicher bleibt dem KV-Pool. `true`: Kopien immer anlegen. Gilt für die Qwen3-/Qwen3.5-Loader. Beim 80B (`qwen3-next-80b-a3b`) `false`: die Kopien kosten 40,5 GiB zusätzlich zu 47,3 GiB Gewichten. |

Regel für den Wert: das native Maximum des Modells (`max_position_embeddings`)
so oft in den KV-Pool legen, wie es hineinpasst, und aufrunden. Es ist immer
nur ein Modell im Speicher; kleine Modelle haben deshalb einen größeren KV-Pool
und mehr Sessions. Whisper und Qwen-Image haben keinen KV-Cache und keine
`[serving]`-Tabelle. Das tatsächliche Fenster steht beim Laden im Log (`Context window: …`) und
in `/internal/state` (`capacity_profile.max_context_tokens`).

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
| `--stall-exit-s` | 300 s | Einziger serverseitiger Abbruch: offene Sequenzen, aber so lange kein Scheduler-Schritt ⇒ Prozess endet mit Exit 1 (Neustart durch die Unit). Kein Zeitlimit pro Anfrage; Clients können `timeout` im Request-Body setzen |
| `--ocr-result-ttl-s` / `--ocr-poll-idle-s` | 900 / 120 | `/convert`-Jobs: Ergebnis-Haltezeit, Verwaist-Schwelle ohne Poll |
| `--ocr-store-max-mb` / `--ocr-max-jobs` | 512 / 64 | `/convert`-Job-Store-Grenzen (voll ⇒ 503 `capacity_exceeded`) |
| `--ocr-page-parallelism` / `--ocr-max-pages` | 4 / 500 | Seiten-Fan-out je Job; Seitenlimit (`document_too_large`) |
| `--ocr-max-failed-pages-pct` | 0 | Toleranz fehlgeschlagener Seiten, 0 = jede ⇒ `failed` |
| `--ocr-cache-dir` / `--ocr-cache-ttl-days` / `--ocr-cache-max-gb` | aus / 90 / 10 | inhaltsadressierter `/convert`-Ergebnis-Cache (ID-Stabilität) |
| `--estimate-safety-factor` / `--estimate-reservation-grace` | 1.25 / 2.0 | `accepted_processing_time`: Aufschlag auf die Schätzung; Backstop-Ablauf einer Reservierung (× Schätzung) |
| `--ntfy-topic` / `--ntfy-url` | aus / ntfy.sh | Anomalie-Pushes (Prio ≥ 3): Panics, CUDA-Fehler, Stalls, 5xx-Rate, Schätzung verfehlt, … |
| `--alert-min-interval-s` / `--alert-max-per-hour` | 300 / 6 | Dedupe je Art+Modell |
| `--event-db` / `--event-db-max-mb` / `--event-snapshot-interval-s` | aus / 512 / 30 | SQLite-Ereignisspeicher: `events`, `snapshots` (/internal/state), `requests` |
| `--alert-stall-s` / `--alert-5xx-per-min` | 60 / 10 | Detektor-Schwellen (Push; Abbruch erst bei `--stall-exit-s`) |
| `ATLAS_NTFY_TOPIC` / `ATLAS_EVENT_DB` (serve.sh) | — | setzen die beiden Flags; serve.sh meldet zusätzlich einen Exit ≠ 0 per ntfy |
| `--oom-guard-mb` | 4096 | Lade-Abbruchschwelle |
| `--dump` | aus | Request/Response-Mitschnitt (Debug) |

### Kapazität & Speicher (deployment-abhängig, gemeinsam für alle Modelle)

| Flag | Default | Tuning |
|---|---|---|
| `--gpu-memory-utilization` | 0.9 | Gesamt-Cap (vLLM-Semantik): Gewichte + KV-Pool + Reserve des geladenen Modells bleiben darunter. Rechnung und Messwerte in `docs/design/memory-model.md` |
| `--max-seq-len` | aus | Festes Kontextfenster für alle Modelle (Testläufe). Ohne Flag gilt `[serving].sessions` aus MODEL.toml; Targets ohne `[serving]` und `--high-speed-swap` bekommen 32768. Ein festes Fenster treibt den KV-Boot-Check (mit `ATLAS_KV_OVERCOMMIT=1` nur Warnung) |
| `--max-batch-size` | 8 | Parallele Decode-Sequenzen (Prod: 4, 8 nach Todo 0.6) |
| `--block-size` | 16 | KV-Blockgröße |
| `--kv-pool-gb` | aus | KV-Pool pinnen (deterministische Hot-Swaps) |
| `--swap-space-gb` | 3 | Disk-Spill für KV-Überlauf (Prod: 32–64) |
| `--high-speed-swap*` (7 Flags) | aus | Block-Level-KV-Streaming auf NVMe |
| `--max-prefill-tokens` | 8192 | Chunked-Prefill-Chunkgröße (0 = aus) |
| `--enable-prefix-caching` | aus | Radix-Prefix-Cache |
| `--ssm-cache-slots` / `--ssm-checkpoint-interval` | 0 / 0 | Marconi-SSM-Snapshots (16 / 256 für Multi-Turn) |
| `--scheduling-policy` / `--tbt-deadline-ms` | fifo | SLAI = SLO-aware |
| `--prefill-max-pause-ms` | 0 (aus) | Prefill-Taktung: längste Pause, die ein Prefill-Schritt laufenden Sessions zumutet. Die Stückgröße ergibt sich daraus (pro Modell zur Laufzeit gemessen, höchstens `--max-prefill-tokens`). Wirkt nur, solange gleichzeitig generiert wird; nicht bei EP, MLA-Modellen und Bild-Prompts. Prod (`ATLAS_PREFILL_MAX_PAUSE_MS`): 500 |
| `--prefill-decode-share` | 50 | Prefill-Taktung: GPU-Zeitanteil in Prozent, den laufende Sessions behalten, während ein Prompt eingelesen wird (0 = Prefill-Schritt in jeder Iteration, 100 = Prefill wartet, bis niemand mehr generiert). Nur mit `--prefill-max-pause-ms` > 0. Prod (`ATLAS_PREFILL_DECODE_SHARE`): 50 |

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
| `--adaptive-sampling` / `--profile` / `--warmup-prompt` | aus | Experimente |

## 3. Env-Variablen (Betrieb/Debug)

| Variable | Wirkung |
|---|---|
| `ATLAS_PRESEED_{GENERATIVE,EMBED,RERANK,TRANSCRIBE,TRANSLATE,OCR,IMAGE}` | Kategorisierte Modell-Registry (JSON-Arrays) |
| `ATLAS_IMAGE_NO_CUBLAS=1` | Qwen-Image-Sidecar: In-Tree-GEMM statt cuBLASLt für die Projektionen (Debug) |
| `ATLAS_KV_OVERCOMMIT=1` | Boot-Worst-Case-Check → Warnung (paged on-demand). Nur mit festem `--max-seq-len` nötig; mit `[serving].sessions` passt jede Session per Konstruktion |
| `ATLAS_NVME_CACHE_DIR` | `serve.sh`: Verzeichnis des NVMe-Session-Stores, Pflicht (wird zu `--nvme-cache-dir`) |
| `ATLAS_MAX_SEQ_LEN` | `serve.sh`: festes Kontextfenster für Testläufe (wird zu `--max-seq-len`); unset = `[serving].sessions` |
| `ATLAS_KV_EXTERNAL_RESERVE_GB` | Manueller Co-Tenant-Abzug im KV-Budget |
| `ATLAS_SKIP_OOM_PREFLIGHT=1` | Loader-Preflight aus (UMA-False-Negatives) |
| `ATLAS_MEM_DEBUG=1` (+ `RUST_LOG=off,memdebug=debug`) | Per-Request-KV-Tracing (Doku.md §7) |
| `ATLAS_DISABLE_WATCHDOGS=1` | ALLE Auto-Watchdogs aus — **nie in Produktion** |
| `ATLAS_MEM_HEARTBEAT_SECS` / `ATLAS_MEM_WARN_AVAIL_GB` / `ATLAS_MEM_WARN_SWAP_GB` | Speicher-Heartbeat (Log-Target `mem`): Intervall (60, 0 = aus) und Warnschwellen für `MemAvailable` (8) und freien Swap (2). Die Zeile nennt auch `late_alloc_mib` / `late_peak_mib` — Geräte-Allokationen nach dem Modell-Build, die das KV-Budget nicht kennt |
| `ATLAS_CUBLAS_GEMM=1` | cuBLASLt-Route für BF16-Dense-GEMMs (Plan.md H.2). Gilt nur für Gewichte, die als BF16 vorliegen; FP8-Gewichte laufen auf den FP8-Kerneln |
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


## NVMe-Tier, Druck-Scheduler und Priorität (Okt 2026)

Neue Serve-Flags (Anforderungsdokument Okt 2026). Der NVMe-Tier ist aus,
solange `--nvme-cache-dir` fehlt — `serve.sh` setzt das Flag immer
(`ATLAS_NVME_CACHE_DIR`), weil nur ein Modell zur Zeit im Speicher ist und
Sessions einen Modellwechsel nur über den Store überstehen. Der Druck-Scheduler ersetzt den
synchronen Lazy-Swap automatisch, sobald eine Preseed-Liste konfiguriert
ist.

| Flag | Default | Bedeutung |
|---|---|---|
| `--nvme-cache-dir` | – (aus) | Wurzel des content-addressierten KV-Block-Stores |
| `--persist-mode` | `on_release` | `write_through` ist Stufe 2 und wird abgelehnt |
| `--session-ttl-hours` | 24 | GC-Alter der Store-Dateien |
| `--gc-interval-s` | 600 | GC-Intervall |
| `--nvme-min-free-pct` | 10 | Mindestreserve auf der NVMe |
| `--write-queue-max-mb` | 2048 | Byte-Limit der Schreib-Queue (voll ⇒ Block bleibt im RAM) |
| `--nvme-write-budget-tb-day` | – | Tagesbudget (TBW); Überschreitung ⇒ Warnung |
| `--persist-timeout-s` | 30 | Flush-Obergrenze vor dem Unload |
| `--nvme-restore-min-tokens` | 512 | kürzere Prefixe werden neu gerechnet |
| `--ewma-alpha` | 0.2 | Glättung für Swap-Kosten, λ, Schrittzeit |
| `--kappa` | 1.0 | Wechselschwellen-Faktor (κ=1 ⇒ Break-even) |
| `--min-residency-factor` | 2.0 | Hysterese in Vielfachen von T_swap |
| `--max-wait-hard-s` | 60 | Starvation-Grenze normaler Requests |
| `--idle-epsilon` | 0.1 | Idle-Schwelle (λ·T_swap) |
| `--scheduler-interval-ms` | 200 | Auswertetakt der Wechselentscheidung |
| `--max-queue-length` | 64 | pro Modell; darüber HTTP 429 |
| `--priority-max-wait-s` | 300 | Wartegrenze priorisierter Requests auf ein fremdes Modell |
| `--nonstream-max-wait-s` | 45 | harte Wartegrenze ohne Streaming (unter Proxy-Timeout!) |
| `--drain-grace-ms` | 2000 | Drain-Frist vor `finish_reason="model_swap"` |
| `--sse-keepalive-interval-s` | 5 | `: waiting`-Kommentare wartender Streams |
| `--default-context-policy` | summarize | E7: Verhalten ohne `X-Atlas-Context-Policy` (`summarize` / `truncate` / `reject`; `reject` = Verhalten vor E7, HTTP 400 `context_length_exceeded`) |
| `--compaction-output-reserve` | 2048 | Fit-Regel: Request passt, wenn `prompt + min(max_tokens, reserve) ≤ max_seq_len`; darüber greift die Policy |
| `--compaction-tail-pct` | 25 | Mindestanteil jüngster Runden am Fenster, die wörtlich bleiben (CMP-5) |
| `--compaction-summary-max-tokens` | 768 | Ausgabebudget der Zusammenfassung (CMP-7) |
| `--compaction-trigger-pct` | 70 | Füllstand für die proaktive Vorbereitung (CMP-8; Hintergrund-Job folgt in Teil 2) |
| `--compaction-prompt-file` | eingebaut | Prompt der Zusammenfassung |
| `--compaction-no-keep-first-user` | aus | erste User-Nachricht nicht im Kopf behalten (CMP-4) |
| `--trust-proxy-headers` | aus | `X-Priority: high` auswerten (nur Loopback-Bind) |
| `--sidecar-small-max-gib` | 0 (aus) | 0 = nur ein Modell zur Zeit im Speicher, Embed/Rerank laufen über die Arena-Queue. Jeder andere Wert gibt das auf: Embed/Rerank-Modelle bis zu dieser Größe bleiben als Sidecars neben dem Arena-Modell resident |
| `--growth-reserve-pct` | 10 | Anteil des KV-Pools, den die Admission für wachsende Sequenzen freilässt |

Hinweis: `--write-batch-interval-ms` aus dem Anforderungsdokument gehört
zur Write-through-Stufe (NVM-13) und kommt erst mit
`--persist-mode write_through`; der Server lehnt diesen Modus derzeit
beim Start ab.

### Traefik (AUTH-5 / PRI-7)

Der Server wertet `X-Priority` nur mit `--trust-proxy-headers` und
Loopback-Bind aus. Traefik muss den Header **setzen** für Anfragen aus
dem internen Netz und **entfernen** für alle anderen, und den
`Authorization`-Header durchreichen (`removeHeadersOnSuccess: false` im
ForwardAuth- bzw. keine Header-Strips im Router):

```yaml
http:
  middlewares:
    strip-priority:
      headers:
        customRequestHeaders:
          X-Priority: ""        # extern: Header entfernen
    internal-priority:
      headers:
        customRequestHeaders:
          X-Priority: "high"    # nur am internen Router (ClientIP 10.0.0.0/8)
```

Bearer-Auth bleibt unverändert aktiv (`--require-auth`); Traefik und
Server verwenden denselben Token.


### Session-Tracking: `POST /v1/atlas/session/state`

Content-addressiertes Tracking des KV-Cache-Zustands einer Session
(keine Session-IDs, E2): Der Client schickt seine Konversation, der
Server meldet den Zustand über alle Ebenen. Nur für das aktive
Arena-Modell (Token-IDs und Ketten-Seed sind modellspezifisch).

```bash
curl -s $BASE/v1/atlas/session/state -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{
    "model": "…", "max_tokens": 512,
    "messages": [{"role":"user","content":"…"}, {"role":"assistant","content":"…"}]
  }'
```

Antwort (gekürzt):

```json
{
  "active": true, "tokens": 3912, "cacheable": true,
  "context": { "max_tokens": 32768, "source": "max_seq_len",
               "remaining_tokens": 28856, "fits_with_max_tokens": true },
  "ram_cache": { "cached_tokens": 3072, "ssm_resume_tokens": 3072 },
  "nvme": { "persisted_tokens": 3840, "tail_persisted": true,
            "ssm_resume_tokens": 2048, "restore_eligible": true },
  "estimated_prefill_tokens": 72
}
```

`context.max_tokens` ist das Serving-Fenster (`--max-seq-len`); die
Kapazität über Sessions hinweg ist blockbasiert (Pool, LRU, Spill) und
hat keine Slot-Grenze. Die Probe ist strikt read-only (keine
Refcount-/LRU-Mutation) und rendert Nachrichten ohne Generation-Prompt —
der Treffer entspricht dem geteilten Prefix eines echten Requests bis
zur letzten Blockgrenze. Multimodale Inhalte werden nicht bewertet
(`media_evaluated: false`).

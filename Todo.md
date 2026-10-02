# Todo — Speicher-Fixes, Logging-System & Test-Bench auf dem DGX Spark

Stand: 2026-09-28. Kontext: `Memory_errors.log` (Hot-Swap-Load-Fehler nach Unload des
80B-Modells, „KV budget auto-measure implausible", monoton wachsendes `used_so_far`
über Retry-Versuche) + NVIDIA Known Issue zu `cudaMemGetInfo` auf UMA-Systemen.

Einordnung: Der NVIDIA-Workaround (`MemAvailable + SwapFree`, HugePages-Sonderfall) ist
bereits implementiert (`crates/spark-runtime/src/meminfo.rs`, Plan.md Phase S/M) und
funktioniert auch im Container — `/proc/meminfo` ist nicht namespaced, `cgroup
memory.max = max` (verifiziert 2026-09-28). Ein Containerless-Umbau wurde geprüft und
**verworfen** (Kosten > Nutzen; das UMA-Reporting-Problem ist ein Treiber-Thema und
verhält sich nativ identisch). Einziger realer Container-Nachteil, den wir bewusst in
Kauf nehmen: io_uring ist per seccomp blockiert, Snapshot-Reload fällt auf sequentielle
O_DIRECT-preads zurück. Host-Hebel unabhängig davon: SwapTotal ist nur 8 GiB — mehr
NVMe-/zram-Swap vergrößert den Reclaim-Puffer, den die Budget-Formel mitzählt.

---

## Phase 0 — Speicherverwaltungs-Bugs (Ursache der Log-Fehler)

- [x] **0.1 Sanity-Gate im KV-Budget: Basen-Clamp.** *(umgesetzt 2026-09-28)*
      Die Baseline aus `free_memory()` = max(cudaMemGetInfo, MemAvailable+SwapFree)
      konnte größer als `total_memory()` sein (Log: 124,0 GB > 119,7 GB); damit war
      `atlas_own > raw_used` permanent wahr → jeder Hot-Swap-Load fiel in den
      Raw-Fallback, der Co-Tenants/Page-Cache als Atlas-Verbrauch zählt.
      Fix: `kv_budget::compute` clampt `actual_free` und `baseline_free` auf
      `total_mem` (eine Basis für alle Terme, inkl. physical cap). Tests:
      `baseline_above_total_is_clamped_not_implausible`,
      `free_above_total_clamps_used_to_zero` (Replay des 2026-09-11-Falls).
- [x] **0.2 Arena-`reset()` gibt physische Pages zurück.** *(umgesetzt 2026-09-28)*
      `WeightsArena::reset()` setzte nur den Bump-Cursor zurück; die ~47 GiB des
      entladenen 80B-Modells blieben als dirty anonymous pages resident, drückten
      MemAvailable und ließen Folge-Loads scheitern.
      Fix: `reset()` ruft `madvise(MADV_DONTNEED)` über die volle Capacity — die VA
      bleibt konstant (Plan-§2-Invariante hält), der Kernel bekommt die Pages zurück;
      übersprungen bei `--arena-host-register` (CUDA-DMA-Referenzen). Greift auch im
      Preseed-Loop und vor Snapshot-Reloads. Test: `reset_releases_physical_pages`.
      **Empirisch bestätigt auf dem Spark (2026-09-28):** Unload des 80B gibt
      47,6 GB zurück (meminfo 26,6 → 74,3 GB; cuda-free 1,25 → 48,95 GB),
      1480 Device-Frees, 0 failed — `scripts/hotswap_release_test.sh`: 12/12 PASS.
- [x] **0.3 Baseline nach Hot-Swap neu erfassen.** *(umgesetzt 2026-09-28)*
      Die Baseline wurde genau einmal beim Boot gesetzt (`preflight.rs:220`) und war
      bei jedem Reload stale. Fix: `unload_model` re-armt sie nach Teardown +
      Arena-Release aus dem frischen max(cuda, meminfo)-Wert (Log-Zeile
      „KV-budget baseline re-armed post-unload", Target `hotswap`). `--kv-pool-gb`
      bleibt als deterministische Alternative empfohlen (Prod-Tabelle).
- [x] **0.4 Leak im Load-Fehlerpfad beseitigt.** *(umgesetzt 2026-09-28)*
      Ursache bestätigt: `build_model` konsumiert das Backend-Box; bei einem
      Build-Abbruch gab es kein Handle mehr, um die bereits getrackten
      Device-Allokationen freizugeben — sie strandeten bis zum Prozessende
      (die +0,1–0,2 GB pro Retry im Log). Fix: neuer handle-loser Helper
      `spark_runtime::cuda_backend::drain_and_free_tracked_allocs()` (der
      Alloc-Tracker ist prozess-global); der Reload-Fehlerzweig in `serve.rs`
      dräniert damit, resettet die Arena (inkl. MADV_DONTNEED des halb
      geschriebenen Payloads) und loggt `load_fail_cleanup` (Target `hotswap`).
      Empirische Bestätigung auf dem Spark: R5-Drift-Check in
      `scripts/hotswap_release_test.sh`.
- [x] **0.5 `/v1/stats` um die Gate-Entscheidung erweitert.** *(umgesetzt 2026-09-28)*
      Neues Feld `memory.kv_budget_decision`: `source` (raw/manual/auto_baseline/
      auto_implausible), `used_so_far_gb`, `total_budget_gb`, `baseline_free_gb`,
      `cotenants_gb` — publiziert aus `factory::build` über
      `kv_stats::publish_budget_decision`. Damit sind 0.1–0.4 von außen prüfbar
      (Test-Bench-Assertion: `source == "auto_baseline"` nach einem Hot-Swap).
- [x] **0.6 Decode-Zeit-Guard gegen Pool-Erschöpfung (Fall A6).** *(umgesetzt +
      live getestet 2026-09-28)*
      Vor jedem Decode-Schritt gilt jetzt die Invariante `free_blocks ≥ active`
      (jede Sequenz braucht ≤ 1 neuen Block/Schritt): Der Scheduler swappt
      präventiv die größte nicht-grammar-aktive Sequenz auf Disk statt beim
      Alloc-Fehler den GANZEN Batch zu killen (`decode_step.rs:59`). Die
      Swap-in-Bedingung bekam einen Headroom-Term (`num_blocks + active + 1 ≤
      free`), sonst ping-pongt dieselbe Sequenz jeden Schritt durch die Platte.
      Bewusste Ausnahme: eine EINZELNE Sequenz, die allein den Pool überwächst,
      wird nicht geswappt (Restore bräuchte dieselben Blöcke) — sie bekommt den
      sauberen Fehler, andere bleiben unberührt.
      Live-Test (TranslateGemma, Pool absichtlich 68 Blöcke = 1088 Token, Batch 4,
      4 parallele Requests): 7 Guard-Swap-outs, Restores korrekt, kein
      Batch-Kill; die zwei 500er waren exakt der Einzelsequenz-Fall.
      Damit ist `--max-batch-size` > 1 produktionstauglich (Prod-Tabelle: 8).
- [x] **0.7 Quant-Einstellungen von global auf per-Modell umgestellt.**
      *(a–c umgesetzt 2026-09-28; nur das Audit d ist offen — siehe 0.7d unten.)*
      `--kv-cache-dtype` und `--mtp-quantization` sind jetzt `Option<String>`:
      ungesetzt (Normalfall) gilt das MODEL.toml (`default_kv_dtype` bzw. der NEUE
      `[behavior]`-Key `default_mtp_quantization`, durch atlas-kernels
      build_parse/build/codegen/lib gezogen), Fallback fp8/bf16; gesetzt = expliziter
      Override mit lauter Warnung bei Abweichung. Beide Zeilen aus serve.sh entfernt.
      Vollständige Optionsübersicht für die Test-Bench: **`Settings.md`** (Repo-Root).
- [ ] **0.7d Audit der Targets ohne `default_kv_dtype`:** ~15 gb10-Targets laufen
      jetzt auf dem fp8-Fallback statt (vorher) serve.sh-nvfp4 — pro Target per
      Test-Bench-Sweep (Settings.md §4) einen expliziten Wert eintragen und
      einmal die Serve-Matrix fahren.
      Ursprünglicher Befund:
      Befund (2026-09-28): `serve.sh` übergibt pauschal `--kv-cache-dtype nvfp4` und
      `--mtp-quantization nvfp4`. Die Präzedenz-Regel 4 in
      `serve_phases/kv_cache.rs::resolve_kv_cache_config` wertet jeden CLI-Wert
      ≠ `fp8` als bewussten Override → die kalibrierten `default_kv_dtype`-Werte
      aus 7 MODEL.toml (deepseek-v4-flash=fp8; gemma-4-26b/-31b, mistral-small-4,
      step3p7-flash, qwen3.6-27b, metal-qwen3-5-4b=bf16) werden überschrieben —
      exakt die im Code dokumentierten Collapse-Fälle (Gemma `<unused>`,
      Mistral-Token-Loops auf NVFP4-KV). Kein einziges MODEL.toml empfiehlt
      nvfp4-KV; für MTP existiert gar kein MODEL.toml-Key (CLI-Default bf16,
      serve.sh erzwingt nvfp4; einzige Rettung ist der ignore_modules-Override).
      Fix: (a) beide Flags als `Option<String>` — nur explizit gesetzt = Override,
      sonst MODEL.toml-Default mit Fallback fp8/bf16 (löst auch die
      Sentinel-Doppeldeutigkeit von „fp8"); (b) neuer `[behavior]`-Key
      `default_mtp_quantization`; (c) beide Zeilen aus serve.sh streichen;
      (d) Audit: Targets ohne `default_kv_dtype` bekommen bewusst einen
      (sonst flippen ~15 Modelle still von nvfp4 auf fp8) + Serve-Matrix-Lauf.


---

## Instabilitätsfall-Katalog

Grundlage für Phase 1 (jeder Fall braucht ein Log-Event) und Phase 2 (jeder Fall
braucht, wo machbar, einen Bench-Testfall). Vorschlag Tracing-Targets in Klammern.

### A. Speicher & Budget (`mem`, `kv_budget`)
| # | Fall | Symptom / Endzustand |
|---|---|---|
| A1 | DRAM+Swap-Erschöpfung | Linux-OOM-Killer killt den Prozess oder System friert (UMA: GPU-Arbeit konkurriert mit Paging). Vorläufer: MemAvailable↓, SwapFree↓ |
| A2 | OOM-Watchdog-Selbstabbruch | 3 Readings unter Schwelle → `exit(1)` (`spawn_oom_watchdog`) |
| A3 | KV-Budget-Fehlrechnung beim Load | „No memory left for KV cache", Server parkt ohne Modell (der 2026-09-11-Fall; 0.1/0.2) |
| A4 | Leak im Build-Fehlerpfad | jeder fehlgeschlagene Load frisst Speicher, bis nichts mehr lädt (0.4) |
| A5 | Arena-Erschöpfung | „weights arena exhausted" wenn Modell > `--weights-arena-gb` |
| A6 | KV-Pool-Erschöpfung zur Laufzeit | decode-time `block_alloc` scheitert bei vielen langen Sequenzen; Preemption/Disk-Spill-Kaskade |
| A7 | Co-Tenant-Druck | externer Prozess allokiert nach der Budget-Messung → physisches Under-delivery trotz korrektem Budget |
| A8 | Swap-Thrashing | SwapFree→0: System unresponsiv, bevor der OOM-Killer greift |

### B. Hot-Swap-Lifecycle (`hotswap`)
| # | Fall | Symptom / Endzustand |
|---|---|---|
| B1 | Unload-Teardown-Fehler | `gpu.free()` schlägt fehl (`free_failed>0`) → geleakte Device-Allocs, „parked as unloaded" |
| B2 | Load-Fehler nach Unload | Zustand `NoModel` dauerhaft; alle Requests scheitern bis manueller Load (Retry-Schleife im Log) |
| B3 | Drain-Grace-Ablauf | Sequenzen force-finished mit `model_swap`; bei Bugs hängende Prefills |
| B4 | Snapshot-Reload-Fehler | Checksum-Mismatch, O_DIRECT/io_uring-Lesefehler, Model-Hash-Mismatch → Reload fail |

### C. Sequenz-Lifecycle & KV-Cache (`memdebug`, existiert)
| # | Fall | Symptom / Endzustand |
|---|---|---|
| C1 | KV-Block-Leak | Release-Pfad übersprungen (Drop-Backstop, `held_after>0`, Idle-Invariante „KV LEAK") → Pool läuft über Tage voll, keine Admission mehr |
| C2 | Client-Disconnect | gedroppter HTTP-Receiver, Decode läuft bis `max_tokens` weiter und hält Blöcke (Task.md-Motivation) |
| C3 | Endlos-/Loop-Generierung | bei `ATLAS_DISABLE_WATCHDOGS=1`: max_tokens-Burn + lange Blockhaltung; Rollback+Re-Steer-Kaskaden (Cap 2) |
| C4 | SSM-Snapshot-Pool-Erschöpfung | Marconi-/Rollback-Slots aufgebraucht bei vielen Sessions |
| C5 | Prefix-Cache-Wachstum | `cached_evictable` wächst unbegrenzt / Eviction-Bug → schleichender Speicherverlust |
| C6 | KV-Spill-I/O-Fehler | Disk voll, Spill-Datei-Fehler, verwaiste Spill-Dateien |

### D. CUDA & Kernels (`cuda_err`)
| # | Fall | Symptom / Endzustand |
|---|---|---|
| D1 | Kernel-Launch-Fehler / device-side assert | sticky CUDA error vergiftet den Kontext → ALLE Folge-Requests scheitern bis Neustart |
| D2 | Illegal memory access in Modell-Kernels | asynchron, erst beim nächsten sync sichtbar → braucht Kernel-Name+Modell-Attribution am Launch |
| D3 | CUDA-Graph capture/replay | Capture-Fehler pro Slot; Graph-Leaks (monoton sinkendes `gpu_drift_mib`) |
| D4 | Kontext-Leak auf Tokio-Workern | Regression des 2026-07-12-Falls (geleaktes `cuCtxSetCurrent` bewaffnete den OOM-Watchdog) |

### E. Prozess & Umfeld (`proc`)
| # | Fall | Symptom / Endzustand |
|---|---|---|
| E1 | Rust-Panic im Scheduler-Thread | Thread besitzt Modell + CUDA-Kontext → ganzer Server tot; Panic-Hook mit letztem Speicher-Snapshot nötig |
| E2 | Tokenizer/Template/Grammar-Fehler beim Modellwechsel | Request-Fehlschleifen (Jinja, EOS-Konfig, tool-call-Token unresolved) |
| E3 | Doppelstart / halbfertiges Cleanup | zwei Prozesse teilen GPU-Speicher (serve.sh-pkill-Fenster); Port belegt |

## Produktions-Konfiguration (Ziel-Setup, weiterhin im Container)

Abweichungen vom heutigen Dev-Setup (`serve.sh` + `.devpod/compose.yml`):

| Setting | Dev heute | Produktion | Warum |
|---|---|---|---|
| `--max-batch-size` | 4 (seit 0.6) | 8 möglich | Decode-Zeit-Guard (0.6) ist live-getestet; serve.sh fährt jetzt 4 |
| `--swap-space-gb` | 32 (seit 0.6) | 32–64 | Disk-Spill ist das aktive Überlaufventil des Decode-Guards |
| `--kv-pool-gb` | nicht gesetzt | fest setzen (z. B. 16–24) | deterministische Hot-Swaps, entkoppelt vom Messproblem, solange 0.3 offen ist |
| `ATLAS_DISABLE_WATCHDOGS` | 1 | **entfernen** | Code-Kommentar selbst: „re-enable for production"; schützt vor Fall C3 |
| `ATLAS_KV_OVERCOMMIT` | 1 | 1 behalten | hohe `--max-seq-len` ohne Boot-Verweigerung; Risiko wird durch 0.6 + Swap-Space getragen |
| `ATLAS_SKIP_OOM_PREFLIGHT` | 1 | 1 behalten | Preflight-False-Negatives sind UMA-bedingt (`cudaMemGetInfo`); per-Shard-Guard bleibt aktiv |
| `ATLAS_MEM_DEBUG` | optional | aus | per-Request-Tracing nur für Debug-Sessions; die `unload_release`-Bilanz läuft immer |
| Host-Swap | 8 GiB | 32+ GiB (zram/NVMe) | SwapFree ist Teil der Budget-Formel (NVIDIA-Guidance); 8 GiB Puffer war beim 2026-09-11-Fehler faktisch die letzte Reserve |
| `drop_caches` | — | NICHT automatisieren | reines Debug-Werkzeug aus den NVIDIA Known Issues; im Normalbetrieb regelt MemAvailable-Accounting das |

## Phase 1 — Logging-System (dateibasiert, ohne Grafana)

Vorhanden: `tracing` mit `RUST_LOG` (main.rs, nur stdout), `/v1/stats`, `/metrics`,
`memdebug.rs` (deckt Kategorie C bereits ab). Ziel: wenn etwas schiefgeht, steht die
Ursache in einer Datei — kein Dashboard, kein externer Stack.

- [x] **1.1 Datei-Sinks neben stdout.** *(umgesetzt 2026-09-28, ohne neue
      Dependencies)* Zwei per-Prozess-Dateien unter `ATLAS_LOG_DIR` (Default
      `logs/`): `atlas-<ts>.log` (RUST_LOG-Filter, Default INFO+) und
      `critical-<ts>.log` (**alle** WARN+ERROR jedes Targets — die eine Datei,
      in die man schaut, wenn etwas schiefging). `ATLAS_FILE_LOG=0` schaltet ab.
      Rotation = pro Serverstart (wie das bisherige `logs/serve-<ts>.log`-Muster);
      Retention: eigene Dateien (atlas-/critical-/crash-) älter als
      `ATLAS_LOG_RETENTION_DAYS` (Default 14) werden beim Start gelöscht.
- [x] **1.2 Katalog-Abdeckung** *(im Wesentlichen erledigt 2026-09-28)*:
      A1/A8-Vorläufer → Heartbeat-WARNs (1.3); A2/E1 → Crash-Snapshots (1.6);
      A4 → `load_fail_cleanup` + memdebug-Build-Phasen (1.5); B1 →
      `unload_release`-Event + `released`-Block in der Unload-Antwort; A6 →
      „Decode-time guard"-Events; D1–D3 → `cuda_err`-Target (1.4); C* →
      memdebug (bestand). Restlücke: B4-Detailfelder (Snapshot-Reload-Fehler
      tragen bislang nur den anyhow-Kontext) — bei Bedarf nachschärfen.
- [x] **1.3 Speicher-Heartbeat statt Dashboard.** *(umgesetzt 2026-09-28)*
      `main_modules/heartbeat.rs`: 1×/min eine INFO-Zeile auf Target `mem`
      (MemAvailable, SwapFree, cudaFree, KV-Pool used/total, Arena used/cap, RSS);
      greppbar via `grep " mem:" logs/atlas-*.log`. Unterschreitet MemAvailable
      8 GB oder SwapFree 2 GB (via `ATLAS_MEM_WARN_{AVAIL,SWAP}_GB` einstellbar),
      wird WARN geloggt → landet automatisch in critical.log (A1/A2/A8-
      Frühwarnung). Intervall `ATLAS_MEM_HEARTBEAT_SECS` (0 = aus).
- [x] **1.4 Zentraler CUDA-Error-Wrapper (`cuda_err`).** *(umgesetzt 2026-09-28)*
      Statt alle verstreuten Call-Sites umzubauen, sind die vier Choke-Points
      instrumentiert, durch die JEDER Modell-Kernel läuft: `GpuBackend::launch`
      (loggt Kernel-Name aus der neuen Handle→Name-Registry + Grid/Block/SMEM),
      `synchronize`/`device_synchronize` (wo asynchrone Device-Fehler auftauchen)
      und `launch_graph` — alles auf Target `cuda_err` (→ critical.log).
      Sticky-Erkennung (`spark_runtime::cuda_err`, Status 700/710/713–716/719/999):
      ein vergifteter Kontext schaltet `/health` dauerhaft auf 503
      `cuda_context_poisoned` statt endloser Request-500er (Fall D1).
- [x] **1.5 memdebug Build-Pfad.** *(umgesetzt 2026-09-28)*
      `memdebug::on_build_phase("start"/"ok"/"fail")` mit meminfo/CUDA/RSS-
      Snapshot, verdrahtet im Hot-Swap-Reload (deckt A4/0.4 messbar ab).
- [x] **1.6 Crash-Forensik.** *(umgesetzt 2026-09-28)*
      `spark_runtime::crash::write_crash_snapshot`: meminfo + raw-CUDA + KV-Pool +
      letzte Budget-Entscheidung + RSS nach `logs/crash-<ts>.log`, zusätzlich als
      ERROR auf Target `proc` (→ critical.log). Angeschlossen an den Panic-Hook
      in main.rs (E1) und den OOM-Watchdog vor `exit(1)` (A2).

## Phase 2 — Automatisierte Test-Bench auf dem DGX Spark

Vorhanden: Serve-Matrix (`tests/run_all_models.py` + `single_gpu_suite.py` +
`gate_results.py`), `scripts/kv_budget_fix_test.sh` (K1–K4), Bench-/Sweep-Skripte,
Kernel-Microbenches (`crates/atlas-spark-bench/benches`).

- [x] **2.1a Release-Testfall für den Modellwechsel** *(umgesetzt 2026-09-28)*:
      `scripts/hotswap_release_test.sh` fährt das 2026-09-11-Szenario gegen einen
      laufenden Server: 80B laden → Completion → unload (assert: `free_failed == 0`
      UND `released.reclaimed_gb` ≥ 20 GB — vor dem MADV_DONTNEED-Fix war das ~0)
      → TranslateGemma laden (muss 200 liefern) → Completion → zweiter Zyklus mit
      Drift-Assertion (< 2 GB Verlust = Leak-Detektor für 0.4). Stützt sich auf das
      neue `unload_release`-Event (Target `hotswap`) und den `released`-Block in der
      Unload-Antwort (`scheduler/hotswap_ctl.rs`). **Erster Lauf 2026-09-28: 12/12
      PASS** (R2 reclaimed 47,6 GB; R3 TranslateGemma-Load OK — der 2026-09-11-Fall
      ist damit geschlossen; R5 Drift 0,06 GB). Der Completion-Check ist
      kategorie-bewusst (Translate-Modelle via `/v1/translate`).
- [x] **2.1b N×-Hot-Swap-Churn.** *(umgesetzt 2026-09-28)*
      `scripts/hotswap_churn_test.sh`: N Zyklen (Default 5) alternierend über
      Modelle; Assertions: `free_failed == 0` je Unload (C1), meminfo-Drift
      < 2 GB über alle Zyklen (C2, Leak-Detektor A4) und
      `kv_budget_decision.source == auto_baseline` am Ende (C3 — das Gate
      bleibt über Swaps hinweg self-relative; nutzt 0.5).
- [x] **2.2/2.3/2.4 Nightly-Bench + Gates.** *(umgesetzt 2026-09-28)*
      `scripts/nightly_bench.sh`: N1 Health (fängt auch das Sticky-CUDA-503),
      N2 Stats-Sanity (Gate nie `auto_implausible`), N3 Release-Suite, N4 Churn,
      N5 **cuda_err-Gate** (kein `cuda_err`-Event in der neuesten critical-*.log
      ⇒ deckt 2.2 ab), N6 **memdebug-Gate** („KV LEAK"/„release path was
      skipped" in atlas-*.log ⇒ FAIL; mit `ATLAS_MEM_DEBUG=1` fahren ⇒ 2.3).
      Scheduling: Host-Crontab-Zeile ist im Skript-Header dokumentiert
      (`podman exec atlas-gb10 …`) — **auf dem Host noch einzutragen** (aus dem
      Container heraus nicht möglich); dieselbe Zeile dient als
      pre-deploy-Gate für Deploy.md. Noch offen: Kernel-Microbenches
      (`cargo bench -p atlas-spark-bench`) nightly + Diff gegen
      `tests/baselines/`, und die volle Serve-Matrix einhängen.

---

## Phase 3 — Neue Modelle & Kernels

### 3.1 nvidia/Qwen3.8-27B-NVFP4 (Text+Vision-Generativ, 262K Kontext)

Architektur `Qwen3_5ForConditionalGeneration` — die Familie ist im Repo schon
angelegt (`weight_map/ssm_qwen35/fused_vl.rs`, Kernel-Targets `qwen3.5-27b` /
`qwen3.6-27b` unter `kernels/gb10/`), aber dieses Checkpoint hat ein NEUES
Quant-Rezept: **NVFP4 auf MLP + lm_head, FP8 auf Self-Attention UND
Linear-Attention** (ModelOpt v0.48.0, Local-Hessian-Kalibrierung).

- [x] **3.1a Kernel-Target `kernels/gb10/qwen3.8-27b/` anlegen** — DONE 2026-10-02.
      MODEL.toml (Geometrie identisch qwen3.6-27b; neu: `mtp_layers = 1`,
      Sampling temp 1.0/top_p 0.95/top_k 20 lt. Modellkarte, `default_kv_dtype
      = "fp8"`), `nvfp4/`-Variante von qwen3.6-27b kopiert. KEINE neuen
      CUDA-Kernels nötig: die FP8-Arme fahren die vorhandenen common-Kernels
      (`w8a16_gemv` Decode, `fp8_gemm_t_blockscaled` Prefill). NEU dabei:
      hf_id-basierte Target-Auswahl (`atlas_kernels::ptx_for_model_id`,
      MODEL.toml `[model].hf_id` → generierte Tabelle), weil die drei dense
      27B jetzt ZWEI exakte `(qwen3_5, 5120)`-Matches hätten und
      first-match-wins das 3.6-Bundle geliefert hätte.
- [x] **3.1b Weight-Map erweitern** — DONE 2026-10-02, aber ANDERS als geplant:
      Das Checkpoint nutzt NICHT `.weight_scale_inv`, sondern per-tensor-SKALAR
      `.weight_scale` (F32) auf den FP8-Armen und nvidia-Standard-NVFP4-Naming
      auf MLP/lm_head — beides war schon ladbar. Einzige echte Lücke: das
      FP8-Overlay im FullAttention-Arm (`qwen35_dense.rs`) erkannte nur
      2D-block-FP8 hinter `ATLAS_DENSE_FP8`. Jetzt: `mixed_fp8_attn`-Route
      (variant Standard + `proj_is_fp8_any_scale` auf q/k/v/o) default-on,
      Kill-Switch `ATLAS_NO_ATTN_FP8`. GDN-FP8 griff bereits default-on.
      Detection bleibt Heuristik (MIXED_PRECISION → Standard via
      dtype-Sniffing; ein expliziter Arm würde FP8-MLP-Mischrezepte à la Holo
      falsch festnageln).
- [x] **3.1c KV-Cache fp8_e4m3** — DONE 2026-10-02. `default_kv_dtype =
      "fp8"` PLUS `fp8_kv_calibration_tokens = 256` (Checkpoint liefert
      KEINE k/v-Scales; unkalibriertes fp8-KV clippt still auf E4M3 und
      fiel den Greedy-Fib-Kanarienvogel 2/2 — kalibriert besteht die
      volle Suite 16/16, gleichauf mit bf16-KV, bei 2x KV-Kapazität und
      NVIDIA-Parität). Selective Boundary 6/16 bf16-Randlayer aktiv.
- [x] **3.1d Verifikation** — DONE on-box 2026-10-02: Boot ✓ (16 Attn-Layer
      nativ FP8 per-tensor, 48 GDN nativ FP8, MTP geladen, lm_head
      pre-packed NVFP4), `single_gpu_suite.py` 16/16 PASS auf fp8-KV
      kalibriert UND bf16-KV (Coherence 3/3, Fib, Tools 2/2, Needle 3/3
      bis 14.6K; TPS 10.1), in `ATLAS_PRESEED_GENERATIVE` (serve.sh +
      compose.yml). A/B `ATLAS_NO_ATTN_FP8=1`: Prefill ident. (899 vs
      898 tok/s @4424), Decode NVFP4-Requant ~5% schneller (10.65/10.17
      vs 10.10/9.57 c=1) — FP8 bleibt default (Qualitäts-Policy wie GDN
      2026-07-03); fused FP8-Dual-GEMV würde die Lücke schließen.
      262K-Check mit `ATLAS_KV_OVERCOMMIT=1`: bootet, Pool 603K KV-Tokens
      (2 volle 262K-Seqs, Batch 4 via On-Demand-Backpressure), Needle-HIT
      bei 29K Tokens, /v1/stats budget decision ok.
      BEKANNTE EIGENHEIT (kein Regression-Blocker): bei temp=0 +
      rep_penalty=1.05 (Suite-Fib-Betriebspunkt, NICHT der Karten-
      Betriebspunkt temp=1.0) ist das Modell über ALLE KV-Dtypes flaky —
      u.a. Rollen-Token-Loops (`user/assistant/…`) im Thinking, die das
      Budget aufbrauchen; Prefix-Cache-Hits kippen das Greedy-Argmax
      (Wiederholungs-Fails 2/4 auch auf bf16). Am Betriebspunkt sauber.
      Noch offen (separate Läufe außerhalb Dev-Container): Docker-
      Release-Matrix `tests/run_all_models.py` (atlas-release-Gate),
      BFCL-artiger Qualitäts-A/B FP8- vs NVFP4-Attention.

### 3.2 Qwen/Qwen-Image-2.1 (Text-zu-Bild / Bild-Editing, eigener Endpoint)

Komplett neue Workload-Klasse: 7B-DiT (32 Single-Stream-Layer, Flow-Matching-
Sampler, VAE, RGBA-Support, bis 10 Referenzbilder) — kein autoregressives
Decoding, kein paged KV. Läuft am ehesten als eigene Kategorie neben dem
Whisper-Sidecar-Muster statt im Generative-Scheduler.

- [x] **3.2a Eigener API-Endpoint** — DONE 2026-10-02 für Text-zu-Bild:
      `/v1/images/generations` (`api/images.rs`) OpenAI-Images-kompatibel:
      `prompt`, `model`, `n` (1–4), `size` ("WxH", auf 32 px gerastert,
      256 px … 2048x2048/2752x1536), `response_format` (`b64_json` |
      `url` = data:-URI, Atlas hostet keine Dateien), `output_format`
      (png|jpeg|webp), `background` (auto|opaque|transparent — der VAE
      liefert immer RGBA; `transparent` verpackt den Prompt im
      Modellkarten-Format, `auto` behält Alpha nur bei echt transparenten
      Bildern). Erweiterungen: `seed`, `num_inference_steps` (Default 40),
      `aspect_ratio` ("16:9" → 1376x768 bei 1024²-Fläche). Antwort pro
      Bild zusätzlich `seed`. OFFEN: `/v1/images/edits` + `/variations`
      antworten weiter 501 — Bildkonditionierung braucht den
      Qwen3-VL-Vision-Tower im Text-Encoder, den VAE-Encoder und
      block-kausale Präfix-Masken für die Referenzbild-Blöcke (die
      Attention-Kernels können das schon: per-Row-Key-Limit).
      Classifier-free Guidance (`negative_prompt`/`true_cfg_scale`) ist
      nicht implementiert (Modell ist für Sampling ohne CFG gedacht).
- [x] **3.2b `ModelCategory::Image`** — DONE 2026-10-02. Kategorie +
      `ATLAS_PRESEED_IMAGE` (serve.sh + compose.yml) + Endpoint-Mapping;
      wie Whisper ein RESIDENTER SIDECAR außerhalb von Scheduler/Arena
      (lädt beim ersten Request in ~10 s, 26.7 GiB BF16 resident: DiT
      13.3 + Text-Encoder 12.9 + VAE-Decoder 0.5; `/v1/stats` zeigt ihn
      als `residency: sidecar`). Kategorien-400 beidseitig
      (`model_wrong_category`). Speicher-Accounting:
      - KV-Budget: neuer Term `sidecar_reserve` in `kv_budget::compute`
        (`gpu::sidecar_pending_reserve_bytes`): solange der Sidecar
        konfiguriert, aber nicht resident ist, bleiben 34 GiB
        (`ATLAS_IMAGE_RESERVE_GB`) aus dem KV-Pool heraus; nach dem Laden
        nur noch 7 GiB Arbeitsspeicher (die Gewichte misst dann
        `baseline − free`). Die Reserve nimmt höchstens die HÄLFTE des
        Pools, den das Modell ohne sie bekäme (WARN im Log) — sonst
        scheiterte auf knapper Box der Modell-Load mit „No memory left
        for KV cache" (so passiert am 2026-10-02).
      - Preflight: Laden und jede Generierung prüfen `free_memory()`
        gegen Bedarf + 4 GiB Marge und antworten sonst 503 statt den
        Host in den OOM-Killer zu treiben.
      - `cuda_backend::UntrackedAllocs`: Sidecar-Allokationen (Image UND
        Whisper) landen nicht mehr im Watermark-Tracker eines parallel
        laufenden Hot-Swap-Builds — sonst würden sie beim Unload dieses
        Modells mit freigegeben (latenter Use-after-free).
      - Eigener cuBLASLt-Workspace (`bf16_gemm_act_weight_t_ws`), weil
        der Sidecar auf eigenem Thread neben dem Scheduler rechnet.
- [x] **3.2c Kernel-Target `kernels/gb10/qwen-image-2.1/`** — DONE
      2026-10-02. `bf16/qwen_image.cu` (16 Kernels: AdaLN-LayerNorm,
      Per-Head-RMSNorm+RoPE, rechteckiger maskierter Row-Softmax,
      NHWC-VAE-Ops) + common-GEMMs/cuBLASLt; Pipeline nativ in
      `spark-model/src/qwen_image/` (st/ops/text/dit/vae/sched):
      - Text-Encoder: Qwen3-VL-8B-LLM standalone (36 Layer, GQA 32/8),
        liefert die Hidden-States VOR der finalen RMSNorm; Embedding-
        Zeilen per pread statt 1.2 GiB Tabelle auf dem Device.
      - DiT: Prompt läuft EINMAL als Präfix-Pass (t=0-Modulation, kausal,
        K/V je Layer gecacht), jeder Denoising-Schritt rechnet nur die
        Target-Tokens gegen [Präfix-K/V, Target-K/V] — mathematisch der
        „extract"+„cached"-Pfad der Referenz ohne Sonderfall im ersten
        Schritt. Attention = GEMM-SDPA pro Head (QKᵀ f32 → Softmax → P·V).
      - VAE-Decoder: NHWC, Conv = im2col-Bänder + GEMM, letzte Conv mit
        f32-Ausgabe (8-bit-Pixel nicht an BF16-Mantisse gebunden).
      - Sampler/Schedule (dynamischer Shift + Terminal-Stretch) hostseitig;
        eigener Seed-RNG (nicht torch-bitkompatibel).
      Verifikation: `examples/qwen_image_golden` gegen die diffusers-
      Referenz (CPU fp32, `scripts/qwen_image_ref_dump.py`): ALLE Stufen
      PASS — Text-Encoder cos 0.9986, DiT-Blöcke/Noise-Preds ≥ 0.9994,
      finale Latents 0.99998, VAE-Bild 0.99998. `tests/image_smoke.py`
      (Serve-Matrix-Baustein) 11/11 PASS. Performance GB10: 1024x1024 /
      40 Schritte ≈ 90 s (2.2 s/Schritt), 512x512 / 8 Schritte 3.4 s.
      Koexistenz-Lauf (Qwen3.8-27B in der Arena + Sidecar, ein Server):
      Chat vor / während / nach einer 1024²-Generierung jeweils 200,
      `/v1/stats` zeigt beide resident (arena 20.4 GiB + sidecar
      26.7 GiB); KV-Pool des Chat-Modells dabei 16.5 GB (Reserve-Regel
      „halber Pool", WARN im Log).
      ERLEDIGT (2026-10-02 nachmittags):
      - Fairness — GELÖST (Commit e915a70): der Sidecar rendert auf
        einem EIGENEN non-blocking CUDA-Stream
        (`AtlasCudaBackend::use_private_stream`, least priority);
        vorher serialisierte die geteilte Registry-Stream-FIFO jede
        Chat-Decode-Launch hinter der kompletten DiT-Kette. Gemessen:
        Chat während 1024²/40-Generierung 9.8/11.9/18.3 s bei 7.3 s
        Baseline (vorher 112 s); Generierung 108 s (solo ~94 s).
        Hinweis: least priority (0) == default priority des
        Registry-Streams (Range 0..-5) — echter Dispatch-Vorrang
        entstünde erst, wenn der Scheduler-Stream Richtung -5
        angehoben wird (Option, nicht verdrahtet).
      - 2048x2048 GEMESSEN: 40 Schritte in 1041.8 s (~26 s/Schritt vs
        2.2 s/Schritt bei 1024²) — der speichergebundene Row-Softmax
        (16384 Tokens, N×S×32 Heads × 32 Layer) dominiert wie
        prognostiziert. Funktional korrekt (Preflight verlangt 6.9 GiB
        + 4 GiB Marge; neben 27B-Arena+KV nur mit verkleinertem
        KV-Pool möglich, sonst korrektes 503). Für praktikable 2048²-
        Latenz bleibt der fusionierte Flash-Attention-Kernel der
        nächste Performance-Schritt.
      OFFEN:
      - Speicher-Befund (2026-10-02): der NVIDIA-Treiber hält den
        Gerätespeicher beendeter Prozesse bzw. entladener KV-Pools
        in einem Pool — in `MemAvailable` „belegt", für neue
        CUDA-Allokationen aber wiederverwendbar; nach einem Kill unter
        Last auch 15–20 min (nach sauberem TERM-Shutdown ~1–2 min).
        Das ist die „co-tenants"-Zeile im KV-Budget-Log; Budget,
        Reserve und Preflight rechnen dadurch konservativ (Details
        Doku.md §3.2).
      - Flash-Attention-Kernel für den DiT (2048²-Latenz, s. o.).
      - `tests/run_all_models.py`-Einhängung (Docker-Release-Matrix).
- [ ] **3.2d Lizenz-Hinweis:** Qwen Research License (nicht Apache-2.0) —
      vor Produktivnutzung klären; betrifft auch das Publishing im
      `avarok/atlas-gb10`-Image (atlas-release MODEL.toml-Gate).

### 3.3 datalab-to/chandra-ocr-2 (OCR, eigener `/v1/ocr`-Endpoint, nur dieses Modell)

Chandra 2 ist ein Qwen-3.5-basiertes OCR-Modell (Markdown/HTML/JSON-Output mit
Layout, SOTA olmocr-Bench 85,8 %) — Architektur-Tag `qwen3_5`, d. h. die Basis
sind die vorhandenen qwen3.5-Targets und der ssm_qwen35-Weight-Loader
(inkl. VL-Pfad: `weight_map/ssm_qwen35/fused_vl.rs` existiert bereits).
Lizenz beachten: modifizierte OpenRAIL-M (frei für Research/Startups < 2 M$,
nicht kompetitiv zur Datalab-API einsetzbar) — vor Produktivnutzung klären.

- [x] **3.3a Kernel-Target `kernels/gb10/chandra-ocr-2/`** — DONE 2026-10-02
      (Commit ec12c86). Geometrie = holo-3.1-4b exakt (dense 4B, 32L 8/24,
      hidden 2560) + `mtp_layers = 1`; Sampling aus chandra/model/vllm.py
      (temp 0.0 / top_p 0.1); thinking + Loop-Watchdog aus (OCR wiederholt
      legitim; Degenerations-Schutz ist der Chandra-Retry im Handler);
      `default_kv_dtype = "bf16"`. BF16-Checkpoint → Bf16Raw-Runtime-NVFP4
      auf den Holo-Kernels. Bewusst OHNE `[[model_types]]` — Auswahl NUR
      über die neue hf_id-Target-Selektion, damit der (qwen3_5, 2560)-
      Fallback bei der Holo-Familie bleibt.
- [x] **3.3b `ModelCategory::Ocr`** — DONE 2026-10-02 (Commit ec12c86).
      Kategorie + `ATLAS_PRESEED_OCR` + `/v1/ocr` (api/ocr.rs), Sperre
      beidseitig verifiziert (400 model_wrong_category am Chat-Endpoint).
      Request: {image: base64|data-URI (Bild ODER PDF, %PDF-Sniff),
      page_range, prompt_type, max_tokens, max_retries,
      include_headers_footers, include_images}; Response pro Seite
      {html, markdown, chunks[{label,bbox,content}], images{name→b64-webp},
      retries} + usage.
- [x] **3.3c Chandra-Pipeline nativ in Rust** — DONE 2026-10-02 (Commit
      a420860): `spark-server/src/ocr/{input,prompts,output,markdown}.rs`
      + Retry im Handler. PDF via pdfium-render/libpdfium (bblanchon
      arm64, Dockerfile-Eintrag; Prozess-Lock + spawn_blocking), DPI-Regel
      und Flatten wie die Referenz; prompts 1:1 inkl. Python-Listen-Repr;
      output: Layout-Chunks/HTML-Filter/WebP-Crops; markdown: Subset-
      Konverter mit markdownify-Referenzoptionen (Byte-Parität → 3.3d-
      Golden-Test, nicht by construction); detect_repeat_token +
      Temperatur-Bump-Retry. Abweichung vom Plan: multipart gibt es
      nicht — ein JSON-Feld nimmt Bild ODER PDF (Magic-Sniff).
      Ursprüngliches Mapping (Referenz):
      - `chandra/input.py` → `ocr/input.rs`: PDF-Rendering via **pdfium**
        (Python nutzt pypdfium2; Rust-Crate `pdfium-render` bindet dieselbe
        libpdfium → identisches Rendering erreichbar). Verhalten exakt
        übernehmen: `FPDFPage_Flatten` (Formulare/Annotationen plätten),
        Render-DPI = max(IMAGE_DPI, min_pdf_image_dim/min_page_dim × 72),
        RGB-Konvertierung; Bilder: Min-Dimension-Upscale mit **Lanczos**
        (image-Crate `Lanczos3`). Settings-Konstanten (IMAGE_DPI,
        MIN_IMAGE_DIM, MIN_PDF_IMAGE_DIM) aus `chandra/settings.py`
        übernehmen. libpdfium muss ins gb10-Image (Dockerfile).
      - `chandra/prompts.py` → `ocr/prompts.rs`: die Prompt-Templates je
        `prompt_type` (ocr_layout/…) 1:1 als Konstanten, inkl.
        ALLOWED_TAGS/ALLOWED_ATTRIBUTES-Listen (HTML-Subset mit
        `data-bbox`/`data-label`, Mathe als KaTeX in `<math>`).
      - `chandra/model/vllm.py` + `schema.py` → entfällt als HTTP-Client:
        der Handler ruft direkt die interne Generative-Pipeline auf
        (gleicher Pfad wie /v1/chat mit Bild-Input; Sampling-Settings aus
        deren vLLM-Aufruf ins MODEL.toml-`[sampling]` übernehmen).
      - `chandra/output.py` → `ocr/output.rs`: Modell-HTML parsen
        (Top-Level-`<div data-label=…>` = Layout-Chunks; `Blank-Page`
        überspringen, Header/Footer optional), `data-bbox`-Crops der
        Image/Figure-Chunks als WebP extrahieren, HTML→Markdown-Konverter
        (markdownify-Äquivalent für das erlaubte Tag-Subset — Tabellen,
        Checkboxen, `<math>` erhalten). HTML-Parsing z. B. mit `scraper`.
      - Response-Format von `/v1/ocr`: `{markdown, html, chunks[{label,
        bbox, content}], images{name→base64-webp}, usage}` — deckungsgleich
        mit dem, was die Python-Lib pro Seite liefert; PDFs mit mehreren
        Seiten als Array + `page_range`-Parameter (Syntax aus
        `input.py::parse_range_str`).
- [ ] **3.3d Verhaltens-Parität testen** — TEILWEISE 2026-10-02:
      `tests/ocr_smoke.py` (Modellkarten-Bilder → /v1/ocr, Keyword- +
      Schema-Checks) läuft 3/3 PASS on-box (bench 16s, multilingual 15s,
      handwritten_form 290s mit 1 Degenerations-Retry — Retry-Mechanik
      greift wie die Referenz); `ATLAS_PRESEED_OCR` in serve.sh +
      compose.yml. OFFEN: Golden-Diff gegen die Python-Referenz
      (braucht torch/vLLM-Umgebung — Kandidat fürs Release-Gate neben
      der Docker-Serve-Matrix) und die Lizenzklärung (OpenRAIL-M
      modifiziert: frei für Research/Startups < 2 M$, nicht kompetitiv
      zur Datalab-API) vor Produktivnutzung/Image-Publishing.

## Bestandsaufnahme Tests/Benchmarks

| Zweck | Existiert | Wo |
|---|---|---|
| Server von außen benchmarken | Ja | `tests/run_all_models.py` (Serve-Matrix), `tests/single_gpu_suite.py`, `tests/atlas_bench_comprehensive.py`, `scripts/bench_per_model.sh`, `scripts/run_full_benchmark.sh`, `bench/` |
| KV-Budget/Speicher-Regression | Ja | Unit: `crates/spark-model/src/factory/kv_budget.rs`; on-box: `scripts/kv_budget_fix_test.sh` (K1–K4 gegen `/v1/stats`) |
| Per-Request-Speichertracking | Ja | `crates/spark-server/src/memdebug.rs` (`ATLAS_MEM_DEBUG=1`, Doku.md §7) — deckt Requests ab, nicht den Build-Fehlerpfad |
| Kernel-Fehler-Logging pro Modell | Nur teilweise | CUDA-Error-Handling verstreut; kein einheitliches Target/kein Bench-Gate → 1.4/2.2 |
| Automatischer Gesamtlauf auf dem Spark | Teilweise | Matrix + Gate existieren, aber nicht geschedult → 2.4 |

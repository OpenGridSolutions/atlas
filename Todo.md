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
- [ ] **0.5 `/v1/stats` um die Gate-Entscheidung erweitern:** `UsedSource`
      (Raw/Manual/AutoBaseline/AutoImplausible), Baseline, atlas_own, cotenants als
      Felder exportieren, damit 0.1–0.4 von außen (Test-Bench, Phase 2) prüfbar sind.
- [ ] **0.6 Decode-Zeit-Guard gegen Pool-Erschöpfung (Fall A6).**
      Der Swap-out-Pfad läuft nur beim Eintreffen NEUER Requests (`mod.rs:392ff`).
      Wachsen die bereits aktiven Sequenzen den Pool leer, ohne dass etwas Neues
      reinkommt, schlägt `decode_batch` beim Block-Alloc fehl und der Fehlerpfad
      (`decode_step.rs:59`) beendet ALLE aktiven Sequenzen. Fix: vor jedem
      Decode-Schritt prüfen, ob für jede wachsende Sequenz ein freier Block
      existiert; sonst EINE Sequenz (größte, nicht grammar-aktiv) swappen statt den
      Batch zu killen. Voraussetzung für `--max-batch-size` > 1 in Produktion.
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
| `--max-batch-size` | 1 | 4 jetzt, 8 nach 0.6 | Continuous Batching aktivieren; ohne 0.6-Guard killt Decode-Zeit-Erschöpfung den ganzen Batch |
| `--swap-space-gb` | 3 (Default) | 32–64 | Disk-Spill ist das Überlaufventil bei Batch > 1 mit langen Kontexten |
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

- [ ] **1.1 Datei-Appender neben stdout:** `tracing-appender` mit täglicher Rotation
      nach `logs/` (ersetzt das ad-hoc `logs/serve-<ts>.log`-Tee). Zwei Sinks:
      `logs/atlas.log` (alles ab INFO, JSON-Lines) und `logs/critical.log`
      (nur WARN+ERROR der Targets `mem`, `kv_budget`, `hotswap`, `cuda_err`,
      `memdebug`, `proc` — unabhängig vom globalen `RUST_LOG`-Filter).
      Retention per Rotation begrenzen (z. B. 14 Tage).
- [ ] **1.2 Katalog-Abdeckung herstellen:** jedes Ereignis aus dem Katalog bekommt ein
      strukturiertes Event auf seinem Target mit den Zahlen, die man zur Diagnose
      braucht (A: meminfo+cuda-Snapshot; B: freed/failed/Bytes/Dauer; D: Kernel-Name,
      Modell, Grid/Block). Lücken heute vor allem: A1/A8-Vorläufer, B4-Details, D*.
      *Teil erledigt (2026-09-28):* B1 hat jetzt das `unload_release`-Event (Target
      `hotswap`): freed/failed, KV-Pool-Belegung vor Teardown, Arena-Payload,
      meminfo+cuda vor/nach, `reclaimed_gib` — zusätzlich als `released`-Block in
      der `/admin/models/unload`-Antwort für externe Assertions.
- [ ] **1.3 Speicher-Heartbeat statt Dashboard:** 1×/min eine kompakte INFO-Zeile auf
      `mem` (MemAvailable, SwapFree, cudaFree, KV-Pool used/total, Arena used, RSS).
      Ein greppbares Zeitprofil ersetzt Grafana: `grep '"target":"mem"'
      logs/atlas.log`. WARN-Schwellen (z. B. SwapFree < 2 GiB, MemAvailable < 8 GiB)
      landen zusätzlich in `critical.log` — als Frühwarnung vor A1/A2/A8.
- [ ] **1.4 Zentraler CUDA-Error-Wrapper (`cuda_err`):** heute verstreut über
      `buffers.rs`, `fast_weights`, `load_fns.rs`. Jeder Launch-/Sync-Fehler loggt
      Kernel-Name, Modell, Shapes, letzte Allokation; sticky-Error-Erkennung markiert
      den Server-Zustand (Health-Endpoint → 503), statt endlos Request-Fehler zu
      produzieren (D1).
- [ ] **1.5 memdebug auf den Build-/Swap-Pfad ausweiten:** Events build_start/
      build_fail/build_ok mit meminfo-Snapshot vorher/nachher (deckt A4/0.4 ab).
- [ ] **1.6 Crash-Forensik:** Panic-Hook, der vor dem Abort einen letzten
      meminfo+cudaMemGetInfo+Arena+Pool-Snapshot nach `logs/crash-<ts>.log` schreibt
      (E1); OOM-Watchdog schreibt dasselbe vor `exit(1)` (A2).

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
- [ ] **2.1b N×-Hot-Swap-Churn** über alle Preseed-Kategorien mit Assertion
      „`used_so_far` wächst nicht monoton" über `/v1/stats` (breiterer Leak-Detektor
      als der 2-Zyklen-Check in 2.1a).
- [ ] **2.2 Kernel-Fehler-Lauf:** pro Modell der Preseed-Matrix ein kurzer
      Inferenz-Sweep; Bench-Ergebnis FAIL bei jedem Event in `critical.log` mit
      Target `cuda_err` (nutzt 1.4). Kernel-Microbenches nightly, Diff gegen
      `tests/baselines/`.
- [ ] **2.3 Per-Request-Speichertracking im Bench:** Läufe mit `ATLAS_MEM_DEBUG=1`,
      memdebug-Zeilen einsammeln; `KV LEAK` / monotones `gpu_drift_mib` ⇒ FAIL;
      Drift-Trend pro Lauf archivieren.
- [ ] **2.4 Scheduling auf dem Spark:** Cron/systemd-Timer auf dem Host, der die
      volle Matrix nightly im bestehenden Container fährt (`podman exec`), plus
      schneller Smoke-Satz (Health, 1 Chat, kv_budget K1, `critical.log` leer) als
      pre-deploy-Gate in `Deploy.md`. Ergebnis: eine Zusammenfassungszeile +
      `tests/all_models_results/`; bei FAIL bleiben die Server-Logs des Laufs liegen.

---

## Phase 3 — Neue Modelle & Kernels

### 3.1 nvidia/Qwen3.8-27B-NVFP4 (Text+Vision-Generativ, 262K Kontext)

Architektur `Qwen3_5ForConditionalGeneration` — die Familie ist im Repo schon
angelegt (`weight_map/ssm_qwen35/fused_vl.rs`, Kernel-Targets `qwen3.5-27b` /
`qwen3.6-27b` unter `kernels/gb10/`), aber dieses Checkpoint hat ein NEUES
Quant-Rezept: **NVFP4 auf MLP + lm_head, FP8 auf Self-Attention UND
Linear-Attention** (ModelOpt v0.48.0, Local-Hessian-Kalibrierung).

- [ ] **3.1a Kernel-Target `kernels/gb10/qwen3.8-27b/` anlegen** (MODEL.toml nach
      dem Muster von `qwen3.6-27b`): Layer-Geometrie aus dem HF-Config übernehmen;
      Quant-Variante „mixed nvfp4-fp8" — NVFP4-GEMMs für MLP/lm_head existieren,
      neu sind die **FP8-Attention-** und **FP8-GDN/Linear-Attention-Kernels**
      (bisherige qwen3.5/3.6-Targets rechnen die Attention-Arme in NVFP4 bzw. BF16).
- [ ] **3.1b Weight-Map erweitern:** `weight_map/ssm_qwen35/` um die
      FP8-Tensor-Namen (`.weight_scale_inv`-Layout aus ModelOpt) für die
      attention/linear-attention-Arme ergänzen; `quant_format`-Erkennung muss das
      Misch-Rezept sauber pro Modul auflösen statt global NVFP4 anzunehmen.
- [ ] **3.1c KV-Cache fp8_e4m3 — pro Modell konfigurieren.** NVIDIA served das
      Modell mit `--kv-cache-dtype fp8_e4m3`. Der Mechanismus dafür existiert
      bereits pro Modell: das Kernel-Target-MODEL.toml kann `default_kv_dtype`
      deklarieren (`serve_phases/kv_cache.rs::resolve_kv_cache_config`,
      Präzedenz: explizite CLI-Angabe > MODEL.toml-Default > CLI-Default `fp8`),
      und der Hot-Swap wendet es per `prepare_swap_target` beim Modellwechsel an
      — wechselnde Dtypes je geladenem Modell sind also schon abgedeckt.
      Zu tun: `default_kv_dtype = "fp8"` im neuen `qwen3.8-27b`-MODEL.toml setzen,
      bei 262K-Kontext gegen den Selective-Boundary-Mix (bf16-Randlayer)
      benchen (Long-Context-Needle), und prüfen, ob die FP8-KV-Kernels für die
      FP8-Attention-Arme dieses Checkpoints durch den Kernel-Preflight kommen.
- [ ] **3.1d Verifikation:** in die Serve-Matrix aufnehmen (`ATLAS_PRESEED_GENERATIVE`),
      Coherence + Tool-Calls + Long-Context-Needle; KV-Budget-Check bei 262K
      max-seq-len nur mit `ATLAS_KV_OVERCOMMIT=1` sinnvoll (Worst-Case-Rechnung!).

### 3.2 Qwen/Qwen-Image-2.1 (Text-zu-Bild / Bild-Editing, eigener Endpoint)

Komplett neue Workload-Klasse: 7B-DiT (32 Single-Stream-Layer, Flow-Matching-
Sampler, VAE, RGBA-Support, bis 10 Referenzbilder) — kein autoregressives
Decoding, kein paged KV. Läuft am ehesten als eigene Kategorie neben dem
Whisper-Sidecar-Muster statt im Generative-Scheduler.

- [ ] **3.2a Eigener API-Endpoint:** die vorhandenen Stubs
      `/v1/images/generations`, `/v1/images/edits`, `/v1/images/variations`
      (`serve_router.rs:119-121`, `api::images_stub`) OpenAI-Images-kompatibel
      implementieren (prompt, size/aspect_ratio, n, response_format b64/url;
      edits mit Referenzbild(ern) + optionaler Maske; RGBA via
      `background: "transparent"`).
- [ ] **3.2b Neue `ModelCategory::Image`** (+ `ATLAS_PRESEED_IMAGE`,
      Endpoint-Mapping in `hotswap.rs`), damit das Kategorien-Routing (400 an
      fremden Endpoints) und Lazy-Swap greifen. Speicher: ~14 GiB BF16 — muss ins
      Arena/KV-Budget-Accounting als Nicht-KV-Consumer eingebucht werden
      (Fall A7-Verwandter: der DiT konkurriert mit dem KV-Pool des
      Generative-Modells um dieselbe UMA).
- [ ] **3.2c Kernel-Target `kernels/gb10/qwen-image-2.1/`:** DiT-Block-Kernels
      (Mixed-Granularity-Attention über Bild-Latents ohne Paging, AdaLN,
      Patch-Embed), VAE-Decode, Flow-Matching-Steps; Text-Encoder ist ein
      Qwen-LLM → vorhandene Prefill-Kernels wiederverwendbar.
- [ ] **3.2d Lizenz-Hinweis:** Qwen Research License (nicht Apache-2.0) —
      vor Produktivnutzung klären; betrifft auch das Publishing im
      `avarok/atlas-gb10`-Image (atlas-release MODEL.toml-Gate).

## Bestandsaufnahme Tests/Benchmarks

| Zweck | Existiert | Wo |
|---|---|---|
| Server von außen benchmarken | Ja | `tests/run_all_models.py` (Serve-Matrix), `tests/single_gpu_suite.py`, `tests/atlas_bench_comprehensive.py`, `scripts/bench_per_model.sh`, `scripts/run_full_benchmark.sh`, `bench/` |
| KV-Budget/Speicher-Regression | Ja | Unit: `crates/spark-model/src/factory/kv_budget.rs`; on-box: `scripts/kv_budget_fix_test.sh` (K1–K4 gegen `/v1/stats`) |
| Per-Request-Speichertracking | Ja | `crates/spark-server/src/memdebug.rs` (`ATLAS_MEM_DEBUG=1`, Doku.md §7) — deckt Requests ab, nicht den Build-Fehlerpfad |
| Kernel-Fehler-Logging pro Modell | Nur teilweise | CUDA-Error-Handling verstreut; kein einheitliches Target/kein Bench-Gate → 1.4/2.2 |
| Automatischer Gesamtlauf auf dem Spark | Teilweise | Matrix + Gate existieren, aber nicht geschedult → 2.4 |

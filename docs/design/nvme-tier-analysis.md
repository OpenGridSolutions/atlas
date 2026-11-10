# NVMe-Tier, Druck-Scheduler, Priorisierung — Code-Impact-Analyse

Stand: 2026-10-05, Branch `modifications`. Analyse der Anforderungen
„AtlasInference Server – Anforderungen" (Jan Groth, Oct 5 2026) gegen den
realen Code-Stand. Dieses Dokument beantwortet die sofort code-beantwortbaren
offenen Punkte aus §14, kartiert den Impact jeder Anforderungsgruppe auf die
bestehenden Module und leitet die Umsetzungsreihenfolge ab.

---

## 1. Antworten auf die offenen Punkte aus §14

### 1.1 Teilt der RAM-Prefix-Cache heute KV zwischen VL-Requests mit gleichem Text und verschiedenen Bildern?

**Nein — aber nur, weil VL-Prompts den Cache komplett umgehen.** Das
Kollisionsproblem ist als Issue #58 bekannt und wurde durch ein hartes Gate
gelöst: jeder Prompt, der Vision-Pad-Tokens enthält, bekommt
`PrefixMatch::empty()` und wird nie in den Cache eingefügt
(`tokens_have_vision_pad`-Guards in `prefill_a.rs:81`, `prefill_c.rs:158`,
`prefill_b/prefix_lookup.rs:27`, `sequence.rs:55`, `decode_checkpoint.rs:81`;
der SSM-Snapshot wird im Vision-Fall explizit verworfen,
`prefill_d.rs:230-240`).

Konsequenz für KV-4: Es gibt **keinen Korrektheitsbug zu fixen**, sondern ein
Gate zu **ersetzen**. Medien-Digests in der Schlüsselkette machen VL-/OCR-
Prompts erstmals cachebar (heute: 0 % Reuse). Der Regressionstest aus KV-4
(gleicher Text, zwei Bilder → kein geteilter Block ab Bild 1) wird zum
Akzeptanztest für das Entfernen des Gates.

### 1.2 Verwendet der Radix-Cache verkettete Schlüssel oder reine Token-Sequenzen?

**Beides, und genau das ist die Falle.** Die Baumkinder sind per
**Token-Chunk** gekeyt (`HashMap<Vec<u32>, NodeId>`,
`radix_tree/inner.rs:13`); zusätzlich führt jeder Knoten einen verketteten
FNV-1a-`context_hash` (Parent-Hash + Chunk-Tokens, `inner.rs:44-51`), der beim
Walk validiert wird (`inner.rs:118`). Heute ist der `context_hash` eine reine
Funktion des Token-Prefixes — deshalb kann `insert` ihn auf bestehenden Knoten
gefahrlos überschreiben (`inner.rs:269`).

Sobald Medien-Digests in die Kette eingehen, gilt das nicht mehr: gleicher
Token-Prefix + verschiedene Bilder ⇒ verschiedene Ketten, aber derselbe
Chunk-Key ⇒ **ein** Knoten. Der Insert-Überschreibpfad würde dann den
physischen Block von Request A mit dem Kontext-Hash von Request B versehen —
ein echter Korruptionspfad. **KV-3 muss daher die Kinder direkt auf den
128-Bit-Kettenschlüssel umstellen** (`HashMap<BlockKey, NodeId>`), nicht nur
die Validierung tauschen. Der Schlüssel wird pro Request vorab aus Tokens +
Medien-Digests berechnet (inkrementell, O(1) pro Block) und dient identisch
als NVMe-Store-Adresse — ein Identitätssystem (E2).

### 1.3 Entspricht das Retokenisieren zurückgeschickter Assistant-Antworten den generierten Token-IDs?

**Für den Prompt-Anteil ja, für den generierten Anteil nicht garantiert.**
Die Historie bis zum Beginn der abgebrochenen Antwort wird beim Folge-Request
durch dasselbe Template deterministisch identisch tokenisiert — dieser Prefix
(typisch der Löwenanteil) trifft sicher. Die generierten Tokens selbst sind
Sampling-Output und nicht zwingend die kanonische BPE-Zerlegung ihres Textes;
beim Retokenisieren können andere Merges entstehen, dann endet der Treffer an
der letzten Blockgrenze vor der Abweichung. Thinking-Inhalte werden in
Folge-Requests ohnehin nicht zurückgeschickt. **DRN-2 bleibt sinnvoll** (der
Tail-Record ist billig und der Prompt-Prefix trifft immer), aber der
Zusatznutzen der Output-Blöcke ist begrenzt — kein Grund für Mehraufwand,
`on_release` persistiert die Sequenz ohnehin als Ganzes. Ein Messpunkt
(NVMe-Restore-Trefferquote, TEL-6) zeigt den realen Effekt.

---

## 2. Impact-Map: Anforderungen → bestehender Code

### 2.1 Block-Schlüssel (KV-3/4/5) — Fundament, zuerst

| Berührt | Änderung |
|---|---|
| `spark-runtime/src/radix_tree/inner.rs` | Kinder-Keying `Vec<u32>` → `BlockKey` (128 bit); `context_hash`-FNV entfällt zugunsten der BLAKE3-Kette; `partial_suffix` bekommt den Tail-Schlüssel |
| `spark-runtime/src/prefix_cache.rs` | Trait-Signaturen: Lookup/Insert/Release bekommen die vorberechnete Schlüsselfolge statt roher Tokens als Cache-Identität |
| neues Modul `spark-runtime/src/block_keys.rs` | `BlockKeyChain`: inkrementelle BLAKE3-128-Kette aus k₀=H(model_hash‖layout_hash), Token-Chunks, Medien-Digests; wächst im Decode mit |
| `spark-model` Prefill-Pfade (`prefill_a/b/c/d`, `sequence.rs`, `decode_checkpoint.rs`) | Schlüsselkette in `SequenceState` führen; Vision-Gates entfernen (durch Digests ersetzt) |
| `spark-server` API-Pfad (`api/chat`, tokenizer Vision-Pad-Expansion) | BLAKE3-Digest der Bild-Rohdaten + Preprocessing-Parameter berechnen; Digest→Token-Span-Zuordnung in den `InferenceRequest` |
| `layout_hash` | neu, aus `ModelConfig`-Geometrie + per-Layer-KV-Dtypes + Blockgröße + RoPE + FP8-Skalen + Format-Version; Quelle: `serve_phases/kv_cache.rs` (Dtype-Aufbau) + Snapshot-`model_version_hash`-Muster |

Risiko: breiteste Signaturänderung des Vorhabens (≈ 8 Dateien in spark-model).
Dafür eliminiert sie das latente Insert-Überschreib-Problem und aktiviert
VL-Caching. Abhängigkeit für ALLES NVMe-seitige.

### 2.2 NVMe-Block-Store (NVM-*) — neu, gut isolierbar

Neues Modul `spark-storage/src/block_store/`. Verzeichnis-Layout, 4-KiB-Header,
xxh3, atomares tmp+fdatasync+rename, In-Memory-Index + 60-s-Sicherung,
GC (TTL + min-free + Waisen), Schreibbudget. I/O über den vorhandenen
io_uring/O_DIRECT-Unterbau (`backend/io_uring.rs`, `backend/posix.rs` —
beide existieren samt Registered-Buffer-Verwaltung); Dev-Container ohne
io_uring (seccomp) → POSIX-O_DIRECT-Fallback ist bereits der etablierte Pfad.
Komplett ohne GPU unit-testbar. Der bestehende Disk-Spill (`kv_spill.rs`,
gepuffert, ohne Checksummen) bleibt in v1 unangetastet (NVM-14 = SOLL).

### 2.3 Persist/Restore-Integration (NVM-1/2, KV-7, BAT-6/7)

- **Persist:** Hook in `scheduler/lifecycle.rs` (`finish_sequence*`,
  `free_sequence`-Umfeld): D2H-Kopie der Blöcke (Muster existiert in
  `sequence.rs:347-391`), Übergabe an die Schreib-Queue (eigener Thread,
  `write_queue_max_bytes`-Limit). Vor-Swap-Persist in `hotswap_ctl.rs::poll`
  zwischen Force-Finish und Unload-Reply, mit `persist_timeout` (DRN-3/6).
- **KV-7:** Radix-Eviction (`inner.rs::evict`) bekommt ein
  Persistiert-Prädikat (Store-Index-Lookup per Knoten-Schlüssel); unpersistierte
  Blöcke werden übersprungen.
- **Restore:** asynchrones Staging in Host-Puffer VOR der Scheduler-Übergabe
  (Dispatcher-Stufe); der `InferenceRequest` trägt die gestagten Blöcke, der
  Scheduler lädt sie beim Prefill-Start hoch und fügt sie in den Radix-Baum
  ein — die bestehende Prefix-Lookup-Logik greift danach unverändert. Kein
  neuer Pfad im Modell-Layer.

### 2.4 SSM (SSM-1..8)

Marconi-Snapshots existieren (`SsmSnapshotIndex` in `radix_tree/snapshot.rs`,
`session_manager.rs`); Schlüssel ist heute `hash_token_prefix` (FNV über
Tokens) + `session_hash`-Isolation. Umstellung auf die Kettenschlüssel ersetzt
beides (Medien-Digests machen die Session-Isolation für die Bild-Tainted-Fälle
überflüssig). Tail-Snapshots + NVMe-Typ `ssm`/`tail` sind neu. Der Disk-Spill
sichert SSM-Zustand bereits roh (`sequence.rs:347-391`) — das D2H-Muster ist
vorhanden. Resume-Punkt-Logik (SSM-4) sitzt im Restore-Staging: erst KV-Kette
bestimmen, dann längsten Snapshot ≤ Kettenende wählen, KV dahinter verwerfen.

### 2.5 Swap-Kosten (MOD-5/7/8, MEM-7)

Messpunkte existieren als Telemetrie (Unload-Accounting in
`hotswap_ctl.rs:157-279`, Swap-Dauer in `serve.rs:895-904`), fließen aber
nirgends in Entscheidungen. Neu: `SwapCostStore` (JSON pro Modell-Hash neben
den Snapshots, Muster `snapshots/<id>-<hash>.raw`), EWMA-Update an den
bestehenden Messstellen, Fallback-Schätzung aus Gewichtsgröße ÷ gemessener
Snapshot-Lesebandbreite (wird beim Preseed geloggt). Laufzeit-Reserve: der
Watermark-Alloc-Tracker (`take_tracked_allocs`) liefert die realen
Post-Build-Bytes; Wert gemessen persistieren und in `kv_budget.rs` als
Override vor die Schätzung aus `preflight.rs:116-190` schalten.

Stand 2026-11-10: Der Override ist wieder entfernt. Der gemessene Wert enthält
alles, was der Build vor dem KV-Budget allokiert, und ist damit nicht die Größe,
für die die Reserve steht; als Override machte er den KV-Pool zu groß. Die
Reserve ist immer die Schätzung, der Messwert bleibt als Diagnose in
`swap_costs.json` (siehe `memory-model.md` §3 und §6).

### 2.6 Druck-Scheduler + Swap-Ablauf (SCH-*, PRI-*, DRN-*, Abschnitt 10)

Kern-Designentscheidung: **der GPU-Scheduler-Thread bleibt, wie er ist**
(Continuous Batching, Chunked Prefill, SLAI). Neu davor: ein
Dispatcher (`spark-server/src/arena_sched/`) mit

- `queues.rs` — pro Arena-Modell zwei FIFO-Stufen (`VecDeque<QueuedRequest>`),
  429 bei `max_queue_length`;
- `decision.rs` — reine Funktion `decide(now, stats, costs, cfg) -> Stay |
  Swap{target}` in der verbindlichen Reihenfolge, vollständig unit-getestet;
- `dispatcher.rs` — Tokio-Task: Auswertung alle `scheduler_interval` + bei
  jedem Einreihen; Weiterreichen an den bestehenden mpsc-Kanal **nur solange
  der Scheduler Kapazität hat** (sonst verlöre die priorisierte Stufe ihre
  Ordnungshoheit); Swap-Orchestrierung über die existierenden
  `ControlMsg::Unload/Load` + `SwapState`;
- `costs.rs` — T_swap/λ/EWMA (2.5).

`ensure_model_active` (`api/lazy_swap.rs`) wird für Arena-Kategorien zum
Queue-Submit; `LazyPhase`/Drittmodell-503 und das „model is unloading"-
Abweisen in `hotswap_ctl.rs:74-78` entfallen (DRN-4 — eingereiht statt
abgewiesen). `unload_active` (Sidecar-Eviction) bleibt und wird vom
Dispatcher-Lock mitserialisiert. MOD-10 (Load-Fehler → voriges Modell zurück)
sitzt im Dispatcher: `reload(None)` lädt das Arena-Vormodell.

**Erhalten bleibt:** `X-Atlas-Model-Swap` (jetzt = Wartezeit), Kategorie-400,
404-mit-Modellliste, Admin-Endpoints (laufen über denselben Dispatcher-Lock).

### 2.7 Wartende Clients (WAIT-*)

`api/chat_stream` öffnet den SSE-Stream heute erst nach `ensure_model_active`.
Neu: Stream sofort öffnen (`: queued`), Keep-alive-Kommentare aus dem
Warte-Task, Fehler nach Stream-Start als SSE-`error`-Event (WAIT-5; axum
`Event::comment` existiert). Blocking-Pfad: Wartezeitschätzung aus
Queue-Position × Schrittzeit + Rest-Residency + T_swap; 503 vor dem Einreihen
oder beim Überschreiten.

### 2.8 Embed/Rerank-Sidecars (MOD-3/4)

Größter Einzel-Neubau außerhalb des Stores: Embeddings/Rerank laufen heute
als `max_tokens=1`-Requests **durch den Scheduler** (`logit_probe` /
`capture_hidden`). Ein residenter Sidecar braucht eine zweite
`TransformerModel`-Instanz außerhalb der Arena (Loader direkt, keine
Arena-Migration, Mini-KV-Pool), eigener Stream + Lock, Prefill-Only-Aufruf am
Scheduler vorbei — Muster: Whisper-Sidecar (`api/audio.rs`, `UntrackedAllocs`,
`OnceCell`) + Qwen-Image (`use_private_stream`). `sidecar_reserve` in
`kv_budget.rs:105-126` wird um die Embed/Rerank-Reserve erweitert (MEM-6).

### 2.9 Auth/Priorität (AUTH-*, PRI-Eingang)

Klein: `--trust-proxy-headers`-Flag (`serve_args.rs`), Startabbruch bei
Nicht-Loopback-Bind (`serve_router.rs:178-195` kennt die Bind-Prüfung),
`X-Priority`-Extraktion in den Handlern → `priority`-Feld im Queue-Submit.
Bearer-Middleware deckt `/internal/*` über das bestehende Pfad-Präfix-Muster
(`middleware.rs:147-150`, um `/internal` ergänzen).

### 2.10 Telemetrie (TEL-*, MEM-9)

`/health`-Zustände `draining`/`swapping` aus `SwapState` ableiten
(`misc_handlers.rs:131-156`); `/internal/state` neu (JSON aus Dispatcher-Stats
+ `kv_stats` + Store-Index + Kapazitätsprofil); Kapazitätsprofil MEM-9 aus
`ModelConfig` + per-Layer-Dtypes + gemessener S_ssm. Neue Prometheus-Metriken
nach dem Muster `metrics.rs`.

---

## 3. Priorisierung und Reihenfolge

Begründung: (a) Die Schlüsselkette ist Fundament für Store, SSM und
Restore; (b) der Store ist GPU-frei testbar und entkoppelt; (c) Swap-Kosten
und die reine Entscheidungsfunktion sind die risikoärmsten Bausteine des
Scheduler-Umbaus und vor der Integration testbar; (d) der Dispatcher-Umbau
ist die invasivste Änderung am Serving-Pfad und kommt erst, wenn seine
Zutaten (Kosten, Entscheidungsfunktion, Persist-Hooks) stehen; (e) Sidecars
für Embed/Rerank sind funktional unabhängig und blockieren nichts.

| # | Task | Anforderungen | Status-Risiko |
|---|---|---|---|
| T2 | Block-Schlüsselkette | KV-1..5, KV-8-Prüfung | breite Signaturänderung, höchster Hebel |
| T3 | NVMe-Block-Store | NVM-3..12 | isoliert, GPU-frei testbar |
| T4 | Persist on_release + pre-swap | NVM-1/2/11, KV-7, DRN-3/6 | Scheduler-Lifecycle-Hooks |
| T5 | Restore-Pfad | BAT-6/7, NVM-6/7 | Staging-Design hält Modell-Layer unberührt |
| T6 | SSM über Kettenschlüssel | SSM-1..8 | komplexester Datenpfad; nach KV-Tier |
| T7 | Swap-Kosten + Reserve | MOD-5/7/8, MEM-7 | klein, Messstellen existieren |
| T8 | Queues + Entscheidungsfunktion | SCH-1..5, PRI-1..3, SCH-8 | pur + testbar vor Integration |
| T9 | Dispatcher-/Drain-Umbau | SCH-6/7/9, DRN-1..5, MOD-10, Abschnitt 10 | invasivster Schritt |
| T10 | Keep-alives + Wartegrenzen | WAIT-1..5 | nach T9 (braucht Queues) |
| T11 | Embed/Rerank-Sidecars | MOD-3/4, WAIT-6, MEM-6 | größter Neubau, unabhängig |
| T12 | trust-proxy-headers | AUTH-3/4/5 | klein |
| T13 | Telemetrie | TEL-2/3/4/6, MEM-9, PRI-6 | sammelt alle Stats ein |
| T14 | Konfig + Validierung | §13 | Abschluss |

**Bewusst NICHT in v1** (SOLL laut Dokument): KV-9 (CoW für angebrochene
Blöcke), NVM-13 (write_through), NVM-14 (Spill-Formatvereinheitlichung),
SSM-6, PRI-5 (Aging), TEL-3-Paginierung.

**Bestand, nur nicht zurückbauen** (keine Arbeit außer Verifikation): MEM-1..6,
MEM-8, KV-1/2/6, SSM-1 (RAM-seitig), MOD-1/2/6/9/11, BAT-1/2/8, AUTH-1/2,
TEL-1/5.

**KV-8 (Spekulative Drafts nie versiegeln):** Prüfauftrag — Drafts leben in
`pending_drafts` und werden erst nach Verify in `seq.tokens` übernommen; die
Radix-Inserts laufen über `seq.tokens`/`kv_valid_tokens`. Erwartung: erfüllt;
wird in T2 mit einem Test festgenagelt.

## 4. Risiken

1. **Signatur-Ripple von T2** durch spark-model — mitigiert durch ein
   Parameterobjekt (`BlockKeyChain` in `SequenceState`) statt n neuer
   Einzelparameter.
2. **Dispatcher-Ordnungshoheit vs. Scheduler-Pull:** priorisierte Requests
   dürfen nicht hinter bereits weitergereichten normalen verschwinden →
   Weiterreichen nur bei freier Scheduler-Kapazität (begrenzte In-Flight-Zahl).
3. **io_uring im Dev-Container** (seccomp) — Store läuft dort auf dem
   POSIX-O_DIRECT-Fallback; Bandbreiten-Verifikation nur on-host.
4. **GB10-Treiber-Pool:** Restore-Staging alloziert Host-Puffer; kein
   GPU-Preflight nötig, aber die Schreib-Queue darf `MemAvailable` nicht
   leerziehen → `write_queue_max_bytes` zählt Host-Puffer mit.
5. **Embed/Rerank-Sidecar** nutzt `build_model` außerhalb der Arena — der
   Loader-Pfad kann das (Boot lädt vor der Migration genauso), aber der
   Watermark-Tracker darf Sidecar-Allocs nicht dem Arena-Unload zuschlagen
   (`UntrackedAllocs`-Muster von Whisper übernehmen).

# Fristbasierte Annahme (`accepted_processing_time`), Datalab-kompatibler OCR-Endpunkt und ntfy-Alarmierung

Stand: 2026-10-11 — Planungsstand, noch nichts implementiert.
Grundlage: Briefing „Atlas: Datalab-kompatibler OCR-Endpunkt als bevorzugter
Weg vor Datalab" (OpenNotes-App, Teil 1–3) und die Code-Analyse vom
2026-10-11 auf `modifications`.

## 0. Kurzfassung

- **Teil 1 (Datalab-Schnittstelle) existiert in Atlas nicht.** Es gibt nur ein
  synchrones `POST /v1/ocr` mit JSON-Body (`api/ocr.rs`). Kein `/convert`,
  kein Job-Store, kein Polling, kein `DELETE`, kein `json.children`-Baum,
  kein `error_code`. Die gesamte Oberfläche ist Neubau; die Chandra-Pipeline
  darunter (`ocr/{input,prompts,output,markdown}.rs`) bleibt.
- **Eine Laufzeitprognose gibt es nur für den Modellwechsel** (WAIT-3,
  `decision.rs::wait_estimate_secs`): Rest-Hysterese + T_swap. Keine
  Prefill-/Decode-Rate, keine Queue-Position, kein Kapazitäts-Ledger. Zwei
  gleichzeitige Anfragen können heute beide die Prüfung passieren.
- **Kein Endpunkt sieht alle Anfragen.** `prepare_model_gate`
  (`api/lazy_swap.rs`) kommt am nächsten, wird aber von `/v1/messages`,
  `/v1/responses`, Whisper und Qwen-Image umgangen und ist im
  Einzelmodell-Betrieb inaktiv.
- **Alarmierung nach außen gibt es nicht.** `critical-<ts>.log` sammelt
  WARN+ERROR, der OOM-Watchdog beendet den Prozess, Panics schreiben
  Crash-Snapshots. Kein HTTP-Client im Server (reqwest fehlt; `ureq` nur
  in `atlas-spark-bench`).

Drei PRs (§4): **A** Schätzkern + globaler Parameter, **B** Datalab-Oberfläche
für OCR, **C** Überwachung (ntfy-Notifier + Ereignisspeicher).

**Entschieden (Jonas, 2026-10-11): Reihenfolge B → A → C.** Erst die
vollständig Datalab-kompatible OCR-API, dann die Schätzung mit
`accepted_processing_time`, dann die Dauerüberwachung. Die Rückmeldung der
App-Session ist in §2b eingearbeitet.

## 1. Befund

### 1.1 OCR heute (`crates/spark-server/src/api/ocr.rs`, `src/ocr/`)

| Aspekt | Ist | Teil-1-Soll |
|---|---|---|
| Transport | JSON `{image: base64\|data-URI}`, Body ≤ 32 MB (`ATLAS_MAX_BODY_BYTES`), PDF ≈ 24 MB netto | `multipart/form-data`, Feld `file` |
| Ablauf | synchron, alle Seiten in einem HTTP-Request, Seiten **sequenziell** | asynchron: `request_id` + Polling bis 900 s |
| Status/Abbruch | – | `pending/processing/complete/failed`, `DELETE` |
| Ergebnis | `pages[{page (1-basiert), html, markdown, chunks[{label,bbox,content}], images, retries}]`, `usage` | `json.children` (Page-Knoten, 0-basiert), `polygon`, `section_hierarchy`, `markdown`, `images`, `metadata.failed_pages`, `runtime`, `versions`, … |
| Blocknamen | Chandra-Labels (`Section-Header`, `Page-Header`, `Equation-Block`, `List-Group`, `Image`, …) | Marker-Namen (`SectionHeader`, `PageHeader`, `Equation`, `ListGroup`, `Picture`, …) |
| Bilder | WebP, Name `md5(html)_<idx>_img.webp` | Base64 am Block **und** im Dokument, Name aus dem Markdown (`_page_0_Picture_6.jpeg`) |
| Fehlercodes | `code: null` bei allen OCR-Fehlern; nur das Swap-Gate liefert `model_not_found`/`model_wrong_category`/`model_wait_timeout` | `error_code` an jeder Fehlerantwort, `unprocessable_document` ≠ Atlas-Problem |
| Seitenfehler | erster Fehler bricht den ganzen Request ab | `metadata.failed_pages` |
| Timeout | `--request-timeout` (300 s) pro Seite; Überschreitung liefert **abgeschnittenen Text als Erfolg** (`decode_logits_step.rs:706`) | Frist pro Auftrag, Überschreitung = `failed` |
| Rendering | pdfium unter prozessweitem `PDFIUM_LOCK`; **alle Seiten vorab** in den Speicher | – |

Weitere Fakten: Chandra läuft in-process durch den Haupt-Scheduler
(`ensure_model_active(ModelCategory::Ocr)`), kein Sidecar; Kontext 16 384,
`max_tokens` 12 384/Seite, bis zu 6 Temperatur-Retries bei Degeneration.
On-Box-Messungen (`Todo.md` 3.3d): bench 16 s, multilingual 15 s,
handwritten_form 290 s (1 Retry). `PDF render error` meldet auch ein
fehlendes `libpdfium.so` als 400 (`input.rs:157`) — falsche Zuordnung.

Lizenz (Todo.md 3.3): Chandra-OCR-2 steht unter modifizierter OpenRAIL-M,
„nicht kompetitiv zur Datalab-API einsetzbar". Der geplante Einsatz ersetzt
Datalab-Aufrufe der eigenen App. Vor Produktivbetrieb klären.

### 1.2 Annahme/Schätzung heute

- `prepare_model_gate` (`lazy_swap.rs:74–255`): Kategorie-/Modellprüfung,
  Fast-Path bei residentem Modell, WAIT-3-Schätzung nur für
  nicht-streamende Anfragen (`estimate_wait_secs` vs. `--nonstream-max-wait-s`,
  503 `model_wait_too_long`), Ticket in `ArenaQueues`.
- Die Schätzung ignoriert Queue-Position, laufende Decodes, laufende Swaps
  und das Persist-Volumen. Sie läuft **außerhalb** des Queue-Mutex.
- Vorhandene Zahlen: `swap_costs.json` (T_load, T_warmup, T_unload,
  T_persist/GiB, EWMA via `--ewma-alpha`), `QueueSnapshot` (Längen, λ,
  Pressure), `scheduler_load()` (aktiv + prefilling), KV-Pool-Belegung,
  `capacity_profile`.
- Fehlend: Prefill-Rate, Decode-Schrittzeit je Batchgröße, Restlänge
  laufender Sequenzen außerhalb des Scheduler-Threads, Token-Länge vor dem
  Swap (Tokenizer gehört dem Zielmodell), Sidecar-Ladezeiten.
- `timeout_at` zählt nur Decode-Zeit. Blockierende Anfragen
  (`/v1/messages`, `/v1/responses` non-stream, Embeddings, Rerank, OCR)
  werden bei Client-Abbruch **nicht** abgebrochen; Sequenz läuft bis
  EOS/`max_tokens`/`timeout_at` und hält Batch-Slot und KV.
- Kein Kapazitäts-Ledger für angenommene, noch nicht gestartete Arbeit.

### 1.3 Anomalie-Erkennung heute

Vorhanden: dreischichtiges Logging (`main.rs::init_logging`), `critical-<ts>.log`
= WARN+, Crash-Snapshots (`crash.rs`) aus Panic-Hook und OOM-Watchdog
(`cuda_backend.rs:542`, `exit(1)` nach 3 Messungen), Sticky-CUDA-Fehler →
`/health` 503, Speicher-Heartbeat (`heartbeat.rs`, WARN unter 8 GB),
KV-Leak-Prüfungen nur mit `ATLAS_MEM_DEBUG=1` (`memdebug.rs`), kv-tier- und
Write-Queue-Fehler als WARN, Prometheus-Metriken (Requests, TTFT, tok/s,
Swaps, Queues, NVMe, Watchdog-Stops).

Fehlend: Ausgangskanal (ntfy/Webhook), NaN-/Leerausgabe-Erkennung,
Fehlerzähler (5xx-Rate), Prozessende-Erkennung (serve.sh hat weder Trap
noch Supervisor), Erkennung „Schätzung verfehlt", Korrelation
App-Fallback ↔ Server-Log.

## 2. Antworten auf Teil 3

### Frage 1 — Prognose für `accepted_processing_time`

Deterministisches Kostenmodell über eine **Momentaufnahme** des Servers,
berechnet als reine Funktion (kein Lernen zur Laufzeit):

```
ETA = T_vor + T_modell + T_arbeit

T_vor    = Σ reservierte Arbeit vor dieser Anfrage (Ledger, §Frage 4)
           + laufende Swap-Restzeit
T_modell = 0, wenn Zielmodell resident
         = Rest-Hysterese(residentes Modell) + T_swap(Ziel)   (WAIT-3, vorhanden)
         + Sidecar-Belegung (Whisper/Qwen-Image halten die GPU exklusiv)
T_arbeit = prompt_tokens / prefill_tok_s(modell)
         + erwartete_output_tokens × schrittzeit(modell, batch_nach_Annahme)
```

Modellkonstanten (`prefill_tok_s`, `schrittzeit` bei Batch 1/2/4/8,
OCR-`tokens_pro_seite_p90`) stehen im `MODEL.toml` des Kernel-Targets in
einem neuen Abschnitt `[estimate]` — SSOT, einmal on-box kalibriert
(`tests/estimate_calibration.py`, PR A). Ohne Abschnitt: kein Versprechen,
`accepted_processing_time` wird mit 503 `unavailable` + Begründung
abgelehnt (PCND: kein stiller Default).

Erwartete Output-Tokens: Chat/Completions nehmen `max_tokens` (Obergrenze,
also konservativ); OCR nimmt `seiten × tokens_pro_seite_p90`;
Embeddings/Rerank/Translate haben feste Profile. Prompt-Länge bei
nicht-residentem Modell: Byte-Schätzung (Bytes / 3,5, konservativ), bei
residentem Modell echte Tokenisierung.

Geteilte Hardware ist im Modell abgebildet: der Swap-Term kommt aus
`swap_costs.json`, der Ledger zählt angenommene Arbeit **aller**
Kategorien, die GPU-exklusiven Sidecars erscheinen als Blockzeit. Für den
80B-Thinker als residentem Modell kann die Rest-Hysterese allein
60–120 s betragen — genau das soll die Schätzung sichtbar machen.

Entscheidung: `ETA ≤ accepted_processing_time` ⇒ annehmen, sonst 503
`capacity_exceeded` mit `estimated_seconds = ceil(ETA)`. Zusätzlich wird die
Frist zur Laufzeit durchgesetzt: läuft sie ab, endet der Auftrag als
`failed` mit `error_code: deadline_exceeded` (Atlas-seitig ⇒ App weicht
aus). Ohne Laufzeitdurchsetzung wäre die Schätzung ein unverbindliches
Versprechen; mit ihr hat die App ein determiniertes Ergebnis.

### Frage 2 — Entscheidungszeit beim Einreichen

Die Entscheidung ist reine Rechnung über In-Memory-Zustand unter einem
Mutex: unter 1 ms. Für OCR kommt die Seitenzahl hinzu (pdfium-Dokument
öffnen, nicht rendern): typischerweise 10–100 ms, bei großen PDFs bis ~1 s.
Voraussetzung: das Rendering wird von „alle Seiten vorab" auf „pro Seite
unter kurzem Lock" umgestellt (PR B), sonst wartet das Einreichen hinter
einem laufenden 50-Seiten-Render.

Empfehlung für die App: **Einreichen 15 s** Timeout ohne Upload-Zeit (Upload
hängt an Dateigröße und Leitung). Antwortet Atlas nicht in 15 s, ist etwas
kaputt — ausweichen.

### Frage 3 — Determinismus ohne Vergangenheitswerte

Ja. Alle Eingaben der Prognose sind entweder Konstanten aus `MODEL.toml`
(einmalig kalibriert, versioniert) oder ein Zustands-Snapshot (Queue,
Ledger, Batch, residentes Modell, Swap-Status). Die Funktion
`estimate(snapshot, profil, konstanten) → Estimate` ist pur und
nebenwirkungsfrei. Die vorhandene `swap_costs.json`-EWMA ist der einzige
adaptive Anteil und bleibt es (sie existiert schon, ist pro Modell
beschränkt und wird in `/internal/state` angezeigt).

Testbarkeit:
- **Unit:** Replay-Vektoren wie in `decision.rs::wait_tests` — Snapshot +
  Profil → erwartete ETA, inklusive Grenzfälle (leere Arena, laufender Swap,
  Sidecar belegt, kein `[estimate]`-Abschnitt).
- **Eigenschaften:** monoton in Seitenzahl, Queue-Tiefe, Prompt-Länge;
  ETA(resident) ≤ ETA(nicht resident).
- **Integration (on-box):** `tests/test_deadline_admission.py` — N Anfragen
  mit knapper Frist, erwartete Verteilung angenommen/abgelehnt, gemessene
  Ist-Zeit ≤ Schätzung × 1,25 bei p90.
- **Kalibrierung:** `tests/estimate_calibration.py` misst Prefill-Rate,
  Schrittzeit je Batchgröße, OCR-Tokens/Seite auf dem Sample-Satz und
  druckt den `[estimate]`-Block; die Zahlen werden committet, nicht zur
  Laufzeit gelernt.

### Frage 4 — Keine Doppelannahme bei einem freien Platz

Ein **Kapazitäts-Ledger** (`arena_sched/ledger.rs`, ein `Mutex`) hält jede
angenommene, noch nicht beendete Anfrage mit Modell, erwarteter Arbeit und
Frist. Schätzen und Reservieren geschehen **atomar unter demselben Lock**;
die Schätzung der zweiten Anfrage sieht die Reservierung der ersten.
Freigabe über einen Drop-Guard (`Reservation`), der bei Erfolg, Fehler,
Client-Abbruch, Ticket-Verfall und `DELETE` greift. Der heutige
WAIT-3-Check zieht in dieselbe Sperre; der Fast-Path für residente Modelle
nimmt sie ebenfalls.

Der GPU-Scheduler bleibt unverändert; der Ledger liegt davor. Nebeneffekt:
der Ledger ist die erste Stelle, die „angenommene Arbeit" überhaupt kennt,
und liefert `/internal/state` eine ETA-Sicht.

Nötige Begleitkorrektur: blockierende Anfragen müssen bei Client-Abbruch
beendet werden (Prüfung `response_tx.is_closed()` im Scheduler neben dem
vorhandenen `TrySendError::Closed`-Pfad), sonst hält ein abgebrochener
Auftrag seine Reservierung und seinen Slot bis `timeout_at`.

### Frage 5 — Was das OCR-Modell nicht oder eingeschränkt liefert

| Teil-1-Element | Lieferbar? | Umsetzung |
|---|---|---|
| Seitenknoten, 0-basierte IDs, Seiten-`bbox` | ja | Rendergröße in Pixeln als Seiten-`bbox`; Blöcke in denselben Pixeln |
| `polygon` | **nur Rechteck** | 4 Ecken der `bbox`; keine Rotation, keine Schräglage |
| Blocktypen | ja, per Mapping | `Section-Header→SectionHeader`, `Page-Header→PageHeader`, `Page-Footer→PageFooter`, `Equation-Block→Equation`, `List-Group→ListGroup`, `Image→Picture`, `Figure→Figure`, `Code-Block→Code`, `Table-Of-Contents→TableOfContents`, `Complex-Block→ComplexRegion`, `Chemical-Block/Diagram→Figure`, `Bibliography→Text`, Rest 1:1 |
| Gruppen mit `children` | **nein** | Chandra liefert eine flache Blockliste pro Seite; alle Blöcke sind Blätter. Die App steigt ohnehin bis zu den Blättern ab |
| `section_hierarchy` | ja, abgeleitet | aus Reihenfolge und `<h1..h6>`-Stufe der SectionHeader |
| `html`/`markdown` je Block | ja | `markdown` nie `null` (Konverter liefert immer String) |
| Bilder am Block und im Dokument | ja | Crops nach JPEG (statt WebP), Name `_page_{n}_Picture_{i}.jpeg`, im Markdown referenziert |
| ID-Stabilität bei Wiederholung | **ja, über Ergebnis-Cache** (§2b.1) | Dieselbe Datei liefert byte-identische Ergebnisse aus dem inhaltsadressierten Cache; Retries sind geseedet. Ohne Cache-Treffer gilt: Greedy-Pfad reproduzierbar bis auf GPU-Rundung, was kein Batch-Server garantieren kann |
| `metadata.failed_pages` | ja, nach Umbau | Seitenfehler werden isoliert statt den Auftrag abzubrechen |
| `runtime`, `versions` | ja | gemessen; `versions = {atlas, model}` |
| `parse_quality_score`, `total_cost`, `checkpoint_id` | `null` | zulässig laut Teil 1 |
| `mode`, `skip_cache`, `save_checkpoint`, `extras`, `include_markdown_in_chunks`, `output_format` | angenommen, ignoriert | ein Modell, ein Format |
| `max_pages` | ja | begrenzt Seiten **und** geht in die Schätzung ein |
| Handschrift/Formulare | eingeschränkt | 290 s für eine Seite im Smoke-Test; die p90-Konstante deckt das nicht, die Laufzeit-Frist fängt es ab (`deadline_exceeded`) |

Zusätzlicher Hebel: Seiten eines Auftrags **parallel** in den Batch geben
(`--ocr-page-parallelism`, ≤ `--max-batch-size`). Das 4B-Modell ist
speichergebunden; 8 Seiten im Batch kosten kaum mehr Wandzeit als eine.
Ein 20-Seiten-PDF fällt damit von ~5 min auf ~1 min — direkt relevant für
die Frist.

### Frage 6 — Aufträge, die niemand abfragt oder abbricht

Job-Store im RAM mit drei Grenzen (alle als Serve-Flags, dokumentiert in
`Settings.md`):

- `--ocr-poll-idle-s` (Vorschlag 120): ein laufender Auftrag ohne `GET`
  seit N Sekunden gilt als verwaist, wird abgebrochen (gibt Ledger und
  Batch-Slot frei) und endet als `failed`, `error_code: abandoned`.
- `--ocr-result-ttl-s` (Vorschlag 900 = Polling-Maximum der App): fertige
  Ergebnisse werden danach gelöscht; `GET` liefert 404,
  `error_code: unknown_request`.
- `--ocr-store-max-bytes`: Obergrenze für gespeicherte Ergebnisse; voll ⇒
  neue Einreichungen 503 `capacity_exceeded`.

GC-Task alle 10 s. Nach Serverneustart ist der Store leer: `GET` auf alte
IDs liefert 404 `unknown_request` ⇒ die App weicht aus. `DELETE` ist
idempotent (unbekannt oder schon fertig ⇒ 204).

## 2b. Rückmeldung der App-Session (2026-10-11) und Auflösung

### 2b.1 Determinismus der Block-IDs — immer, nicht nur im Greedy-Pfad

Die App hängt Quellenverweise (`source_id`) und Abbildungsadressen
(`Block::figureKey()`) an die IDs. Ein erneuter Lauf derselben Datei muss
dieselben IDs liefern.

Ehrliche Grenze: Kein Batch-Inferenzserver garantiert bitgleiche Logits über
verschiedene Batch-Zusammensetzungen. Selbst Greedy kann bei Beinahe-Gleichstand
kippen, und ein Block mehr oder weniger verschiebt alle Folge-Indizes auf der
Seite. Datalab selbst löst das mit einem Datei-Cache (`skip_cache`), nicht
mit GPU-Determinismus. Atlas macht es genauso:

1. **Inhaltsadressierter Ergebnis-Cache** (`--ocr-cache-dir`, auf NVMe).
   Schlüssel = BLAKE3(PDF-Bytes) + Modell-Revision + `prompt_type` +
   Seitenauswahl. Ein Treffer liefert das gespeicherte Ergebnis
   byte-identisch, ohne Inferenz. `skip_cache=true` umgeht den Cache (und
   überschreibt den Eintrag). Ablauf über `--ocr-cache-ttl-days`
   (Vorschlag 90) und Bytes-Obergrenze, LRU. Das ist der Garant für
   ID-Stabilität zwischen Läufen.
2. **Seitenweiser Cache-Schlüssel**: BLAKE3(gerenderte Seite als PNG) +
   Modell-Revision. Ein Dokument, dessen Seiten unverändert sind, trifft
   seitenweise, auch wenn andere Seiten geändert wurden.
3. **Geseedete Retries**: Degenerations-Retries bekommen
   `seed = BLAKE3(seitenschlüssel, versuch)` statt ungeseedeter Sampling-
   Zufälle (`InferenceRequest.seed` existiert, `ocr.rs` setzt heute `None`).
4. **Greedy bleibt Pass 0** (temperature 0, wie heute).

Konsequenz für die App: dieselbe Datei ⇒ dieselben IDs, solange der
Cache-Eintrag lebt. Nach Ablauf oder `skip_cache=true` ist das Ergebnis mit
hoher Wahrscheinlichkeit identisch, aber nicht garantiert. Wer absolute
Stabilität braucht, verlässt sich auf den Cache und nicht auf einen
Neu-Lauf. Datalab bietet exakt dieselbe Zusage.

### 2b.2 Abgeschnittener Text zählt als Fehler

Für `/convert` verbindlich: Eine Seite, deren Generierung durch
`max_tokens`, `timeout_at` oder die Auftragsfrist abgebrochen wurde, gilt als
fehlgeschlagen und steht in `metadata.failed_pages`. Ihr Inhalt wird **nicht**
in `markdown` oder `json.children` aufgenommen. Zusätzlich endet der Auftrag
als `failed` mit `error_code: incomplete_document`, wenn mehr als
`--ocr-max-failed-pages-pct` (Vorschlag 0, also jede fehlgeschlagene Seite
⇒ `failed`) der Seiten fehlen. Mit 0 % gibt es nie ein `complete` mit
Lücken; die App entscheidet dann über den Fallback. Der Grund je Seite steht
in `metadata.page_errors[{page, error_code}]` (Atlas-Erweiterung, von der App
ignorierbar).

Dasselbe gilt für Degenerations-Retries, die nach `max_retries` immer noch
degenerieren: `failed_pages` + `page_errors.error_code = degenerate_output`.

### 2b.3 Kontraktgarantien (zusammengefasst, verbindlich für PR B)

Die App verlässt sich auf diese Punkte; jeder ist Teil des Kontrakt-Tests
`tests/test_convert_contract.py`:

- `json.children` sind Seitenknoten `block_type: "Page"`, ID
  `/page/{n}/Page/{n}`, **n ab 0** in Dokumentreihenfolge der verarbeiteten
  Seiten.
- Jeder Seitenknoten hat `bbox = [0, 0, breite_px, hoehe_px]` der gerenderten
  Seite. Alle Block-`bbox`/`polygon` liegen im selben Pixelraum, nicht
  normiert.
- Block-IDs `/page/{n}/{BlockType}/{i}` mit `i` als seitenweitem Laufindex in
  Lesereihenfolge (wie im Datalab-Beispiel: `SectionHeader/0`, `Text/1`).
- `polygon` = vier Ecken der `bbox` im Uhrzeigersinn ab oben links.
- `block_type` aus der Marker-Namensmenge (Mapping-Tabelle in Frage 5).
- `section_hierarchy` an jedem Block: `{"<level>": "<id des SectionHeader>"}`
  für alle zum Zeitpunkt des Blocks offenen Überschriftenstufen; Stufe aus
  `<h1..h6>` im Block-HTML, Standard 1. Vor der ersten Überschrift `{}`.
- `html` und `markdown` an jedem Block, `markdown` immer String, nie `null`.
- `images` am Block: `{"<block-id>": "<base64 jpeg>"}`; im Dokument unter
  `_page_{n}_{BlockType}_{i}.jpeg`, exakt der Name, der im Markdown
  referenziert wird.
- `metadata.failed_pages` 0-basiert; `runtime` in Sekunden; `versions =
  {"atlas": "<version>", "model": "<hf-id@revision>"}`.
- `parse_quality_score`, `total_cost`, `checkpoint_id` sind `null`.
- `error_code` an **jeder** Fehlerantwort. `unprocessable_document` nur, wenn
  das Dokument selbst die Ursache ist: kein PDF, verschlüsselt, pdfium kann es
  nicht öffnen, 0 Seiten in der Auswahl, Seitenlimit des Servers
  überschritten. Alles andere (`pdfium_unavailable`, `model_unavailable`,
  `capacity_exceeded`, `deadline_exceeded`, `incomplete_document`,
  `abandoned`, `internal_error`, `unknown_request`, `unavailable`) ist
  Atlas-seitig.
- Statuswerte exakt `pending`, `processing`, `complete`, `failed`.

## 3. Globaler Parameter — Oberfläche

| Endpunkt | Parameter | Antwort bei Annahme |
|---|---|---|
| JSON-APIs (`/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/v1/responses`, `/v1/embeddings`, `/v1/rerank`, `/v1/translate`, `/v1/ocr`) | Body-Feld `accepted_processing_time` (Sekunden, float) | Header `X-Atlas-Estimated-Seconds` |
| Multipart/Binär (`/convert`, `/v1/audio/transcriptions`, `/v1/images/*`) | Formfeld `accepted_processing_time` **oder** Header `X-Atlas-Accepted-Processing-Time` | `/convert`: Body-Feld `estimated_seconds`; sonst Header |
| Ablehnung | 503, `{"success": false, "error", "error_code": "capacity_exceeded", "estimated_seconds"}` auf `/convert`; OpenAI-Fehlerform mit `code: "capacity_exceeded"` + `estimated_seconds` auf den übrigen | `Retry-After` gesetzt |
| Nicht bereit | 503 `error_code: "unavailable"` (CUDA poisoned, Modell nicht registriert, kein `[estimate]`, Store voll) | – |

Ohne Parameter: Verhalten wie heute (keine Ablehnung, aber Schätzung im
Header, wenn verfügbar). Protokolldrift OpenAI↔Anthropic wird durch das
gemeinsame Gate vermieden (beide Pfade rufen dieselbe Funktion).

## 4. Umsetzungsplan

### PR A — Schätzkern und globaler Parameter (`spark-server`)

1. `arena_sched/estimate.rs`: `CostProfile` (aus `MODEL.toml [estimate]`,
   geparst in `atlas-kernels/build.rs` neben `ModelBehavior`), `WorkProfile`
   (prompt_tokens, expected_output_tokens, pages), reine `estimate()`;
   Replay-Tests.
2. `arena_sched/ledger.rs`: `CapacityLedger` + `Reservation`-Guard;
   Schätzen+Reservieren atomar; Freigabe-Pfade; Sicht in `/internal/state`.
3. Gate-Vereinheitlichung: `prepare_model_gate` auch für `/v1/messages`,
   `/v1/responses`, Einzelmodell-Betrieb; Haken für Whisper/Qwen-Image.
   WAIT-3 in die Ledger-Sperre ziehen.
4. Parameter in allen Request-Typen + Header-Variante; 503-Formen;
   `X-Atlas-Estimated-Seconds`.
5. Laufzeit-Frist: `deadline_at` auf `InferenceRequest` (ergänzt
   `timeout_at`, zählt ab Einreichen); Überschreitung ⇒ Fehler, nicht
   abgeschnittener Erfolg.
6. Korrektur: blockierende Anfragen bei Client-Abbruch beenden.
7. `tests/estimate_calibration.py` + `[estimate]`-Blöcke für die vier
   generativen Preseed-Modelle und Chandra; `tests/test_deadline_admission.py`.

### PR B — Datalab-Oberfläche für OCR

1. `api/convert/` (Modulordner, Dateien ≤ 500 LoC): `submit.rs` (multipart,
   Feldaufnahme, Seitenzahl, Schätzung, Job anlegen), `status.rs` (`GET`),
   `cancel.rs` (`DELETE`), `job_store.rs` (RAM, TTL, Idle, Bytes-Cap, GC),
   `result_tree.rs` (Chandra-Chunks → `json.children`, Mapping-Tabelle,
   `section_hierarchy`, Bilder als JPEG, `metadata.failed_pages`).
2. `ocr/`-Umbau: Rendering pro Seite unter kurzem Lock; Seitenfehler
   isolieren; Abschneiden (`max_tokens`, Timeout, Frist) ⇒ `failed_pages`
   (§2b.2); Seiten-Fan-out mit `--ocr-page-parallelism`; `max_pages`;
   geseedete Retries.
3. `error_code` überall; `PDF render error` aufteilen in
   `unprocessable_document` (Dokument) vs. `pdfium_unavailable` (Server).
3b. `ocr/result_cache.rs`: inhaltsadressierter Ergebnis-Cache auf NVMe
   (§2b.1), Datei- und Seitenschlüssel, `skip_cache`, TTL + LRU, über die
   `StorageBackend`-Abstraktion (SBIO).
4. Routen `/convert`, `/convert/{id}` (`GET`, `DELETE`); CORS um `DELETE`
   erweitern; Multipart-Limit aus `ATLAS_MAX_BODY_BYTES`.
5. Tests: Kontrakt-Test mit dem Teil-1-Beispiel als Fixture
   (`tests/test_convert_contract.py`), Fehlerpfade (kaputtes PDF, Abbruch,
   verwaist, Frist), bestehender `ocr_smoke.py` bleibt grün.
6. `/v1/ocr` bleibt unverändert bestehen.

### PR C — ntfy-Notifier (SBIO)

1. `notify/mod.rs`: Trait `Notifier` (`notify(event)`), Implementierungen
   `NtfyNotifier` (ureq 3 mit rustls, eigener Thread, begrenzter Kanal,
   `try_send`, Dedupe/Rate-Limit je Schlüssel) und `NoopNotifier`.
   Flags `--ntfy-url` (Default `https://ntfy.sh`), `--ntfy-topic` (ohne
   Angabe: Noop), `--alert-min-interval-s`.
2. Tracing-Layer in `init_logging`: ERROR aller Targets + WARN-Allowlist
   (`kv_tier`, Block-Store-Writer, `mem`, OOM-Watchdog).
3. Panic-Hook und OOM-Watchdog: synchroner Best-Effort-Send vor `exit`.
4. Neue Signale: Zähler `atlas_request_failures_total{endpoint,code}` +
   5xx-Rate-Schwelle; `estimate_miss` (Ist > Schätzung × 1,5);
   `deadline_exceeded`-Rate; NaN-Wächter im Sampling (alle Logits NaN ⇒
   Fehler + Event); leere Ausgabe bei `finish_reason=stop` nach 0 Tokens;
   Watchdog-Stops pro Minute über Schwelle.
5. Prozessende: `serve.sh` ohne `exec`, `PIPESTATUS[0]` auswerten, bei
   Exit ≠ 0 `curl` an ntfy. Keine Host-Probe (Entscheidung §6).
5b. Ereignisspeicher SQLite (§7) — entschieden 2026-10-11.
6. Korrelation: `X-Atlas-Request-Id` in jede Antwort und jede Logzeile des
   Auftrags, damit ein App-seitiger Fallback im Server-Log auffindbar ist.

Topic für die erste Phase: `jonas-dgx-spark001` auf ntfy.sh (öffentlich;
Nachrichten enthalten keine Prompts, nur Art, Modell, Zahlen, Request-Id).

## 5. Entscheidungen

Entschieden 2026-10-11: Reihenfolge **B → A → C**; `/convert` vollständig
Datalab-kompatibel; IDs immer stabil (über Cache, §2b.1); abgeschnittene
Seiten sind Fehler (§2b.2).

Noch offen:

1. Lizenz Chandra-OCR-2 (OpenRAIL-M, Datalab-Klausel) vor Produktivbetrieb.
2. Vorschlagswerte: `--ocr-poll-idle-s 120`, `--ocr-result-ttl-s 900`,
   `--ocr-page-parallelism 4`, `--ocr-cache-ttl-days 90`,
   `--ocr-max-failed-pages-pct 0`, Einreich-Timeout der App 15 s.
3. `/v1/ocr` bekommt `error_code` und Seitenisolation mit (empfohlen).
4. ~~Ereignisspeicher als SQLite~~ — entschieden: ja, Teil von PR C.

## 6. Instabilitäts-Katalog für die Überwachung (PR C)

Jede Zeile ist ein Signal, das ntfy auslösen soll. Spalte „Quelle" nennt,
woher das Signal kommt: **L** = bestehende Logzeile (Tracing-Layer fängt sie),
**M** = bestehende Metrik (Schwellwert-Prüfer), **N** = neu zu bauen,
**H** = nur von außen erkennbar. Entschieden (Jonas, 2026-10-11): **keine
Host-Probe.** Ein toter Prozess fällt dadurch auf, dass die Apps nur noch
den Fallback nutzen; die H-Zeilen bleiben zur Vollständigkeit stehen, werden
aber nicht gebaut. Priorität nach ntfy-Skala 1–5.

### 6.1 GPU und Speicher

| Signal | Quelle | Prio |
|---|---|---|
| CUDA-OOM beim Laden, Prefill oder Decode (GB10 teilt Host- und GPU-Speicher) | L (`cuda_err`) | 5 |
| OOM-Watchdog: 3 niedrige Messungen ⇒ `exit(1)` | L + sync Send vor `exit` | 5 |
| Sticky-CUDA-Fehler (illegal address, launch failure) ⇒ Kontext vergiftet, `/health` 503 | L (`cuda_err`) | 5 |
| NaN/Inf in Logits (FP8-/NVFP4-Pfade, BF16-Cache in FP8-Kernel) | N: Wächter im Sampling, zählt und bricht Sequenz ab | 4 |
| KV-Pool erschöpft trotz Eviction (`KV cache exhausted`) | L | 4 |
| KV-Leak: Blöcke nach `free_sequence` gehalten / im Idle gehalten | L (`memdebug`, nur mit `ATLAS_MEM_DEBUG=1`) ⇒ N: Idle-Prüfung immer aktiv | 4 |
| Weights-Arena: Load scheitert, obwohl Größe passt (Fragmentierung) | L (hotswap) | 4 |
| Host-`available` unter Schwelle, Swap-Frei unter Schwelle | L (`mem` WARN) | 3 |
| Treiber-Pool hält Speicher beendeter Prozesse (Load dauert, Preflight scheitert) | N: Load-Dauer > 2 × `swap_costs`-Erwartung | 3 |
| GPU-Temperatur / Takt-Drosselung | H (`nvidia-smi` in Host-Probe) | 3 |

### 6.2 Scheduler und Arena

| Signal | Quelle | Prio |
|---|---|---|
| Stillstand: aktive Sequenzen, aber kein Decode-Schritt seit N s | N: Fortschritts-Heartbeat des Scheduler-Threads | 5 |
| Dispatcher hängt in `swapping` länger als 3 × T_swap | N | 5 |
| Modell-Load oder -Unload fehlgeschlagen | L (`Hot-swap: load failed`) | 5 |
| Drain-Grace überschritten, Sequenzen zwangsbeendet | L | 3 |
| Swap-Sturm: mehr als N Swaps pro 10 min (Hysterese versagt, Image-Sidecar evictet Arena) | M (`atlas_model_swaps_total`) | 3 |
| Scheduler-Queue voll (503 `Scheduler queue full`) | N: Fehlerzähler | 3 |
| Ticket-Verfall in Serie (PRI-4 / WAIT-4) | N: Zähler | 3 |
| Watchdog-Stops / Rollbacks über Schwelle pro Minute | M (`atlas_watchdog_stops_total`) | 2 |
| SSM-Snapshot-Restore fehlgeschlagen | L (`rollback.rs:271`) | 4 |

### 6.3 Schätzung und Frist (PR A)

| Signal | Quelle | Prio |
|---|---|---|
| **Alles abgelehnt**: Ablehnungsrate `capacity_exceeded` 100 % über 60 s, während GPU-Last < 20 % und Ledger leer ⇒ Schätzung defekt | N | 5 |
| Schätzung verfehlt: Ist-Dauer > Schätzung × 1,5 bei mehr als 10 % der Aufträge in 10 min | N | 3 |
| `deadline_exceeded`-Rate über Schwelle | N | 3 |
| Modell ohne `[estimate]`-Block angefragt mit Frist (`unavailable`) | N | 2 |
| Ledger-Inkonsistenz: Reservierungen ohne lebende Anfrage (Guard-Leck) | N: Abgleich mit `scheduler_load` | 4 |

### 6.4 NVMe-Tier

| Signal | Quelle | Prio |
|---|---|---|
| Write-Queue voll, Blöcke verworfen | L (`writer.rs`), N: Zähler als Metrik | 3 |
| Export/Import-Fehler, `persist_timeout` überschritten | L (`kv_tier`) | 3 |
| Dateisystem voll / I/O-Fehler im Store | L | 4 |
| Checksummen-Fehler beim Restore | N (falls nicht vorhanden) | 4 |
| Restore-Miss-Rate steigt (Eviction-Ordnung defekt, wie am 2026-10-07) | M (`atlas_nvme_restored_tokens_total` vs. Erwartung) | 2 |

### 6.5 Generierungsqualität

| Signal | Quelle | Prio |
|---|---|---|
| Leere Ausgabe (`finish_reason=stop` nach 0 Tokens) über Schwelle | N | 3 |
| Abbruch bei `max_tokens` über Schwelle (Loop ohne Watchdog) | N | 2 |
| Tool-Call-Parsefehler über Schwelle (Template-Drift) | N | 3 |
| OCR: Degenerations-Retries erschöpft, `degenerate_output` | N | 3 |
| Spec-Decode-Verify-Rate bricht ein | M | 2 |

### 6.6 HTTP und API

| Signal | Quelle | Prio |
|---|---|---|
| 5xx-Rate über Schwelle (je Endpunkt) | N: `atlas_request_failures_total{endpoint,code}` | 4 |
| Handler-Panic (CatchPanicLayer) | N: Panic-Hook + Zähler | 5 |
| TTFT-p95 oder Queue-Wait-p95 über SLO | M | 2 |
| Blockierende Anfrage läuft nach Client-Abbruch weiter (Leck) | N: Prüfung `response_tx.is_closed()` + Zähler | 3 |
| Body-Limit-Ablehnungen in Serie (App schickt zu große PDFs) | N | 2 |
| Auth-Fehler in Serie (Token rotiert, App nicht) | N | 3 |

### 6.7 OCR-Pfad

| Signal | Quelle | Prio |
|---|---|---|
| pdfium-Absturz (native Bibliothek ⇒ Prozess-Abort, kein Panic) | H (Prozessende) + N: Render in eigenem Thread mit Timeout | 5 |
| `libpdfium.so` fehlt (`pdfium_unavailable`) | N | 5 |
| `PDFIUM_LOCK` länger als N s gehalten | N | 3 |
| Job-Store voll / Cache-Verzeichnis nicht beschreibbar | N | 4 |
| Verwaiste Aufträge über Schwelle (App pollt nicht) | N | 2 |
| Seitenzeit über p99 der Kalibrierung | N | 2 |

### 6.8 Prozess und Host

| Signal | Quelle | Prio |
|---|---|---|
| Panic / Abort / Exit ≠ 0 | Panic-Hook (sync Send) + `serve.sh`-Trap (`PIPESTATUS[0]`) | 5 |
| Server antwortet nicht auf `/health` (hart beendet, hängt, Port weg) | H (`scripts/atlas_watch.sh`, Cron 60 s) | 5 |
| Neustart-Schleife (mehr als 2 Starts in 10 min) | H | 4 |
| Log-Platte voll, Logdatei nicht schreibbar | N | 4 |
| HF-Cache: Gewichte fehlen oder Snapshot-Hash abweichend | L (Load-Fehler) | 4 |
| Sidecar (Whisper, Qwen-Image, Text) Load fehlgeschlagen | L | 4 |
| Sidecar-Mutex länger als N s gehalten (Serialisierung staut) | N | 3 |

Alle Signale dedupliziert je Schlüssel (Art + Modell) mit
`--alert-min-interval-s`, maximal N Meldungen je Schlüssel pro Stunde, und
mit `request_id` im Text, wo es einen gibt.

## 7. Logs oder Datenbank?

Textlogs allein reichen für die Fehlersuche im Nachhinein **nicht**: Der
interessante Zustand (Pool-Belegung, Queue, residentes Modell, Swap-Status,
Ledger) steht zum Zeitpunkt des Fehlers nirgends als Ganzes, und drei
Logströme (stdout, `atlas-*.log`, `critical-*.log`) lassen sich nicht nach
`request_id` oder Zeitfenster abfragen. Genau deshalb schreibt der
NVMe-Sweep sein eigenes `events.jsonl` mit Speicherstand vor und nach jedem
Request.

Eine externe Datenbank (Postgres, ClickHouse) ist aber das falsche Werkzeug
für eine einzelne Box: zusätzlicher Dienst, Netz, Betrieb. Empfehlung:
**eingebettetes SQLite** (`rusqlite`, `bundled`, eine Datei auf NVMe,
`--event-db <pfad>`), geschrieben vom selben Hintergrund-Thread wie der
Notifier (SBIO: Business-Logik sendet Events in einen Kanal, der Thread
schreibt). Tabellen:

- `events(ts, level, kind, model, request_id, detail_json)` — jedes
  Anomalie-Signal aus §6 und jede WARN+-Logzeile.
- `snapshots(ts, trigger, state_json)` — Inhalt von `/internal/state` +
  `/v1/stats`, alle 30 s **und** bei jedem Event der Prio ≥ 3. Damit steht zu
  jedem Fehler der Zustand unmittelbar davor.
- `requests(request_id, ts_in, ts_out, endpoint, model, status, error_code,
  estimated_s, actual_s, prompt_tokens, output_tokens)` — eine Zeile je
  Anfrage. Beantwortet „welche Anfragen liefen, als Pool X voll war" und
  liefert die Datenbasis für die Schätzungs-Nachkalibrierung.

Rotation nach Größe (`--event-db-max-mb`), alte Zeilen löschen. Textlogs
bleiben, weil sie ohne Werkzeug lesbar sind; `critical-*.log` und
Crash-Snapshots bleiben die Rückfallebene, wenn SQLite selbst nicht
schreibbar ist. Prometheus bleibt für Zeitreihen, ersetzt aber keine
Ereignis-Korrelation.

Der Ereignisspeicher ist Teil von PR C, nicht später: Ohne ihn finden die
Alarme den Fehler, aber nicht die Ursache. Entschieden (Jonas, 2026-10-11).

## 8. Umsetzungsstand

- 2026-10-11: Dokument erstellt, Entscheidungen §5 getroffen.
- 2026-10-11, PR B umgesetzt (uncommittiert auf `modifications`):
  - `ocr/pipeline.rs`: gemeinsame Seiten-Pipeline für `/v1/ocr` und
    `/convert`, typisierte Fehler mit `error_code`, Abschneiden
    (`finish_reason != stop`) ist ein Fehler. **Verhaltensänderung
    `/v1/ocr`:** abgeschnittene Seiten liefern jetzt 500
    `incomplete_document` statt gekürztem Erfolg; alle Fehler tragen `code`.
  - `api/convert/`: `params` (Multipart, alle Datalab-Felder), `job_store`
    (TTL, Verwaist, Frist, Bytes-/Job-Cap, CSPRNG-IDs), `cache`
    (inhaltsadressiert, BLAKE3, TTL + LRU-Prune, `skip_cache`),
    `result_tree` (Marker-Baum, Mapping, `section_hierarchy`, JPEG-Bilder),
    `worker` (Seitenzahl, Auswahl, Cache, Fan-out mit Re-Gate je Seite,
    Seitenisolation, Fristdurchsetzung, geseedete Retries), `mod`
    (Routen, Datalab-Fehlerform, `request_check_url`).
  - Routen `POST /convert`, `GET|DELETE /convert/{id}`; CORS um `DELETE`
    erweitert; Auth deckt `/convert` ab und akzeptiert `X-API-Key` neben
    `Bearer`; `/internal/state.convert`; 9 Serve-Flags (Settings.md).
  - `ocr/input.rs`: `pdf_page_count`, `render_pdf_page` (Lock nur pro
    Seite), `is_pdfium_missing`; `ocr/output.rs`: `fixup_block_content`,
    `crop_block`, `encode_jpeg`.
  - Tests: 39 Unit-Tests grün (`convert::`, `ocr::`, Middleware);
    `tests/test_convert_contract.py` On-Box grün (2026-10-11, Testserver
    Port 11435, nur Chandra, `--ocr-page-parallelism 3`, dreiseitiges
    Text-PDF): Einreichen 15 ms; 1 Seite 12,0 s; 3 Seiten parallel
    28,9–36,4 s (Fan-out bringt bei diesem Modell kaum Wandzeit — Prefill
    mit Vision-Encoder dominiert; die Annahme „8 Seiten ≈ 1 Seite“ in
    Frage 5 gilt so nicht, Kalibrierung in PR A); Cache-Treffer sofort,
    IDs und Markdown byte-identisch; `DELETE` idempotent; 0,5-s-Frist ⇒
    `deadline_exceeded`; Server-Log ohne ERROR/Panic.
  - Noch nicht in PR B: seitenweiser Cache-Schlüssel (nur Datei-Ebene);
    `estimated_seconds` ist `null` bis PR A; laufende Generierungen laufen
    nach `DELETE` bis zum Seitenende weiter (Abbruch im Scheduler für
    blockierende Anfragen = PR A, Punkt 6); Rate-Limiter zählt `/convert`
    nicht (nur `/v1/*`).

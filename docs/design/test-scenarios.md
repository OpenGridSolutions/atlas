# AtlasInference Server – Testszenarien (bereinigt)

Stand: 2026-10-07. Bereinigte Fassung des Katalogs „AtlasInference Server –
Testszenarien“ (Jan Groth, 2026-10-06). Gestrichen sind Szenarien, die
Pfade prüfen, die es im Code nicht gibt, oder Infrastruktur außerhalb des
Repos. Die Spalte **Stand** bindet jedes Szenario an vorhandene Tests.

Abkürzungen in **Stand**: `Unit` = Rust-Unit-Test vorhanden (Datei),
`neu` = in dieser Runde ergänzt, `offen` = noch zu schreiben,
`On-Box` = nur mit GB10 und Produktivmodell prüfbar.

## 0. Änderungen gegenüber dem Ursprungskatalog

### Gestrichen

| ID | Grund |
| --- | --- |
| T-KV-07 | Es gibt keine Laufzeit-Kalibrierung, die neue FP8-Skalen erzeugt. Die Skalen hängen am Modellfile und damit an `layout_hash`; T-KV-06 deckt das ab. |
| T-NVM-14 | Der Block-Store hat keinen io_uring-Pfad. Er schreibt mit O_DIRECT und fällt auf gepuffertes I/O zurück. Die io_uring-Backends unter `spark-storage/src/backend/` gehören zum High-Speed-Weight-Swap. |
| T-BAT-05 | BAT-4 (Schrittzeit-Lernen) ist bewusst nicht gebaut; die Bremse ist `--tbt-deadline-ms` (T-BAT-01). |
| T-SEC-01, 02, 03, 07, 08, 10, 11 | Prüfen die Traefik-Konfiguration, die nicht im Repo liegt (nur ein Snippet in `Settings.md`). Gehören in eine Deployment-Checkliste neben die Traefik-Konfiguration. |
| T-CMP-02/05/06/11/12/17 bis 22, T-SCH-13, T-WAIT-08 | Kontext-Kompaktierung Teil 2 (Hintergrund-Jobs, Qualitätsmessung, Bilder, TTL-Ablauf, Endpunkt) ist noch nicht implementiert; Teil 1 (synchroner Pfad) ist seit 2026-10-12 auf dem Branch, siehe Abschnitt 10. |

### Zurückgestellt (Stufe 2, Pfad nicht implementiert)

| ID | Grund |
| --- | --- |
| T-SSM-02, 03, 06 | Die `.tail`-Datei wird geschrieben, der Restore daraus ist nicht implementiert. Heute ist der korrekte Rückfall der letzte blockalignierte Snapshot; das prüft T-SSM-04. |
| T-NVM-13 (Teil `--high-speed-swap`) | Zählung der Swap-Schreibvorgänge im Schreibbudget ist nicht vorgesehen. Der Budget-Zähler selbst bleibt (siehe T-NVM-13). |

### Entfallen durch Architekturentscheidung (2026-10-07)

Die Session-Slots (E6: `slots_total`, `slot_tokens`, MOD-12, PRI-7) wurden
zurückgenommen. Kapazität und Nebenläufigkeit sind rein blockbasiert
(`--max-batch-size`, KV-Pool, LRU-Verdrängung, Spill). Damit entfällt
Invariante 5 „Speicherzaun“ in ihrer Slot-Form: Overcommit führt zu
Verdrängung und Spill, nicht zu Warteschlange. T-SLOT-01 bis 06, T-MOD-09
und T-PRI-03 sind gestrichen.

### Umformuliert

- **Determinismus-Vergleich.** „Tokenidentisch“ ist bei gebatchtem Decode
  nicht garantiert, weil die Batch-Zusammensetzung die Reduktionsreihenfolge
  ändert. Referenz ist der Vergleich auf Logit-Ebene mit
  `tests/metal_kv_kld_compare.py` (KLD und Top-1-Übereinstimmung je
  Schritt). Betrifft T-KV-01, T-KV-02, T-SSM-01, T-NVM-01, T-LOAD-02.
- **T-LOAD-01.** „Kein Wachstum des Speicherverbrauchs“ wird an der
  Allokation des Prozesses gemessen, nicht an `MemAvailable`: der GB10-Treiber
  hält Speicher beendeter Prozesse.
- **T-WAIT-01.** Sofort-SSE gibt es für `/v1/chat/completions`;
  `/v1/completions` wartet inline. Die Erwartung gilt für Chat.
- **T-BAT-06.** `ATLAS_KV_OVERCOMMIT=1` schaltet nur den Ladefehler in eine
  Warnung um. Die Präemption wird mit einem bewusst kleinen Pool erzwungen.

### Neu aufgenommen

| ID | Szenario | Grund |
| --- | --- | --- |
| T-KV-11 | P1 · Integration – Modellfile unter gleichem Namen ersetzt (neuer Snapshot-Hash). Erwartet: keine Treffer auf NVMe-Einträge der alten Revision. | Die Cache-Identität nutzt die Modell-ID, nicht den Snapshot-Hash (`compute_kv_identity`). Loch unter Invariante 1. |
| T-KV-12 | P1 · Unit – Generierte Token-IDs gegen die Retokenisierung des zurückgeschickten Assistant-Textes. | T-DRN-03 prüft das nur implizit; ein Unit-Test trifft das Risiko aus `nvme-tier-analysis.md` §1.3 direkt. |
| T-API-01 | P1 · Integration – Jeder Header-Test (`X-Atlas-Model-Swap`, `X-Atlas-Compacted` sobald vorhanden) auch auf `/v1/messages`. | Protokolldrift zwischen OpenAI- und Anthropic-Pfad ist laut AGENTS.md ein wiederkehrender Fehler. |

## 1. Prefix-Cache und Block-Schlüssel

| ID | Szenario und Erwartung | Bezug | Stand |
| --- | --- | --- | --- |
| T-KV-01 | P1 · Integration – Derselbe Prompt zweimal. Zweiter Request trifft alle vollen Blöcke; Logits im Toleranzband. | KV-2, KV-3 | offen |
| T-KV-02 | P1 · Integration – Gemeinsamer Anfang, Abweichung ab Token 100. Treffer genau bis zum ersten Unterschied. | KV-3 | Unit `block_keys_tests.rs` (Präfixstabilität); Integration offen |
| T-KV-03 | P1 · Unit – Gleicher Text, zwei Bilder. Kein geteilter Block ab dem ersten Bild-Platzhalter. | KV-4 | Unit `block_keys_tests.rs::media_divergence_starts_at_first_media_block` |
| T-KV-04 | P1 · Unit – Dasselbe Bild, andere Vorverarbeitung. Unterschiedliche Schlüssel. | KV-4 | teilweise: Digest aus vorverarbeiteten Pixeln (`media_digests_of`), Unit für den Digest offen |
| T-KV-05 | P1 · Integration – Dasselbe Bild, gleicher Text. Treffer über den Bildblock hinweg. | KV-4 | offen |
| T-KV-06 | P1 · Unit – Jeder Bestandteil von `layout_hash` einzeln geändert. Alte Blöcke unauffindbar. | KV-5 | Unit `block_keys_tests.rs::different_seed_diverges_from_block_zero`, `block_store/header.rs` |
| T-KV-08 | P1 · Unit + On-Box – Verworfene Draft-Tokens (Self-Spec, N-Gram, DFlash) und Watchdog-Rollback. Schlüssel und Inhalt versiegelter Blöcke identisch zum Lauf ohne Spekulation. | KV-8 | Unit **neu** `block_keys_tests.rs::rejected_drafts_never_seal_a_block` (Fix: `BlockKeyChain::truncate_to`, `SequenceState::pop_tokens`); On-Box offen |
| T-KV-09 | P2 · Integration – Gemeinsamer Systemprompt, Refcount > 1, sinkt beim Ende. | KV-2 | Unit `radix_tree/tests/`; Integration offen |
| T-KV-10 | P2 · Integration – Pool voll mit Refcount-0-Blöcken. LRU verdrängt nur persistierte Blöcke. | KV-6, KV-7 | Unit `prefix_cache.rs` (Evict-Guard); Integration offen |
| T-KV-11 | P1 · Integration – Modellrevision ohne Namensänderung. | KV-5 | offen |
| T-KV-12 | P1 · Unit – Retokenisierung der Antwort. | DRN-2 | offen |

## 2. SSM-Zustand der Hybridmodelle

| ID | Szenario und Erwartung | Bezug | Stand |
| --- | --- | --- | --- |
| T-SSM-01 | P1 · On-Box – Ende an Blockgrenze mit Snapshot; Folge-Request setzt dort fort. | SSM-1, SSM-4 | On-Box 2026-10-07 Qwen3-Next-80B (`tests/nvme_cold_cache_bench.py`, 7,7k-Token-Sessions): 28/28 Restores setzen am Prompt-Tail-Checkpoint (7680) fort, TTFT 0,31 s statt 8,5 s |
| T-SSM-04 | P1 · On-Box – KV bis Block 40, Snapshots bei 16 und 32. Resume bei 32, Blöcke 33–40 neu berechnet. | SSM-4 | Unit `nvme_tier.rs::keeps_every_aligned_checkpoint_deepest_first`; Staging wählt den tiefsten `.ssm` im Bereich (On-Box indirekt über T-SSM-01) |
| T-SSM-05 | P1 · Integration – Checksumme des Snapshots bei 32 manipuliert. Rückfall auf 16, Datei gelöscht. | SSM-5, NVM-6 | Unit `block_store` (Checksumme); Rückfall offen |
| T-SSM-07 | P2 · Unit – Reines Attention-Modell: keine `.ssm`-Dateien, Tail nur KV. | SSM-8 | Unit `radix_tree/tests/snapshot.rs`; NVMe-Seite offen |
| T-SSM-08 | P3 · On-Box – Snapshot-Dichte 1, 4, 16. | SSM-7, MEM-9 | offen |

## 3. NVMe-Block-Store

| ID | Szenario und Erwartung | Bezug | Stand |
| --- | --- | --- | --- |
| T-NVM-01 | P1 · On-Box – Modellwechsel hin und zurück, Restore von NVMe, Logits im Toleranzband. | NVM-1, BAT-6 | offen (`scripts/nvme_tier_smoke.sh`) |
| T-NVM-02 | P1 · Integration – `kill -9` während des Schreibens. Keine halbe Datei unter gültigem Namen. | NVM-5 | Unit `block_store/writer.rs` (tmp + rename); Crash-Test offen |
| T-NVM-03 | P1 · Unit – Ein Payload-Byte gekippt. Restore endet davor, Datei gelöscht. | NVM-6 | Unit **neu** `kv_tier_tests.rs::fetch_blocks_ends_the_chain_at_an_invalid_record` |
| T-NVM-04 | P1 · Unit – Datei abgeschnitten. Wie T-NVM-03. | NVM-6 | Unit **neu** (gleicher Test) |
| T-NVM-05 | P1 · Unit – Falscher `model_hash`, `layout_hash`, Format-Version. Datei verworfen. | NVM-7 | Unit `block_store/header.rs`, `block_store/mod.rs` |
| T-NVM-06 | P2 · Unit – Block zweimal persistiert. Ein Schreibvorgang. | NVM-3 | Unit `writer.rs::duplicate_key_skipped_at_enqueue` |
| T-NVM-07 | P2 · Unit – Index gelöscht, Neustart. Aufbau aus Verzeichnis-Scan. | NVM-8 | Unit `block_store/index.rs`; Neustart **neu** `kv_tier_tests.rs::flush_persists_the_index_for_a_restart` |
| T-NVM-08 | P2 · Unit – Tiefer Zugriff aktualisiert `last_access` der Vorgänger. | NVM-9 | Unit `gc.rs::recency_propagation_protects_whole_chain` |
| T-NVM-09 | P2 · Unit – TTL-Ablauf und Waisen. | NVM-10 | Unit `gc.rs::ttl_expires_old_records_and_cascades_orphans` |
| T-NVM-10 | P2 · On-Box – Dateisystem unter `nvme_min_free`. GC räumt, Server läuft. | NVM-10 | Unit `gc.rs::pressure_evicts_lru_first`; On-Box offen |
| T-NVM-11 | P2 · Integration – GC läuft während eines Restores derselben Kette. | NVM-10, BAT-6 | offen |
| T-NVM-12 | P2 · On-Box – NVMe gedrosselt. Queue füllt sich, Generierung läuft weiter. | NVM-11, KV-7 | RAM-Seite **neu** `kv_tier_tests.rs::rejected_writes_are_parked_and_pumped_later`; On-Box offen |
| T-NVM-13 | P2 · Integration – Schreibbudget niedrig. Warnung, Zähler korrekt. | NVM-12 | offen |
| T-NVM-15 | P3 · On-Box – Restore- und Schreibdurchsatz gegen Neuberechnung. | BAT-7, NVM-13 | offen |

## 4. Speicherbudget, Modelle und Sidecars

| ID | Szenario und Erwartung | Bezug | Stand |
| --- | --- | --- | --- |
| T-MEM-01 | P1 · Unit – Replay von `c847244` mit großem `SwapFree`. | MEM-4 | Unit `kv_budget.rs` (SwapFree-Ausschluss); Replay-Fall prüfen |
| T-MEM-02 | P2 · On-Box – Fremdprozess belegt und beendet; Budget nutzt Baseline. | MEM-5 | offen |
| T-MEM-03 | P2 · On-Box – Whisper, Qwen-Image, Embed nacheinander; Budget wird neu berechnet, kein OOM. | MEM-1, MEM-6 | offen |
| T-MEM-04 | P2 · Unit – Gemischte Layer-Dtypes; Bytes/Token = Summe. | MEM-3, MEM-9 | Unit `kv_budget.rs` teilweise |
| T-MOD-01 bis 04 | P2 – Messung, Persistenz, Verwurf bei neuem Hash, EWMA. | MOD-5..8 | Unit `swap_costs.rs` teilweise; On-Box offen |
| T-MOD-05 | P2 · Integration – Laden schlägt fehl; vorheriges Modell zurück, Wartende 503. | MOD-10 | Unit **neu** `queued_sse_tests.rs::rejected_ticket_reports_the_failed_load` (Stream-Seite); Integration offen |
| T-MOD-06 | P2 · Integration – Embed unter `sidecar_small_max_gib` als Sidecar, Arena unverändert (nur noch mit explizitem Flag; Default 0 = aus). | MOD-3, MOD-4 | offen (`text_sidecar.rs` ohne Tests) |
| T-MOD-07 | P2 · Integration – Embed über der Grenze in die Arena-Queue. | MOD-4, SCH-2 | offen |
| T-MOD-08 | P2 · Integration – Unbekanntes Modell: 404 mit Liste. | MOD-9 | offen |

## 5. Scheduler und Modellwechsel-Entscheidung

| ID | Szenario und Erwartung | Bezug | Stand |
| --- | --- | --- | --- |
| T-SCH-01 bis 08 | P2 · Unit – Replay gegen die Entscheidungsfunktion: Schwelle, Druck, Idle, Ping-Pong, Starvation, Kandidatenwahl, Priorität. | SCH-4, PRI-2, PRI-3 | Unit `arena_sched/decision.rs` (11 Fälle); Abgleich auf fehlende Fälle |
| T-SCH-09 | P2 · Unit – Eine Stunde aufgezeichneter Verkehr, κ = 0,5 / 1 / 2. | SCH-5 | offen; Voraussetzung: TEL-4-Events aufzeichnen |
| T-SCH-10 | P2 · Integration – Queue über `max_queue_length`: 429 mit `Retry-After`. | SCH-8 | Unit `queues.rs::queue_full_rejects_with_429_semantics`; Integration offen |
| T-SCH-11 | P2 · Integration – Translate/OCR für nicht geladenes Modell werden eingereiht. | SCH-2 | offen |
| T-SCH-12 | P2 · Integration – Header `X-Atlas-Model-Swap` mit Wartezeit. | SCH-9 | Schätzung Unit **neu** `decision.rs::wait_tests`; Header offen |

## 6. Batching und Admission (blockbasiert)

Seit 2026-10-07 gibt es keine Session-Slots mehr (E6 wurde zurückgenommen).
Nebenläufigkeit ist `--max-batch-size`, Kapazität ist der KV-Pool: LRU-Verdrängung
des Prefix-Caches (persistierte Blöcke zuerst), dann Spill auf Platte.

| ID | Szenario und Erwartung | Bezug | Stand |
| --- | --- | --- | --- |
| T-BAT-01 | P3 · On-Box – 30k-Prompt neben vier Streams; TBT unter `--tbt-deadline-ms`. | BAT-2 | offen |
| T-BAT-02 | P2 · On-Box – Langer NVMe-Prefix neben laufenden Streams; Restore asynchron. | BAT-6 | On-Box 2026-10-07, Qwen3-VL-30B (2,8k): Import 35 ms, max. Stall 0,31 s (vorher 2,6 s), kalte TTFT 0,22 s (vorher 2,5 s), 22/22 Treffer. Qwen3-Next-80B (7,7k, hybrid): max. Stall 0,38 s (vorher 8,5 s), kalte TTFT 0,31 s (vorher 8,5 s), 28/28 Treffer mit SSM-Resume |
| T-BAT-03 | P3 · Integration – Prefix kürzer als `nvme_restore_min_tokens`: Neuberechnung. | BAT-7 | offen |
| T-BAT-04 | P2 · Integration – Priorisierter und normaler Request, ein Platz frei. | BAT-3, PRI-1 | Unit `queues.rs::grant_wakes_waiters_priority_first`; Admission offen |
| T-BAT-06 | P2 · On-Box – Overcommit mit kleinem Pool: Cache-Verdrängung vor Spill, Spill-Resume funktioniert. | BAT-5 | On-Box 2026-10-07 (`tests/concurrency_bench.py`, `--kv-pool-gb 1`): N > Batch wartet, kein Reject/Drop; Fixes a721f36b, dfa141f0 |
| T-BAT-07 | P2 · Integration – Prefill-Allokation bei Pool voll mit Cache: Verdrängung greift, kein `KV cache exhausted`. | KV-6, KV-7 | Unit `radix_tree/tests/evict_guard.rs` |

## 7. Ablauf eines Modellwechsels

| ID | Szenario und Erwartung | Bezug | Stand |
| --- | --- | --- | --- |
| T-DRN-01 | P2 · On-Box – Kurze Streams enden regulär innerhalb `drain_grace_ms`. | DRN-1 | offen |
| T-DRN-02 | P2 · On-Box – Langer Stream endet mit `finish_reason="model_swap"` und `[DONE]`. | DRN-1, E1 | offen |
| T-DRN-03 | P1 · On-Box – Fortsetzung nach Abbruch nutzt den Prefix bis zur Abbruchstelle. | DRN-2 | offen (dazu T-KV-12) |
| T-DRN-04 | P2 · On-Box – NVMe gedrosselt; Wechsel nach `persist_timeout`. | DRN-3 | Unit `kv_tier_tests.rs::flush_persists_the_index_for_a_restart` (Flush-Pfad); On-Box offen |
| T-DRN-05 | P2 · Integration – Requests für A, B, C während Drain werden eingereiht. | DRN-4, SCH-7 | offen |
| T-DRN-06 | P2 · Integration – Client trennt in der Queue; Request verschwindet. | DRN-5 | Unit `queues.rs::dropped_ticket_leaves_the_queue` |
| T-DRN-07 | P2 · Integration – Persistierung beginnt beim Ende jeder Sequenz. | DRN-6 | offen |
| T-DRN-08 | P2 · On-Box – `/health` zeigt `draining`, `swapping`, `ready`; `/health/live` bleibt 200. | TEL-1, TEL-2 | Unit **neu** `middleware_tests.rs::swap_gate_spares_introspection_and_lazy_endpoints`; On-Box offen |
| T-DRN-09 | P2 · On-Box – Zweite Entscheidung während eines Wechsels wartet. | Abschnitt 10 | offen |
| T-DRN-10 | P2 · On-Box – Wechsel während Qwen-Image arbeitet. | MOD-2 | offen |

## 8. Wartende Clients und Timeouts

| ID | Szenario und Erwartung | Bezug | Stand |
| --- | --- | --- | --- |
| T-WAIT-01 | P2 · Integration – Chat-Stream eingereiht: sofort 200, SSE-Header, `: queued`. | WAIT-1 | Unit **neu** `queued_sse_tests.rs::queued_comment_arrives_before_the_grant` |
| T-WAIT-02 | P2 · On-Box – Drei Minuten Wartezeit durch die Proxy-Kette. | WAIT-2 | Server-Seite **neu** `queued_sse_tests.rs::keepalive_comments_tick_while_queued`; Proxy-Kette offen |
| T-WAIT-03 | P2 · Integration – OpenAI-SDK und JS-Client ignorieren Kommentare. | WAIT-2 | offen |
| T-WAIT-04 | P2 · Integration – Ohne Streaming, Schätzung über `nonstream_max_wait`: sofort 503. | WAIT-3 | Schätzung Unit **neu** `decision.rs::wait_tests`; Integration offen |
| T-WAIT-05 | P2 · Integration – Reale Wartezeit über der Grenze: 503. | WAIT-4 | Unit `queues.rs::expiry_hits_foreign_priority_and_nonstream` |
| T-WAIT-06 | P2 · Integration – Laden schlägt nach `: queued` fehl: SSE-`error`, dann `[DONE]`. | WAIT-5 | Unit **neu** `queued_sse_tests.rs` (Expired, Rejected, Non-SSE-Fehler) |
| T-WAIT-07 | P2 · Integration – Embed über Sidecar während eines Wechsels: sofort. | WAIT-6 | offen |

## 9. Priorität und Authentifizierung (Server-Seite)

| ID | Szenario und Erwartung | Bezug | Stand |
| --- | --- | --- | --- |
| T-SEC-04 | P1 · On-Box – Port 11434 von einem anderen Host nicht erreichbar. | AUTH-2 | offen; die laufende Instanz mit `--bind 0.0.0.0` fällt durch |
| T-SEC-05 | P1 · Integration – `--trust-proxy-headers` mit `--bind 0.0.0.0`: Start verweigert. | AUTH-4 | Unit **neu** `middleware_tests.rs::trust_proxy_headers_requires_a_loopback_bind` |
| T-SEC-06 | P1 · Integration – Ohne `--trust-proxy-headers` wird `X-Priority` ignoriert. | AUTH-3 | Unit **neu** `middleware_tests.rs::x_priority_is_ignored_without_trust` |
| T-SEC-09 | P1 · On-Box – `/internal/state` und `/admin` ohne Token: 401. | TEL-7 | Unit **neu** `middleware_tests.rs::introspection_and_admin_paths_need_auth`; Integration `tests/test_internal_state.py` |
| T-PRI-01 | P2 · Integration – Zehn normale, dann ein priorisierter: der wird als nächster aufgenommen. | PRI-1 | Unit `queues.rs::grant_wakes_waiters_priority_first` |
| T-PRI-02 | P2 · Integration – Priorisiert für nicht geladenes Modell: 503 nach `priority_max_wait`. | PRI-4 | Unit `queues.rs` (Expiry) + **neu** `queued_sse_tests.rs::expired_ticket_ends_the_stream_with_an_sse_error` |

## 10. Telemetrie und Health

| ID | Szenario und Erwartung | Bezug | Stand |
| --- | --- | --- | --- |
| T-TEL-01 | P2 · Integration – `/internal/state` unter Last: alle TEL-3-Felder, konsistent mit Batch und Pool. | TEL-3 | **neu** Unit `internal_state.rs::capacity_profile_derives_from_pool_geometry`; Integration `tests/test_internal_state.py`. Hinweis: die Route fehlte bis 2026-10-07 im Router. |
| T-TEL-02 | P2 · On-Box – Drei Wechsel mit verschiedenen Auslösern: je ein Swap-Event. | TEL-4 | offen |
| T-TEL-03 | P2 · Integration – Metriken gegen gezählte Werte. | TEL-5, TEL-6, TEL-8 | offen |
| T-TEL-04 | P3 · On-Box – 10 Abfragen/s auf `/internal/state` und `/metrics` ohne TPS-Einbruch. | TEL-3 | offen |
| T-TEL-05 | P2 · Integration – Kapazitätsprofil gegen unabhängige Nachrechnung. | MEM-9 | Unit **neu** (siehe T-TEL-01); Nachrechnung offen |

## 11. Fehlerinjektion und Dauerlast

| ID | Szenario und Erwartung | Bezug | Stand |
| --- | --- | --- | --- |
| T-LOAD-01 | P2 · Last – 24 h gemischte Last; kein Absturz, kein OOM, keine wachsende Prozessallokation. | alle | offen |
| T-LOAD-02 | P1 · Last – Alle 10 min Kontroll-Requests mit bekannter Referenz; Logits im Toleranzband. | KV-3, NVM-6 | offen; wertvollster Test des Katalogs |
| T-LOAD-03 | P2 · Last – `kill -9` während Wechsel und Persistierung; sauberer Neustart. | NVM-5, NVM-9 | offen |
| T-LOAD-04 | P2 · Last – Bitfehler in 1 % der Cache-Dateien; alle erkannt. | NVM-6, SSM-5 | Unit T-NVM-03/04; Last offen |
| T-LOAD-05 | P2 · Last – NVMe wechselt zwischen voll und gedrosselt. | NVM-11 | offen |
| T-LOAD-06 | P2 · Last – NVMe läuft voll; GC räumt, Persistierung scheitert kontrolliert. | NVM-10 | offen |
| T-LOAD-07 | P2 · Last – 50 Requests in 2 s für ein nicht geladenes Modell: genau ein Wechsel. | SCH-4 | offen |
| T-LOAD-08 | P2 · Last – 50 % der Streams brechen ab; Plätze und Blöcke werden frei. | DRN-5 | Unit `queues.rs::dropped_ticket_leaves_the_queue`; Last offen |
| T-LOAD-09 | P1 · Last – Komplette Rust-Unit-Suite und Qualitäts-Suite (FP8-KV) grün. | Bestand | `cargo test --workspace`, `tests/gate_results.py` |

## P1-Prüfpfad (bereinigt)

1. **T-SEC-04** – Server nicht am Proxy vorbei erreichbar.
2. **T-SEC-06** – `X-Priority` ohne `--trust-proxy-headers` wirkungslos.
3. **T-SEC-09** – `/internal/state` nur mit Token.
4. **T-TEL-01** – `/internal/state` erreichbar und konsistent.
5. **T-KV-03** – Gleicher Text, anderes Bild: kein Treffer.
6. **T-KV-08** – Verworfene Drafts und Rollback versiegeln nichts.
7. **T-NVM-01** – Restore nach Modellwechsel im Toleranzband.
8. **T-NVM-03** – Beschädigter Block wird nie verwendet.
9. **T-SSM-04** – Resume am längsten blockalignierten Snapshot.
10. **T-DRN-03** – Fortsetzung nach `model_swap` nutzt den Prefix.

## 10. Kontext-Kompaktierung (Teil 1, seit 2026-10-12)

Spezifikation: `docs/design/context-compaction.md`. Unit-Tests in
`crates/spark-server/src/compaction/tests.rs`, On-Box-Lauf mit
`ATLAS_MAX_SEQ_LEN=4096 ./serve.sh` und `tests/test_compaction.py --model <id>`.

| ID | Szenario und Erwartung | Stand | Bezug |
| --- | --- | --- | --- |
| T-CMP-01 | P1 · Integration – Kurze Session. Erwartet: keine Kompaktierung, kein `X-Atlas-Compacted`. | test_compaction.py | CMP-9 |
| T-CMP-03 | P1 · Integration – Historie passt nicht mehr. Erwartet: synchrone Kompaktierung, `X-Atlas-Compacted: true` + `X-Atlas-Compaction-Id`. | test_compaction.py | CMP-2, CMP-10, CTX-4 |
| T-CMP-04 | P1 · Integration – Folge-Request mit voller Historie + neuer Nachricht. Erwartet: Eintrag wiedererkannt (gleiche Id), Prompt bleibt klein, Fakt aus dem zusammengefassten Teil beantwortbar. | test_compaction.py | CMP-1, CMP-2 |
| T-CMP-07 | P1 · Integration – Zusammengefasste Nachricht geändert. Erwartet: neue Kompaktierung mit anderer Id. | test_compaction.py | CMP-12 |
| T-CMP-08 | P1 · Integration – Nachricht nach dem Schnitt geändert. Erwartet: Eintrag bleibt angewendet. | test_compaction.py | CMP-2 |
| T-CMP-09 | P1 · Unit – Tool-Call und Tool-Ergebnis an der Schnittstelle bleiben zusammen. | `cut_never_separates_a_tool_call_from_its_result`, `cut_respects_tool_pairs_when_shrinking` | CMP-5 |
| T-CMP-10 | P1 · Unit – Kopf bei mehreren System-, Developer-, ohne System-Nachricht, mit/ohne `keep_first_user`; ohne System-Nachricht wird eine erzeugt. | `head_detection_cases`, `effective_creates_system_message_when_head_has_none` | CMP-4, CMP-6 |
| T-CMP-13 | P1 · Integration – Einzelne Nachricht größer als das Fenster. Erwartet: 400 `context_length_exceeded` mit Größenangaben. | test_compaction.py | CTX-6 |
| T-CMP-14 | P2 · Integration – Header `reject`. Erwartet: 400, keine Kompaktierung. | test_compaction.py | CTX-1, CTX-2 |
| T-CMP-15 | P2 · Integration – Header `truncate`. Erwartet: Eintrag mit leerer Zusammenfassung. | test_compaction.py | CTX-3 |
| T-CMP-16 | P2 · Integration – Kein Header = `summarize`; ungültiger Wert ⇒ 400 `invalid_context_policy`. | test_compaction.py | CTX-1 |
| T-CMP-23 | P2 · Integration – Streaming-Request mit synchroner Kompaktierung. Erwartet: `: compacting`-Keep-alives, `: compacted id=`-Kommentar, vollständiger Stream. | test_compaction.py | CMP-10, WAIT-2 |
| T-API-01 | P1 · Integration – `/v1/messages` trägt die Kompaktierungs-Header. | test_compaction.py | CTX-4 |
| T-CMP-24 | P1 · Unit – Testvektoren der kanonischen Serialisierung und der Kette; NFC; `reasoning_content` ohne Einfluss; Bild-Digests. | `canonical_*`, `chain_*` | CMP-1 |
| T-NVM-xx | P2 · Unit – `.cmp`-Einträge hängen am `.cmp`-Vorgänger, nicht an einem KV-Block; TTL kaskadiert. | spark-storage `cmp_entries_chain_to_cmp_predecessors_not_kv` | §5, NVM-10 |

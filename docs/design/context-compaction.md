# Kontext-Kompaktierung (E7, Runde B)

Stand: 2026-10-12, Branch `modifications`. Anforderungen aus „AtlasInference
Server – Anforderungen" (Jan Groth, v3 vom 2026-10-06, §12 und
Querbezüge), die vier Umsetzungsentscheidungen vom 2026-10-06 und der
Umsetzungsplan. Die Testszenarien T-CMP-01 bis 22, T-SCH-13 und T-WAIT-08
stehen im Ursprungskatalog „AtlasInference Server – Testszenarien"
(2026-10-06) und kommen mit dieser Runde zurück in
`docs/design/test-scenarios.md`.

## 1. Architekturentscheidung E7

Die Kompaktierung (Kopf + Zusammenfassung + jüngste Runden) ist
standardmäßig aktiv. Abweisen mit `context_length_exceeded` gibt es nur
noch, wenn der Client es ausdrücklich verlangt oder ein Request auch nach
Kompaktierung nicht passt. Sessions laufen unbegrenzt weiter, ohne dass
Clients eigenes Kontextmanagement brauchen; Clients mit eigenem
Kontextmanagement sind nicht betroffen, weil sie das Fenster nie
überschreiten.

**Abweichung vom Dokument:** Die Session-Slots (E6) wurden am 2026-10-07
auf Wunsch des Betreibers wieder entfernt (Commit ef839fc9); Kapazität ist
blockbasiert. Überall, wo das Dokument `slot_tokens` sagt, gilt hier das
effektive Kontextfenster des Modells (`--max-seq-len`, bzw. die
Modellgrenze, wenn sie kleiner ist). `slot_buffer_tokens` entfällt;
`max_tokens` wird gegen das verbleibende Fenster geklemmt.

## 2. Anforderungen (v3, Status „Neu")

### Nachrichten-Kette und Kompaktierungseinträge

Jede Nachricht einer Konversation erhält einen verketteten Schlüssel über
Rolle, Inhalt, Tool-Calls und Medien:

    m_0 = H(model_hash)
    m_j = H(m_{j-1} || role_j || content_j || tool_calls_j || media_digests_j)

Ein Kompaktierungseintrag hat den Schlüssel m_c der letzten
zusammengefassten Nachricht und sagt: „Nachrichten nach dem Kopf bis
einschließlich c werden durch Zusammenfassung Z ersetzt." Effektiver
Kontext für einen Request mit Nachrichten 1 bis n:

    Kopf + Zusammenfassung Z + Nachrichten c+1 bis n

Der effektive Kontext ist eine gewöhnliche Token-Sequenz; sein KV-Cache
wird über die Block-Schlüssel gefunden und persistiert. Nach der ersten
Kompaktierung treffen alle Folge-Requests wieder den Cache.

| ID | Anforderung | Prio |
| --- | --- | --- |
| CMP-1 | Der Server berechnet für jeden eingehenden Request die Nachrichten-Schlüssel m_1 bis m_n (BLAKE3-128, kanonische JSON-Serialisierung von Rolle, Inhalt und Tool-Calls). | MUSS |
| CMP-2 | Der Server sucht den größten Index c, für den ein Kompaktierungseintrag zu m_c existiert, und baut daraus den effektiven Kontext. Die Nachrichten bis c werden verworfen, bevor das Chat-Template angewendet wird. | MUSS |
| CMP-3 | Kompaktierungen sind kumulativ: Eine neue Zusammenfassung entsteht aus der vorherigen Zusammenfassung plus den neu herausfallenden Nachrichten, nicht aus der gesamten Historie. Der neue Eintrag erhält den Schlüssel der neuen Schnittstelle; der alte bleibt bis zum GC bestehen. | MUSS |
| CMP-4 | Kopf: alle führenden `system`-/`developer`-Nachrichten und, wenn `compaction_keep_first_user` gesetzt ist, die erste `user`-Nachricht. Der Kopf wird nie zusammengefasst. | MUSS |
| CMP-5 | Jüngste Runden: die letzten Nachrichten, deren Summe mindestens `compaction_tail_pct` des Fensters erreicht. Geschnitten wird nur an Nachrichtengrenzen; ein Tool-Call und sein Tool-Ergebnis werden nie getrennt. | MUSS |
| CMP-6 | Die Zusammenfassung wird an die System-Nachricht des Kopfs als eigener Abschnitt angehängt (gibt es keine, wird eine erzeugt). Das hält Chat-Templates ein, die nur eine System-Nachricht am Anfang erlauben. | MUSS |
| CMP-7 | Die Zusammenfassung erstellt das geladene Arena-Modell, mit Temperatur 0, höchstens `compaction_summary_max_tokens` und einem Prompt aus `compaction_prompt_file`. Der Standard-Prompt verlangt, Fakten, Zahlen, Entscheidungen, Code-Bezeichner, offene Aufgaben und Inhalte von Bildern zu erhalten. | MUSS |
| CMP-8 | Proaktiv: Erreicht der effektive Kontext einer Session nach Request-Ende `compaction_trigger_pct` des Fensters, wird die Kompaktierung als Hintergrund-Job eingeplant. Er läuft nur, wenn keine Requests für das geladene Modell warten, löst keinen Modellwechsel aus und wird beim Drain abgebrochen. | MUSS |
| CMP-9 | Ein vorbereiteter Eintrag wird erst angewendet, wenn ein Request ohne ihn nicht ins Fenster passt. Bis dahin läuft die Session ungekürzt weiter. | MUSS |
| CMP-10 | Ist beim Überlauf kein Eintrag vorbereitet, wird synchron kompaktiert; Streaming-Requests erhalten währenddessen Keep-alives (WAIT-2). | MUSS |
| CMP-11 | Einträge werden als `.cmp`-Datei im Block-Store unter `<model_hash>/` gespeichert (JSON: Schlüssel m_c, Kopf-Länge, Zusammenfassung, Vorgänger-Eintrag, Erstellungszeit). Sie unterliegen derselben TTL und demselben GC; jede Verwendung aktualisiert `last_access`. | MUSS |
| CMP-12 | Ändert der Client eine bereits zusammengefasste Nachricht, passt m_c nicht mehr; der Eintrag wird nicht angewendet, und die Historie wird bei Bedarf neu kompaktiert. | MUSS |

### Steuerung durch den Client

| ID | Anforderung | Prio |
| --- | --- | --- |
| CTX-1 | Request-Header `X-Atlas-Context-Policy` mit den Werten `summarize`, `truncate`, `reject`. Ohne Header gilt `--default-context-policy`, Default `summarize`. | MUSS |
| CTX-2 | `reject` gilt nur, wenn der Client es per Header wählt oder der Betreiber den Default umstellt: Passt der Request nicht ins Fenster, antwortet der Server mit HTTP 400 und dem Fehlercode `context_length_exceeded`. | MUSS |
| CTX-3 | `truncate`: wie `summarize`, aber ohne Zusammenfassung; die Nachrichten zwischen Kopf und jüngsten Runden entfallen ersatzlos. Dafür wird ein Eintrag mit leerer Zusammenfassung angelegt. | MUSS |
| CTX-4 | Wurde kompaktiert, enthält die Antwort `X-Atlas-Compacted: true` und `X-Atlas-Compaction-Id` (Hex von m_c). | MUSS |
| CTX-5 | `GET /v1/atlas/compactions/{id}` liefert die Zusammenfassung und die Anzahl ersetzter Nachrichten. | SOLL |
| CTX-6 | Passt ein Request auch nach Kompaktierung nicht ins Fenster, weil Kopf, Zusammenfassung und die neueste Nachricht zusammen mit `max_tokens` zu groß sind, antwortet der Server unabhängig von der Policy mit HTTP 400 und `context_length_exceeded`. Die Fehlermeldung nennt die Größe von Kopf und neuester Nachricht. | MUSS |

### Querbezüge

| ID | Anforderung | Prio |
| --- | --- | --- |
| §5 | Block-Store speichert Kompaktierungseinträge als `.cmp` neben `kvb`/`ssm`/`tail`; der GC behandelt sie typbewusst (ein `.cmp`-Vorgänger ist ein `.cmp`, kein `.kvb`). | MUSS |
| SCH-10 | Hintergrund-Kompaktierungen zählen weder zu N_active,A noch zu N_queued,A, verhindern die Idle-Regel nicht und werden bei einer Wechselentscheidung sofort abgebrochen. Ein abgebrochener Job wird beim nächsten Request der Session neu eingeplant. | MUSS |
| WAIT-3 | Die Wartezeit-Schätzung für Requests ohne Streaming enthält, falls synchron kompaktiert werden muss, die erwartete Kompaktierungsdauer. | MUSS |
| TEL-3 | `/internal/state` zeigt laufende und geplante Kompaktierungen. | MUSS |
| TEL-8 | Kompaktierungs-Metriken: Anzahl je Policy und Auslöser (proaktiv, synchron), Dauer, ersetzte Tokens, Anteil der Requests mit angewendeter Kompaktierung, Anzahl CTX-6-Ablehnungen. | MUSS |

### Konfiguration

| Parameter | Default | Bedeutung |
| --- | --- | --- |
| `--default-context-policy` | `summarize` | Verhalten ohne `X-Atlas-Context-Policy` |
| `--compaction-trigger-pct` | 70 | Ab diesem Füllstand wird im Hintergrund vorbereitet (Dokument: 75, Entscheidung 3) |
| `--compaction-tail-pct` | 25 | Mindestanteil jüngster Runden am Fenster |
| `--compaction-summary-max-tokens` | 768 | Länge der Zusammenfassung (Dokument: 2048, Entscheidung 3) |
| `--compaction-prompt-file` | eingebauter Prompt | Prompt für die Zusammenfassung |
| `--compaction-keep-first-user` | an | Erste User-Nachricht gehört zum Kopf |

## 3. Umsetzungsentscheidungen (2026-10-06)

1. **Eingabe des Hintergrund-Jobs:** Der Server speichert keine Sessions
   (E2). Die Konversation des letzten Requests wird pro Session im RAM
   gehalten (gekeyt über die m-Kette), Obergrenze 256 MB gesamt mit
   LRU-Verwurf. Ein verworfener Job läuft dann eben synchron beim Überlauf
   (CMP-10).
2. **Job-Abbruch bei jeder Arena-Ankunft**, nicht nur bei der
   Wechselentscheidung (SCH-10 allein reicht nicht; echte Requests haben
   immer Vorrang).
3. **`compaction_summary_max_tokens` Default 768** statt 2048 (Decode
   dominiert die synchrone Latenz: 27B ≈ 10 tok/s), Trigger bei 70 %.
4. **Kanonische Serialisierung** präzise definiert (sortierte Keys, NFC,
   Array-Content normalisiert) mit Testvektoren im Code.

Offene Punkte aus dem Dokument (§15): Qualität des eingebauten
Kompaktierungs-Prompts an realen Sessions; Verhalten der Chat-Templates
(Qwen3-Next, Qwen3.5, Qwen3-VL) mit angehängter Zusammenfassung (CMP-6).

## 4. Umsetzung

Zwei Teile, wie am 2026-10-06 vereinbart.

### Teil 1 (2026-10-12, umgesetzt): synchroner Pfad

Modul `crates/spark-server/src/compaction/`:

| Datei | Inhalt | Anforderung |
| --- | --- | --- |
| `canonical.rs` | kanonisches JSON einer Nachricht: sortierte Keys, kein Whitespace, NFC; `content` immer `{images:[blake3(hex) der rohen Referenz], text}`; `tool_calls` mit kanonisch re-serialisierten Argumenten; `reasoning_content` ist nicht Teil des Schlüssels | CMP-1, Entscheidung 4 |
| `chain.rs` | `m_0 = H("atlas-msg-v1" ‖ model)`, `m_j = H(domain ‖ m_{j-1} ‖ canonical_j)`; BLAKE3-128 über `block_keys::chain_step` mit eigener Domäne | CMP-1, CMP-12 |
| `entry.rs` | JSON-Eintrag `{version, key, model, policy, head_len, cut, replaced_messages, replaced_tokens, summary, pred, created_at}` | CMP-11 |
| `index.rs` | RAM-Index (LRU, 4096 Einträge) vor den `.cmp`-Records des aktiven Block-Stores; jeder Treffer aktualisiert `last_access`; TEL-8-Zähler | CMP-11, TEL-8 |
| `head.rs` | Kopf = führende system/developer + erste user-Nachricht | CMP-4 |
| `plan.rs` | Schnittwahl: minimaler Tail ≥ `tail_pct`, verkleinert bis er ins Budget passt; nie zwischen Tool-Call und Tool-Ergebnis; die neueste Einheit bleibt immer | CMP-5 |
| `effective.rs` | Kopf + Abschnitt `<conversation_summary>` an der System-Nachricht (oder neue System-Nachricht) + Nachrichten nach dem Schnitt | CMP-2, CMP-6, CTX-3 |
| `summarize.rs` | Zusammenfassung auf dem Arena-Modell (Temperatur 0, `summary_max_tokens`, eingebauter Prompt oder Datei); kumulativ über vorherige Zusammenfassung; passt die Eingabe nicht in ein Fenster, wird in mehreren Runden gefaltet statt geschnitten | CMP-3, CMP-7 |
| `policy.rs` | `X-Atlas-Context-Policy` als Request-Extension (Middleware), ungültiger Wert ⇒ 400 `invalid_context_policy` | CTX-1 |
| `request.rs` | pro Request: Kette, angewendeter Eintrag, neue Einträge | CMP-2, CMP-10 |

Pipeline (`api/chat/`): `chat_completions_inner` berechnet die Kette, wendet
den jüngsten passenden Eintrag an, rendert den Prompt (`prepare.rs`) und
prüft die Fit-Regel. Bei Überlauf: `reject` ⇒ 400 (`overflow.rs`), sonst
synchrone Kompaktierung, zweites Rendern, dann `finish.rs`. Streaming-Clients
bekommen währenddessen `: compacting`-Keep-alives und nach Abschluss den
Kommentar `: compacted id=<hex>` (`queued_sse::respond_while_compacting`),
weil der Antwortkopf schon gesendet ist. Nicht-streamende Antworten und
`/v1/messages` tragen `X-Atlas-Compacted` und `X-Atlas-Compaction-Id`
(Anthropic-Stream-Wrapper reicht alle `x-atlas-*`-Header durch).

Block-Store: `Kind::Cmp` (`.cmp`, Tag 4); der GC löst `pred` typbewusst auf
(`.cmp` ⇒ `.cmp`, sonst `.kvb`); `touch_chain` wandert auch über `.cmp`.

Abweichungen und Festlegungen:

- **Fit-Regel ohne Slots:** `prompt + min(max_tokens, --compaction-output-reserve) ≤ max_seq_len`.
  Die Policy `reject` behält die alte Regel (`prompt < max_seq_len`, Ausgabe geklemmt).
- **Anwenden statt Vorhalten:** Ein synchron erzeugter Eintrag gilt ab sofort für jeden
  Folge-Request (CMP-2); CMP-9 („erst anwenden, wenn es nicht passt") betrifft nur die
  proaktiv vorbereiteten Einträge aus Teil 2.
- **CTX-6:** Meldung nennt Kopf, neueste Nachricht, Zusammenfassung und Reserve in Tokens.
- **Bilder** werden in der Zusammenfassung nur als „[image attached]" erwähnt (T-CMP-19 gestrichen).
- Die alte `--auto-compact`-Heuristik (ohne Modell, Trunkierung der Mitte) ist entfernt.
- Das vorherige Team hatte Auto-Kompaktierung als Quelle von Agent-Schleifen abgeschaltet
  (synthetische Fortsetzungen, Mitte-Trunkierung). E7 arbeitet mit stabilem Schnitt,
  Zusammenfassung an der System-Nachricht und sichtbaren Headern; die Stabilität mit
  Agent-Clients (opencode) ist on-box zu beobachten.

Tests: `cargo test -p spark-server compaction::` (Testvektoren, Kette, Kopf,
Schnittwahl, Effektiv-Kontext, Eintrag, Index, Policy, Fit-Regel),
`cargo test -p spark-storage` (`.cmp`-GC), `tests/test_compaction.py` on-box.

### Teil 2 (offen): proaktive Hintergrund-Jobs

- RAM-Retention der letzten Konversation pro Session (256 MB LRU) als Eingabe
  des Hintergrund-Jobs (Entscheidung 1); Job nur bei leerer Arena-Queue,
  Abbruch bei jeder Arena-Ankunft und jeder Wechselentscheidung (SCH-10,
  Entscheidung 2); Job-Sequenzen nicht in `scheduler_load()` zählen.
- CMP-9: vorbereitete Einträge mit `applied=false`, die erst beim Überlauf greifen.
- WAIT-3: Kompaktierungsdauer in der Wartezeit-Schätzung (nicht-streamend ⇒ 503).
- CTX-5 `GET /v1/atlas/compactions/{id}`, `session/state`-Fragment, TEL-8 vollständig.

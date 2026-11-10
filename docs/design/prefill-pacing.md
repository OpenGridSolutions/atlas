# Prefill-Taktung: zeitbasierte Prefill-Stücke und Decode-Anteil

Stand: 2026-11-10 — umgesetzt und mit Unit-Tests geprüft, **auf der Box noch
nicht gemessen** (§8). Im Server standardmäßig aus; im GB10-Deployment über
`.devpod/atlas-gb10.env` mit 500 ms / 50 % eingeschaltet.

## 0. Kurzfassung

Solange eine Session generiert und gleichzeitig ein neuer Prompt eingelesen
wird (Prefill), teilen sich beide die GPU. Bisher lief der Prefill in festen
Stücken von bis zu 8192 Tokens, das erste Stück allein und blockierend — die
laufenden Sessions standen so lange still.

Die Taktung ersetzt das durch zwei Betreiber-Werte:

| Flag | `serve.sh` / Env | Bedeutung |
|---|---|---|
| `--prefill-max-pause-ms` | `ATLAS_PREFILL_MAX_PAUSE_MS` | Längste Pause, die ein Prefill-Schritt den laufenden Sessions zumutet. 0 = Taktung aus (Default). |
| `--prefill-decode-share` | `ATLAS_PREFILL_DECODE_SHARE` | GPU-Zeitanteil in Prozent, den die laufenden Sessions während eines Prefills behalten. Default 50. |

Daraus folgt:

1. **Stückgröße aus Zeit statt aus Tokens.** Der Server misst pro Modell, was
   ein Prefill-Schritt kostet, und wählt die Stückgröße so, dass ein Schritt
   etwa die eingestellte Pause dauert.
2. **Takt als Zeitanteil.** Nach jedem Prefill-Stück bekommen die laufenden
   Sessions reine Decode-Schritte, bis ihr Anteil erreicht ist. Bei 100 wartet
   der Prefill, bis niemand mehr generiert; bei 0 läuft in jeder
   Scheduler-Runde ein Prefill-Stück.
3. **Mitfahren bleibt.** Die mittleren Stücke laufen im gemeinsamen Durchlauf
   mit dem Decode-Schritt (`mixed_forward`), die Gewichte werden also nur
   einmal durchlaufen.
4. **Das erste Stück gehört dazu.** Es wird nicht mehr sofort und allein
   gerechnet, sondern wie jedes andere eingetaktet.

Die Taktung greift nur, wenn Prefill und Decode gleichzeitig anstehen. Ein
Prefill ohne laufende Session behält seine vollen Stücke. Mehrere wartende
Prompts kommen nacheinander dran; die Reihenfolge bestimmt weiter die
Zulassung (`--scheduling-policy`).

## 1. Ausgangslage

Der Scheduler ist eine einzelne Schleife (`scheduler/mod.rs::run`). Pro Runde
nimmt er neue Anfragen an, setzt laufende Prefills fort und macht einen
Decode-Schritt für alle generierenden Sessions. Was die laufenden Sessions als
Stocken sehen, ist die Dauer einer Runde.

- Das erste Stück eines neuen Prompts („chunk 0") wurde bei der Annahme sofort
  gerechnet — bis zu `--max-prefill-tokens` (8192) Tokens ohne Decode-Schritt
  dazwischen.
- Die Stückgröße war eine Token-Zahl. Wie lange ein Stück dauert, hängt aber
  vom Modell ab; dieselbe Zahl bedeutet auf einem kleinen Modell eine kurze
  und auf einem großen eine lange Pause.
- Die Policy (`slai`) kannte ein Zeitziel, das zu Mini-Stücken führte. Eine
  Messung vom 2026-06-24 (Kommentar in `scheduling_policy.rs`) zeigte, warum
  das nicht funktioniert: Ein gemeinsamer Schritt hat einen festen Sockel von
  rund 250 ms (ein voller Durchlauf über die Gewichte), unabhängig von der
  Stückgröße. 32-Token-Stücke machten den Prefill 4,6-mal langsamer, ohne die
  Pause zu verkürzen.

## 2. Stückgröße: der Regler

`crates/spark-server/src/prefill_pacer.rs`, reine Rechenlogik ohne Uhr und
ohne GPU-Zugriff. Der Scheduler misst jede Runde und meldet Dauer und
gerechnete Tokens.

**Kostenmodell.** `Dauer = Sockel + Kosten_pro_Token × Tokens`. Beide Größen
werden aus den gemessenen Runden geschätzt (gleitende Regression, neue Messung
mit Gewicht 0,2).

- Der Sockel lässt sich erst bestimmen, wenn unterschiedlich große Stücke
  gemessen wurden (Streuung ≥ 15 % des Mittelwerts). Bis dahin gilt die
  gemessene Dauer eines reinen Decode-Schritts als Sockel.
- Kurze Reststücke (weniger als die halbe aktuelle Stückgröße) zählen nicht
  als Messung.

**Ziel.** `Zieldauer = max(Pause, 2 × Sockel)`,
`Stückgröße = (Zieldauer − Sockel) / Kosten_pro_Token`.

Liegt die eingestellte Pause unter dem, was das Modell kann (Sockel größer als
die halbe Pause), zielt der Regler auf den doppelten Sockel statt auf
Mini-Stücke, und im Log steht einmalig:

```
prefill pacer: --prefill-max-pause-ms 200 is below what this model can do
(fixed cost per prefill step ≈ 250 ms); pacing at ≈ 500 ms per step
```

**Grenzen.**

| Größe | Wert |
|---|---|
| Startgröße vor der ersten Messung | 512 Tokens |
| Kleinstes Stück | 128 Tokens |
| Größtes Stück | `--max-prefill-tokens` |
| Wachstum | höchstens Verdopplung pro Schritt |
| Kürzen | sofort; dauert ein Schritt mehr als das Doppelte des Ziels, wird direkt proportional gekürzt |
| Ausrichtung | auf KV-Blockgrenzen (`--block-size`) |

**Rechenbeispiel** (Sockel 250 ms, 0,13 ms pro Token, Pause 500 ms): Der
Regler landet bei rund 1920 Tokens pro Stück. Ohne Sockel wären es bei
gleichen Token-Kosten 3840.

Die Messwerte gehören zum Modell. Bei einem Modellwechsel fängt der Regler neu
an.

## 3. Takt: der Decode-Anteil

Der Regler führt ein Zeitkonto für die laufenden Sessions. Nach einer Runde
mit Prefill-Stück der Dauer `t` schuldet er ihnen

```
Schuld += (Anteil × t − Decode-Schritt) / (1 − Anteil)
```

an reiner Decode-Zeit; der Decode-Schritt, der im Prefill-Schritt mitgefahren
ist, ist bereits abgezogen. Solange eine Schuld offen ist, läuft kein
Prefill-Stück, nur Decode-Schritte. Jeder davon zahlt seine Dauer ab.

- **50 %** und ein 500-ms-Stück bei 100 ms Decode-Schritt: drei reine
  Decode-Schritte, dann das nächste Stück.
- **100 %**: Der Prefill wartet, bis keine Session mehr generiert. Bei
  Dauerlast kann das beliebig lange dauern.
- **0 %**: In jeder Runde ein Prefill-Stück; die Sessions bekommen nur den
  mitfahrenden Decode-Schritt.
- Guthaben wird nicht angespart: Decode-Zeit ohne offenen Prefill verschafft
  einem späteren Prefill keinen Vorsprung.
- Generiert niemand mehr, ist das Konto leer und der Prefill läuft mit vollen
  Stücken weiter.

Unter Taktung ersetzt das Zeitkonto die Prefill-Entscheidung der
Scheduling-Policy (`should_prefill`, `prefill_slice_budget`). Die Policy
bestimmt weiter, in welcher Reihenfolge Anfragen zugelassen werden.

## 4. Welches Stück welchen Weg nimmt

Der gemeinsame Durchlauf (`mixed_forward`) kann weniger als der normale
Prefill-Pfad (`prefill_chunk`): kein Nachschlagen im Präfix-Cache, kein
Überspringen bereits gecachter Tokens, kein Checkpoint am Prompt-Ende, kein
Eintrag in den Präfix-Cache. Deshalb:

| Stück | Weg | Grund |
|---|---|---|
| Erstes Stück | normaler Pfad, in derselben Runde wie ein Decode-Schritt | Präfix-Cache wird hier nachgeschlagen |
| Stücke im gecachten Präfix | normaler Pfad | überspringt gecachte Tokens, spielt bei SSM-Modellen das Zustandsfenster nach |
| Mittlere Stücke | gemeinsamer Durchlauf | Gewichte nur einmal durchlaufen |
| Letztes Stück | normaler Pfad | Checkpoint am Prompt-Ende, Eintrag in den Präfix-Cache, erstes Token |

Einzelheiten der Planung (`plan_chunk`):

- Gecachte Tokens kosten keine Rechenzeit. Ein Stück umfasst deshalb den
  gecachten Teil plus die Stückgröße an echter Arbeit, höchstens
  `--max-prefill-tokens`.
- Der normale Pfad legt den Checkpoint an, indem er das letzte Stück an einer
  Blockgrenze teilt. Ein Stück, das in diesem Endbereich enden würde, wird
  deshalb zum letzten Stück verlängert oder endet davor.
- Zwischenstücke enden auf einem Vielfachen von 4 Tokens (Vorgabe der
  SSM-Schichten).
- Der gemeinsame Durchlauf hat einen begrenzten Puffer; die Stückgröße ist
  zusätzlich durch `max_batch_tokens` minus Decode-Plätze begrenzt.

Stehen mehrere Prefills an, während Sessions generieren, wird der älteste
zuerst fertig eingelesen; die anderen warten.

Mit `--speculative`/MTP gibt es keinen gemeinsamen Durchlauf. Die Stücke
laufen dann alle über den normalen Pfad, Stückgröße und Zeitkonto gelten
trotzdem.

## 5. Ausnahmen

Für diese Fälle bleibt alles wie ohne Taktung:

- **EP-Betrieb** (zwei Knoten): kein gemeinsamer Durchlauf.
- **MLA-Modelle** (Mistral-Small-4): Der Prefill muss in einem Stück laufen,
  weil es keinen seitenweisen MLA-Prefill-Kernel gibt.
- **Prompts mit Bildern**: Die Bild-Einbettung zählt Platzhalter-Tokens pro
  Stück (`prefill_b/embed_chunk.rs`); andere Stückgrenzen würden ein Bild
  zerschneiden. Das erste Stück eines Bild-Prompts läuft deshalb weiter
  sofort und allein, auch bei 100 %.
- **Ohne Chunked Prefill** (`--max-prefill-tokens 0`) ist die Taktung
  wirkungslos.

## 6. Einstellungen

```bash
# Server direkt
spark serve … --prefill-max-pause-ms 500 --prefill-decode-share 50

# Deployment: .devpod/atlas-gb10.env (serve.sh reicht die Werte durch)
ATLAS_PREFILL_MAX_PAUSE_MS=500
ATLAS_PREFILL_DECODE_SHARE=50
```

Abschalten: `ATLAS_PREFILL_MAX_PAUSE_MS` auskommentieren oder auf 0 setzen.
Beim Start steht im Log `Prefill pacing: max pause 500 ms, decode share 50 %
(chunk ≤ 8192 tokens)`. Mit `RUST_LOG=info,spark::prefill_pacer=debug` zeigt
`prefill pacer: chunk A → B tokens (…)` jede Änderung der Stückgröße.

Richtwerte: Eine kürzere Pause macht laufende Streams flüssiger und den
Prefill langsamer (mehr Schritte, jeder mit Sockel). Ein höherer Decode-Anteil
verschiebt Zeit vom neuen Prompt zu den laufenden Sessions; die Wartezeit bis
zum ersten Token des neuen Prompts wächst entsprechend.

## 7. Zwei Korrekturen am gemeinsamen Durchlauf

Beide wirken auch bei abgeschalteter Taktung, sobald ein Prefill-Stück
zusammen mit einem Decode-Schritt läuft:

1. **Stücke im gecachten Präfix laufen nicht mehr gemeinsam.** Der gemeinsame
   Durchlauf kennt keinen Cache-Treffer. Er hat solche Stücke neu gerechnet
   und bei SSM-Modellen den Zustand ein zweites Mal weitergeschoben
   (`run_standard.rs`, `fuse_ok`).
2. **Der gemeinsame Durchlauf führt `kv_valid_tokens` mit.** Der Eintrag in
   den Präfix-Cache am Prompt-Ende reicht nur so weit, wie dieser Zähler
   geht; vorher endete er am ersten gemeinsam gerechneten Stück
   (`decode_b.rs`, Schritt 8).

Beides sind Befunde aus dem Code. Auf der Box wurde der Fehler nicht
nachgestellt.

## 8. Prüfstand und offene Punkte

**Geprüft**

- `prefill_pacer_tests.rs`, 30 Tests: Regler (Zielgröße mit und ohne Sockel,
  Grenzen, Wachstum, verrauschte Messwerte), Zeitkonto (25/50/75 %, 0 %,
  100 %, Leerlauf, wechselnde Batchgröße), Stückplanung (erstes Stück,
  Cache-Präfix, SSM-Fenster, Prompt-Ende, Blockgrenzen) und ein Durchlauf
  eines 40 000-Token-Prompts mit halb gecachtem Präfix, bei dem kein Schritt
  länger als 0,55 s dauert.
- `main_modules/tests.rs::test_cli_parse_prefill_pacing`: Defaults, Parsen,
  Ablehnung von Anteilen über 100.

**Nicht geprüft**

- Die Verdrahtung in der Scheduler-Schleife hat keinen automatischen Test (es
  gibt kein Modell-Mock für die Schleife).
- Kein Lauf auf der Box. Offen pro Modellfamilie:
  1. Pause eines laufenden Streams, während ein langer Prompt ankommt, mit und
     ohne Taktung.
  2. Zeit bis zum ersten Token des neuen Prompts, mit und ohne Taktung.
  3. Gleiche Ausgabe bei Temperatur 0 mit und ohne Taktung. Wichtig bei den
     SSM-Hybriden (Qwen3-Next-80B): Kleinere Stücke bedeuten häufigeres
     Normalisieren des SSM-Zustands.
  4. Präfix-Cache-Treffer nach einem getakteten Prefill (zweite Anfrage mit
     gleichem Präfix).

**Bekannte Grenzen**

- Das Zurückstellen des ersten Stücks war bisher nur hinter
  `ATLAS_HOLO_ALWAYS_MIXED` im Einsatz und ist entsprechend wenig erprobt.
- Gemeinsam gerechnete Stücke legen keine Zwischen-Checkpoints im
  Präfix-Cache an (wie bisher); das tun nur Stücke auf dem normalen Pfad. Neu
  ist, dass das erste Stück kleiner ist (Startgröße 512 statt bis zu 8192
  Tokens): Zwischen seinem Checkpoint und dem am Prompt-Ende liegt bei einem
  kalten Prompt unter Last keiner mehr.
- Größtes Stück und Blockgröße stammen vom Boot-Modell und werden bei einem
  Modellwechsel nicht neu bestimmt.
- Die Schätzung der Bearbeitungszeit (`accepted_processing_time`,
  `docs/design/deadline-admission.md`) rechnet die Taktung nicht ein: Ein
  Prefill unter Last dauert länger als dort angenommen.

## 9. Fundstellen

| Datei | Inhalt |
|---|---|
| `crates/spark-server/src/prefill_pacer.rs` | Regler, Zeitkonto, Stückplanung |
| `crates/spark-server/src/prefill_pacer_tests.rs` | Tests dazu |
| `crates/spark-server/src/scheduler/mod.rs` | Messung der Runde, Reset beim Modellwechsel, Ausnahme EP/MLA |
| `crates/spark-server/src/scheduler/phase_continue_prefills.rs` | Zeitkonto als Tor vor dem Prefill-Schritt, Puffergrenze `fuse_cap` |
| `crates/spark-server/src/scheduler/phase_continue_prefills/run_standard.rs` | Stückplanung, Wahl des Wegs, Meldung der gerechneten Tokens |
| `crates/spark-server/src/scheduler/types.rs` | `PrefillInProgress.has_media` (Bild-Prompts) |
| `crates/spark-model/src/model/trait_impl/decode_b.rs` | `kv_valid_tokens` im gemeinsamen Durchlauf |
| `crates/spark-server/src/cli/serve_args.rs`, `main_modules/serve.rs` | Flags, Aufbau des Reglers |
| `serve.sh`, `.devpod/atlas-gb10.env` | Deployment-Werte |

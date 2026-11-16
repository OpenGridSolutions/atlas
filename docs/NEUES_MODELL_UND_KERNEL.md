# Neues Modell, neue Kernel: der Ablauf

Wie ein Modell, für das Atlas noch keine Kernel hat, vom Vorschlag bis zum
Eintrag in der `MODEL.toml` kommt. Die sechs Schritte laufen in dieser
Reihenfolge; jeder hat ein Ergebnis, ohne das der nächste nicht beginnt.

Durchgespieltes Beispiel: Qwen3.8-Flash-Next,
[`design/qwen4-exp-port.md`](design/qwen4-exp-port.md). Die technischen
Handgriffe (Loader-Arm, Target-Verzeichnis) stehen in
[`AGENTS.md`](../AGENTS.md), „Adding a new model"; das Speichermodell in
[`design/memory-model.md`](design/memory-model.md).

| Schritt | Ergebnis |
|---|---|
| 1. Modellauswahl | Entscheidung für genau einen Checkpoint |
| 2. Eigene Analyse | Plan in `docs/design/<modell>-port.md` |
| 3. Abgleich mit den Referenz-Engines | nachgeschärfter Plan, Entscheidungen des Betreibers |
| 4. Kernel | Kernel-Target mit Microtests gegen CPU-Referenz |
| 5. Loader und Schichten | Modell lädt und antwortet auf der GB10 |
| 6. Messung | `[serving]`, `[estimate]`, Zeile in der Messwert-Tabelle |

## 1. Modellauswahl

Geprüft wird ein konkretes Hugging-Face-Repository, nicht eine Modellfamilie:
Quantisierungs-Varianten desselben Modells unterscheiden sich in Größe und
Format.

Ohne Download lässt sich klären (Header per HTTP-Range: 8 Byte Länge, dann
JSON):

- **Passt es?** Summe der Tensoren gegen das Budget (Gerätespeicher ×
  `--gpu-memory-utilization`, auf der GB10 107,7 GiB). Was bleibt für KV-Pool
  und Reserve?
- **Wie schnell kann es werden?** Decode ist auf der GB10 durch die
  Speicherbandbreite begrenzt. Maßgeblich sind die je Token gelesenen
  Gewichts-Bytes (aktive Parameter × Bytes je Parameter), nicht die
  Gesamtgröße.
- **Was ist neu?** `model_type` gegen `crates/spark-model/src/factory.rs`,
  Tensor-Dtypes gegen `WeightDtype::from_safetensors`, Quantisierungsformate
  gegen die vorhandenen Kernel.
- **Lizenz** des Checkpoints und des Basismodells.

Ergebnis ist eine Empfehlung mit Zahlen. Die Wahl trifft der Betreiber.

## 2. Eigene Analyse

Bevor fremde Implementierungen gelesen werden, entsteht ein eigenes Bild aus
drei Quellen: `config.json`, dem Tensor-Inventar aus den Headern und der
Referenz-Mathematik (in der Regel `modeling_*.py` in HF transformers).

Der Plan in `docs/design/<modell>-port.md` enthält:

- die Bausteine des Modells und zu jedem, ob Atlas ihn hat, parametrisieren
  kann oder neu bauen muss;
- die Gewichtsformate. Grundsatz: Gewichte werden in dem Format gerechnet, in
  dem sie vorliegen; jede Umwandlung beim Laden ist Speicher, den das Budget
  nicht kennt;
- die Speicherrechnung (Gewichte, KV je Token, Zustand je Sequenz);
- den Zustand je Sequenz neben dem KV-Cache und was davon gesichert werden
  muss (Präfix-Cache, NVMe-Tier, Kompaktierung, Rollback);
- die Aufteilung in PRs und was sich jeweils wie prüfen lässt;
- die Entscheidungen, die der Betreiber treffen muss.

Annahmen über Formate werden an echten Bytes des Checkpoints geprüft, bevor
Code darauf aufbaut (Beispiel: csf-Skalen und N-Gramm-Konstanten in §2 des
Flash-Next-Plans).

## 3. Abgleich mit den Referenz-Engines

Eingang ist der Hugging-Face-Link des Checkpoints. Gelesen werden

- die Modellkarte und die des Basismodells (Start-Rezepte, Sampling,
  Denk-Schalter, Kontextgrenzen),
- die Diskussionen (Speicherprobleme, Messwerte auf derselben Hardware),
- die Implementierungen der dort genannten Engines (vLLM, SGLang, FlashInfer).

Aus den Engines wird Verhalten übernommen, kein Code: Die Quellen sind meist
Apache-2.0, Atlas ist AGPL-3.0-only. Jeder Befund wird eingeordnet:

- **Pflicht** — ohne das rechnet das Modell falsch (Reihenfolgen,
  Rundungs- und Randfälle, Zustand, der mitgeführt werden muss);
- **Tempo** — Fusionen, Kernel-Varianten, Ladewege;
- **bewusst anders** — mit Begründung.

Der Plan wird danach nachgeschärft, und die offenen Entscheidungen gehen an
den Betreiber. Die Messwerte der anderen Engines auf derselben Hardware sind
die Messlatte für Schritt 6.

## 4. Kernel

Ein Target je Modell: `kernels/gb10/<modell>/MODEL.toml` und
`kernels/gb10/<modell>/<quant>/` mit `KERNEL.toml` und den `.cu`-Dateien.
Gemeinsame Kernel kommen aus `kernels/gb10/common/`; eine gleichnamige Datei
im Target ersetzt sie. Jede Quelldatei beginnt mit der SPDX-Zeile.

Je Kernel:

1. **CPU-Referenz zuerst.** Ein Microtest unter
   `crates/spark-model/examples/<name>_microtest.rs` rechnet die Formel in
   f64 nach und vergleicht. Die Toleranz ergibt sich aus dem Ausgabeformat
   (BF16: ein Rundungsschritt je Element), nicht aus dem, was gerade besteht.
2. **Echte Gewichte.** Der Test nimmt optional das Snapshot-Verzeichnis und
   liest die betroffenen Tensoren aus dem Checkpoint. Zufallsgewichte allein
   decken Wertebereiche und Formen nicht ab.
3. **Ein Target, eigenes Build-Verzeichnis.** Die Produktions-Binary in
   `/build/target` bleibt unberührt:

   ```bash
   CARGO_TARGET_DIR=/build/target-<modell> ATLAS_TARGET_MODEL=<modell> \
   ATLAS_TARGET_QUANT=<quant> nice -n 10 cargo build --release -j 6 \
     -p spark-model --features cuda,gpu-examples --example <name>_microtest
   ```

4. **Erst richtig, dann schnell.** Eine Variante je Token (GEMV) für Decode
   genügt für den ersten Lauf; die Stapel-Variante für Prefill ist ein
   eigener Kernel mit eigenem Test.

Builds und GPU-Tests laufen nur, wenn auf dem Server kein Modell geladen ist
(`/internal/state`, `MemAvailable` ≥ 60 GiB).

## 5. Loader und Schichten

Registrierstellen, die ein neues Modell berührt:

- `crates/atlas-core/src/config/dispatch.rs` (Config-Arm),
- `crates/spark-model/src/factory.rs` und `weight_loader/`,
- `crates/spark-server/tool_defaults.toml` (Tool- und Reasoning-Parser),
- `crates/atlas-kernels/tests/serving_parse.rs` (Liste der Targets),
- `serve.sh` und `.devpod/atlas-gb10.env` (Preseed-Listen),
- die Tabellen in `MODELS_AND_ENDPOINTS.md`, `GB10_DEPLOYMENT_GUIDE.md` §2
  und `README.md`.

Der Loader prüft das Inventar (Name, Form, Dtype) und bricht bei Abweichung
ab. Zahlenpfade werden mit dem echten Modell geprüft, bevor sie als fertig
gelten.

## 6. Messung und `MODEL.toml`

Erst messen, dann eintragen. Vor dem ersten Laden wird der Bedarf gerechnet;
geladen wird einzeln, und unter 6 GiB freiem Speicher wird abgebrochen.

**Speicherbelegung.** Der Server schreibt beim Laden vier Zeilen
(`memory-model.md` §8). Abzulesen sind:

- `Atlas-own` — Gewichte, abgeleitete Kopien, Puffer;
- `KV pool` und die Zahl der KV-Tokens;
- `after the KV pool` gegen `reserve estimate` — der erste Wert muss unter
  dem zweiten liegen;
- im Betrieb `late_alloc_mib` (höchstens 64) und der kleinste Wert von
  `MemAvailable` während eines langen Prefills.

**Sessions.** Aus dem gemessenen Pool folgt die Regelzahl
(`memory-model.md` §4):

```
Regel-Sessions = aufrunden(nutzbare KV-Tokens / natives Fenster)
Fenster        = min(natives Fenster, abrunden(nutzbare Blöcke / sessions) × 16)
```

Dazu kommt der Zustand je Sequenz außerhalb des KV-Pools (SSM-Zustand und
was das Modell sonst mitführt); er steckt in der Reserve und wächst mit
`--max-batch-size`. `[serving].sessions` muss mindestens `--max-batch-size`
sein. Mehr Sessions als die Regelzahl verkleinern das Fenster je Session.
Die Zahl legt der Betreiber fest.

**Laufzeit.** `tests/estimate_calibration.py` misst Prefill-Rate und
Decode-Schritt und liefert den `[estimate]`-Block.

Eingetragen wird in `kernels/gb10/<modell>/MODEL.toml`: `[serving].sessions`,
`[estimate]`, `[behavior]` (Denk-Standard, KV-Dtype, Thinking-Budget). Die
Messwerte kommen als Zeile in `memory-model.md` §7, und der Vergleich mit den
Werten der Referenz-Engines aus Schritt 3 in den Plan.

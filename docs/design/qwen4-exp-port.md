# Qwen3.8-Flash-Next (`qwen4_exp`) in Atlas

Stand: 2026-11-10 — Planungsstand, noch kein Code. Ziel-Checkpoint:
`nvidia/Qwen3.8-Flash-Next-FP4-Local` (Revision `b9ce18a6`, 21 Shards,
95,1 GiB).

## 0. Kurzfassung

- Das Modell ist **kein größeres 80B**. Von den Bausteinen des 80B bleiben
  GDN-Schichten, gegatete Attention und der MoE-Block; neu sind vier
  Residualströme mit gelernter Mischung, eine gehashte N-Gramm-Tabelle,
  Attention mit Blockauswahl (QSA-Indexer), zwei neue Gewichtsformate und ein
  anderer MTP-Kopf.
- Atlas lehnt den Checkpoint heute an drei Stellen ab: Config-Parser
  (`text_config` verschachtelt), Tensor-Dtype `U32`, Loader-Fabrik
  (`qwen4_exp` unbekannt). Der Lade-Preflight (Gewichte × 1,3–1,5) käme als
  vierte dazu.
- Der Speicher ist knapp: 95,1 GiB Gewichte bei 107,7 GiB Budget. Davon sind
  26,9 GiB die N-Gramm-Tabelle, von der ein Token 16 Zeilen (≈ 1,4 KB) liest.
- Umsetzung in neun PRs (§5a). PR 1 bis 3 brauchen kein geladenes Modell.

## 1. Was das Modell rechnet

Quelle der Formeln: Referenzimplementierung in HF transformers
(`models/qwen4_exp/`) und vLLM (`vllm/models/qwen4_exp/`), beide Apache-2.0.
Atlas implementiert nach diesen Formeln neu und übernimmt keinen Code.

| Baustein | Inhalt | Verhältnis zu Atlas |
|---|---|---|
| Residualströme | 4 Ströme × 2560. Vor jedem Block: Gruppen-RMSNorm, Gewichte `sigmoid(Up(silu(Down(n)/4)))` über einen 320er-Engpass, Mittel über die Ströme. Nach dem Block: `strom_i += block · 2·sigmoid(Inject(n)/4)_i`. Am Ende ein Mischer ohne Inject, dann `lm_head`. Keine weiteren Layernorms. | DeepSeek-V4 hat Ströme (`hyper_connection.cu`), aber andere Mathematik (ein RMS, Skalar je Strom, Sinkhorn). Neuer Kernel. |
| GDN (36 Schichten) | Wie 80B, aber 16 Key-/48 Value-Köpfe (Faktor 3), Ausgangs-Gate `sigmoid`, Projektionen getrennt (`in_proj_qkv`, `_z`, `_a`, `_b`). | `Qwen3SsmLayer`-Kernel wiederverwendbar (Kopf-Dim 128 passt), Loader und Gate-Art neu. |
| Attention (12 Schichten) | 24 Köpfe × 256, 2 KV-Köpfe, Query + Gate je Kopf, MRoPE auf 64 Dimensionen. | Gegatete Attention existiert. |
| QSA-Indexer | Je Token ein Index-Key (128). Sichtbare Tokens in 4er-Blöcken, Blockschlüssel = Mittel der 4 Keys. `score = Σ_köpfe relu(q·k)/√128`, die 512 besten Blöcke plus der unvollständige Rest sind sichtbar. Bis 2048 Tokens ist das volle Attention. | Fehlt vollständig (Schlüsselablage, Bewertung, Top-512, Attention über eine Auswahl). |
| MoE | 512 Experten, Top-10, Breite 640, ein geteilter Experte (640). | `MoeLayer` passt in den Grenzen (512/32). Geteilter Experte hat ein anderes Format als die gerouteten. |
| N-Gramm-Einbettung (PLE) | Nur an Schicht 2 (`layers.1.ple`). 16 Hash-Köpfe über das aktuelle und die 2 vorigen Tokens → 16 Zeilen × 160 = 2560. Gate je Strom, dann Tiefen-Faltung (Kern 4, Dilatation 3). | Fehlt vollständig. |
| MTP-Kopf | Eigene Schicht (Attention + MoE + Ströme), Eingang ist der Mehrstrom-Zustand vor dem Endmischer. | Anderer Aufbau als `MtpHead`; Weg über `DraftProposer` wie bei DeepSeek-V4. |
| Vision | 27 Blöcke, 0,30 GiB. | Eigenes Format (MXFP8/NVFP4), später. |

## 2. Gewichtsformate

| Teil | Format im Checkpoint | Atlas heute |
|---|---|---|
| Geroutete Experten (Decoder) | NVFP4, Blockskalen **csf-komprimiert** (`nvfp4_csf_fixed` U8, `nvfp4_csf_exceptions` U32), `weight_scale_2`, `input_scale` | NVFP4 ja, csf nein, `U32` wird abgelehnt |
| GDN, Attention, Indexer, geteilter Experte | **MXFP8**: E4M3-Werte, ein Exponenten-Byte je 32 Spalten | fehlt (FP8 nur mit 128×128-Blockskalen) |
| N-Gramm-Tabelle | NVFP4, 128 Teile × 2 500 012 Zeilen × 80 Byte + Skalen | fehlt |
| Router, Normen, Mischer, `embed_tokens`, `lm_head`, MTP-Attention | BF16/F32 | vorhanden |
| MTP-Experten | NVFP4 mit gewöhnlichen Skalen | vorhanden (`quantized`) |

csf: je 16 Zeilen ein Block aus 16 Basis-Bytes und den 4-Bit-Codes der Zeilen
(zwei je Byte, niedriges Nibble zuerst); Skalen-Byte = Basis + Code. Ausnahmen:
untere 24 Bit = `zeile·spalten + spalte`, obere 8 Bit = das exakte Byte.

Am Checkpoint geprüft (2026-11-10, Schicht 0, Experte 0, per HTTP-Range):
- csf-Dekodierung von `gate_proj` und `down_proj`: Formen stimmen, Ausnahmen
  aufsteigend und alle außerhalb des Zeilenfensters, Bytes 86–126 (gültige
  positive E4M3-Werte).
- N-Gramm-Konstanten (Multiplikatoren, 16 Primzahl-Größen, Offsets) aus der
  Referenzregel nachgerechnet und mit den I64-Puffern im Checkpoint
  verglichen: identisch. `seed` steht nicht in `config.json`; Atlas liest die
  Puffer aus dem Checkpoint statt sie zu berechnen.

## 3. Speicher

Budget 107,7 GiB (119,7 × 0,90). Gewichte: 95,1 GiB, davon N-Gramm-Tabelle
26,9, Vision 0,30, MTP 1,49. Alles Folgende ist Rechnung, nicht Messung.

- KV je Token ≈ 14 KB (12 Schichten × 2 Köpfe × 256 × K+V in FP8 ≈ 12,3 KB,
  dazu 12 Index-Keys ≈ 1,5 KB).
- Zustand je Sequenz ≈ 113 MB GDN (36 × 48 × 128 × 128 × 4 B) + Faltungs- und
  PLE-Zustand.
- Tabelle im Speicher: für Puffer, Zustands-Pools, Reserve und KV bleiben
  ≈ 12 GiB. Tabelle auf NVMe: ≈ 39 GiB.
- Der Lade-Preflight rechnet Gewichte × 1,3 (mit FP8-Tensoren × 1,5) und
  lehnt 95 GiB damit ab. Er braucht für dieses Modell die tatsächliche
  Rechnung (keine abgeleiteten Kopien, Migration shardweise).
- Abgeleitete Kopien (transponierte Experten, entpackte Skalen) sind bei
  diesem Modell nicht bezahlbar; `moe_prefill_transpose = false`.

## 4. Entscheidungen des Betreibers

Entschieden am 2026-11-11:

1. **N-Gramm-Tabelle auf NVMe.** Gemessen auf den Checkpoint-Dateien ohne
   Cache: 0,19 ms je 4K-Zugriff, mindestens 32 000 Zugriffe/s parallel. Ein
   Token braucht 16 Zugriffe. Der Ladeweg der Tabelle ist damit PR 3 (§5a).
2. **csf-Skalen bleiben komprimiert im Speicher.** Je Schritt werden nur die
   Skalen der gerouteten Experten in einen festen Arbeitspuffer entpackt.

Offen: `[serving].sessions` (nach der ersten Messung), `--speculative`
(nach PR 6).

### Abgleich mit der Modellkarte und den Erfahrungsberichten (2026-11-11)

- Die Karte betreibt die DGX Spark ebenfalls mit Tabelle auf Platte
  (`VLLM_PLE_TABLE_MEMORY=disk`, io_uring), FP8-KV, Blockgröße 16, MTP mit
  4 Entwürfen, Parser `qwen3` / `qwen3_xml`.
- Vergleichswerte eines Spark-Nutzers mit dem vLLM-Container (Diskussion 13,
  131K Kontext, MTP an): 35,3 tok/s einzeln, 96,6 tok/s bei 8 parallelen
  Anfragen, 16K-Prompt in 5,15 s, KV-Pool 336K Tokens, Start 11 min,
  Speichereinbruch 35–60 s nach dem Laden.
- Denkmodus: `enable_thinking` (Standard an), `preserve_thinking` (Standard
  an), `reasoning_effort` (`xhigh` Standard, `medium`, `low`). Bekanntes
  Verhalten: vereinzelt endet das Denken nicht und verbraucht das ganze
  Budget (7 von 492 Läufen bei 32K Budget). Betrifft PR 4 (Template-Argumente,
  `max_thinking_budget`).
- Sampling laut Basiskarte: Denkmodus 1,0 / 0,95 / 20; ohne Denken
  0,7 / 0,80 / 20 mit `presence_penalty` 1,5.
- Kontext über 262 144 Tokens geht nur mit statischem YaRN, das kurze Texte
  verschlechtern kann. Nicht vorgesehen.
- SGLang entpackt die csf-Skalen beim Laden; die Karte nennt als Laufzeit
  nur vLLM.

## 5. Abgleich mit vLLM, FlashInfer und SGLang (2026-11-11)

Gelesen: upstream vLLM `vllm/models/qwen4_exp/`, FlashInfer (b12x, csf,
PLE-Leser), die drei offenen SGLang-PRs (#41624, #46153, #46128). Atlas
übernimmt Verhalten, keinen Code. „Pflicht" heißt: ohne das rechnet das
Modell falsch.

**Experten (MoE)**
- Pflicht: `W[r,c] = e2m1(nibble) · e4m3(skala[r, c/16]) · weight_scale_2`;
  niedriges Nibble = gerade Spalte; E2M1-Beträge 0, 0.5, 1, 1.5, 2, 3, 4, 6,
  Bit 3 = Vorzeichen.
- Die 48 Decoder-Expertenschichten sind im Checkpoint als `NVFP4` markiert
  (Aktivierungen 4 Bit, dafür `input_scale`). Beide Rezepte der Modellkarte
  erzwingen trotzdem BF16-Aktivierungen im Decode
  (`VLLM_B12X_MOE_FP4_FORCE_A16=1`); `input_scale` wird dann nicht gelesen.
  Atlas rechnet ebenso mit BF16-Aktivierungen (vorhandener W4A16-Pfad) und
  lädt `input_scale` nicht. Die 4-Bit-Aktivierung im Prefill
  (`B12X_W4A16_A4_PREFILL`) ist eine Beschleunigung, die die Ausgabe ändert;
  nicht vorgesehen.
- csf: vLLM und SGLang entpacken beim Laden, nur FlashInfer hält die Skalen
  komprimiert und entpackt je Aufruf die gerouteten Experten (ohne
  Duplikate), bei langem Prefill alle. Atlas folgt FlashInfer (§4).
- Pflicht-Prüfungen beim Laden: Basis-Bytes ≤ 240, `zeilen·spalten ≤ 2^24`,
  Ausnahmen streng aufsteigend und kleiner als `zeilen·spalten`, Slab-Breite
  `16·(1 + spalten/2)`.

**MXFP8**
- Pflicht: `wert = e4m3(byte) · 2^(skala − 127)`, eine Skala je 32 Spalten.
  Entspricht `mxfp8_gemv.cu`. SGLang wandelt die kleinen Tensoren
  (`in_proj_a/b`, 48 Zeilen) nach BF16; Atlas liest auch sie direkt.

**N-Gramm-Tabelle und PLE-Schicht**
- Pflicht: Zeile = `id // 2500012` (Teil), Rest = Zeile im Teil. Je Token
  und Kopf 80 Byte Nibbles + 10 Byte E4M3-Skalen; ein FP32-Globalwert für
  die ganze Tabelle. Dequantisierung wie bei den Experten.
- Pflicht (EOS): Vorgänger vor Position 0 sind EOS. Ist ein Vorgänger EOS,
  gelten alle älteren auch als EOS. Das aktuelle Token löst das nicht aus.
- Der N-Gramm-Kontext ist kein eigener Zustand: vLLM bildet ihn je Schritt
  aus der Token-Historie.
- Pflicht (Faltung): Der Zustand hält die **normierten Eingänge** der
  Faltung (nicht die Ausgänge), 9 Werte je Kanal, älteste zuerst; am
  Sequenzanfang Null. Mit MTP braucht das Fenster zusätzlich Platz für die
  Entwürfe.
- Platte: FlashInfer liest mit O_DIRECT und io_uring, sortiert nach Datei
  und Offset, fasst benachbarte 4K-Blöcke bis 64 KiB zusammen; kein
  Zeilen-Cache. vLLM liest gepuffert über den Seiten-Cache. Für Atlas:
  O_DIRECT (der Seiten-Cache würde den Speicher belegen, den die Auslagerung
  freimacht).

**Attention mit Blockauswahl (QSA)**
- Pflicht: Index-Keys werden **immer** gebildet und abgelegt, auch unter
  2048 Tokens. Es gibt keinen Schwellwert, der den Indexer abschaltet.
- Pflicht: Roh-Keys (ungenormt, unrotiert) der laufenden Vierergruppe liegen
  in einem Ring je Sequenz. Bei `(pos+1) % 4 == 0`: Mittel der vier in FP32
  → `k_layernorm` → RoPE an der ersten Position der Gruppe → Ablage bei
  `pos // 4`.
- Pflicht: Sichtbar für ein Query-Token sind `min((pos+1)//4, len//4)`
  Blöcke; daraus die 512 besten; der unvollständige Rest der eigenen Gruppe
  immer. Im Chunk-Prefill werden die Keys des Chunks vor der Auswahl
  geschrieben.
- Pflicht (MTP): Ring-Kapazität = 4 + Entwürfe, aufgerundet auf 4. Bei
  genau 4 überschreibt ein verworfener Entwurf einen bestätigten Key.
- Präfix-Cache: gepoolte Keys teilen den Lebenszyklus der KV-Blöcke. Die
  Blockgröße muss ein Vielfaches von 4 sein (Atlas: 16); ein Treffer liegt
  dann auf einer Gruppengrenze und braucht den Ring nicht.
- Reihenfolge der Bindungen bei Gleichstand der Scores ist in vLLM
  unbestimmt; Tests vergleichen Wertemengen.

**GDN**
- Flaches Layout wie Qwen3.5 (`qkv`, dann `z`; `b`, dann `a`), nicht das
  verschachtelte des 80B. `A_log` in Float exponentiieren.

**Residualströme**
- Reihenfolge je Schicht: PLE addieren → Mischer → GDN/QSA → Mischer → MoE.
  Kein Endnorm; der Endmischer hat kein `block_inject`.
- vLLM hält die Ströme in BF16 und rundet an jedem Kernel-Ausgang; Atlas
  hält sie in FP32. Toleranz der fusionierten vLLM-Kernel gegen die
  Referenz: 1e-2.

**MTP**
- Eingang ist der Mehrstrom-Zustand vor dem Zusammenfassen im Endmischer
  plus die Einbettung des nächsten Tokens; die MTP-Schicht hat keine PLE.
- Höchstens 4 Entwürfe (= Gruppengröße). Die Blockauswahl aus Schritt 0 wird
  in den Folgeschritten wiederverwendet.
- vLLM prüft Entwürfe probabilistisch (Rejection Sampling), nicht greedy.
- Akzeptanzlänge laut SGLang 2,5–2,8 beim Denken, 3,3–3,7 sonst.

**Laden**
- Je Shard rund 23 000 Tensoren. SGLang maß 31 s statt 3 s je Shard, wenn
  die Datei nicht erst sequentiell gelesen wird.
- Sichern/Wiederherstellen: mit dem GDN-Zustand zusammen nur der
  PLE-Faltungszustand; Index-Keys über die KV-Blöcke.

**Vergleichswerte auf der Spark (mit MTP):** vLLM 35 tok/s, SGLang
27,5 tok/s (ohne MTP 15,9).

## 5a. PRs

| PR | Inhalt | Prüfbar |
|---|---|---|
| 1 | Config-Arm `qwen4_exp` (`atlas-core/config`), Dtype `U32`, csf-Prüfungen, N-Gramm-Hash mit EOS-Regel, Inventarprüfung der Tensoren (Name, Form, Dtype; Abbruch bei Abweichung) | CPU |
| 2 | Kernel-Target `kernels/gb10/qwen3.8-flash-next/nvfp4/`: Strom-Mischer ✔, MXFP8-GEMV ✔, MXFP8-GEMM (Prefill), csf-Entpackung, PLE-Schicht, Index-Key-Pooling; je Kernel CPU-Referenz + Microtest | GPU, ohne Modell |
| 3 | N-Gramm-Tabelle von NVMe (O_DIRECT über die I/O-Grenze) | GPU, ohne Modell |
| 4 | Loader, Schichten (GDN im flachen Layout), Decode und Prefill für Text. Index-Keys werden abgelegt; Auswahl noch nicht, daher Kontext auf 2048 Tokens begrenzt (darüber Ablehnung). `MODEL.toml`, Tool-/Reasoning-Zuordnung, Denk-Schalter, Preflight. Vision- und MTP-Tensoren und `input_scale` werden nicht geladen. | erstes Laden auf der GB10 |
| 5 | Blockauswahl: Bewertung, Top-512, Attention über die Auswahl; hebt die 2048er-Grenze auf | GPU, mit Modell |
| 6 | Sitzungszustand: PLE-Faltungszustand und Roh-Key-Ring durch Sichern/Wiederherstellen, NVMe-Tier, Kompaktierung, Rollback | Sweep |
| 7 | MTP-Kopf als `DraftProposer` (Ring und Faltungsfenster um die Entwürfe erweitert) | GPU, mit Modell |
| 8 | Vision | GPU, mit Modell |
| 9 | Geschwindigkeit, Messwerte, Support-Matrix, README | — |

Bis PR 6 bleibt die Wiederverwendung von Präfixen und Sitzungen für dieses
Modell abgeschaltet, weil der neue Zustand je Sequenz sonst fehlt.

Übergreifende Änderungen (für die Reviews): `WeightDtype` (U32),
Config-Dispatch, Lade-Preflight, `Model`-Trait-Methoden für Zustand je
Sequenz (PR 6), Index-Key-Ebene im KV-Cache (PR 4). Bestehende
Kernel-Targets werden nicht angefasst; die neuen Kernel liegen im eigenen
Target.

## 5b. Stand der Umsetzung (2026-11-11)

| Teil | Stand |
|---|---|
| Config-Arm, `Qwen4ExpConfig`, Dtype `U32` | fertig, Tests mit der Config des Checkpoints |
| csf-Prüfungen, N-Gramm-Hash mit EOS-Regel | fertig, Vektoren aus unabhängiger Nachrechnung |
| Kernel: Strom-Mischer, MXFP8-GEMV, csf-Entpackung, PLE-Schicht, GDN-Gates und Ausgangsnorm, Blockauswahl (QSA) | fertig, Microtests gegen CPU-Referenz, auch mit echten Gewichten |
| Tabellen-Leser (`spark_runtime::row_table`) | fertig. io_uring ist im Container gesperrt; der Leser nutzt dann 16 pread-Threads |
| Loader (`weight_loader/qwen4_exp`), Schicht (`layers/qwen4_exp`) | fertig für Decode. Vier echte Schichten × zwölf Tokens stimmen mit der CPU-Referenz überein (Cosinus ≥ 0,99999) |
| Blockauswahl in der Attention, Index-Daten in einer Aux-Ebene des KV-Caches | fertig |
| Modell: Zustand nur in Prompt-Reihenfolge (`token_serial`) | fertig. Präfix-Cache, NVMe-Tier und Sitzungs-Auslagerung sind für dieses Modell aus |
| Prefill als Stapel (§5d) | fertig für Text: Stücke von 512 Tokens, jede Projektion und jeder Experte je Stück einmal gelesen. Hinter 2051 Tokens Kontext wählt noch jedes Token einzeln seine Blöcke |
| Server: Lade-Filter, Tabelle auf Platte (auch beim Snapshot-Reload), `MODEL.toml`, Parser, Preseed | fertig; `[serving].sessions = 4` ist vorläufig |
| Volltest mit allen 48 Schichten, Speicher- und Session-Messung | gelaufen am 2026-11-11, siehe §5c |
| Gebündelte Blockauswahl im Prefill, FP8-KV, MTP, Vision, Wiederverwendung von Präfixen und Sitzungen | offen |

Abweichungen von der Referenz, die die vorhandenen Atlas-Kernel mitbringen:
`gated_delta_rule_decode_f32` begrenzt den Zerfallsfaktor auf
[1e-6, 1 − 1e-6] und die Norm des GDN-Zustands je Kopf auf 1000. Im
Schichtentest wirkte sich das nicht aus.

## 5c. Volltest auf der GB10 (2026-11-11)

Produktionsserver, `--gpu-memory-utilization 0.9`, `--max-batch-size 4`,
BF16-KV, Denkmodus aus (`enable_thinking: false`), Temperatur 0.

| Prüfung | Ergebnis |
|---|---|
| Kurze Frage (25 Prompt-Tokens) | „Die Hauptstadt von Frankreich ist Paris." |
| Code + Erklärung (210 Ausgabe-Tokens) | korrekte Primzahl-Funktion, schlüssige Erklärung |
| Prompt mit 4293 Tokens, Frage nach einer Angabe im ersten Satz und nach einer aus Absatz 3 | beide Antworten richtig (Blockauswahl aktiv: mehr als doppelt so viele Tokens wie das Budget) |

| Messwert | Wert |
|---|---|
| Decode | 14,5–16,3 tok/s (eine Anfrage, ohne MTP) |
| Prefill | ≈ 150 tok/s als Stapel (§5d); vorher ≈ 16 tok/s Token für Token |
| Atlas-eigen / KV-Pool / nach dem Pool (Reserve) | 73,2 / 27,15 / 6,88 (7,38) GiB |
| KV-Tokens | 1.025.760 → Regel 4 Sessions, 256.432 Tokens je Session |
| `MemAvailable` geladen | 9,5–11,9 GiB; `late_alloc_mib` 0 |
| Laden aus dem Snapshot | 66,5 GiB in 121 s (0,55 GiB/s, kalter Cache), Modellwechsel 135 s |
| Snapshot einmalig anlegen | 38 min, davon 37 min Umlagern von ≈ 225.000 Tensoren in die Arena |

Vergleich (mit MTP): vLLM 35 tok/s, SGLang 27,5 tok/s, SGLang ohne MTP 15,9.

Beim Volltest gefunden und behoben: Der schnelle Loader fand keine Tensoren
in Checkpoints mit Shards in einem Unterordner; das Kernel-Build-Skript
bemerkte neue Targets nicht; die Einbettungstabelle hat mehr Zeilen als das
Tokenizer-Vokabular; das Template lehnt die Denkstufe „high" ab
(`tool_defaults.toml`, `[template_reasoning_effort]`).

Offen aus dem Volltest: Der SSM-Snapshot-Pool reserviert 2 GiB für
Präfix-Snapshots, die dieses Modell nicht nutzt. `[estimate]` in der
`MODEL.toml` fehlt (Kalibrierung mit `tests/estimate_calibration.py`).

## 5d. Prefill als Stapel (2026-11-11)

Decode liest je Token alle Projektionen und zehn Experten je Schicht. Für
einen Prompt ist das die falsche Reihenfolge: Bei 512 Tokens je Stück wird
praktisch jeder der 512 Experten einer Schicht gebraucht, aber jeder nur
einmal gelesen. Der Treiber (`model/token_serial.rs`) schickt den Prompt
deshalb in Stücken von bis zu `PREFILL_ROWS = 512` Tokens durch
`Qwen4ExpLayer::prefill`; ein einzelnes Rest-Token läuft über `decode`.
Beide Wege schreiben denselben Zustand je Sequenz (GDN-Zustand, Conv- und
N-Gramm-Historie, KV- und Index-Ebene), ein Prompt kann also an jeder Stelle
zwischen ihnen wechseln.

| Baustein | im Stück |
|---|---|
| Strom-Mischer | Gruppennorm nach BF16, Down/Inject/Up als BF16-GEMM (`gr_norm_bf16`, `gr_act`, `gr_gate_mix`) |
| MXFP8-Projektionen | für die Dauer einer GEMM nach BF16 entpackt (`mxfp8_expand_bf16`, höchstens 63 MB); das residente Gewicht bleibt MXFP8 |
| GDN | Conv und Rekurrenz je ein Launch über alle Tokens (`causal_conv1d_update_prefill`, `gated_delta_rule_prefill_persistent`) |
| Attention | K/V und Index-Keys des Stücks zuerst; bis 2051 Tokens Kontext ein kausaler Durchlauf über den Cache (`inferspark_prefill_paged`), dahinter Auswahl und Attention je Token wie im Decode |
| MoE | Tokens nach Experten sortiert, csf-Skalen der gerouteten Experten entpackt, gruppierte W4A16-GEMM über die Zeigertabellen des Decode-Pfads |
| N-Gramm-Schicht | Zeilen aller Tokens in einem Lesevorgang, Faltung über die Token-Achse in einem Launch |

Arbeitsspeicher des Prefill-Pfads: ≈ 235 MB, einmal beim Laden angelegt.

**Zahlen.** Der Prefill-Pfad rundet Zwischenwerte auf BF16, wo Decode FP32
hält (Mischer, GEMM-Ausgänge). Bei identischen Eingängen stimmen alle Stufen
einer Schicht auf 0,1–0,9 % (RMS) überein. Größer wird die Abweichung nur,
wenn die Rundung den schwächsten der zehn Experten eines Tokens gegen den
elften tauscht (Gewicht ≈ 0,04; bis 4 % am MoE-Ausgang dieses Tokens). Das
ist die Empfindlichkeit der Expertenwahl, die jede BF16-Engine hat, kein
Fehler des Pfads. Der Schichtentest prüft deshalb dreifach: jede Schicht
einzeln gegen Decode auf identischen Eingängen (Cosinus > 0,999), die
Stückelung exakt (ein Stück = zwei Stücke = Token für Token) und Ende-zu-Ende
gegen die CPU-Referenz mit lockerer Schranke.

**Messung am vollen Modell** (Produktionsserver, Denkmodus aus, Temperatur
0; Prompt aus nummerierten Absätzen, gefragt wird nach Angaben am Anfang, in
der Mitte und am Ende — `logs/flashnext_prefill_check.py`):

| Prompt-Tokens | Zeit bis zum ersten Token | Antworten |
|---|---|---|
| 284 | 6,6 s | richtig |
| 1.104 | 9,3 s | richtig |
| 1.932 | 12,5 s | richtig |
| 4.211 | 28,4 s (Token für Token: 4.293 Tokens in 270 s) | richtig |
| 7.936 | 53,3 s | richtig |

Das sind ≈ 150 tok/s, vor und hinter der Grenze der vollen Attention gleich:
Die Auswahl je Token ist nicht der Engpass. Die Zeit hängt an der Zahl der
Stücke, nicht der Tokens — rund 3,2 s je Stück von bis zu 512 Tokens, auch
wenn es nur 284 enthält. Offen ist, wohin diese Zeit geht (Entpacken der
csf-Skalen aller Experten, gruppierte GEMM mit vielen leeren Kacheln,
Entpacken der MXFP8-Gewichte); das zu messen ist der nächste Schritt.
vLLM erreicht auf derselben Hardware ≈ 3100 tok/s.

Ein Fehler, den der Schichtentest nicht sah: Der Treiber legte die KV-Slots
(i64) hinter die Positionen (u32) — bei ungerader Tokenzahl auf eine nicht
ausgerichtete Adresse. Der Schichtentest hat ein eigenes Metadaten-Layout;
der Treiber wird erst am vollen Modell geprüft.

## 6. Prüfung der Zahlen

Im Container gibt es kein PyTorch, und das Original in BF16 ist für die CPU zu
groß. Vorgesehen:

1. Je Kernel eine CPU-Referenz in Rust (Konvention `*_microtest.rs`,
   `spark-storage/src/*_ref.rs`). Prüft Kernel gegen die Formel.
2. Je Baustein ein Vergleich mit der HF-Referenz auf echten Gewichten einer
   Schicht (PyTorch-CPU in einer venv außerhalb des Repos, Ergebnisse unter
   `golden/`, nicht eingecheckt). Prüft die Formel gegen das Original.
3. Ende-zu-Ende: Greedy-Ausgabe und Log-Wahrscheinlichkeiten gegen den
   vLLM-Container der Modellkarte auf demselben Host (Host-Schritt).

## 7. Offen

- Ob `jinja-templates/` einen Eintrag für `qwen4_exp` braucht. Das Template
  des Checkpoints ist bytegleich mit dem des Qwen3.8-27B, aber der 27B läuft
  über das Override `qwen3_5.jinja`, das davon abweicht.
- Index-Keys als dritte Ebene im KV-Cache oder als Pseudo-Schichten.
- Neue Schichttypen (`layers/qwen4_exp/`) gegenüber Schaltern in
  `Qwen3AttentionLayer`/`Qwen3SsmLayer`.

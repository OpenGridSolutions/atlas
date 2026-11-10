# Speichermodell: KV-Budget, Reserve und Buchführung der Geräte-Allokationen

Stand: 2026-11-10 — umgesetzt und auf der GB10 an allen zehn Arena-Modellen
gemessen (§7), allerdings nur unter leichter Last. Was offen ist, steht in
`Todo.md`, Phase 4 (4.1–4.10); dieses Dokument wiederholt die Liste nicht.

## 0. Kurzfassung

Die GB10 hat einen gemeinsamen Speicher für CPU und GPU (119,7 GiB). Atlas hält
immer nur ein Modell darin und gibt ihm allen Platz, den
`--gpu-memory-utilization` erlaubt. Die einzige frei wählbare Größe ist der
KV-Pool; er wird beim Laden einmal bemessen und danach nicht mehr verändert.

```
Budget        = Gerätespeicher × --gpu-memory-utilization      (119,7 × 0,90 = 107,7 GiB)
Atlas-eigen   = freier Speicher vor dem Laden − freier Speicher jetzt
KV-Pool       = Budget − Atlas-eigen − Reserve
```

„Atlas-eigen" ist alles, was der Prozess bis zu diesem Moment für das Modell
belegt hat (Gewichte, abgeleitete Kopien, Puffer). Die Reserve ist eine
Schätzung dessen, was der Build **nach** dem KV-Pool noch allokiert. Die
Rechnung stimmt genau dann, wenn nach dem Laden nichts Ungeplantes mehr
dazukommt. Das war bis zum 2026-11-10 an vier Stellen nicht der Fall (§6); die
Fehler sind behoben, und der Server zählt jetzt selbst mit, was nach dem Build
noch allokiert wird (§5).

Ergebnis der Messung: Bei allen zehn Modellen bleibt der Prozess im Budget,
geladen sind 6,4–7,7 GiB `MemAvailable` frei, und nach dem Build kommen höchstens
64 MiB dazu.

## 1. Die Rechnung

`crates/spark-model/src/factory/kv_budget.rs::compute`, reine Arithmetik ohne
GPU-Zugriff. `factory/build.rs` liefert die Messwerte und schreibt das Ergebnis
ins Log.

**Freier Speicher.** `gpu.free_memory()` ist das Maximum aus `cudaMemGetInfo`
und `MemAvailable + SwapFree` aus `/proc/meminfo`. Auf der GB10 gewinnt die
zweite Zahl; sie kann über dem Gerätespeicher liegen, weil der Swap (8 GiB) nicht
Teil davon ist.

**Baseline.** Der freie Speicher, bevor das Modell geladen wird. Sie wird beim
Start des Servers genommen und nach jedem Entladen neu gesetzt
(`scheduler/hotswap_ctl.rs`, Logzeile `KV-budget baseline re-armed post-unload`).
Was schon vorher belegt war (andere Prozesse, der Server selbst ohne Modell,
rund 1,15 GiB), steckt in der Baseline und zählt nicht zum Budget.

**Atlas-eigen.** `Baseline − freier Speicher jetzt`, beide Werte mit Swap und
keiner davon auf den Gerätespeicher begrenzt. Die Differenz gilt als plausibel,
wenn sie größer als 0 und nicht größer als der Gerätespeicher ist; sonst fällt
die Rechnung auf `Gerätespeicher − frei` zurück und meldet das
(`KV budget auto-measure implausible`). `ATLAS_KV_EXTERNAL_RESERVE_GB` ersetzt
die Messung durch einen festen Abzug für fremde Prozesse.

**Physische Grenze.** Unabhängig vom Budget wird der Pool nie größer als
`MemAvailable − Reserve`, hier **ohne** Swap: KV-Seiten werden von der GPU
angefasst und sind gepinnt, Swap ist für sie kein Platz.

**Weitere Abzüge.** `--kv-pool-gb` legt die Poolgröße fest (weiterhin durch die
physische Grenze gedeckelt). Mit residenten Embedding-/Reranker-Sidecars
(`--sidecar-small-max-gib` > 0; Standard 0 = aus) wird deren Speicher vom Pool
abgezogen, höchstens aber die Hälfte des Pools. Whisper und Qwen-Image liegen nie
neben einem Arena-Modell im Speicher und brauchen keinen Abzug.

## 2. Reihenfolge beim Laden

Was wohin zählt, hängt davon ab, wann es allokiert wird:

1. **Vor dem KV-Budget:** Gewichte in der Arena, abgeleitete Gewichte, Puffer.
   Das misst die Rechnung als „Atlas-eigen".
2. **KV-Pool:** wird in voller Größe angelegt. Das spätere Füllen mit Tokens
   kostet keinen weiteren Speicher.
3. **Nach dem KV-Pool, noch im Build:** SSM-Zustands- und Snapshot-Pools,
   Prefill-Puffer der GDN-Schichten. Dafür steht die Reserve.
4. **Nach dem Build:** soll nichts mehr kommen. Was trotzdem allokiert wird, war
   nie Teil der Rechnung (§5, „late").

## 3. Reserve-Schätzung

`crates/spark-server/src/main_modules/serve_phases/preflight.rs::preflight_reserve`.
Die Reserve ist die Summe aus

- dem SSM-Zustands-Pool (`spark_model::model::ssm_state_pool_bytes` — dieselbe
  Formel, mit der der Pool allokiert; `--max-batch-size` + 1 Slots, der
  zusätzliche ist der Dummy-Slot),
- dem SSM-Snapshot-Pool (`--ssm-cache-slots` plus der Rollback-Ring je aktiver
  Sequenz),
- den Zwei-Phasen-Puffern des GDN-Prefills (skalieren mit der Prefill-Stückgröße),
- 0,5 GiB Sicherheitsabstand (4 GiB mit Spekulation).

Modelle ohne SSM-Schichten haben nur den Sicherheitsabstand. Die Schätzung gilt
für jeden Load, auch für den Lazy-Load beim Modellwechsel. Der Server misst
zusätzlich je Modell, um wie viel `MemAvailable` über den Build ohne Gewichte und
KV-Pool sinkt, und legt den Wert in `snapshots/swap_costs.json` ab
(`runtime_reserve_bytes`); das ist eine andere Größe (sie enthält alles vor dem
KV-Budget) und dient nur der Diagnose.

## 4. Fenster und Sessions

`crates/spark-model/src/factory/context_window.rs`. Der Pool besteht aus Blöcken
zu 16 Tokens; ein Block ist für den Server selbst reserviert.

```
Regel-Sessions = aufrunden(nutzbare Tokens im Pool / natives Fenster des Modells)
Fenster        = min(natives Fenster, abrunden(nutzbare Blöcke / sessions) × 16)
```

`sessions` kommt aus `[serving].sessions` in der `MODEL.toml` des Modells und
muss mindestens `--max-batch-size` sein. Ist dort die Regelzahl oder weniger
eingetragen, bekommt jede Session das native Fenster (oder knapp darunter). Sind
mehr Sessions eingetragen, wird das Fenster kleiner und folgt der beim Laden
gemessenen Poolgröße — es kann dann von Load zu Load um einige tausend Tokens
schwanken (Todo 4.7).

## 5. Buchführung der Geräte-Allokationen

`crates/spark-runtime/src/cuda_backend/alloc_tracker.rs`. Jede
Geräte-Allokation des Prozesses wird mit ihrer Größe gezählt und fällt in eine
von drei Gruppen:

| Gruppe | Wann | Beim Entladen |
|---|---|---|
| **build** | während eines Modell-Builds | wird freigegeben |
| **sidecar** | innerhalb eines Sidecar-Bereichs | gibt das Sidecar-Ledger frei |
| **late** | außerhalb eines Builds, dazu die Arbeitsbereiche der Bibliotheken (cuBLASLt, CUTLASS, FlashInfer) | bleibt bestehen |

„late" ist die Gruppe, auf die es ankommt: Speicher, den das KV-Budget nie
gesehen hat. Die Zähler sind über `GpuBackend::alloc_stats()` abrufbar (im Trait
mit Standardwert 0, andere Backends sind nicht betroffen).

Sichtbar an drei Stellen:

- **Beim Laden**, eine Zeile je Build:
  `Build allocations: 21.77 GiB before the KV budget, KV pool 56.25 GiB, 8.19 GiB after the KV pool (reserve estimate 8.69 GiB)`.
  Der dritte Wert muss unter dem vierten liegen.
- **Im Heartbeat** (Log-Target `mem`, alle 60 s, `ATLAS_MEM_HEARTBEAT_SECS`):
  `late_alloc_mib` (jetzt belegt) und `late_peak_mib` (Höchststand seit dem
  letzten Build).
- **Beim Entladen:**
  `Hot-swap: device allocations of the unloaded model — build … GiB (freed); after the build: peak … MiB, still live … MiB in … allocation(s) (not freed)`.

Die Buchführung misst nur; sie begrenzt nichts und bricht keinen Load ab
(Todo 4.2 und 4.3).

## 6. Was am 2026-11-10 geändert wurde

Auslöser: Beim Laden des Qwen3.8-27B wurde der KV-Pool 64 GB groß, wo 56 GB
gepasst hätten. Während der ersten Anfrage liefen RAM und Swap voll, und der Host
beendete den Server. Die Fehler waren älter, wirkten sich aber erst aus, seit der
Treiber 620.70.06 seinen Speicher korrekt als frei meldet (unten).

| Fehler | Wirkung | Änderung |
|---|---|---|
| Die Baseline wurde auf den Gerätespeicher begrenzt, bevor der aktuelle Wert abgezogen wurde. | „Atlas-eigen" fiel um die Swap-Größe zu klein aus, der Pool um 2,8–3,5 GiB zu groß — bei jedem Load. | Unbegrenzte Differenz; die Plausibilitätsprüfung vergleicht mit dem Gerätespeicher. |
| Beim Lazy-Load ersetzte der gespeicherte Messwert aus `swap_costs.json` die Reserve-Schätzung. | 27B: 4,3 statt 8,5 GiB Reserve. Der Messwert beschreibt eine andere Größe (§3). | Die Reserve ist immer die Schätzung. |
| Mit `ATLAS_CUBLAS_GEMM=1` liefen auch FP8-Gewichte über cuBLASLt; dafür wurde beim ersten Prefill je Gewicht eine BF16-Kopie angelegt. | Allokation nach dem Build, außerhalb des Budgets, beim Entladen nicht freigegeben. 27B: rund 10,6 GiB (gerechnet). | Der Schalter gilt nur noch für Gewichte, die als BF16 vorliegen. FP8-Projektionen laufen auf den FP8-Kerneln (W8A8 blockskaliert, W8A16). |
| Die Reserve rechnete den SSM-Zustands-Pool ohne den Dummy-Slot. | 76,5 MiB (80B) bis 151,5 MiB (27B) zu wenig. | Eine Formel für Schätzung und Pool (`ssm_state_pool_bytes`). |

Grundsatz aus der dritten Zeile: Gewichte werden in dem Format gerechnet, in dem
sie vorliegen. Eine Umwandlung zur Laufzeit ist Speicher, den das Budget nicht
kennt.

Dazu kam die Buchführung aus §5, damit eine fünfte Stelle dieser Art im Log
steht, statt erst als Absturz aufzufallen.

**Treiber.** Der NVIDIA-Treiber hält freigegebenen Gerätespeicher in eigenen
Pools. Im 610er-Zweig sind diese Pools nirgends als frei verbucht: Der Server sah
nach einem Modellwechsel kaum freien Speicher und bemaß den KV-Pool viel zu klein
(80B: 11 GB). Ab 620.70.06 zählen sie in `MemAvailable` und `cudaMemGetInfo` mit
(80B: 52 GB). Die zwischenzeitliche Modul-Option
`NVreg_EnableSystemMemoryPools` in `install.sh` ist wieder entfernt. Die Rechnung
setzt einen Treiber voraus, der korrekt verbucht.

## 7. Messwerte

GB10, Treiber 620.70.06, `--gpu-memory-utilization 0.9`, `--max-batch-size 4`,
2026-11-10. Je Modell ein kurzer Request und einer mit rund 4000 Tokens. Alle
Speicherwerte in GiB.

| Modell | Atlas-eigen | KV-Pool | Nach dem Pool: gemessen / Reserve | Sessions: Regel / eingetragen | Fenster (Tokens) | `MemAvailable` geladen |
|---|---|---|---|---|---|---|
| Qwen3-Next-80B Thinking | 51,0 | 52,0 | 4,21 / 4,71 | 12 / 11 | 262.144 | 7,7 |
| Qwen3-Coder-Next | 48,0 | 55,0 | 4,21 / 4,71 | 13 / 11 | 262.144 | 7,3 |
| Qwen3-VL-30B | 52,3 | 55,0 | 0,00 / 0,50 | 5 / 6 | 177.888 | 7,5 |
| Qwen3.8-27B | 42,8 | 56,25 | 8,19 / 8,69 | 6 / 7 | 191.488 | 7,0–7,4 |
| Qwen3-Embedding-8B | 19,3 | 87,9 | 0,00 / 0,50 | 27 / 26 | 40.960 | 6,6 |
| Qwen3-Reranker-8B | 20,5 | 86,8 | 0,00 / 0,50 | 27 / 26 | 40.960 | 6,6 |
| Qwen3-VL-Embedding-8B | 22,6 | 84,6 | 0,00 / 0,50 | 5 / 4 | 262.144 | 6,9 |
| Qwen3-VL-Reranker-8B | 23,7 | 83,5 | 0,00 / 0,50 | 4 / 4 | 260.560 | 6,8 |
| TranslateGemma-12B | 31,2 | 76,0 | 0,00 / 0,50 | 3 / 4 | 92.192 | 7,3 |
| Chandra-OCR-2 | 20,0 | 84,4 | 2,89 / 3,39 | 11 / 11 | 251.280 | 6,4 |

Ablesen lässt sich daraus:

- **Die Summe geht auf.** Atlas-eigen + KV-Pool + Reserve ergibt bei jedem
  Modell das Budget von 107,7 GiB. Der Prozess belegt geladen 108,3–109,1 GiB
  (Budget plus die 1,15 GiB des Servers ohne Modell).
- **Die Reserve stimmt.** Nach dem Pool wird bei jedem Modell genau die
  Schätzung abzüglich der 0,5 GiB Sicherheitsabstand allokiert.
- **Nach dem Build kommt nichts dazu.** `late_alloc_mib` ist 0 oder 64 (der
  Arbeitsbereich von cuBLASLt), der Höchststand ist derselbe — auch beim 27B nach
  zwei Prefills mit 4000 Tokens, wo vorher rund 10,6 GiB dazukamen.
- **27B unter Last:** kleinster Wert von `MemAvailable` während des Prefills
  6,57 GiB.

Nicht gemessen: parallele Anfragen, volle Fenster, Bildeingaben, Whisper und
Qwen-Image (Todo 4.1 und 4.9).

## 8. Ablesen im Betrieb

Beim Laden schreibt der Server vier Zeilen, die zusammen die Rechnung zeigen
(Qwen3.8-27B, 2026-11-10):

```
KV budget self-relative (auto): baseline-free 122.0 GB − free-now 79.2 GB = Atlas-own 42.8 GB; co-tenants 0.0 GB excluded (…)
KV cache: 119.7 GB total × 90% util = 107.7 GB budget; 42.8 GB pre-KV + 8.7 GB reserve → 56.2 GB for KV → 83779 blocks × 16 tok/block = 1340464 max KV tokens
Context window: the native window (262144 tokens) fits 5.11x into the KV pool → rule: 6 session(s); MODEL.toml [serving].sessions = 7 → 191488 tokens per session
Build allocations: 21.77 GiB before the KV budget, KV pool 56.25 GiB, 8.19 GiB after the KV pool (reserve estimate 8.69 GiB)
```

(Die Log-Zeilen schreiben „GB", gerechnet wird in GiB.)

Worauf zu achten ist:

- `after the KV pool` größer als `reserve estimate`: Die Schätzung passt für
  dieses Modell nicht; der Pool ist um die Differenz zu groß.
- `late_alloc_mib` oder `late_peak_mib` im Heartbeat deutlich über 64: Ein
  Codepfad allokiert zur Laufzeit Gerätespeicher.
- `still live … MiB` beim Entladen wächst von Wechsel zu Wechsel: ein Leck über
  Modellwechsel hinweg.
- `KV budget auto-measure implausible`: Die Baseline passt nicht zum aktuellen
  Zustand; der Pool wird dann aus `Gerätespeicher − frei` gerechnet und fällt
  kleiner aus.

Der Heartbeat warnt mit `memory LOW`, sobald `MemAvailable` unter
`ATLAS_MEM_WARN_AVAIL_GB` (8) oder der freie Swap unter `ATLAS_MEM_WARN_SWAP_GB`
(2) fällt. Mit geladenem Modell bleiben planmäßig rund 7 GiB frei; die Warnung
steht deshalb in jeder Heartbeat-Zeile (Todo 4.6).

## 9. Was die Rechnung nicht erfasst

- **Speicher außerhalb des Prozesses.** Kernel und Treiber belegen für den
  Prozess bis zu 1,3 GiB, die kein Zähler des Servers sieht.
- **Wachstum des Prozesses zur Laufzeit.** Der Speicher außerhalb der
  Geräte-Allokationen wuchs schon bei einer Anfrage um bis zu 0,7 GiB.
- **Veränderungen am System während des Ladens.** „Atlas-eigen" ist eine
  Differenz zweier Systemmessungen. Lagert der Kernel während des Ladens aus oder
  gibt ein anderer Prozess Speicher frei, landet das in der Rechnung (einmal
  beobachtet: Pool 2,7 GiB zu klein).
- **Der Abstand zum Speicherende.** Außerhalb des Budgets liegen 12 GiB, davon
  bleiben rund 7 frei. Dieser Abstand ist ein Erfahrungswert, keine hergeleitete
  Größe.

Die Arbeitspunkte dazu stehen in `Todo.md`, Phase 4.

## 10. Tests

| Was | Wo |
|---|---|
| Budget-Arithmetik, mit Nachstellung der Loads vom 2026-11-10 (80B, 27B) und der früheren Ausfälle | `spark-model/src/factory/kv_budget.rs` |
| Formel des SSM-Zustands-Pools gegen die tatsächliche Allokation, mit und ohne Spekulation | `spark-model/src/model/ssm_pool.rs` |
| Zuordnung build / late, Höchststand, Entladen | `spark-runtime/src/cuda_backend/alloc_tracker.rs` |
| Fenster-Regel | `spark-model/src/factory/context_window.rs` |

Die On-Box-Messung aus §7 lief mit einem Skript, das noch nicht im Repository
liegt (Todo 4.8).

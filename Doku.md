# Atlas — Architektur-Dokumentation

Stand: 2026-07-12 (Fork-Branch `modifications`). Diese Doku erklärt den
Aufbau des Projekts, die Komponenten und ihre Aufgaben, und wie neue
CUDA-Kernels geschrieben und eingebunden werden. Datei-Referenzen zeigen
auf den realen Code, damit man von hier aus direkt weiterlesen kann.

---

## 1. Was ist Atlas?

Atlas ist ein Inference-Server für LLMs auf der NVIDIA DGX Spark (GB10,
aarch64, 128 GB Unified Memory, SM 12.1). Ein einziges Rust-Binary
(`spark serve <model>`) bedient OpenAI-kompatible HTTP-Endpunkte und
kann **mehrere Modelle** verwalten: Genau ein Modell ist zur Zeit in der
GPU-Gewichts-Arena resident, weitere registrierte Modelle werden bei
Bedarf per **Lazy Hot-Swap** in Sekunden aus Snapshots eingewechselt.
Whisper läuft als dauerhaft residenter **Sidecar** daneben.

Die wichtigsten Designentscheidungen:

- **Kein Python, keine externen Inference-Libraries** — Tokenizer
  (HF `tokenizers`), Jinja-Templates (`minijinja`), Scheduler, Sampler,
  Kernels: alles im Workspace.
- **CUDA-Kernels als PTX zur Buildzeit einkompiliert**, gruppiert in
  „Targets" pro Modellfamilie (Abschnitt 5).
- **Unified Memory ausgenutzt**: Gewichte liegen in einer großen
  vorallozierten Arena; Modellwechsel = Arena leeren + Snapshot-Datei
  (`.raw`) per O_DIRECT einströmen (~8 GiB/s), kein Neuladen von
  Safetensors.

---

## 2. Workspace-Layout (Crates und ihre Aufgaben)

```
crates/
├── atlas-core          Config-Parsing + Kernel-Registry-Grundtypen
├── atlas-kernels       CUDA-Build-Pipeline (build.rs) + PTX-Einbettung
├── spark-runtime       GPU-Backend, Arena, KV-Cache, Snapshots, Sampler
├── spark-model         Weight-Loader, Layer-Implementierungen, Forward-Pass
├── spark-server        HTTP-API, Scheduler, Tokenizer/Templates, Hot-Swap
├── spark-storage       Low-Level-I/O (io_uring/O_DIRECT), minimale CUDA-FFI
├── spark-comm          Multi-GPU-Kommunikation (EP/TP, auf GB10 kaum genutzt)
├── cufile-sys          FFI für GPUDirect Storage (libcufile)
├── xgrammar            Constrained Decoding (Tool-Call-Grammatiken)
├── atlas-quant/-norm/-activation/-reduce/-embed
│                       kleine Hilfs-Crates (Quant-Helfer, Norm-, Aktivierungs-,
│                       Reduktionsprimitiven; atlas-embed = Pooling-Scaffold)
└── atlas-spark-bench   Benchmarks + Korrektheitstests gegen den Server
```

### 2.1 `atlas-core` — Konfiguration & Registry

- **`config.rs` + `config/parsers/*.rs`**: Übersetzt HF-`config.json`
  in das interne `ModelConfig` (Single Source of Truth für alle
  Dimensionen). Der Einstieg ist `config/dispatch.rs`: ein `match` auf
  `model_type` wählt einen **schema-spezifischen Parser-Arm**
  (`qwen3`, `qwen3_vl`, `gemma3`, `gemma4`, `whisper`, …) oder den
  generischen Flat-Fallback. Verschachtelte Configs (`text_config`)
  brauchen zwingend einen eigenen Arm — der Flat-Fallback stirbt sonst
  mit „missing field hidden_size" (wiederkehrendes Muster; siehe
  `parsers/gemma3.rs` und `parsers/whisper.rs` als Vorlagen).
- **`registry.rs`**: `AtlasRegistry` — prozessweiter Singleton, der
  PTX-Module pro Kernel-Target in den CUDA-Kontext lädt. Targets werden
  **additiv** geladen (Hot-Swap zu einer neuen Modellfamilie lädt deren
  Module nach, entlädt nie).

### 2.2 `atlas-kernels` — die Kernel-Build-Pipeline

- **`build.rs`** scannt `kernels/gb10/` zur Buildzeit, kompiliert jede
  `.cu` mit `nvcc` zu PTX (SM 12.1) und dedupliziert identische
  (Quelle, Arch, Flags)-Tupel über alle Targets (~8× weniger
  nvcc-Aufrufe, parallelisiert). Ergebnis: generierter Code
  (`OUT_DIR/target_ptx.rs`) mit den PTX-Bytes **im Binary eingebettet**.
- **`src/lib.rs`**: `ptx_for_config(model_type, hidden_size)` — die
  Laufzeit-Target-Auswahl. Exakter Match auf `[[model_types]]`-Einträge
  der MODEL.toml zuerst, dann Wildcard. Liefert `TargetPtxSet`
  (Module + `behavior`-Defaults + Sampling-Presets aus MODEL.toml).

### 2.3 `spark-runtime` — GPU-Substrat

- **`gpu.rs` / `cuda_backend/`**: `GpuBackend`-Trait (alloc, memcpy,
  Kernel-Launch, Streams) + CUDA-Treiber-FFI. `AtlasCudaBackend::new`
  bindet sich an den prozessweiten Kontext; `bind_to_thread()` für
  weitere Threads. Diagnose-Helfer wie `cuda_mem_get_info_raw()`
  (für /v1/stats) binden den Kontext on-demand und **entbinden ihn
  wieder** (geteilte Tokio-Worker!).
- **`weights.rs` / `fast_weights/`**: `WeightStore` — namens-indizierte
  GPU-Tensoren; Safetensors-Fast-Loader (O_DIRECT, pipelined).
- **`weights_arena.rs`**: die große Arena (z. B. 50 GiB). Modelle
  werden per `migrate_weight_store` hineinkopiert;
  `snapshot.rs` schreibt/liest das ATLASRAW-Format (hash-verifiziert).
- **`kv_cache.rs`**: `PagedKvCache` — paged KV mit Blockliste,
  per-Layer-Dtypes (BF16/FP8/NVFP4-Mischbetrieb), Prefix-Cache-Refcounts.
  Publiziert Belegungszähler nach `kv_stats.rs` (für /v1/stats).
- **`meminfo.rs`**: /proc/meminfo-Leser (MemAvailable + SwapFree,
  HugePages-Fall) — Anzeige in /v1/stats, Grundlage für den geplanten
  KV-Budget-Fix.
- **`sampler.rs`**: Sampling (top-k/p, min-p, Penalties, DRY, Seeds).

### 2.4 `spark-model` — Modelle & Layer

- **`factory.rs`**: `loader_for_config(config)` — `match` auf
  `model_type` → **Weight-Loader** (z. B. `Qwen3WeightLoader`,
  `Qwen3VLWeightLoader`, `Gemma3WeightLoader`). `factory/build.rs`
  (`build_model`) ist modell­agnostisch: ruft Loader-Methoden
  (`load_layers`, `load_embedding`, `load_lm_head`, …), berechnet das
  KV-Budget und baut das `TransformerModel`.
- **`weight_loader/*.rs`**: pro Familie ein Loader. Hier lebt die
  gesamte Format-Intelligenz: NVFP4-Varianten (`weight_map/nvfp4_detect.rs`),
  FP8-block-skaliert (dequant → NVFP4-Requant, `weight_map/loaders_fp8.rs`),
  BF16-roh (Runtime-Quant), fused/stacked Experten-Layouts
  (`weight_map/ssm_qwen35/fused_vl.rs`), Norm-Konventionen
  (zero-zentriert vs. Standard → `dense` vs. `dense_minus_one`).
- **`layers/`**: die Bausteine — `Qwen3AttentionLayer` (GQA, paged
  attention, per-Layer-Overrides für Rope-Theta, Sliding-Window,
  Attention-Scale), `DenseFfnLayer` (SwiGLU/GeGLU), `MoeLayer`,
  `vision_encoder/` (Qwen3-VL-ViT mit DeepStack),
  `ops/` (Rust-Wrapper um Kernel-Launches).
- **`model/`**: Forward-Pass (Prefill chunked, Decode mit CUDA-Graphs),
  Logit-Finalize, Embedding-Scale, MTP/Spekulation.
- **`whisper/`**: eigenständiges Encoder-Decoder-Modul für den Sidecar
  (bewusst außerhalb des Model-Traits/Schedulers).
- **`vision_preprocess.rs`**: Bild-Dekodierung + smart_resize
  (OCR-Pixel-Floor, Pixel-Budget = ViT-Puffergrenze) + Patch-Packung.

### 2.5 `spark-server` — API & Orchestrierung

- **`main_modules/serve.rs`**: der Boot-Ablauf (Abschnitt 3.1) —
  Preseed-Kategorien, Registry, Arena, Scheduler-Start, AppState,
  Router.
- **`main_modules/serve_phases/`**: Boot-Bausteine (Config laden,
  Weight-Store, KV-Sizing, Tokenizer-Runtime, Preseed, Swap-Target).
- **`main_modules/hotswap.rs`**: `HotswapCtx` — Registry
  (`RegisteredModel` mit **Kategorie**: generative/embed/rerank/
  transcribe/translate), Swap-Zustandsmaschine, Snapshot-Pfade.
- **`scheduler/`**: eigener OS-Thread mit gebundenem CUDA-Kontext.
  Nimmt `InferenceRequest` (Blocking/Streaming) über einen mpsc-Kanal,
  fährt Prefill→Decode, verwaltet Thinking-Phasen (<think>-Erkennung,
  Budget, Watchdogs), Stop-Tokens, Tool-Call-Grammatiken; Steuerkanal
  für Unload/Load (Hot-Swap).
- **`api/`**: die Endpunkte — `chat/` (+Streaming), `completions.rs`,
  `embeddings.rs` (pooled hidden states), `rerank.rs` (yes/no-Logits),
  `translate.rs` (erzwungenes TranslateGemma-Template),
  `audio.rs` (Whisper-Sidecar), `stats.rs` (Speicher-Diagnose),
  `lazy_swap.rs` (Kategorie-Check + Swap-Trigger), `admin.rs`.
- **`tokenizer/`**: Chat-Template-Rendering (minijinja, inkl.
  Python-Jinja-Konvertierung), Vision-Pad-Expansion, Stop-Token-Setup.

---

## 3. Lebenszyklus

### 3.1 Boot (`serve.rs`, Reihenfolge)

1. **Kategorisierte Preseed-Listen** aus den sieben ENV-Vars
   `ATLAS_PRESEED_{GENERATIVE,EMBED,RERANK,TRANSCRIBE,TRANSLATE,OCR,IMAGE}`
   parsen (jede Kategorie ↔ genau ein Endpunkt; falscher Endpunkt ⇒ 400).
   Ist ein Image-Modell gelistet, wird sofort die Sidecar-Reserve fürs
   KV-Budget gesetzt (`ATLAS_IMAGE_RESERVE_GB`).
2. Boot-Modell: config.json parsen (dispatch) → Kernel-Target wählen
   (`ptx_for_config`) → Quant-Kompatibilität prüfen
   (`quant_pair_compatible`: nvfp4↔fp8↔bf16-Matrix) → Safetensors
   fast-laden → `loader_for_config` → `build_model` → Arena-Migration.
3. **Registry** aufbauen: pro Preseed-Eintrag `registered_model_checked`
   (Config parst? Loader existiert? Target existiert? Quant passt?) —
   nicht ladbare Einträge werden mit WARN übersprungen, Boot läuft
   weiter. Whisper und Qwen-Image werden nie registriert (Sidecars).
4. Scheduler-Thread starten, **Boot-Preseed**: Boot-Modell snapshotten,
   entladen, fehlende Snapshots aller anderen Modelle einmalig erzeugen
   (`serve_phases/preseed.rs`). Danach: Server idle, Arena leer.
5. AppState + axum-Router (`serve_router.rs`), HTTP auf `--port`.

### 3.2 Chat-Request (repräsentativ)

```
POST /v1/chat/completions
  └─ api/chat: ensure_model_active(model, Kategorie=Generative)
       ├─ Kategorie falsch → 400 („categorized as embed …")
       ├─ Modell nicht aktiv → Lazy Swap: Unload → Snapshot → Load (~4–9 s,
       │    Antwort-Header x-atlas-model-swap)
       └─ aktiv → weiter
  └─ ModelRuntime-Snapshot (Tokenizer, Template, Sampling-Defaults)
  └─ Jinja-Rendering (+ Vision: Bild → Patches → <|image_pad|>-Expansion)
  └─ resolve_thinking (CLI-Flag > Request > MODEL.toml thinking_default;
       Template-forced <think> wird automatisch erkannt)
  └─ InferenceRequest über mpsc an den Scheduler
       └─ Prefill (chunked; ViT-Encode falls Bilder) → Decode
            (CUDA-Graph, Stop-Tokens, Thinking-Split, Watchdogs)
  └─ Antwort (blocking JSON oder SSE-Stream)
```

Embeddings/Rerank nutzen denselben Weg mit `max_tokens=1`
(immediate-finish nach Prefill) und lesen pooled hidden states bzw.
zwei Logit-Einträge. `/v1/translate` rendert das TranslateGemma-Template
serverseitig (kein generisches Template). `/v1/audio/transcriptions`
geht am Scheduler vorbei in den Whisper-Sidecar.

`/v1/images/generations` geht ebenfalls am Scheduler vorbei: der
Qwen-Image-Sidecar (`spark-model/src/qwen_image/`, lädt beim ersten
Request, 26.7 GiB resident) fährt Text-Encoder → DiT-Sampler →
VAE-Decoder synchron auf einem Blocking-Thread; ein Bild zur Zeit,
weitere Requests warten am Sidecar-Lock. Das aktive Chat-Modell bleibt
geladen. Vor dem Laden und vor jeder Generierung prüft ein
Speicher-Preflight `MemAvailable` (ohne Swap) gegen Bedarf + 4 GiB
Marge (sonst 503).

Speicher-Eigenheit der GB10 (gemessen 2026-10-02): der NVIDIA-Treiber
gibt freigegebenen Gerätespeicher — beendete Prozesse, der beim
Boot-Preseed entladene KV-Pool — nicht sofort an den Kernel zurück,
sondern hält ihn minutenlang in einem Pool. In `MemAvailable` zählt er
als belegt (im KV-Budget-Log die „co-tenants"-Zeile), neue
CUDA-Allokationen werden aber daraus bedient. Folge: KV-Budget,
Sidecar-Reserve und Preflight sehen weniger freien Speicher als für
Gerätespeicher real verfügbar ist und entscheiden konservativ;
Host-Speicher (Weights-Arena) kann den Pool dagegen nicht nutzen.

---

## 4. Kernkonzepte im Detail

### 4.1 Arena, Snapshots, Hot-Swap

- Die **Arena** ist ein einziger großer GPU-Alloc (`--weights-arena-gb`),
  vor Überschreiben durch „protected ranges" geschützt.
- Ein **Snapshot** (`snapshots/<id>-<hash>.raw`) ist das Arena-Abbild
  eines Modells inkl. Manifest + 32-Byte-Versions-Hash (Modell-ID,
  Revision, Layoutversion). Stale Snapshots werden erkannt und neu
  erzeugt.
- **Lazy Swap** (`api/lazy_swap.rs`): erster Request auf ein nicht
  aktives, registriertes Modell bezahlt den Swap; gleichzeitige
  Requests auf dasselbe Ziel warten, ein drittes Modell bekommt 503 +
  Retry-After. Beim Load läuft der Weight-Loader **erneut** gegen den
  arena-gestützten Store (deshalb sind auch Runtime-Dequants wie
  FP8→NVFP4 pro Swap zu bezahlen).

### 4.2 Quantisierungspfade (Loader-Ebene)

`detect_nvfp4_variant` entscheidet config-first, dann heuristisch:

| Variante | On-Disk | Pfad |
|---|---|---|
| `Standard` | ModelOpt NVFP4 (weight + weight_scale + weight_scale_2) | zero-copy |
| `CompressedTensors` | RedHat/Sehyo NVFP4 (weight_packed + global_scale) | zero-copy |
| `Fp8Dequanted` | FP8-E4M3 + `weight_scale_inv` (block-skaliert) | Dequant→BF16→NVFP4-Requant (`quantized_from_fp8`) |
| `Bf16Raw` | unquantisiert | Runtime-Quant zu NVFP4 (oder BF16-Pfade via `set_bf16_weights`/`set_o_dense_bf16`) |

Qualitätskritische Klassen (Embedding/Rerank/Gemma-Attention) bleiben
bewusst BF16 (dense_gemv-Pfad), MoE-Experten gehen nach NVFP4.

### 4.3 Scheduler & Thinking

Der Scheduler ist eine Zustandsmaschine pro Sequenz: `inside_thinking`
startet true, wenn Thinking aktiv ist (Template-forced oder
MODEL.toml-Default) — alles bis `</think>` landet in
`reasoning_content`. Budget-Erschöpfung injiziert `</think>`
(Satzgrenzen-Defer). **Fallstrick aus der Praxis:** `thinking_default =
true` auf einem Instruct-Modell ohne Thinking-Training ⇒ leerer
content + Halluzinations-Drift (Phase V in Plan.md).

### 4.4 Kategorien & Sidecar

Jedes registrierte Modell trägt die Kategorie seiner ENV-Liste
(`ModelCategory`), `ensure_model_active` erzwingt sie **vor** dem
Fast-Path. Whisper ist bewusst kein Scheduler-Modell: der Sidecar
(`api/audio.rs`) lädt beim ersten Request eigenständig (eigenes
Kernel-Target, eigener Backend-Handle, tokio-OnceCell) und bleibt
resident — ein registriertes Whisper wäre lazy-swappable und würde am
fehlenden Model-Trait-Loader sterben.

---

## 5. Kernel-System: Aufbau, neue Kernels schreiben & einbinden

### 5.1 Verzeichnisstruktur

```
kernels/gb10/
├── HARDWARE.toml               Arch-Definition (sm_121)
├── common/                     ~150 Basis-Kernels (.cu) + KERNEL.toml
│   ├── rope.cu, rms_norm.cu, paged_decode_attn*.cu, w4a16_gemm.cu, …
│   └── KERNEL.toml             [modules]-Umbenennungen Datei→Modulname
└── <modell-target>/            z. B. qwen3-next-80b-a3b, translategemma-12b
    ├── MODEL.toml              Geometrie, [behavior], [sampling.*], [[model_types]]
    └── <quant>/                nvfp4/ oder bf16/
        ├── KERNEL.toml         [build] extra_nvcc_flags, [modules]
        └── *.cu                Overrides (shadowen common per DATEISTAMM)
```

**Ein Target = (Hardware, Modellfamilie, Quant).** Die Laufzeit wählt
es über die `[[model_types]]`-Einträge (`model_type` + `hidden_size`).
Entscheidend: fast alle Dimensionen (hidden, heads, vocab, Experten)
sind **Launch-Parameter zur Laufzeit** — der einzige Compile-Time-Bake
ist `head_dim` über `-DHDIM` (Default 256; Targets mit head_dim 128
setzen `-DHDIM=128`). Deshalb ist ein neues Target fast gratis: leeres
Quant-Verzeichnis erbt den kompletten common-Satz, und dank Build-Dedup
sind identische Kompilate Cache-Hits.

### 5.2 Wie ein Kernel vom .cu-File zum Launch kommt

1. **CUDA-Seite** (`kernels/gb10/common/foo.cu`):
   ```cuda
   extern "C" __global__ void foo_forward(
       __nv_bfloat16* __restrict__ x, unsigned int n, float alpha) { … }
   ```
   `extern "C"` ist Pflicht (Symbol-Lookup per Name). Modulname =
   Dateistamm, außer `[modules]` in KERNEL.toml benennt um
   (z. B. `dense_gemm_bf16 = "gemm"`).
2. **Build**: `atlas-kernels/build.rs` kompiliert automatisch — kein
   Registrieren nötig, jede `.cu` im gescannten Baum wird Teil des
   Targets. `cargo build -p atlas-kernels` zeigt pro Target
   „compiled N kernels (M model-specific overrides)".
3. **Laden**: `AtlasRegistry` lädt die PTX-Module des gewählten Targets
   beim Boot (bzw. additiv beim Swap auf eine neue Familie).
4. **Rust-Wrapper** (`spark-model/src/layers/ops/…`):
   ```rust
   pub fn foo(gpu: &dyn GpuBackend, kernel: KernelHandle,
              x: DevicePtr, n: u32, alpha: f32, stream: u64) -> Result<()> {
       KernelLaunch::new(gpu, kernel)
           .grid([n.div_ceil(256), 1, 1])
           .block([256, 1, 1])
           .arg_ptr(x).arg_u32(n).arg_f32(alpha)
           .launch(stream)
   }
   ```
   Argument-Reihenfolge und -Typen müssen exakt der CUDA-Signatur
   entsprechen (keine ABI-Prüfung!).
5. **Handle-Lookup** an der Nutzungsstelle:
   ```rust
   let k = gpu.kernel("foo", "foo_forward")?;          // hart (Fehler wenn fehlt)
   let k = try_kernel(gpu, "foo", "foo_forward");       // weich (Handle 0 = fehlt)
   ```
   `try_kernel` + Handle-0-Guard ist das Muster für optionale Kernels,
   die nur manche Targets haben (Beispiel: `rope_forward_linear` — nur
   von gemma3 genutzt, aber in common für alle kompiliert; die
   Dispatch-Kette in `qwen3_attention/prefill/*.rs` prüft
   `self.rope_linear_k.0 != 0`).

### 5.3 Einen neuen Kernel schreiben — Kochrezept

Beispiel aus dieser Codebasis: `dequant_fp8_blockscaled_bf16_slice_t`
(Dequant + Slice + Transpose in einem Pass, für fused FP8-Experten).

1. Kernel in die passende Datei/neue Datei unter `common/` (wenn
   familienübergreifend) oder `<target>/<quant>/` (wenn Override).
   Bestehende Konventionen übernehmen: `__restrict__`, BF16 via
   `__nv_bfloat16`, FP8 via Byte + `E4M3_LUT`, Grid/Block-Kommentar
   über der Signatur.
2. Rust-Wrapper daneben legen (in `layers/ops/` oder — bei
   Loader-Helfern — direkt im Loader-Modul, siehe
   `weight_map/ssm_qwen35/dequant_fp8.rs`).
3. `cargo build -p atlas-kernels` — prüfen, dass das Symbol im
   richtigen Target landet.
4. Aufrufer schreiben; bei Hot-Path-Kerneln einen `try_kernel`-Fallback
   vorsehen, damit alte Targets funktionsfähig bleiben (additiv!).
5. Microtest, wenn numerisch heikel: `crates/spark-model/examples/`
   enthält Kernel-Microtests (z. B. `w8a16_microtest.rs`) als Vorlage.

**Stolperfallen (alle real passiert):**
- *Eager-Symbole:* `Qwen3AttentionLayer`/`DenseFfnLayer` resolven
  einige w4a16-Symbole **hart**, die nur in Modell-Overrides existieren
  — ein „leeres" bf16-Target braucht deshalb mindestens die
  `w4a16_gemm.cu`-Kopie (siehe Kommentar in
  `qwen3-dense-8b/bf16/KERNEL.toml`).
- *Norm-Konvention:* der common-`rms_norm` rechnet `x·rms·(1+w)`
  (zero-zentrierte Gewichte, Qwen3-Next/Gemma). Modelle mit
  Standard-Gewichten laden Norms per `dense_minus_one` ODER überschreiben
  den Kernel im Target (gemma-4 tut Letzteres — nicht blind kopieren!).
- *Puffer-Kopplung:* feste Scratch-Größen (z. B. ViT `p_max`, SDPA-
  Score-Matrix) müssen mit den Preprocessing-Budgets aus **derselben
  Konstante** kommen (`vision_preprocess::VISION_MAX_PATCHES`).

### 5.4 Ein neues Modell anbinden — Checkliste

So sind TranslateGemma (gemma3) und Qwen3-VL-FP8 integriert worden:

1. **Config-Parser**: braucht `config.json` einen eigenen Arm?
   (verschachtelt / abweichendes Rope-Schema / fehlende Felder) →
   `atlas-core/src/config/parsers/<familie>.rs` + Arm in `dispatch.rs`
   + Fixture-Test in `config/tests/multiclass.rs`.
2. **Weight-Loader**: `factory.rs`-Arm → bestehender Loader oder neuer
   (Vorlagen: `qwen3_dense.rs` für BF16-Policy, `gemma3.rs` für
   per-Layer-Rope/Sliding, `fused_vl.rs` für exotische Layouts).
   Tensor-Layout **vorher am Checkpoint verifizieren** (safetensors-
   Header lesen: Namen, Shapes, Dtypes, Scale-Grids) — nicht raten.
3. **Kernel-Target**: `kernels/gb10/<name>/MODEL.toml` (Geometrie,
   `[[model_types]]`, `[behavior]` — Achtung `thinking_default`!,
   `[sampling.*]`) + `<quant>/KERNEL.toml` + nötige Override-Kopien.
4. **Kategorie**: Modell-ID in die passende `ATLAS_PRESEED_*`-Liste
   (compose.yml + serve.sh-Bridge).
5. **Boot**: Preseed erzeugt den Snapshot automatisch; Log prüfen
   („Preseed [kategorie] … registered").
6. **Akzeptanztest** nach dem Muster `scripts/*_test.sh` (Responses in
   Temp-Files, `python3 - <<'EOF'` liest `os.environ["RESP"]`) + volle
   Regression der bestehenden Suiten.

---

## 6. Betriebs-Cheatsheet

| Was | Wo |
|---|---|
| Server starten | `./serve.sh` (baut Release, killt Altinstanz, exportiert Preseed-Defaults) |
| Modell-Registry live | `GET /admin/models`, `GET /v1/stats` (inkl. Speicher-Diagnose) |
| Kategorien | `ATLAS_PRESEED_{GENERATIVE,EMBED,RERANK,TRANSCRIBE,TRANSLATE}` (JSON-Listen) |
| Snapshots | `snapshots/*.raw` — löschen = Neuerzeugung beim nächsten Boot |
| Test-Suiten | `scripts/categories_translate_test.sh`, `embed_rerank_test.sh`, `whisper_test.sh`, `stats_test.sh` |
| Kernel-Rebuild | automatisch via `cargo build` (nvcc-Cache macht Wiederholungen billig) |
| Projekt-Historie | `Plan.md` (Phasen K/T/W/S/V/Q mit Befunden), `CHANGELOG.md` |

## 7. KV-Cache-Debugging (`ATLAS_MEM_DEBUG`)

Per-Request/Per-Phase-Sicht auf den paged KV-Cache — für Block-Leak-Jagd
(z. B. Client-Timeouts) und um zuzusehen, wie Prefill/Thinking/Generation
Blöcke konsumieren. Reine Observability: ändert kein Scheduling, kein
Caching, keine Eviction; ohne die Env-Var entstehen keine Kosten.

```bash
RUST_LOG="off,memdebug=debug" ATLAS_MEM_DEBUG=1 ./serve.sh
```

`RUST_LOG=off,memdebug=debug` stummschaltet alle anderen Module; alle
Ausgaben laufen als strukturierte tracing-Felder auf dem Target `memdebug`
(Level `debug`; `error` nur bei Invariantenverletzungen).

Events pro Request (Feld `seq` = memdebug-ID, `resumed=true` nach
Disk-Swap-Resume):

| `event` | Wann | Schlüsselfelder |
|---|---|---|
| `server_ready` | Preseed fertig, Listener gebunden — ab hier testen | `addr`, `registered_models` (Modelle laden lazy beim ersten Request) |
| `admission` | Request angenommen | `pool_total/free`, `active_used`, `cached_evictable`, `unaccounted`, `gpu_free_mib`, `rss_mib`, `*_drift_mib` |
| `prefill` | Prefill fertig | `prompt_tokens`, `prefix_hit_tokens`, `reused_blocks`, `new_blocks`, `blocks_held` |
| `block_alloc` | Neuer Block im Decode | `phase` (`thinking`/`generation`), `new_blocks`, `blocks_held`, `output_tokens`, `pool_free` |
| `decode_summary` | Alle 128 Output-Tokens | `blocks_held`, `thinking_blocks`, `generation_blocks`, `pool_free` |
| `phase_transition` | `inside_thinking`-Flip | `thinking_tokens`, `thinking_blocks` (fehlt bei Modellen ohne Thinking, z. B. TranslateGemma — dort ist alles `generation`) |
| `finish` | Jeder Abschluss (stop/length/error/cancel/receiver-dropped/model_swap) | `blocks_released`, `freed_to_pool`, `kept_by_prefix_cache`, `held_after` (muss 0 sein), Per-Phase-Breakdown, `gpu_delta_mib`, `rss_delta_mib`, `*_drift_mib` |
| `abort` / `swap_out` / `resume` | Reguläre Nicht-Finish-Exits | `why` bzw. `blocks_released` |
| Idle-Zeile | Scheduler wird leer | `idle invariant ok` oder `error` `KV LEAK: blocks held by sequences while scheduler is idle` |

So sieht ein Leak aus:
- `finish` mit `held_after > 0` → `free_sequence` hat die Blocktabelle
  nicht geräumt (error-Zeile).
- Idle-`KV LEAK` mit `leaked_blocks > 0` → Pool-Blöcke, die weder dem
  Prefix-Cache noch dem permanenten Baseline-Overhead (z. B. der
  CUDA-Graph-Dummy-Block aus dem Modellbau) gehören.
- `error` „sequence dropped without finish/abort accounting" → ein
  Exit-Pfad hat die Freigabe-Buchung übersprungen.
- Monoton sinkende `gpu_drift_mib` über viele Requests → CUDA-Graphen
  oder andere GPU-Allokationen lecken (pro Request wird ein Graph
  gecaptured).

Rechen-Hinweis: `unaccounted = total − free − active_used −
cached_evictable` kann **negativ** werden, weil ein per Prefix-Cache
wiederverwendeter Block gleichzeitig in `active_used` und
`cached_evictable` zählt. Die harte Invariante ist die Idle-Prüfung.

---

## 8. NVMe-Tier, Druck-Scheduler, Sidecars (Stand 2026-10-06)

Umsetzung des Anforderungsdokuments vom 2026-10-05 (Analyse:
`docs/design/nvme-tier-analysis.md`). Die wichtigsten neuen Bausteine:

| Baustein | Ort |
|---|---|
| Verkettete Block-Schlüssel (BLAKE3-128, Medien-Digests) | `spark-runtime/src/block_keys.rs`, Radix-Keying in `radix_tree/inner.rs` |
| NVMe-Block-Store (Header/Checksummen/GC/Write-Queue) | `spark-storage/src/block_store/` |
| Persist/Restore-Glue (on_release, KV-7-Guard, Staging) | `spark-server/src/kv_tier.rs` + Scheduler-Hooks |
| SSM-Snapshots auf NVMe (Resume-Punkte, Tail-Records) | `trait_impl/nvme_tier.rs`, `kv_tier.rs` |
| Swap-Kosten (EWMA, `snapshots/swap_costs.json`) | `spark-server/src/swap_costs.rs` |
| Queues + reine Wechselentscheidung + Dispatcher | `spark-server/src/arena_sched/` |
| Warte-SSE (`: queued`/`: waiting`) | `api/queued_sse.rs` |
| Embed/Rerank-Sidecars | `main_modules/text_sidecar.rs` |
| `/internal/state`, neue Metriken | `api/internal_state.rs`, `metrics.rs` |
| Feste Session-Slots (E6: Resolution, Fit, Plan) | `spark-server/src/slots.rs`, Admission in `scheduler/mod_helpers.rs` |
| Session-KV-Tracking `POST /v1/atlas/session/state` | `api/session_state.rs`, `main_modules/cache_registry.rs` |

Ablauf eines Modellwechsels (Abschnitt 10 des Dokuments): Gate
(`api/lazy_swap.rs`) reiht Tickets ein → Dispatcher entscheidet
(`arena_sched/decision.rs`, rein + replay-getestet) → bestehende
`ControlMsg`-Drain/Unload/Load-Maschinerie → Persist-Flush
(`persist_timeout`) → Grant der Ziel-Tickets (Priorität zuerst).
Laufende Requests werden nach `--drain-grace-ms` mit
`finish_reason="model_swap"` beendet; ihr KV/SSM-Zustand ist zuvor
persistiert (E1), sodass der Folge-Request den Prefix wiederfindet.

Feste Slots (E6): `slots::resolve` löst beim (Re-)Load
CLI `--slot-tokens` → MODEL.toml `[behavior].slot_tokens` →
`--max-seq-len` auf (MOD-12 verweigert den Load bei Verstoß),
`publish_from_pool` leitet `slots_total` aus der echten Pool-Geometrie
ab. Slots sind logisch: Blöcke bleiben demand-paged, unbenutzte Blöcke
dienen dem Prefix-Cache (SLOT-3). Die Admission nimmt höchstens
`min(max_batch_size, slots_total)` Sequenzen an (BAT-8; Staging zählt
mit), normale Requests lassen `--priority-reserved-slots` frei (PRI-7),
Priority-Requests werden zuerst zugelassen (PRI-1). `max_blocks_per_seq`
der Attention ist auf die Slot-Größe dimensioniert — jede
Handler-Längenprüfung nutzt daher die Slot-Grenze (`slots::prompt_limit`).
Requests über der Slot-Grenze: Prompt zu groß → 400
`context_length_exceeded` (SLOT-4); `max_tokens` wird auf
`slot_buffer_tokens` und den Slot-Rest geklemmt; jede Antwort trägt
`X-Atlas-Context-Remaining` (SLOT-6; Streaming konservativ
Prompt+Budget, Blocking exakt).

Bewusst offen (Stufe 2 / SOLL): Write-through (NVM-13, inkl.
`--write-batch-interval-ms`), KV-9 (CoW für angebrochene Blöcke),
NVM-14 (Spill-Formatvereinheitlichung), SSM-6, PRI-5 (Aging), BAT-4
(gelernte Schrittzeit-Kurve — die SLAI-Policy mit `--tbt-deadline-ms`
deckt die Admission-Bremse heute ab), TEL-3-Paginierung.

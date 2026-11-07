# Atlas — verfügbare Modelle und Endpunkte

Stand: 2026-11-07, Deployment auf der GB10-Box (`serve.sh`, Commit `72c84d2bb`).
Quelle für die Modellliste ist der laufende Server (`GET /v1/models`),
Quelle für das Verhalten der Endpunkte ist `crates/spark-server/src/`.
Die vollständige Kompatibilitätsmatrix aller unterstützten Checkpoints steht
in [`GB10_DEPLOYMENT_GUIDE.md`](GB10_DEPLOYMENT_GUIDE.md) §2; dieses Dokument
beschreibt, was auf **diesem** Deployment konfiguriert ist und wie man es anspricht.

## 1. Basis-URL und Authentifizierung

| Zugriff | Basis-URL | Auth |
|---|---|---|
| Öffentlich (hinter Reverse-Proxy) | `https://atlas.soliton-sh.net` | `Authorization: Bearer $ATLAS_TOKEN` (alternativ `X-API-Key: $ATLAS_TOKEN` für Datalab-Clients) |
| Lokal auf der Box / im Dev-Container | `http://127.0.0.1:11434` | keine |

Hinweise:

- Der Reverse-Proxy hat ein Read-Timeout von ca. 60 s. Lange Anfragen
  (Bildgenerierung, große OCR-Dokumente, Modellwechsel + langer Prompt)
  gehören auf die lokale URL oder in den asynchronen `/convert`-Pfad.
- `/health`, `/health/live` und `/metrics` sind nie auth-geschützt.
  Alles unter `/v1/`, `/admin`, `/internal`, `/convert`, `/tokenize`,
  `/detokenize` ist es, sobald der Server mit `--require-auth` läuft bzw.
  der Proxy davor steht.
- Für den OpenAI-SDK-Client reicht `base_url=<BASE>/v1` und ein beliebiger
  `api_key` (lokal) bzw. der echte Token (öffentlich). Siehe
  [`examples/python/chat_completions.py`](../examples/python/chat_completions.py).

```bash
export ATLAS_BASE_URL=https://atlas.soliton-sh.net
export ATLAS_TOKEN=...   # nicht ins Repo
curl -s "$ATLAS_BASE_URL/v1/models" -H "Authorization: Bearer $ATLAS_TOKEN" | jq '.data[].id'
```

## 2. Modelle auf diesem Deployment

Die Modelle sind per Kategorie konfiguriert (`ATLAS_PRESEED_<KATEGORIE>` in
`serve.sh` / `.devpod/compose.yml`). Jedes Modell ist auf **eine** Kategorie
und damit auf **einen** Satz Endpunkte festgelegt. Ein Modell am falschen
Endpunkt ergibt `400 model_wrong_category`; ein unbekanntes Modell ergibt
`404 model_not_found` mit der Liste der verfügbaren Modelle im Body.

### 2.1 Generative Modelle (Chat, Completions, Responses, Anthropic Messages)

| Modell-ID (`model`) | Was es ist | Besonderheiten |
|---|---|---|
| `nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4` | 80B MoE (3B aktiv), Reasoning-Modell, Boot-Modell | Denkt immer (Template-forciert), `reasoning_content` wird separat ausgegeben. Schnellstes Modell im Set, ~75 tok/s Decode. |
| `vincentzed-hf/Qwen3-Coder-Next-NVFP4` | 80B MoE (3B aktiv), Coding-Spezialist | Tool-Calling mit `qwen3_coder`-Parser. Agentische Qualität sinkt nach ca. 5–8 Tool-Runden (Modell-Eigenschaft). |
| `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` | 30B MoE (3B aktiv), **Vision** | Bild-Eingabe über `image_url` in den Messages. Kein Thinking. |
| `nvidia/Qwen3.8-27B-NVFP4` | 27B dense, Reasoning + Vision | Thinking standardmäßig an (Budget 2048), FP8-KV. Langsamer (~10 tok/s), höhere Qualität pro Token. |

**Lazy Swap:** Es ist immer nur **ein** generatives Modell im Speicher.
Wird ein anderes angefragt, entlädt der Server das aktive Modell und lädt
das gewünschte aus dem Snapshot. Die auslösende Anfrage zahlt den Wechsel
(einige Sekunden bis ~30 s) und bekommt den Header `X-Atlas-Model-Swap: <ms>`.
Ein Warmup oder Preload ist nicht vorgesehen; das Verhalten ist gewollt.
Aktuellen Zustand zeigt `GET /admin/models` oder `GET /health`.

### 2.2 Spezialmodelle (eigene Endpunkte)

| Modell-ID | Kategorie | Endpunkt(e) |
|---|---|---|
| `Qwen/Qwen3-Embedding-8B` | embed | `POST /v1/embeddings` (Text) |
| `Qwen/Qwen3-VL-Embedding-8B` | embed | `POST /v1/embeddings` (Text und/oder Bild) |
| `Qwen/Qwen3-Reranker-8B` | rerank | `POST /v1/rerank`, `POST /rerank` (Text) |
| `Qwen/Qwen3-VL-Reranker-8B` | rerank | `POST /v1/rerank` (Text und/oder Bild-Dokumente) |
| `google/translategemma-12b-it` | translate | `POST /v1/translate` |
| `datalab-to/chandra-ocr-2` | ocr | `POST /v1/ocr` (synchron), `POST /convert` (asynchron, Datalab-kompatibel) |
| `openai/whisper-large-v3` | transcribe | `POST /v1/audio/transcriptions` |
| `Qwen/Qwen-Image-2.1` | image | `POST /v1/images/generations`, `POST /v1/images/edits` |

Embed-, Rerank-, Translate- und OCR-Modelle laufen in derselben Arena wie die
generativen Modelle und lösen ebenfalls einen Swap aus. Whisper und Qwen-Image
sind **Sidecars**: Sie werden beim ersten Request geladen, bleiben resident
und verdrängen das Chat-Modell nicht (außer der Speicher reicht nicht; dann
wird das Arena-Modell entladen und beim nächsten Chat-Request neu geladen).

## 3. Gemeinsame Konventionen

- **Fehlerformat:** OpenAI-Shape `{"error": {"message", "type", "code"}}`.
  Wichtige Codes: `model_not_found` (404), `model_wrong_category` (400),
  `capacity_exceeded` (503), `context_length_exceeded` (400).
  `/convert` antwortet stattdessen Datalab-Shape `{"success": false, "error", "error_code"}`.
- **`accepted_processing_time`** (Atlas-Erweiterung, Sekunden als Float):
  Der Server schätzt die Bearbeitungsdauer vorab und lehnt mit
  `503 capacity_exceeded` ab, wenn er die Frist nicht halten kann. Bei
  JSON-Endpunkten als Body-Feld, bei Multipart-Endpunkten als Formfeld oder
  Header `X-Atlas-Accepted-Processing-Time`. Die Schätzung kommt immer im
  Antwort-Header `X-Atlas-Estimated-Seconds` zurück (bei `/convert` im Feld
  `estimated_seconds`). Details: [`design/deadline-admission.md`](design/deadline-admission.md).
- **Kontextfenster:** Der Server läuft aktuell mit `--max-seq-len 16384`.
  Überlaufende Chat-Historien werden komprimiert statt abgelehnt
  (siehe [`design/context-compaction.md`](design/context-compaction.md)).
- **Body-Limit:** 32 MB (`ATLAS_MAX_BODY_BYTES`).
- **Request-ID:** jede Antwort trägt `X-Request-Id`.

## 4. Endpunkte im Detail

### 4.1 Chat Completions — `POST /v1/chat/completions`

OpenAI-kompatibel inkl. Streaming (SSE), Tools, `response_format`
(`json_object`, `json_schema`), `logprobs`, `seed`, `stop`, `n` (nur blocking).

```bash
curl -s "$ATLAS_BASE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $ATLAS_TOKEN" -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4",
    "messages": [{"role": "user", "content": "Erkläre in zwei Sätzen, was ein KV-Cache ist."}],
    "max_tokens": 512,
    "stream": false
  }'
```

Die Antwort enthält bei Reasoning-Modellen `choices[0].message.reasoning_content`
(und im Stream `delta.reasoning_content`) getrennt von `content`.
`usage.completion_tokens_details.reasoning_tokens` zählt die Denk-Tokens.

**Thinking steuern** (höchste Priorität zuerst, der Client gewinnt immer
gegen den Modell-Default aus `MODEL.toml`):

| Form | Beispiel |
|---|---|
| Anthropic-Stil | `"thinking": {"budget_tokens": 1024}` |
| vLLM-PR-Stil | `"thinking_token_budget": 1024` (Alias `max_thinking_tokens`) |
| OpenAI-Stil | `"reasoning": {"effort": "low"}` oder `"reasoning_effort": "low"` |
| vLLM-stable | `"chat_template_kwargs": {"enable_thinking": true, "thinking_budget": 1024}` |
| Atlas-legacy | `"enable_thinking": true` (Default-Budget 256 Tokens) |

Das 80B-Thinking-Modell denkt template-bedingt immer; ein Budget begrenzt
nur die Länge. Qwen3.8-27B denkt per Default, lässt sich aber mit
`"chat_template_kwargs": {"enable_thinking": false}` abschalten.

**Vision** (Qwen3-VL-30B, Qwen3.8-27B): Bild als `image_url` im Content-Array,
**nur** als `data:`-URI oder rohes Base64 (JPEG/PNG). Externe http(s)-URLs
werden nicht abgerufen; der Client muss das Bild selbst einbetten.

```json
{"model": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
 "messages": [{"role": "user", "content": [
   {"type": "text", "text": "Was steht auf dem Schild?"},
   {"type": "image_url", "image_url": {"url": "data:image/png;base64,...."}}]}],
 "max_tokens": 300}
```

**Tool-Calling:** `tools` / `tool_choice` wie bei OpenAI; Atlas emittiert
derzeit einen Tool-Call pro Turn (`parallel_tool_calls` wird akzeptiert,
ändert aber nichts). Streaming liefert Tool-Calls live als `delta.tool_calls`.

Weitere Sampling-Parameter neben den OpenAI-Standardfeldern: `top_k`,
`min_p`, `top_n_sigma`, `repetition_penalty`, `min_tokens`,
`repetition_detection` (Loop-Watchdog-Override), `return_token_ids`.
Ohne Angabe gelten die Presets aus der jeweiligen `kernels/gb10/<modell>/MODEL.toml`.

Python mit dem OpenAI-SDK:

```python
from openai import OpenAI
client = OpenAI(base_url=f"{ATLAS_BASE_URL}/v1", api_key=ATLAS_TOKEN)
r = client.chat.completions.create(
    model="vincentzed-hf/Qwen3-Coder-Next-NVFP4",
    messages=[{"role": "user", "content": "Schreibe eine Rust-Funktion, die FizzBuzz ausgibt."}],
    max_tokens=800, stream=True)
for chunk in r:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### 4.2 Weitere generative Oberflächen

| Endpunkt | Zweck |
|---|---|
| `POST /v1/completions` | Roh-Prompt ohne Chat-Template. `prompt` als String, String-Array, Token-ID-Array oder Batch von Token-ID-Arrays (letztere umgehen die Tokenisierung). |
| `POST /v1/responses`, `GET/DELETE /v1/responses/{id}`, `GET …/input_items`, `POST …/cancel` | OpenAI Responses API inkl. Streaming und gespeicherten Responses. |
| `POST /v1/conversations`, `GET/POST/DELETE /v1/conversations/{id}`, `…/items` | Conversation-Objekte zur Responses API. |
| `POST /v1/messages`, `POST /v1/messages/count_tokens` | **Anthropic Messages API** (Streaming und blocking). Für Clients, die das Anthropic-SDK sprechen: `base_url=$ATLAS_BASE_URL`, `api_key=$ATLAS_TOKEN`, `model` = eine der generativen IDs. |
| `GET /v1/chat/completions/{id}` | Abruf gespeicherter Completions. |
| `POST /tokenize`, `POST /detokenize` | `{"model", "prompt"}` oder `{"model", "messages"}` → `{"tokens", "count"}`. |

Beispiel Anthropic-Format:

```bash
curl -s "$ATLAS_BASE_URL/v1/messages" \
  -H "Authorization: Bearer $ATLAS_TOKEN" -H "Content-Type: application/json" \
  -d '{"model": "nvidia/Qwen3.8-27B-NVFP4", "max_tokens": 400,
       "thinking": {"type": "enabled", "budget_tokens": 1024},
       "messages": [{"role": "user", "content": "Nenne drei Risiken bei FP8-KV-Caches."}]}'
```

### 4.3 Embeddings — `POST /v1/embeddings`

OpenAI-Shape. `input` ist ein String, ein String-Array oder ein Array aus
`{"text": …}` / `{"image": "<data-URI oder base64>"}` Objekten (Bilder nur
mit `Qwen/Qwen3-VL-Embedding-8B`). Vektoren sind L2-normalisiert.

- `dimensions` (optional): Matryoshka-Kürzung, 32…4096.
- `instruction` (optional, nicht-Standard): Task-Instruktion für Queries
  (`Instruct: {task}\nQuery: {text}`); Dokumente ohne Instruktion einbetten.
- Alternativ `multipart/form-data` mit `file`-Feldern für Bilder.

```bash
curl -s "$ATLAS_BASE_URL/v1/embeddings" \
  -H "Authorization: Bearer $ATLAS_TOKEN" -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-Embedding-8B",
       "input": ["Wie konfiguriere ich den KV-Pool?", "Atlas KV-Pool Dokumentation"],
       "dimensions": 1024,
       "instruction": "Given a web search query, retrieve relevant passages"}'
```

### 4.4 Rerank — `POST /v1/rerank` (Alias `POST /rerank`)

Jina/Cohere-kompatibel.

```bash
curl -s "$ATLAS_BASE_URL/v1/rerank" \
  -H "Authorization: Bearer $ATLAS_TOKEN" -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-Reranker-8B",
       "query": "Wie starte ich den Server neu?",
       "documents": ["serve.sh baut und startet spark serve", "Die Lizenz ist AGPL-3.0", "pkill beendet den Prozess"],
       "top_n": 2, "return_documents": true}'
```

Antwort: `results[] = {index, relevance_score, document?}`; `index` zeigt auf
die Position im Request-Array. `instruction` überschreibt die Default-Aufgabe.
Mit `Qwen/Qwen3-VL-Reranker-8B` dürfen Dokumente `{"image": …}`-Objekte sein
(oder Multipart mit `documents`-Dateifeldern).

### 4.5 Übersetzung — `POST /v1/translate`

Strukturierte Anfrage, **kein** Chat-Format (TranslateGemma lässt sich nicht
über `/v1/chat/completions` ansprechen). Prompt-Budget ca. 2K Tokens; längere
Texte werden mit 400 abgelehnt, Chunking liegt beim Client.

```bash
curl -s "$ATLAS_BASE_URL/v1/translate" \
  -H "Authorization: Bearer $ATLAS_TOKEN" -H "Content-Type: application/json" \
  -d '{"model": "google/translategemma-12b-it",
       "source_lang_code": "de", "target_lang_code": "en",
       "text": "Der Server lädt Modelle erst beim ersten Request."}'
```

Antwort: `{"text", "model", "source_lang", "target_lang", "usage"}`.
Sprachcodes entsprechen der `languages`-Map des Checkpoints (ISO-639-1-Stil).

### 4.6 OCR synchron — `POST /v1/ocr`

Chandra-OCR-2. `image` ist ein `data:`-URI oder Roh-Base64 eines Bildes
(PNG/JPEG/WebP) **oder eines PDFs** (wird serverseitig per pdfium gerendert).

```bash
curl -s "$ATLAS_BASE_URL/v1/ocr" \
  -H "Authorization: Bearer $ATLAS_TOKEN" -H "Content-Type: application/json" \
  -d "{\"model\": \"datalab-to/chandra-ocr-2\",
       \"image\": \"data:application/pdf;base64,$(base64 -w0 rechnung.pdf)\",
       \"page_range\": \"0-1\", \"include_images\": false}"
```

Felder: `page_range` (0-basiert, `"0,2-5,9"`), `prompt_type`
(`ocr_layout` Default | `ocr`), `max_tokens`, `max_retries`,
`include_headers_footers`, `include_images`, `accepted_processing_time`.
Antwort pro Seite: `html`, `markdown`, `chunks` (Layout-Blöcke mit Bounding
Boxes), `images` (WebP-Crops als Base64), `retries`.
Für mehrseitige Dokumente über den Proxy ist `/convert` die bessere Wahl.

### 4.7 OCR asynchron — `POST /convert` (Datalab-kompatibel)

Ein unveränderter Datalab/Marker-Client funktioniert, wenn er auf Atlas zeigt.
Auth über `X-API-Key` oder Bearer.

```bash
# 1) Einreichen (multipart)
curl -s "$ATLAS_BASE_URL/convert" -H "X-API-Key: $ATLAS_TOKEN" \
  -F file=@rechnung.pdf -F page_range=0-9 -F accepted_processing_time=120
# → {"success": true, "request_id": "...", "request_check_url": "https://.../convert/<id>", "estimated_seconds": 37}

# 2) Pollen
curl -s "$ATLAS_BASE_URL/convert/<id>" -H "X-API-Key: $ATLAS_TOKEN"
# → {"status": "pending" | "processing"} … → {"status": "complete", ...Marker-Dokument...}

# 3) Abbrechen (idempotent)
curl -s -X DELETE "$ATLAS_BASE_URL/convert/<id>" -H "X-API-Key: $ATLAS_TOKEN"
```

Akzeptierte Formfelder: `file` (Pflicht), `page_range`, `max_pages`,
`skip_cache`, `mode` (ignoriert), `output_format`, `extras` u. a. (ignoriert),
`accepted_processing_time`. Ergebnisse sind inhaltsadressiert gecacht
(gleiches Dokument ⇒ identische Block-IDs). Fehlgeschlagene Seiten stehen in
`metadata.failed_pages`; nur `error_code: unprocessable_document` bedeutet
„Dokument ist schuld“, alles andere ist serverseitig. Jobs, die niemand
abfragt, werden nach einer Idle-Zeit abgebrochen (`abandoned`).

### 4.8 Transkription — `POST /v1/audio/transcriptions`

Whisper-large-v3 als Sidecar, OpenAI-kompatibles Multipart.

```bash
curl -s "$ATLAS_BASE_URL/v1/audio/transcriptions" \
  -H "Authorization: Bearer $ATLAS_TOKEN" \
  -F file=@aufnahme.wav -F model=openai/whisper-large-v3 \
  -F language=de -F response_format=json
```

`response_format`: `json` (Default, `{"text"}`), `text`, `verbose_json`.
`prompt`, `temperature`, `timestamp_granularities` werden akzeptiert und
ignoriert. `/v1/audio/translations` und `/v1/audio/speech` antworten 501.

### 4.9 Bildgenerierung — `POST /v1/images/generations`

Qwen-Image-2.1 als Sidecar (ca. 27 GiB resident). Eine Generierung läuft zur
Zeit; weitere Requests warten. 1024² mit 40 Schritten dauert ca. 60 s, 2048²
ca. 450 s. Über den Proxy (60 s Timeout) nur kleine Bilder; sonst lokal.

```bash
curl -s "$ATLAS_BASE_URL/v1/images/generations" \
  -H "Authorization: Bearer $ATLAS_TOKEN" -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen-Image-2.1",
       "prompt": "Ein Leuchtturm im Nebel, Ölgemälde",
       "size": "1024x1024", "num_inference_steps": 30, "seed": 42,
       "response_format": "b64_json", "output_format": "webp"}' \
  | jq -r '.data[0].b64_json // empty' | base64 -d > leuchtturm.webp
```

Felder: `prompt`, `n`, `size` (`"WxH"` oder `"auto"`), `aspect_ratio`
(`"16:9"`, wenn kein `size`), `response_format` (`b64_json` | `url`; `url`
liefert eine `data:`-URI, Atlas hostet keine Dateien), `output_format`
(`png` | `jpeg` | `webp`), `background` (`auto` | `opaque` | `transparent`),
`seed`, `num_inference_steps` (Alias `steps`), `accepted_processing_time`.
Reicht der Speicher nicht, antwortet der Server 503 statt in den OOM-Killer
zu laufen; vorher versucht er, das Chat-Modell zu entladen.

### 4.10 Bildbearbeitung — `POST /v1/images/edits`

Bildkonditionierte Generierung (bis 10 Referenzbilder). OpenAI-Multipart
(`image`-Dateifelder + Textfelder) oder JSON mit `image` als Base64-String
bzw. Array. Zielgröße folgt dem Seitenverhältnis des letzten Bildes.

```bash
curl -s "$ATLAS_BASE_URL/v1/images/edits" \
  -H "Authorization: Bearer $ATLAS_TOKEN" \
  -F image=@wuerfel.png -F model=Qwen/Qwen-Image-2.1 \
  -F prompt="Mache den Würfel blau" -F response_format=b64_json
```

`/v1/images/variations` antwortet 501 (kein promptfreier Modus).

### 4.11 Betrieb und Diagnose

| Endpunkt | Inhalt |
|---|---|
| `GET /health` | `{"status": "ready", "model": "<aktives Modell>"}` |
| `GET /health/live` | Liveness |
| `GET /metrics` | Prometheus |
| `GET /v1/models`, `GET /v1/models/{id}` | Modellliste (alle Kategorien inkl. Sidecars) |
| `GET /admin/models` | Lazy-Swap-Zustand: welches Modell resident ist, Snapshots, Arena-Belegung |
| `POST /admin/models/load` · `/unload` · `/snapshot` | Manuelles Laden/Entladen/Snapshotten |
| `GET /v1/stats` | Speicher (CUDA vs. `/proc/meminfo`), KV-Budget, Residenz pro Modell |
| `GET /internal/state` | Arena, Sidecars, KV-Pool, Scheduler-Queues, Kompaktierungszähler, Kapazitätsprofil |
| `POST /v1/atlas/session/state` | Prüft, ob der Prompt-Präfix einer Session noch im RAM-/NVMe-Cache liegt |

Stubs mit 501 und OpenAI-Fehlerbody (damit Auto-Probing-Clients sauber
zurückfallen): `/v1/batches*`, `/v1/files*`, `/v1/moderations`.

## 5. Typische Stolpersteine

- **Erster Request nach Modellwechsel ist langsam.** Das ist der Lazy Swap,
  kein Hänger. `X-Atlas-Model-Swap` im Response-Header zeigt die Dauer.
- **`model_wrong_category`.** Embedding-, Rerank-, Translate- oder OCR-Modell
  an `/v1/chat/completions` geschickt (oder umgekehrt). Tabelle in §2 prüfen.
- **TranslateGemma per Chat.** Geht nicht, nur `/v1/translate`.
- **Proxy-Timeout.** Alles, was länger als etwa 60 s dauert, über
  `/convert` (OCR) oder lokal fahren; `accepted_processing_time` setzen,
  damit der Server statt des Proxys ablehnt.
- **Thinking-Tokens fressen `max_tokens`.** Bei Reasoning-Modellen ein
  `thinking_token_budget` setzen oder `max_tokens` großzügig wählen; das
  Budget wird hart durchgesetzt (`</think>` wird erzwungen).
- **`jq` ohne `// empty`.** Bei einem 503 landet sonst der Fehler-JSON in
  der Bilddatei.

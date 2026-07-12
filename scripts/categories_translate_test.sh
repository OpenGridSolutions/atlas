#!/usr/bin/env bash
# Acceptance test for categorized preseed + /v1/translate (multiclass
# Plan.md Phase K + T).
#
# Run ON THE BOX against a RUNNING server (five-category preseed):
#   scripts/categories_translate_test.sh
# Env: BASE (default http://localhost:11434),
#      CHAT_MODEL      (default vincentzed-hf/Qwen3-Coder-Next-NVFP4)
#      EMBED_MODEL     (default Qwen/Qwen3-Embedding-8B)
#      RERANK_MODEL    (default Qwen/Qwen3-Reranker-8B)
#      WHISPER_MODEL   (default openai/whisper-large-v3)
#      TRANSLATE_MODEL (default google/translategemma-12b-it)
#
# Checks:
#   C1  real names accepted on the RIGHT endpoints (embed + translate)
#   C2  wrong endpoint -> explaining 400 (embed model on chat; translate
#       model on chat; chat model on translate; embed model on audio)
#   C3  openai/whisper-large-v3 REAL name transcribes (whisper-1 alias too)
#   T1  de->en /v1/translate: plausible English (keyword, case-insensitive)
#   T2  de->fr: recognizably French (arbitrary target language)
#   T3  forced-template proof: bogus lang code -> 400 naming the language
#       list (the server renders the frame itself), and the shipped chat
#       template rejects OpenAI-shaped messages so no generic-template
#       fallback path exists
#   R1  every category still answers: chat, embed, rerank, transcribe
#
# NOTE: server responses are written to a temp FILE and read by python via
# json.load(open(path)). We deliberately do NOT pass the body as argv (large
# responses exceed ARG_MAX) and do NOT pipe the body into `python3 - <<EOF`
# (stdin is already consumed by the heredoc script body).
set -euo pipefail

BASE="${BASE:-http://localhost:11434}"
CHAT_MODEL="${CHAT_MODEL:-vincentzed-hf/Qwen3-Coder-Next-NVFP4}"
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}"
RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-Reranker-8B}"
WHISPER_MODEL="${WHISPER_MODEL:-openai/whisper-large-v3}"
TRANSLATE_MODEL="${TRANSLATE_MODEL:-google/translategemma-12b-it}"
JFK="${JFK:-/workspace/test_data/jfk.wav}"
PASS=0; FAIL=0
ok()   { echo "PASS: $1"; PASS=$((PASS+1)); }
bad()  { echo "FAIL: $1"; FAIL=$((FAIL+1)); }

RESP="$(mktemp)"
export RESP
trap 'rm -f "$RESP"' EXIT

# curl helper capturing the HTTP status into $CODE and the body into $RESP.
post_json() { # $1 = path, $2 = body
  CODE=$(curl -sS -o "$RESP" -w '%{http_code}' "$BASE$1" \
    -H 'Content-Type: application/json' -d "$2")
  export CODE
}

translate() { # $1 = src, $2 = tgt, $3 = text
  post_json /v1/translate \
    "{\"source_lang_code\":\"$1\",\"target_lang_code\":\"$2\",\"text\":\"$3\"}"
}

echo "== C1: real names on the right endpoints =="
post_json /v1/embeddings "{\"model\":\"$EMBED_MODEL\",\"input\":\"hello world\"}"
python3 - <<'EOF' && ok "C1a embed real name" || bad "C1a embed real name"
import os, json
assert os.environ["CODE"] == "200", os.environ["CODE"]
d = json.load(open(os.environ["RESP"]))
assert len(d["data"][0]["embedding"]) == 4096
EOF
translate de en "Guten Morgen"
python3 - <<'EOF' && ok "C1b translate real name" || bad "C1b translate real name"
import os, json
assert os.environ["CODE"] == "200", os.environ["CODE"]
d = json.load(open(os.environ["RESP"]))
assert d["text"].strip(), d
EOF

echo "== C2: wrong endpoint -> explaining 400 =="
post_json /v1/chat/completions \
  "{\"model\":\"$EMBED_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"
python3 - <<'EOF' && ok "C2a embed model on chat -> 400" || bad "C2a embed model on chat"
import os, json
assert os.environ["CODE"] == "400", os.environ["CODE"]
msg = json.load(open(os.environ["RESP"]))["error"]["message"]
assert "categorized as embed" in msg and "/v1/embeddings" in msg, msg
EOF
post_json /v1/chat/completions \
  "{\"model\":\"$TRANSLATE_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"
python3 - <<'EOF' && ok "C2b translate model on chat -> 400" || bad "C2b translate model on chat"
import os, json
assert os.environ["CODE"] == "400", os.environ["CODE"]
msg = json.load(open(os.environ["RESP"]))["error"]["message"]
assert "categorized as translate" in msg and "/v1/translate" in msg, msg
EOF
post_json /v1/translate \
  "{\"model\":\"$CHAT_MODEL\",\"source_lang_code\":\"de\",\"target_lang_code\":\"en\",\"text\":\"Hallo\"}"
python3 - <<'EOF' && ok "C2c chat model on translate -> 400" || bad "C2c chat model on translate"
import os, json
assert os.environ["CODE"] == "400", os.environ["CODE"]
msg = json.load(open(os.environ["RESP"]))["error"]["message"]
assert "categorized as generative" in msg and "/v1/chat/completions" in msg, msg
EOF
CODE=$(curl -sS -o "$RESP" -w '%{http_code}' "$BASE/v1/audio/transcriptions" \
  -F "file=@$JFK" -F "model=$EMBED_MODEL"); export CODE
python3 - <<'EOF' && ok "C2d embed model on audio -> 400" || bad "C2d embed model on audio"
import os, json
assert os.environ["CODE"] == "400", os.environ["CODE"]
msg = json.load(open(os.environ["RESP"]))["error"]["message"]
assert "categorized as embed" in msg, msg
EOF

echo "== C3: whisper REAL name on /v1/audio/transcriptions =="
CODE=$(curl -sS -o "$RESP" -w '%{http_code}' "$BASE/v1/audio/transcriptions" \
  -F "file=@$JFK" -F "model=$WHISPER_MODEL"); export CODE
python3 - <<'EOF' && ok "C3 whisper real name" || bad "C3 whisper real name"
import os, json
assert os.environ["CODE"] == "200", os.environ["CODE"]
t = json.load(open(os.environ["RESP"]))["text"].lower()
assert "country" in t, t
EOF

echo "== T1: de->en translation =="
translate de en "Die Katze sitzt auf dem Tisch und trinkt Milch."
python3 - <<'EOF' && ok "T1 de->en keywords" || bad "T1 de->en keywords"
import os, json
assert os.environ["CODE"] == "200", os.environ["CODE"]
d = json.load(open(os.environ["RESP"]))
t = d["text"].lower()
print(f"  en: {d['text']!r}")
assert "cat" in t and ("table" in t or "desk" in t) and "milk" in t, t
assert d["target_lang"] == "English", d
EOF

echo "== T2: de->fr (arbitrary target) =="
translate de fr "Der Hund schläft im Garten unter dem großen Baum."
python3 - <<'EOF' && ok "T2 de->fr French" || bad "T2 de->fr French"
import os, json
assert os.environ["CODE"] == "200", os.environ["CODE"]
d = json.load(open(os.environ["RESP"]))
t = d["text"].lower()
print(f"  fr: {d['text']!r}")
assert "chien" in t and ("jardin" in t or "arbre" in t), t
assert d["target_lang"] == "French", d
EOF

echo "== T3: forced-template path =="
# 3a: the server itself resolves language codes from the checkpoint's own
# template map — a bogus code fails BEFORE any generation, proving the
# frame is rendered server-side, not by a generic chat template.
translate xx en "Hallo"
python3 - <<'EOF' && ok "T3a bogus lang code -> 400 from server-side frame" || bad "T3a bogus lang code"
import os, json
assert os.environ["CODE"] == "400", os.environ["CODE"]
msg = json.load(open(os.environ["RESP"]))["error"]["message"]
assert "source_lang_code" in msg and "language list" in msg, msg
EOF
# 3b: no OpenAI-shaped fallback exists for this model: the chat endpoint
# refuses it by category (C2b) and /v1/translate accepts only the
# structured fields — a messages-array body is rejected as invalid JSON
# shape (missing required fields), never silently reinterpreted.
post_json /v1/translate \
  "{\"messages\":[{\"role\":\"user\",\"content\":\"translate this\"}]}"
python3 - <<'EOF' && ok "T3b messages-shaped body -> 400" || bad "T3b messages-shaped body"
import os, json
assert os.environ["CODE"] == "400", os.environ["CODE"]
msg = json.load(open(os.environ["RESP"]))["error"]["message"]
assert "source_lang_code" in msg, msg
EOF

echo "== R1: every category still answers =="
post_json /v1/chat/completions \
  "{\"model\":\"$CHAT_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: pong\"}],\"max_tokens\":2000}"
python3 - <<'EOF' && ok "R1a chat" || bad "R1a chat"
import os, json
assert os.environ["CODE"] == "200", os.environ["CODE"]
c = json.load(open(os.environ["RESP"]))["choices"][0]["message"]["content"]
assert "pong" in c.lower(), c
EOF
post_json /v1/embeddings "{\"model\":\"$EMBED_MODEL\",\"input\":\"still alive\"}"
python3 - <<'EOF' && ok "R1b embed" || bad "R1b embed"
import os, json
assert os.environ["CODE"] == "200", os.environ["CODE"]
assert len(json.load(open(os.environ["RESP"]))["data"][0]["embedding"]) == 4096
EOF
post_json /v1/rerank \
  "{\"model\":\"$RERANK_MODEL\",\"query\":\"capital of France\",\"documents\":[\"Paris is the capital of France.\",\"Bread needs flour.\"]}"
python3 - <<'EOF' && ok "R1c rerank" || bad "R1c rerank"
import os, json
assert os.environ["CODE"] == "200", os.environ["CODE"]
d = json.load(open(os.environ["RESP"]))
rs = sorted(d["results"], key=lambda r: r["relevance_score"], reverse=True)
assert rs[0]["index"] == 0, d
EOF
CODE=$(curl -sS -o "$RESP" -w '%{http_code}' "$BASE/v1/audio/transcriptions" \
  -F "file=@$JFK" -F "model=whisper-1"); export CODE
python3 - <<'EOF' && ok "R1d transcribe (whisper-1 alias)" || bad "R1d transcribe"
import os, json
assert os.environ["CODE"] == "200", os.environ["CODE"]
assert "country" in json.load(open(os.environ["RESP"]))["text"].lower()
EOF

echo
echo "== $PASS PASS, $FAIL FAIL =="
exit $((FAIL > 0))

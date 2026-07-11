#!/usr/bin/env bash
# Acceptance test for /v1/audio/transcriptions (multiclass Plan.md W.4).
#
# Run ON THE BOX against a RUNNING server:
#   scripts/whisper_test.sh
#
# Fixture: test_data/whisper_jfk.wav (the canonical 11 s whisper.cpp JFK
# sample, 16 kHz mono: "...ask not what your country can do for you...").
# A3's >30 s fixture is built on the fly: JFK at 0 s and again at ~35 s
# (each utterance fully inside its own 30 s chunk), so the keyword
# appearing TWICE proves the second chunk was transcribed.
#
# Checks:
#   A1  short (<30 s) audio -> non-empty transcript
#   A2  transcript contains expected keywords (case-insensitive)
#   A3  >30 s audio -> chunking works, late-audio keyword present (2x)
#   A4  chat model still answers afterwards (sidecar never swaps models)
#
# NOTE first request pays sidecar init (fp32 load + BF16 convert, ~1 min).
# Response bodies go through temp files (ARG_MAX / heredoc-stdin rules).
set -euo pipefail

BASE="${BASE:-http://localhost:11434}"
CHAT_MODEL="${CHAT_MODEL:-vincentzed-hf/Qwen3-Coder-Next-NVFP4}"
FIX="$(cd "$(dirname "$0")/.." && pwd)/test_data/whisper_jfk.wav"
export FIX
PASS=0; FAIL=0
ok()  { echo "PASS: $1"; PASS=$((PASS+1)); }
bad() { echo "FAIL: $1"; FAIL=$((FAIL+1)); }

RESP="$(mktemp)"; export RESP
LONGWAV="$(mktemp --suffix=.wav)"; export LONGWAV
trap 'rm -f "$RESP" "$LONGWAV"' EXIT

echo "== A1/A2: short audio =="
curl -sS "$BASE/v1/audio/transcriptions" \
  -F "file=@$FIX" -F "model=whisper-1" -F "language=en" > "$RESP"
python3 - <<'EOF' && ok "A1 non-empty transcript" || bad "A1"
import os, json
d = json.load(open(os.environ["RESP"]))
t = d.get("text", "")
print(f"  transcript: {t[:120]!r}")
assert t.strip(), "empty transcript"
EOF
python3 - <<'EOF' && ok "A2 keywords present" || bad "A2"
import os, json
t = json.load(open(os.environ["RESP"])).get("text", "").lower()
for kw in ("country", "americans"):
    assert kw in t, f"keyword {kw!r} missing from transcript"
EOF

echo "== A3: >30s audio (JFK at 0s and ~35s) =="
python3 - <<'EOF'
import os, wave
src = wave.open(os.environ["FIX"])
frames = src.readframes(src.getnframes())
rate = src.getframerate()
gap = b"\x00\x00" * int(rate * 24)  # 24 s silence between utterances
out = wave.open(os.environ["LONGWAV"], "wb")
out.setnchannels(1); out.setsampwidth(2); out.setframerate(rate)
out.writeframes(frames + gap + frames)  # ~11 + 24 + 11 = 46 s
out.close()
EOF
curl -sS "$BASE/v1/audio/transcriptions" \
  -F "file=@$LONGWAV" -F "model=whisper-1" -F "language=en" > "$RESP"
python3 - <<'EOF' && ok "A3 late-chunk keyword (2x)" || bad "A3"
import os, json
t = json.load(open(os.environ["RESP"])).get("text", "").lower()
n = t.count("country")
print(f"  'country' occurrences: {n}; transcript: {t[:160]!r}...")
assert n >= 2, "second 30s chunk was not transcribed"
EOF

echo "== A4: chat model still serves =="
curl -sS "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\":\"$CHAT_MODEL\",
  \"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: OK\"}],
  \"max_tokens\":10,\"stream\":false}" > "$RESP"
python3 - <<'EOF' && ok "A4 chat after whisper" || bad "A4"
import os, json
d = json.load(open(os.environ["RESP"]))
assert d["choices"][0]["message"]["content"].strip(), "empty chat reply"
EOF

echo
echo "== $PASS PASS, $FAIL FAIL =="
[ "$FAIL" -eq 0 ]

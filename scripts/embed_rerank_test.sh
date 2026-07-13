#!/usr/bin/env bash
# Acceptance test for /v1/embeddings + /v1/rerank (multiclass Plan.md 1.4).
#
# Run ON THE BOX against a RUNNING server (full 6-model preseed list):
#   scripts/embed_rerank_test.sh
# Env: BASE (default http://localhost:11434),
#      EMBED_MODEL (default Qwen/Qwen3-Embedding-8B),
#      RERANK_MODEL (default Qwen/Qwen3-Reranker-8B),
#      CHAT_MODEL (default vincentzed-hf/Qwen3-Coder-Next-NVFP4)
#
# Checks:
#   E1  embedding is 4096-dim, unit-norm
#   E2  self-similarity == 1 (deterministic), related > unrelated
#   E3  dimensions=64 Matryoshka truncation, unit-norm
#   E4  multipart form field embeds identically to the JSON path (Phase U)
#   R1  rerank orders an obvious (query, docs) set correctly, scores in [0,1]
#   R2  multipart `documents` fields rank identically to R1 (Phase U)
#   S1  chat model still answers after the swaps (lazy multi-model intact)
# Image-file multipart (the actual Phase U point) needs the VL embed/rerank
# models resident — covered by the laptop endpoint-matrix test instead.
#
# NOTE: server responses are written to a temp FILE and read by python via
# json.load(open(path)). We deliberately do NOT pass the body as argv (multi-
# vector embedding responses exceed ARG_MAX -> "Argument list too long") and we
# do NOT pipe the body into `python3 - <<EOF` (stdin is already consumed by the
# heredoc script body -> python reads the wrong stream -> "Expecting value").
set -euo pipefail

BASE="${BASE:-http://localhost:11434}"
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}"
RERANK_MODEL="${RERANK_MODEL:-Qwen/Qwen3-Reranker-8B}"
CHAT_MODEL="${CHAT_MODEL:-vincentzed-hf/Qwen3-Coder-Next-NVFP4}"
PASS=0; FAIL=0
ok()   { echo "PASS: $1"; PASS=$((PASS+1)); }
bad()  { echo "FAIL: $1"; FAIL=$((FAIL+1)); }

# scratch file for the current response body; RESP is exported so the heredoc
# python (which runs in a child process) can pick it up from the environment.
RESP="$(mktemp)"
export RESP
trap 'rm -f "$RESP"' EXIT

embed() { # $1 = json input value, $2 = extra fields (optional) -> body in $RESP
  curl -sS "$BASE/v1/embeddings" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$EMBED_MODEL\",\"input\":$1$2}" > "$RESP"
}

echo "== E1: dimension + unit norm =="
embed '"The quick brown fox jumps over the lazy dog."' ''
python3 - <<'EOF' && ok "E1 4096-dim unit-norm" || bad "E1"
import os, json, math
d = json.load(open(os.environ["RESP"])); v = d["data"][0]["embedding"]
assert len(v) == 4096, f"dim {len(v)}"
n = math.sqrt(sum(x*x for x in v))
assert abs(n - 1.0) < 1e-3, f"norm {n}"
EOF

echo "== E2: self-similarity + related > unrelated =="
embed '["What is the capital of France?","The capital of France is Paris.","How do I bake sourdough bread?","What is the capital of France?"]' ''
python3 - <<'EOF' && ok "E2 similarity ordering" || bad "E2"
import os, json
d = json.load(open(os.environ["RESP"]))
vs = [e["embedding"] for e in sorted(d["data"], key=lambda e: e["index"])]
dot = lambda a, b: sum(x*y for x, y in zip(a, b))
self_sim = dot(vs[0], vs[3])
related = dot(vs[0], vs[1])
unrelated = dot(vs[0], vs[2])
print(f"  self={self_sim:.4f} related={related:.4f} unrelated={unrelated:.4f}")
assert self_sim > 0.999, "identical inputs must embed identically"
assert related > unrelated + 0.05, "related must beat unrelated clearly"
EOF

echo "== E3: Matryoshka dimensions=64 =="
embed '"hello world"' ',"dimensions":64'
python3 - <<'EOF' && ok "E3 dimensions=64" || bad "E3"
import os, json, math
v = json.load(open(os.environ["RESP"]))["data"][0]["embedding"]
assert len(v) == 64, f"dim {len(v)}"
n = math.sqrt(sum(x*x for x in v))
assert abs(n - 1.0) < 1e-3, f"norm {n}"
EOF

echo "== E4: multipart == JSON (Phase U) =="
embed '"the multipart bridge test sentence"' ''
cp "$RESP" "${RESP}.json"
curl -sS "$BASE/v1/embeddings" \
  -F "model=$EMBED_MODEL" \
  -F "input=the multipart bridge test sentence" > "$RESP"
python3 - <<'EOF' && ok "E4 multipart == JSON" || bad "E4"
import os, json
a = json.load(open(os.environ["RESP"] + ".json"))["data"][0]["embedding"]
b = json.load(open(os.environ["RESP"]))["data"][0]["embedding"]
sim = sum(x*y for x, y in zip(a, b))
print(f"  cosine(json, multipart) = {sim:.6f}")
assert sim > 0.9999, "multipart field must take the identical encode path"
EOF
rm -f "${RESP}.json"

echo "== R1: rerank ordering =="
curl -sS "$BASE/v1/rerank" -H 'Content-Type: application/json' -d "{
  \"model\":\"$RERANK_MODEL\",
  \"query\":\"What is the capital of France?\",
  \"documents\":[
    \"Sourdough bread needs a mature starter and long fermentation.\",
    \"Paris is the capital and largest city of France.\",
    \"Berlin is the capital of Germany.\"
  ]}" > "$RESP"
python3 - <<'EOF' && ok "R1 rerank ordering" || bad "R1"
import os, json
d = json.load(open(os.environ["RESP"]))
rs = d["results"]
print("  " + ", ".join(f"idx{r['index']}={r['relevance_score']:.4f}" for r in rs))
assert all(0.0 <= r["relevance_score"] <= 1.0 for r in rs), "scores out of [0,1]"
assert rs[0]["index"] == 1, "Paris document must rank first"
assert rs[0]["relevance_score"] > rs[-1]["relevance_score"], "no separation"
EOF

echo "== R2: multipart rerank == R1 (Phase U) =="
curl -sS "$BASE/v1/rerank" \
  -F "model=$RERANK_MODEL" \
  -F "query=What is the capital of France?" \
  -F "documents=Sourdough bread needs a mature starter and long fermentation." \
  -F "documents=Paris is the capital and largest city of France." \
  -F "documents=Berlin is the capital of Germany." > "$RESP"
python3 - <<'EOF' && ok "R2 multipart rerank" || bad "R2"
import os, json
d = json.load(open(os.environ["RESP"]))
rs = d["results"]
print("  " + ", ".join(f"idx{r['index']}={r['relevance_score']:.4f}" for r in rs))
assert rs[0]["index"] == 1, "Paris document must rank first via multipart too"
EOF

echo "== S1: chat model still serves after swaps =="
curl -sS "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\":\"$CHAT_MODEL\",
  \"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: OK\"}],
  \"max_tokens\":10,\"stream\":false}" > "$RESP"
python3 - <<'EOF' && ok "S1 chat after swap" || bad "S1"
import os, json
d = json.load(open(os.environ["RESP"]))
assert d["choices"][0]["message"]["content"].strip(), "empty chat reply"
EOF

echo
echo "== $PASS PASS, $FAIL FAIL =="
[ "$FAIL" -eq 0 ]

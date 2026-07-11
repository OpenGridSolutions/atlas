#!/usr/bin/env bash
# Acceptance test for the Qwen3-VL embedding/reranker models on the
# shared endpoints (multiclass Plan.md Phase 2).
#
# Run ON THE BOX against a RUNNING server (full 6-model preseed list):
#   scripts/vl_embed_test.sh
# NOTE: the first VL request pays snapshot creation on a fresh boot
# (safetensors -> .raw, ~1-2 min) plus the lazy swap (~seconds).
#
# Checks:
#   V1  VL model embeds TEXT (4096-dim unit-norm; Phase-1 parity)
#   V2  image embeds closer to its caption than to unrelated text
#   V3  VL rerank: image document ranks above an unrelated text doc
#   V4  image input on the TEXT embedding model returns a clear 400
#
# Response bodies go through a temp file (ARG_MAX + heredoc-stdin, see
# embed_rerank_test.sh).
set -euo pipefail

BASE="${BASE:-http://localhost:11434}"
VL_EMBED="${VL_EMBED:-Qwen/Qwen3-VL-Embedding-8B}"
VL_RERANK="${VL_RERANK:-Qwen/Qwen3-VL-Reranker-8B}"
TEXT_EMBED="${TEXT_EMBED:-Qwen/Qwen3-Embedding-8B}"
PASS=0; FAIL=0
ok()  { echo "PASS: $1"; PASS=$((PASS+1)); }
bad() { echo "FAIL: $1"; FAIL=$((FAIL+1)); }

RESP="$(mktemp)"; export RESP
IMG="$(mktemp)"; export IMG
trap 'rm -f "$RESP" "$IMG"' EXIT

# Deterministic test image: solid red 64x64 PNG (pure python, no PIL).
python3 - <<'EOF'
import os, struct, zlib
w = h = 64
def chunk(tag, data):
    c = tag + data
    return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))
raw = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))
png = (b"\x89PNG\r\n\x1a\n"
       + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(raw))
       + chunk(b"IEND", b""))
import base64
open(os.environ["IMG"], "w").write("data:image/png;base64," + base64.b64encode(png).decode())
EOF
IMG_URI=$(cat "$IMG")

echo "== V1: VL model embeds text =="
curl -sS "$BASE/v1/embeddings" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$VL_EMBED\",\"input\":\"a plain sentence about nothing in particular\"}" > "$RESP"
python3 - <<'EOF' && ok "V1 VL text embedding" || bad "V1"
import os, json, math
d = json.load(open(os.environ["RESP"])); v = d["data"][0]["embedding"]
assert len(v) == 4096, f"dim {len(v)}"
n = math.sqrt(sum(x*x for x in v)); assert abs(n - 1.0) < 1e-3, f"norm {n}"
EOF

echo "== V2: image closer to caption than to unrelated text =="
curl -sS "$BASE/v1/embeddings" -H 'Content-Type: application/json' -d "{
  \"model\":\"$VL_EMBED\",
  \"input\":[
    {\"image\":\"$IMG_URI\"},
    {\"text\":\"a solid red square\"},
    {\"text\":\"sourdough bread needs a mature starter and long fermentation\"}
  ]}" > "$RESP"
python3 - <<'EOF' && ok "V2 image/caption similarity" || bad "V2"
import os, json
d = json.load(open(os.environ["RESP"]))
vs = [e["embedding"] for e in sorted(d["data"], key=lambda e: e["index"])]
dot = lambda a, b: sum(x*y for x, y in zip(a, b))
caption, unrelated = dot(vs[0], vs[1]), dot(vs[0], vs[2])
print(f"  image~caption={caption:.4f} image~unrelated={unrelated:.4f}")
assert caption > unrelated, "image must embed closer to its caption"
EOF

echo "== V3: VL rerank ranks the image doc for a matching query =="
curl -sS "$BASE/v1/rerank" -H 'Content-Type: application/json' -d "{
  \"model\":\"$VL_RERANK\",
  \"query\":\"a red square\",
  \"documents\":[
    \"sourdough bread needs a mature starter and long fermentation\",
    {\"image\":\"$IMG_URI\"}
  ]}" > "$RESP"
python3 - <<'EOF' && ok "V3 VL rerank" || bad "V3"
import os, json
d = json.load(open(os.environ["RESP"]))
rs = d["results"]
print("  " + ", ".join(f"idx{r['index']}={r['relevance_score']:.4f}" for r in rs))
assert rs[0]["index"] == 1, "image document must rank first for 'a red square'"
EOF

echo "== V4: image on the text-only embedding model -> 400 =="
CODE=$(curl -sS -o "$RESP" -w '%{http_code}' "$BASE/v1/embeddings" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$TEXT_EMBED\",\"input\":[{\"image\":\"$IMG_URI\"}]}")
if [ "$CODE" = "400" ]; then ok "V4 clean 400 on text-only model"; else bad "V4 (got $CODE)"; fi

echo
echo "== $PASS PASS, $FAIL FAIL =="
[ "$FAIL" -eq 0 ]

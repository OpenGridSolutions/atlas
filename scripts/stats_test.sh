#!/usr/bin/env bash
# Acceptance test for GET /v1/stats (multiclass Plan.md Phase S).
#
# Run ON THE BOX against a RUNNING server:
#   scripts/stats_test.sh
# Env: BASE (default http://localhost:11434),
#      CHAT_MODEL (default vincentzed-hf/Qwen3-Coder-Next-NVFP4)
#
# Checks:
#   S1  valid JSON with BOTH memory readings + discrepancy_gb
#   S2  discrepancy_gb > 20 — documents the UNFIXED cudaMemGetInfo
#       state (DGX Spark known issue). After the separate KV-budget fix
#       this same metric is expected small/stable.
#   S3  after a chat request the generative model is resident (arena)
#       with plausible weights_bytes and kv_reserved_blocks > 0
#   S4  after a transcription the whisper sidecar shows
#       residency=sidecar with plausible weights_bytes
#
# NOTE: server responses are written to a temp FILE and read by python via
# json.load(open(path)) — not argv (ARG_MAX), not heredoc-stdin (already
# consumed by the script body). Same convention as embed_rerank_test.sh.
set -euo pipefail

BASE="${BASE:-http://localhost:11434}"
CHAT_MODEL="${CHAT_MODEL:-vincentzed-hf/Qwen3-Coder-Next-NVFP4}"
JFK="${JFK:-$(cd "$(dirname "$0")/.." && pwd)/test_data/whisper_jfk.wav}"
PASS=0; FAIL=0
ok()   { echo "PASS: $1"; PASS=$((PASS+1)); }
bad()  { echo "FAIL: $1"; FAIL=$((FAIL+1)); }

RESP="$(mktemp)"
export RESP
trap 'rm -f "$RESP"' EXIT

stats() { curl -sS "$BASE/v1/stats" > "$RESP"; }

echo "== S1: both memory readings + discrepancy =="
stats
python3 - <<'EOF' && ok "S1 memory block complete" || bad "S1 memory block"
import os, json
d = json.load(open(os.environ["RESP"]))
m = d["memory"]
assert isinstance(m["cuda_mem_get_info_free_gb"], (int, float)), m
assert isinstance(m["proc_meminfo_available_gb"], (int, float)), m
assert isinstance(m["discrepancy_gb"], (int, float)), m
assert m["total_unified_memory_gb"] > 0, m
print(f"  cuda_free={m['cuda_mem_get_info_free_gb']} GB, "
      f"proc_available={m['proc_meminfo_available_gb']} GB, "
      f"discrepancy={m['discrepancy_gb']} GB")
EOF

echo "== S2: discrepancy documents the unfixed state (> 20 GB) =="
# The discrepancy is exactly "reclaimable memory cudaMemGetInfo cannot
# see". Depending on when this runs, the page cache may be empty (Atlas
# reads snapshots O_DIRECT), which HIDES the bug. Reproduce the
# documented state deterministically: warm the cache with existing
# snapshot files (read-only) until ~40 GB of reclaimable cache exists —
# MemAvailable counts it, cudaMemGetInfo does not.
SNAPDIR="${SNAPDIR:-/workspace/snapshots}"
warmed=0
for f in "$SNAPDIR"/*.raw; do
  [ -e "$f" ] || break
  sz=$(stat -c%s "$f")
  cat "$f" > /dev/null 2>&1 || true
  warmed=$((warmed + sz))
  [ "$warmed" -ge $((45 * 1024 * 1024 * 1024)) ] && break
done
echo "  (page cache warmed with $((warmed / 2**30)) GiB of snapshot reads)"
stats
python3 - <<'EOF' && ok "S2 discrepancy > 20 GB (bug visible)" || bad "S2 discrepancy"
import os, json
m = json.load(open(os.environ["RESP"]))["memory"]
# Pre-fix expectation: cudaMemGetInfo misses the reclaimable DRAM (the
# warmed cache). The follow-up budget fix is verified against
# memory.kv_budget_last.free_now_seen_gb approaching
# proc_meminfo_available_gb — this raw-vs-meminfo discrepancy itself
# documents the driver behaviour and stays visible.
print(f"  cuda_free={m['cuda_mem_get_info_free_gb']} GB, "
      f"proc_available={m['proc_meminfo_available_gb']} GB, "
      f"discrepancy={m['discrepancy_gb']} GB")
assert m["discrepancy_gb"] > 20, m["discrepancy_gb"]
EOF

echo "== S3: generative model resident after a chat request =="
curl -sS "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$CHAT_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: pong\"}],\"max_tokens\":2000}" \
  > "$RESP"
python3 - <<'EOF' || bad "S3 chat request failed"
import os, json
d = json.load(open(os.environ["RESP"]))
assert "pong" in d["choices"][0]["message"]["content"].lower(), d
EOF
stats
CHAT_MODEL="$CHAT_MODEL" python3 - <<'EOF' && ok "S3 resident chat model with KV reservation" || bad "S3 residency"
import os, json
d = json.load(open(os.environ["RESP"]))
me = [m for m in d["models"] if m["id"] == os.environ["CHAT_MODEL"]]
assert len(me) == 1, d["models"]
m = me[0]
assert m["resident"] and m["residency"] == "arena", m
assert m["category"] == "generative", m
# 30B-A3B NVFP4 ≈ 16-20 GiB; sanity band 4-80 GiB.
assert 4e9 < m["weights_bytes"] < 8e10, m["weights_bytes"]
assert m["kv_reserved_blocks"] > 0, m
assert m["kv_reserved_bytes"] > 0, m
assert 0 <= m["kv_used_blocks"] <= m["kv_reserved_blocks"], m
print(f"  weights={m['weights_bytes']/2**30:.1f} GiB, "
      f"kv={m['kv_reserved_blocks']} blocks "
      f"({m['kv_reserved_bytes']/2**30:.1f} GiB), used={m['kv_used_blocks']}")
# Non-resident registered models report zero usage.
others = [x for x in d["models"] if not x["resident"]]
assert all(x["weights_bytes"] == 0 and x["kv_reserved_blocks"] == 0 for x in others), others
EOF

echo "== S4: whisper sidecar resident after a transcription =="
curl -sS "$BASE/v1/audio/transcriptions" -F "file=@$JFK" -F "model=whisper-1" > "$RESP"
python3 - <<'EOF' || bad "S4 transcription failed"
import os, json
assert "country" in json.load(open(os.environ["RESP"]))["text"].lower()
EOF
stats
python3 - <<'EOF' && ok "S4 sidecar residency" || bad "S4 sidecar residency"
import os, json
d = json.load(open(os.environ["RESP"]))
ws = [m for m in d["models"] if m["category"] == "transcribe"]
assert len(ws) == 1, d["models"]
w = ws[0]
assert w["resident"] and w["residency"] == "sidecar", w
# whisper-large-v3 BF16 ≈ 3 GiB; sanity band 1-8 GiB.
assert 1e9 < w["weights_bytes"] < 8e9, w["weights_bytes"]
assert w["kv_reserved_blocks"] == 0, w
print(f"  sidecar weights={w['weights_bytes']/2**30:.2f} GiB")
EOF

echo
echo "== $PASS PASS, $FAIL FAIL =="
exit $((FAIL > 0))

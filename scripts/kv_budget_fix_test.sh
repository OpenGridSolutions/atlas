#!/usr/bin/env bash
# Phase M acceptance (Plan.md): KV budget uses MemAvailable+SwapFree.
#
# Run ON THE BOX against a RUNNING server built with the fix:
#   scripts/kv_budget_fix_test.sh
# Env: BASE (default http://localhost:11434)
#
# Checks:
#   K1  stats scale proof: right after a fresh load, kv_budget_last.
#       free_now_seen_gb tracks proc_meminfo (available+swap), not the
#       raw cuda value (pre-fix it equals the smaller reading)
#   K2  induced-pressure repro of the 2026-07-13 failure: flood the page
#       cache with snapshot reads, then hot-swap TranslateGemma (the
#       --max-seq-len 104416 config that died with "KV cache can hold at
#       most 0 concurrent sequences") -> 200 + sane budget
#   K3  churn drift: swap through all five categories, K2's translate
#       again -> budget does not degrade with hot-swap history
#   K4  overcommit guard: budget stays under the util ceiling
#
# NOTE: responses go to a temp FILE and python reads $RESP from the env
# (not argv -> ARG_MAX; not piped stdin -> heredoc conflict).
set -uo pipefail

BASE="${BASE:-http://localhost:11434}"
SNAPDIR="${SNAPDIR:-/workspace/snapshots}"
PASS=0; FAIL=0
ok()  { echo "PASS: $1"; PASS=$((PASS+1)); }
bad() { echo "FAIL: $1"; FAIL=$((FAIL+1)); }

RESP="$(mktemp)"; export RESP
trap 'rm -f "$RESP"' EXIT

stats() { curl -sS -m 30 "$BASE/v1/stats" > "$RESP"; }

translate() { # returns http code
  curl -s -m 900 -o "$RESP" -w '%{http_code}' "$BASE/v1/translate" \
    -H 'Content-Type: application/json' \
    -d '{"model":"google/translategemma-12b-it","source_lang_code":"de","target_lang_code":"en","text":"Guten Morgen."}'
}

echo "== K2a: induced memory pressure (page-cache flood) =="
# The S2 trick: stream the big snapshot files through the page cache
# (plain reads, NOT O_DIRECT). Reproduces the low-MemAvailable state
# deterministically instead of waiting hours of uptime.
# Same capped warm loop as stats_test.sh S2 (the proven repro of the
# documented discrepancy state): read snapshots until ~45 GiB cached.
warmed=0
for f in "$SNAPDIR"/*.raw; do
  [ -e "$f" ] || break
  sz=$(stat -c%s "$f")
  cat "$f" > /dev/null 2>&1 || true
  warmed=$((warmed + sz))
  [ "$warmed" -ge $((45 * 1024 * 1024 * 1024)) ] && break
done
if [ "$warmed" -gt 0 ]; then
  echo "  page cache warmed with $((warmed / 2**30)) GiB of snapshot reads"
else
  echo "  WARN: no snapshots under $SNAPDIR — continuing without extra pressure"
fi

echo "== K2b: TranslateGemma hot-swap under pressure (the crash repro) =="
code=$(translate)
if [ "$code" = "200" ]; then ok "K2 translate under pressure (was: 'can hold at most 0 sequences')"
else bad "K2 translate returned $code: $(head -c 200 "$RESP")"; fi

echo "== K1: budget input tracks meminfo scale (fresh load just happened) =="
stats
python3 - <<'EOF' && ok "K1 free_now_seen on meminfo scale" || bad "K1"
import os, json
m = json.load(open(os.environ["RESP"]))["memory"]
seen = m["kv_budget_last"]["free_now_seen_gb"]
avail = m["proc_meminfo_available_gb"]
cuda = m["cuda_mem_get_info_free_gb"]
print(f"  free_now_seen={seen}  meminfo(avail+swap)={avail}  cuda_raw={cuda}")
# Post-fix the budget input is max(cuda, avail+swap). Allow drift between
# the load instant and this stats call.
assert seen >= max(cuda, avail) - 5.0, "budget input below the meminfo scale"
EOF

echo "== K1b: pool viable (informational — the hard criterion is K2's 200) =="
KV1=$(python3 -c 'import os,json;print(json.load(open(os.environ["RESP"]))["memory"]["kv_budget_last"]["kv_budget_gb"])')
echo "  kv_budget_last=$KV1 GB"
python3 - <<'EOF' && ok "K1b pool viable" || bad "K1b"
import os, json
kv = json.load(open(os.environ["RESP"]))["memory"]["kv_budget_last"]["kv_budget_gb"]
assert kv > 0, "empty KV budget"
EOF

echo "== K3: five-category churn, then the repro again =="
curl -s -m 900 -o /dev/null "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"vincentzed-hf/Qwen3-Coder-Next-NVFP4","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":8}'
curl -s -m 900 -o /dev/null "$BASE/v1/embeddings" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-Embedding-8B","input":"hello"}'
curl -s -m 900 -o /dev/null "$BASE/v1/rerank" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-Reranker-8B","query":"a","documents":["b","c"]}'
code=$(translate)
if [ "$code" = "200" ]; then ok "K3 translate after churn"
else bad "K3 translate returned $code"; fi
stats
KV1="$KV1" python3 - <<'EOF' && ok "K3b budget stable after churn" || bad "K3b"
import os, json
m = json.load(open(os.environ["RESP"]))["memory"]
kv = m["kv_budget_last"]["kv_budget_gb"]
before = float(os.environ["KV1"])
print(f"  kv_budget_last after churn = {kv} GB (before churn: {before} GB)")
# The pre-fix failure mode was monotonic degradation with hot-swap
# history; allow small state noise but no collapse.
assert kv >= before - 2.0, "budget degraded with hot-swap history"
EOF

echo "== K4: overcommit guard (util ceiling holds) =="
python3 - <<'EOF' && ok "K4 under util ceiling" || bad "K4"
import os, json
d = json.load(open(os.environ["RESP"]))
m = d["memory"]
total = m["total_unified_memory_gb"]
kv = m["kv_budget_last"]["kv_budget_gb"]
weights = sum(x["weights_bytes"] for x in d["models"] if x["resident"]) / 2**30
print(f"  weights(resident)={weights:.1f} GB + kv_budget={kv} GB vs ceiling {0.9*total:.1f} GB")
assert weights + kv <= 0.9 * total + 1.0, "weights + KV exceed the 90% util ceiling"
EOF

echo
echo "== $PASS PASS, $FAIL FAIL =="
[ "$FAIL" -eq 0 ]

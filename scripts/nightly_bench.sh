#!/usr/bin/env bash
# Nightly test bench on the DGX Spark (Todo.md 2.2/2.3/2.4).
#
# Assumes a RUNNING server (started via ./serve.sh, ideally with
# ATLAS_MEM_DEBUG=1 for the 2.3 leak checks). Runs the stability suite
# and gates on the file-logging sinks from Phase 1:
#
#   N1  smoke: /health 200 (also catches the sticky-CUDA 503, case D1)
#   N2  KV-budget sanity: kv_budget_fix_test.sh K1 equivalent via /v1/stats
#   N3  hot-swap release: scripts/hotswap_release_test.sh (12 assertions)
#   N4  churn leak detector: scripts/hotswap_churn_test.sh
#   N5  cuda_err gate: no `cuda_err` events in the newest critical-*.log
#   N6  memdebug gate: no "KV LEAK" / "release path was skipped" in the
#       newest atlas-*.log (meaningful when ATLAS_MEM_DEBUG=1)
#
# Scheduling (host side, container name atlas-gb10):
#   crontab: 0 3 * * *  podman exec atlas-gb10 /workspace/scripts/nightly_bench.sh \
#              >> /var/log/atlas-nightly.log 2>&1
# Exit 0/1; one summary line at the end. Also usable as the pre-deploy
# gate in Deploy.md (run before pushing a new image).
set -uo pipefail

URL="${ATLAS_URL:-http://127.0.0.1:11434}"
LOG_DIR="${ATLAS_LOG_DIR:-/workspace/logs}"
cd "$(dirname "$0")/.."

PASS=0
FAIL=0
step() { printf '\n==> %s\n' "$*"; }
ok()   { PASS=$((PASS+1)); printf '    PASS: %s\n' "$*"; }
bad()  { FAIL=$((FAIL+1)); printf '    FAIL: %s\n' "$*"; }

step "N1: smoke — /health"
CODE=$(curl -sS -o /tmp/nightly_health.json -w '%{http_code}' --max-time 10 "${URL}/health")
if [[ "${CODE}" == "200" ]]; then
    ok "health 200"
elif grep -q "cuda_context_poisoned" /tmp/nightly_health.json 2>/dev/null; then
    bad "health ${CODE}: STICKY CUDA ERROR — $(head -c 200 /tmp/nightly_health.json)"
else
    bad "health ${CODE}"
fi

step "N2: KV-budget sanity via /v1/stats"
python3 - "$URL" <<'EOF' && ok "stats sane" || bad "stats sanity failed"
import json, sys, urllib.request
d = json.load(urllib.request.urlopen(sys.argv[1] + "/v1/stats", timeout=10))
m = d["memory"]
assert m["proc_meminfo_available_gb"] > 4, f"available too low: {m}"
dec = m.get("kv_budget_decision")
assert dec is None or dec["source"] != "auto_implausible", f"gate degraded: {dec}"
EOF

step "N3: hot-swap release suite"
./scripts/hotswap_release_test.sh >/tmp/nightly_release.log 2>&1 \
    && ok "hotswap_release_test PASS" \
    || { bad "hotswap_release_test FAIL"; tail -20 /tmp/nightly_release.log; }

step "N4: churn leak detector"
CYCLES="${CYCLES:-5}" ./scripts/hotswap_churn_test.sh >/tmp/nightly_churn.log 2>&1 \
    && ok "hotswap_churn_test PASS" \
    || { bad "hotswap_churn_test FAIL"; tail -20 /tmp/nightly_churn.log; }

step "N5: cuda_err gate (critical log)"
CRIT=$(ls -t "${LOG_DIR}"/critical-*.log 2>/dev/null | head -1)
if [[ -z "${CRIT}" ]]; then
    bad "no critical-*.log found in ${LOG_DIR} (file logging off?)"
elif grep -q "cuda_err" "${CRIT}"; then
    bad "cuda_err events in ${CRIT}:"; grep "cuda_err" "${CRIT}" | tail -5
else
    ok "no cuda_err events in $(basename "${CRIT}")"
fi

step "N6: memdebug leak gate (atlas log)"
ATLAS_LOG=$(ls -t "${LOG_DIR}"/atlas-*.log 2>/dev/null | head -1)
if [[ -z "${ATLAS_LOG}" ]]; then
    bad "no atlas-*.log found in ${LOG_DIR}"
elif grep -Eq "KV LEAK|release path was skipped" "${ATLAS_LOG}"; then
    bad "leak markers in ${ATLAS_LOG}:"; grep -E "KV LEAK|release path" "${ATLAS_LOG}" | tail -5
else
    ok "no leak markers in $(basename "${ATLAS_LOG}")"
fi

printf '\n== nightly_bench: %d PASS, %d FAIL ==\n' "${PASS}" "${FAIL}"
exit $((FAIL > 0))

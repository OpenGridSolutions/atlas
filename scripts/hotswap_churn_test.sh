#!/usr/bin/env bash
# Hot-swap CHURN leak detector (Todo.md 2.1b — broader than the 2-cycle
# check in hotswap_release_test.sh).
#
# Drives a RUNNING server through N load->completion->unload cycles
# alternating across models and asserts, via /v1/stats:
#   C1  every unload reports free_failed == 0
#   C2  meminfo available after the LAST cycle is within DRIFT_GB of the
#       value after the FIRST cycle (monotonic loss = leak, case A4)
#   C3  the final kv_budget_decision.source is auto_baseline (the 0.1/0.3
#       fixes keep the self-relative gate alive across swaps)
#
# Env overrides:
#   ATLAS_URL     (default http://127.0.0.1:11434)
#   ATLAS_API_KEY (optional bearer token)
#   CYCLES        (default 5)
#   MODELS        (space-separated, default: the two snapshot-backed
#                  models of the 2026-09-11 case)
#   DRIFT_GB      (default 2)
set -uo pipefail

URL="${ATLAS_URL:-http://127.0.0.1:11434}"
CYCLES="${CYCLES:-5}"
MODELS="${MODELS:-google/translategemma-12b-it vincentzed-hf/Qwen3-Coder-Next-NVFP4}"
DRIFT_GB="${DRIFT_GB:-2}"
AUTH_ARGS=()
[[ -n "${ATLAS_API_KEY:-}" ]] && AUTH_ARGS=(-H "Authorization: Bearer ${ATLAS_API_KEY}")

PASS=0
FAIL=0
step() { printf '\n==> %s\n' "$*"; }
ok()   { PASS=$((PASS+1)); printf '    PASS: %s\n' "$*"; }
bad()  { FAIL=$((FAIL+1)); printf '    FAIL: %s\n' "$*"; }

if command -v jq >/dev/null 2>&1; then
    jget() { jq -r "$1" 2>/dev/null; }
else
    jget() { python3 -c "
import sys, json
path = sys.argv[1].lstrip('.')
try:
    doc = json.load(sys.stdin)
    for part in path.replace(']','').replace('[','.').split('.'):
        doc = doc[int(part)] if part.isdigit() else doc[part]
    print(doc)
except Exception:
    print('null')
" "$1"; }
fi
fcmp() { python3 -c "import sys; sys.exit(0 if eval(f'{float(sys.argv[1])} {sys.argv[2]} {float(sys.argv[3])}') else 1)" $1 $2 $3; }

avail_gb() { curl -sS --max-time 10 "${AUTH_ARGS[@]}" "${URL}/v1/stats" | jget .memory.proc_meminfo_available_gb; }

completion() { # $1 = model — category-aware (translate models 400 on /v1/chat)
    local code
    code=$(curl -sS -o /tmp/churn_cmpl.json -w '%{http_code}' --max-time 900 \
        "${AUTH_ARGS[@]}" -H "Content-Type: application/json" \
        -d "{\"model\": \"$1\", \"temperature\": 0, \"max_tokens\": 32, \
             \"messages\": [{\"role\": \"user\", \"content\": \"Say OK.\"}]}" \
        "${URL}/v1/chat/completions")
    if [[ "${code}" == "400" ]] \
        && [[ "$(jget .model_endpoint < /tmp/churn_cmpl.json)" == "/v1/translate" ]]; then
        code=$(curl -sS -o /tmp/churn_cmpl.json -w '%{http_code}' --max-time 900 \
            "${AUTH_ARGS[@]}" -H "Content-Type: application/json" \
            -d "{\"model\": \"$1\", \"text\": \"Guten Morgen\", \
                 \"source_lang_code\": \"de\", \"target_lang_code\": \"en\"}" \
            "${URL}/v1/translate")
    fi
    echo "${code}"
}

step "C0: server up"
STATE=$(curl -sS --max-time 10 "${AUTH_ARGS[@]}" "${URL}/admin/models" | jget .state) || {
    echo "Server not reachable at ${URL}"; exit 1; }
ok "state: ${STATE}"
[[ "${STATE}" != "unloaded" && "${STATE}" != "null" ]] \
    && curl -sS --max-time 120 -X POST "${AUTH_ARGS[@]}" "${URL}/admin/models/unload" >/dev/null

AVAIL_FIRST=""
read -ra MODEL_ARR <<< "${MODELS}"
for ((c = 1; c <= CYCLES; c++)); do
    MODEL="${MODEL_ARR[$(( (c - 1) % ${#MODEL_ARR[@]} ))]}"
    step "Cycle ${c}/${CYCLES}: ${MODEL}"
    LOAD=$(curl -sS --max-time 1000 -X POST "${AUTH_ARGS[@]}" \
        -H "Content-Type: application/json" -d "{\"model\": \"${MODEL}\"}" \
        "${URL}/admin/models/load")
    [[ "$(echo "${LOAD}" | jget .duration_ms)" != "null" ]] \
        && ok "loaded" || { bad "load failed: ${LOAD:0:250}"; continue; }
    CODE=$(completion "${MODEL}")
    [[ "${CODE}" == "200" ]] && ok "completion 200" || bad "completion ${CODE}"
    UNLOAD=$(curl -sS --max-time 120 -X POST "${AUTH_ARGS[@]}" "${URL}/admin/models/unload")
    FREE_FAILED=$(echo "${UNLOAD}" | jget .free_failed)
    [[ "${FREE_FAILED}" == "0" ]] \
        && ok "unload: 0 failed frees (reclaimed $(echo "${UNLOAD}" | jget .released.reclaimed_gb) GB)" \
        || bad "unload free_failed=${FREE_FAILED} (case B1)"
    AVAIL=$(avail_gb)
    echo "    meminfo available after cycle ${c}: ${AVAIL} GB"
    [[ -z "${AVAIL_FIRST}" ]] && AVAIL_FIRST="${AVAIL}"
done

step "C2: drift over ${CYCLES} cycles"
AVAIL_LAST=$(avail_gb)
if [[ "${AVAIL_FIRST}" != "null" && "${AVAIL_LAST}" != "null" ]] \
    && fcmp "${AVAIL_LAST}" ">=" "$(python3 -c "print(float('${AVAIL_FIRST}') - float('${DRIFT_GB}'))")"; then
    ok "no monotonic loss: ${AVAIL_FIRST} GB -> ${AVAIL_LAST} GB (drift < ${DRIFT_GB} GB)"
else
    bad "drift: ${AVAIL_FIRST} GB -> ${AVAIL_LAST} GB (case A4 leak)"
fi

step "C3: budget gate stays self-relative"
SOURCE=$(curl -sS --max-time 10 "${AUTH_ARGS[@]}" "${URL}/v1/stats" | jget .memory.kv_budget_decision.source)
[[ "${SOURCE}" == "auto_baseline" ]] \
    && ok "kv_budget_decision.source = auto_baseline" \
    || bad "kv_budget_decision.source = ${SOURCE} (gate degraded; 0.1/0.3 regression)"

printf '\n== %d PASS, %d FAIL ==\n' "${PASS}" "${FAIL}"
exit $((FAIL > 0))

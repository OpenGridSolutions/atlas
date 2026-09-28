#!/usr/bin/env bash
# Hot-swap RELEASE acceptance (Todo.md 0.2/2.1 — the 2026-09-11 regression).
#
# Drives a RUNNING server (start it first via ./serve.sh) through the exact
# failure sequence from Memory_errors.log and asserts memory actually comes
# back at each step:
#   R1  load BIG model + one completion (fill arena, KV pool, snapshots)
#   R2  unload -> `released` accounting in the reply: free_failed == 0 and
#       reclaimed_gb >= MIN_RECLAIM_GB (0 before the MADV_DONTNEED fix)
#   R3  load SMALL model -> MUST succeed (failed 5x in a row on 2026-09-11
#       with "No memory left for KV cache")
#   R4  completion on the small model works
#   R5  second unload->load cycle: meminfo after cycle 2 within
#       CYCLE_DRIFT_GB of cycle 1 (leak detector for the load/fail paths)
#
# Env overrides:
#   ATLAS_URL        (default http://127.0.0.1:11434)
#   ATLAS_API_KEY    (optional bearer token)
#   MODEL_BIG        (default nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4)
#   MODEL_SMALL      (default google/translategemma-12b-it)
#   MIN_RECLAIM_GB   (default 20 — the 80B arena payload is ~47 GiB)
#   CYCLE_DRIFT_GB   (default 2)
set -uo pipefail

URL="${ATLAS_URL:-http://127.0.0.1:11434}"
MODEL_BIG="${MODEL_BIG:-nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4}"
MODEL_SMALL="${MODEL_SMALL:-google/translategemma-12b-it}"
MIN_RECLAIM_GB="${MIN_RECLAIM_GB:-20}"
CYCLE_DRIFT_GB="${CYCLE_DRIFT_GB:-2}"
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

# float comparison: fcmp "a >= b"
fcmp() { python3 -c "import sys; a=float(sys.argv[1]); b=float(sys.argv[3]); \
op=sys.argv[2]; sys.exit(0 if eval(f'a {op} b') else 1)" $1 $2 $3; }

meminfo_avail_gb() {
    curl -sS --max-time 10 "${AUTH_ARGS[@]}" "${URL}/v1/stats" \
        | jget .memory.proc_meminfo_available_gb
}

completion() { # $1 = model — category-aware (Phase K routes models to
    # exactly one endpoint; translate-category models 400 on /v1/chat).
    local code
    code=$(curl -sS -o /tmp/hotswap_release_cmpl.json -w '%{http_code}' --max-time 900 \
        "${AUTH_ARGS[@]}" -H "Content-Type: application/json" \
        -d "{\"model\": \"$1\", \"temperature\": 0, \"max_tokens\": 32, \
             \"messages\": [{\"role\": \"user\", \"content\": \"Say OK.\"}]}" \
        "${URL}/v1/chat/completions")
    if [[ "${code}" == "400" ]] \
        && [[ "$(jget .error.code < /tmp/hotswap_release_cmpl.json)" == "model_wrong_category" ]] \
        && [[ "$(jget .model_endpoint < /tmp/hotswap_release_cmpl.json)" == "/v1/translate" ]]; then
        code=$(curl -sS -o /tmp/hotswap_release_cmpl.json -w '%{http_code}' --max-time 900 \
            "${AUTH_ARGS[@]}" -H "Content-Type: application/json" \
            -d "{\"model\": \"$1\", \"text\": \"Guten Morgen\", \
                 \"source_lang_code\": \"de\", \"target_lang_code\": \"en\"}" \
            "${URL}/v1/translate")
    fi
    echo "${code}"
}

unload() {
    curl -sS --max-time 120 -X POST "${AUTH_ARGS[@]}" "${URL}/admin/models/unload"
}

load() { # $1 = model
    curl -sS --max-time 1000 -X POST "${AUTH_ARGS[@]}" \
        -H "Content-Type: application/json" -d "{\"model\": \"$1\"}" \
        "${URL}/admin/models/load"
}

# ── R0: server up, note state ─────────────────────────────────────────────
step "R0: GET /admin/models"
STATE=$(curl -sS --max-time 10 "${AUTH_ARGS[@]}" "${URL}/admin/models" | jget .state) || {
    echo "Server not reachable at ${URL} — start it first (./serve.sh)"; exit 1;
}
ok "server up (state: ${STATE})"
if [[ "${STATE}" != "unloaded" && "${STATE}" != "null" ]]; then
    step "R0b: clearing pre-loaded model"
    unload >/dev/null && ok "cleared" || bad "pre-unload failed"
fi

# ── R1: big model in, memory under load ──────────────────────────────────
step "R1: load ${MODEL_BIG} + one completion"
LOAD_JSON=$(load "${MODEL_BIG}")
if [[ "$(echo "${LOAD_JSON}" | jget .duration_ms)" != "null" ]]; then
    ok "big model loaded"
else
    bad "big model load failed: ${LOAD_JSON:0:300}"; echo "Aborting."; exit 1
fi
CODE=$(completion "${MODEL_BIG}")
[[ "${CODE}" == "200" ]] && ok "completion 200" || bad "completion ${CODE}"
AVAIL_LOADED=$(meminfo_avail_gb)
echo "    meminfo available with big model: ${AVAIL_LOADED} GB"

# ── R2: unload must actually give the memory back ────────────────────────
step "R2: unload -> released accounting"
UNLOAD_JSON=$(unload)
echo "    $(echo "${UNLOAD_JSON}" | head -c 500)"
FREE_FAILED=$(echo "${UNLOAD_JSON}" | jget .free_failed)
RECLAIMED=$(echo "${UNLOAD_JSON}" | jget .released.reclaimed_gb)
[[ "$(echo "${UNLOAD_JSON}" | jget .state)" == "unloaded" ]] \
    && ok "state unloaded" || bad "unload failed"
[[ "${FREE_FAILED}" == "0" ]] \
    && ok "0 failed device frees" || bad "free_failed=${FREE_FAILED} (case B1)"
if [[ "${RECLAIMED}" != "null" ]] && fcmp "${RECLAIMED}" ">=" "${MIN_RECLAIM_GB}"; then
    ok "reclaimed ${RECLAIMED} GB >= ${MIN_RECLAIM_GB} GB (arena pages returned)"
else
    bad "reclaimed ${RECLAIMED} GB < ${MIN_RECLAIM_GB} GB — unload kept pages resident"
fi

# ── R3: the 2026-09-11 regression: small model must now fit ──────────────
step "R3: load ${MODEL_SMALL} after big-model unload"
LOAD_JSON=$(load "${MODEL_SMALL}")
if [[ "$(echo "${LOAD_JSON}" | jget .duration_ms)" != "null" ]]; then
    ok "small model loaded (was: 'No memory left for KV cache' x5)"
else
    bad "small model load failed: ${LOAD_JSON:0:300}"
fi

# ── R4: it actually serves ────────────────────────────────────────────────
step "R4: completion on ${MODEL_SMALL}"
CODE=$(completion "${MODEL_SMALL}")
[[ "${CODE}" == "200" ]] && ok "completion 200" || bad "completion ${CODE}"
AVAIL_CYCLE1=$(meminfo_avail_gb)
echo "    meminfo available after cycle 1: ${AVAIL_CYCLE1} GB"

# ── R5: second cycle, no monotonic loss ───────────────────────────────────
step "R5: second unload->load cycle (leak detector)"
unload >/dev/null
LOAD_JSON=$(load "${MODEL_SMALL}")
[[ "$(echo "${LOAD_JSON}" | jget .duration_ms)" != "null" ]] \
    && ok "cycle-2 load ok" || bad "cycle-2 load failed: ${LOAD_JSON:0:300}"
CODE=$(completion "${MODEL_SMALL}")
[[ "${CODE}" == "200" ]] && ok "cycle-2 completion 200" || bad "cycle-2 completion ${CODE}"
AVAIL_CYCLE2=$(meminfo_avail_gb)
echo "    meminfo available after cycle 2: ${AVAIL_CYCLE2} GB"
if [[ "${AVAIL_CYCLE1}" != "null" && "${AVAIL_CYCLE2}" != "null" ]] \
    && fcmp "${AVAIL_CYCLE2}" ">=" "$(python3 -c "print(float('${AVAIL_CYCLE1}') - float('${CYCLE_DRIFT_GB}'))")"; then
    ok "no monotonic loss across cycles (drift < ${CYCLE_DRIFT_GB} GB)"
else
    bad "cycle drift: ${AVAIL_CYCLE1} GB -> ${AVAIL_CYCLE2} GB (case A4 leak)"
fi

printf '\n== %d PASS, %d FAIL ==\n' "${PASS}" "${FAIL}"
exit $((FAIL > 0))

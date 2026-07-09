#!/usr/bin/env bash
# Hot-swap MVP acceptance drive (HOTSWAP_MVP_PLAN.md Tasks 3-5).
#
# Drives a RUNNING server (start it first via ./serve.sh) through:
#   baseline temp-0 completion -> snapshot -> unload -> 503 check
#   -> load -> temp-0 completion again -> diff vs baseline.
#
# Env overrides:
#   ATLAS_URL      (default http://127.0.0.1:11434)
#   ATLAS_API_KEY  (optional bearer token when serving with --require-auth)
#   PROMPT         (default below)
#   MAX_TOKENS     (default 64)
set -uo pipefail

URL="${ATLAS_URL:-http://127.0.0.1:11434}"
PROMPT="${PROMPT:-Explain in one sentence why the sky is blue.}"
MAX_TOKENS="${MAX_TOKENS:-64}"
AUTH_ARGS=()
[[ -n "${ATLAS_API_KEY:-}" ]] && AUTH_ARGS=(-H "Authorization: Bearer ${ATLAS_API_KEY}")

PASS=0
FAIL=0
step() { printf '\n==> %s\n' "$*"; }
ok()   { PASS=$((PASS+1)); printf '    PASS: %s\n' "$*"; }
bad()  { FAIL=$((FAIL+1)); printf '    FAIL: %s\n' "$*"; }

# JSON field extraction: jq if present, else python3.
if command -v jq >/dev/null 2>&1; then
    jget() { jq -r "$1" 2>/dev/null; }
else
    jget() { python3 -c "
import sys, json, functools
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

completion_request() {
    curl -sS --max-time 600 "${AUTH_ARGS[@]}" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${MODEL}\", \"temperature\": 0, \"max_tokens\": ${MAX_TOKENS}, \
             \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}]}" \
        "${URL}/v1/chat/completions"
}

# ── Step 0: server + hot-swap availability ────────────────────────────────
step "Step 0: GET /admin/models (server up, hot-swap enabled)"
ADMIN_STATE=$(curl -sS --max-time 10 "${AUTH_ARGS[@]}" "${URL}/admin/models") || {
    bad "server unreachable at ${URL}"; echo "Aborting."; exit 1; }
echo "    ${ADMIN_STATE}"
MODEL=$(echo "${ADMIN_STATE}" | jget .model)
STATE=$(echo "${ADMIN_STATE}" | jget .state)
ARENA_BASE_0=$(echo "${ADMIN_STATE}" | jget .arena.base)
if [[ "${STATE}" == "running" && "${MODEL}" != "null" ]]; then
    ok "state=running, model=${MODEL}, arena_base=${ARENA_BASE_0}"
else
    bad "expected state=running with a model (got state=${STATE})"
    echo "Aborting: hot-swap unavailable (--weights-arena-gb 0, EP, or --dflash?)."
    exit 1
fi

# ── Step 1: temperature-0 baseline ────────────────────────────────────────
step "Step 1: baseline temperature-0 completion"
BASELINE_JSON=$(completion_request)
BASELINE=$(echo "${BASELINE_JSON}" | jget '.choices[0].message.content')
if [[ -n "${BASELINE}" && "${BASELINE}" != "null" ]]; then
    ok "baseline captured (${#BASELINE} chars)"
    printf '    ---\n%s\n    ---\n' "${BASELINE}"
else
    bad "no completion content; raw: ${BASELINE_JSON:0:400}"
    echo "Aborting."; exit 1
fi

# ── Step 2: snapshot ──────────────────────────────────────────────────────
step "Step 2: POST /admin/models/snapshot"
T_SNAP0=$(date +%s)
SNAP_JSON=$(curl -sS --max-time 900 -X POST "${AUTH_ARGS[@]}" "${URL}/admin/models/snapshot")
T_SNAP1=$(date +%s)
echo "    ${SNAP_JSON}"
SNAP_PATH=$(echo "${SNAP_JSON}" | jget .path)
SNAP_XXH3=$(echo "${SNAP_JSON}" | jget .payload_xxh3)
SNAP_BYTES=$(echo "${SNAP_JSON}" | jget .bytes)
if [[ "${SNAP_PATH}" != "null" && "${SNAP_XXH3}" != "null" ]]; then
    ok "snapshot ${SNAP_PATH} (${SNAP_BYTES} bytes, xxh3 ${SNAP_XXH3}, wall $((T_SNAP1-T_SNAP0)) s)"
else
    bad "snapshot failed: ${SNAP_JSON:0:400}"
    echo "Aborting."; exit 1
fi

# ── Step 3: unload ────────────────────────────────────────────────────────
step "Step 3: POST /admin/models/unload"
UNLOAD_JSON=$(curl -sS --max-time 120 -X POST "${AUTH_ARGS[@]}" "${URL}/admin/models/unload")
echo "    ${UNLOAD_JSON}"
UNLOAD_STATE=$(echo "${UNLOAD_JSON}" | jget .state)
FREED=$(echo "${UNLOAD_JSON}" | jget .freed_allocs)
if [[ "${UNLOAD_STATE}" == "unloaded" ]]; then
    ok "unloaded (${FREED} device allocations freed)"
else
    bad "unload failed: ${UNLOAD_JSON:0:400}"
fi
ARENA_BASE_1=$(curl -sS "${AUTH_ARGS[@]}" "${URL}/admin/models" | jget .arena.base)
if [[ "${ARENA_BASE_1}" == "${ARENA_BASE_0}" ]]; then
    ok "arena base unchanged after unload (${ARENA_BASE_1})"
else
    bad "arena base CHANGED: ${ARENA_BASE_0} -> ${ARENA_BASE_1}"
fi

# ── Step 4: inference must 503 ────────────────────────────────────────────
step "Step 4: /v1/chat/completions returns 503 while unloaded"
HTTP_CODE=$(curl -sS -o /tmp/hotswap_503_body.json -w '%{http_code}' --max-time 30 \
    "${AUTH_ARGS[@]}" -H "Content-Type: application/json" \
    -d "{\"model\": \"${MODEL}\", \"temperature\": 0, \"max_tokens\": 8, \
         \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}" \
    "${URL}/v1/chat/completions")
BODY_STATE=$(jget .state < /tmp/hotswap_503_body.json)
if [[ "${HTTP_CODE}" == "503" ]]; then
    ok "503 received (body state: ${BODY_STATE})"
else
    bad "expected 503, got ${HTTP_CODE}: $(head -c 300 /tmp/hotswap_503_body.json)"
fi

# ── Step 5: load ──────────────────────────────────────────────────────────
step "Step 5: POST /admin/models/load"
LOAD_JSON=$(curl -sS --max-time 1000 -X POST "${AUTH_ARGS[@]}" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${MODEL}\"}" \
    "${URL}/admin/models/load")
echo "    ${LOAD_JSON}"
LOAD_MS=$(echo "${LOAD_JSON}" | jget .duration_ms)
LOAD_GIBS=$(echo "${LOAD_JSON}" | jget .gib_per_s)
LOAD_READ_MS=$(echo "${LOAD_JSON}" | jget .read_ms)
if [[ "${LOAD_MS}" != "null" && -n "${LOAD_MS}" ]]; then
    ok "reload: total ${LOAD_MS} ms (payload read ${LOAD_READ_MS} ms @ ${LOAD_GIBS} GiB/s)"
else
    bad "load failed: ${LOAD_JSON:0:400}"
    echo "Aborting."; exit 1
fi
ARENA_BASE_2=$(curl -sS "${AUTH_ARGS[@]}" "${URL}/admin/models" | jget .arena.base)
if [[ "${ARENA_BASE_2}" == "${ARENA_BASE_0}" ]]; then
    ok "arena base constant across the full cycle (${ARENA_BASE_2})"
else
    bad "arena base CHANGED after load: ${ARENA_BASE_0} -> ${ARENA_BASE_2}"
fi

# ── Step 6: temperature-0 identity ────────────────────────────────────────
step "Step 6: temperature-0 completion after reload, diff vs baseline"
AFTER_JSON=$(completion_request)
AFTER=$(echo "${AFTER_JSON}" | jget '.choices[0].message.content')
if [[ -z "${AFTER}" || "${AFTER}" == "null" ]]; then
    bad "no completion after reload; raw: ${AFTER_JSON:0:400}"
elif [[ "${AFTER}" == "${BASELINE}" ]]; then
    ok "temperature-0 output IDENTICAL before unload and after reload"
else
    bad "temperature-0 output DIFFERS after reload"
    diff <(printf '%s\n' "${BASELINE}") <(printf '%s\n' "${AFTER}") | head -30 || true
fi

# ── Summary ───────────────────────────────────────────────────────────────
printf '\n== hotswap_curl_test: %d PASS, %d FAIL ==\n' "${PASS}" "${FAIL}"
[[ -n "${LOAD_MS:-}" && "${LOAD_MS}" != "null" ]] && \
    printf 'reload duration: %s ms, payload read: %s ms @ %s GiB/s\n' \
        "${LOAD_MS}" "${LOAD_READ_MS}" "${LOAD_GIBS}"
[[ "${FAIL}" -eq 0 ]]

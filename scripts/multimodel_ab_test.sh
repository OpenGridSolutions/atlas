#!/usr/bin/env bash
# Task 6 Phase 1.4 acceptance gate: different-model swap correctness (A→B→A).
#
# Usage:
#   scripts/multimodel_ab_test.sh <MODEL_A> <MODEL_B> [REF_B_FILE]
#
#   MODEL_A / MODEL_B  HF ids with DIFFERENT tokenizers/templates. Use the
#                      exact same id strings the server is booted with —
#                      the snapshot identity hashes over the id.
#   REF_B_FILE         Optional path to a temperature-0 completion produced
#                      by a FRESH STANDALONE MODEL_B process (same PROMPT /
#                      MAX_TOKENS). If absent (or the file doesn't exist),
#                      phase 0 boots B standalone and generates it — that
#                      boot also writes B's snapshot (preseed).
#
# Flow (drives ./serve.sh; requires GPU; run on the GB10 box):
#   Phase 0: ensure B is preseeded (snapshot) + fresh-standalone-B reference.
#   Phase 1: boot A (fresh standalone A) -> snapshot A -> P = temp-0 completion.
#   Phase 2: unload -> load B -> Q = temp-0 completion.
#            Q MUST equal the fresh-standalone-B reference. Comparing B to
#            itself would NOT catch a stale tokenizer/mask — this does.
#   Phase 3: unload -> load A -> P2 = temp-0 completion. P2 MUST equal P.
#
# Env overrides: ATLAS_URL (default http://127.0.0.1:11434), PROMPT,
#   MAX_TOKENS (64), SNAPSHOT_DIR (snapshots), BOOT_TIMEOUT_S (1800),
#   SERVE_SH (./serve.sh), ATLAS_API_KEY (optional bearer token).
set -uo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <MODEL_A> <MODEL_B> [REF_B_FILE]" >&2
    exit 2
fi
MODEL_A="$1"
MODEL_B="$2"
SLUG_B="${MODEL_B//\//--}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-snapshots}"
REF_B_FILE="${3:-${SNAPSHOT_DIR}/${SLUG_B}.ref-temp0.txt}"

URL="${ATLAS_URL:-http://127.0.0.1:11434}"
PROMPT="${PROMPT:-Explain in one sentence why the sky is blue.}"
MAX_TOKENS="${MAX_TOKENS:-64}"
BOOT_TIMEOUT_S="${BOOT_TIMEOUT_S:-1800}"
SERVE_SH="${SERVE_SH:-./serve.sh}"
AUTH_ARGS=()
[[ -n "${ATLAS_API_KEY:-}" ]] && AUTH_ARGS=(-H "Authorization: Bearer ${ATLAS_API_KEY}")

PASS=0
FAIL=0
step() { printf '\n==> %s\n' "$*"; }
ok()   { PASS=$((PASS+1)); printf '    PASS: %s\n' "$*"; }
bad()  { FAIL=$((FAIL+1)); printf '    FAIL: %s\n' "$*"; }
die()  { printf 'ABORT: %s\n' "$*" >&2; stop_server; exit 1; }

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

SERVER_LOG=""
start_server() { # $1 = model id
    SERVER_LOG="logs/multimodel-$(date +%Y%m%d-%H%M%S)-$(basename "$1").log"
    mkdir -p logs
    printf '    booting server with %s (log: %s)\n' "$1" "${SERVER_LOG}"
    ATLAS_MODEL="$1" nohup bash "${SERVE_SH}" >"${SERVER_LOG}" 2>&1 &
    disown || true
}

stop_server() {
    pkill -TERM -f "spark serve" 2>/dev/null || true
    for _ in $(seq 1 20); do
        pgrep -f "spark serve" >/dev/null 2>&1 || return 0
        sleep 1
    done
    pkill -KILL -f "spark serve" 2>/dev/null || true
    sleep 2
}

wait_ready() { # $1 = expected model id
    local deadline=$(( $(date +%s) + BOOT_TIMEOUT_S ))
    while (( $(date +%s) < deadline )); do
        local body state model
        body=$(curl -sS --max-time 5 "${AUTH_ARGS[@]}" "${URL}/admin/models" 2>/dev/null) || {
            sleep 5; continue; }
        state=$(echo "${body}" | jget .state)
        model=$(echo "${body}" | jget .model)
        if [[ "${state}" == "running" && "${model}" == "$1" ]]; then
            return 0
        fi
        sleep 5
    done
    echo "    server did not become ready with '$1' within ${BOOT_TIMEOUT_S}s" >&2
    [[ -n "${SERVER_LOG}" ]] && tail -20 "${SERVER_LOG}" >&2
    return 1
}

completion() { # $1 = model id; prints content
    curl -sS --max-time 600 "${AUTH_ARGS[@]}" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$1\", \"temperature\": 0, \"max_tokens\": ${MAX_TOKENS}, \
             \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}]}" \
        "${URL}/v1/chat/completions" | jget '.choices[0].message.content'
}

snapshot() {
    curl -sS --max-time 900 -X POST "${AUTH_ARGS[@]}" "${URL}/admin/models/snapshot"
}

admin_post() { # $1 = endpoint, $2 = optional JSON body
    if [[ -n "${2:-}" ]]; then
        curl -sS --max-time 1000 -X POST "${AUTH_ARGS[@]}" \
            -H "Content-Type: application/json" -d "$2" "${URL}$1"
    else
        curl -sS --max-time 1000 -X POST "${AUTH_ARGS[@]}" "${URL}$1"
    fi
}

b_snapshot_present() {
    ls "${SNAPSHOT_DIR}/${SLUG_B}-"*.raw >/dev/null 2>&1
}

# ── Phase 0: preseed B + fresh-standalone-B reference ─────────────────────
step "Phase 0: fresh-standalone-B reference + B snapshot"
if [[ -s "${REF_B_FILE}" ]] && b_snapshot_present; then
    ok "reusing reference ${REF_B_FILE} and existing B snapshot"
else
    stop_server
    start_server "${MODEL_B}"
    wait_ready "${MODEL_B}" || die "standalone ${MODEL_B} failed to boot"
    SNAP_B=$(snapshot)
    [[ "$(echo "${SNAP_B}" | jget .path)" != "null" ]] \
        && ok "B snapshot: $(echo "${SNAP_B}" | jget .path)" \
        || bad "B snapshot failed: ${SNAP_B:0:300}"
    REF_B_CONTENT=$(completion "${MODEL_B}")
    if [[ -n "${REF_B_CONTENT}" && "${REF_B_CONTENT}" != "null" ]]; then
        mkdir -p "$(dirname "${REF_B_FILE}")"
        printf '%s' "${REF_B_CONTENT}" > "${REF_B_FILE}"
        ok "fresh-standalone-B reference captured (${#REF_B_CONTENT} chars) -> ${REF_B_FILE}"
    else
        die "no completion from standalone ${MODEL_B}"
    fi
    stop_server
fi
REF_B=$(cat "${REF_B_FILE}")

# ── Phase 1: boot A (fresh standalone), snapshot, baseline P ──────────────
step "Phase 1: boot ${MODEL_A}, snapshot, temperature-0 baseline P"
start_server "${MODEL_A}"
wait_ready "${MODEL_A}" || die "standalone ${MODEL_A} failed to boot"
ARENA_BASE_0=$(curl -sS "${AUTH_ARGS[@]}" "${URL}/admin/models" | jget .arena.base)
SNAP_A=$(snapshot)
[[ "$(echo "${SNAP_A}" | jget .path)" != "null" ]] \
    && ok "A snapshot: $(echo "${SNAP_A}" | jget .path)" \
    || bad "A snapshot failed: ${SNAP_A:0:300}"
P=$(completion "${MODEL_A}")
[[ -n "${P}" && "${P}" != "null" ]] \
    && { ok "P captured (${#P} chars — this IS the fresh-standalone-A reference)"; \
         printf '    ---\n%s\n    ---\n' "${P}"; } \
    || die "no baseline completion from ${MODEL_A}"

# ── Phase 2: swap to B, compare against fresh-standalone-B ────────────────
step "Phase 2: unload -> load ${MODEL_B} -> Q vs fresh-standalone-B"
UNLOAD=$(admin_post /admin/models/unload)
[[ "$(echo "${UNLOAD}" | jget .state)" == "unloaded" ]] \
    && ok "unloaded ($(echo "${UNLOAD}" | jget .freed_allocs) allocations freed)" \
    || bad "unload failed: ${UNLOAD:0:300}"
LOAD_B=$(admin_post /admin/models/load "{\"model\": \"${MODEL_B}\"}")
LOAD_B_MS=$(echo "${LOAD_B}" | jget .duration_ms)
if [[ "${LOAD_B_MS}" != "null" && -n "${LOAD_B_MS}" ]]; then
    ok "loaded ${MODEL_B}: ${LOAD_B_MS} ms total ($(echo "${LOAD_B}" | jget .gib_per_s) GiB/s, \
kernel target $(echo "${LOAD_B}" | jget .kernel_target))"
else
    bad "load B failed: ${LOAD_B:0:400}"
    die "cannot continue without B"
fi
ACTIVE=$(curl -sS "${AUTH_ARGS[@]}" "${URL}/admin/models" | jget .model)
[[ "${ACTIVE}" == "${MODEL_B}" ]] \
    && ok "active model reports ${MODEL_B}" \
    || bad "active model is '${ACTIVE}', expected '${MODEL_B}'"
Q=$(completion "${MODEL_B}")
if [[ -z "${Q}" || "${Q}" == "null" ]]; then
    bad "no completion from swapped-in ${MODEL_B}"
elif [[ "${Q}" == "${REF_B}" ]]; then
    ok "Q IDENTICAL to fresh-standalone-B reference (tokenizer + masks swapped correctly)"
else
    bad "Q DIFFERS from fresh-standalone-B — THE SEAMS ARE INCOMPLETE (stale tokenizer/mask/runtime). Do not paper over this."
    printf '    --- Q (swapped-in B) ---\n%s\n    --- reference (fresh B) ---\n%s\n' "${Q}" "${REF_B}"
    diff <(printf '%s\n' "${REF_B}") <(printf '%s\n' "${Q}") | head -30 || true
fi

# ── Phase 3: swap back to A, P2 must equal P ──────────────────────────────
step "Phase 3: unload -> load ${MODEL_A} -> P2 vs P"
UNLOAD2=$(admin_post /admin/models/unload)
[[ "$(echo "${UNLOAD2}" | jget .state)" == "unloaded" ]] \
    && ok "unloaded B" \
    || bad "unload B failed: ${UNLOAD2:0:300}"
LOAD_A=$(admin_post /admin/models/load "{\"model\": \"${MODEL_A}\"}")
LOAD_A_MS=$(echo "${LOAD_A}" | jget .duration_ms)
[[ "${LOAD_A_MS}" != "null" && -n "${LOAD_A_MS}" ]] \
    && ok "loaded ${MODEL_A} back: ${LOAD_A_MS} ms total" \
    || { bad "load A failed: ${LOAD_A:0:400}"; die "cannot continue"; }
P2=$(completion "${MODEL_A}")
if [[ -z "${P2}" || "${P2}" == "null" ]]; then
    bad "no completion from swapped-back ${MODEL_A}"
elif [[ "${P2}" == "${P}" ]]; then
    ok "P2 IDENTICAL to P (A -> B -> A round trip is clean)"
else
    bad "P2 DIFFERS from P after the round trip"
    diff <(printf '%s\n' "${P}") <(printf '%s\n' "${P2}") | head -30 || true
fi
ARENA_BASE_1=$(curl -sS "${AUTH_ARGS[@]}" "${URL}/admin/models" | jget .arena.base)
[[ "${ARENA_BASE_1}" == "${ARENA_BASE_0}" ]] \
    && ok "arena base constant across the whole cycle (${ARENA_BASE_1})" \
    || bad "arena base CHANGED: ${ARENA_BASE_0} -> ${ARENA_BASE_1}"

# ── Summary ────────────────────────────────────────────────────────────────
printf '\n== multimodel_ab_test (%s <-> %s): %d PASS, %d FAIL ==\n' \
    "${MODEL_A}" "${MODEL_B}" "${PASS}" "${FAIL}"
[[ -n "${LOAD_B_MS:-}" && "${LOAD_B_MS}" != "null" ]] && \
    printf 'swap A->B: %s ms, swap B->A: %s ms\n' "${LOAD_B_MS}" "${LOAD_A_MS:-?}"
stop_server
[[ "${FAIL}" -eq 0 ]]

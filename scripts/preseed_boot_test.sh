#!/usr/bin/env bash
# Task 6 Phase 2.2 acceptance: preseed boot ends IDLE.
#
# Usage:
#   scripts/preseed_boot_test.sh <MODEL_A> <MODEL_B>
#
# Boots the server with ATLAS_PRESEED_MODELS='["A","B"]' (A also the boot
# model) and asserts the plan's post-condition once HTTP is up:
#   - GET /admin/models: state == "unloaded", arena used_bytes == 0
#   - both models listed with snapshot_present == true
#   - GET /v1/models lists both ids
#
# First run initializes missing snapshots from safetensors (minutes);
# subsequent runs skip via the hash check and boot fast.
#
# Env: ATLAS_URL (http://127.0.0.1:11434), BOOT_TIMEOUT_S (2400),
#      SERVE_SH (./serve.sh), KEEP_RUNNING=1 to leave the idle server up
#      (e.g. to chain scripts/lazy_swap_concurrency_test.sh).
set -uo pipefail

[[ $# -lt 2 ]] && { echo "usage: $0 <MODEL_A> <MODEL_B>" >&2; exit 2; }
MODEL_A="$1"
MODEL_B="$2"
URL="${ATLAS_URL:-http://127.0.0.1:11434}"
BOOT_TIMEOUT_S="${BOOT_TIMEOUT_S:-2400}"
SERVE_SH="${SERVE_SH:-./serve.sh}"
AUTH_ARGS=()
[[ -n "${ATLAS_API_KEY:-}" ]] && AUTH_ARGS=(-H "Authorization: Bearer ${ATLAS_API_KEY}")

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); printf '    PASS: %s\n' "$*"; }
bad() { FAIL=$((FAIL+1)); printf '    FAIL: %s\n' "$*"; }

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
    print(json.dumps(doc) if isinstance(doc,(list,dict)) else doc)
except Exception:
    print('null')
" "$1"; }
fi

stop_server() {
    pkill -TERM -f "spark serve" 2>/dev/null || true
    for _ in $(seq 1 20); do pgrep -f "spark serve" >/dev/null 2>&1 || return 0; sleep 1; done
    pkill -KILL -f "spark serve" 2>/dev/null || true
}

printf '==> booting with ATLAS_PRESEED_MODELS=["%s","%s"]\n' "${MODEL_A}" "${MODEL_B}"
stop_server
mkdir -p logs
LOG="logs/preseed-boot-$(date +%Y%m%d-%H%M%S).log"
ATLAS_MODEL="${MODEL_A}" \
ATLAS_PRESEED_MODELS="[\"${MODEL_A}\",\"${MODEL_B}\"]" \
    nohup bash "${SERVE_SH}" >"${LOG}" 2>&1 &
disown || true
printf '    log: %s\n' "${LOG}"

deadline=$(( $(date +%s) + BOOT_TIMEOUT_S ))
ADMIN=""
while (( $(date +%s) < deadline )); do
    ADMIN=$(curl -sS --max-time 5 "${AUTH_ARGS[@]}" "${URL}/admin/models" 2>/dev/null) && \
        [[ -n "${ADMIN}" ]] && break
    if ! pgrep -f "spark serve" >/dev/null 2>&1 && (( $(date +%s) > $(date +%s) )); then
        # give the pipeline a few seconds to actually spawn before judging
        sleep 5
        pgrep -f "spark serve" >/dev/null 2>&1 || { echo "server process died — log tail:"; tail -30 "${LOG}"; exit 1; }
    fi
    sleep 5
done
[[ -n "${ADMIN}" ]] || { echo "HTTP never came up within ${BOOT_TIMEOUT_S}s"; tail -30 "${LOG}"; exit 1; }
echo "==> server up; asserting preseed post-condition"
echo "    ${ADMIN}"

STATE=$(echo "${ADMIN}" | jget .state)
USED=$(echo "${ADMIN}" | jget .arena.used_bytes)
[[ "${STATE}" == "unloaded" ]] && ok "state == unloaded" || bad "state == '${STATE}' (want unloaded)"
[[ "${USED}" == "0" ]] && ok "arena used_bytes == 0" || bad "arena used_bytes == ${USED} (want 0)"

for M in "${MODEL_A}" "${MODEL_B}"; do
    ENTRY=$(echo "${ADMIN}" | jget ".models" | python3 -c "
import sys, json
models = json.load(sys.stdin)
for m in models if isinstance(models, list) else []:
    if m.get('id') == '$M':
        print(json.dumps(m)); break
else:
    print('null')
" 2>/dev/null || echo null)
    if [[ "${ENTRY}" == "null" || -z "${ENTRY}" ]]; then
        bad "'${M}' missing from /admin/models models[]"
        continue
    fi
    SNAP=$(echo "${ENTRY}" | jget .snapshot_present)
    MSTATE=$(echo "${ENTRY}" | jget .state)
    [[ "${SNAP}" == "true" ]] && ok "'${M}' snapshot_present" || bad "'${M}' snapshot_present == ${SNAP}"
    [[ "${MSTATE}" == "unloaded" ]] && ok "'${M}' state unloaded" || bad "'${M}' state == ${MSTATE}"
done

V1=$(curl -sS "${AUTH_ARGS[@]}" "${URL}/v1/models")
for M in "${MODEL_A}" "${MODEL_B}"; do
    echo "${V1}" | grep -q "\"${M}\"" \
        && ok "/v1/models lists '${M}'" \
        || bad "/v1/models missing '${M}': ${V1:0:300}"
done

printf '\n== preseed_boot_test: %d PASS, %d FAIL ==\n' "${PASS}" "${FAIL}"
if [[ "${KEEP_RUNNING:-0}" != "1" ]]; then
    stop_server
else
    echo "(KEEP_RUNNING=1 — idle preseeded server left up for follow-up tests)"
fi
[[ "${FAIL}" -eq 0 ]]

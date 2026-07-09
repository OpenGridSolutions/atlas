#!/usr/bin/env bash
# Task 6 Phase 3.2 acceptance: lazy-swap concurrency semantics.
#
# Usage:
#   scripts/lazy_swap_concurrency_test.sh <MODEL_A> <MODEL_B>
#
# Expects a PRESEEDED server already running (e.g. via
#   KEEP_RUNNING=1 scripts/preseed_boot_test.sh <A> <B>).
#
# Asserts:
#   1. First request per model lazy-loads it (X-Atlas-Model-Swap header).
#   2. A concurrent same-target burst produces EXACTLY ONE swap (only one
#      response carries the swap header; all return the same temp-0 content).
#   3. Interleaved A/B bursts: every 200 echoes the requested model and its
#      exact per-model temp-0 reference content (no wrong-model answers);
#      requests hitting a foreign in-flight swap get 503 + Retry-After with
#      code model_swap_in_flight; everything terminates (no deadlock).
#
# Env: ATLAS_URL (http://127.0.0.1:11434), PROMPT, MAX_TOKENS (48),
#      BURST (4), ROUNDS (6).
set -uo pipefail

[[ $# -lt 2 ]] && { echo "usage: $0 <MODEL_A> <MODEL_B>" >&2; exit 2; }
MODEL_A="$1"; MODEL_B="$2"
URL="${ATLAS_URL:-http://127.0.0.1:11434}"
PROMPT="${PROMPT:-Explain in one sentence why the sky is blue.}"
MAX_TOKENS="${MAX_TOKENS:-48}"
BURST="${BURST:-4}"
ROUNDS="${ROUNDS:-6}"
AUTH_ARGS=()
[[ -n "${ATLAS_API_KEY:-}" ]] && AUTH_ARGS=(-H "Authorization: Bearer ${ATLAS_API_KEY}")
WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT

PASS=0; FAIL=0
step() { printf '\n==> %s\n' "$*"; }
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
    print(doc)
except Exception:
    print('null')
" "$1"; }
fi

# fire <model> <outprefix>: writes .code/.headers/.body
fire() {
    local model="$1" out="$2"
    local code
    code=$(curl -sS --max-time 1200 -o "${out}.body" -D "${out}.headers" -w '%{http_code}' \
        "${AUTH_ARGS[@]}" -H "Content-Type: application/json" \
        -d "{\"model\": \"${model}\", \"temperature\": 0, \"max_tokens\": ${MAX_TOKENS}, \
             \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}]}" \
        "${URL}/v1/chat/completions" 2>"${out}.err")
    echo "${code}" > "${out}.code"
}
swap_hdr()   { grep -i '^x-atlas-model-swap:' "$1.headers" | tr -d '\r' | awk '{print $2}'; }
body_model() { jget .model < "$1.body"; }
body_text()  { jget '.choices[0].message.content' < "$1.body"; }
body_code()  { jget .error.code < "$1.body"; }

ADMIN=$(curl -sS --max-time 5 "${AUTH_ARGS[@]}" "${URL}/admin/models") || {
    echo "no server at ${URL} — run preseed_boot_test.sh with KEEP_RUNNING=1 first" >&2; exit 1; }
echo "server state: $(echo "${ADMIN}" | jget .state), active: $(echo "${ADMIN}" | jget .model)"

# ── Phase 1: sequential lazy loads + per-model references ────────────────
step "Phase 1: first request per model lazy-loads it"
fire "${MODEL_A}" "${WORK}/refA"
[[ "$(cat "${WORK}/refA.code")" == "200" ]] || { bad "A request failed: $(head -c 200 "${WORK}/refA.body")"; exit 1; }
REF_A=$(body_text "${WORK}/refA")
HDR_A=$(swap_hdr "${WORK}/refA")
[[ -n "${HDR_A}" ]] && ok "A lazy-loaded (swap ${HDR_A} ms)" \
                    || ok "A already active (no swap header — fine after warm boot)"
fire "${MODEL_B}" "${WORK}/refB"
[[ "$(cat "${WORK}/refB.code")" == "200" ]] || { bad "B request failed: $(head -c 200 "${WORK}/refB.body")"; exit 1; }
REF_B=$(body_text "${WORK}/refB")
HDR_B=$(swap_hdr "${WORK}/refB")
[[ -n "${HDR_B}" ]] && ok "B lazy-swapped in (swap ${HDR_B} ms)" || bad "B request carried no swap header"
[[ -n "${REF_A}" && -n "${REF_B}" && "${REF_A}" != "null" && "${REF_B}" != "null" ]] \
    && ok "temp-0 references captured (A ${#REF_A} chars, B ${#REF_B} chars)" \
    || { bad "empty reference content"; exit 1; }

# ── Phase 2: concurrent same-target burst → exactly one swap ─────────────
step "Phase 2: ${BURST} concurrent requests for ${MODEL_A} (B active) — one swap"
for i in $(seq 1 "${BURST}"); do fire "${MODEL_A}" "${WORK}/burst${i}" & done
wait
SWAPS=0; OKS=0
for i in $(seq 1 "${BURST}"); do
    CODE=$(cat "${WORK}/burst${i}.code")
    if [[ "${CODE}" == "200" ]]; then
        OKS=$((OKS+1))
        [[ "$(body_text "${WORK}/burst${i}")" == "${REF_A}" ]] \
            || bad "burst ${i}: content differs from A reference"
        [[ "$(body_model "${WORK}/burst${i}")" == "${MODEL_A}" ]] \
            || bad "burst ${i}: model echo '$(body_model "${WORK}/burst${i}")' != ${MODEL_A}"
        [[ -n "$(swap_hdr "${WORK}/burst${i}")" ]] && SWAPS=$((SWAPS+1))
    else
        bad "burst ${i}: HTTP ${CODE} (same-target requests must wait, not fail): $(head -c 200 "${WORK}/burst${i}.body")"
    fi
done
[[ "${OKS}" -eq "${BURST}" ]] && ok "all ${BURST} same-target requests served" || true
[[ "${SWAPS}" -eq 1 ]] && ok "exactly one swap in the burst (no double-swap)" \
                       || bad "${SWAPS} swap headers in same-target burst (want exactly 1)"

# ── Phase 3: interleaved A/B under concurrency ────────────────────────────
step "Phase 3: ${ROUNDS} rounds of interleaved A/B pairs"
IDX=0
for r in $(seq 1 "${ROUNDS}"); do
    IDX=$((IDX+1)); fire "${MODEL_A}" "${WORK}/mix${IDX}" &
    IDX=$((IDX+1)); fire "${MODEL_B}" "${WORK}/mix${IDX}" &
    wait
done
WRONG=0; REJ=0; SRV=0; SWAPT=0
for i in $(seq 1 "${IDX}"); do
    WANT=$([[ $((i % 2)) == 1 ]] && echo "${MODEL_A}" || echo "${MODEL_B}")
    WANT_REF=$([[ $((i % 2)) == 1 ]] && echo "${REF_A}" || echo "${REF_B}")
    CODE=$(cat "${WORK}/mix${i}.code")
    case "${CODE}" in
        200)
            SRV=$((SRV+1))
            [[ -n "$(swap_hdr "${WORK}/mix${i}")" ]] && SWAPT=$((SWAPT+1))
            if [[ "$(body_model "${WORK}/mix${i}")" != "${WANT}" || \
                  "$(body_text "${WORK}/mix${i}")" != "${WANT_REF}" ]]; then
                WRONG=$((WRONG+1))
                bad "mix ${i}: WRONG-MODEL ANSWER (asked ${WANT}, got model '$(body_model "${WORK}/mix${i}")')"
            fi
            ;;
        503)
            REJ=$((REJ+1))
            [[ "$(body_code "${WORK}/mix${i}")" == "model_swap_in_flight" ]] \
                || bad "mix ${i}: 503 without model_swap_in_flight code: $(head -c 200 "${WORK}/mix${i}.body")"
            grep -qi '^retry-after:' "${WORK}/mix${i}.headers" \
                || bad "mix ${i}: 503 without Retry-After header"
            ;;
        *)
            bad "mix ${i}: unexpected HTTP ${CODE}: $(head -c 200 "${WORK}/mix${i}.body")"
            ;;
    esac
done
[[ "${WRONG}" -eq 0 ]] && ok "no wrong-model answers across ${IDX} interleaved requests"
ok "served ${SRV}, rejected-during-swap ${REJ}, swaps paid ${SWAPT} (thrash is expected; wrongness is not)"

# Optional metrics cross-check (present once the metrics commit is live).
METRICS=$(curl -sS --max-time 5 "${URL}/metrics" 2>/dev/null | grep -E "^atlas_model_swaps_total" || true)
[[ -n "${METRICS}" ]] && printf '    metrics: %s\n' "${METRICS}"

printf '\n== lazy_swap_concurrency_test: %d PASS, %d FAIL ==\n' "${PASS}" "${FAIL}"
[[ "${FAIL}" -eq 0 ]]

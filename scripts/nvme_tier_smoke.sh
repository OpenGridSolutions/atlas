#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# On-box smoke test for the NVMe KV tier + pressure scheduler
# (requirements doc 2026-10-05; analysis: docs/design/nvme-tier-analysis.md).
#
# Boots the release server with two preseeded generative models and the
# NVMe tier enabled, then checks:
#   1. /health reaches ready (idle arena after preseed).
#   2. First chat request lazy-loads model A via the dispatcher
#      (X-Atlas-Model-Swap header = queue wait) and answers.
#   3. on_release persist: block-store files appear under the cache dir.
#   4. Identical second request hits the RAM prefix cache
#      (usage.prompt_tokens_details.cached_tokens > 0).
#   5. A request for model B queues, the dispatcher swaps (idle trigger),
#      B answers; /internal/state shows the swap decision + swap costs.
#   6. After the swap back to A (third request), the NVMe restore path
#      can serve A's prefix (atlas_nvme_restored_tokens_total > 0 when
#      the prompt is long enough; informational otherwise).
#
# Usage:
#   MODEL_A=... MODEL_B=... ./scripts/nvme_tier_smoke.sh
# Defaults match the deployment preseed set. NEVER run while a
# production instance is serving on the same port.

set -euo pipefail

PORT="${PORT:-8899}"
BASE="http://127.0.0.1:${PORT}"
MODEL_A="${MODEL_A:?set MODEL_A to a preseeded generative model id}"
MODEL_B="${MODEL_B:?set MODEL_B to a second preseeded generative model id}"
CACHE_DIR="${CACHE_DIR:-/var/tmp/atlas-nvme-smoke}"
BIN="${BIN:-target/release/spark}"
LOG="logs/nvme-smoke-$(date +%Y%m%d-%H%M%S).log"

mkdir -p "$(dirname "$LOG")" "$CACHE_DIR"

if pgrep -af "spark serve" | grep -v $$ | grep -q serve; then
    echo "FATAL: a spark server is already running — refusing to start" >&2
    exit 1
fi

echo "== booting (log: $LOG)"
ATLAS_PRESEED_GENERATIVE="[\"$MODEL_A\",\"$MODEL_B\"]" \
    "$BIN" serve "$MODEL_A" \
    --port "$PORT" \
    --enable-prefix-caching \
    --nvme-cache-dir "$CACHE_DIR" \
    --max-wait-hard-s 60 --scheduler-interval-ms 200 \
    >"$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "== waiting for /health ready"
for _ in $(seq 1 600); do
    if curl -sf "$BASE/health" | grep -q '"ready"'; then break; fi
    sleep 2
done
curl -sf "$BASE/health" || { echo "FATAL: health never ready"; exit 1; }
echo

chat() { # chat <model> <prompt> -> response json
    curl -sf -D /tmp/smoke_headers.txt "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}],\"max_tokens\":32,\"temperature\":0}"
}

PROMPT="Explain in two sentences why the sky is blue. Then count from one to twenty in words."

echo "== 1st request on $MODEL_A (dispatcher lazy-load)"
R1=$(chat "$MODEL_A" "$PROMPT")
grep -i "x-atlas-model-swap" /tmp/smoke_headers.txt || echo "(no swap header — model was already resident)"
echo "$R1" | python3 -c 'import json,sys; r=json.load(sys.stdin); assert r["choices"][0]["message"]["content"], "empty answer"; print("OK: answered,", r["usage"])'

echo "== waiting for on_release persist"
sleep 3
KVB=$(find "$CACHE_DIR" -name "*.kvb" | wc -l)
echo "block-store .kvb files: $KVB"
[ "$KVB" -gt 0 ] || { echo "FATAL: no persisted KV blocks (NVM-1)"; exit 1; }

echo "== 2nd identical request (RAM prefix-cache hit)"
R2=$(chat "$MODEL_A" "$PROMPT")
echo "$R2" | python3 -c '
import json,sys
r=json.load(sys.stdin)
cached=r["usage"].get("prompt_tokens_details",{}).get("cached_tokens",0)
print("cached_tokens:",cached)
assert cached>0, "expected a RAM prefix-cache hit"
print("OK")'

echo "== request for $MODEL_B (pressure/idle swap via dispatcher)"
R3=$(chat "$MODEL_B" "hello, which model are you?")
echo "$R3" | python3 -c 'import json,sys; r=json.load(sys.stdin); assert r["choices"][0]["message"]["content"]; print("OK: model B answered")'
grep -i "x-atlas-model-swap" /tmp/smoke_headers.txt && echo "OK: request waited for the swap"

echo "== /internal/state"
curl -sf "$BASE/internal/state" | python3 -c '
import json,sys
s=json.load(sys.stdin)
print("arena:",s["arena"]["model"],s["arena"]["state"])
print("last_decision:",s["scheduler"]["last_decision"].get("decision") if s.get("scheduler") else None)
print("nvme entries:",(s.get("nvme_store") or {}).get("entries"))
assert s.get("nvme_store"), "nvme tier missing from state"'

echo "== back to $MODEL_A (restore path exercises on long prompts)"
chat "$MODEL_A" "$PROMPT" >/dev/null
curl -sf "$BASE/metrics" | grep -E "atlas_nvme_(restored_tokens|read_bytes|write_bytes)_total|atlas_model_swaps_total" || true
grep -E "swap_complete|nvme-restore" "$LOG" | tail -5 || true

echo "== PASS (inspect $LOG for swap_decision/swap_complete events)"

#!/usr/bin/env bash
# Full test cycle inside the dev container: cleanup -> build -> serve.
# Keep this in the repo (bind-mounted at /workspace) so flags stay editable
# without rebuilding the image. Run it AFTER `podman exec -it atlas-gb10-dev bash`.
set -euo pipefail

BIN="${ATLAS_BIN:-/build/target/release/spark}"
MODEL="${ATLAS_MODEL:-nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4}"

echo "==> [1/3] cleanup: stopping any running spark & freeing GPU memory ..."
# Terminate a previous server so it releases its VRAM before we rebuild/rerun.
pkill -TERM -f "${BIN} serve" 2>/dev/null || true
for _ in $(seq 1 10); do
    pgrep -f "${BIN} serve" >/dev/null 2>&1 || break
    sleep 1
done
pkill -KILL -f "${BIN} serve" 2>/dev/null || true
sleep 1
echo "    GPU memory after cleanup (used, free):"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null || true

echo "==> [2/3] building spark-server (release) ..."
cargo build --release -p spark-server
# Confirm the cache is doing its job.
command -v sccache >/dev/null 2>&1 && sccache --show-stats || true

echo "==> [3/3] serving ${MODEL} for testing ..."
# GB10 unified memory: cudaMemGetInfo cannot see reclaimable UVM pages from a
# previous run, so the loader's OOM pre-flight false-negatives after a restart.
# Safety nets that remain: the per-shard OOM guard (unconditional per shard
# since the Task 0b fix) and the per-tensor UVM-fallback allocation.
export ATLAS_SKIP_OOM_PREFLIGHT=1
# Phase K bridge for containers created before the categorized preseed
# vars landed in .devpod/compose.yml: the removed flat var must not leak
# into the server (it hard-fails on it), and the five category vars get
# the compose.yml defaults when the stale container env lacks them.
# After a container recreation the compose-provided values win.
unset ATLAS_PRESEED_MODELS
export ATLAS_PRESEED_GENERATIVE="${ATLAS_PRESEED_GENERATIVE:-[
  \"nvidia/Qwen3-Next-80B-A3B-Thinking-NVFP4\",
  \"vincentzed-hf/Qwen3-Coder-Next-NVFP4\",
  \"Qwen/Qwen3-VL-30B-A3B-Instruct-FP8\"
]}"
export ATLAS_PRESEED_EMBED="${ATLAS_PRESEED_EMBED:-[
  \"Qwen/Qwen3-Embedding-8B\",
  \"Qwen/Qwen3-VL-Embedding-8B\"
]}"
export ATLAS_PRESEED_RERANK="${ATLAS_PRESEED_RERANK:-[
  \"Qwen/Qwen3-Reranker-8B\",
  \"Qwen/Qwen3-VL-Reranker-8B\"
]}"
export ATLAS_PRESEED_TRANSCRIBE="${ATLAS_PRESEED_TRANSCRIBE:-[\"openai/whisper-large-v3\"]}"
export ATLAS_PRESEED_TRANSLATE="${ATLAS_PRESEED_TRANSLATE:-[\"google/translategemma-12b-it\"]}"
# Retired checkpoint (2026-07-12, Plan.md Q): ig1's NVFP4 quant has a
# defective vision encoder — container envs created before the swap still
# list it; substitute the official FP8 release until the container is
# recreated from the updated compose.yml.
export ATLAS_PRESEED_GENERATIVE="${ATLAS_PRESEED_GENERATIVE//ig1\/Qwen3-VL-30B-A3B-Instruct-NVFP4/Qwen\/Qwen3-VL-30B-A3B-Instruct-FP8}"
LOG="/workspace/logs/serve-$(date +%Y%m%d-%H%M%S).log"
echo "==> logging to ${LOG}"

exec "${BIN}" serve "${MODEL}" \
    --bind 0.0.0.0 \
    --port 11434 \
    --max-seq-len 104416 \
    --max-batch-size 1 \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization 0.90 \
    --kv-high-precision-layers 3 \
    --scheduling-policy slai \
    --mtp-quantization nvfp4 \
    --enable-prefix-caching \
    2>&1 | tee "${LOG}"

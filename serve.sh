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
# The per-shard OOM guard and the UVM fallback are the real safety nets.
export ATLAS_SKIP_OOM_PREFLIGHT=1
exec "${BIN}" serve "${MODEL}" \
    --bind 0.0.0.0 \
    --port 11434 \
    --max-seq-len 104416 \
    --max-batch-size 1 \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization 0.85 \
    --kv-high-precision-layers 3 \
    --scheduling-policy slai \
    --mtp-quantization nvfp4 \
    --enable-prefix-caching

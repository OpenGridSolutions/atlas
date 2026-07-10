#!/usr/bin/env bash
# bf16 target smoke — Plan.md (multiclass) step 1.0 HARD GATE.
#
# Builds the kernel tree including the new qwen3-dense-8b/bf16 target and
# runs the GPU smoke harness (dense_gemm_bf16 / dense_gemv_bf16 / rms_norm
# vs CPU references on the bf16 target's own modules).
#
# Run ON THE GB10 BOX:   scripts/bf16_smoke_test.sh
# Exit 0 = gate passed (1.1+ work may start), non-zero = bf16 surprise.
set -euo pipefail
cd "$(dirname "$0")/.."

export ATLAS_TARGET_HW="${ATLAS_TARGET_HW:-gb10}"
export ATLAS_TARGET_MODEL='*'
export ATLAS_TARGET_QUANT='*'

echo "── build (must include target qwen3-dense-8b/bf16) ──"
# A NEWLY ADDED target dir is not covered by the build script's
# rerun-if-changed set (it only watches files that existed at the last
# run) — force the target rescan.
touch crates/atlas-kernels/build.rs
BUILD_LOG=$(mktemp)
cargo build --release -p spark-model --features cuda,gpu-examples \
    --example bf16_target_smoke 2>&1 | tee "$BUILD_LOG" | grep -E "atlas-kernels: compiled|error" || true
if ! grep -q "qwen3-dense-8b, bf16" "$BUILD_LOG"; then
    echo "FAIL: build log has no 'qwen3-dense-8b, bf16' target line" >&2
    exit 1
fi
echo "── target compiled ✓ — running GPU smoke ──"

cargo run --release -p spark-model --features cuda,gpu-examples \
    --example bf16_target_smoke

echo "── bf16 smoke: PASS — Plan.md 1.0 gate satisfied ──"

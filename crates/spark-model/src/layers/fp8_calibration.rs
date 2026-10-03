// SPDX-License-Identifier: AGPL-3.0-only

//! Online FP8 KV cache scale calibration.
//!
//! Computes per-tensor scales from the FIRST observed K/V batch (the first
//! prefill chunk, before anything is written to the cache):
//! `scale = absmax * HEADROOM / 448.0`, frozen immediately and forever.
//!
//! Freeze-at-first-observe is load-bearing, not a simplification: scales
//! are per-layer globals shared by the write AND read kernels, so any entry
//! written under scale A and read under a later scale B is off by B/A —
//! there is no requantization pass. The previous design (256-token warmup
//! at scale 2.0, then freeze) corrupted exactly the first short-prompt
//! request of a fresh build: its early KV entries, written at 2.0, were
//! attenuated ~10-100× once the mid-generation freeze switched the scale,
//! and the response degenerated into token loops from that point on. Long
//! first prompts (>= warmup tokens in one chunk) froze before the first
//! write and masked the bug — which is why suites with system prompts
//! passed while a minimal curl failed.
//!
//! The HEADROOM factor absorbs the smaller sample (one chunk instead of
//! 256 tokens). It is nearly free: E4M3 is a float format, so a generous
//! scale costs no relative precision — it only raises the denormal floor
//! (~scale * 0.002), far below meaningful K/V magnitudes.
//!
//! Thread safety: uses `parking_lot::Mutex` for interior mutability. The lock
//! is uncontended (single inference thread) so lock overhead is negligible.

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// FP8 E4M3 max representable magnitude.
const FP8_E4M3_MAX: f32 = 448.0;

/// Minimum scale to prevent division by zero or denormalized values.
const MIN_SCALE: f32 = 1e-12;

/// Safety factor on the first-batch absmax: later tokens may exceed the
/// first chunk's dynamic range. Overestimation is nearly free for a float
/// format (see module docs), underestimation clips.
const HEADROOM: f32 = 4.0;

/// Mutable calibration state protected by Mutex for Send + Sync.
struct CalibrationInner {
    /// Running max of |K| values observed so far.
    k_running_max: f32,
    /// Running max of |V| values observed so far.
    v_running_max: f32,
    /// Total tokens processed during calibration.
    tokens_seen: usize,
    /// Whether calibration is complete (scales frozen).
    frozen: bool,
    /// Calibrated k_scale (set once after warmup).
    k_scale: f32,
    /// Calibrated v_scale (set once after warmup).
    v_scale: f32,
}

/// Online FP8 KV cache scale calibration tracker for one attention layer.
///
/// Wraps calibration state in a Mutex so it can live inside a `Send + Sync`
/// struct (required by `TransformerLayer` trait).
pub struct Fp8KvCalibration {
    inner: Mutex<CalibrationInner>,
    /// MODEL.toml `fp8_kv_calibration_tokens` — kept for the decode path's
    /// CUDA-graph re-enable heuristic; the freeze itself happens at the
    /// first observation regardless of this value.
    #[allow(dead_code)]
    warmup_tokens: usize,
    /// GPU buffer for absmax reduction output: `[1]` f32 for K, `[1]` f32 for V.
    /// Layout: `[k_absmax: f32, v_absmax: f32]` = 8 bytes.
    absmax_buf: DevicePtr,
    /// Kernel handle for bf16_absmax reduction.
    absmax_kernel: KernelHandle,
}

// SAFETY: DevicePtr is a raw GPU pointer (u64). It is only accessed from the
// inference thread that owns the CUDA context. The Mutex guards the mutable
// calibration state. All kernel launches are serialized on the CUDA stream.
unsafe impl Send for Fp8KvCalibration {}
unsafe impl Sync for Fp8KvCalibration {}

impl Fp8KvCalibration {
    /// Create a new calibration tracker.
    ///
    /// `warmup_tokens`: the MODEL.toml calibration-token count (scales
    /// freeze at the first observed batch; see module docs).
    /// `gpu`: GPU backend for allocating the absmax reduction buffer.
    pub fn new(warmup_tokens: usize, gpu: &dyn GpuBackend) -> Result<Self> {
        let absmax_kernel = gpu.kernel("reshape_and_cache", "bf16_absmax")?;
        // Allocate 8 bytes: [k_absmax: f32, v_absmax: f32]
        let absmax_buf = gpu.alloc(8)?;
        // Initialize to zero
        let zeros = [0u8; 8];
        gpu.copy_h2d(&zeros, absmax_buf)?;

        Ok(Self {
            inner: Mutex::new(CalibrationInner {
                k_running_max: 0.0,
                v_running_max: 0.0,
                tokens_seen: 0,
                frozen: false,
                // Pre-observe default, never used for a cache write
                // (observe() runs before the first write and freezes the
                // real scales). 2.0 covers ±896, safe for all known models
                // (Gemma-4 26B / Mistral reach |K| ~600).
                k_scale: 2.0,
                v_scale: 2.0,
            }),
            warmup_tokens,
            absmax_buf,
            absmax_kernel,
        })
    }

    /// Whether calibration is still in warmup phase (scales not yet frozen).
    pub fn is_calibrating(&self) -> bool {
        let inner = self.inner.lock();
        !inner.frozen
    }

    /// Get current scales. Returns (k_scale, v_scale).
    ///
    /// Before the first `observe` call: the conservative (2.0, 2.0)
    /// defaults — never used for a cache write, because `observe` runs
    /// first. After it: the frozen calibrated scales.
    pub fn scales(&self) -> (f32, f32) {
        let inner = self.inner.lock();
        (inner.k_scale, inner.v_scale)
    }

    /// Observe K/V projection outputs and update running max.
    ///
    /// Launches absmax reduction kernels on the K and V buffers, then reads
    /// the results back to CPU after a sync. Call this AFTER K/V projections
    /// and BEFORE writing to the KV cache.
    ///
    /// `k_data`: device BF16 buffer of K projection output
    /// `v_data`: device BF16 buffer of V projection output
    /// `num_tokens`: number of tokens in the batch
    /// `num_kv_heads`: number of KV heads
    /// `head_dim`: dimension per head
    pub fn observe(
        &self,
        gpu: &dyn GpuBackend,
        k_data: DevicePtr,
        v_data: DevicePtr,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        stream: u64,
    ) -> Result<()> {
        // Observe during warmup (always) and periodically after (every 512 tokens)
        // to catch distribution shifts in multi-turn conversations. Without periodic
        // recalibration, FP8 KV scales become stale as context grows, causing recent
        // K values to collapse into fewer E4M3 quantization buckets.
        // Recalibrate every 128 tokens (was 512) to catch distribution shifts
        // in multi-turn conversations where K/V statistics change frequently.
        let should_observe = {
            let inner = self.inner.lock();
            !inner.frozen || (inner.tokens_seen % 128 < num_tokens as usize)
        };
        if !should_observe {
            return Ok(());
        }

        let n_elems = num_tokens * num_kv_heads * head_dim;

        // Reset absmax buffer to 0.0 before reduction (async to avoid sync/async conflict)
        gpu.memset_async(self.absmax_buf, 0, 8, stream)?;

        // Launch absmax for K
        let k_out = self.absmax_buf;
        super::ops::bf16_absmax(gpu, self.absmax_kernel, k_data, k_out, n_elems, stream)?;

        // Launch absmax for V (write to offset 4 = second f32)
        let v_out = self.absmax_buf.offset(4);
        super::ops::bf16_absmax(gpu, self.absmax_kernel, v_data, v_out, n_elems, stream)?;

        // Sync and read back
        gpu.synchronize(stream)?;
        let mut result_buf = [0u8; 8];
        gpu.copy_d2h(self.absmax_buf, &mut result_buf)?;
        let k_max =
            f32::from_le_bytes([result_buf[0], result_buf[1], result_buf[2], result_buf[3]]);
        let v_max =
            f32::from_le_bytes([result_buf[4], result_buf[5], result_buf[6], result_buf[7]]);

        // Update running max and check if warmup is complete
        let mut inner = self.inner.lock();
        inner.k_running_max = inner.k_running_max.max(k_max);
        inner.v_running_max = inner.v_running_max.max(v_max);
        inner.tokens_seen += num_tokens as usize;

        if !inner.frozen {
            // Freeze on the FIRST observation: it runs before the first
            // cache write, so every entry this process ever stores is
            // quantized with the same scales the read kernels use — a
            // mid-sequence scale switch is structurally impossible (the
            // old 256-token warmup corrupted the first short-prompt
            // request exactly that way; see module docs).
            inner.k_scale = (inner.k_running_max * HEADROOM / FP8_E4M3_MAX).max(MIN_SCALE);
            inner.v_scale = (inner.v_running_max * HEADROOM / FP8_E4M3_MAX).max(MIN_SCALE);
            inner.frozen = true;
            tracing::debug!(
                "FP8 KV scales frozen at first batch ({} tokens): k_scale={:.6} \
                 (max={:.2}), v_scale={:.6} (max={:.2})",
                inner.tokens_seen,
                inner.k_scale,
                inner.k_running_max,
                inner.v_scale,
                inner.v_running_max,
            );
        } else if inner.frozen
            && inner.tokens_seen % 128 < num_tokens as usize
            && std::env::var("ATLAS_FP8_KV_EMA_RECAL")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
        {
            // F5 (2026-05-26): Periodic EMA recalibration is now OPT-IN
            // via `ATLAS_FP8_KV_EMA_RECAL=1`, default OFF. Rationale:
            // changing `k_scale` / `v_scale` after `frozen=true` makes
            // previously-written KV cache entries (quantized with the
            // OLD scales) stale relative to the NEW scales — every
            // attention read after a recalibration sees the historical
            // cache through a shifted quantization basis. The comment
            // "responds faster to multi-turn topic switches" was correct
            // about the calibration signal but missed that retroactively
            // rescaling the cache corrupts already-stored multi-turn
            // context. The forensic study of the canonical opencode
            // probe shows reasoning-channel collapse + drift-to-phantom-
            // path patterns whose timing is consistent with deep-layer
            // KV reading through a rescaled basis.
            let new_k = (k_max / FP8_E4M3_MAX).max(MIN_SCALE);
            let new_v = (v_max / FP8_E4M3_MAX).max(MIN_SCALE);
            let k_shift = (new_k - inner.k_scale).abs() / inner.k_scale.max(MIN_SCALE);
            let v_shift = (new_v - inner.v_scale).abs() / inner.v_scale.max(MIN_SCALE);
            let alpha = if k_shift > 0.2 || v_shift > 0.2 {
                0.3
            } else {
                0.1
            };
            inner.k_scale = (1.0 - alpha) * inner.k_scale + alpha * new_k;
            inner.v_scale = (1.0 - alpha) * inner.v_scale + alpha * new_v;
            // Reset running max for next observation window
            inner.k_running_max = k_max;
            inner.v_running_max = v_max;

            tracing::info!(
                "FP8 KV calibrated after {} tokens: k_scale={:.6} (max={:.2}), v_scale={:.6} (max={:.2})",
                inner.tokens_seen,
                inner.k_scale,
                inner.k_running_max,
                inner.v_scale,
                inner.v_running_max,
            );
        }

        Ok(())
    }
}

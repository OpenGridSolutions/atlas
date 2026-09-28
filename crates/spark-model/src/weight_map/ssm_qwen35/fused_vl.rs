// SPDX-License-Identifier: AGPL-3.0-only

//! Fused-stacked expert loading for Qwen3-VL-MoE FP8 checkpoints
//! (Plan.md Phase Q), split out of `ssm_qwen35.rs` for the ≤500 LoC cap.
//!
//! Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 serializes ALL experts of a layer
//! as one tensor per projection in the transformers fused format, stored
//! **[in, out]** (transposed relative to the per-expert convention):
//!
//!   experts.gate_up_proj            [E, H, 2I]   FP8-E4M3
//!   experts.gate_up_proj_scale_inv  [E, H/128, 2I/128]  F32
//!   experts.down_proj               [E, I, H]    FP8-E4M3
//!   experts.down_proj_scale_inv     [E, I/128, H/128]   F32
//!
//! (Columns of gate_up: gate = 0..I, up = I..2I — transformers
//! `chunk(2, dim=-1)`.) This differs from the AgentWorld/Qwen3.6 fused
//! layout ([E, 2I, H], out-first) that `load_moe_qwen35`'s fused branch
//! handles — orientation is detected from the shapes (H != 2I here).
//!
//! Each expert projection is dequanted + column-sliced + TRANSPOSED to
//! the [out, in] convention in one kernel pass
//! (`dequant_fp8_blockscaled_bf16_slice_t`), runtime-quantized to NVFP4,
//! and the BF16 transient freed — the MiniMax M2 memory pattern. The
//! FP8 source stays in the (arena-backed) store untouched.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::super::{DenseWeight, ExpertWeight, QuantizeCtx, quantize_to_nvfp4};

/// Dequant one column range `[out_start, out_start+out_count)` of a fused
/// `[in_dim, out_total]` FP8 expert slice into a fresh BF16 buffer of
/// shape `[out_count, in_dim]` (transposed to the [out, in] convention).
#[allow(clippy::too_many_arguments)]
fn dequant_slice_transposed(
    gpu: &dyn GpuBackend,
    weight_ptr: DevicePtr,
    scale_ptr: DevicePtr,
    in_dim: usize,
    out_total: usize,
    out_start: usize,
    out_count: usize,
    sn: usize,
    sk: usize,
    scale_is_f32: bool,
) -> Result<DevicePtr> {
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
    let out = gpu.alloc(out_count * in_dim * 2)?; // BF16
    let stream = gpu.default_stream();
    let kernel = gpu.kernel(
        "dequant_fp8_blockscaled_bf16",
        "dequant_fp8_blockscaled_bf16_slice_t",
    )?;
    KernelLaunch::new(gpu, kernel)
        .grid([
            div_ceil(in_dim as u32, 64),
            div_ceil(out_count as u32, 4),
            1,
        ])
        .block([64, 4, 1])
        .arg_ptr(weight_ptr)
        .arg_ptr(scale_ptr)
        .arg_ptr(out)
        .arg_u32(in_dim as u32)
        .arg_u32(out_total as u32)
        .arg_u32(out_start as u32)
        .arg_u32(out_count as u32)
        .arg_u32((in_dim / sn) as u32)
        .arg_u32((out_total / sk) as u32)
        .arg_u32(sk as u32)
        .arg_u32(scale_is_f32 as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    Ok(out)
}

/// Load every routed expert of one layer from the transposed fused-FP8
/// layout. `p` is the layer's `.mlp` prefix.
pub(super) fn load_experts_fused_fp8_transposed(
    store: &WeightStore,
    p: &str,
    num_experts: usize,
    inter: usize,
    h: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
    qctx: QuantizeCtx,
) -> Result<Vec<ExpertWeight>> {
    let gu_key = format!("{p}.experts.gate_up_proj");
    let d_key = format!("{p}.experts.down_proj");
    let gu = store.get(&gu_key)?;
    let d = store.get(&d_key)?;
    ensure!(
        gu.dtype == WeightDtype::FP8E4M3 && d.dtype == WeightDtype::FP8E4M3,
        "{gu_key}: expected FP8E4M3 fused experts, got {:?}/{:?}",
        gu.dtype,
        d.dtype,
    );
    // Orientation gate (Plan.md Q.2 risk): this path is the [in, out]
    // (transposed) layout ONLY. Shapes make it unambiguous when H != 2I.
    ensure!(
        gu.shape == vec![num_experts, h, 2 * inter],
        "{gu_key}: expected transposed fused shape [{num_experts}, {h}, {}], got {:?} \
         (the [E, 2*inter, h] out-first layout is handled by load_moe_qwen35)",
        2 * inter,
        gu.shape,
    );
    ensure!(
        d.shape == vec![num_experts, inter, h],
        "{d_key}: expected [{num_experts}, {inter}, {h}], got {:?}",
        d.shape,
    );
    let gu_s = store.get(&format!("{gu_key}_scale_inv"))?;
    let d_s = store.get(&format!("{d_key}_scale_inv"))?;
    ensure!(
        gu_s.shape.len() == 3 && d_s.shape.len() == 3,
        "fused expert scales must be 3-D, got {:?} / {:?}",
        gu_s.shape,
        d_s.shape,
    );
    let (gu_sn, gu_sk) = (gu_s.shape[1], gu_s.shape[2]);
    let (d_sn, d_sk) = (d_s.shape[1], d_s.shape[2]);
    // 128×128 block grid must divide the dims exactly (2048/128, 1536/128,
    // 768/128 all exact for this family) — a ceil-grid on non-multiples
    // would silently misalign scale lookups in the slice kernel.
    ensure!(
        gu_sn > 0 && gu_sk > 0 && h.is_multiple_of(gu_sn) && (2 * inter).is_multiple_of(gu_sk),
        "{gu_key}_scale_inv grid {:?} does not evenly divide [{h}, {}]",
        gu_s.shape,
        2 * inter,
    );
    ensure!(
        d_sn > 0 && d_sk > 0 && inter.is_multiple_of(d_sn) && h.is_multiple_of(d_sk),
        "{d_key}_scale_inv grid {:?} does not evenly divide [{inter}, {h}]",
        d_s.shape,
    );
    let gu_s_f32 = gu_s.dtype == WeightDtype::FP32;
    let d_s_f32 = d_s.dtype == WeightDtype::FP32;
    let gu_s_elem = if gu_s_f32 { 4 } else { 2 };
    let d_s_elem = if d_s_f32 { 4 } else { 2 };
    let gu_w_stride = h * 2 * inter; // FP8 = 1 byte/element
    let d_w_stride = inter * h;
    let gu_s_stride = gu_sn * gu_sk * gu_s_elem;
    let d_s_stride = d_sn * d_sk * d_s_elem;

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        if !config.is_local_expert(e) {
            experts.push(ExpertWeight::null());
            continue;
        }
        let gu_ptr = gu.ptr.offset(e * gu_w_stride);
        let gu_s_ptr = gu_s.ptr.offset(e * gu_s_stride);
        // gate = columns 0..inter, up = columns inter..2*inter
        // (transformers chunk(2, dim=-1)); each lands as [inter, h] BF16.
        let gate_bf16 = dequant_slice_transposed(
            gpu, gu_ptr, gu_s_ptr, h, 2 * inter, 0, inter, gu_sn, gu_sk, gu_s_f32,
        )?;
        let up_bf16 = dequant_slice_transposed(
            gpu, gu_ptr, gu_s_ptr, h, 2 * inter, inter, inter, gu_sn, gu_sk, gu_s_f32,
        )?;
        // down: [inter, h] source → [h, inter] BF16.
        let down_bf16 = dequant_slice_transposed(
            gpu,
            d.ptr.offset(e * d_w_stride),
            d_s.ptr.offset(e * d_s_stride),
            inter,
            h,
            0,
            h,
            d_sn,
            d_sk,
            d_s_f32,
        )?;
        let out = ExpertWeight {
            gate_proj: quantize_to_nvfp4(
                &DenseWeight { weight: gate_bf16 },
                inter,
                h,
                gpu,
                qctx.absmax_k,
                qctx.quantize_k,
                qctx.stream,
            )?,
            up_proj: quantize_to_nvfp4(
                &DenseWeight { weight: up_bf16 },
                inter,
                h,
                gpu,
                qctx.absmax_k,
                qctx.quantize_k,
                qctx.stream,
            )?,
            down_proj: quantize_to_nvfp4(
                &DenseWeight { weight: down_bf16 },
                h,
                inter,
                gpu,
                qctx.absmax_k,
                qctx.quantize_k,
                qctx.stream,
            )?,
        };
        // Free the BF16 transients — only the NVFP4 copies live on. The
        // FP8 source is arena/store-owned and stays.
        gpu.free(gate_bf16)?;
        gpu.free(up_bf16)?;
        gpu.free(down_bf16)?;
        experts.push(out);
    }
    tracing::info!(
        "{p}: {} fused-FP8 experts (transposed [E, {h}, {}] layout) dequanted → NVFP4",
        num_experts,
        2 * inter,
    );
    Ok(experts)
}

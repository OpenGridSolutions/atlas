// SPDX-License-Identifier: AGPL-3.0-only

//! Strict tensor access for the `qwen4_exp` loader: every tensor is checked
//! against the dtype and shape the model code assumes; a deviation aborts
//! the load instead of mis-reading memory later.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layers::qwen4_exp::weights::{Mixer, Mxfp8Linear};

/// Device pointer of `name`, which must have exactly `dtype` and `shape`.
pub(super) fn tensor(
    store: &WeightStore,
    name: &str,
    dtype: WeightDtype,
    shape: &[usize],
) -> Result<DevicePtr> {
    let t = store.get(name)?;
    if t.dtype != dtype || t.shape != shape {
        bail!(
            "{name}: expected {dtype:?} {shape:?}, checkpoint has {:?} {:?}",
            t.dtype,
            t.shape
        );
    }
    Ok(t.ptr)
}

/// Device pointer of a `[rows, cols]` table with at least `min_rows` rows.
/// The token embedding and the LM head have one row per id of the model's
/// vocabulary, which may be larger than the tokenizer's (`config.vocab_size`
/// follows the tokenizer).
pub(super) fn table(
    store: &WeightStore,
    name: &str,
    dtype: WeightDtype,
    min_rows: usize,
    cols: usize,
) -> Result<DevicePtr> {
    let t = store.get(name)?;
    if t.dtype != dtype || t.shape.len() != 2 || t.shape[0] < min_rows || t.shape[1] != cols {
        bail!(
            "{name}: expected {dtype:?} [>= {min_rows}, {cols}], checkpoint has {:?} {:?}",
            t.dtype,
            t.shape
        );
    }
    Ok(t.ptr)
}

/// MXFP8 linear `[n, k]`: E4M3 weight plus one scale byte per 32 columns.
pub(super) fn mxfp8(store: &WeightStore, prefix: &str, n: usize, k: usize) -> Result<Mxfp8Linear> {
    if !k.is_multiple_of(32) {
        bail!("{prefix}: MXFP8 needs a multiple of 32 columns, got {k}");
    }
    Ok(Mxfp8Linear {
        weight: tensor(
            store,
            &format!("{prefix}.weight"),
            WeightDtype::FP8E4M3,
            &[n, k],
        )?,
        scale: tensor(
            store,
            &format!("{prefix}.weight_scale"),
            WeightDtype::UInt8,
            &[n, k / 32],
        )?,
        n: n as u32,
        k: k as u32,
    })
}

/// Stream mixer under `prefix`; `inject` is absent in the final mixer.
pub(super) fn mixer(
    store: &WeightStore,
    prefix: &str,
    hidden: usize,
    hc: usize,
    rank: usize,
    inject: bool,
) -> Result<Mixer> {
    let wide = hc * hidden;
    Ok(Mixer {
        norm_w: tensor(
            store,
            &format!("{prefix}.hc_norm.weight"),
            WeightDtype::FP32,
            &[wide],
        )?,
        down: tensor(
            store,
            &format!("{prefix}.input_mix_weight_down.weight"),
            WeightDtype::BF16,
            &[rank, wide],
        )?,
        up: tensor(
            store,
            &format!("{prefix}.input_mix_weight_up.weight"),
            WeightDtype::BF16,
            &[wide, rank],
        )?,
        inject: if inject {
            Some(tensor(
                store,
                &format!("{prefix}.block_inject_weight.weight"),
                WeightDtype::BF16,
                &[hc, wide],
            )?)
        } else {
            None
        },
    })
}

/// Host copy of a small device tensor.
pub(super) fn host_bytes(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    name: &str,
    dtype: WeightDtype,
    shape: &[usize],
) -> Result<Vec<u8>> {
    let ptr = tensor(store, name, dtype, shape)?;
    let mut out = vec![0u8; shape.iter().product::<usize>() * dtype.byte_size()];
    gpu.copy_d2h(ptr, &mut out)?;
    Ok(out)
}

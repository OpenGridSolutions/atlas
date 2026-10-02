// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen-Image-2.1 VAE encoder (`QwenImage21Encoder3d`, single frame) — the
//! condition-image half of `/v1/images/edits`.
//!
//! Mirrors the decoder's NHWC formulation: conv = im2col + GEMM, channel
//! RMS norm per row. The encoder-specific pieces are the stride-2
//! downsampling convolution (ZeroPad2d((0,1,0,1)) + 3×3 stride 2 →
//! `qimg_im2col3x3_s2`) and the parameter-free AvgDown3D shortcut
//! (`qimg_vae_avg_down`). For a single frame the temporal machinery
//! degenerates: the `time_conv`s never run (first-chunk cache path) and
//! the temporal blocks' AvgDown zero-pads the time axis IN FRONT, which
//! zeroes the even output channels of their shortcuts.
//!
//! `encode` returns argmax/mode latents — the first 64 of the 128
//! `quant_conv` channels — already (x − latents_mean) / latents_std
//! normalized, packed row-major `[lat_h*lat_w, 64]`.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::ops::{Ops, Pool, WeightSet};
use super::st::{StDir, bf16_bytes_to_f32s, f32s_to_bf16_bytes};
use super::vae::{Conv, Loader, Res, attn_nhwc, conv_nhwc, res_nhwc};

struct Down {
    res: Vec<Res>,
    /// Stride-2 downsample conv (None on the final, non-downsampling block).
    downsample: Option<Conv>,
    cin: usize,
    cout: usize,
    /// AvgDown3D temporal factor (2 on the blocks that downsample time).
    ft: usize,
}

pub(crate) struct VaeEncoder {
    conv_in: Conv,
    downs: Vec<Down>,
    mid_res0: Res,
    attn_norm: DevicePtr,
    attn_qkv: Conv,
    attn_proj: Conv,
    mid_res1: Res,
    norm_out: DevicePtr,
    conv_out: Conv,
    quant_conv: Conv,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
    pub weights_bytes: usize,
}

impl VaeEncoder {
    pub fn load(dir: &Path, ws: &mut WeightSet<'_>) -> Result<Self> {
        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)
                .context("vae/config.json")?;
        let floats = |k: &str| -> Vec<f32> {
            cfg[k]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_f64().map(|x| x as f32))
                        .collect()
                })
                .unwrap_or_default()
        };
        let (latents_mean, latents_std) = (floats("latents_mean"), floats("latents_std"));
        ensure!(
            latents_mean.len() == 64 && latents_std.len() == 64,
            "vae/config.json: expected 64 latents_mean/latents_std entries"
        );
        let num_res = cfg["num_res_blocks"].as_u64().unwrap_or(2) as usize;
        let temporal: Vec<bool> = cfg["temperal_downsample"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_bool().unwrap_or(false)).collect())
            .unwrap_or_default();

        let before = ws.bytes;
        let mut l = Loader {
            st: StDir::open(dir)?,
            ws,
        };
        let conv_in = l.conv("encoder.conv_in")?;
        let mut downs = Vec::new();
        let mut i = 0;
        while l
            .st
            .contains(&format!("encoder.down_blocks.{i}.resnets.0.conv1.weight"))
        {
            let base = format!("encoder.down_blocks.{i}");
            let mut res = Vec::new();
            for j in 0..num_res {
                res.push(l.res(&format!("{base}.resnets.{j}"))?);
            }
            let downsample = if l
                .st
                .contains(&format!("{base}.downsampler.resample.1.weight"))
            {
                Some(l.conv(&format!("{base}.downsampler.resample.1"))?)
            } else {
                None
            };
            let (cin, cout) = (res[0].conv1.cin, res[0].conv1.cout);
            let ft = if downsample.is_some() && temporal.get(i).copied().unwrap_or(false) {
                2
            } else {
                1
            };
            downs.push(Down {
                res,
                downsample,
                cin,
                cout,
                ft,
            });
            i += 1;
        }
        ensure!(!downs.is_empty(), "VAE encoder has no down blocks");
        let mid_res0 = l.res("encoder.mid_block.resnets.0")?;
        let attn_norm = l.gamma("encoder.mid_block.attentions.0.norm.gamma")?;
        let attn_qkv = l.conv("encoder.mid_block.attentions.0.to_qkv")?;
        let attn_proj = l.conv("encoder.mid_block.attentions.0.proj")?;
        let mid_res1 = l.res("encoder.mid_block.resnets.1")?;
        let norm_out = l.gamma("encoder.norm_out.gamma")?;
        let conv_out = l.conv("encoder.conv_out")?;
        let quant_conv = l.conv("quant_conv")?;
        Ok(Self {
            conv_in,
            downs,
            mid_res0,
            attn_norm,
            attn_qkv,
            attn_proj,
            mid_res1,
            norm_out,
            conv_out,
            quant_conv,
            latents_mean,
            latents_std,
            weights_bytes: l.ws.bytes - before,
        })
    }

    /// Stride-2 downsampling convolution; returns a `[h/2 * w/2, cout]`
    /// buffer (h, w are the input dims).
    fn down_conv(
        &self,
        g: &dyn GpuBackend,
        ops: &Ops,
        pool: &mut Pool<'_>,
        x: DevicePtr,
        h: usize,
        w: usize,
        c: &Conv,
    ) -> Result<DevicePtr> {
        let (ho, wo) = (h / 2, w / 2);
        let k = 9 * c.cin;
        let out = pool.alloc(ho * wo * c.cout * 2)?;
        let band = (super::vae::IM2COL_BUDGET / (wo * k * 2)).clamp(1, ho);
        let cols = pool.alloc(band * wo * k * 2)?;
        let mut row0 = 0;
        while row0 < ho {
            let rows = band.min(ho - row0);
            ops.im2col3x3_s2(g, x, cols, h, w, c.cin, row0, rows)?;
            ops.gemm(
                g,
                cols,
                c.w,
                out.offset(row0 * wo * c.cout * 2),
                rows * wo,
                c.cout,
                k,
            )?;
            row0 += rows;
        }
        g.synchronize(g.default_stream())?;
        pool.free(cols);
        ops.bias_add(g, out, c.bias, ho * wo, c.cout)?;
        Ok(out)
    }

    /// Encode a preprocessed image `[h*w, 4]` f32 in [-1, 1] (RGBA, NHWC,
    /// row-major) to normalized mode latents `[lat_h*lat_w, 64]`.
    pub fn encode(
        &self,
        g: &dyn GpuBackend,
        ops: &Ops,
        pixels: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>> {
        ensure!(
            pixels.len() == height * width * self.conv_in.cin,
            "encoder input shape mismatch"
        );
        ensure!(
            height.is_multiple_of(16) && width.is_multiple_of(16),
            "encoder input must be 16-aligned"
        );
        let mut pool = Pool::new(g);
        let pool = &mut pool;
        let (mut h, mut w) = (height, width);

        let img = pool.upload(&f32s_to_bf16_bytes(pixels))?;
        let mut x = conv_nhwc(g, ops, pool, img, h, w, &self.conv_in)?;
        pool.free(img);

        for d in &self.downs {
            let block_in = x;
            let mut cur = block_in;
            for r in &d.res {
                let n = res_nhwc(g, ops, pool, cur, h, w, r)?;
                g.synchronize(g.default_stream())?;
                if cur != block_in {
                    pool.free(cur);
                }
                cur = n;
            }
            match &d.downsample {
                Some(conv) => {
                    let n = self.down_conv(g, ops, pool, cur, h, w, conv)?;
                    g.synchronize(g.default_stream())?;
                    if cur != block_in {
                        pool.free(cur);
                    }
                    // AvgDown3D shortcut added onto the downsampled output.
                    let factor = d.ft * 4;
                    let group = d.cin * factor / d.cout;
                    ops.vae_avg_down(g, block_in, n, h, w, d.cin, d.cout, group, factor, d.ft)?;
                    g.synchronize(g.default_stream())?;
                    pool.free(block_in);
                    x = n;
                    h /= 2;
                    w /= 2;
                }
                None => {
                    // Identity AvgDown (factor 1): plain residual add.
                    ops.gated_add(g, cur, block_in, DevicePtr::NULL, h * w, d.cout)?;
                    g.synchronize(g.default_stream())?;
                    pool.free(block_in);
                    x = cur;
                }
            }
        }

        let n = res_nhwc(g, ops, pool, x, h, w, &self.mid_res0)?;
        g.synchronize(g.default_stream())?;
        pool.free(x);
        x = n;
        let n = attn_nhwc(
            g,
            ops,
            pool,
            x,
            h,
            w,
            self.attn_norm,
            &self.attn_qkv,
            &self.attn_proj,
        )?;
        pool.free(x);
        x = n;
        let n = res_nhwc(g, ops, pool, x, h, w, &self.mid_res1)?;
        g.synchronize(g.default_stream())?;
        pool.free(x);
        x = n;

        let c = self.conv_out.cin;
        ops.vae_norm(g, x, self.norm_out, x, h * w, c, true)?;
        let moments = conv_nhwc(g, ops, pool, x, h, w, &self.conv_out)?;
        pool.free(x);
        let enc = conv_nhwc(g, ops, pool, moments, h, w, &self.quant_conv)?;
        g.synchronize(g.default_stream())?;
        pool.free(moments);

        // mode() = mean = first 64 of the 128 channels, then normalize.
        let zc = self.latents_mean.len();
        let full = self.quant_conv.cout;
        let mut raw = vec![0u8; h * w * full * 2];
        g.copy_d2h(enc, &mut raw)?;
        let vals = bf16_bytes_to_f32s(&raw);
        let mut out = Vec::with_capacity(h * w * zc);
        for p in 0..h * w {
            for ch in 0..zc {
                let v = vals[p * full + ch];
                out.push((v - self.latents_mean[ch]) / self.latents_std[ch]);
            }
        }
        Ok(out)
    }
}

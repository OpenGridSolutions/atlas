// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen-Image-2.1 VAE decoder (`AutoencoderKLQwenImage21`, single frame).
//!
//! The checkpoint is the image specialization of Wan's causal 3-D VAE: every
//! "causal conv" is a plain 2-D convolution and the temporal branches are
//! no-ops for one frame. Activations are kept NHWC (`[pixels, C]`, BF16) so
//! a convolution is im2col + the common projection GEMM and the channel
//! RMS norm is a per-row kernel. The final convolution produces FP32 so the
//! 8-bit pixels are not limited by BF16's 8-bit mantissa.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::dit::Trace;
use super::ops::{Ops, Pool, WeightSet, pad_keys};
use super::st::{StDir, bf16_bytes_to_f32s, bytes_to_f32s, f32s_to_bf16_bytes, f32s_to_bytes};

/// Upper bound for one im2col band (BF16 bytes).
const IM2COL_BUDGET: usize = 768 << 20;

struct Conv {
    /// `[cout, 9*cin]` (3×3, columns ordered ky, kx, c) or `[cout, cin]` (1×1), BF16.
    w: DevicePtr,
    /// `[cout]` FP32 on device.
    bias: DevicePtr,
    /// Host copy of the bias (the FP32 output convolution adds it on the host).
    bias_host: Vec<f32>,
    cin: usize,
    cout: usize,
    k3: bool,
}

struct Res {
    norm1: DevicePtr,
    conv1: Conv,
    norm2: DevicePtr,
    conv2: Conv,
    shortcut: Option<Conv>,
}

struct Up {
    res: Vec<Res>,
    upsample: Option<Conv>,
    cin: usize,
    cout: usize,
    /// DupUp3D shortcut parameters (factor, offset, repeats).
    dup: Option<(usize, usize, usize)>,
}

pub(crate) struct VaeDecoder {
    post_quant: Conv,
    conv_in: Conv,
    mid_res0: Res,
    attn_norm: DevicePtr,
    attn_qkv: Conv,
    attn_proj: Conv,
    mid_res1: Res,
    ups: Vec<Up>,
    norm_out: DevicePtr,
    conv_out: Conv,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
    pub out_channels: usize,
    pub weights_bytes: usize,
}

struct Loader<'a, 'g> {
    st: StDir,
    ws: &'a mut WeightSet<'g>,
}

impl Loader<'_, '_> {
    fn upload(&mut self, raw: &[u8]) -> Result<DevicePtr> {
        self.ws.upload(raw)
    }

    fn conv(&mut self, base: &str) -> Result<Conv> {
        let e = self.st.entry(&format!("{base}.weight"))?.clone();
        ensure!(
            e.shape.len() == 4,
            "{base}.weight: expected a 4-D conv weight"
        );
        let (cout, cin, kh, kw) = (e.shape[0], e.shape[1], e.shape[2], e.shape[3]);
        ensure!(
            (kh, kw) == (3, 3) || (kh, kw) == (1, 1),
            "{base}: unsupported kernel {kh}x{kw}"
        );
        let w = self.st.read_f32(&format!("{base}.weight"))?;
        let taps = kh * kw;
        // [cout, cin, ky, kx] → [cout, (ky, kx), cin]
        let mut m = vec![0f32; w.len()];
        for o in 0..cout {
            for c in 0..cin {
                for t in 0..taps {
                    m[(o * taps + t) * cin + c] = w[(o * cin + c) * taps + t];
                }
            }
        }
        let bias_host = self.st.read_f32(&format!("{base}.bias"))?;
        Ok(Conv {
            w: self.upload(&f32s_to_bf16_bytes(&m))?,
            bias: self.upload(&f32s_to_bytes(&bias_host))?,
            bias_host,
            cin,
            cout,
            k3: taps == 9,
        })
    }

    fn gamma(&mut self, name: &str) -> Result<DevicePtr> {
        let v = self.st.read_f32(name)?;
        self.upload(&f32s_to_bytes(&v))
    }

    fn res(&mut self, base: &str) -> Result<Res> {
        let shortcut = if self.st.contains(&format!("{base}.conv_shortcut.weight")) {
            Some(self.conv(&format!("{base}.conv_shortcut"))?)
        } else {
            None
        };
        Ok(Res {
            norm1: self.gamma(&format!("{base}.norm1.gamma"))?,
            conv1: self.conv(&format!("{base}.conv1"))?,
            norm2: self.gamma(&format!("{base}.norm2.gamma"))?,
            conv2: self.conv(&format!("{base}.conv2"))?,
            shortcut,
        })
    }
}

impl VaeDecoder {
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
        // Decoder temporal upsampling is the reversed encoder downsampling.
        let temporal: Vec<bool> = cfg["temperal_downsample"]
            .as_array()
            .map(|a| {
                a.iter()
                    .rev()
                    .map(|v| v.as_bool().unwrap_or(false))
                    .collect()
            })
            .unwrap_or_default();

        let before = ws.bytes;
        let mut l = Loader {
            st: StDir::open(dir)?,
            ws,
        };
        let post_quant = l.conv("post_quant_conv")?;
        let conv_in = l.conv("decoder.conv_in")?;
        let mid_res0 = l.res("decoder.mid_block.resnets.0")?;
        let attn_norm = l.gamma("decoder.mid_block.attentions.0.norm.gamma")?;
        let attn_qkv = l.conv("decoder.mid_block.attentions.0.to_qkv")?;
        let attn_proj = l.conv("decoder.mid_block.attentions.0.proj")?;
        let mid_res1 = l.res("decoder.mid_block.resnets.1")?;

        let mut ups = Vec::new();
        let mut i = 0;
        while l
            .st
            .contains(&format!("decoder.up_blocks.{i}.resnets.0.conv1.weight"))
        {
            let base = format!("decoder.up_blocks.{i}");
            let mut res = Vec::new();
            for j in 0..=num_res {
                res.push(l.res(&format!("{base}.resnets.{j}"))?);
            }
            let (cin, cout) = (res[0].conv1.cin, res[0].conv1.cout);
            let upsample = if l
                .st
                .contains(&format!("{base}.upsampler.resample.1.weight"))
            {
                Some(l.conv(&format!("{base}.upsampler.resample.1"))?)
            } else {
                None
            };
            let dup = upsample.as_ref().map(|_| {
                // Single frame of a temporal (factor_t = 2) block keeps the
                // second duplicated frame → channel offset factor_s².
                let factor_t = if temporal.get(i).copied().unwrap_or(false) {
                    2
                } else {
                    1
                };
                let factor = factor_t * 4;
                (factor, (factor_t - 1) * 4, cout * factor / cin)
            });
            ups.push(Up {
                res,
                upsample,
                cin,
                cout,
                dup,
            });
            i += 1;
        }
        ensure!(!ups.is_empty(), "VAE decoder has no up blocks");
        let norm_out = l.gamma("decoder.norm_out.gamma")?;
        let conv_out = l.conv("decoder.conv_out")?;
        Ok(Self {
            out_channels: conv_out.cout,
            post_quant,
            conv_in,
            mid_res0,
            attn_norm,
            attn_qkv,
            attn_proj,
            mid_res1,
            ups,
            norm_out,
            conv_out,
            latents_mean,
            latents_std,
            weights_bytes: l.ws.bytes - before,
        })
    }

    /// Convolution over an NHWC buffer; returns a new `[h*w, cout]` buffer.
    fn conv(
        &self,
        g: &dyn GpuBackend,
        ops: &Ops,
        pool: &mut Pool<'_>,
        x: DevicePtr,
        h: usize,
        w: usize,
        c: &Conv,
    ) -> Result<DevicePtr> {
        let pixels = h * w;
        let out = pool.alloc(pixels * c.cout * 2)?;
        if c.k3 {
            let k = 9 * c.cin;
            let band = (IM2COL_BUDGET / (w * k * 2)).clamp(1, h);
            let cols = pool.alloc(band * w * k * 2)?;
            let mut row0 = 0;
            while row0 < h {
                let rows = band.min(h - row0);
                ops.im2col3x3(g, x, cols, h, w, c.cin, row0, rows)?;
                ops.gemm(
                    g,
                    cols,
                    c.w,
                    out.offset(row0 * w * c.cout * 2),
                    rows * w,
                    c.cout,
                    k,
                )?;
                row0 += rows;
            }
            g.synchronize(g.default_stream())?;
            pool.free(cols);
        } else {
            ops.gemm(g, x, c.w, out, pixels, c.cout, c.cin)?;
        }
        ops.bias_add(g, out, c.bias, pixels, c.cout)?;
        Ok(out)
    }

    /// Residual block; consumes nothing, returns a new `[h*w, cout]` buffer.
    fn res(
        &self,
        g: &dyn GpuBackend,
        ops: &Ops,
        pool: &mut Pool<'_>,
        x: DevicePtr,
        h: usize,
        w: usize,
        r: &Res,
    ) -> Result<DevicePtr> {
        let pixels = h * w;
        let (cin, cout) = (r.conv1.cin, r.conv1.cout);
        let t = pool.alloc(pixels * cin * 2)?;
        ops.vae_norm(g, x, r.norm1, t, pixels, cin, true)?;
        let a = self.conv(g, ops, pool, t, h, w, &r.conv1)?;
        pool.free(t);
        ops.vae_norm(g, a, r.norm2, a, pixels, cout, true)?;
        let b = self.conv(g, ops, pool, a, h, w, &r.conv2)?;
        pool.free(a);
        match &r.shortcut {
            Some(sc) => {
                let s = self.conv(g, ops, pool, x, h, w, sc)?;
                ops.gated_add(g, b, s, DevicePtr::NULL, pixels, cout)?;
                g.synchronize(g.default_stream())?;
                pool.free(s);
            }
            None => ops.gated_add(g, b, x, DevicePtr::NULL, pixels, cout)?,
        }
        Ok(b)
    }

    /// Single-head self-attention over all pixels (mid block).
    fn attn(
        &self,
        g: &dyn GpuBackend,
        ops: &Ops,
        pool: &mut Pool<'_>,
        x: DevicePtr,
        h: usize,
        w: usize,
    ) -> Result<DevicePtr> {
        let pixels = h * w;
        let c = self.attn_proj.cin;
        let k_rows = pad_keys(pixels);
        let xn = pool.alloc(pixels * c * 2)?;
        ops.vae_norm(g, x, self.attn_norm, xn, pixels, c, false)?;
        let qkv = self.conv(g, ops, pool, xn, h, w, &self.attn_qkv)?;
        let q = xn; // reuse: the normalized input is consumed
        let k = pool.alloc_zeroed(k_rows * c * 2)?;
        let vt = pool.alloc_zeroed(c * k_rows * 2)?;
        ops.vae_qkv_split(g, qkv, q, k, vt, pixels, c, k_rows)?;
        let scores = pool.alloc(pixels * k_rows * 4)?;
        let probs = pool.alloc(pixels * k_rows * 2)?;
        ops.gemm_f32out(g, q, k, scores, pixels, k_rows, c)?;
        ops.softmax_rows(
            g,
            scores,
            probs,
            pixels,
            k_rows,
            pixels,
            1.0 / (c as f32).sqrt(),
        )?;
        ops.gemm(g, probs, vt, q, pixels, c, k_rows)?;
        let out = self.conv(g, ops, pool, q, h, w, &self.attn_proj)?;
        ops.gated_add(g, out, x, DevicePtr::NULL, pixels, c)?;
        g.synchronize(g.default_stream())?;
        for p in [qkv, q, k, vt, scores, probs] {
            pool.free(p);
        }
        Ok(out)
    }

    /// Decode normalized latents `[lat_h*lat_w, 64]` (row-major tokens) to
    /// `[16*lat_h * 16*lat_w, out_channels]` FP32 pixels clamped to [-1, 1].
    pub fn decode(
        &self,
        g: &dyn GpuBackend,
        ops: &Ops,
        latents: &[f32],
        lat_h: usize,
        lat_w: usize,
        mut trace: Option<&mut Trace>,
    ) -> Result<Vec<f32>> {
        let zc = self.latents_mean.len();
        ensure!(
            latents.len() == lat_h * lat_w * zc,
            "latents shape mismatch"
        );
        let z: Vec<f32> = latents
            .iter()
            .enumerate()
            .map(|(i, v)| v * self.latents_std[i % zc] + self.latents_mean[i % zc])
            .collect();
        let mut pool = Pool::new(g);
        let pool = &mut pool;
        let (mut h, mut w) = (lat_h, lat_w);
        let mut snap = |name: &str, p: DevicePtr, n: usize| -> Result<()> {
            if let Some(t) = trace.as_deref_mut() {
                g.synchronize(g.default_stream())?;
                let mut raw = vec![0u8; n * 2];
                g.copy_d2h(p, &mut raw)?;
                t.push((name.to_string(), bf16_bytes_to_f32s(&raw)));
            }
            Ok(())
        };

        let z = pool.upload(&f32s_to_bf16_bytes(&z))?;
        let mut x = self.conv(g, ops, pool, z, h, w, &self.post_quant)?;
        pool.free(z);
        snap("post_quant_conv", x, h * w * zc)?;
        let step = |pool: &mut Pool<'_>, x: &mut DevicePtr, next: DevicePtr| -> Result<()> {
            g.synchronize(g.default_stream())?;
            pool.free(*x);
            *x = next;
            Ok(())
        };

        let n = self.conv(g, ops, pool, x, h, w, &self.conv_in)?;
        step(pool, &mut x, n)?;
        let mut c = self.conv_in.cout;
        snap("conv_in", x, h * w * c)?;
        let n = self.res(g, ops, pool, x, h, w, &self.mid_res0)?;
        step(pool, &mut x, n)?;
        snap("mid.res0", x, h * w * c)?;
        let n = self.attn(g, ops, pool, x, h, w)?;
        step(pool, &mut x, n)?;
        snap("mid.attn0", x, h * w * c)?;
        let n = self.res(g, ops, pool, x, h, w, &self.mid_res1)?;
        step(pool, &mut x, n)?;
        snap("mid", x, h * w * c)?;

        for (i, up) in self.ups.iter().enumerate() {
            // The block input feeds the DupUp shortcut after the main path.
            let block_in = x;
            let mut cur = block_in;
            for r in &up.res {
                let n = self.res(g, ops, pool, cur, h, w, r)?;
                g.synchronize(g.default_stream())?;
                if cur != block_in {
                    pool.free(cur);
                }
                cur = n;
            }
            if let (Some(conv), Some((factor, off, repeats))) = (&up.upsample, up.dup) {
                let big = pool.alloc(4 * h * w * up.cout * 2)?;
                ops.upsample2x(g, cur, big, h, w, up.cout)?;
                g.synchronize(g.default_stream())?;
                pool.free(cur);
                let n = self.conv(g, ops, pool, big, 2 * h, 2 * w, conv)?;
                pool.free(big);
                ops.dup_shortcut_add(g, n, block_in, h, w, up.cin, up.cout, factor, off, repeats)?;
                cur = n;
                h *= 2;
                w *= 2;
            }
            g.synchronize(g.default_stream())?;
            pool.free(block_in);
            x = cur;
            c = up.cout;
            snap(&format!("up{i}"), x, h * w * c)?;
        }

        // head: norm + SiLU, then the FP32-output 3×3 convolution
        ops.vae_norm(g, x, self.norm_out, x, h * w, c, true)?;
        let co = &self.conv_out;
        let k = 9 * co.cin;
        let band = (IM2COL_BUDGET / (w * k * 2)).clamp(1, h);
        let cols = pool.alloc(band * w * k * 2)?;
        let out = pool.alloc(h * w * co.cout * 4)?;
        let mut row0 = 0;
        while row0 < h {
            let rows = band.min(h - row0);
            ops.im2col3x3(g, x, cols, h, w, co.cin, row0, rows)?;
            ops.gemm_f32out(
                g,
                cols,
                co.w,
                out.offset(row0 * w * co.cout * 4),
                rows * w,
                co.cout,
                k,
            )?;
            row0 += rows;
        }
        g.synchronize(g.default_stream())?;
        let mut raw = vec![0u8; h * w * co.cout * 4];
        g.copy_d2h(out, &mut raw)?;
        let mut img = bytes_to_f32s(&raw);
        for (i, v) in img.iter_mut().enumerate() {
            *v = (*v + co.bias_host[i % co.cout]).clamp(-1.0, 1.0);
        }
        Ok(img)
    }
}

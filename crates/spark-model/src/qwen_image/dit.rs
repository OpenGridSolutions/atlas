// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen-Image-2.1 single-stream DiT (`QwenImage21Transformer2DModel`).
//!
//! The joint sequence is `[prompt tokens, target latents]` with block-causal
//! attention: the prompt is causal, the target block attends to everything.
//! With `causal_condition` the prompt tokens are modulated from t = 0, so
//! their activations never depend on the denoising step. The reference
//! exploits that with an "extract" first step plus cached K/V afterwards;
//! here the prompt runs ONCE as its own prefix pass (it cannot see the
//! target, so that is exactly the extract step's prefix rows) and every
//! denoising step — the first included — only computes the target rows
//! against `[cached prefix K/V, target K/V]`.

use std::path::Path;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::ops::{AttnBufs, Ops, Pool, WeightSet, pad_keys};
use super::sched;
use super::st::{StDir, bytes_to_f32s, f32s_to_bf16_bytes, f32s_to_bytes};

pub(crate) const HIDDEN: usize = 4096;
pub(crate) const LATENT_CH: usize = 64;
const LAYERS: usize = 32;
const HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const FFN: usize = 12288;
const EPS: f32 = 1e-6;

struct Block {
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    o: DevicePtr,
    norm_q: DevicePtr,
    norm_k: DevicePtr,
    gate: DevicePtr,
    proj: DevicePtr,
    out: DevicePtr,
}

pub(crate) struct Dit {
    blocks: Vec<Block>,
    img_in: DevicePtr,
    txt_norm: DevicePtr,
    txt_in: DevicePtr,
    txt_out: DevicePtr,
    t_lin1: DevicePtr,
    t_lin2: DevicePtr,
    modulation: DevicePtr,
    norm_out_lin: DevicePtr,
    proj_out: DevicePtr,
    pub weights_bytes: usize,
}

/// Named intermediate activations captured for the golden harness.
pub type Trace = Vec<(String, Vec<f32>)>;

fn download_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut raw = vec![0u8; n * 2];
    g.copy_d2h(p, &mut raw)?;
    Ok(super::st::bf16_bytes_to_f32s(&raw))
}

fn download_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut raw = vec![0u8; n * 4];
    g.copy_d2h(p, &mut raw)?;
    Ok(bytes_to_f32s(&raw))
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// Per-pass modulation vectors, f32 on device, `HIDDEN` each.
struct ModVecs {
    scale1: DevicePtr,
    gate1: DevicePtr,
    scale2: DevicePtr,
    gate2: DevicePtr,
    scale_out: DevicePtr,
}

impl Dit {
    pub fn load(dir: &Path, ws: &mut WeightSet<'_>) -> Result<Self> {
        let st = StDir::open(dir)?;
        let before = ws.bytes;
        let mut up = |name: String| -> Result<DevicePtr> { ws.upload(&st.read_bf16(&name)?) };
        let mut blocks = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("transformer_blocks.{i}");
            blocks.push(Block {
                q: up(format!("{p}.attn.to_q.weight"))?,
                k: up(format!("{p}.attn.to_k.weight"))?,
                v: up(format!("{p}.attn.to_v.weight"))?,
                o: up(format!("{p}.attn.to_out.0.weight"))?,
                norm_q: up(format!("{p}.attn.norm_q.weight"))?,
                norm_k: up(format!("{p}.attn.norm_k.weight"))?,
                gate: up(format!("{p}.img_mlp.gate_layer.weight"))?,
                proj: up(format!("{p}.img_mlp.proj.weight"))?,
                out: up(format!("{p}.img_mlp.out.weight"))?,
            });
        }
        let dit = Self {
            blocks,
            img_in: up("img_in.weight".into())?,
            txt_norm: up("txt_in.text_norm.weight".into())?,
            txt_in: up("txt_in.in_layer.weight".into())?,
            txt_out: up("txt_in.out_layer.weight".into())?,
            t_lin1: up("time_text_embed.timestep_embedder.linear_1.weight".into())?,
            t_lin2: up("time_text_embed.timestep_embedder.linear_2.weight".into())?,
            modulation: up("modulation.1.weight".into())?,
            norm_out_lin: up("norm_out.linear.weight".into())?,
            proj_out: up("proj_out.weight".into())?,
            weights_bytes: 0,
        };
        Ok(Self {
            weights_bytes: ws.bytes - before,
            ..dit
        })
    }

    /// Timestep embedding → the shared block modulation and the final
    /// AdaLN scale for one timestep. Gates are stored already `tanh`-ed.
    fn mod_vectors(
        &self,
        g: &dyn GpuBackend,
        ops: &Ops,
        pool: &mut Pool<'_>,
        sigma: f32,
    ) -> Result<ModVecs> {
        let row = pool.alloc(HIDDEN * 2)?;
        let out = pool.alloc(4 * HIDDEN * 4)?;
        // One-row GEMM with an f32 result: y[n] = W[n, k] · x[k].
        let matvec = |input: &[f32], w: DevicePtr, n: usize| -> Result<Vec<f32>> {
            g.copy_h2d(&f32s_to_bf16_bytes(input), row)?;
            ops.gemm_f32out(g, row, w, out, 1, n, input.len())?;
            g.synchronize(g.default_stream())?;
            download_f32(g, out, n)
        };
        let proj = sched::timestep_proj(sigma);
        let h1: Vec<f32> = matvec(&proj, self.t_lin1, HIDDEN)?
            .into_iter()
            .map(silu)
            .collect();
        let temb = matvec(&h1, self.t_lin2, HIDDEN)?;
        let act: Vec<f32> = temb.iter().copied().map(silu).collect();
        let mut m = matvec(&act, self.modulation, 4 * HIDDEN)?;
        let scale_out = matvec(&act, self.norm_out_lin, HIDDEN)?;
        pool.free(row);
        pool.free(out);
        // [mod1.scale, mod1.gate, mod2.scale, mod2.gate]
        for chunk in [1usize, 3] {
            for v in &mut m[chunk * HIDDEN..(chunk + 1) * HIDDEN] {
                *v = v.tanh();
            }
        }
        let dev = pool.upload(&f32s_to_bytes(&m))?;
        let stride = HIDDEN * 4;
        Ok(ModVecs {
            scale1: dev,
            gate1: dev.offset(stride),
            scale2: dev.offset(2 * stride),
            gate2: dev.offset(3 * stride),
            scale_out: pool.upload(&f32s_to_bytes(&scale_out))?,
        })
    }

    /// Run the flow-matching sampler over `latents` (`[lat_h*lat_w, 64]`
    /// f32, target tokens in row-major order) in place.
    ///
    /// `prompt` is the text encoder output `[tp, 4096]` BF16 on device;
    /// `sigmas` has `steps + 1` entries.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        g: &dyn GpuBackend,
        ops: &Ops,
        prompt: DevicePtr,
        tp: usize,
        lat_h: usize,
        lat_w: usize,
        latents: &mut [f32],
        sigmas: &[f32],
        mut trace: Option<&mut Trace>,
        mut on_step: impl FnMut(usize),
    ) -> Result<()> {
        let n = lat_h * lat_w;
        anyhow::ensure!(latents.len() == n * LATENT_CH, "latents shape mismatch");
        let s = tp + n;
        let (tp_pad, s_pad) = (pad_keys(tp), pad_keys(s));
        let hd = HEADS * HEAD_DIM;
        let half = HEAD_DIM / 2;
        let mut pool = Pool::new(g);
        let pool = &mut pool;

        let positions = sched::t2i_positions(tp, lat_h, lat_w);
        let (cos, sin) = sched::rope_tables(&positions);
        let cos_p = pool.upload(&f32s_to_bytes(&cos[..tp * half]))?;
        let sin_p = pool.upload(&f32s_to_bytes(&sin[..tp * half]))?;
        let cos_t = pool.upload(&f32s_to_bytes(&cos[tp * half..]))?;
        let sin_t = pool.upload(&f32s_to_bytes(&sin[tp * half..]))?;

        // ── prefix pass: prompt tokens, t = 0 modulation, causal ──
        let mut kc = Vec::with_capacity(LAYERS);
        let mut vc = Vec::with_capacity(LAYERS);
        {
            let m0 = self.mod_vectors(g, ops, pool, 0.0)?;
            let limits: Vec<u8> = (0..tp as u32).flat_map(|r| (r + 1).to_le_bytes()).collect();
            let limits = pool.upload(&limits)?;
            let h = pool.alloc(tp * HIDDEN * 2)?;
            let x = pool.alloc(tp * HIDDEN * 2)?;
            let q = pool.alloc(tp * hd * 2)?;
            let kv = pool.alloc(tp * hd * 2)?;
            let ff_a = pool.alloc(tp * FFN * 2)?;
            let ff_b = pool.alloc(tp * FFN * 2)?;
            let qh = pool.alloc(tp * hd * 2)?;
            let scores = pool.alloc(tp * tp_pad * 4)?;
            let probs = pool.alloc(tp * tp_pad * 2)?;
            let o_stage = pool.alloc(tp * HEAD_DIM * 2)?;

            // txt_in: zero-centred RMSNorm → Linear → GELU(tanh) → Linear
            ops.rms_norm(g, prompt, self.txt_norm, x, tp, HIDDEN, EPS, 1.0)?;
            ops.gemm(g, x, self.txt_in, q, tp, HIDDEN, HIDDEN)?;
            ops.gelu_tanh(g, q, tp * HIDDEN)?;
            ops.gemm(g, q, self.txt_out, h, tp, HIDDEN, HIDDEN)?;
            if let Some(t) = trace.as_deref_mut() {
                g.synchronize(g.default_stream())?;
                t.push(("txt_in".into(), download_bf16(g, h, tp * HIDDEN)?));
            }

            for (li, b) in self.blocks.iter().enumerate() {
                let kc_l = pool.alloc_zeroed(tp_pad * hd * 2)?;
                let vc_l = pool.alloc_zeroed(tp_pad * hd * 2)?;
                ops.layernorm_mod(g, h, m0.scale1, x, tp, HIDDEN, EPS)?;
                ops.gemm(g, x, b.q, q, tp, hd, HIDDEN)?;
                ops.norm_rope_heads(
                    g, q, b.norm_q, cos_p, sin_p, qh, tp, HEADS, HEAD_DIM, 0, tp, EPS, 0,
                )?;
                ops.gemm(g, x, b.k, kv, tp, hd, HIDDEN)?;
                ops.norm_rope_heads(
                    g, kv, b.norm_k, cos_p, sin_p, kc_l, tp, HEADS, HEAD_DIM, 0, tp_pad, EPS, 0,
                )?;
                ops.gemm(g, x, b.v, kv, tp, hd, HIDDEN)?;
                ops.v_transpose(g, kv, vc_l, tp, HEADS, HEAD_DIM, 0, tp_pad)?;
                let bufs = AttnBufs {
                    qh,
                    kh: kc_l,
                    vt: vc_l,
                    scores,
                    probs,
                    o_stage,
                };
                ops.attention(g, &bufs, q, limits, tp, tp, tp_pad, HEADS, HEADS, HEAD_DIM)?;
                ops.gemm(g, q, b.o, x, tp, HIDDEN, hd)?;
                ops.gated_add(g, h, x, m0.gate1, tp, HIDDEN)?;

                ops.layernorm_mod(g, h, m0.scale2, x, tp, HIDDEN, EPS)?;
                ops.gemm(g, x, b.gate, ff_a, tp, FFN, HIDDEN)?;
                ops.gemm(g, x, b.proj, ff_b, tp, FFN, HIDDEN)?;
                ops.silu_mul(g, ff_a, ff_b, ff_a, tp * FFN)?;
                ops.gemm(g, ff_a, b.out, x, tp, HIDDEN, FFN)?;
                ops.gated_add(g, h, x, m0.gate2, tp, HIDDEN)?;
                kc.push(kc_l);
                vc.push(vc_l);
                if let Some(t) = trace.as_deref_mut()
                    && matches!(li, 0 | 1 | 15 | 31)
                {
                    g.synchronize(g.default_stream())?;
                    t.push((
                        format!("prefix.block{li}"),
                        download_bf16(g, h, tp * HIDDEN)?,
                    ));
                }
            }
            g.synchronize(g.default_stream())?;
            for p in [h, x, q, kv, ff_a, ff_b, qh, scores, probs, o_stage] {
                pool.free(p);
            }
        }

        // ── target pass buffers ──
        let lat = pool.alloc(n * LATENT_CH * 2)?;
        let h = pool.alloc(n * HIDDEN * 2)?;
        let x = pool.alloc(n * HIDDEN * 2)?;
        let q = pool.alloc(n * hd * 2)?;
        let kv = pool.alloc(n * hd * 2)?;
        let ff_a = pool.alloc(n * FFN * 2)?;
        let ff_b = pool.alloc(n * FFN * 2)?;
        let pred = pool.alloc(n * LATENT_CH * 4)?;
        let bufs = AttnBufs {
            qh: pool.alloc(n * hd * 2)?,
            kh: pool.alloc_zeroed(s_pad * hd * 2)?,
            vt: pool.alloc_zeroed(s_pad * hd * 2)?,
            scores: pool.alloc(n * s_pad * 4)?,
            probs: pool.alloc(n * s_pad * 2)?,
            o_stage: pool.alloc(n * HEAD_DIM * 2)?,
        };

        for step in 0..sigmas.len() - 1 {
            let sigma = sigmas[step];
            let m = self.mod_vectors(g, ops, pool, sigma)?;
            g.copy_h2d(&f32s_to_bf16_bytes(latents), lat)?;
            ops.gemm(g, lat, self.img_in, h, n, HIDDEN, LATENT_CH)?;

            for (li, b) in self.blocks.iter().enumerate() {
                ops.layernorm_mod(g, h, m.scale1, x, n, HIDDEN, EPS)?;
                ops.gemm(g, x, b.q, q, n, hd, HIDDEN)?;
                ops.norm_rope_heads(
                    g, q, b.norm_q, cos_t, sin_t, bufs.qh, n, HEADS, HEAD_DIM, 0, n, EPS, 0,
                )?;
                ops.copy_prefix(
                    g, kc[li], vc[li], bufs.kh, bufs.vt, tp, tp_pad, s_pad, HEADS, HEAD_DIM,
                )?;
                ops.gemm(g, x, b.k, kv, n, hd, HIDDEN)?;
                ops.norm_rope_heads(
                    g, kv, b.norm_k, cos_t, sin_t, bufs.kh, n, HEADS, HEAD_DIM, tp, s_pad, EPS, 0,
                )?;
                ops.gemm(g, x, b.v, kv, n, hd, HIDDEN)?;
                ops.v_transpose(g, kv, bufs.vt, n, HEADS, HEAD_DIM, tp, s_pad)?;
                ops.attention(
                    g,
                    &bufs,
                    q,
                    DevicePtr::NULL,
                    n,
                    s,
                    s_pad,
                    HEADS,
                    HEADS,
                    HEAD_DIM,
                )?;
                ops.gemm(g, q, b.o, x, n, HIDDEN, hd)?;
                ops.gated_add(g, h, x, m.gate1, n, HIDDEN)?;

                ops.layernorm_mod(g, h, m.scale2, x, n, HIDDEN, EPS)?;
                ops.gemm(g, x, b.gate, ff_a, n, FFN, HIDDEN)?;
                ops.gemm(g, x, b.proj, ff_b, n, FFN, HIDDEN)?;
                ops.silu_mul(g, ff_a, ff_b, ff_a, n * FFN)?;
                ops.gemm(g, ff_a, b.out, x, n, HIDDEN, FFN)?;
                ops.gated_add(g, h, x, m.gate2, n, HIDDEN)?;
                if let Some(t) = trace.as_deref_mut()
                    && step < 2
                    && matches!(li, 0 | 1 | 15 | 31)
                {
                    g.synchronize(g.default_stream())?;
                    t.push((
                        format!("step{step}.block{li}"),
                        download_bf16(g, h, n * HIDDEN)?,
                    ));
                }
            }

            ops.layernorm_mod(g, h, m.scale_out, x, n, HIDDEN, EPS)?;
            ops.gemm_f32out(g, x, self.proj_out, pred, n, LATENT_CH, HIDDEN)?;
            g.synchronize(g.default_stream())?;
            let p = download_f32(g, pred, n * LATENT_CH)?;
            let dt = sigmas[step + 1] - sigma;
            for (l, v) in latents.iter_mut().zip(&p) {
                *l += dt * v;
            }
            if let Some(t) = trace.as_deref_mut() {
                t.push((format!("noise_pred.{step}"), p));
                t.push((format!("latents.{step}"), latents.to_vec()));
            }
            for p in [m.scale1, m.scale_out] {
                pool.free(p);
            }
            on_step(step);
        }
        Ok(())
    }
}

// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3-VL-8B text tower as the Qwen-Image prompt encoder.
//!
//! Only the language model runs (text-to-image prompts carry no vision
//! tokens): 36 dense causal layers, GQA 32/8 heads, per-head q/k RMSNorm,
//! SwiGLU MLP. The output is the LAST LAYER'S hidden states BEFORE the
//! final RMSNorm — the DiT was trained on the un-normalized states (see
//! the diffusers pipeline's forward-hook note).

use std::path::Path;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::ops::{AttnBufs, Ops, Pool, WeightSet, pad_keys};
use super::st::{StDir, f32s_to_bytes};

pub(crate) const TE_HIDDEN: usize = 4096;
const TE_LAYERS: usize = 36;
const TE_HEADS: usize = 32;
const TE_KV_HEADS: usize = 8;
const TE_HEAD_DIM: usize = 128;
const TE_FFN: usize = 12288;
const TE_EPS: f32 = 1e-6;
const TE_ROPE_THETA: f64 = 5_000_000.0;

struct TeLayer {
    ln1: DevicePtr,
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    o: DevicePtr,
    q_norm: DevicePtr,
    k_norm: DevicePtr,
    ln2: DevicePtr,
    gate: DevicePtr,
    up: DevicePtr,
    down: DevicePtr,
}

pub(crate) struct TextEncoder {
    layers: Vec<TeLayer>,
    /// Checkpoint handle for host-side embedding row reads.
    st: StDir,
    pub weights_bytes: usize,
}

impl TextEncoder {
    pub fn load(dir: &Path, ws: &mut WeightSet<'_>) -> Result<Self> {
        let st = StDir::open(dir)?;
        let before = ws.bytes;
        let mut up = |name: String| -> Result<DevicePtr> { ws.upload(&st.read_bf16(&name)?) };
        let mut layers = Vec::with_capacity(TE_LAYERS);
        for i in 0..TE_LAYERS {
            let p = format!("model.language_model.layers.{i}");
            layers.push(TeLayer {
                ln1: up(format!("{p}.input_layernorm.weight"))?,
                q: up(format!("{p}.self_attn.q_proj.weight"))?,
                k: up(format!("{p}.self_attn.k_proj.weight"))?,
                v: up(format!("{p}.self_attn.v_proj.weight"))?,
                o: up(format!("{p}.self_attn.o_proj.weight"))?,
                q_norm: up(format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: up(format!("{p}.self_attn.k_norm.weight"))?,
                ln2: up(format!("{p}.post_attention_layernorm.weight"))?,
                gate: up(format!("{p}.mlp.gate_proj.weight"))?,
                up: up(format!("{p}.mlp.up_proj.weight"))?,
                down: up(format!("{p}.mlp.down_proj.weight"))?,
            });
        }
        Ok(Self {
            layers,
            st,
            weights_bytes: ws.bytes - before,
        })
    }

    /// Run the text tower over `tokens`; returns the pre-final-norm hidden
    /// states `[tokens, 4096]` as a BF16 device buffer owned by `pool`.
    pub fn forward(
        &self,
        g: &dyn GpuBackend,
        ops: &Ops,
        pool: &mut Pool<'_>,
        tokens: &[u32],
    ) -> Result<DevicePtr> {
        let t = tokens.len();
        let t_pad = pad_keys(t);
        let (hd, kvd) = (TE_HEADS * TE_HEAD_DIM, TE_KV_HEADS * TE_HEAD_DIM);

        let h = pool.upload(
            &self
                .st
                .read_rows_bf16("model.language_model.embed_tokens.weight", tokens)?,
        )?;

        // 1-D RoPE (text-only MRoPE: all three axes share the position).
        let half = TE_HEAD_DIM / 2;
        let mut cos = vec![0f32; t * half];
        let mut sin = vec![0f32; t * half];
        for pos in 0..t {
            for i in 0..half {
                let ang = pos as f64 / TE_ROPE_THETA.powf(i as f64 / half as f64);
                cos[pos * half + i] = ang.cos() as f32;
                sin[pos * half + i] = ang.sin() as f32;
            }
        }
        let cos = pool.upload(&f32s_to_bytes(&cos))?;
        let sin = pool.upload(&f32s_to_bytes(&sin))?;
        // Causal mask: row r sees keys [0, r].
        let limits: Vec<u8> = (0..t as u32).flat_map(|r| (r + 1).to_le_bytes()).collect();
        let limits = pool.upload(&limits)?;

        let x = pool.alloc(t * TE_HIDDEN * 2)?;
        let q = pool.alloc(t * hd * 2)?;
        let k = pool.alloc(t * kvd * 2)?;
        let v = pool.alloc(t * kvd * 2)?;
        let o = pool.alloc(t * hd * 2)?;
        let ff_a = pool.alloc(t * TE_FFN * 2)?;
        let ff_b = pool.alloc(t * TE_FFN * 2)?;
        let bufs = AttnBufs {
            qh: pool.alloc(t * hd * 2)?,
            kh: pool.alloc_zeroed(t_pad * kvd * 2)?,
            vt: pool.alloc_zeroed(t_pad * kvd * 2)?,
            scores: pool.alloc(t * t_pad * 4)?,
            probs: pool.alloc(t * t_pad * 2)?,
            o_stage: pool.alloc(t * TE_HEAD_DIM * 2)?,
        };

        for l in &self.layers {
            ops.rms_norm(g, h, l.ln1, x, t, TE_HIDDEN, TE_EPS, 0.0)?;
            ops.gemm(g, x, l.q, q, t, hd, TE_HIDDEN)?;
            ops.gemm(g, x, l.k, k, t, kvd, TE_HIDDEN)?;
            ops.gemm(g, x, l.v, v, t, kvd, TE_HIDDEN)?;
            ops.norm_rope_heads(
                g,
                q,
                l.q_norm,
                cos,
                sin,
                bufs.qh,
                t,
                TE_HEADS,
                TE_HEAD_DIM,
                0,
                t,
                TE_EPS,
                1,
            )?;
            ops.norm_rope_heads(
                g,
                k,
                l.k_norm,
                cos,
                sin,
                bufs.kh,
                t,
                TE_KV_HEADS,
                TE_HEAD_DIM,
                0,
                t_pad,
                TE_EPS,
                1,
            )?;
            ops.v_transpose(g, v, bufs.vt, t, TE_KV_HEADS, TE_HEAD_DIM, 0, t_pad)?;
            ops.attention(
                g,
                &bufs,
                o,
                limits,
                t,
                t,
                t_pad,
                TE_HEADS,
                TE_KV_HEADS,
                TE_HEAD_DIM,
            )?;
            ops.gemm(g, o, l.o, x, t, TE_HIDDEN, hd)?;
            ops.gated_add(g, h, x, DevicePtr::NULL, t, TE_HIDDEN)?;

            ops.rms_norm(g, h, l.ln2, x, t, TE_HIDDEN, TE_EPS, 0.0)?;
            ops.gemm(g, x, l.gate, ff_a, t, TE_FFN, TE_HIDDEN)?;
            ops.gemm(g, x, l.up, ff_b, t, TE_FFN, TE_HIDDEN)?;
            ops.silu_mul(g, ff_a, ff_b, ff_a, t * TE_FFN)?;
            ops.gemm(g, ff_a, l.down, x, t, TE_HIDDEN, TE_FFN)?;
            ops.gated_add(g, h, x, DevicePtr::NULL, t, TE_HIDDEN)?;
        }
        g.synchronize(g.default_stream())?;
        Ok(h)
    }
}

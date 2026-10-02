// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3-VL-8B text tower as the Qwen-Image prompt encoder.
//!
//! 36 dense causal layers, GQA 32/8 heads, per-head q/k RMSNorm, SwiGLU
//! MLP. The output is the LAST LAYER'S hidden states BEFORE the final
//! RMSNorm — the DiT was trained on the un-normalized states (see the
//! diffusers pipeline's forward-hook note).
//!
//! For `/v1/images/edits` the prompt carries condition images: their ViT
//! features (from [`super::vision::VisionTower`]) replace the
//! `<|image_pad|>` embedding rows, positions follow Qwen3-VL's
//! interleaved 3-axis MRoPE (text advances all axes; an image freezes the
//! frame axis and spans an h/w grid), and the deepstack tap features are
//! added to the image rows after decoder layers 0..2 — exactly
//! transformers' `_deepstack_process`.

use std::path::Path;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::ops::{AttnBufs, Ops, Pool, WeightSet, pad_keys};
use super::st::{StDir, f32s_to_bytes};
use super::vision::{ImgFeats, VisionTower};

pub(crate) const TE_HIDDEN: usize = 4096;
const TE_LAYERS: usize = 36;
const TE_HEADS: usize = 32;
const TE_KV_HEADS: usize = 8;
const TE_HEAD_DIM: usize = 128;
const TE_FFN: usize = 12288;
const TE_EPS: f32 = 1e-6;
const TE_ROPE_THETA: f64 = 5_000_000.0;

/// text_config.rope_parameters: interleaved MRoPE sections (t, h, w) over
/// the 64 rotary frequency indices.
const MROPE_SECTION: [usize; 3] = [24, 20, 20];

/// Which position axis frequency index `i` reads under the interleaved
/// layout (transformers `recomposition_frequencies`): H takes indices
/// 1, 4, …, < 3·sec_h; W takes 2, 5, …, < 3·sec_w; everything else stays
/// on the frame/text axis.
fn mrope_axis(i: usize) -> usize {
    match i % 3 {
        1 if i < 3 * MROPE_SECTION[1] => 1,
        2 if i < 3 * MROPE_SECTION[2] => 2,
        _ => 0,
    }
}

/// Condition-image context for an edit-prompt forward pass.
pub(crate) struct VlConditioning<'a> {
    pub tower: &'a VisionTower,
    /// Preprocessed (pixels, grid_h, grid_w) per image, template order.
    pub images: Vec<(&'a [f32], usize, usize)>,
    /// Contiguous `<|image_pad|>` token span per image: (start, len) in
    /// the token sequence, template order. `len == (grid_h/2)*(grid_w/2)`.
    pub spans: Vec<(usize, usize)>,
}

/// Per-token (t, h, w) positions: text runs advance all three axes, image
/// spans freeze the frame axis at the current position, lay out an
/// `h × w` grid offset by it, and advance by `max(h, w)` (Qwen3-VL
/// `get_rope_index` / `get_vision_position_ids`).
fn vl_positions(t: usize, spans: &[(usize, usize)], merged: &[(usize, usize)]) -> Vec<[i32; 3]> {
    let mut pos = Vec::with_capacity(t);
    let mut cur = 0i32;
    let mut idx = 0usize;
    for (k, &(start, len)) in spans.iter().enumerate() {
        while idx < start {
            pos.push([cur, cur, cur]);
            cur += 1;
            idx += 1;
        }
        let (h, w) = merged[k];
        debug_assert_eq!(len, h * w, "span/grid mismatch");
        for r in 0..h as i32 {
            for c in 0..w as i32 {
                pos.push([cur, cur + r, cur + c]);
            }
        }
        idx += len;
        cur += h.max(w) as i32;
    }
    while idx < t {
        pos.push([cur, cur, cur]);
        cur += 1;
        idx += 1;
    }
    pos
}

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
    /// Checkpoint handle (the edit stack loads the ViT from the same
    /// text_encoder shards).
    pub fn st(&self) -> &StDir {
        &self.st
    }

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
    ///
    /// With `vl` the condition images' ViT features replace the
    /// `<|image_pad|>` rows before layer 0 and the deepstack taps are
    /// added after layers 0..2; positions switch to the 3-axis layout.
    pub fn forward(
        &self,
        g: &dyn GpuBackend,
        ops: &Ops,
        pool: &mut Pool<'_>,
        tokens: &[u32],
        vl: Option<&VlConditioning<'_>>,
    ) -> Result<DevicePtr> {
        let t = tokens.len();
        let t_pad = pad_keys(t);
        let (hd, kvd) = (TE_HEADS * TE_HEAD_DIM, TE_KV_HEADS * TE_HEAD_DIM);

        let h = pool.upload(
            &self
                .st
                .read_rows_bf16("model.language_model.embed_tokens.weight", tokens)?,
        )?;

        // Run the ViT and splice its merged features over the image rows;
        // remember the per-span deepstack sources for the layer-0..2 adds.
        let mut deepstack: Vec<(usize, usize, usize)> = Vec::new(); // (start, len, feature row_off)
        let mut merged_grids = Vec::new();
        let mut total_merged = 0usize;
        if let Some(vl) = vl {
            ensure!(
                vl.images.len() == vl.spans.len(),
                "images/spans count mismatch"
            );
            let feats: Vec<ImgFeats>;
            (feats, total_merged) = vl.tower.forward(&vl.images, g, g.default_stream())?;
            for (k, f) in feats.iter().enumerate() {
                let (start, len) = vl.spans[k];
                ensure!(
                    f.merged_p == len,
                    "image {k}: {} merged ViT tokens but a {len}-token pad span",
                    f.merged_p,
                );
                ensure!(start + len <= t, "image span out of range");
                g.copy_d2d(
                    vl.tower.final_ptr(f),
                    h.offset(start * TE_HIDDEN * 2),
                    len * TE_HIDDEN * 2,
                )?;
                deepstack.push((start, len, f.row_off));
                let (_, gh, gw) = vl.images[k];
                merged_grids.push((gh / 2, gw / 2));
            }
        }
        let spans: Vec<(usize, usize)> = vl.map(|v| v.spans.clone()).unwrap_or_default();

        // Interleaved 3-axis MRoPE tables. Without images every token sits
        // at the same position on all axes, which reproduces the 1-D case.
        let positions = vl_positions(t, &spans, &merged_grids);
        let half = TE_HEAD_DIM / 2;
        let mut cos = vec![0f32; t * half];
        let mut sin = vec![0f32; t * half];
        for (pos_idx, p) in positions.iter().enumerate() {
            for i in 0..half {
                let ang = p[mrope_axis(i)] as f64 / TE_ROPE_THETA.powf(i as f64 / half as f64);
                cos[pos_idx * half + i] = ang.cos() as f32;
                sin[pos_idx * half + i] = ang.sin() as f32;
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

        for (li, l) in self.layers.iter().enumerate() {
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

            // Deepstack: after decoder layers 0..2 add ViT tap `li` to the
            // image rows (`_deepstack_process`; each span is contiguous).
            if let Some(vl) = vl
                && li < vl.tower.deepstack_taps()
            {
                for &(start, len, row_off) in &deepstack {
                    let f = ImgFeats {
                        row_off,
                        merged_p: len,
                    };
                    ops.gated_add(
                        g,
                        h.offset(start * TE_HIDDEN * 2),
                        vl.tower.deepstack_ptr(li, total_merged, &f),
                        DevicePtr::NULL,
                        len,
                        TE_HIDDEN,
                    )?;
                }
            }
        }
        g.synchronize(g.default_stream())?;
        Ok(h)
    }
}

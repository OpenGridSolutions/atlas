// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3-VL-8B vision tower for the image-conditioned (edit) path.
//!
//! The Qwen-Image text encoder is a full Qwen3-VL model; for `/v1/images/
//! edits` the condition images run through its ViT so the prompt tokens are
//! visually grounded. The tower itself is the SAME architecture Atlas
//! already serves for the chat VL models, so this is a thin loader around
//! [`crate::layers::VisionEncoder`] (27 blocks, hidden 1152, 16 heads,
//! spatial merge 2, out 4096, deepstack taps after blocks 8/16/24) fed from
//! the sidecar's safetensors reader instead of a `WeightStore`.
//!
//! Loaded LAZILY on the first edit request — text-to-image deployments do
//! not pay the ~0.9 GiB of tower weights plus the encoder's ~2 GiB scratch.
//! The `VisionEncoder` scratch buffers are plain `gpu.alloc`s owned by the
//! encoder for the sidecar's (process) lifetime.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use crate::layers::{MergerLayer, ViTBlock, VisionEncoder};

use super::ops::WeightSet;
use super::st::StDir;

/// text_encoder/config.json `vision_config` (checkpoint constants).
const VIT_DEPTH: usize = 27;
const VIT_HIDDEN: usize = 1152;
const VIT_HEADS: usize = 16;
const VIT_MERGE: usize = 2;
const VIT_OUT: usize = 4096;
const VIT_FFN: usize = 4304;
const DEEPSTACK_INDEXES: [usize; 3] = [8, 16, 24];

/// Per-image result of a tower pass, pointing into the encoder's output
/// buffer (valid until the next forward).
pub(crate) struct ImgFeats {
    /// Row offset of this image's merged tokens in `buf_out`.
    pub row_off: usize,
    /// Merged token count ((grid_h/2) * (grid_w/2)).
    pub merged_p: usize,
}

pub(crate) struct VisionTower {
    enc: VisionEncoder,
    pub weights_bytes: usize,
}

impl VisionTower {
    /// Load the tower from the text_encoder safetensors (keys under
    /// `model.visual.`).
    pub fn load(st: &StDir, ws: &mut WeightSet<'_>, gpu: &dyn GpuBackend) -> Result<Self> {
        let before = ws.bytes;
        let vp = "model.visual";
        let mut up = |name: String| -> Result<_> { ws.upload(&st.read_bf16(&name)?) };

        let patch_embed_w = up(format!("{vp}.patch_embed.proj.weight"))?;
        let patch_embed_b = up(format!("{vp}.patch_embed.proj.bias"))?;
        let pos_embed = up(format!("{vp}.pos_embed.weight"))?;
        let num_position_embeddings = st.entry(&format!("{vp}.pos_embed.weight"))?.shape[0];

        let mut blocks = Vec::with_capacity(VIT_DEPTH);
        for i in 0..VIT_DEPTH {
            let bp = format!("{vp}.blocks.{i}");
            blocks.push(ViTBlock {
                norm1_w: up(format!("{bp}.norm1.weight"))?,
                norm1_b: up(format!("{bp}.norm1.bias"))?,
                qkv_w: up(format!("{bp}.attn.qkv.weight"))?,
                qkv_b: up(format!("{bp}.attn.qkv.bias"))?,
                proj_w: up(format!("{bp}.attn.proj.weight"))?,
                proj_b: up(format!("{bp}.attn.proj.bias"))?,
                norm2_w: up(format!("{bp}.norm2.weight"))?,
                norm2_b: up(format!("{bp}.norm2.bias"))?,
                fc1_w: up(format!("{bp}.mlp.linear_fc1.weight"))?,
                fc1_b: up(format!("{bp}.mlp.linear_fc1.bias"))?,
                fc2_w: up(format!("{bp}.mlp.linear_fc2.weight"))?,
                fc2_b: up(format!("{bp}.mlp.linear_fc2.bias"))?,
            });
        }
        let mut merger_at = |base: String| -> Result<MergerLayer> {
            Ok(MergerLayer {
                norm_w: up(format!("{base}.norm.weight"))?,
                norm_b: up(format!("{base}.norm.bias"))?,
                fc1_w: up(format!("{base}.linear_fc1.weight"))?,
                fc1_b: up(format!("{base}.linear_fc1.bias"))?,
                fc2_w: up(format!("{base}.linear_fc2.weight"))?,
                fc2_b: up(format!("{base}.linear_fc2.bias"))?,
            })
        };
        let mut deepstack = Vec::with_capacity(DEEPSTACK_INDEXES.len());
        for i in 0..DEEPSTACK_INDEXES.len() {
            deepstack.push(merger_at(format!("{vp}.deepstack_merger_list.{i}"))?);
        }
        let merger = merger_at(format!("{vp}.merger"))?;

        let enc = VisionEncoder::new(
            patch_embed_w,
            patch_embed_b,
            pos_embed,
            num_position_embeddings,
            blocks,
            deepstack,
            DEEPSTACK_INDEXES.to_vec(),
            merger,
            VIT_HIDDEN,
            VIT_HEADS,
            VIT_MERGE,
            VIT_OUT,
            VIT_FFN,
            gpu,
        )?;
        Ok(Self {
            enc,
            weights_bytes: ws.bytes - before,
        })
    }

    /// Run the tower over the condition images. Returns per-image feature
    /// locations plus the total merged row count (`Σ merged_p`, the stride
    /// between deepstack taps in the output buffer).
    pub fn forward(
        &self,
        images: &[(&[f32], usize, usize)],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<(Vec<ImgFeats>, usize)> {
        let per_image = self.enc.forward_batched(images, gpu, stream)?;
        let mut out = Vec::with_capacity(per_image.len());
        let mut row_off = 0;
        for (_, _, merged_p) in per_image {
            out.push(ImgFeats { row_off, merged_p });
            row_off += merged_p;
        }
        Ok((out, row_off))
    }

    /// Device pointer to one image's FINAL merger rows `[merged_p, 4096]`.
    pub fn final_ptr(&self, f: &ImgFeats) -> spark_runtime::gpu::DevicePtr {
        self.enc.buf_out.offset(f.row_off * VIT_OUT * 2)
    }

    /// Device pointer to one image's deepstack-`k` rows `[merged_p, 4096]`.
    /// Layout per `VisionEncoder::forward_batched`: tap `k` starts at row
    /// `(k + 1) * Σmerged_p`.
    pub fn deepstack_ptr(
        &self,
        k: usize,
        total_merged: usize,
        f: &ImgFeats,
    ) -> spark_runtime::gpu::DevicePtr {
        self.enc
            .buf_out
            .offset(((k + 1) * total_merged + f.row_off) * VIT_OUT * 2)
    }

    pub fn deepstack_taps(&self) -> usize {
        DEEPSTACK_INDEXES.len()
    }
}

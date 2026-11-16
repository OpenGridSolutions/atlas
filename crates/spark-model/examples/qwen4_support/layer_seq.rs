// SPDX-License-Identifier: AGPL-3.0-only
//! Harness of `qwen4_layers_microtest`: one sequence driven through
//! `Qwen4ExpLayer::decode` / `prefill` on the GPU, and quiet checks.
use anyhow::Result;
use atlas_core::config::{LayerType, ModelConfig};
use spark_model::layer::{
    AttnMetadataDev, DecodePrep, EmptyLayerState, ForwardContext, LayerState, SsmLayerState,
    TransformerLayer,
};
use spark_model::layers::qwen4_exp::Qwen4ExpShared;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::support::reference::{self, H, HC};
use super::support::{self, f32_to_bf16, f32s, le16};
use super::{BLOCK, TOKENS};

/// Fresh per-sequence state for every layer. The n-gram history starts
/// dirty: it must not leak into position 0.
pub fn new_states(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    n_layers: usize,
) -> Result<Vec<Box<dyn LayerState>>> {
    let mut states: Vec<Box<dyn LayerState>> = Vec::new();
    for i in 0..n_layers {
        if config.layer_type(i) != LayerType::LinearAttention {
            states.push(Box::new(EmptyLayerState));
            continue;
        }
        let (hb, cb) = (config.ssm_h_state_bytes(), config.ssm_conv_state_bytes());
        let (h_state, conv_state) = (gpu.alloc(hb)?, gpu.alloc(cb)?);
        let gdn = config.gdn_conv_state_bytes();
        gpu.memset(h_state, 0, hb)?;
        gpu.memset(conv_state, 0, gdn)?;
        gpu.memset(conv_state.offset(gdn), 0x41, cb - gdn)?;
        states.push(Box::new(SsmLayerState {
            h_state,
            conv_state,
            h_state_checkpoint: None,
            conv_state_checkpoint: None,
            h_state_intermediates: Vec::new(),
            conv_state_intermediates: Vec::new(),
        }));
    }
    Ok(states)
}

/// One sequence on the GPU: layers, state, KV blocks.
pub struct Seq<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub config: &'a ModelConfig,
    pub buffers: &'a BufferArena,
    pub shared: &'a Qwen4ExpShared,
    pub layers: &'a [Box<dyn TransformerLayer>],
    pub dir: &'a str,
    /// Positions u32 @0, KV slots i64 @1024, length i32 @2048, table @4096.
    pub meta: DevicePtr,
    pub states: Vec<Box<dyn LayerState>>,
    pub block_table: Vec<u32>,
}

impl<'a> Seq<'a> {
    /// The same sequence setup over other layers, with fresh state.
    pub fn with(
        &self,
        layers: &'a [Box<dyn TransformerLayer>],
        states: Vec<Box<dyn LayerState>>,
    ) -> Seq<'a> {
        Seq {
            layers,
            states,
            block_table: self.block_table.clone(),
            ..*self
        }
    }

    /// Uploads embeddings and metadata of tokens `start..start + n`, runs
    /// the layers' host preparation and returns the forward context.
    pub fn stage(&self, start: usize, n: usize, prefill: bool) -> Result<ForwardContext<'_>> {
        let (gpu, meta) = (self.gpu, self.meta);
        let mut emb_bf = Vec::with_capacity(n * H);
        let (mut pos_b, mut slot_b) = (Vec::new(), Vec::new());
        for pos in start..start + n {
            let emb = reference::embedding(self.dir, TOKENS[pos])?;
            emb_bf.extend(emb.iter().map(|v| f32_to_bf16(*v as f32)));
            pos_b.extend((pos as u32).to_le_bytes());
            let slot = self.block_table[pos / BLOCK] as i64 * BLOCK as i64 + (pos % BLOCK) as i64;
            slot_b.extend(slot.to_le_bytes());
        }
        gpu.copy_h2d(&le16(&emb_bf), self.buffers.hidden_states())?;
        gpu.copy_h2d(&pos_b, meta)?;
        gpu.copy_h2d(&slot_b, meta.offset(1024))?;
        gpu.copy_h2d(&((start + n) as i32).to_le_bytes(), meta.offset(2048))?;
        let bt: Vec<u8> = self
            .block_table
            .iter()
            .flat_map(|b| (*b as i32).to_le_bytes())
            .collect();
        gpu.copy_h2d(&bt, meta.offset(4096))?;
        let stream = gpu.default_stream();
        let prep = DecodePrep {
            history: &TOKENS[..start],
            tokens: &TOKENS[start..start + n],
            prefill,
            gpu,
            stream,
        };
        for l in self.layers {
            l.prepare_decode(&prep)?;
        }
        gpu.synchronize(stream)?;
        Ok(ForwardContext {
            buffers: self.buffers,
            gpu,
            config: self.config,
            attn_metadata: Some(AttnMetadataDev {
                positions: meta,
                positions_h: meta,
                positions_w: meta,
                slot: meta.offset(1024),
                seq_len: meta.offset(2048),
                block_table: meta.offset(4096),
                max_blocks_per_seq: self.block_table.len() as u32,
                num_seqs: 1,
            }),
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
        })
    }

    /// Decodes token `pos`; `seen(layer, streams)` after every layer.
    pub fn decode(
        &mut self,
        kv: &mut PagedKvCache,
        pos: usize,
        mut seen: impl FnMut(usize, Vec<f32>),
    ) -> Result<()> {
        let (gpu, stream) = (self.gpu, self.gpu.default_stream());
        let (hidden, residual) = (self.buffers.hidden_states(), self.buffers.residual());
        let (mut disk, mut off, mut bt) = (Vec::new(), Vec::new(), self.block_table.clone());
        let mut states = std::mem::take(&mut self.states);
        let ctx = self.stage(pos, 1, false)?;
        for (i, l) in self.layers.iter().enumerate() {
            l.decode(
                hidden,
                residual,
                states[i].as_mut(),
                kv,
                pos,
                &mut bt,
                &mut disk,
                &mut off,
                &ctx,
                stream,
            )?;
            gpu.synchronize(stream)?;
            let bytes = support::download(gpu, self.buffers.hc_streams(), HC * H * 4)?;
            seen(i, f32s(&bytes));
        }
        drop(ctx);
        self.states = states;
        Ok(())
    }

    /// Prefills tokens `start..start + n` as one chunk; `seen(layer, pos,
    /// streams)` for every token after every layer.
    pub fn prefill(
        &mut self,
        kv: &mut PagedKvCache,
        start: usize,
        n: usize,
        mut seen: impl FnMut(usize, usize, &[f32]),
    ) -> Result<()> {
        let (gpu, stream) = (self.gpu, self.gpu.default_stream());
        let (hidden, residual) = (self.buffers.hidden_states(), self.buffers.residual());
        let (mut disk, mut off, mut bt) = (Vec::new(), Vec::new(), self.block_table.clone());
        let mut states = std::mem::take(&mut self.states);
        let ctx = self.stage(start, n, true)?;
        for (i, l) in self.layers.iter().enumerate() {
            l.prefill(
                hidden,
                residual,
                n,
                states[i].as_mut(),
                kv,
                start,
                &mut bt,
                &mut disk,
                &mut off,
                0,
                &ctx,
                stream,
            )?;
            gpu.synchronize(stream)?;
            let bytes = support::download(gpu, self.shared.prefill_streams(), n * HC * H * 4)?;
            for (t, row) in f32s(&bytes).chunks_exact(HC * H).enumerate() {
                seen(i, start + t, row);
            }
        }
        drop(ctx);
        self.states = states;
        Ok(())
    }
}

/// Checks of one section that only print failures and a summary.
pub struct Quiet {
    ok: bool,
    worst_cos: f64,
    worst_diff: f64,
    count: usize,
}
impl Quiet {
    pub fn new() -> Self {
        Self {
            ok: true,
            worst_cos: 1.0,
            worst_diff: 0.0,
            count: 0,
        }
    }
    /// `got` within `min_cos` of `want`; with `min_cos` = 1.0: equal.
    pub fn add(&mut self, name: &str, got: &[f32], want: &[f32], min_cos: f64) {
        let (mut dot, mut ng, mut nw, mut diff) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (g, w) in got.iter().zip(want) {
            let (g, w) = (*g as f64, *w as f64);
            dot += g * w;
            ng += g * g;
            nw += w * w;
            diff = diff.max((g - w).abs());
        }
        let cos = dot / (ng.sqrt() * nw.sqrt());
        let ok = if min_cos >= 1.0 {
            diff == 0.0
        } else {
            cos.is_finite() && cos > min_cos
        };
        if !ok {
            println!("  {name}: cosine={cos:.7} max_diff={diff:.2e} FAIL");
        }
        self.ok &= ok;
        self.worst_cos = self.worst_cos.min(if cos.is_finite() { cos } else { -1.0 });
        self.worst_diff = self.worst_diff.max(diff);
        self.count += 1;
    }
    pub fn done(self, what: &str) -> bool {
        println!(
            "  {what}: {} checks, worst cosine={:.7}, largest difference={:.2e} {}",
            self.count,
            self.worst_cos,
            self.worst_diff,
            if self.ok { "ok" } else { "FAIL" }
        );
        self.ok
    }
}

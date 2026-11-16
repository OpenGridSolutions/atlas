// SPDX-License-Identifier: AGPL-3.0-only

//! Decoder layer of the `qwen4_exp` family (Qwen3.8-Flash-Next).
//!
//! The model carries `hc` residual streams per token (FP32, in the arena's
//! `hc_streams` buffer). One layer is
//!
//! ```text
//! [layer 0]   streams = embedding, copied into every stream
//! [PLE layer] streams += n-gram features
//! x = mix(streams);  streams += (GDN | attention)(x) * inject
//! x = mix(streams);  streams += MoE(x) * inject
//! [last]      hidden = final_mix(streams)        -> LM head, no final norm
//! ```
//!
//! `decode` runs one token, `prefill` a chunk of up to [`PREFILL_ROWS`]
//! tokens of one sequence. Both advance the same per-sequence state, so a
//! prompt may be split between them at any token.

mod attention;
mod kernels;
mod ple;
mod prefill;
mod scratch;
mod sublayers;
pub mod weights;

use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{ModelConfig, Qwen4ExpConfig};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use self::kernels::{Kernels, StreamDims};
pub use self::ple::Ple;
pub use self::prefill::PREFILL_ROWS;
use self::prefill::{Pass, PrefillKernels, PrefillScratch};
use self::scratch::Scratch;
use self::weights::{Mixer, MoeWeights, Sublayer};
use crate::layer::{
    DecodePrep, EmptyLayerState, ForwardContext, LayerState, SsmLayerState, TransformerLayer,
};

/// State shared by all layers of one model: kernel handles, working memory.
pub struct Qwen4ExpShared {
    kernels: Kernels,
    scratch: Scratch,
    prefill_kernels: PrefillKernels,
    prefill_scratch: PrefillScratch,
    family: Qwen4ExpConfig,
    dims: StreamDims,
}

impl Qwen4ExpShared {
    pub fn new(gpu: &dyn GpuBackend, config: &ModelConfig) -> Result<Arc<Self>> {
        let Some(family) = config.qwen4_exp.clone() else {
            bail!("qwen4_exp layer needs the family parameters of the config");
        };
        let dims = StreamDims {
            hidden: config.hidden_size as u32,
            hc: config.hc_mult as u32,
            rank: family.hc_lowrank as u32,
            eps: config.rms_norm_eps as f32,
        };
        Ok(Arc::new(Self {
            kernels: Kernels::load(gpu)?,
            scratch: Scratch::new(gpu, config, &family)?,
            prefill_kernels: PrefillKernels::load(gpu)?,
            prefill_scratch: PrefillScratch::new(gpu, config, &family)?,
            family,
            dims,
        }))
    }

    /// The residual streams the last `prefill` call left behind, FP32
    /// `[tokens, hc, H]` (diagnostics and tests).
    pub fn prefill_streams(&self) -> DevicePtr {
        self.prefill_scratch.streams
    }

    /// Bytes of the shared working memory.
    pub fn scratch_bytes(&self) -> usize {
        self.scratch.bytes + self.prefill_scratch.bytes
    }

    /// Counts malformed entries in the compressed scales of one expert
    /// projection (`rows` x `cols` scale bytes per expert); 0 = valid.
    pub fn csf_errors(
        &self,
        gpu: &dyn GpuBackend,
        proj: &weights::ExpertProj,
        num_experts: usize,
        rows: usize,
        cols: usize,
    ) -> Result<u32> {
        use spark_runtime::kernel_args::KernelLaunch;
        let stream = gpu.default_stream();
        // The first u32 of the router-logit scratch serves as the counter.
        let counter = self.scratch.router_logits;
        gpu.copy_h2d(&0u32.to_le_bytes(), counter)?;
        KernelLaunch::new(gpu, self.kernels.csf_validate)
            .grid([num_experts as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(proj.fixed_ptrs)
            .arg_ptr(proj.exc_ptrs)
            .arg_ptr(proj.exc_counts)
            .arg_ptr(counter)
            .arg_u32(rows as u32)
            .arg_u32(cols as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut out = [0u8; 4];
        gpu.copy_d2h(counter, &mut out)?;
        Ok(u32::from_le_bytes(out))
    }
}

pub struct Qwen4ExpLayer {
    pub shared: Arc<Qwen4ExpShared>,
    pub layer_idx: usize,
    pub is_first: bool,
    pub ple: Option<Ple>,
    pub attn_mixer: Mixer,
    pub sublayer: Sublayer,
    pub mlp_mixer: Mixer,
    pub moe: MoeWeights,
    /// Present on the last layer only.
    pub final_mixer: Option<Mixer>,
}

impl Qwen4ExpLayer {
    /// Host work before a decode step or a prefill chunk: the table rows of
    /// the tokens. Runs outside the decode graph.
    pub fn prepare(&self, p: &DecodePrep) -> Result<()> {
        let Some(ple) = &self.ple else {
            return Ok(());
        };
        let (rows, scales) = if p.prefill {
            if p.tokens.len() > PREFILL_ROWS {
                bail!("{} tokens exceed a prefill chunk", p.tokens.len());
            }
            let s = &self.shared.prefill_scratch;
            (s.ple_rows, s.ple_row_scales)
        } else {
            let s = &self.shared.scratch;
            (s.ple_rows, s.ple_row_scales)
        };
        ple.prepare(p.history, p.tokens, p.gpu, rows, scales, p.stream)
    }

    fn gdn_state<'a>(&self, state: &'a dyn LayerState) -> Result<&'a SsmLayerState> {
        match state.as_any().downcast_ref::<SsmLayerState>() {
            Some(st) => Ok(st),
            None => bail!("layer {}: no recurrent state", self.layer_idx),
        }
    }
}

impl TransformerLayer for Qwen4ExpLayer {
    fn decode(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (gpu, config) = (ctx.gpu, ctx.config);
        let (k, s, d) = (&self.shared.kernels, &self.shared.scratch, self.shared.dims);
        if hidden != ctx.buffers.hidden_states() {
            bail!("qwen4_exp layers decode one token at row 0 of the hidden buffer");
        }
        let Some(meta) = ctx.attn_metadata.as_ref() else {
            bail!("qwen4_exp layers need the decode metadata (positions, KV slot)");
        };
        let streams = ctx.buffers.hc_streams();

        if self.is_first {
            kernels::expand(gpu, k, d, hidden, streams, stream)?;
        }
        if let Some(ple) = &self.ple {
            // History lives behind the GDN conv state in this layer's slot.
            let st = self.gdn_state(state)?;
            let history = st.conv_state.offset(config.gdn_conv_state_bytes());
            ple.forward(gpu, k, s, d, streams, history, meta.positions, stream)?;
        }

        kernels::mix(
            gpu,
            k,
            d,
            &self.attn_mixer,
            streams,
            s.normed,
            s.act,
            s.inj_a,
            s.x,
            stream,
        )?;
        match &self.sublayer {
            Sublayer::Gdn(w) => {
                let st = self.gdn_state(state)?;
                let sigmoid = self.shared.family.gdn_output_gate_sigmoid;
                sublayers::gdn(gpu, k, s, config, sigmoid, w, st, stream)?;
            }
            Sublayer::Attn(w) => {
                if kv_cache.dtype_for_layer(w.attn_idx) != KvCacheDtype::Bf16 {
                    bail!(
                        "qwen4_exp attention supports a BF16 KV cache only (--kv-cache-dtype bf16)"
                    );
                }
                let q = &self.shared.family;
                attention::attention(gpu, k, s, config, q, w, kv_cache, meta, stream)?;
            }
        }
        kernels::post(gpu, k, d, s.y, s.inj_a, streams, stream)?;

        kernels::mix(
            gpu,
            k,
            d,
            &self.mlp_mixer,
            streams,
            s.normed,
            s.act,
            s.inj_b,
            s.x,
            stream,
        )?;
        sublayers::moe(gpu, k, s, config, &self.moe, stream)?;
        kernels::post(gpu, k, d, s.y, s.inj_b, streams, stream)?;

        if let Some(m) = &self.final_mixer {
            kernels::mix(
                gpu,
                k,
                d,
                m,
                streams,
                s.normed,
                s.act,
                DevicePtr::NULL,
                hidden,
                stream,
            )?;
        }
        Ok(())
    }

    /// Chunk of `num_tokens` tokens at positions `seq_len_start..`, the
    /// embeddings in `hidden` (`[T, H]`). The last layer leaves the final
    /// hidden states there.
    fn prefill(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (config, d) = (ctx.config, self.shared.dims);
        if num_tokens == 0 || num_tokens > PREFILL_ROWS {
            bail!("qwen4_exp prefill takes 1..={PREFILL_ROWS} tokens, got {num_tokens}");
        }
        let Some(meta) = ctx.attn_metadata.as_ref() else {
            bail!("qwen4_exp layers need the prefill metadata (positions, KV slots)");
        };
        let p = Pass {
            gpu: ctx.gpu,
            k: &self.shared.kernels,
            pk: &self.shared.prefill_kernels,
            s: &self.shared.prefill_scratch,
            s1: &self.shared.scratch,
            stream,
            n: num_tokens as u32,
        };
        let s = p.s;

        if self.is_first {
            p.expand(d, hidden)?;
        }
        if let Some(ple) = &self.ple {
            let history = self
                .gdn_state(state)?
                .conv_state
                .offset(config.gdn_conv_state_bytes());
            ple.forward_prefill(&p, d, history, meta.positions)?;
        }

        p.mix(d, &self.attn_mixer, s.inj_a, s.x)?;
        match &self.sublayer {
            Sublayer::Gdn(w) => {
                let sigmoid = self.shared.family.gdn_output_gate_sigmoid;
                prefill::gdn(&p, config, sigmoid, w, self.gdn_state(state)?)?;
            }
            Sublayer::Attn(w) => {
                if kv_cache.dtype_for_layer(w.attn_idx) != KvCacheDtype::Bf16 {
                    bail!(
                        "qwen4_exp attention supports a BF16 KV cache only (--kv-cache-dtype bf16)"
                    );
                }
                let q = &self.shared.family;
                prefill::attention(&p, config, q, w, kv_cache, meta, seq_len_start)?;
            }
        }
        p.post(d, s.y, s.inj_a)?;

        p.mix(d, &self.mlp_mixer, s.inj_b, s.x)?;
        prefill::moe(&p, config, &self.moe)?;
        p.post(d, s.y, s.inj_b)?;

        if let Some(m) = &self.final_mixer {
            p.mix(d, m, DevicePtr::NULL, hidden)?;
        }
        Ok(())
    }

    fn prepare_decode(&self, p: &DecodePrep) -> Result<()> {
        self.prepare(p)
    }

    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        match self.sublayer {
            Sublayer::Attn(_) => Ok(Box::new(EmptyLayerState)),
            // GDN state comes from the model's state pool, sized from the
            // config's linear_* fields; there is no standalone allocation.
            Sublayer::Gdn(_) => bail!("qwen4_exp GDN state is allocated by the model's state pool"),
        }
    }
}

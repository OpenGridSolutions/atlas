// SPDX-License-Identifier: AGPL-3.0-only

//! Whisper encoder-decoder as a RESIDENT SIDECAR MODULE (multiclass
//! Plan.md Phase W) — deliberately OUTSIDE the `Model` trait / scheduler
//! / hot-swap stack (gate finding W.0): the audio encoder follows the
//! `VisionEncoder` pattern (self-contained GPU transformer driven
//! synchronously), the text decoder is a greedy loop with dense
//! per-layer KV buffers (max 448 positions — no paged KV).
//!
//! Precision: the checkpoint's FP32 shards convert to BF16 at load; all
//! compute is BF16 (common `dense_gemm_bf16`/`dense_gemv_bf16` + the
//! whisper.cu kernels). NO NVFP4 quantization anywhere in this path.

mod forward;

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::weights::WeightStore;

pub const D_MODEL: usize = 1280;
pub const N_HEADS: usize = 20;
pub const HEAD_DIM: usize = 64;
pub const N_LAYERS: usize = 32; // encoder AND decoder depth
pub const FFN_DIM: usize = 5120;
pub const N_AUDIO_CTX: usize = 1500;
pub const N_TEXT_CTX: usize = 448;
pub const VOCAB: usize = 51_866;
pub const N_MELS: usize = crate::audio_preprocess::N_MELS;
const LN_EPS: f32 = 1e-5;

/// (weight, bias) device pointers, BF16.
#[derive(Clone, Copy)]
pub(crate) struct WB {
    pub w: DevicePtr,
    pub b: DevicePtr,
}

pub(crate) struct AttnWeights {
    pub q: WB,
    /// k_proj is biasless in Whisper — `k_b` is a zero vector so the
    /// shared bias-add path stays uniform.
    pub k: WB,
    pub v: WB,
    pub out: WB,
}

pub(crate) struct EncLayer {
    pub ln1: WB,
    pub attn: AttnWeights,
    pub ln2: WB,
    pub fc1: WB,
    pub fc2: WB,
}

pub(crate) struct DecLayer {
    pub ln1: WB,
    pub self_attn: AttnWeights,
    pub ln2: WB,
    pub cross_attn: AttnWeights,
    pub ln3: WB,
    pub fc1: WB,
    pub fc2: WB,
    /// Decoder self-attention cache, dense `[N_TEXT_CTX, N_HEADS, HEAD_DIM]`.
    pub k_cache: DevicePtr,
    pub v_cache: DevicePtr,
    /// Cross-attention K/V over the CURRENT chunk's encoder output,
    /// `[N_AUDIO_CTX, N_HEADS, HEAD_DIM]` — recomputed per chunk.
    pub k_cross: DevicePtr,
    pub v_cross: DevicePtr,
}

pub(crate) struct Kernels {
    pub gemm: KernelHandle,
    pub gemv: KernelHandle,
    pub layernorm: KernelHandle,
    pub bias_add: KernelHandle,
    pub bias_gelu: KernelHandle,
    pub attention: KernelHandle,
    pub add_inplace: KernelHandle,
}

/// Scratch buffers, allocated once for the worst-case shapes.
pub(crate) struct Scratch {
    /// [N_AUDIO_CTX(×2 for conv1 out at 3000), max(D_MODEL, FFN_DIM)]
    pub h_a: DevicePtr,
    pub h_b: DevicePtr,
    pub h_c: DevicePtr,
    /// q/k/v/attn-out for the widest case [N_AUDIO_CTX, D_MODEL]
    pub q: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub o: DevicePtr,
    /// decode-step logits [VOCAB]
    pub logits: DevicePtr,
    /// conv im2col upload buffer (host-filled): [3000, N_MELS*3] then
    /// [1500, D_MODEL*3] — sized for the larger.
    pub im2col: DevicePtr,
}

pub struct Whisper {
    pub(crate) gpu: Box<dyn GpuBackend>,
    pub(crate) kern: Kernels,
    pub(crate) conv1: WB, // [D_MODEL, N_MELS*3] rearranged for im2col GEMM
    pub(crate) conv2: WB, // [D_MODEL, D_MODEL*3]
    pub(crate) enc_pos: DevicePtr,
    pub(crate) enc_layers: Vec<EncLayer>,
    pub(crate) enc_ln: WB,
    pub(crate) tok_emb: DevicePtr, // [VOCAB, D_MODEL] (tied lm head)
    pub(crate) dec_pos: DevicePtr, // [N_TEXT_CTX, D_MODEL]
    pub(crate) dec_layers: Vec<DecLayer>,
    pub(crate) dec_ln: WB,
    pub(crate) enc_out: DevicePtr, // [N_AUDIO_CTX, D_MODEL] current chunk
    pub(crate) scratch: Scratch,
}

fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    (bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

/// Download an FP32 store tensor to host f32.
fn f32_host(store: &WeightStore, gpu: &dyn GpuBackend, name: &str) -> Result<Vec<f32>> {
    let w = store.get(name).with_context(|| format!("missing {name}"))?;
    anyhow::ensure!(
        w.dtype == spark_runtime::weights::WeightDtype::FP32,
        "{name}: expected FP32 checkpoint, got {:?}",
        w.dtype
    );
    let n = w.num_elements();
    let mut raw = vec![0u8; n * 4];
    gpu.copy_d2h(w.ptr, &mut raw)?;
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn upload_bf16(gpu: &dyn GpuBackend, vals: &[f32]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = vals
        .iter()
        .flat_map(|v| f32_to_bf16_bits(*v).to_le_bytes())
        .collect();
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}

/// FP32 store tensor → BF16 device tensor.
fn bf16(store: &WeightStore, gpu: &dyn GpuBackend, name: &str) -> Result<DevicePtr> {
    upload_bf16(gpu, &f32_host(store, gpu, name)?)
}

fn wb(store: &WeightStore, gpu: &dyn GpuBackend, base: &str) -> Result<WB> {
    Ok(WB {
        w: bf16(store, gpu, &format!("{base}.weight"))?,
        b: bf16(store, gpu, &format!("{base}.bias"))?,
    })
}

fn attn(store: &WeightStore, gpu: &dyn GpuBackend, base: &str) -> Result<AttnWeights> {
    Ok(AttnWeights {
        q: wb(store, gpu, &format!("{base}.q_proj"))?,
        k: WB {
            w: bf16(store, gpu, &format!("{base}.k_proj.weight"))?,
            b: upload_bf16(gpu, &vec![0f32; D_MODEL])?, // biasless k_proj
        },
        v: wb(store, gpu, &format!("{base}.v_proj"))?,
        out: wb(store, gpu, &format!("{base}.out_proj"))?,
    })
}

/// Rearrange a conv1d weight [out, in, k] into an im2col GEMM matrix
/// [out, in*k] with the column order the host im2col produces
/// (frame-major: [in0k0, in1k0, ..] NO — kernel-major per input frame:
/// for tap t in 0..k, all `in` channels of frame (pos+t)). We emit
/// columns as (tap, channel): col = t*in + c.
fn conv_as_gemm(w: &[f32], out: usize, inp: usize, k: usize) -> Vec<f32> {
    let mut m = vec![0f32; out * inp * k];
    for o in 0..out {
        for c in 0..inp {
            for t in 0..k {
                m[o * (inp * k) + t * inp + c] = w[o * (inp * k) + c * k + t];
            }
        }
    }
    m
}

impl Whisper {
    /// The sidecar's GPU backend (for cross-thread binding and for
    /// releasing the transient FP32 load store).
    pub fn gpu_backend(&self) -> &dyn GpuBackend {
        self.gpu.as_ref()
    }

    /// Load whisper-large-v3 from an FP32 WeightStore onto `gpu` (a
    /// backend bound to the whisper kernel target). ~3.2 GiB BF16
    /// resident + ~350 MiB caches/scratch.
    pub fn load(store: &WeightStore, gpu: Box<dyn GpuBackend>) -> Result<Self> {
        let g = gpu.as_ref();
        let kern = Kernels {
            gemm: g.kernel("gemm", "dense_gemm_bf16")?,
            gemv: g.kernel("gemv", "dense_gemv_bf16")?,
            layernorm: g.kernel("whisper", "whisper_layernorm")?,
            bias_add: g.kernel("whisper", "whisper_bias_add")?,
            bias_gelu: g.kernel("whisper", "whisper_bias_gelu")?,
            attention: g.kernel("whisper", "whisper_attention")?,
            add_inplace: g.kernel("bf16_add", "bf16_add_inplace")?,
        };

        let c1 = f32_host(store, g, "model.encoder.conv1.weight")?;
        let c2 = f32_host(store, g, "model.encoder.conv2.weight")?;
        let conv1 = WB {
            w: upload_bf16(g, &conv_as_gemm(&c1, D_MODEL, N_MELS, 3))?,
            b: bf16(store, g, "model.encoder.conv1.bias")?,
        };
        let conv2 = WB {
            w: upload_bf16(g, &conv_as_gemm(&c2, D_MODEL, D_MODEL, 3))?,
            b: bf16(store, g, "model.encoder.conv2.bias")?,
        };

        let mut enc_layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            let p = format!("model.encoder.layers.{i}");
            enc_layers.push(EncLayer {
                ln1: wb(store, g, &format!("{p}.self_attn_layer_norm"))?,
                attn: attn(store, g, &format!("{p}.self_attn"))?,
                ln2: wb(store, g, &format!("{p}.final_layer_norm"))?,
                fc1: wb(store, g, &format!("{p}.fc1"))?,
                fc2: wb(store, g, &format!("{p}.fc2"))?,
            });
            tracing::debug!("whisper: encoder layer {i} loaded");
        }
        let mut dec_layers = Vec::with_capacity(N_LAYERS);
        let kv_self = N_TEXT_CTX * D_MODEL * 2;
        let kv_cross = N_AUDIO_CTX * D_MODEL * 2;
        for i in 0..N_LAYERS {
            let p = format!("model.decoder.layers.{i}");
            dec_layers.push(DecLayer {
                ln1: wb(store, g, &format!("{p}.self_attn_layer_norm"))?,
                self_attn: attn(store, g, &format!("{p}.self_attn"))?,
                ln2: wb(store, g, &format!("{p}.encoder_attn_layer_norm"))?,
                cross_attn: attn(store, g, &format!("{p}.encoder_attn"))?,
                ln3: wb(store, g, &format!("{p}.final_layer_norm"))?,
                fc1: wb(store, g, &format!("{p}.fc1"))?,
                fc2: wb(store, g, &format!("{p}.fc2"))?,
                k_cache: g.alloc(kv_self)?,
                v_cache: g.alloc(kv_self)?,
                k_cross: g.alloc(kv_cross)?,
                v_cross: g.alloc(kv_cross)?,
            });
            tracing::debug!("whisper: decoder layer {i} loaded");
        }

        let big = N_AUDIO_CTX * 2 * FFN_DIM.max(D_MODEL * 3) * 2;
        let scratch = Scratch {
            h_a: g.alloc(big)?,
            h_b: g.alloc(big)?,
            h_c: g.alloc(big)?,
            q: g.alloc(N_AUDIO_CTX * D_MODEL * 2)?,
            k: g.alloc(N_AUDIO_CTX * D_MODEL * 2)?,
            v: g.alloc(N_AUDIO_CTX * D_MODEL * 2)?,
            o: g.alloc(N_AUDIO_CTX * D_MODEL * 2)?,
            logits: g.alloc(VOCAB * 2)?,
            im2col: g.alloc(3000 * D_MODEL * 3 * 2)?,
        };

        let whisper = Self {
            kern,
            conv1,
            conv2,
            enc_pos: bf16(store, g, "model.encoder.embed_positions.weight")?,
            enc_ln: wb(store, g, "model.encoder.layer_norm")?,
            tok_emb: bf16(store, g, "model.decoder.embed_tokens.weight")?,
            dec_pos: bf16(store, g, "model.decoder.embed_positions.weight")?,
            dec_ln: wb(store, g, "model.decoder.layer_norm")?,
            dec_layers,
            enc_layers,
            enc_out: g.alloc(N_AUDIO_CTX * D_MODEL * 2)?,
            scratch,
            gpu,
        };
        tracing::info!(
            "Whisper sidecar loaded: {} enc + {} dec layers, d_model {D_MODEL}, BF16 resident",
            N_LAYERS,
            N_LAYERS
        );
        Ok(whisper)
    }
}

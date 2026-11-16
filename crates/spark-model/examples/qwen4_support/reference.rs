// SPDX-License-Identifier: AGPL-3.0-only
//! CPU reference of Qwen3.8-Flash-Next decoder layers, written from the
//! published model definition and independent of the GPU kernels. It reads
//! the weights of the checkpoint directly and computes in f64; values are
//! rounded to BF16 wherever the model's activations are BF16 tensors.

use anyhow::Result;

use super::{bf16_to_f32, e2m1, e4m3, f32_to_bf16, f32s, read_at, read_tensor, sigmoid, u16s};

pub const H: usize = 2560;
pub const HC: usize = 4;
pub const RANK: usize = 320;
pub const EPS: f64 = 1e-6;
const LM: &str = "model.language_model";

/// Round to the nearest BF16 value.
pub fn r16(v: f64) -> f64 {
    bf16_to_f32(f32_to_bf16(v as f32)) as f64
}
pub(super) fn r16v(v: Vec<f64>) -> Vec<f64> {
    v.into_iter().map(r16).collect()
}
pub(super) fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}
/// y = W x for a row-major `[n, k]` matrix.
pub(super) fn matvec(w: &[f32], n: usize, k: usize, x: &[f64]) -> Vec<f64> {
    (0..n)
        .map(|r| {
            w[r * k..(r + 1) * k]
                .iter()
                .zip(x)
                .map(|(a, b)| *a as f64 * b)
                .sum()
        })
        .collect()
}
/// RMS norm over the whole slice with weight (1 + w).
pub(super) fn rms1p(x: &[f64], w: &[f32]) -> Vec<f64> {
    let rs = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + EPS).sqrt();
    x.iter()
        .zip(w)
        .map(|(v, w)| v * rs * (1.0 + *w as f64))
        .collect()
}

pub(super) fn bf16_tensor(dir: &str, name: &str) -> Result<Vec<f32>> {
    Ok(u16s(&read_tensor(dir, name, "BF16")?.1)
        .into_iter()
        .map(bf16_to_f32)
        .collect())
}
pub(super) fn f32_tensor(dir: &str, name: &str) -> Result<Vec<f32>> {
    Ok(f32s(&read_tensor(dir, name, "F32")?.1))
}
/// MXFP8 weight as dense f32: e4m3(byte) * 2^(scale - 127), scale per 32 columns.
fn mxfp8(dir: &str, prefix: &str) -> Result<Vec<f32>> {
    let (shape, w) = read_tensor(dir, &format!("{prefix}.weight"), "F8_E4M3")?;
    let (_, s) = read_tensor(dir, &format!("{prefix}.weight_scale"), "U8")?;
    let k = shape[1];
    Ok(w.iter()
        .enumerate()
        .map(|(i, b)| {
            (e4m3(*b) * 2f64.powi(s[(i / k) * (k / 32) + (i % k) / 32] as i32 - 127)) as f32
        })
        .collect())
}
/// Routed-expert weight as dense f32 `[rows, in_dim]`:
/// e2m1(nibble) * e4m3(block scale) * weight_scale_2, scales csf-compressed.
fn nvfp4_csf(dir: &str, prefix: &str, rows: usize, in_dim: usize) -> Result<Vec<f32>> {
    let (_, packed) = read_tensor(dir, &format!("{prefix}.weight"), "U8")?;
    let (_, fixed) = read_tensor(dir, &format!("{prefix}.weight_scale.nvfp4_csf_fixed"), "U8")?;
    let (_, exc) = read_tensor(
        dir,
        &format!("{prefix}.weight_scale.nvfp4_csf_exceptions"),
        "U32",
    )?;
    let scale2 = f32s(&read_tensor(dir, &format!("{prefix}.weight_scale_2"), "F32")?.1)[0] as f64;
    let cols = in_dim / 16;
    let slab = 16 * (1 + cols / 2);
    let mut scales = vec![0u8; rows * cols];
    for r in 0..rows {
        let sl = &fixed[(r / 16) * slab..(r / 16 + 1) * slab];
        for c in 0..cols {
            let code = (sl[16 + (r % 16) * (cols / 2) + c / 2] >> (4 * (c & 1))) & 0xF;
            scales[r * cols + c] = sl[r % 16] + code;
        }
    }
    for rec in exc.chunks_exact(4) {
        let rec = u32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]);
        scales[(rec & 0xFF_FFFF) as usize] = (rec >> 24) as u8;
    }
    Ok((0..rows * in_dim)
        .map(|i| {
            let (r, c) = (i / in_dim, i % in_dim);
            let nib = (packed[r * (in_dim / 2) + c / 2] >> (4 * (c & 1))) & 0xF;
            (e2m1(nib) * e4m3(scales[r * cols + c / 16]) * scale2) as f32
        })
        .collect())
}

pub struct Mixer {
    norm: Vec<f32>,
    down: Vec<f32>,
    up: Vec<f32>,
    inject: Option<Vec<f32>>,
}
impl Mixer {
    pub fn load(dir: &str, prefix: &str, inject: bool) -> Result<Self> {
        Ok(Self {
            norm: f32_tensor(dir, &format!("{prefix}.hc_norm.weight"))?,
            down: bf16_tensor(dir, &format!("{prefix}.input_mix_weight_down.weight"))?,
            up: bf16_tensor(dir, &format!("{prefix}.input_mix_weight_up.weight"))?,
            inject: if inject {
                Some(bf16_tensor(
                    dir,
                    &format!("{prefix}.block_inject_weight.weight"),
                )?)
            } else {
                None
            },
        })
    }
    /// (sublayer input rounded to BF16, injection weights).
    pub fn mix(&self, streams: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut n = Vec::with_capacity(HC * H);
        for i in 0..HC {
            n.extend(rms1p(
                &streams[i * H..(i + 1) * H],
                &self.norm[i * H..(i + 1) * H],
            ));
        }
        let act: Vec<f64> = matvec(&self.down, RANK, HC * H, &n)
            .into_iter()
            .map(|v| silu(v / HC as f64))
            .collect();
        let gate = matvec(&self.up, HC * H, RANK, &act);
        let mixed = (0..H)
            .map(|d| {
                r16((0..HC)
                    .map(|i| sigmoid(gate[i * H + d]) * n[i * H + d])
                    .sum::<f64>()
                    / HC as f64)
            })
            .collect();
        let inj = self.inject.as_ref().map_or(Vec::new(), |w| {
            matvec(w, HC, HC * H, &n)
                .into_iter()
                .map(|v| 2.0 * sigmoid(v / HC as f64))
                .collect()
        });
        (mixed, inj)
    }
}

fn post(streams: &mut [f64], y: &[f64], inj: &[f64]) {
    for i in 0..HC {
        for d in 0..H {
            streams[i * H + d] += y[d] * inj[i];
        }
    }
}

const NK: usize = 16;
const NV: usize = 48;
const KD: usize = 128;
const KEY: usize = NK * KD;
const VAL: usize = NV * KD;
const CONV: usize = 2 * KEY + VAL;

pub struct Gdn {
    qkv: Vec<f32>,
    z: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    out: Vec<f32>,
    conv: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
    /// Last three conv inputs per channel, oldest first.
    hist: Vec<[f64; 3]>,
    /// Recurrent state `[NV][KD (key)][KD (value)]`.
    state: Vec<f64>,
}
impl Gdn {
    fn load(dir: &str, p: &str) -> Result<Self> {
        Ok(Self {
            qkv: mxfp8(dir, &format!("{p}.in_proj_qkv"))?,
            z: mxfp8(dir, &format!("{p}.in_proj_z"))?,
            a: mxfp8(dir, &format!("{p}.in_proj_a"))?,
            b: mxfp8(dir, &format!("{p}.in_proj_b"))?,
            out: mxfp8(dir, &format!("{p}.out_proj"))?,
            conv: bf16_tensor(dir, &format!("{p}.conv1d.weight"))?,
            a_log: bf16_tensor(dir, &format!("{p}.A_log"))?,
            dt_bias: bf16_tensor(dir, &format!("{p}.dt_bias"))?,
            norm: bf16_tensor(dir, &format!("{p}.norm.weight"))?,
            hist: vec![[0.0; 3]; CONV],
            state: vec![0.0; NV * KD * KD],
        })
    }
    fn forward(&mut self, x: &[f64]) -> Vec<f64> {
        let qkv = r16v(matvec(&self.qkv, CONV, H, x));
        let z = r16v(matvec(&self.z, VAL, H, x));
        let a = r16v(matvec(&self.a, NV, H, x));
        let b = r16v(matvec(&self.b, NV, H, x));
        // causal depthwise conv (kernel 4) + SiLU
        let mut c = vec![0.0f64; CONV];
        for ch in 0..CONV {
            let w = &self.conv[ch * 4..ch * 4 + 4];
            let h = &mut self.hist[ch];
            c[ch] = silu(
                w[0] as f64 * h[0]
                    + w[1] as f64 * h[1]
                    + w[2] as f64 * h[2]
                    + w[3] as f64 * qkv[ch],
            );
            *h = [h[1], h[2], qkv[ch]];
        }
        let l2 = |v: &mut [f64]| {
            let rs = 1.0 / (v.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt();
            v.iter_mut().for_each(|x| *x *= rs);
        };
        for hd in 0..2 * NK {
            l2(&mut c[hd * KD..(hd + 1) * KD]);
        }
        let (q, rest) = c.split_at(KEY);
        let (k, v) = rest.split_at(KEY);
        let mut core = vec![0.0f64; VAL];
        for vh in 0..NV {
            let kh = vh / (NV / NK);
            let (qh, kh_v) = (&q[kh * KD..(kh + 1) * KD], &k[kh * KD..(kh + 1) * KD]);
            let beta = sigmoid(b[vh]);
            let sp = (a[vh] + self.dt_bias[vh] as f64).exp().ln_1p();
            let decay = (-(self.a_log[vh] as f64).exp() * sp).exp();
            let s = &mut self.state[vh * KD * KD..(vh + 1) * KD * KD];
            s.iter_mut().for_each(|x| *x *= decay);
            for d in 0..KD {
                let mem: f64 = (0..KD).map(|i| s[i * KD + d] * kh_v[i]).sum();
                let delta = (v[vh * KD + d] - mem) * beta;
                for i in 0..KD {
                    s[i * KD + d] += kh_v[i] * delta;
                }
            }
            for d in 0..KD {
                core[vh * KD + d] =
                    (0..KD).map(|i| s[i * KD + d] * qh[i]).sum::<f64>() / (KD as f64).sqrt();
            }
        }
        // gated RMS norm per value head: w * rmsnorm(core) * sigmoid(z)
        let mut normed = vec![0.0f64; VAL];
        for vh in 0..NV {
            let o = &core[vh * KD..(vh + 1) * KD];
            let rs = 1.0 / (o.iter().map(|x| x * x).sum::<f64>() / KD as f64 + EPS).sqrt();
            for d in 0..KD {
                normed[vh * KD + d] =
                    r16(o[d] * rs * self.norm[d] as f64 * sigmoid(z[vh * KD + d]));
            }
        }
        r16v(matvec(&self.out, H, VAL, &normed))
    }
}

const NQ: usize = 24;
const NKV: usize = 2;
const HD: usize = 256;
const ROT: usize = 64;
const THETA: f64 = 1e7;

pub struct Attn {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    /// Cached keys / values per token: `[NKV * HD]` each.
    keys: Vec<Vec<f64>>,
    values: Vec<Vec<f64>>,
}
fn rope(x: &mut [f64], pos: usize) {
    for i in 0..ROT / 2 {
        let ang = pos as f64 * THETA.powf(-2.0 * i as f64 / ROT as f64);
        let (a, b) = (x[i], x[i + ROT / 2]);
        x[i] = r16(a * ang.cos() - b * ang.sin());
        x[i + ROT / 2] = r16(b * ang.cos() + a * ang.sin());
    }
}
impl Attn {
    fn load(dir: &str, p: &str) -> Result<Self> {
        Ok(Self {
            q: mxfp8(dir, &format!("{p}.q_proj"))?,
            k: mxfp8(dir, &format!("{p}.k_proj"))?,
            v: mxfp8(dir, &format!("{p}.v_proj"))?,
            o: mxfp8(dir, &format!("{p}.o_proj"))?,
            q_norm: bf16_tensor(dir, &format!("{p}.q_norm.weight"))?,
            k_norm: bf16_tensor(dir, &format!("{p}.k_norm.weight"))?,
            keys: Vec::new(),
            values: Vec::new(),
        })
    }
    fn forward(&mut self, x: &[f64], pos: usize) -> Vec<f64> {
        let qg = r16v(matvec(&self.q, 2 * NQ * HD, H, x));
        let mut k = r16v(matvec(&self.k, NKV * HD, H, x));
        let v = r16v(matvec(&self.v, NKV * HD, H, x));
        for h in 0..NKV {
            let n = r16v(rms1p(&k[h * HD..(h + 1) * HD], &self.k_norm));
            k[h * HD..(h + 1) * HD].copy_from_slice(&n);
            rope(&mut k[h * HD..(h + 1) * HD], pos);
        }
        self.keys.push(k);
        self.values.push(v);
        let mut gated = vec![0.0f64; NQ * HD];
        for h in 0..NQ {
            // per head [query | gate]
            let mut q = r16v(rms1p(&qg[h * 2 * HD..h * 2 * HD + HD], &self.q_norm));
            rope(&mut q, pos);
            let gate = &qg[h * 2 * HD + HD..(h + 1) * 2 * HD];
            let kvh = h / (NQ / NKV);
            let scores: Vec<f64> = self
                .keys
                .iter()
                .map(|k| {
                    q.iter()
                        .zip(&k[kvh * HD..(kvh + 1) * HD])
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
                        / (HD as f64).sqrt()
                })
                .collect();
            let mx = scores.iter().cloned().fold(f64::MIN, f64::max);
            let p: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
            let sum: f64 = p.iter().sum();
            for d in 0..HD {
                let o: f64 = p
                    .iter()
                    .zip(&self.values)
                    .map(|(p, v)| p / sum * v[kvh * HD + d])
                    .sum();
                gated[h * HD + d] = r16(r16(o) * sigmoid(gate[d]));
            }
        }
        r16v(matvec(&self.o, H, NQ * HD, &gated))
    }
}

const E: usize = 512;
const TOP_K: usize = 10;
const INTER: usize = 640;

pub struct Moe {
    dir: String,
    prefix: String,
    router: Vec<f32>,
    shared_gate: Vec<f32>,
    sh_gate: Vec<f32>,
    sh_up: Vec<f32>,
    sh_down: Vec<f32>,
    /// Experts routed to by the last call, best first.
    pub last_experts: Vec<usize>,
}
impl Moe {
    fn load(dir: &str, p: &str) -> Result<Self> {
        Ok(Self {
            dir: dir.to_string(),
            prefix: p.to_string(),
            router: bf16_tensor(dir, &format!("{p}.gate.weight"))?,
            shared_gate: bf16_tensor(dir, &format!("{p}.shared_expert_gate.weight"))?,
            sh_gate: mxfp8(dir, &format!("{p}.shared_expert.gate_proj"))?,
            sh_up: mxfp8(dir, &format!("{p}.shared_expert.up_proj"))?,
            sh_down: mxfp8(dir, &format!("{p}.shared_expert.down_proj"))?,
            last_experts: Vec::new(),
        })
    }
    fn forward(&mut self, x: &[f64]) -> Result<Vec<f64>> {
        let logits = r16v(matvec(&self.router, E, H, x));
        let mx = logits.iter().cloned().fold(f64::MIN, f64::max);
        let probs: Vec<f64> = logits.iter().map(|l| (l - mx).exp()).collect();
        let total: f64 = probs.iter().sum();
        let mut order: Vec<usize> = (0..E).collect();
        order.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap().then(a.cmp(b)));
        let top = &order[..TOP_K];
        let top_sum: f64 = top.iter().map(|e| probs[*e] / total).sum();
        let mut y = vec![0.0f64; H];
        for &e in top {
            let p = format!("{}.experts.{e}", self.prefix);
            let gate = r16v(matvec(
                &nvfp4_csf(&self.dir, &format!("{p}.gate_proj"), INTER, H)?,
                INTER,
                H,
                x,
            ));
            let up = r16v(matvec(
                &nvfp4_csf(&self.dir, &format!("{p}.up_proj"), INTER, H)?,
                INTER,
                H,
                x,
            ));
            let act: Vec<f64> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
            let down = r16v(matvec(
                &nvfp4_csf(&self.dir, &format!("{p}.down_proj"), H, INTER)?,
                H,
                INTER,
                &act,
            ));
            let w = probs[e] / total / top_sum;
            y.iter_mut().zip(&down).for_each(|(y, d)| *y += w * d);
        }
        self.last_experts = top.to_vec();
        let g = r16v(matvec(&self.sh_gate, INTER, H, x));
        let u = r16v(matvec(&self.sh_up, INTER, H, x));
        let act: Vec<f64> = g.iter().zip(&u).map(|(g, u)| r16(silu(*g) * u)).collect();
        let shared = r16v(matvec(&self.sh_down, H, INTER, &act));
        let sg = sigmoid(
            self.shared_gate
                .iter()
                .zip(x)
                .map(|(a, b)| *a as f64 * b)
                .sum(),
        );
        Ok((0..H).map(|d| r16(y[d] + sg * shared[d])).collect())
    }
}

pub enum Sub {
    Gdn(Box<Gdn>),
    Attn(Box<Attn>),
}

/// One decoder layer of the reference model with its per-sequence state.
pub struct Layer {
    ple: Option<super::reference_ple::Ple>,
    attn_mixer: Mixer,
    sub: Sub,
    mlp_mixer: Mixer,
    pub moe: Moe,
}
impl Layer {
    /// `is_attention` and `has_ple` come from the checkpoint's config.
    pub fn load(dir: &str, i: usize, is_attention: bool, has_ple: bool) -> Result<Self> {
        let lp = format!("{LM}.layers.{i}");
        Ok(Self {
            ple: if has_ple {
                Some(super::reference_ple::Ple::load(dir, &format!("{lp}.ple"))?)
            } else {
                None
            },
            attn_mixer: Mixer::load(dir, &format!("{lp}.attn_hyper_connection"), true)?,
            sub: if is_attention {
                Sub::Attn(Box::new(Attn::load(dir, &format!("{lp}.self_attn"))?))
            } else {
                Sub::Gdn(Box::new(Gdn::load(dir, &format!("{lp}.linear_attn"))?))
            },
            mlp_mixer: Mixer::load(dir, &format!("{lp}.mlp_hyper_connection"), true)?,
            moe: Moe::load(dir, &format!("{lp}.mlp"))?,
        })
    }
    /// Advances `streams` (`[HC * H]`) by this layer for the token at `pos`.
    /// `ctx` = (token, previous token, token before that), EOS before the start.
    pub fn forward(&mut self, streams: &mut [f64], pos: usize, ctx: (i64, i64, i64)) -> Result<()> {
        if let Some(ple) = &mut self.ple {
            ple.forward(streams, ctx.0, ctx.1, ctx.2)?;
        }
        let (x, inj) = self.attn_mixer.mix(streams);
        let y = match &mut self.sub {
            Sub::Gdn(g) => g.forward(&x),
            Sub::Attn(a) => a.forward(&x, pos),
        };
        post(streams, &y, &inj);
        let (x, inj) = self.mlp_mixer.mix(streams);
        let y = self.moe.forward(&x)?;
        post(streams, &y, &inj);
        Ok(())
    }
}

/// BF16 embedding row of `token`, as f64.
pub fn embedding(dir: &str, token: u32) -> Result<Vec<f64>> {
    let loc = super::locate(dir, &format!("{LM}.embed_tokens.weight"), "BF16")?;
    let b = read_at(&loc, token as u64 * (H as u64 * 2), H * 2)?;
    Ok(u16s(&b)
        .into_iter()
        .map(|x| bf16_to_f32(x) as f64)
        .collect())
}

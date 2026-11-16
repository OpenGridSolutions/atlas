// SPDX-License-Identifier: AGPL-3.0-only
//! CPU reference of the n-gram layer (PLE), split from `reference.rs`.

use anyhow::Result;

use super::reference::{H, HC, bf16_tensor, f32_tensor, matvec, r16, r16v, rms1p, silu};
use super::{e2m1, e4m3, read_at, read_tensor, sigmoid};

const HEADS: usize = 16;
const HEAD_DIM: usize = 160;
const EMB: usize = HEADS * HEAD_DIM;
const TAPS: usize = 4;
const DIL: usize = 3;
const EOS: i64 = 248044;

pub struct Ple {
    dir: String,
    prefix: String,
    key_proj: Vec<f32>,
    value_proj: Vec<f32>,
    conv: Vec<f32>,
    w_key: Vec<f32>,
    w_query: Vec<f32>,
    w_conv: Vec<f32>,
    scale2: f64,
    mult: Vec<i64>,
    sizes: Vec<i64>,
    offs: Vec<i64>,
    /// Normalised conv inputs of all earlier tokens, `[HC * H]` each.
    hist: Vec<Vec<f64>>,
}
impl Ple {
    pub fn load(dir: &str, p: &str) -> Result<Self> {
        let i64s = |n: &str| -> Result<Vec<i64>> {
            let b = read_tensor(dir, &format!("{p}.ple_embedding.{n}"), "I64")?.1;
            Ok(b.chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8])))
                .collect())
        };
        Ok(Self {
            dir: dir.to_string(),
            prefix: p.to_string(),
            key_proj: bf16_tensor(dir, &format!("{p}.key_proj.weight"))?,
            value_proj: bf16_tensor(dir, &format!("{p}.value_proj.weight"))?,
            conv: bf16_tensor(dir, &format!("{p}.conv1d.weight"))?,
            w_key: f32_tensor(dir, &format!("{p}.norm_key.weight"))?,
            w_query: f32_tensor(dir, &format!("{p}.norm_query.weight"))?,
            w_conv: f32_tensor(dir, &format!("{p}.norm_conv.weight"))?,
            scale2: f32_tensor(
                dir,
                &format!("{p}.ple_embedding.ngram_embedding.weight_scale_2"),
            )?[0] as f64,
            mult: i64s("layer_multipliers")?,
            sizes: i64s("ngram_heads_vocab_sizes")?,
            offs: i64s("ngram_heads_offsets")?,
            hist: Vec::new(),
        })
    }
    /// The token's 2560-wide n-gram embedding, read from the table on disk.
    fn embedding(&self, tok: i64, prev: i64, prev2: i64) -> Result<Vec<f64>> {
        let prev2 = if prev == EOS { EOS } else { prev2 };
        let bi = tok.wrapping_mul(self.mult[0]) ^ prev.wrapping_mul(self.mult[1]);
        let tri = bi ^ prev2.wrapping_mul(self.mult[2]);
        let table = format!("{}.ple_embedding.ngram_embedding", self.prefix);
        let shard0 = super::locate(&self.dir, &format!("{table}.shard_0.weight"), "U8")?;
        let shard_rows = shard0.shape[0] as i64;
        let mut emb = Vec::with_capacity(EMB);
        for h in 0..HEADS {
            let id =
                (if h < HEADS / 2 { bi } else { tri }).rem_euclid(self.sizes[h]) + self.offs[h];
            let (shard, row) = (id / shard_rows, (id % shard_rows) as u64);
            let w = super::locate(&self.dir, &format!("{table}.shard_{shard}.weight"), "U8")?;
            let s = super::locate(
                &self.dir,
                &format!("{table}.shard_{shard}.weight_scale"),
                "F8_E4M3",
            )?;
            let rows = read_at(&w, row * (HEAD_DIM as u64 / 2), HEAD_DIM / 2)?;
            let scales = read_at(&s, row * (HEAD_DIM as u64 / 16), HEAD_DIM / 16)?;
            for j in 0..HEAD_DIM {
                let nib = (rows[j / 2] >> (4 * (j & 1))) & 0xF;
                emb.push(r16(e2m1(nib) * e4m3(scales[j / 16]) * self.scale2));
            }
        }
        Ok(emb)
    }
    pub fn forward(&mut self, streams: &mut [f64], tok: i64, prev: i64, prev2: i64) -> Result<()> {
        let emb = self.embedding(tok, prev, prev2)?;
        let key = r16v(matvec(&self.key_proj, HC * H, EMB, &emb));
        let value = r16v(matvec(&self.value_proj, H, EMB, &emb));
        let (mut gated, mut conv_in) = (vec![0.0f64; HC * H], vec![0.0f64; HC * H]);
        for s in 0..HC {
            let r = s * H..(s + 1) * H;
            let kn = rms1p(&key[r.clone()], &self.w_key[r.clone()]);
            let qn = rms1p(&streams[r.clone()], &self.w_query[r.clone()]);
            let d = kn.iter().zip(&qn).map(|(a, b)| a * b).sum::<f64>() / (H as f64).sqrt();
            let g = sigmoid(d.abs().max(1e-6).sqrt() * if d == 0.0 { 0.0 } else { d.signum() });
            let gv: Vec<f64> = value.iter().map(|v| g * v).collect();
            conv_in[r.clone()].copy_from_slice(&rms1p(&gv, &self.w_conv[r.clone()]));
            gated[r].copy_from_slice(&gv);
        }
        self.hist.push(conv_in);
        let t = self.hist.len() - 1;
        for c in 0..HC * H {
            // taps at t - 9, t - 6, t - 3, t; missing history is zero
            let y: f64 = (0..TAPS)
                .filter_map(|k| {
                    t.checked_sub(DIL * (TAPS - 1 - k))
                        .map(|j| self.conv[c * TAPS + k] as f64 * self.hist[j][c])
                })
                .sum();
            streams[c] += gated[c] + silu(y);
        }
        Ok(())
    }
}

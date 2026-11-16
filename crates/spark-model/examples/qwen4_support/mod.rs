// SPDX-License-Identifier: AGPL-3.0-only
//! Shared helpers of the Qwen3.8-Flash-Next kernel microtests
//! (`qwen4_*_microtest.rs`): deterministic random numbers, BF16 / FP8 / FP4
//! scalar codecs, raw tensor access to a checkpoint snapshot, and the GPU
//! backend for the `qwen3.8-flash-next` kernel target.
#![allow(dead_code)] // each microtest uses a different subset

pub mod reference;
pub mod reference_ple;

use anyhow::{Context, Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::io::{Read, Seek, SeekFrom};

pub struct Rng(pub u64);
impl Rng {
    pub fn n(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn u(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.n() >> 40) as f32 / (1u64 << 24) as f32)
    }
    pub fn bf16s(&mut self, n: usize, lo: f32, hi: f32) -> Vec<u16> {
        (0..n).map(|_| f32_to_bf16(self.u(lo, hi))).collect()
    }
}

pub fn f32_to_bf16(f: f32) -> u16 {
    let b = f.to_bits();
    ((b.wrapping_add(0x7FFF + ((b >> 16) & 1))) >> 16) as u16
}
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
/// FP8 E4M3 (sign 1, exponent 4 bias 7, mantissa 3); NaN codes read as zero.
pub fn e4m3(b: u8) -> f64 {
    if b & 0x7F == 0x7F {
        return 0.0;
    }
    let (e, m) = (((b >> 3) & 0xF) as i32, (b & 7) as f64);
    let v = if e == 0 {
        m / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    if b & 0x80 != 0 { -v } else { v }
}
/// FP4 E2M1: magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6; bit 3 is the sign.
pub fn e2m1(n: u8) -> f64 {
    let v = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(n & 7) as usize];
    if n & 8 != 0 { -v } else { v }
}

pub fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}
pub fn u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
pub fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
pub fn le16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
pub fn le32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
pub fn le_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Where one tensor lives in a sharded safetensors snapshot.
pub struct TensorLoc {
    pub file: String,
    pub offset: u64,
    pub len: usize,
    pub shape: Vec<usize>,
}

/// Locates `name` through `model.safetensors.index.json` and checks its dtype.
pub fn locate(dir: &str, name: &str, dtype: &str) -> Result<TensorLoc> {
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(format!(
        "{dir}/model.safetensors.index.json"
    ))?)?;
    let shard = index["weight_map"][name]
        .as_str()
        .with_context(|| format!("{name} not in the index"))?;
    let file = format!("{dir}/{shard}");
    let mut f = std::fs::File::open(&file)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let hlen = u64::from_le_bytes(len);
    let mut hdr = vec![0u8; hlen as usize];
    f.read_exact(&mut hdr)?;
    let hdr: serde_json::Value = serde_json::from_slice(&hdr)?;
    let e = &hdr[name];
    if e["dtype"] != dtype {
        bail!("{name}: expected {dtype}, found {e}");
    }
    let shape = e["shape"]
        .as_array()
        .context("shape")?
        .iter()
        .map(|d| d.as_u64().unwrap_or(0) as usize)
        .collect();
    let (a, b) = (
        e["data_offsets"][0].as_u64().context("offset")?,
        e["data_offsets"][1].as_u64().context("offset")?,
    );
    Ok(TensorLoc {
        file,
        offset: 8 + hlen + a,
        len: (b - a) as usize,
        shape,
    })
}

/// `len` bytes of a tensor starting `skip` bytes into its data.
pub fn read_at(loc: &TensorLoc, skip: u64, len: usize) -> Result<Vec<u8>> {
    if skip as usize + len > loc.len {
        bail!("read past the end of a tensor in {}", loc.file);
    }
    let mut f = std::fs::File::open(&loc.file)?;
    let mut out = vec![0u8; len];
    f.seek(SeekFrom::Start(loc.offset + skip))?;
    f.read_exact(&mut out)?;
    Ok(out)
}

/// Shape and all bytes of one tensor.
pub fn read_tensor(dir: &str, name: &str, dtype: &str) -> Result<(Vec<usize>, Vec<u8>)> {
    let loc = locate(dir, name, dtype)?;
    let bytes = read_at(&loc, 0, loc.len)?;
    Ok((loc.shape, bytes))
}

/// Backend with the modules of the `qwen3.8-flash-next` kernel target.
pub fn backend() -> Result<AtlasCudaBackend> {
    let set = atlas_kernels::ptx_for_model("qwen3.8-flash-next").with_context(|| {
        format!(
            "kernel target qwen3.8-flash-next is not compiled in (build with \
             ATLAS_TARGET_MODEL=qwen3.8-flash-next). Compiled: {:?}",
            atlas_kernels::available_targets()
                .iter()
                .map(|t| t.target.to_string())
                .collect::<Vec<_>>()
        )
    })?;
    AtlasCudaBackend::new(0, &set.modules)
}

/// Device copy of `bytes`; empty input still gets an address.
pub fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(4))?;
    if !bytes.is_empty() {
        gpu.copy_h2d(bytes, p)?;
    }
    Ok(p)
}
pub fn download(gpu: &dyn GpuBackend, p: DevicePtr, len: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; len];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// Prints RESULT and exits non-zero on failure.
pub fn finish(ok: bool) -> Result<()> {
    if ok {
        println!("RESULT: PASS");
        Ok(())
    } else {
        println!("RESULT: FAIL");
        std::process::exit(1);
    }
}

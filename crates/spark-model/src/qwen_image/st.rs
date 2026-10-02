// SPDX-License-Identifier: AGPL-3.0-only

//! Minimal safetensors directory reader for the Qwen-Image sidecar.
//!
//! The diffusers layout (`transformer/diffusion_pytorch_model-0000x-of-…`,
//! `vae/diffusion_pytorch_model.safetensors`, `text_encoder/model-…`) is not
//! what `FastSafetensorsLoader::resolve_shards` recognizes, and that loader
//! puts EVERY tensor of a directory on the device — the sidecar wants a
//! subset (no `lm_head`, no VAE encoder, host-side embedding rows). This
//! reader indexes every `*.safetensors` in one directory and serves tensors
//! by name with positional reads.

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StDtype {
    Bf16,
    F32,
    I64,
}

impl StDtype {
    fn size(self) -> usize {
        match self {
            StDtype::Bf16 => 2,
            StDtype::F32 => 4,
            StDtype::I64 => 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StEntry {
    pub dtype: StDtype,
    pub shape: Vec<usize>,
    file: usize,
    /// Absolute byte offset of the tensor data in its file.
    offset: u64,
}

impl StEntry {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct StDir {
    files: Vec<File>,
    entries: HashMap<String, StEntry>,
}

pub fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

pub fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    (bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

pub fn f32s_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for v in vals {
        out.extend_from_slice(&f32_to_bf16_bits(*v).to_le_bytes());
    }
    out
}

pub fn bf16_bytes_to_f32s(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

pub fn f32s_to_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn bytes_to_f32s(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

impl StDir {
    /// Index every `*.safetensors` file directly inside `dir`.
    pub fn open(dir: &Path) -> Result<Self> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        paths.sort();
        ensure!(
            !paths.is_empty(),
            "no .safetensors files in {}",
            dir.display()
        );
        Self::open_files(&paths)
    }

    pub fn open_files<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        let mut files = Vec::with_capacity(paths.len());
        let mut entries = HashMap::new();
        for (idx, path) in paths.iter().enumerate() {
            let path = path.as_ref();
            let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
            let mut len = [0u8; 8];
            file.read_exact_at(&mut len, 0)?;
            let hdr_len = u64::from_le_bytes(len);
            ensure!(
                hdr_len < 256 << 20,
                "{}: implausible header size",
                path.display()
            );
            let mut hdr = vec![0u8; hdr_len as usize];
            file.read_exact_at(&mut hdr, 8)?;
            let json: serde_json::Value = serde_json::from_slice(&hdr)
                .with_context(|| format!("{}: safetensors header", path.display()))?;
            let map = json
                .as_object()
                .context("safetensors header is not an object")?;
            for (name, meta) in map {
                if name == "__metadata__" {
                    continue;
                }
                // Tensors in a dtype the sidecar never reads are not indexed.
                let dtype = match meta["dtype"].as_str() {
                    Some("BF16") => StDtype::Bf16,
                    Some("F32") => StDtype::F32,
                    Some("I64") => StDtype::I64,
                    _ => continue,
                };
                let shape: Vec<usize> = meta["shape"]
                    .as_array()
                    .context("shape")?
                    .iter()
                    .filter_map(|v| v.as_u64().map(|x| x as usize))
                    .collect();
                let begin = meta["data_offsets"][0].as_u64().context("data_offsets")?;
                let end = meta["data_offsets"][1].as_u64().context("data_offsets")?;
                let elems: usize = shape.iter().product();
                ensure!(
                    (end - begin) as usize == elems * dtype.size(),
                    "{name}: data size does not match shape"
                );
                entries.insert(
                    name.clone(),
                    StEntry {
                        dtype,
                        shape,
                        file: idx,
                        offset: 8 + hdr_len + begin,
                    },
                );
            }
            files.push(file);
        }
        Ok(Self { files, entries })
    }

    pub fn entry(&self, name: &str) -> Result<&StEntry> {
        self.entries
            .get(name)
            .with_context(|| format!("missing tensor '{name}'"))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }

    /// Raw little-endian tensor bytes in the checkpoint dtype.
    pub fn read_raw(&self, name: &str) -> Result<Vec<u8>> {
        let e = self.entry(name)?;
        let mut buf = vec![0u8; e.num_elements() * e.dtype.size()];
        self.files[e.file].read_exact_at(&mut buf, e.offset)?;
        Ok(buf)
    }

    /// Tensor as host f32 (BF16 widened, F32 as-is).
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let e = self.entry(name)?;
        let raw = self.read_raw(name)?;
        match e.dtype {
            StDtype::Bf16 => Ok(bf16_bytes_to_f32s(&raw)),
            StDtype::F32 => Ok(bytes_to_f32s(&raw)),
            StDtype::I64 => bail!("{name}: integer tensor requested as f32"),
        }
    }

    /// Tensor as BF16 little-endian bytes (F32 rounded to nearest-even).
    pub fn read_bf16(&self, name: &str) -> Result<Vec<u8>> {
        let e = self.entry(name)?;
        let raw = self.read_raw(name)?;
        match e.dtype {
            StDtype::Bf16 => Ok(raw),
            StDtype::F32 => Ok(f32s_to_bf16_bytes(&bytes_to_f32s(&raw))),
            StDtype::I64 => bail!("{name}: integer tensor requested as bf16"),
        }
    }

    pub fn read_i64(&self, name: &str) -> Result<Vec<i64>> {
        let e = self.entry(name)?;
        ensure!(e.dtype == StDtype::I64, "{name}: not an I64 tensor");
        Ok(self
            .read_raw(name)?
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    /// Selected rows of a 2-D BF16 tensor, concatenated (embedding lookup
    /// without loading the table).
    pub fn read_rows_bf16(&self, name: &str, rows: &[u32]) -> Result<Vec<u8>> {
        let e = self.entry(name)?;
        ensure!(
            e.dtype == StDtype::Bf16 && e.shape.len() == 2,
            "{name}: expected 2-D BF16"
        );
        let width = e.shape[1] * 2;
        let mut out = vec![0u8; rows.len() * width];
        for (i, &r) in rows.iter().enumerate() {
            ensure!((r as usize) < e.shape[0], "{name}: row {r} out of range");
            self.files[e.file].read_exact_at(
                &mut out[i * width..(i + 1) * width],
                e.offset + r as u64 * width as u64,
            )?;
        }
        Ok(out)
    }
}

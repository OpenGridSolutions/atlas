// SPDX-License-Identifier: AGPL-3.0-only

//! Row lookups in tables that stay on disk.
//!
//! Some checkpoints carry a lookup table that is far larger than what one
//! token reads from it (the n-gram embedding table of Qwen3.8-Flash-Next:
//! 26.9 GiB, 16 rows of 90 bytes per token). Such a table is not loaded;
//! its rows are read from the checkpoint files on demand.
//!
//! SBIO contract: model code never opens a file. It names rows as
//! [`RowRead`]s and gets their bytes from a [`RowSource`]. [`DirectRows`]
//! is the production source (O_DIRECT, so the page cache does not take back
//! the memory the table was kept out of); [`MemoryRows`] serves tests.

use anyhow::{Result, bail};
use std::path::PathBuf;

mod direct;
#[cfg(test)]
mod tests;

pub use direct::DirectRows;

/// One fixed-row-size tensor inside a file.
#[derive(Debug, Clone)]
pub struct RowPlane {
    /// File holding the tensor.
    pub path: PathBuf,
    /// Byte offset of the tensor's first row in the file.
    pub data_offset: u64,
    /// Bytes per row.
    pub row_bytes: usize,
    /// Number of rows.
    pub rows: u64,
}

/// One row to read: `plane` indexes the planes the source was opened with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowRead {
    pub plane: u32,
    pub row: u64,
}

/// The I/O boundary for on-disk tables.
pub trait RowSource: Send {
    /// Bytes per row of `plane`.
    fn row_bytes(&self, plane: u32) -> Result<usize>;

    /// Reads every requested row into `out`, concatenated in request order.
    /// `out` must be exactly as long as the rows together.
    fn read_rows(&mut self, reads: &[RowRead], out: &mut [u8]) -> Result<()>;
}

/// Total bytes of `reads`, checking planes and row indices.
fn checked_len(planes: &[(usize, u64)], reads: &[RowRead]) -> Result<usize> {
    let mut total = 0usize;
    for r in reads {
        let Some(&(row_bytes, rows)) = planes.get(r.plane as usize) else {
            bail!("row read names plane {} of {}", r.plane, planes.len());
        };
        if r.row >= rows {
            bail!("row {} outside plane {} ({rows} rows)", r.row, r.plane);
        }
        total += row_bytes;
    }
    Ok(total)
}

/// In-memory table for tests: plane = (row size, bytes).
pub struct MemoryRows {
    planes: Vec<(usize, Vec<u8>)>,
}

impl MemoryRows {
    pub fn new(planes: Vec<(usize, Vec<u8>)>) -> Result<Self> {
        for (i, (row_bytes, data)) in planes.iter().enumerate() {
            if *row_bytes == 0 || data.len() % row_bytes != 0 {
                bail!(
                    "plane {i}: {} bytes are not whole rows of {row_bytes}",
                    data.len()
                );
            }
        }
        Ok(Self { planes })
    }

    fn geometry(&self) -> Vec<(usize, u64)> {
        self.planes
            .iter()
            .map(|(rb, d)| (*rb, (d.len() / rb) as u64))
            .collect()
    }
}

impl RowSource for MemoryRows {
    fn row_bytes(&self, plane: u32) -> Result<usize> {
        match self.planes.get(plane as usize) {
            Some((rb, _)) => Ok(*rb),
            None => bail!("no plane {plane}"),
        }
    }

    fn read_rows(&mut self, reads: &[RowRead], out: &mut [u8]) -> Result<()> {
        if checked_len(&self.geometry(), reads)? != out.len() {
            bail!("row output buffer has the wrong length");
        }
        let mut at = 0;
        for r in reads {
            let (rb, data) = &self.planes[r.plane as usize];
            let start = r.row as usize * rb;
            out[at..at + rb].copy_from_slice(&data[start..start + rb]);
            at += rb;
        }
        Ok(())
    }
}

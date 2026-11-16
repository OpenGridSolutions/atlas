// SPDX-License-Identifier: AGPL-3.0-only

//! csf-compressed NVFP4 block scales (`weight_scale_encoding: "csf"`).
//!
//! A `[rows, cols]` matrix of E4M3 scale bytes is stored as
//!
//! * `fixed`: u8 `[rows / 16, 16 * (1 + cols / 2)]` — per 16-row slab the 16
//!   row base bytes, then per row `cols / 2` bytes of 4-bit codes (low nibble
//!   = even column); scale byte = base + code;
//! * `exceptions`: u32 `[n]` — low 24 bits `row * cols + col`, high 8 bits
//!   the exact byte, applied after base + code.
//!
//! Atlas keeps this form in memory and expands the routed experts per step
//! on the GPU (`csf_expand.cu`). This module holds the load-time checks and
//! the host decoder used by tests.

use anyhow::{Result, bail};

const SLAB_ROWS: usize = 16;
/// Largest base that cannot overflow a byte with a 4-bit code added.
const MAX_BASE: u8 = 240;
/// Positions are stored in 24 bits.
const MAX_POSITIONS: usize = 1 << 24;

/// Shape of the `fixed` tensor for a `[rows, cols]` scale matrix.
pub fn fixed_shape(rows: usize, cols: usize) -> [usize; 2] {
    [rows / SLAB_ROWS, SLAB_ROWS * (1 + cols / 2)]
}

/// Checks one compressed scale matrix against the geometry of its weight.
/// `fixed` and `exceptions` are the tensors' bytes (little-endian u32 records).
pub fn validate(
    name: &str,
    rows: usize,
    cols: usize,
    fixed: &[u8],
    exceptions: &[u8],
) -> Result<()> {
    if rows == 0 || !rows.is_multiple_of(SLAB_ROWS) || cols == 0 || !cols.is_multiple_of(2) {
        bail!("{name}: csf scales need rows % 16 == 0 and even cols, got [{rows}, {cols}]");
    }
    if rows * cols > MAX_POSITIONS {
        bail!("{name}: {rows} x {cols} scales exceed the 24-bit exception positions");
    }
    let [slabs, slab_len] = fixed_shape(rows, cols);
    if fixed.len() != slabs * slab_len {
        bail!(
            "{name}: csf fixed stream has {} bytes, expected {}",
            fixed.len(),
            slabs * slab_len
        );
    }
    for (s, slab) in fixed.chunks_exact(slab_len).enumerate() {
        if let Some(r) = slab[..SLAB_ROWS].iter().position(|b| *b > MAX_BASE) {
            bail!(
                "{name}: csf row base {} of row {} overflows a byte",
                slab[r],
                s * SLAB_ROWS + r
            );
        }
    }
    if !exceptions.len().is_multiple_of(4) {
        bail!("{name}: csf exception stream is not a whole number of u32 records");
    }
    let mut prev: Option<u32> = None;
    for rec in exceptions.chunks_exact(4) {
        let pos = u32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]) & 0x00FF_FFFF;
        if pos as usize >= rows * cols {
            bail!("{name}: csf exception position {pos} outside [{rows}, {cols}]");
        }
        if prev.is_some_and(|p| pos <= p) {
            bail!("{name}: csf exception positions are not strictly increasing at {pos}");
        }
        prev = Some(pos);
    }
    Ok(())
}

/// Host decoder: the `[rows, cols]` scale bytes, row-major. Input must have
/// passed [`validate`].
pub fn decode(rows: usize, cols: usize, fixed: &[u8], exceptions: &[u8]) -> Vec<u8> {
    let slab_len = fixed_shape(rows, cols)[1];
    let mut out = vec![0u8; rows * cols];
    for (r, row) in out.chunks_exact_mut(cols).enumerate() {
        let slab = &fixed[(r / SLAB_ROWS) * slab_len..][..slab_len];
        let base = slab[r % SLAB_ROWS];
        let codes = &slab[SLAB_ROWS + (r % SLAB_ROWS) * (cols / 2)..][..cols / 2];
        for (c, b) in row.iter_mut().enumerate() {
            *b = base + ((codes[c / 2] >> (4 * (c & 1))) & 0xF);
        }
    }
    for rec in exceptions.chunks_exact(4) {
        let rec = u32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]);
        out[(rec & 0x00FF_FFFF) as usize] = (rec >> 24) as u8;
    }
    out
}

#[cfg(test)]
#[path = "csf_tests.rs"]
mod tests;

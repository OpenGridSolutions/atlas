// SPDX-License-Identifier: AGPL-3.0-only

//! N-gram hashing of the Qwen3.8-Flash-Next embedding table (PLE).
//!
//! Every token reads `heads` table rows. For each n-gram order n = 2..=N a
//! group of heads hashes the current token and its n-1 predecessors:
//!
//! ```text
//! mix = tok[t]*m[0] ^ tok[t-1]*m[1] ^ ... ^ tok[t-n+1]*m[n-1]     (wrapping i64)
//! row = offset[head] + mix mod size[head]                          (non-negative)
//! ```
//!
//! Predecessors before the start of the sequence read as EOS, and once a
//! predecessor is EOS every older one reads as EOS too (n-grams do not cross
//! an end-of-sequence token). The multipliers, head sizes and offsets are
//! buffers of the checkpoint; they are read, not recomputed.

use anyhow::{Result, bail};

/// Hash constants of one n-gram table.
#[derive(Debug, Clone, PartialEq)]
pub struct NgramHasher {
    multipliers: Vec<i64>,
    head_sizes: Vec<i64>,
    head_offsets: Vec<i64>,
    heads_per_ngram: usize,
    eos_token: i64,
    rows_per_shard: i64,
}

/// Position of one table row in the split table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowAddress {
    /// Index of the `shard_<n>` tensor pair.
    pub shard: usize,
    /// Row inside that shard.
    pub row: usize,
}

impl NgramHasher {
    /// `multipliers` has one entry per n-gram position (`ngram_size`),
    /// `head_sizes`/`head_offsets` one per head. `rows_per_shard` is the row
    /// count of every `shard_<n>` tensor and `num_shards` their number.
    pub fn new(
        multipliers: Vec<i64>,
        head_sizes: Vec<i64>,
        head_offsets: Vec<i64>,
        heads_per_ngram: usize,
        eos_token: u32,
        rows_per_shard: usize,
        num_shards: usize,
    ) -> Result<Self> {
        let ngram_size = multipliers.len();
        if ngram_size < 2 || heads_per_ngram == 0 {
            bail!("n-gram table: ngram_size {ngram_size} / heads_per_ngram {heads_per_ngram}");
        }
        let heads = (ngram_size - 1) * heads_per_ngram;
        if head_sizes.len() != heads || head_offsets.len() != heads {
            bail!(
                "n-gram table: {heads} heads expected, checkpoint has {} sizes and {} offsets",
                head_sizes.len(),
                head_offsets.len()
            );
        }
        let mut next = 0i64;
        for (h, (&size, &offset)) in head_sizes.iter().zip(&head_offsets).enumerate() {
            if size <= 0 || offset != next {
                bail!(
                    "n-gram table: head {h} has size {size} at offset {offset}, expected offset {next}"
                );
            }
            next += size;
        }
        let capacity = (rows_per_shard as i64).saturating_mul(num_shards as i64);
        if rows_per_shard == 0 || next > capacity {
            bail!(
                "n-gram table: heads address {next} rows, the {num_shards} shards hold {capacity}"
            );
        }
        Ok(Self {
            multipliers,
            head_sizes,
            head_offsets,
            heads_per_ngram,
            eos_token: eos_token as i64,
            rows_per_shard: rows_per_shard as i64,
        })
    }

    /// Rows read per token.
    pub fn heads(&self) -> usize {
        self.head_sizes.len()
    }

    /// Predecessor tokens a lookup depends on.
    pub fn context_len(&self) -> usize {
        self.multipliers.len() - 1
    }

    /// Global row ids of the token at the end of `history` (`history` = the
    /// sequence's tokens up to and including the current one; only the last
    /// `context_len() + 1` are read). Appends `heads()` ids to `out`.
    pub fn row_ids(&self, history: &[u32], out: &mut Vec<i64>) -> Result<()> {
        let Some((&tok, before)) = history.split_last() else {
            bail!("n-gram lookup needs at least the current token");
        };
        let mut mix = (tok as i64).wrapping_mul(self.multipliers[0]);
        let mut cut = false; // an EOS predecessor was seen
        for pos in 1..self.multipliers.len() {
            let prev = match before.len().checked_sub(pos) {
                Some(i) if !cut => before[i] as i64,
                _ => self.eos_token,
            };
            cut |= prev == self.eos_token;
            mix ^= prev.wrapping_mul(self.multipliers[pos]);
            // Heads of order pos + 1 use the mix of the first pos + 1 tokens.
            let first = (pos - 1) * self.heads_per_ngram;
            for h in first..first + self.heads_per_ngram {
                out.push(self.head_offsets[h] + mix.rem_euclid(self.head_sizes[h]));
            }
        }
        Ok(())
    }

    /// Shard and row of a global row id.
    pub fn address(&self, row_id: i64) -> RowAddress {
        RowAddress {
            shard: (row_id / self.rows_per_shard) as usize,
            row: (row_id % self.rows_per_shard) as usize,
        }
    }
}

#[cfg(test)]
#[path = "ngram_tests.rs"]
mod tests;

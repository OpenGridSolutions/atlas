// SPDX-License-Identifier: AGPL-3.0-only

//! Auxiliary per-block planes of the paged KV cache.
//!
//! Some attention variants keep data per KV block besides keys and values
//! (the index keys of Qwen3.8-Flash-Next's sparse attention). An auxiliary
//! plane gives every layer `bytes_per_block` bytes for each physical KV
//! block, addressed by the same block ids, so the data shares the block's
//! lifetime and ownership.

use anyhow::{Result, bail};

use super::PagedKvCache;
use crate::gpu::{DevicePtr, GpuBackend};

impl PagedKvCache {
    /// Allocates the auxiliary plane of every layer. Call once, right after
    /// the cache is created; the caller budgets the memory.
    pub fn alloc_aux(&mut self, bytes_per_block: usize, gpu: &dyn GpuBackend) -> Result<()> {
        if bytes_per_block == 0 || !self.aux_pools.is_empty() {
            bail!("KV aux plane: needs a non-zero block size and may be allocated once");
        }
        for _ in 0..self.layers.len() {
            self.aux_pools
                .push(gpu.alloc(self.num_blocks * bytes_per_block)?);
        }
        self.aux_block_bytes = bytes_per_block;
        tracing::info!(
            "KV cache aux plane: {} blocks × {} layers × {} bytes/block = {:.2} GB",
            self.num_blocks,
            self.layers.len(),
            bytes_per_block,
            (self.num_blocks * self.layers.len() * bytes_per_block) as f64
                / (1024.0 * 1024.0 * 1024.0),
        );
        Ok(())
    }

    /// Auxiliary plane of `layer_idx`, if one was allocated.
    pub fn aux_pool_ptr(&self, layer_idx: usize) -> Option<DevicePtr> {
        self.aux_pools.get(layer_idx).copied()
    }

    /// Bytes per block of the auxiliary planes (0 without planes).
    pub fn aux_block_bytes(&self) -> usize {
        self.aux_block_bytes
    }

    /// Blocks that fit into `available_bytes` when every block also carries
    /// `aux_bytes_per_block` in each layer's auxiliary plane.
    pub fn compute_num_blocks_with_aux(
        config: &super::KvCacheConfig,
        available_bytes: usize,
        aux_bytes_per_block: usize,
    ) -> Result<usize> {
        let per_block =
            config.block_bytes_kv_all_layers() + config.num_layers * aux_bytes_per_block;
        if per_block == 0 {
            bail!("KV cache block size is zero");
        }
        Ok(available_bytes / per_block)
    }
}

// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::gpu::mock::MockGpuBackend;

fn test_config() -> KvCacheConfig {
    KvCacheConfig {
        block_size: 16,
        num_kv_heads: 2,
        head_dim: 256,
        num_layers: 12,
        dtype: KvCacheDtype::Bf16,
        layer_dtypes: Vec::new(),
        layer_dims: Vec::new(),
        cache_blocks_per_seq: None,
    }
}

#[test]
fn aux_plane_is_per_layer_and_budgeted_per_block() {
    let gpu = MockGpuBackend::new();
    let cfg = test_config();
    let kv_block = cfg.block_bytes_kv_all_layers();
    // 5120 aux bytes per block and layer shrink the block count accordingly.
    let budget = 100 * (kv_block + cfg.num_layers * 5120) + 17;
    assert_eq!(
        PagedKvCache::compute_num_blocks_with_aux(&cfg, budget, 5120).unwrap(),
        100
    );
    assert_eq!(
        PagedKvCache::compute_num_blocks_with_aux(&cfg, budget, 0).unwrap(),
        PagedKvCache::compute_num_blocks(&cfg, budget).unwrap()
    );

    let layers = cfg.num_layers;
    let mut kv = PagedKvCache::new(cfg, 8, &gpu).unwrap();
    assert!(kv.aux_pool_ptr(0).is_none());
    assert_eq!(kv.aux_block_bytes(), 0);
    kv.alloc_aux(5120, &gpu).unwrap();
    assert_eq!(kv.aux_block_bytes(), 5120);
    let ptrs: Vec<_> = (0..layers).map(|l| kv.aux_pool_ptr(l).unwrap()).collect();
    assert!(kv.aux_pool_ptr(layers).is_none());
    let unique: std::collections::HashSet<u64> = ptrs.iter().map(|p| p.0).collect();
    assert_eq!(unique.len(), layers, "every layer has its own plane");
    assert!(
        kv.alloc_aux(5120, &gpu).is_err(),
        "planes are allocated once"
    );
}

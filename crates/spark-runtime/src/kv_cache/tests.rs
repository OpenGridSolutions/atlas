// SPDX-License-Identifier: AGPL-3.0-only

// Module is gated by parent's `#[cfg(test)] mod tests;` declaration —
// no inner `#![cfg(test)]` needed (and nesting them is a duplicated
// attribute under recent rustc).

use super::*;
use crate::gpu::mock::MockGpuBackend;

fn test_config() -> KvCacheConfig {
    KvCacheConfig {
        block_size: 16,
        num_kv_heads: 2,
        head_dim: 256,
        num_layers: 12,
        dtype: KvCacheDtype::Fp8,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    }
}

#[test]
fn test_block_bytes_fp8() {
    let cfg = test_config();
    // 16 tokens * 2 heads * 256 dim * 1 byte = 8192
    assert_eq!(cfg.block_bytes(), 8192);
    assert_eq!(cfg.block_bytes_kv(), 16384);
}

#[test]
fn test_block_bytes_bf16() {
    let cfg = KvCacheConfig {
        dtype: KvCacheDtype::Bf16,
        ..test_config()
    };
    // 16 * 2 * 256 * 2 = 16384
    assert_eq!(cfg.block_bytes(), 16384);
}

#[test]
fn test_block_bytes_nvfp4() {
    let cfg = KvCacheConfig {
        dtype: KvCacheDtype::Nvfp4,
        ..test_config()
    };
    // data: 16 * 2 * 256 / 2 = 4096
    // scales: 16 * 2 * 256 / 16 = 512
    // total: 4608
    assert_eq!(cfg.block_bytes(), 4608);
    assert_eq!(cfg.nvfp4_data_bytes(), 4096);
    assert_eq!(cfg.nvfp4_scale_bytes(), 512);
}

#[test]
fn test_alloc_free() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(test_config(), 10, &gpu).unwrap();
    assert_eq!(cache.num_free_blocks(), 10);

    let b0 = cache.alloc_block().unwrap();
    let b1 = cache.alloc_block().unwrap();
    assert_ne!(b0, b1);
    assert_eq!(cache.num_free_blocks(), 8);

    cache.free_block(b0);
    assert_eq!(cache.num_free_blocks(), 9);
}

#[test]
fn test_exhaust() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(test_config(), 2, &gpu).unwrap();
    cache.alloc_block().unwrap();
    cache.alloc_block().unwrap();
    assert!(cache.alloc_block().is_err());
}

#[test]
fn test_compute_num_blocks() {
    let cfg = test_config();
    // Each block: 16384 bytes * 12 layers = 196608 bytes
    let n = PagedKvCache::compute_num_blocks(&cfg, 1_000_000).unwrap();
    assert_eq!(n, 1_000_000 / 196608); // = 5
}

#[test]
fn test_compute_num_blocks_nvfp4() {
    let cfg = KvCacheConfig {
        dtype: KvCacheDtype::Nvfp4,
        ..test_config()
    };
    // Each block: 4608 * 2 (K+V) * 12 layers = 110592 bytes
    let n = PagedKvCache::compute_num_blocks(&cfg, 1_000_000).unwrap();
    assert_eq!(n, 1_000_000 / 110592); // = 9
}

#[test]
fn test_ref_counting() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(test_config(), 5, &gpu).unwrap();

    let b0 = cache.alloc_block().unwrap();
    assert_eq!(cache.ref_count(b0), 1);
    assert_eq!(cache.num_free_blocks(), 4);

    // inc_ref: block shared with prefix cache
    cache.inc_ref(b0);
    assert_eq!(cache.ref_count(b0), 2);

    // dec_ref (sequence release): refcount 2→1, block NOT freed
    let freed = cache.dec_ref(b0);
    assert!(!freed);
    assert_eq!(cache.num_free_blocks(), 4); // still not free

    // dec_ref (cache eviction): refcount 1→0, block freed
    let freed = cache.dec_ref(b0);
    assert!(freed);
    assert_eq!(cache.num_free_blocks(), 5); // back in free pool
}

#[test]
fn test_try_alloc_block() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(test_config(), 2, &gpu).unwrap();

    assert!(cache.try_alloc_block().is_some());
    assert!(cache.try_alloc_block().is_some());
    assert!(cache.try_alloc_block().is_none()); // exhausted
}

#[test]
fn test_return_evicted_block() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(test_config(), 2, &gpu).unwrap();

    let b0 = cache.alloc_block().unwrap();
    let _b1 = cache.alloc_block().unwrap();
    assert_eq!(cache.num_free_blocks(), 0);

    // Simulate eviction: block returned directly
    cache.return_evicted_block(b0);
    assert_eq!(cache.num_free_blocks(), 1);
    assert_eq!(cache.ref_count(b0), 0);

    // Can re-allocate it
    let b2 = cache.alloc_block().unwrap();
    assert_eq!(b2, b0);
    assert_eq!(cache.ref_count(b2), 1);
}

#[test]
fn test_read_write_block_roundtrip() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(test_config(), 4, &gpu).unwrap();
    let b0 = cache.alloc_block().unwrap();

    // Write known pattern to K and V for layer 0.
    let stride = cache.block_stride_bytes();
    let k_data: Vec<u8> = (0..stride).map(|i| (i % 256) as u8).collect();
    let v_data: Vec<u8> = (0..stride).map(|i| ((i + 128) % 256) as u8).collect();
    cache.write_block(0, b0, &k_data, &v_data, &gpu).unwrap();

    // Read back and verify.
    let (k_out, v_out) = cache.read_block(0, b0, &gpu).unwrap();
    assert_eq!(k_out, k_data);
    assert_eq!(v_out, v_data);
}

#[test]
fn test_read_write_block_multiple_layers() {
    let gpu = MockGpuBackend::new();
    let cfg = test_config(); // 12 layers
    let mut cache = PagedKvCache::new(cfg, 4, &gpu).unwrap();
    let b0 = cache.alloc_block().unwrap();

    let stride = cache.block_stride_bytes();
    for layer in 0..12 {
        let k: Vec<u8> = vec![layer as u8; stride];
        let v: Vec<u8> = vec![(layer + 100) as u8; stride];
        cache.write_block(layer, b0, &k, &v, &gpu).unwrap();
    }
    for layer in 0..12 {
        let (k, v) = cache.read_block(layer, b0, &gpu).unwrap();
        assert!(k.iter().all(|&b| b == layer as u8));
        assert!(v.iter().all(|&b| b == (layer + 100) as u8));
    }
}

#[test]
fn test_num_layers() {
    let cfg = test_config();
    let gpu = MockGpuBackend::new();
    let cache = PagedKvCache::new(cfg, 2, &gpu).unwrap();
    assert_eq!(cache.num_layers(), 12);
}

#[test]
fn test_kv_cache_dtype_parse() {
    assert_eq!("fp8".parse::<KvCacheDtype>().unwrap(), KvCacheDtype::Fp8);
    assert_eq!("bf16".parse::<KvCacheDtype>().unwrap(), KvCacheDtype::Bf16);
    assert_eq!(
        "nvfp4".parse::<KvCacheDtype>().unwrap(),
        KvCacheDtype::Nvfp4
    );
    assert!("int8".parse::<KvCacheDtype>().is_err());
}

#[test]
fn test_dtype_for_layer_empty_fallback() {
    let cfg = test_config(); // layer_dtypes is empty, dtype is Fp8
    assert_eq!(cfg.dtype_for_layer(0), KvCacheDtype::Fp8);
    assert_eq!(cfg.dtype_for_layer(11), KvCacheDtype::Fp8);
}

#[test]
fn test_dtype_for_layer_mixed() {
    // 12 attention layers: first 2 BF16, last 2 BF16, middle 8 NVFP4
    let mut layer_dtypes = vec![KvCacheDtype::Nvfp4; 12];
    layer_dtypes[0] = KvCacheDtype::Bf16;
    layer_dtypes[1] = KvCacheDtype::Bf16;
    layer_dtypes[10] = KvCacheDtype::Bf16;
    layer_dtypes[11] = KvCacheDtype::Bf16;

    let cfg = KvCacheConfig {
        layer_dtypes,
        dtype: KvCacheDtype::Nvfp4,
        ..test_config()
    };

    assert_eq!(cfg.dtype_for_layer(0), KvCacheDtype::Bf16);
    assert_eq!(cfg.dtype_for_layer(1), KvCacheDtype::Bf16);
    assert_eq!(cfg.dtype_for_layer(2), KvCacheDtype::Nvfp4);
    assert_eq!(cfg.dtype_for_layer(5), KvCacheDtype::Nvfp4);
    assert_eq!(cfg.dtype_for_layer(9), KvCacheDtype::Nvfp4);
    assert_eq!(cfg.dtype_for_layer(10), KvCacheDtype::Bf16);
    assert_eq!(cfg.dtype_for_layer(11), KvCacheDtype::Bf16);
}

#[test]
fn test_block_bytes_for_layer_mixed() {
    let mut layer_dtypes = vec![KvCacheDtype::Nvfp4; 12];
    layer_dtypes[0] = KvCacheDtype::Bf16;
    layer_dtypes[11] = KvCacheDtype::Bf16;

    let cfg = KvCacheConfig {
        layer_dtypes,
        dtype: KvCacheDtype::Nvfp4,
        ..test_config()
    };

    // BF16 layer: 16 * 2 * 256 * 2 = 16384 bytes
    assert_eq!(cfg.block_bytes_for_layer(0), 16384);
    assert_eq!(cfg.block_bytes_for_layer(11), 16384);
    // NVFP4 layer: data(4096) + scales(512) = 4608 bytes
    assert_eq!(cfg.block_bytes_for_layer(1), 4608);
    assert_eq!(cfg.block_bytes_for_layer(5), 4608);
}

#[test]
fn test_block_bytes_kv_all_layers_uniform() {
    let cfg = test_config(); // 12 layers, FP8
    // FP8: 8192 bytes/block, K+V = 16384, × 12 layers = 196608
    assert_eq!(cfg.block_bytes_kv_all_layers(), 196608);
}

#[test]
fn test_block_bytes_kv_all_layers_mixed() {
    // 12 layers: 2 BF16 + 10 NVFP4
    let mut layer_dtypes = vec![KvCacheDtype::Nvfp4; 12];
    layer_dtypes[0] = KvCacheDtype::Bf16;
    layer_dtypes[11] = KvCacheDtype::Bf16;

    let cfg = KvCacheConfig {
        layer_dtypes,
        dtype: KvCacheDtype::Nvfp4,
        ..test_config()
    };

    // BF16: 16384 * 2 layers = 32768 (K only); K+V = 65536
    // NVFP4: 4608 * 10 layers = 46080 (K only); K+V = 92160
    // Total: 65536 + 92160 = 157696
    let expected = 2 * 16384 * 2 + 10 * 4608 * 2;
    assert_eq!(cfg.block_bytes_kv_all_layers(), expected);
}

#[test]
fn test_compute_num_blocks_mixed_dtype() {
    let mut layer_dtypes = vec![KvCacheDtype::Nvfp4; 12];
    layer_dtypes[0] = KvCacheDtype::Bf16;
    layer_dtypes[11] = KvCacheDtype::Bf16;

    let cfg = KvCacheConfig {
        layer_dtypes,
        dtype: KvCacheDtype::Nvfp4,
        ..test_config()
    };

    let bytes_per_block = cfg.block_bytes_kv_all_layers();
    let n = PagedKvCache::compute_num_blocks(&cfg, 1_000_000).unwrap();
    assert_eq!(n, 1_000_000 / bytes_per_block);
}

#[test]
fn test_mixed_dtype_pool_allocation() {
    let gpu = MockGpuBackend::new();
    let mut layer_dtypes = vec![KvCacheDtype::Fp8; 4];
    layer_dtypes[0] = KvCacheDtype::Bf16;
    layer_dtypes[3] = KvCacheDtype::Bf16;

    let cfg = KvCacheConfig {
        block_size: 16,
        num_kv_heads: 2,
        head_dim: 256,
        num_layers: 4,
        dtype: KvCacheDtype::Fp8,
        layer_dtypes,
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    };
    let cache = PagedKvCache::new(cfg, 4, &gpu).unwrap();

    // BF16 layers have larger block stride
    assert_eq!(cache.block_stride_bytes_for_layer(0), 16384); // BF16
    assert_eq!(cache.block_stride_bytes_for_layer(1), 8192); // FP8
    assert_eq!(cache.block_stride_bytes_for_layer(2), 8192); // FP8
    assert_eq!(cache.block_stride_bytes_for_layer(3), 16384); // BF16

    // dtype_for_layer returns correct per-layer dtype
    assert_eq!(cache.dtype_for_layer(0), KvCacheDtype::Bf16);
    assert_eq!(cache.dtype_for_layer(1), KvCacheDtype::Fp8);
    assert_eq!(cache.dtype_for_layer(3), KvCacheDtype::Bf16);
}

/// Phase 6.1.i: simulate the `--high-speed-swap` sliding-window
/// behavior — many alloc/free cycles per sequence, with the
/// production HBM cache sized to a tiny `cache_blocks_per_seq` cap.
/// Verifies the free list never under-runs and physical block IDs
/// are correctly recycled.
#[test]
fn sliding_window_recycles_blocks() {
    let gpu = MockGpuBackend::new();
    // Tiny pool: 2 sequences × 4 blocks/seq = 8 total physical blocks.
    let cfg = KvCacheConfig {
        cache_blocks_per_seq: Some(4),
        ..test_config()
    };
    let mut cache = PagedKvCache::new(cfg, 8, &gpu).unwrap();

    // Per-sequence sliding window: keep at most 4 HBM-resident blocks.
    // Simulate 2 sequences each writing 100 blocks.
    for seq_id in 0..2 {
        let mut block_table: Vec<u32> = Vec::new();
        for step in 0..100 {
            let blk = cache.alloc_block().unwrap_or_else(|_| {
                panic!("alloc failed at seq {seq_id} step {step}: no free blocks")
            });
            block_table.push(blk);
            // Slide window: when block_table > cap, evict the oldest.
            while block_table.len() > 4 {
                let evicted = block_table.remove(0);
                cache.free_block(evicted);
            }
        }
        // Free remaining blocks as if the sequence completed.
        for &blk in &block_table {
            cache.free_block(blk);
        }
        // After full free, all 4 should be back in the pool.
        assert!(
            cache.num_free_blocks() >= 4,
            "after seq {seq_id} all blocks freed: {} free",
            cache.num_free_blocks()
        );
    }
    // After both sequences finish, all 8 blocks are free.
    assert_eq!(cache.num_free_blocks(), 8);
}

/// Phase 6.3 sliding-window precondition: after `free_block` returns
/// a block to the pool, the very next `alloc_block` returns the same
/// block (LIFO). This is what makes the slide-then-alloc round-trip
/// in `ensure_blocks_through_decode` produce a fresh physical slot
/// without changing the pool size.
#[test]
fn alloc_after_free_round_trip_returns_same_block_lifo() {
    let gpu = MockGpuBackend::new();
    let cfg = test_config();
    let mut cache = PagedKvCache::new(cfg, 4, &gpu).unwrap();

    let b0 = cache.alloc_block().unwrap();
    let b1 = cache.alloc_block().unwrap();
    let b2 = cache.alloc_block().unwrap();
    let b3 = cache.alloc_block().unwrap();
    assert_eq!(cache.num_free_blocks(), 0);
    assert!(cache.alloc_block().is_err(), "pool exhausted as expected");

    // Free b0 → free list = [b0]. Next alloc must return b0.
    cache.free_block(b0);
    let recycled = cache.alloc_block().unwrap();
    assert_eq!(
        recycled, b0,
        "alloc-after-free must return the just-freed block (LIFO precondition for HSS slide)"
    );

    // Free b1, b2 in order → LIFO pops b2 first.
    cache.free_block(b1);
    cache.free_block(b2);
    assert_eq!(cache.alloc_block().unwrap(), b2);
    assert_eq!(cache.alloc_block().unwrap(), b1);

    assert_eq!(cache.num_free_blocks(), 0);
    cache.free_block(b3);
    cache.free_block(recycled);
}

// ── Phase M (Plan.md): KV-budget → block-count derivation per cache dtype ──
// The meminfo fix changes the BUDGET input; these tests pin down that the
// budget→blocks step stays exact for every cache type in use: uniform
// bf16/fp8/nvfp4, the production selective-boundary mix (6×bf16 + rest
// nvfp4 — the TranslateGemma/qwen3 pattern), asymmetric K/V dtypes, and
// heterogeneous per-layer dims (Gemma sliding vs full).

fn phase_m_config(dtype: KvCacheDtype, layer_dtypes: Vec<KvCacheDtype>) -> KvCacheConfig {
    KvCacheConfig {
        block_size: 16,
        num_kv_heads: 8,
        head_dim: 128,
        num_layers: 48,
        dtype,
        layer_dtypes,
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    }
}

#[test]
fn phase_m_blocks_are_exact_and_never_overcommit_per_dtype() {
    let budget = 8usize * 1024 * 1024 * 1024; // 8 GiB
    let mut counts = std::collections::HashMap::new();
    for dtype in [KvCacheDtype::Bf16, KvCacheDtype::Fp8, KvCacheDtype::Nvfp4] {
        let cfg = phase_m_config(dtype, vec![]);
        let bytes = cfg.block_bytes_kv_all_layers();
        let n = PagedKvCache::compute_num_blocks(&cfg, budget).unwrap();
        assert_eq!(n, budget / bytes, "{dtype:?}: floor division exactly");
        assert!(n * bytes <= budget, "{dtype:?}: pool must fit the budget");
        assert!((n + 1) * bytes > budget, "{dtype:?}: pool must be maximal");
        counts.insert(format!("{dtype:?}"), n);
    }
    // Cheaper dtypes buy strictly more blocks from the same budget.
    assert!(counts["Nvfp4"] > counts["Fp8"], "{counts:?}");
    assert!(counts["Fp8"] > counts["Bf16"], "{counts:?}");
    // Exact BYTE ratios (block counts floor-divide, so assert on bytes):
    // fp8 = bf16/2; nvfp4 = 0.5 data + 1/16 scale bytes/elem = 9/32 of bf16.
    let bf16_bytes = phase_m_config(KvCacheDtype::Bf16, vec![]).block_bytes_kv_all_layers();
    let fp8_bytes = phase_m_config(KvCacheDtype::Fp8, vec![]).block_bytes_kv_all_layers();
    let nvfp4_bytes = phase_m_config(KvCacheDtype::Nvfp4, vec![]).block_bytes_kv_all_layers();
    assert_eq!(fp8_bytes * 2, bf16_bytes);
    assert_eq!(
        nvfp4_bytes * 32,
        bf16_bytes * 9,
        "nvfp4 = 9/32 of bf16 (0.5625/2)"
    );
}

#[test]
fn phase_m_selective_boundary_mix_is_layer_sum() {
    // Production pattern from the failing TranslateGemma load: 6 boundary
    // layers at bf16, the remaining 42 at nvfp4 ("Selective boundary KV
    // cache: 6/48 attention layers at bf16, rest at nvfp4").
    let mut layer_dtypes = vec![KvCacheDtype::Nvfp4; 48];
    for slot in layer_dtypes.iter_mut().take(3) {
        *slot = KvCacheDtype::Bf16;
    }
    for slot in layer_dtypes.iter_mut().skip(45) {
        *slot = KvCacheDtype::Bf16;
    }
    let mixed = phase_m_config(KvCacheDtype::Nvfp4, layer_dtypes);
    let uniform_bf16 = phase_m_config(KvCacheDtype::Bf16, vec![]);
    let uniform_nvfp4 = phase_m_config(KvCacheDtype::Nvfp4, vec![]);

    // Identity: mixed total == Σ per-layer (K+V) bytes.
    let per_layer_sum: usize = (0..48)
        .map(|i| mixed.k_block_bytes_for_layer(i) + mixed.v_block_bytes_for_layer(i))
        .sum();
    assert_eq!(mixed.block_bytes_kv_all_layers(), per_layer_sum);
    // And equals 6 bf16-layers + 42 nvfp4-layers exactly.
    let bf16_layer = uniform_bf16.block_bytes_kv_all_layers() / 48;
    let nvfp4_layer = uniform_nvfp4.block_bytes_kv_all_layers() / 48;
    assert_eq!(
        mixed.block_bytes_kv_all_layers(),
        6 * bf16_layer + 42 * nvfp4_layer
    );

    // Same budget buys a block count strictly between the uniform extremes.
    let budget = 8usize * 1024 * 1024 * 1024;
    let n_mixed = PagedKvCache::compute_num_blocks(&mixed, budget).unwrap();
    let n_bf16 = PagedKvCache::compute_num_blocks(&uniform_bf16, budget).unwrap();
    let n_nvfp4 = PagedKvCache::compute_num_blocks(&uniform_nvfp4, budget).unwrap();
    assert!(
        n_bf16 < n_mixed && n_mixed < n_nvfp4,
        "{n_bf16} {n_mixed} {n_nvfp4}"
    );
    assert!(n_mixed * mixed.block_bytes_kv_all_layers() <= budget);
}

#[test]
fn phase_m_asymmetric_kv_dtype_sums_both_sides() {
    // Bf16KTurbo3V: K at 2 B/elem, V at 3-bit + per-group scale.
    let cfg = phase_m_config(KvCacheDtype::Bf16KTurbo3V, vec![]);
    let k = cfg.k_block_bytes_for_layer(0);
    let v = cfg.v_block_bytes_for_layer(0);
    assert!(k != v, "asym dtype must produce different K/V sizes");
    assert_eq!(cfg.block_bytes_kv_all_layers(), 48 * (k + v));
    let budget = 4usize * 1024 * 1024 * 1024;
    let n = PagedKvCache::compute_num_blocks(&cfg, budget).unwrap();
    assert!(n * cfg.block_bytes_kv_all_layers() <= budget);
    assert!((n + 1) * cfg.block_bytes_kv_all_layers() > budget);
}

#[test]
fn phase_m_heterogeneous_layer_dims_use_per_layer_geometry() {
    // Gemma-4 pattern: sliding layers 16×256, full layers 4×512 — the
    // budget derivation must price each layer at ITS geometry.
    let mut cfg = phase_m_config(KvCacheDtype::Bf16, vec![]);
    cfg.num_layers = 4;
    cfg.layer_dims = vec![(16, 256), (4, 512), (16, 256), (4, 512)];
    // (child test module: private `block_bytes_dims` is accessible)
    let sliding = cfg.block_bytes_dims(KvCacheDtype::Bf16, 16, 256);
    let full = cfg.block_bytes_dims(KvCacheDtype::Bf16, 4, 512);
    assert_eq!(
        cfg.block_bytes_kv_all_layers(),
        2 * (2 * sliding) + 2 * (2 * full),
        "2 sliding + 2 full layers, K+V each"
    );
}

// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]

pub mod buffers;
#[cfg(feature = "cuda")]
pub mod cublaslt;
// Metal/no-cuda builds get unreachable stubs so spark-model's unconditional
// references to these cuda-only entry points still resolve (compile-only).
pub mod crash;
#[cfg(not(feature = "cuda"))]
#[path = "cublaslt_metal_stub.rs"]
pub mod cublaslt;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
pub mod cuda_err;
#[cfg(feature = "cuda")]
pub mod cutlass;
#[cfg(not(feature = "cuda"))]
#[path = "cutlass_metal_stub.rs"]
pub mod cutlass;
#[cfg(unix)]
pub mod fast_weights;
#[cfg(feature = "cuda")]
pub mod flashinfer;
#[cfg(not(feature = "cuda"))]
#[path = "flashinfer_metal_stub.rs"]
pub mod flashinfer;
pub mod gpu;
pub mod kernel_args;
pub mod kernel_audit;
pub mod kv_cache;
pub mod kv_dequant;
pub mod kv_spill;
pub mod kv_stats;
pub mod meminfo;
#[cfg(feature = "metal")]
pub mod metal_backend;
pub mod prefix_cache;
pub mod radix_tree;
pub mod sampler;
#[cfg(unix)]
pub mod snapshot;
pub mod weights;
#[cfg(unix)]
pub mod weights_arena;

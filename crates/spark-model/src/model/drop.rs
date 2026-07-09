// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl Drop for TransformerModel {
    fn drop(&mut self) {
        self.destroy_captured_graphs();
        self.drop_pinned_staging();
    }
}

impl TransformerModel {
    /// Destroy every captured CUDA graph exec (`cuGraphExecDestroy`).
    ///
    /// Graph exec objects pin device memory (baked-in buffers, the graph
    /// structure itself). Nothing destroyed them before — irrelevant for a
    /// process-lifetime model, but a per-swap leak once hot-swap drops and
    /// rebuilds models repeatedly (Task 6 Phase 1.3 teardown; the
    /// 100-cycle MemFree-drift test watches exactly this).
    fn destroy_captured_graphs(&self) {
        let mut graphs: Vec<GraphHandle> = Vec::new();
        graphs.extend(self.decode_graph.lock().drain().map(|(_, g)| g));
        graphs.extend(self.batch_decode_graphs.lock().drain().map(|(_, g)| g));
        graphs.extend(self.verify2_graph.lock().drain().map(|(_, g)| g));
        graphs.extend(self.verify3_graph.lock().drain().map(|(_, g)| g));
        graphs.extend(self.verify4_graph.lock().drain().map(|(_, g)| g));
        graphs.extend(self.verify_kgamma_graph.lock().drain().map(|(_, g)| g));
        // GraphHandle(0) is the no-graph sentinel of backends without
        // capture support (mock/metal) — nothing to destroy.
        let mut destroyed = 0usize;
        let mut failed = 0usize;
        for g in graphs.into_iter().filter(|g| g.0 != 0) {
            match self.gpu.destroy_graph(g) {
                Ok(()) => destroyed += 1,
                Err(e) => {
                    failed += 1;
                    tracing::warn!("cuGraphExecDestroy failed on model drop: {e}");
                }
            }
        }
        if destroyed + failed > 0 {
            tracing::info!(
                "Model drop: destroyed {destroyed} captured CUDA graph execs ({failed} failed)"
            );
        }
    }
}

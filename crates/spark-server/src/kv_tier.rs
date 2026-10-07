// SPDX-License-Identifier: AGPL-3.0-only

//! NVMe KV tier glue (requirements §5, E3 `persist_mode = on_release`):
//! owns the per-model [`BlockStore`] + background [`StoreWriter`] and the
//! scheduler-side persist hooks.
//!
//! Lifecycle: one `KvTier` per loaded arena model (the store directory is
//! keyed by the model hash, the records by the KV layout hash). The
//! scheduler installs the active tier via [`set_active`] at start and on
//! every hot-swap load; `finish_sequence` calls [`KvTier::persist_sequence`]
//! on each sequence end (NVM-1, DRN-6), and the pre-unload drain calls
//! [`KvTier::flush`] bounded by `persist_timeout` (NVM-2, DRN-3).
//!
//! Enqueue rejections (queue at its byte limit, NVM-11) land in a bounded
//! retry buffer pumped once per scheduler iteration — the payload is
//! already host-resident, so retries cost no GPU work. Blocks that are
//! neither persisted nor retryable simply stay pinned in RAM by the KV-7
//! evict guard until pool pressure forces the spill path.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use spark_model::traits::{Model, SequenceState};
use spark_storage::block_store::{
    BlockStore, BlockStoreConfig, Kind, StoreStats, StoreWriter, WriteJob,
};

/// Cap on host bytes the retry buffer may pin beyond the writer queue.
const RETRY_MAX_BYTES: u64 = 512 << 20;

pub struct KvTier {
    store: Arc<BlockStore>,
    writer: StoreWriter,
    pub persist_timeout: Duration,
    /// BAT-7: restorable extensions below this are recomputed instead.
    pub restore_min_tokens: usize,
    retry: Mutex<RetryBuf>,
}

#[derive(Default)]
struct RetryBuf {
    jobs: VecDeque<WriteJob>,
    bytes: u64,
    dropped: u64,
}

/// Per-model identity the tier binds records to.
pub struct KvTierIdentity {
    pub model_hash: [u8; 32],
    pub layout_hash: [u8; 32],
}

/// Tier construction parameters, resolved from CLI flags by the caller.
pub struct KvTierParams {
    pub root: std::path::PathBuf,
    pub session_ttl: Duration,
    pub gc_interval: Duration,
    pub min_free_pct: u8,
    pub write_queue_max_bytes: u64,
    pub write_budget_per_day: Option<u64>,
    pub persist_timeout: Duration,
    pub restore_min_tokens: usize,
}

impl KvTier {
    /// Resolve the CLI flags into a tier for `identity`'s model. `None`
    /// when `--nvme-cache-dir` is unset.
    pub fn build_from_args(
        args: &crate::cli::ServeArgs,
        identity: &KvTierIdentity,
    ) -> Result<Option<Arc<KvTier>>> {
        let Some(root) = args.nvme_cache_dir.clone() else {
            return Ok(None);
        };
        match args.persist_mode.as_str() {
            "on_release" => {}
            "write_through" => anyhow::bail!(
                "--persist-mode write_through is a later stage (NVM-13); use on_release"
            ),
            other => anyhow::bail!("unknown --persist-mode '{other}' (want on_release)"),
        }
        let params = KvTierParams {
            root,
            session_ttl: Duration::from_secs(args.session_ttl_hours * 3600),
            gc_interval: Duration::from_secs(args.gc_interval_s.max(1)),
            min_free_pct: args.nvme_min_free_pct.min(90),
            write_queue_max_bytes: args.write_queue_max_mb.max(1) << 20,
            write_budget_per_day: args.nvme_write_budget_tb_day.map(|tb| (tb * 1e12) as u64),
            restore_min_tokens: args.nvme_restore_min_tokens,
            persist_timeout: Duration::from_secs(args.persist_timeout_s.max(1)),
        };
        let tier = KvTier::open(&params, identity)?;
        // Safe: Arc has a single owner right here.
        Ok(Some(tier))
    }

    pub fn open(params: &KvTierParams, identity: &KvTierIdentity) -> Result<Arc<KvTier>> {
        let store = Arc::new(
            BlockStore::open(BlockStoreConfig {
                root: params.root.clone(),
                model_hash: identity.model_hash,
                layout_hash: identity.layout_hash,
                session_ttl: params.session_ttl,
                min_free_pct: params.min_free_pct,
                write_budget_per_day: params.write_budget_per_day,
            })
            .context("open NVMe block store")?,
        );
        let writer = StoreWriter::spawn(
            Arc::clone(&store),
            params.write_queue_max_bytes,
            params.gc_interval,
            Duration::from_secs(60),
        );
        tracing::info!(
            root = %params.root.display(),
            entries = store.stats().entries,
            "NVMe KV tier active (persist_mode=on_release)"
        );
        Ok(Arc::new(KvTier {
            store,
            writer,
            persist_timeout: params.persist_timeout,
            restore_min_tokens: params.restore_min_tokens,
            retry: Mutex::new(RetryBuf::default()),
        }))
    }

    pub fn store(&self) -> &Arc<BlockStore> {
        &self.store
    }

    /// BAT-6 probe: how many CONTIGUOUS blocks starting at `keys[0]` the
    /// store holds (index-only, no I/O).
    pub fn probe_restore(&self, keys: &[[u8; 16]]) -> usize {
        keys.iter()
            .take_while(|k| self.store.contains(k, Kind::Kv))
            .count()
    }

    /// Staging fetch (runs on a blocking thread, NOT the scheduler):
    /// read `keys` in chain order, stopping at the first missing or
    /// invalid record (NVM-6 — the rest is recomputed). Returns
    /// `(logical_block, payload)` pairs.
    pub fn fetch_blocks(
        store: &BlockStore,
        start_block: usize,
        keys: &[[u8; 16]],
    ) -> Vec<(usize, Vec<u8>)> {
        let mut out = Vec::new();
        for (i, key) in keys.iter().enumerate() {
            match store.get(key, Kind::Kv) {
                Some(rec) => {
                    crate::metrics::NVME_READ_BYTES_TOTAL.inc_by(rec.payload.len() as u64);
                    out.push((start_block + i, rec.payload));
                }
                None => break,
            }
        }
        out
    }

    /// The KV-7 eviction guard bound to this tier's store.
    pub fn evict_guard(&self) -> spark_runtime::prefix_cache::EvictGuard {
        let store = Arc::clone(&self.store);
        Arc::new(move |key: &[u8; 16], is_tail: bool| {
            let kind = if is_tail { Kind::Tail } else { Kind::Kv };
            store.contains(key, kind)
        })
    }

    /// NVM-1 / DRN-6: persist every not-yet-persisted block of a finished
    /// (or swap-aborted) sequence, plus its tail record. Runs on the
    /// scheduler thread BEFORE `free_sequence` — the blocks must still be
    /// HBM-resident for the D2H export. Never blocks on disk: payloads go
    /// to the writer queue; rejections land in the retry buffer.
    pub fn persist_sequence(&self, model: &dyn Model, seq: &mut SequenceState) {
        seq.block_keys.extend_to(&seq.tokens);
        if !seq.block_keys.is_cacheable() || seq.hss_window_start() != 0 {
            return;
        }
        // Validity cap (mirrors the prefill stale-V cap): prefill wrote
        // KV for [0, kv_valid_tokens); decode wrote [prompt_len, len)
        // inline. A capped prefill leaves an unwritten gap — persist only
        // below it.
        let valid_tokens = if seq.kv_valid_tokens >= seq.prompt_len || seq.prompt_len == 0 {
            seq.tokens.len()
        } else {
            seq.kv_valid_tokens
        };
        let view = seq.block_keys.view(&seq.tokens, seq.tokens.len());
        let bs = view.block_size();
        if bs == 0 {
            return;
        }
        let keys = view.full_keys();
        let full = keys.len().min(seq.block_table.len()).min(valid_tokens / bs);

        for (i, key) in keys.iter().take(full).enumerate() {
            if self.store.contains(&key.0, Kind::Kv) {
                self.store.touch(&key.0);
                continue;
            }
            let payload = match model.export_kv_block(seq, i) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("kv-tier: export block {i} failed: {e:#}");
                    return; // successors would be orphans anyway
                }
            };
            let pred = if i == 0 { [0u8; 16] } else { keys[i - 1].0 };
            self.submit(WriteJob {
                kind: Kind::Kv,
                key: key.0,
                pred,
                token_count: bs as u32,
                payload,
            });
        }

        let hybrid = model.has_recurrent_state();

        // Interior SSM resume points (SSM-1/2/7): every block-aligned
        // Marconi snapshot of this sequence, each keyed by the chain key
        // of the KV block it sits behind. `pred` is that same key's KV
        // record, so the GC orphans a snapshot exactly when its KV chain
        // dies. The prompt-tail checkpoint serves a replay of the prompt
        // alone; a deeper decode checkpoint serves the next turn.
        if hybrid && full > 0 {
            match model.export_aligned_ssm_snapshots(seq) {
                Ok(snaps) => {
                    for (tokens, payload) in snaps {
                        let blk = tokens / bs;
                        if blk < 1 || blk > full {
                            continue;
                        }
                        let key = keys[blk - 1].0;
                        if self.store.contains(&key, Kind::Ssm) {
                            continue;
                        }
                        self.submit(WriteJob {
                            kind: Kind::Ssm,
                            key,
                            pred: key,
                            token_count: tokens as u32,
                            payload,
                        });
                    }
                }
                Err(e) => tracing::warn!("kv-tier: ssm snapshot export failed: {e:#}"),
            }
        }

        let tail_len = seq.tokens.len() % bs;
        if tail_len == 0 {
            // Block-aligned end: persist the live end state as an `.ssm`
            // record at the final chain key (SSM-3's aligned arm).
            if hybrid
                && full > 0
                && full == keys.len()
                && valid_tokens == seq.tokens.len()
                && !self.store.contains(&keys[full - 1].0, Kind::Ssm)
            {
                match model.export_ssm_state(seq) {
                    Ok(payload) if !payload.is_empty() => self.submit(WriteJob {
                        kind: Kind::Ssm,
                        key: keys[full - 1].0,
                        pred: keys[full - 1].0,
                        token_count: seq.tokens.len() as u32,
                        payload,
                    }),
                    Ok(_) => {}
                    Err(e) => tracing::warn!("kv-tier: ssm end-state export failed: {e:#}"),
                }
            }
            return;
        }

        // Tail record (SSM-3): partial KV block + live end SSM state in
        // one framed payload — `[u64 kv_len][kv][ssm]` (ssm empty for
        // attention-only models, SSM-8). Only when every full block is
        // valid — a tail behind an unwritten gap would be unreachable
        // anyway (its predecessors get GC'd as orphans).
        if valid_tokens == seq.tokens.len()
            && seq.block_table.len() > full
            && full == keys.len()
            && let Some(tail_key) = view.tail_key()
            && !self.store.contains(&tail_key.0, Kind::Tail)
        {
            let pred = if full == 0 {
                [0u8; 16]
            } else {
                keys[full - 1].0
            };
            let kv = match model.export_kv_block(seq, full) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("kv-tier: export tail block failed: {e:#}");
                    return;
                }
            };
            let ssm = if hybrid {
                model.export_ssm_state(seq).unwrap_or_else(|e| {
                    tracing::warn!("kv-tier: tail ssm export failed: {e:#}");
                    Vec::new()
                })
            } else {
                Vec::new()
            };
            let mut payload = Vec::with_capacity(8 + kv.len() + ssm.len());
            payload.extend_from_slice(&(kv.len() as u64).to_le_bytes());
            payload.extend_from_slice(&kv);
            payload.extend_from_slice(&ssm);
            self.submit(WriteJob {
                kind: Kind::Tail,
                key: tail_key.0,
                pred,
                token_count: tail_len as u32,
                payload,
            });
        }
    }

    fn submit(&self, job: WriteJob) {
        crate::metrics::NVME_WRITE_BYTES_TOTAL.inc_by(job.payload.len() as u64);
        if let Some(rejected) = self.writer.enqueue(job) {
            let mut retry = self.retry.lock().expect("retry lock");
            let bytes = rejected.payload.len() as u64;
            if retry.bytes + bytes > RETRY_MAX_BYTES {
                retry.dropped += 1;
                if retry.dropped.is_power_of_two() {
                    tracing::warn!(
                        dropped = retry.dropped,
                        "kv-tier: retry buffer full — block remains RAM-only \
                         until pool pressure evicts it unpersisted"
                    );
                }
                return;
            }
            retry.bytes += bytes;
            retry.jobs.push_back(rejected);
        }
    }

    /// Re-offer rejected jobs to the writer (called once per scheduler
    /// iteration; cheap when empty).
    pub fn pump_retries(&self) {
        let mut retry = self.retry.lock().expect("retry lock");
        while let Some(job) = retry.jobs.pop_front() {
            let bytes = job.payload.len() as u64;
            match self.writer.enqueue(job) {
                None => retry.bytes = retry.bytes.saturating_sub(bytes),
                Some(rejected) => {
                    retry.jobs.push_front(rejected);
                    break;
                }
            }
        }
    }

    /// NVM-2 / DRN-3: drain the write queue before a model unload, bounded
    /// by `persist_timeout`. Returns false when the deadline cut it short
    /// (only cache is lost, never results).
    pub fn flush(&self) -> bool {
        self.pump_retries();
        let ok = self.writer.flush(Instant::now() + self.persist_timeout);
        if !ok {
            tracing::warn!(
                timeout_s = self.persist_timeout.as_secs(),
                pending_mb = self.writer.pending_bytes() / (1 << 20),
                "kv-tier: persist_timeout exceeded before unload — remaining \
                 writes dropped (cache only, no results lost)"
            );
        }
        let _ = self.store.persist_index();
        ok
    }

    /// Telemetry: store stats + queue depth.
    pub fn stats(&self) -> (StoreStats, u64, u64) {
        (
            self.store.stats(),
            self.writer.pending_bytes(),
            self.writer.dropped_full(),
        )
    }
}

// ── Scheduler-thread active tier (installed at start + every hot-swap) ──

static ACTIVE: Mutex<Option<Arc<KvTier>>> = Mutex::new(None);

/// Install the tier for the currently loaded model (None = tier off or
/// model without a store).
pub fn set_active(tier: Option<Arc<KvTier>>) {
    *ACTIVE.lock().expect("kv-tier active lock") = tier;
}

/// Run `f` with the active tier, if any.
pub fn with_active<R>(f: impl FnOnce(&KvTier) -> R) -> Option<R> {
    let guard = ACTIVE.lock().expect("kv-tier active lock");
    guard.as_ref().map(|t| f(t))
}

#[cfg(test)]
#[path = "kv_tier_tests.rs"]
mod tests;

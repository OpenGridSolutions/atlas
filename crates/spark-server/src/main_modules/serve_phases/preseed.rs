// SPDX-License-Identifier: AGPL-3.0-only

//! Boot-time preseed orchestration (Task 6 Phase 2.2).
//!
//! For every registered model, ensure a hash-verified ATLASRAW snapshot
//! exists in `--snapshot-dir`: present + matching hash → skip (stays on
//! SSD); missing or stale → load from safetensors into the (empty) arena,
//! write the snapshot, reset the arena. Strictly one model in the arena at
//! a time — peak-safe because the 0.1 streaming migration bounds the load
//! peak at weights + one shard.

use std::sync::Arc;

use anyhow::{Context, Result, bail};

use crate::cli;
use crate::main_modules::hotswap::{HotswapCtx, RegisteredModel, SwapState};

/// Resolve a preseed id into its registry entry (revision-hashed). The
/// checkpoint must be resolvable (cached or downloadable) at boot.
pub(crate) fn registered_model(
    args: &cli::ServeArgs,
    model_id: &str,
) -> Result<RegisteredModel> {
    let model_dir = crate::model_resolver::resolve_model_dir(model_id, args.cache_dir.as_deref())
        .with_context(|| format!("resolving preseed model '{model_id}'"))?;
    let revision = model_dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "local".to_string());
    Ok(RegisteredModel {
        model_id: model_id.to_string(),
        model_hash: spark_runtime::snapshot::model_version_hash(model_id, &revision),
    })
}

/// True when `path` holds a snapshot whose full 32-byte model_version_hash
/// matches — the filename alone (4 hash bytes) is NOT trusted, per plan:
/// a stale snapshot after a kernel/format change must be regenerated.
fn snapshot_current(path: &std::path::Path, hash: &[u8; 32]) -> bool {
    if !path.exists() {
        return false;
    }
    match spark_runtime::snapshot::read_snapshot_meta(path) {
        Ok((header, _)) => header.model_version_hash == *hash,
        Err(e) => {
            tracing::warn!(
                "Preseed: snapshot {} unreadable/corrupt ({e:#}) — regenerating",
                path.display()
            );
            false
        }
    }
}

/// Ensure snapshots for every model in `models`. Models whose snapshot is
/// already current are skipped (this includes the boot model — its
/// snapshot is written from the live arena before this runs). The arena
/// MUST be empty on entry and is left empty.
pub(crate) fn preseed_missing_snapshots(
    args: &cli::ServeArgs,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    arena: &spark_runtime::weights_arena::WeightsArena,
    snapshot_dir: &std::path::Path,
    models: &[RegisteredModel],
) -> Result<()> {
    for m in models {
        let path = snapshot_dir.join(spark_runtime::snapshot::snapshot_file_name(
            &m.model_id,
            &m.model_hash,
        ));
        if snapshot_current(&path, &m.model_hash) {
            tracing::info!(
                "Preseed: '{}' snapshot current at {} — skipping init",
                m.model_id,
                path.display()
            );
            continue;
        }
        if arena.used() != 0 {
            bail!(
                "preseed invariant violated: arena not empty ({} B used) before loading '{}'",
                arena.used(),
                m.model_id
            );
        }
        tracing::info!(
            "Preseed: '{}' snapshot missing or stale — initializing from safetensors",
            m.model_id
        );
        let t0 = std::time::Instant::now();

        // Per-model config, mirroring the boot path's pre-load phases.
        let model_dir =
            crate::model_resolver::resolve_model_dir(&m.model_id, args.cache_dir.as_deref())?;
        let (mut config, _json) = super::load_model_config(&model_dir)?;
        super::apply_lm_head_dtype_override(args, &mut config)?;
        super::merge_sidecar_quant_config(&model_dir, &mut config);
        // Quant gate: refuse to preseed a model the kernel set cannot
        // serve — its snapshot would only ever produce a load-time error.
        let ptx_set = atlas_kernels::ptx_for_config(&config.model_type, config.hidden_size)
            .with_context(|| {
                format!("no compiled kernel target matches preseed model '{}'", m.model_id)
            })?;
        let model_quant = crate::main_modules::serve::canonicalize_model_quant(&config);
        if !crate::main_modules::serve::quant_pair_compatible(ptx_set.target.quant, &model_quant)
        {
            bail!(
                "preseed model '{}': kernel/model quant mismatch (kernel {} vs model {})",
                m.model_id,
                ptx_set.target.quant,
                model_quant,
            );
        }

        // Load weights (streamed) and migrate into the arena. Migration
        // frees the loader's allocations shard by shard (0.1), so peak =
        // weights + one shard.
        let mut store = super::load_weight_store(
            args,
            &config,
            &model_dir,
            gpu,
            0,
            1,
            args.oom_guard_mb * 1024 * 1024,
        )?;
        let need = store.total_bytes();
        if (need as f64 * 1.01) as usize > arena.capacity() {
            bail!(
                "preseed model '{}' ({:.2} GiB) does not fit the {:.1} GiB arena — raise \
                 --weights-arena-gb",
                m.model_id,
                need as f64 / (1024.0 * 1024.0 * 1024.0),
                arena.capacity() as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }
        let (manifest, mig) =
            spark_runtime::weights_arena::migrate_weight_store(gpu, arena, &mut store)?;
        drop(store);
        let stats = spark_runtime::snapshot::write_snapshot(&path, arena, &manifest, m.model_hash)
            .with_context(|| format!("writing preseed snapshot {}", path.display()))?;
        arena.reset();
        gpu.device_synchronize()?;
        tracing::info!(
            "Preseed: '{}' done in {:.1} s ({} tensors, {:.2} GiB migrated, snapshot {:.2} \
             GiB @ {}, O_DIRECT {})",
            m.model_id,
            t0.elapsed().as_secs_f64(),
            mig.tensors,
            mig.bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            stats.bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            path.display(),
            stats.o_direct,
        );
    }
    Ok(())
}

/// Full boot preseed sequence (runs after the scheduler thread is up,
/// before the HTTP server starts):
///
/// 1. Snapshot the BOOT model from the live arena (unless already current).
/// 2. Unload it through the scheduler control path (empty drain).
/// 3. For every other registered model: hash-verified snapshot or
///    safetensors→arena→snapshot→reset, one at a time.
/// 4. Assert the plan's post-condition: state `unloaded`, arena
///    `used_bytes == 0`, no model in the scheduler. The server then waits
///    idle; the first request naming a registered model lazy-loads it.
pub(crate) async fn run_boot_preseed(
    args: &cli::ServeArgs,
    ctx: &Arc<HotswapCtx>,
    boot_modules: &[(&'static str, &'static [u8])],
) -> Result<()> {
    let t0 = std::time::Instant::now();
    let active = ctx.active_model();

    // 1. Boot model's snapshot from the live arena (no reload needed).
    let boot_path = ctx.snapshot_path_for(&active.model_id, &active.model_hash);
    if snapshot_current(&boot_path, &active.model_hash) {
        tracing::info!(
            "Preseed: boot model '{}' snapshot current at {}",
            active.model_id,
            boot_path.display()
        );
    } else {
        ctx.snapshot_blocking()
            .context("preseed: snapshotting the boot model")?;
    }

    // 2. Unload through the scheduler (drain is empty — HTTP not started).
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    ctx.scheduler
        .push_control(crate::scheduler::ControlMsg::Unload { reply: reply_tx });
    match tokio::time::timeout(std::time::Duration::from_secs(120), reply_rx).await {
        Ok(Ok(Ok(_stats))) => {}
        Ok(Ok(Err(e))) => return Err(e.context("preseed: boot-model unload failed")),
        Ok(Err(_)) => bail!("preseed: unload reply channel dropped (scheduler died?)"),
        Err(_) => bail!("preseed: boot-model unload timed out after 120 s"),
    }
    // The scheduler sets NoModel just after replying; give it a moment.
    for _ in 0..100 {
        if ctx.state() == SwapState::NoModel {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    if ctx.state() != SwapState::NoModel {
        bail!(
            "preseed: scheduler did not reach state 'unloaded' (state: {})",
            ctx.state().as_str()
        );
    }

    // 3. Remaining models, one at a time in the (now empty) arena. Fresh
    // backend handle onto the boot kernel target — the boot `gpu` box was
    // consumed by build_model.
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(args.gpu_ordinal, boot_modules)?;
    preseed_missing_snapshots(
        args,
        &gpu,
        &ctx.arena,
        &ctx.snapshot_dir,
        &ctx.registry,
    )?;

    // 4. Post-condition (plan 2.2): assert it, loudly.
    if ctx.state() != SwapState::NoModel || ctx.arena.used() != 0 {
        bail!(
            "preseed post-condition violated: state={}, arena used={} B (want unloaded / 0)",
            ctx.state().as_str(),
            ctx.arena.used(),
        );
    }
    tracing::info!(
        "Preseed complete in {:.1} s: {} model(s) registered, server idle (state: unloaded, \
         arena used: 0) — first request naming a registered model loads it (~seconds)",
        t0.elapsed().as_secs_f64(),
        ctx.registry.len(),
    );
    Ok(())
}

// SPDX-License-Identifier: AGPL-3.0-only

//! Handler-side view onto the ACTIVE arena model's cache identity: the
//! chain parameters (seed, block size, vision-pad id) and a shared
//! handle on its RAM prefix cache. Published at boot and on every
//! hot-swap load, next to `kv_tier::set_active`.
//!
//! Consumers (the `/v1/atlas/session/state` probe) are READ-ONLY: they
//! build chains and call `peek_*` methods, never `lookup`/`insert` —
//! those mutate refcounts/LRU state and belong to the scheduler thread.
//! Queries must gate on the hotswap state being `Running`; between
//! unload and load the registry still holds the previous model's tree
//! (CPU-side metadata only — its GPU pool is gone).

use std::sync::{Arc, Mutex};

use spark_runtime::block_keys::BlockKey;
use spark_runtime::prefix_cache::PrefixCache;

/// Chain parameters of the active model (everything
/// `BlockKeyChain::new` needs besides the tokens).
#[derive(Clone)]
pub struct ChainParams {
    pub model_id: String,
    pub seed: BlockKey,
    pub block_size: usize,
    /// 0 = no vision tower (no token is a media placeholder).
    pub vision_pad_token_id: u32,
}

struct Registry {
    params: ChainParams,
    cache: Arc<dyn PrefixCache>,
}

static ACTIVE: Mutex<Option<Registry>> = Mutex::new(None);

/// Install the active model's cache identity (boot + every reload).
pub fn set_active(params: ChainParams, cache: Arc<dyn PrefixCache>) {
    *ACTIVE.lock().expect("cache registry lock") = Some(Registry { params, cache });
}

/// Chain params + cache handle of the active model, if published.
pub fn active() -> Option<(ChainParams, Arc<dyn PrefixCache>)> {
    let g = ACTIVE.lock().expect("cache registry lock");
    g.as_ref().map(|r| (r.params.clone(), Arc::clone(&r.cache)))
}

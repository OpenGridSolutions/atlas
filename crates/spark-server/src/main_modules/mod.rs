// SPDX-License-Identifier: AGPL-3.0-only

//! Sub-modules of `main.rs`, factored out to keep the binary entry-point file ≤500 LoC.

pub(crate) mod app_state;
pub(crate) mod cache_registry;
pub(crate) mod heartbeat;
pub(crate) mod hotswap;
pub(crate) mod kv_dtypes;
pub(crate) mod middleware;
pub(crate) mod serve;
pub(crate) mod serve_phases;
pub(crate) mod serve_router;
pub(crate) mod text_sidecar;

#[cfg(test)]
mod middleware_tests;
#[cfg(test)]
mod tests;

pub(crate) use app_state::{AppState, ModelRuntime};
pub(crate) use kv_dtypes::{auto_high_precision_layers, build_layer_kv_dtypes};
pub(crate) use serve::serve;

// SPDX-License-Identifier: AGPL-3.0-only

//! The `qwen4_exp` family (Qwen3.8-Flash-Next): checkpoint formats and the
//! weight loader.

pub mod csf;
pub mod inventory;
pub mod layers;
mod loader;
pub mod ngram;
mod tensors;

pub use loader::Qwen4ExpWeightLoader;

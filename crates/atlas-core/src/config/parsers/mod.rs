// SPDX-License-Identifier: AGPL-3.0-only

//! Per-model-family JSON parsers, split out of `config.rs` for file-size
//! budget.

mod deepseek_v4;
mod gemma3;
mod gemma4;
mod minimax;
mod mistral;
mod quantization;
mod qwen3_dense;
mod step3p7;
mod vision;
mod whisper;

pub(crate) use deepseek_v4::parse_deepseek_v4;
pub(crate) use gemma3::parse_gemma3_params;
pub(crate) use gemma4::parse_gemma4_params;
pub(crate) use minimax::parse_minimax_m2;
pub(crate) use qwen3_dense::{parse_qwen3_dense, parse_qwen3_vl_dense};
pub(crate) use whisper::parse_whisper;
pub use mistral::parse_mistral_params;
pub use quantization::parse_quantization_config;
pub(crate) use step3p7::parse_step3p7;
pub(crate) use vision::parse_vision_config;

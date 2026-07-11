// SPDX-License-Identifier: AGPL-3.0-only

//! `/v1/embeddings` request/response types (multiclass Plan.md 1.4, E2)
//! — OpenAI-compatible shape.

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    Single(String),
    Batch(Vec<String>),
}

impl EmbeddingsInput {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Batch(v) => v,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: EmbeddingsInput,
    /// Matryoshka truncation: keep the first N dimensions and
    /// re-normalize (Qwen3-Embedding supports 32..4096).
    #[serde(default)]
    pub dimensions: Option<usize>,
    /// Only "float" is supported (OpenAI default).
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// Non-standard: task instruction for the Qwen3-Embedding instruct
    /// prefix. Per the model card, QUERIES should carry an instruction
    /// ("Instruct: {task}\nQuery: {text}") while documents embed raw —
    /// so this is opt-in per request; omitted = raw text.
    #[serde(default)]
    pub instruction: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingObject {
    pub object: &'static str, // "embedding"
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingsUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
    pub object: &'static str, // "list"
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingsUsage,
}

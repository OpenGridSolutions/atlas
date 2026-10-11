// SPDX-License-Identifier: AGPL-3.0-only

//! `/v1/rerank` request/response types (multiclass Plan.md 1.4, R2).
//!
//! Jina/Cohere-compatible shape — the de-facto rerank API also served
//! by vLLM and TEI: `{model, query, documents[], top_n?}` in,
//! `{results: [{index, relevance_score}]}` out, results sorted by
//! descending score, `index` referring to the ORIGINAL documents order.

use serde::{Deserialize, Serialize};

use crate::api::multimodal_input::MultimodalItem;

#[derive(Debug, Deserialize)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    /// Plain strings and/or `{text?, image?}` objects (multiclass 2.3 —
    /// Qwen3-VL multimodal reranking).
    pub documents: Vec<MultimodalItem>,
    /// Return only the best `top_n` results (default: all).
    #[serde(default)]
    pub top_n: Option<usize>,
    /// Echo each result's document text (Jina `return_documents`).
    #[serde(default)]
    pub return_documents: bool,
    /// Task instruction for the Qwen3-Reranker prompt. Defaults to the
    /// model card's retrieval instruction; override for domain-specific
    /// reranking ("Given a legal question, retrieve statutes that…").
    #[serde(default)]
    pub instruction: Option<String>,
    /// Atlas extension: seconds the client accepts from submit to
    /// completion. The server estimates up front and answers 503
    /// `capacity_exceeded` when it cannot meet it; an accepted request's
    /// deadline is clamped to it.
    #[serde(default)]
    pub accepted_processing_time: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct RerankResult {
    /// Position of this document in the REQUEST's `documents` array.
    pub index: usize,
    pub relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<RerankDocument>,
}

#[derive(Debug, Serialize)]
pub struct RerankDocument {
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct RerankUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct RerankResponse {
    pub model: String,
    pub results: Vec<RerankResult>,
    pub usage: RerankUsage,
}

// SPDX-License-Identifier: AGPL-3.0-only

//! Native OCR pipeline (Todo 3.3c) — Rust port of datalab-to/chandra.
//!
//! The `/v1/ocr` endpoint (api/ocr.rs) drives the Chandra-2 checkpoint
//! through the in-process generative pipeline — no Python, no HTTP
//! side-car. Module map mirrors the upstream package:
//!
//! | upstream (Python)    | here          | status |
//! |----------------------|---------------|--------|
//! | chandra/prompts.py   | `prompts`     | ported 1:1 |
//! | chandra/input.py     | `input`       | TODO 3.3c (pdfium rendering, Lanczos min-dim upscale) |
//! | chandra/output.py    | `output`      | TODO 3.3c (layout-chunk parse, bbox crops, HTML→Markdown) |
//! | chandra/model/*.py   | — (replaced by the internal scheduler path) | n/a |
//!
//! Reference pipeline constants (chandra/settings.py @ master):
//! IMAGE_DPI=192, MIN_PDF_IMAGE_DIM=1024, MIN_IMAGE_DIM=1536,
//! MAX_OUTPUT_TOKENS=12384, BBOX_SCALE=1000, MAX_VLLM_RETRIES=6.

pub mod prompts;

/// Chandra reference default for generated tokens per page
/// (settings.MAX_OUTPUT_TOKENS).
pub const MAX_OUTPUT_TOKENS: usize = 12384;

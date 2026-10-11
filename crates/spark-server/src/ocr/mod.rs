// SPDX-License-Identifier: AGPL-3.0-only

//! Native OCR pipeline (Todo 3.3c) — Rust port of datalab-to/chandra.
//!
//! The `/v1/ocr` endpoint (api/ocr.rs) drives the Chandra-2 checkpoint
//! through the in-process generative pipeline — no Python, no HTTP
//! side-car. Module map mirrors the upstream package:
//!
//! | upstream (Python)       | here       | status |
//! |-------------------------|------------|--------|
//! | chandra/prompts.py      | `prompts`  | ported 1:1 |
//! | chandra/input.py        | `input`    | images ported (min-dim upscale, scale_to_fit, page ranges); PDF via pdfium pending (needs libpdfium in the image) |
//! | chandra/output.py       | `output`   | ported (layout chunks, filtered HTML, WebP bbox crops) |
//! | markdownify subclass    | `markdown` | subset converter (golden-diffed in 3.3d) |
//! | chandra/model/util.py   | here       | `detect_repeat_token` for the degeneration retry |
//! | chandra/model/vllm.py   | `pipeline` | per-page generation + degeneration retry, shared by /v1/ocr and /convert |
//!
//! Reference pipeline constants (chandra/settings.py @ master):
//! IMAGE_DPI=192, MIN_PDF_IMAGE_DIM=1024, MIN_IMAGE_DIM=1536,
//! MAX_OUTPUT_TOKENS=12384, BBOX_SCALE=1000, MAX_VLLM_RETRIES=6.

pub mod input;
pub mod markdown;
pub mod output;
pub mod pipeline;
pub mod prompts;

/// Chandra reference default for generated tokens per page
/// (settings.MAX_OUTPUT_TOKENS).
pub const MAX_OUTPUT_TOKENS: usize = 12384;

/// Chandra reference default retry budget (settings.MAX_VLLM_RETRIES).
pub const MAX_RETRIES: usize = 6;

/// chandra/model/util.py::detect_repeat_token — true when the text tail
/// is a short sequence repeated implausibly often (inverse scaling:
/// shorter sequences need more repeats). Operates on chars, like the
/// Python original operates on str positions.
pub fn detect_repeat_token(text: &str) -> bool {
    detect_repeat_token_with(text, 4, 500, 0, 3.0)
}

pub fn detect_repeat_token_with(
    text: &str,
    base_max_repeats: usize,
    window_size: usize,
    cut_from_end: usize,
    scaling_factor: f64,
) -> bool {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len().saturating_sub(cut_from_end);
    let chars = &chars[..len];
    for seq_len in 1..=(window_size / 2) {
        if seq_len > chars.len() {
            break;
        }
        let candidate = &chars[chars.len() - seq_len..];
        let max_repeats =
            (base_max_repeats as f64 * (1.0 + scaling_factor / seq_len as f64)) as usize;
        let mut repeat_count = 0usize;
        let mut pos = chars.len() as isize - seq_len as isize;
        while pos >= 0 {
            if &chars[pos as usize..pos as usize + seq_len] == candidate {
                repeat_count += 1;
                pos -= seq_len as isize;
            } else {
                break;
            }
        }
        if repeat_count > max_repeats {
            return true;
        }
    }
    false
}

/// The reference checks the raw output AND the output with the last 50
/// chars cut (a repeat can be masked by a trailing partial sequence).
pub fn has_degenerate_tail(raw: &str) -> bool {
    detect_repeat_token(raw)
        || (raw.chars().count() > 50 && detect_repeat_token_with(raw, 4, 500, 50, 3.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeat_detected() {
        let s = format!("preamble {}", "same row | ".repeat(40));
        assert!(detect_repeat_token(&s));
    }

    #[test]
    fn normal_text_clean() {
        let s = "A regular paragraph of prose, with varied content and no loops. \
                 Tables may repeat separators | a | b | but not endlessly.";
        assert!(!detect_repeat_token(s));
    }

    #[test]
    fn masked_tail_detected() {
        let s = format!("x {}{}", "pattern-block ".repeat(40), "and then an ending");
        assert!(
            has_degenerate_tail(&s),
            "cut_from_end=50 should expose the repeat"
        );
    }
}

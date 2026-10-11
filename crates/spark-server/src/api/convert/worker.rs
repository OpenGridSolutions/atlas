// SPDX-License-Identifier: AGPL-3.0-only

//! Background execution of one `/convert` job.
//!
//! Page count → page selection → result-cache lookup → page fan-out
//! (bounded by `--ocr-page-parallelism`, each page re-gates the OCR
//! model so a swap between pages is survived) → Datalab document.
//! Failed pages are isolated (`metadata.failed_pages`); a truncated or
//! degenerate page never reaches the result. The job fails as a whole
//! when more than `--ocr-max-failed-pages-pct` of its pages failed, when
//! the deadline passed, when the client cancelled, or when pdfium is
//! missing on the server.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use futures::StreamExt;

use super::ConvertService;
use super::cache::{cache_key, page_seed};
use super::params::ConvertParams;
use super::result_tree::{DocumentInputs, PageFailure, PageInput, PageNode, assemble_document};
use crate::main_modules::AppState;
use crate::main_modules::hotswap::ModelCategory;
use crate::ocr;

pub struct JobSpec {
    pub id: String,
    pub params: ConvertParams,
    pub cancel: Arc<AtomicBool>,
    pub deadline: Option<Instant>,
    pub submitted: Instant,
}

const PROMPT_TYPE: &str = "ocr_layout";
const JPEG_QUALITY: u8 = 85;

enum InputKind {
    Pdf,
    Image,
}

fn sniff(bytes: &[u8]) -> Option<InputKind> {
    if ocr::input::is_pdf(bytes) {
        Some(InputKind::Pdf)
    } else if image::guess_format(bytes).is_ok() {
        Some(InputKind::Image)
    } else {
        None
    }
}

/// Pages to process: `page_range` ∩ [0, count), then the first
/// `max_pages`. Empty when the range selects nothing.
pub fn select_pages(
    count: usize,
    page_range: Option<&str>,
    max_pages: Option<usize>,
) -> Vec<usize> {
    let mut pages: Vec<usize> = match page_range {
        Some(r) => ocr::input::parse_range_str(r)
            .unwrap_or_default()
            .into_iter()
            .filter(|&p| p < count)
            .collect(),
        None => (0..count).collect(),
    };
    if let Some(m) = max_pages {
        pages.truncate(m);
    }
    pages
}

/// Entry point spawned by the submit handler.
pub async fn run_job(state: Arc<AppState>, svc: Arc<ConvertService>, spec: JobSpec) {
    let id = spec.id.clone();
    let fail = |code: &str, msg: String| {
        tracing::warn!("convert {id}: failed ({code}): {msg}");
        svc.jobs.fail(&id, code, msg, Instant::now());
    };

    let Some(kind) = sniff(&spec.params.file) else {
        fail(
            "unprocessable_document",
            "file is neither a PDF nor a supported image".into(),
        );
        return;
    };
    let file = Arc::new(spec.params.file);
    let page_count = match kind {
        InputKind::Pdf => {
            let f = Arc::clone(&file);
            match tokio::task::spawn_blocking(move || ocr::input::pdf_page_count(&f)).await {
                Ok(Ok(n)) => n,
                Ok(Err(e)) if ocr::input::is_pdfium_missing(&e) => {
                    fail("pdfium_unavailable", format!("{e:#}"));
                    return;
                }
                Ok(Err(e)) => {
                    fail("unprocessable_document", format!("{e:#}"));
                    return;
                }
                Err(e) => {
                    fail("internal_error", format!("page count task failed: {e}"));
                    return;
                }
            }
        }
        InputKind::Image => 1,
    };
    let pages = select_pages(
        page_count,
        spec.params.page_range.as_deref(),
        spec.params.max_pages,
    );
    if pages.is_empty() {
        fail(
            "unprocessable_document",
            format!("document has {page_count} pages; the requested page_range selects none"),
        );
        return;
    }
    if pages.len() > svc.cfg.max_pages {
        fail(
            "document_too_large",
            format!(
                "{} pages selected, server limit is {} (--ocr-max-pages)",
                pages.len(),
                svc.cfg.max_pages
            ),
        );
        return;
    }
    let Some(model_id) = crate::api::ocr::resolve_ocr_model(&state) else {
        fail(
            "model_unavailable",
            "no OCR model registered (ATLAS_PRESEED_OCR)".into(),
        );
        return;
    };

    // Result cache (design doc §2b.1): a hit is byte-identical, no GPU.
    let key = cache_key(&file, &model_id, PROMPT_TYPE, &pages);
    if !spec.params.skip_cache
        && let Some(cache) = svc.cache.as_ref()
        && let Some(bytes) = cache.get(&key)
        && let Ok(mut doc) = serde_json::from_slice::<serde_json::Value>(&bytes)
    {
        doc["metadata"]["cached"] = serde_json::Value::Bool(true);
        if let Some(m) = doc["json"].get_mut("metadata") {
            m["cached"] = serde_json::Value::Bool(true);
        }
        tracing::info!("convert {id}: cache hit ({} pages)", pages.len());
        svc.jobs
            .complete(&id, doc, bytes.len() as u64, Instant::now());
        return;
    }

    svc.jobs.set_processing(&id);
    let is_pdf = matches!(kind, InputKind::Pdf);
    let results: Vec<Result<PageNode, PageFailure>> = futures::stream::iter(pages.clone())
        .map(|idx| {
            let ctx = PageCtx {
                state: Arc::clone(&state),
                file: Arc::clone(&file),
                is_pdf,
                model_id: model_id.clone(),
                seed: page_seed(&key, idx),
                cancel: Arc::clone(&spec.cancel),
                deadline: spec.deadline,
            };
            process_page(ctx, idx)
        })
        .buffer_unordered(svc.cfg.page_parallelism.max(1))
        .collect()
        .await;

    if spec.cancel.load(Ordering::Relaxed) {
        // DELETE / abandon / deadline sweep already set the terminal state.
        tracing::info!(
            "convert {id}: cancelled, {} page results dropped",
            results.len()
        );
        return;
    }
    let mut ok_pages = Vec::new();
    let mut failures = Vec::new();
    for r in results {
        match r {
            Ok(p) => ok_pages.push(p),
            Err(f) => failures.push(f),
        }
    }
    if let Some(f) = failures
        .iter()
        .find(|f| f.error_code == "pdfium_unavailable")
    {
        fail("pdfium_unavailable", f.error.clone());
        return;
    }
    if spec.deadline.is_some_and(|d| Instant::now() >= d)
        || failures.iter().any(|f| f.error_code == "deadline_exceeded")
    {
        fail(
            "deadline_exceeded",
            "accepted_processing_time exceeded".into(),
        );
        return;
    }
    if let Some(f) = failures
        .iter()
        .find(|f| f.error_code == "capacity_exceeded")
    {
        fail("capacity_exceeded", f.error.clone());
        return;
    }
    let failed_pct = failures.len() * 100 / pages.len();
    if !failures.is_empty() && failed_pct > usize::from(svc.cfg.max_failed_pages_pct) {
        let mut idxs: Vec<usize> = failures.iter().map(|f| f.page_idx).collect();
        idxs.sort_unstable();
        let first = &failures[0];
        fail(
            "incomplete_document",
            format!(
                "{} of {} pages failed (pages {:?}); first: {} ({})",
                failures.len(),
                pages.len(),
                idxs,
                first.error,
                first.error_code
            ),
        );
        return;
    }

    let doc = assemble_document(DocumentInputs {
        pages: ok_pages,
        failures: &failures,
        page_count: pages.len(),
        runtime_secs: spec.submitted.elapsed().as_secs_f64(),
        model_id: &model_id,
        cached: false,
    });
    let bytes = serde_json::to_vec(&doc).unwrap_or_default();
    if failures.is_empty()
        && let Some(cache) = svc.cache.as_ref()
    {
        cache.put(&key, &bytes);
    }
    tracing::info!(
        "convert {id}: complete, {} pages, {} failed, {:.1}s, {} KB",
        pages.len(),
        failures.len(),
        spec.submitted.elapsed().as_secs_f64(),
        bytes.len() / 1024
    );
    svc.jobs
        .complete(&id, doc, bytes.len() as u64, Instant::now());
}

struct PageCtx {
    state: Arc<AppState>,
    file: Arc<Vec<u8>>,
    is_pdf: bool,
    model_id: String,
    seed: u64,
    cancel: Arc<AtomicBool>,
    deadline: Option<Instant>,
}

fn failure(page_idx: usize, error_code: &'static str, error: String) -> PageFailure {
    PageFailure {
        page_idx,
        error_code,
        error,
    }
}

/// Gate → render → OCR → assemble for one page. A `model_swap` finish
/// (the arena swapped the OCR model out between pages) is retried once
/// after re-gating.
async fn process_page(ctx: PageCtx, idx: usize) -> Result<PageNode, PageFailure> {
    let cancelled = || ctx.cancel.load(Ordering::Relaxed);
    if cancelled() {
        return Err(failure(idx, "cancelled", "job cancelled".into()));
    }
    // Render once; the retry below only redoes the generation.
    let f = Arc::clone(&ctx.file);
    let is_pdf = ctx.is_pdf;
    let raw = tokio::task::spawn_blocking(move || -> anyhow::Result<image::RgbImage> {
        if is_pdf {
            ocr::input::render_pdf_page(&f, idx)
        } else {
            Ok(image::load_from_memory(&f)?.to_rgb8())
        }
    })
    .await
    .map_err(|e| failure(idx, "internal_error", format!("render task failed: {e}")))?
    .map_err(|e| {
        if ocr::input::is_pdfium_missing(&e) {
            failure(idx, "pdfium_unavailable", format!("{e:#}"))
        } else {
            failure(idx, "unprocessable_document", format!("page {idx}: {e:#}"))
        }
    })?;
    let page_image = ocr::input::load_image(raw, ocr::input::MIN_IMAGE_DIM);

    let mut attempt = 0u8;
    let out = loop {
        if cancelled() {
            return Err(failure(idx, "cancelled", "job cancelled".into()));
        }
        gate(&ctx, idx).await?;
        let rt = ctx.state.runtime();
        let Some(vcfg) = rt.vision_config.as_ref() else {
            return Err(failure(
                idx,
                "model_unavailable",
                format!("OCR model '{}' has no vision tower", ctx.model_id),
            ));
        };
        let prompt_text = ocr::prompts::prompt_for(PROMPT_TYPE).unwrap_or_default();
        let prompt_text: &str = &prompt_text;
        let res = ocr::pipeline::ocr_page(ocr::pipeline::PageJob {
            state: &ctx.state,
            rt: &rt,
            vcfg,
            prompt_text,
            page_image: &page_image,
            max_tokens: None,
            max_retries: ocr::MAX_RETRIES,
            seed_base: Some(ctx.seed),
            deadline: ctx.deadline,
            cancel: Some(&ctx.cancel),
        })
        .await;
        match res {
            Ok(o) => break o,
            Err(ocr::pipeline::PageError::Truncated { finish_reason, .. })
                if finish_reason == "model_swap" && attempt == 0 =>
            {
                attempt += 1;
                tracing::info!("convert page {idx}: model swapped mid-page, re-gating once");
                continue;
            }
            Err(ocr::pipeline::PageError::Truncated { finish_reason, .. })
                if finish_reason == "deadline" =>
            {
                return Err(failure(idx, "deadline_exceeded", "deadline passed".into()));
            }
            Err(e) => return Err(failure(idx, e.error_code(), e.to_string())),
        }
    };
    if out.degenerate {
        return Err(failure(
            idx,
            "degenerate_output",
            format!("page still degenerate after {} retries", out.retries),
        ));
    }

    // Layout + markdown + JPEG crops are CPU work — off the runtime.
    let html = out.html;
    tokio::task::spawn_blocking(move || {
        let chunks = ocr::output::parse_layout(&html, page_image.width(), page_image.height());
        super::result_tree::build_page(PageInput {
            page_idx: idx,
            chunks: &chunks,
            image: &page_image,
            include_headers_footers: false,
            jpeg_quality: JPEG_QUALITY,
        })
    })
    .await
    .map_err(|e| failure(idx, "internal_error", format!("assembly task failed: {e}")))
}

/// Ensure the OCR model is resident, bounded by the job deadline. The
/// gate's HTTP-shaped rejection is folded into a page failure.
async fn gate(ctx: &PageCtx, idx: usize) -> Result<(), PageFailure> {
    let wait = crate::api::lazy_swap::ensure_model_active(
        &ctx.state,
        &ctx.model_id,
        ModelCategory::Ocr,
        false,
        // "streaming" here means "I can wait"; the job has its own
        // deadline instead of the non-streaming WAIT-3 bound.
        true,
    );
    let res = match ctx.deadline {
        Some(d) => {
            match tokio::time::timeout(d.saturating_duration_since(Instant::now()), wait).await {
                Ok(r) => r,
                Err(_) => {
                    return Err(failure(
                        idx,
                        "deadline_exceeded",
                        "deadline passed while waiting for the model".into(),
                    ));
                }
            }
        }
        None => wait.await,
    };
    match res {
        Ok(_) => Ok(()),
        Err(resp) => {
            let status = resp.status();
            let body = axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap_or_default();
            let v: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
            let msg = v["error"]["message"]
                .as_str()
                .unwrap_or("model gate rejected the request")
                .to_string();
            let code = match status.as_u16() {
                503 | 429 => "model_unavailable",
                404 | 400 => "model_unavailable",
                _ => "internal_error",
            };
            Err(failure(idx, code, msg))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_selection() {
        assert_eq!(select_pages(5, None, None), vec![0, 1, 2, 3, 4]);
        assert_eq!(select_pages(5, None, Some(2)), vec![0, 1]);
        assert_eq!(select_pages(5, Some("1,3-9"), None), vec![1, 3, 4]);
        assert_eq!(select_pages(5, Some("7-9"), None), Vec::<usize>::new());
        assert_eq!(select_pages(5, Some("0-4"), Some(3)), vec![0, 1, 2]);
    }

    #[test]
    fn sniff_kinds() {
        assert!(matches!(sniff(b"%PDF-1.7 ..."), Some(InputKind::Pdf)));
        let png = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0];
        assert!(matches!(sniff(&png), Some(InputKind::Image)));
        assert!(sniff(b"hello world").is_none());
    }
}

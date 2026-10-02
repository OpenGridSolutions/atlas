// SPDX-License-Identifier: AGPL-3.0-only

//! Port of chandra/input.py and the image shaping from
//! chandra/model/util.py. PDF rendering uses pdfium-render over the
//! same libpdfium that pypdfium2 wraps upstream — identical rasterizer,
//! so page images match the reference. libpdfium.so must be on the
//! library search path (gb10 image: bblanchon/pdfium-binaries
//! linux-arm64 → /usr/lib/libpdfium.so).

use anyhow::{Context, Result, bail};
use base64::Engine;

/// chandra/settings.py: IMAGE_DPI — baseline render DPI for PDF pages.
pub const IMAGE_DPI: f64 = 192.0;

/// chandra/settings.py: MIN_PDF_IMAGE_DIM — a page's short side is
/// rendered to at least this many pixels.
pub const MIN_PDF_IMAGE_DIM: f64 = 1024.0;

/// chandra/settings.py: MIN_IMAGE_DIM — plain images below this minimum
/// dimension are Lanczos-upscaled so small scans still produce enough
/// vision tokens.
pub const MIN_IMAGE_DIM: u32 = 1536;

/// Decode an `image` request field: full `data:` URI or bare base64.
/// The image format is sniffed from the bytes.
pub fn decode_request_image(field: &str) -> Result<image::RgbImage> {
    let b64 = match field.strip_prefix("data:") {
        Some(rest) => rest
            .split_once("base64,")
            .map(|(_, payload)| payload)
            .context("data: URI without base64 payload")?,
        None => field,
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .context("invalid base64 image payload")?;
    let img = image::load_from_memory(&bytes).context("undecodable image bytes")?;
    Ok(img.to_rgb8())
}

/// chandra/input.py::load_image — RGB + min-dimension Lanczos upscale.
pub fn load_image(img: image::RgbImage, min_image_dim: u32) -> image::RgbImage {
    let (w, h) = (img.width(), img.height());
    if w >= min_image_dim && h >= min_image_dim {
        return img;
    }
    let scale = min_image_dim as f64 / w.min(h) as f64;
    let new_w = (w as f64 * scale) as u32;
    let new_h = (h as f64 * scale) as u32;
    image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Lanczos3)
}

/// chandra/model/util.py::scale_to_fit — fit the image into the pixel
/// budget the model was trained at, on a grid raster, preferring the
/// resize direction that preserves aspect ratio. Defaults mirror the
/// reference: max (3072, 2048), min (1792, 28), grid 28.
pub fn scale_to_fit(img: image::RgbImage) -> image::RgbImage {
    scale_to_fit_with(img, (3072, 2048), (1792, 28), 28)
}

pub fn scale_to_fit_with(
    img: image::RgbImage,
    max_size: (u32, u32),
    min_size: (u32, u32),
    grid_size: u32,
) -> image::RgbImage {
    let (width, height) = (img.width(), img.height());
    if width == 0 || height == 0 {
        return img;
    }
    let original_ar = width as f64 / height as f64;
    let current_pixels = (width as u64) * (height as u64);
    let max_pixels = (max_size.0 as u64) * (max_size.1 as u64);
    let min_pixels = (min_size.0 as u64) * (min_size.1 as u64);

    let mut scale = 1.0f64;
    if current_pixels > max_pixels {
        scale = (max_pixels as f64 / current_pixels as f64).sqrt();
    } else if current_pixels < min_pixels {
        scale = (min_pixels as f64 / current_pixels as f64).sqrt();
    }

    let g = grid_size as f64;
    let mut w_blocks = ((width as f64 * scale) / g).round().max(1.0) as u64;
    let mut h_blocks = ((height as f64 * scale) / g).round().max(1.0) as u64;

    let gsq = (grid_size as u64) * (grid_size as u64);
    while w_blocks * h_blocks * gsq > max_pixels {
        if w_blocks == 1 && h_blocks == 1 {
            break;
        }
        if w_blocks == 1 {
            h_blocks -= 1;
            continue;
        }
        if h_blocks == 1 {
            w_blocks -= 1;
            continue;
        }
        let ar_w_loss = ((w_blocks - 1) as f64 / h_blocks as f64 - original_ar).abs();
        let ar_h_loss = (w_blocks as f64 / (h_blocks - 1) as f64 - original_ar).abs();
        if ar_w_loss < ar_h_loss {
            w_blocks -= 1;
        } else {
            h_blocks -= 1;
        }
    }

    let new_width = (w_blocks as u32) * grid_size;
    let new_height = (h_blocks as u32) * grid_size;
    if (new_width, new_height) == (width, height) {
        return img;
    }
    image::imageops::resize(
        &img,
        new_width,
        new_height,
        image::imageops::FilterType::Lanczos3,
    )
}

/// True when the decoded request payload is a PDF (reference sniffs
/// via `filetype`; the `%PDF-` magic is equivalent for our purposes).
pub fn is_pdf(bytes: &[u8]) -> bool {
    bytes.starts_with(b"%PDF-")
}

/// Decode the request field to raw bytes (data: URI or bare base64)
/// without assuming an image format — used for the PDF sniff.
pub fn decode_request_bytes(field: &str) -> Result<Vec<u8>> {
    let b64 = match field.strip_prefix("data:") {
        Some(rest) => rest
            .split_once("base64,")
            .map(|(_, payload)| payload)
            .context("data: URI without base64 payload")?,
        None => field,
    };
    base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .context("invalid base64 payload")
}

/// PDFium is not thread-safe; every render call binds + runs under this
/// process-wide lock inside `spawn_blocking`.
static PDFIUM_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// chandra/input.py::load_pdf_images — render selected pages (0-based
/// indices; empty range = all) at max(IMAGE_DPI, short-side→1024px)
/// with flattened forms/annotations, RGB.
pub fn load_pdf_images(bytes: &[u8], page_range: &[usize]) -> Result<Vec<(usize, image::RgbImage)>> {
    use pdfium_render::prelude::*;
    let _guard = PDFIUM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let pdfium = Pdfium::new(
        Pdfium::bind_to_system_library().context(
            "libpdfium.so not found — install the bblanchon/pdfium-binaries \
             linux-arm64 build to /usr/lib (see the gb10 Dockerfile)",
        )?,
    );
    let doc = pdfium
        .load_pdf_from_byte_slice(bytes, None)
        .context("unreadable PDF")?;
    let mut out = Vec::new();
    for (idx, mut page) in doc.pages().iter().enumerate() {
        if !page_range.is_empty() && !page_range.contains(&idx) {
            continue;
        }
        let (w_pt, h_pt) = (page.width().value as f64, page.height().value as f64);
        let min_page_dim = w_pt.min(h_pt).max(1.0);
        let scale_dpi = ((MIN_PDF_IMAGE_DIM / min_page_dim) * 72.0).max(IMAGE_DPI);
        // Reference flattens forms/annotations before rendering
        // (FPDFPage_Flatten); a failure is logged, not fatal.
        if let Err(e) = page.flatten() {
            tracing::warn!("PDF page {idx}: flatten failed ({e}) — rendering as-is");
        }
        let rendered = page
            .render_with_config(
                &PdfRenderConfig::new().scale_page_by_factor((scale_dpi / 72.0) as f32),
            )
            .with_context(|| format!("rendering PDF page {idx}"))?;
        out.push((idx, rendered.as_image().to_rgb8()));
    }
    Ok(out)
}

/// chandra/input.py::parse_range_str — "0,2-5,9" → sorted dedup'd page
/// indices.
pub fn parse_range_str(range_str: &str) -> Result<Vec<usize>> {
    let mut pages = std::collections::BTreeSet::new();
    for part in range_str.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            let s: usize = start.trim().parse().context("invalid page_range start")?;
            let e: usize = end.trim().parse().context("invalid page_range end")?;
            if e < s {
                bail!("page_range '{part}': end before start");
            }
            pages.extend(s..=e);
        } else {
            pages.insert(part.parse().context("invalid page_range entry")?);
        }
    }
    Ok(pages.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_str_mixed() {
        assert_eq!(parse_range_str("0,2-5,9,3").unwrap(), vec![0, 2, 3, 4, 5, 9]);
    }

    #[test]
    fn scale_to_fit_downscales_to_budget() {
        let img = image::RgbImage::new(4000, 3000);
        let out = scale_to_fit(img);
        assert!((out.width() as u64) * (out.height() as u64) <= 3072 * 2048);
        assert_eq!(out.width() % 28, 0);
        assert_eq!(out.height() % 28, 0);
    }

    #[test]
    fn scale_to_fit_upscales_small() {
        let img = image::RgbImage::new(200, 100);
        let out = scale_to_fit(img);
        assert!((out.width() as u64) * (out.height() as u64) >= 1792 * 28);
    }

    #[test]
    fn min_dim_upscale() {
        let img = image::RgbImage::new(800, 600);
        let out = load_image(img, MIN_IMAGE_DIM);
        assert_eq!(out.height(), 1536);
        assert_eq!(out.width(), 2048);
    }
}

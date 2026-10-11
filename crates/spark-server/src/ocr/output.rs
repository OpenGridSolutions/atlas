// SPDX-License-Identifier: AGPL-3.0-only

//! Port of chandra/output.py — layout-chunk parsing, filtered-HTML
//! assembly, and Image/Figure bbox crops.
//!
//! The reference mutates a BeautifulSoup DOM; `scraper`'s DOM is
//! read-only, so the (flat, tag-whitelisted) model output is adjusted
//! with targeted string surgery instead. That is safe here because the
//! model emits only the Chandra tag/attribute subset — `data-bbox`
//! always appears as a plain double-quoted attribute and `<img>` tags
//! are simple void elements.

use md5::{Digest, Md5};
use scraper::{ElementRef, Html};

/// chandra/settings.py: BBOX_SCALE — data-bbox coordinates are
/// normalized 0..=1000.
pub const BBOX_SCALE: u32 = 1000;

#[derive(Clone, serde::Serialize)]
pub struct LayoutBlock {
    pub label: String,
    /// Pixel coordinates on the (upscaled) input image, x0 y0 x1 y1.
    pub bbox: [u32; 4],
    pub content: String,
}

fn md5_hex(s: &str) -> String {
    let mut h = Md5::new();
    h.update(s.as_bytes());
    format!("{:x}", h.finalize())
}

/// chandra/output.py::get_image_name.
fn image_name(html: &str, div_idx: usize) -> String {
    format!("{}_{div_idx}_img.webp", md5_hex(html))
}

/// Top-level `<div>` elements of the model output fragment.
fn top_level_divs(fragment: &Html) -> Vec<ElementRef<'_>> {
    fragment
        .root_element()
        .children()
        .filter_map(ElementRef::wrap)
        .filter(|el| el.value().name() == "div")
        .collect()
}

/// Strip `data-bbox="..."` attributes from nested content (the
/// reference deletes them from every nested tag before emitting chunk
/// content).
fn strip_nested_bboxes(content: &str) -> String {
    let mut out = String::with_capacity(content.len());
    let mut rest = content;
    while let Some(i) = rest.find(" data-bbox=\"") {
        out.push_str(&rest[..i]);
        let after = &rest[i + " data-bbox=\"".len()..];
        match after.find('"') {
            Some(j) => rest = &after[j + 1..],
            None => {
                rest = after;
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

/// chandra/output.py::parse_layout — top-level divs → labeled blocks
/// with pixel bboxes (invalid bboxes default to [0,0,1,1] pre-scale,
/// like the reference). `Blank-Page` divs are dropped.
pub fn parse_layout(html: &str, img_w: u32, img_h: u32) -> Vec<LayoutBlock> {
    let fragment = Html::parse_fragment(html);
    let width_scaler = img_w as f64 / BBOX_SCALE as f64;
    let height_scaler = img_h as f64 / BBOX_SCALE as f64;
    let mut blocks = Vec::new();
    for div in top_level_divs(&fragment) {
        let label = div.value().attr("data-label").unwrap_or("");
        if label == "Blank-Page" {
            continue;
        }
        let raw_bbox: Vec<i64> = div
            .value()
            .attr("data-bbox")
            .map(|b| {
                b.split_whitespace()
                    .filter_map(|t| t.parse().ok())
                    .collect()
            })
            .unwrap_or_default();
        let raw_bbox: [i64; 4] = match raw_bbox.as_slice() {
            [a, b, c, d] => [*a, *b, *c, *d],
            other => {
                tracing::debug!("invalid data-bbox {:?}, defaulting to full image", other);
                [0, 0, 1, 1]
            }
        };
        let bbox = [
            ((raw_bbox[0] as f64 * width_scaler) as i64).max(0) as u32,
            ((raw_bbox[1] as f64 * height_scaler) as i64).max(0) as u32,
            ((raw_bbox[2] as f64 * width_scaler) as u32).min(img_w),
            ((raw_bbox[3] as f64 * height_scaler) as u32).min(img_h),
        ];
        let content = strip_nested_bboxes(&div.inner_html());
        blocks.push(LayoutBlock {
            label: if label.is_empty() {
                "block".to_string()
            } else {
                label.to_string()
            },
            bbox,
            content,
        });
    }
    blocks
}

/// chandra/output.py::extract_images — crop Image/Figure chunks that
/// contain an `<img>` tag, encode as (lossless) WebP. Keyed by the
/// reference naming scheme md5(html)_{div_idx}_img.webp, with div_idx
/// counted over the chunk list exactly like the reference.
pub fn extract_images(
    html: &str,
    chunks: &[LayoutBlock],
    image: &image::RgbImage,
) -> std::collections::BTreeMap<String, Vec<u8>> {
    let mut out = std::collections::BTreeMap::new();
    for (idx, chunk) in chunks.iter().enumerate() {
        let div_idx = idx + 1;
        if chunk.label != "Image" && chunk.label != "Figure" {
            continue;
        }
        if !chunk.content.contains("<img") {
            continue;
        }
        let Some(crop) = crop_block(image, chunk.bbox) else {
            continue;
        };
        let mut bytes = Vec::new();
        let enc = image::codecs::webp::WebPEncoder::new_lossless(&mut bytes);
        if enc
            .encode(
                crop.as_raw(),
                crop.width(),
                crop.height(),
                image::ExtendedColorType::Rgb8,
            )
            .is_err()
        {
            continue;
        }
        out.insert(image_name(html, div_idx), bytes);
    }
    out
}

/// Reference content fixups for ONE block (`parse_html` body, factored
/// out so the Datalab tree can apply them per block): Image/Figure
/// blocks get `image_src` injected into their first `<img>` (or a
/// synthesized `<img>`), srcless `<img>` hallucinations are stripped
/// from every other block, and bare Text blocks are wrapped in `<p>`.
pub fn fixup_block_content(label: &str, mut content: String, image_src: Option<&str>) -> String {
    if let Some(src) = image_src.filter(|_| label == "Image" || label == "Figure") {
        if let Some(i) = content.find("<img") {
            // Inject/overwrite src on the first img tag.
            let tag_end = content[i..].find('>').map(|j| i + j).unwrap_or(i + 4);
            let tag = content[i..=tag_end].to_string();
            let new_tag = if let Some(s) = tag.find(" src=\"") {
                let after = &tag[s + 6..];
                match after.find('"') {
                    Some(e) => format!("{} src=\"{}\"{}", &tag[..s], src, &after[e + 1..],),
                    None => tag.clone(),
                }
            } else {
                format!("{} src=\"{}\"{}", &tag[..4], src, &tag[4..])
            };
            content.replace_range(i..=tag_end, &new_tag);
        } else {
            content.push_str(&format!("<img src='{src}'/>"));
        }
    } else {
        // Strip srcless img hallucinations.
        while let Some(i) = content.find("<img") {
            let tag_end = match content[i..].find('>') {
                Some(j) => i + j,
                None => break,
            };
            if content[i..=tag_end].contains(" src=") {
                break;
            }
            content.replace_range(i..=tag_end, "");
        }
    }
    if label == "Text" && !content.trim().contains('<') {
        content = format!("<p>{}</p>", content.trim());
    }
    content
}

/// chandra/output.py::parse_html — concatenated inner HTML of the
/// top-level divs with the reference's filtering and fixups:
/// Blank-Page dropped, Page-Header/Footer dropped unless requested,
/// Image/Figure divs get an `src` injected (or a synthesized `<img>`),
/// srcless `<img>` hallucinations stripped from non-image blocks, and
/// bare Text blocks wrapped in `<p>`.
pub fn parse_html(html: &str, include_headers_footers: bool, include_images: bool) -> String {
    let fragment = Html::parse_fragment(html);
    let mut out = String::new();
    let mut div_idx = 0usize;
    for div in top_level_divs(&fragment) {
        div_idx += 1;
        let label = div.value().attr("data-label").unwrap_or("");
        if label == "Blank-Page" {
            continue;
        }
        if !include_headers_footers && (label == "Page-Header" || label == "Page-Footer") {
            continue;
        }
        if !include_images && (label == "Image" || label == "Figure") {
            continue;
        }
        let src = image_name(html, div_idx);
        out.push_str(&fixup_block_content(label, div.inner_html(), Some(&src)));
    }
    out
}

/// Crop one block's bbox out of the page image. None for empty boxes.
pub fn crop_block(image: &image::RgbImage, bbox: [u32; 4]) -> Option<image::RgbImage> {
    let [x0, y0, x1, y1] = bbox;
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(image::imageops::crop_imm(image, x0, y0, x1 - x0, y1 - y0).to_image())
}

/// Encode an RGB crop as JPEG (the Datalab wire format for block
/// images). None on encoder failure.
pub fn encode_jpeg(img: &image::RgbImage, quality: u8) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, quality);
    enc.encode(
        img.as_raw(),
        img.width(),
        img.height(),
        image::ExtendedColorType::Rgb8,
    )
    .ok()?;
    Some(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "<div data-bbox=\"0 0 500 100\" data-label=\"Section-Header\"><h1>Title</h1></div>\n<div data-bbox=\"0 100 1000 900\" data-label=\"Text\">Hello <b data-bbox=\"1 2 3 4\">world</b></div>\n<div data-bbox=\"0 900 1000 1000\" data-label=\"Page-Footer\"><p>42</p></div>";

    #[test]
    fn layout_blocks_scaled_and_stripped() {
        let blocks = parse_layout(SAMPLE, 2000, 1000);
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].label, "Section-Header");
        assert_eq!(blocks[0].bbox, [0, 0, 1000, 100]);
        assert!(!blocks[1].content.contains("data-bbox"));
    }

    #[test]
    fn html_filters_footer_and_wraps_text() {
        let html = parse_html(
            "<div data-label=\"Text\">bare text</div><div data-label=\"Page-Footer\"><p>f</p></div>",
            false,
            true,
        );
        assert_eq!(html, "<p>bare text</p>");
    }

    #[test]
    fn image_div_gets_src() {
        let html = parse_html(
            "<div data-label=\"Figure\"><img alt=\"a chart\"></div>",
            false,
            true,
        );
        assert!(html.contains("src=\""), "{html}");
        assert!(html.contains("_1_img.webp"), "{html}");
    }

    #[test]
    fn jpeg_crop_roundtrip() {
        let img = image::RgbImage::from_pixel(64, 32, image::Rgb([200, 10, 10]));
        let crop = crop_block(&img, [0, 0, 32, 16]).unwrap();
        assert_eq!((crop.width(), crop.height()), (32, 16));
        assert!(crop_block(&img, [5, 5, 5, 9]).is_none());
        let jpeg = encode_jpeg(&crop, 85).unwrap();
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "JPEG SOI marker");
    }

    #[test]
    fn srcless_img_stripped_from_text() {
        let html = parse_html(
            "<div data-label=\"Text\"><p>a<img alt=\"x\">b</p></div>",
            false,
            true,
        );
        assert_eq!(html, "<p>ab</p>");
    }
}

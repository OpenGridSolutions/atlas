// SPDX-License-Identifier: AGPL-3.0-only

//! Chandra page output → Datalab/Marker result document.
//!
//! Pure assembly (no I/O, no HTTP). The contract the client relies on
//! (design doc §2b.3): page nodes `/page/{n}/Page/{n}` (n 0-based, the
//! original PDF index) with a pixel `bbox`; blocks `/page/{n}/{Type}/{i}`
//! with `i` a page-wide running index in reading order; `polygon` =
//! bbox corners; Marker block-type names; `section_hierarchy` per
//! block; `html` + `markdown` on every block (`markdown` never null);
//! block images as base64 JPEG under the block id and again in the
//! document `images` under the name the markdown references.

use std::collections::BTreeMap;

use base64::Engine;
use serde_json::{Value, json};

use crate::ocr::output::LayoutBlock;

/// Chandra label → Marker `block_type`. Unknown labels pass through
/// with hyphens removed so the client still gets a CamelCase name.
pub fn map_block_type(label: &str) -> String {
    match label {
        "Section-Header" => "SectionHeader",
        "Page-Header" => "PageHeader",
        "Page-Footer" => "PageFooter",
        "Equation-Block" => "Equation",
        "List-Group" => "ListGroup",
        "Image" => "Picture",
        "Figure" => "Figure",
        "Code-Block" => "Code",
        "Table-Of-Contents" => "TableOfContents",
        "Complex-Block" => "ComplexRegion",
        "Chemical-Block" | "Diagram" => "Figure",
        "Bibliography" => "Text",
        "Text" => "Text",
        "Table" => "Table",
        "Caption" => "Caption",
        "Footnote" => "Footnote",
        "Form" => "Form",
        "" | "block" => "Text",
        other => return other.replace('-', ""),
    }
    .to_string()
}

fn polygon(b: [u32; 4]) -> Value {
    let [x0, y0, x1, y1] = b.map(f64::from);
    json!([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
}

fn bbox_json(b: [u32; 4]) -> Value {
    Value::Array(b.iter().map(|v| json!(f64::from(*v))).collect())
}

/// Heading level of a SectionHeader block from its `<hN>` tag, default 1.
fn heading_level(html: &str) -> u8 {
    let lower = html.to_ascii_lowercase();
    for lvl in 1..=6u8 {
        if lower.contains(&format!("<h{lvl}")) {
            return lvl;
        }
    }
    1
}

/// One assembled page.
pub struct PageNode {
    pub page_idx: usize,
    /// The `Page` node for `json.children`.
    pub node: Value,
    /// Page markdown (headers/footers excluded like the reference).
    pub markdown: String,
    /// Document-level images: markdown name → base64 JPEG.
    pub images: BTreeMap<String, String>,
}

/// Inputs for one page: the raw model HTML already split into layout
/// chunks (pixel bboxes on `image`), and the page image for crops.
pub struct PageInput<'a> {
    pub page_idx: usize,
    pub chunks: &'a [LayoutBlock],
    pub image: &'a image::RgbImage,
    pub include_headers_footers: bool,
    /// JPEG quality for block crops.
    pub jpeg_quality: u8,
}

/// Build the Datalab page node. Every chunk becomes a leaf block (Chandra
/// emits a flat list, no groups); PageHeader/PageFooter blocks are kept
/// in the tree but excluded from the page markdown unless requested.
pub fn build_page(input: PageInput<'_>) -> PageNode {
    let n = input.page_idx;
    let (w, h) = (input.image.width(), input.image.height());
    let mut children = Vec::with_capacity(input.chunks.len());
    let mut page_md = Vec::new();
    let mut doc_images = BTreeMap::new();
    // Open heading levels → SectionHeader id, snapshot per block.
    let mut open_headings: BTreeMap<u8, String> = BTreeMap::new();

    for (i, chunk) in input.chunks.iter().enumerate() {
        let block_type = map_block_type(&chunk.label);
        let id = format!("/page/{n}/{block_type}/{i}");
        let is_picture = block_type == "Picture" || block_type == "Figure";
        let image_name = is_picture.then(|| format!("_page_{n}_{block_type}_{i}.jpeg"));

        let html = crate::ocr::output::fixup_block_content(
            &chunk.label,
            chunk.content.clone(),
            image_name.as_deref(),
        );
        let markdown = crate::ocr::markdown::convert(&html).trim().to_string();

        let section_hierarchy: Value = open_headings
            .iter()
            .map(|(lvl, hid)| (lvl.to_string(), Value::String(hid.clone())))
            .collect::<serde_json::Map<_, _>>()
            .into();
        if block_type == "SectionHeader" {
            let lvl = heading_level(&html);
            open_headings.retain(|l, _| *l < lvl);
            open_headings.insert(lvl, id.clone());
        }

        let mut block_images = serde_json::Map::new();
        if let Some(name) = &image_name
            && let Some(crop) = crate::ocr::output::crop_block(input.image, chunk.bbox)
            && let Some(jpeg) = crate::ocr::output::encode_jpeg(&crop, input.jpeg_quality)
        {
            let b64 = base64::engine::general_purpose::STANDARD.encode(jpeg);
            block_images.insert(id.clone(), Value::String(b64.clone()));
            doc_images.insert(name.clone(), b64);
        }

        let is_header_footer = block_type == "PageHeader" || block_type == "PageFooter";
        if (input.include_headers_footers || !is_header_footer) && !markdown.is_empty() {
            page_md.push(markdown.clone());
        }

        children.push(json!({
            "id": id,
            "block_type": block_type,
            "page": n,
            "bbox": bbox_json(chunk.bbox),
            "polygon": polygon(chunk.bbox),
            "html": html,
            "markdown": markdown,
            "section_hierarchy": section_hierarchy,
            "images": Value::Object(block_images),
            "children": Value::Null,
        }));
    }

    let page_bbox = [0, 0, w, h];
    let node = json!({
        "id": format!("/page/{n}/Page/{n}"),
        "block_type": "Page",
        "page": n,
        "bbox": bbox_json(page_bbox),
        "polygon": polygon(page_bbox),
        "html": "",
        "markdown": "",
        "section_hierarchy": {},
        "images": {},
        "children": children,
    });
    PageNode {
        page_idx: n,
        node,
        markdown: page_md.join("\n\n"),
        images: doc_images,
    }
}

/// One failed page with its reason (Atlas extension next to
/// `metadata.failed_pages`).
pub struct PageFailure {
    pub page_idx: usize,
    pub error_code: &'static str,
    pub error: String,
}

/// Everything the status endpoint returns for a finished job.
pub struct DocumentInputs<'a> {
    pub pages: Vec<PageNode>,
    pub failures: &'a [PageFailure],
    pub page_count: usize,
    pub runtime_secs: f64,
    pub model_id: &'a str,
    pub cached: bool,
}

/// Assemble the `complete` document. Pages are emitted in document
/// order; `metadata.failed_pages` lists 0-based indices.
pub fn assemble_document(mut inp: DocumentInputs<'_>) -> Value {
    inp.pages.sort_by_key(|p| p.page_idx);
    let mut images = BTreeMap::new();
    let mut md_parts = Vec::with_capacity(inp.pages.len());
    let mut children = Vec::with_capacity(inp.pages.len());
    for p in inp.pages {
        images.extend(p.images);
        md_parts.push(p.markdown);
        children.push(p.node);
    }
    let mut failed: Vec<usize> = inp.failures.iter().map(|f| f.page_idx).collect();
    failed.sort_unstable();
    let page_errors: Vec<Value> = inp
        .failures
        .iter()
        .map(|f| json!({"page": f.page_idx, "error_code": f.error_code, "error": f.error}))
        .collect();
    let metadata = json!({
        "failed_pages": failed,
        "page_errors": page_errors,
        "pages": inp.page_count,
        "cached": inp.cached,
    });
    json!({
        "status": "complete",
        "success": true,
        "error": "",
        "error_code": null,
        "result_url": null,
        "json": {
            "children": children,
            "block_type": "Document",
            "metadata": metadata,
        },
        "markdown": md_parts.join("\n\n"),
        "images": images,
        "metadata": metadata,
        "parse_quality_score": null,
        "total_cost": null,
        "runtime": inp.runtime_secs,
        "checkpoint_id": null,
        "versions": {
            "atlas": env!("CARGO_PKG_VERSION"),
            "model": inp.model_id,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(label: &str, bbox: [u32; 4], content: &str) -> LayoutBlock {
        LayoutBlock {
            label: label.into(),
            bbox,
            content: content.into(),
        }
    }

    #[test]
    fn block_type_mapping() {
        assert_eq!(map_block_type("Section-Header"), "SectionHeader");
        assert_eq!(map_block_type("Image"), "Picture");
        assert_eq!(map_block_type("Equation-Block"), "Equation");
        assert_eq!(map_block_type("Chemical-Block"), "Figure");
        assert_eq!(map_block_type("block"), "Text");
        assert_eq!(map_block_type("New-Thing"), "NewThing");
    }

    #[test]
    fn page_tree_matches_datalab_contract() {
        let img = image::RgbImage::from_pixel(2000, 1000, image::Rgb([255, 255, 255]));
        let chunks = vec![
            chunk("Page-Header", [0, 0, 2000, 40], "<p>Running head</p>"),
            chunk(
                "Section-Header",
                [128, 164, 1362, 272],
                "<h1>Deckungsbeitragsrechnung</h1>",
            ),
            chunk(
                "Text",
                [128, 373, 959, 432],
                "DB I = Umsatz - variable Kosten",
            ),
            chunk("Section-Header", [128, 500, 900, 560], "<h2>Teil</h2>"),
            chunk("Image", [100, 600, 500, 900], "<img alt=\"chart\">"),
            chunk("Text", [128, 920, 959, 980], "Nach dem Bild"),
        ];
        let page = build_page(PageInput {
            page_idx: 3,
            chunks: &chunks,
            image: &img,
            include_headers_footers: false,
            jpeg_quality: 80,
        });
        assert_eq!(page.node["id"], "/page/3/Page/3");
        assert_eq!(page.node["block_type"], "Page");
        assert_eq!(page.node["bbox"], json!([0.0, 0.0, 2000.0, 1000.0]));
        let kids = page.node["children"].as_array().unwrap();
        assert_eq!(kids.len(), 6, "headers stay in the tree");
        assert_eq!(kids[0]["id"], "/page/3/PageHeader/0");
        assert_eq!(kids[1]["id"], "/page/3/SectionHeader/1");
        assert_eq!(kids[2]["id"], "/page/3/Text/2");
        assert_eq!(kids[2]["page"], 3);
        assert_eq!(
            kids[2]["polygon"],
            json!([
                [128.0, 373.0],
                [959.0, 373.0],
                [959.0, 432.0],
                [128.0, 432.0]
            ])
        );
        // Hierarchy: header itself sees {}, text under h1, text after h2 sees both.
        assert_eq!(kids[1]["section_hierarchy"], json!({}));
        assert_eq!(
            kids[2]["section_hierarchy"],
            json!({"1": "/page/3/SectionHeader/1"})
        );
        assert_eq!(
            kids[5]["section_hierarchy"],
            json!({"1": "/page/3/SectionHeader/1", "2": "/page/3/SectionHeader/3"})
        );
        // Markdown on every block, never null.
        for k in kids {
            assert!(k["markdown"].is_string(), "{k}");
            assert!(k["html"].is_string());
        }
        assert_eq!(kids[1]["markdown"], "# Deckungsbeitragsrechnung");
        // Picture: block image under its id, doc image under the markdown name.
        assert_eq!(kids[4]["block_type"], "Picture");
        let name = "_page_3_Picture_4.jpeg";
        assert!(kids[4]["images"]["/page/3/Picture/4"].is_string());
        assert!(page.images.contains_key(name));
        assert!(
            kids[4]["markdown"].as_str().unwrap().contains(name),
            "{}",
            kids[4]["markdown"]
        );
        // Page markdown excludes the header.
        assert!(!page.markdown.contains("Running head"));
        assert!(page.markdown.starts_with("# Deckungsbeitragsrechnung"));
    }

    #[test]
    fn document_assembly_orders_pages_and_lists_failures() {
        let img = image::RgbImage::new(100, 100);
        let c = vec![chunk("Text", [0, 0, 50, 50], "b")];
        let p1 = build_page(PageInput {
            page_idx: 1,
            chunks: &c,
            image: &img,
            include_headers_footers: false,
            jpeg_quality: 80,
        });
        let p0 = build_page(PageInput {
            page_idx: 0,
            chunks: &c,
            image: &img,
            include_headers_footers: false,
            jpeg_quality: 80,
        });
        let failures = [PageFailure {
            page_idx: 2,
            error_code: "incomplete_document",
            error: "truncated".into(),
        }];
        let doc = assemble_document(DocumentInputs {
            pages: vec![p1, p0],
            failures: &failures,
            page_count: 3,
            runtime_secs: 1.5,
            model_id: "datalab-to/chandra-ocr-2",
            cached: false,
        });
        assert_eq!(doc["status"], "complete");
        assert_eq!(doc["success"], true);
        let kids = doc["json"]["children"].as_array().unwrap();
        assert_eq!(kids[0]["id"], "/page/0/Page/0");
        assert_eq!(kids[1]["id"], "/page/1/Page/1");
        assert_eq!(doc["metadata"]["failed_pages"], json!([2]));
        assert_eq!(
            doc["metadata"]["page_errors"][0]["error_code"],
            "incomplete_document"
        );
        assert!(doc["parse_quality_score"].is_null());
        assert!(doc["checkpoint_id"].is_null());
        assert_eq!(doc["runtime"], 1.5);
        assert_eq!(doc["versions"]["model"], "datalab-to/chandra-ocr-2");
    }
}

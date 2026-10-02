// SPDX-License-Identifier: AGPL-3.0-only

//! 1:1 port of chandra/prompts.py (datalab-to/chandra @ master).
//!
//! The upstream prompts are Python f-strings that interpolate the
//! ALLOWED_TAGS / ALLOWED_ATTRIBUTES lists via their Python repr
//! (single-quoted, comma-space separated). The model was trained on
//! exactly that rendering, so the strings below embed the repr
//! verbatim — do NOT reformat them into prose or JSON.

/// `PROMPT_ENDING` — shared tail of every prompt variant.
const PROMPT_ENDING: &str = "Only use these tags ['math', 'br', 'i', 'b', 'u', 'del', 'sup', 'sub', 'table', 'tr', 'td', 'p', 'th', 'div', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'ul', 'ol', 'li', 'input', 'a', 'span', 'img', 'hr', 'tbody', 'small', 'caption', 'strong', 'thead', 'big', 'code', 'chem'], and these attributes ['class', 'colspan', 'rowspan', 'display', 'checked', 'type', 'border', 'value', 'style', 'href', 'alt', 'align', 'data-bbox', 'data-label'].

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
* Lists: Preserve indents and proper list markers.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural.";

/// `OCR_LAYOUT_PROMPT` — layout-block HTML with data-bbox/data-label.
const OCR_LAYOUT_HEAD: &str = "OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format.  Bboxes are normalized 0-1000. The data-label attribute is the label for the block.

Use the following labels:
- Caption
- Footnote
- Equation-Block
- List-Group
- Page-Header
- Page-Footer
- Image
- Section-Header
- Table
- Text
- Complex-Block
- Code-Block
- Form
- Table-Of-Contents
- Figure
- Chemical-Block
- Diagram
- Bibliography
- Blank-Page";

/// `OCR_PROMPT` — plain HTML transcription without layout blocks.
const OCR_HEAD: &str = "OCR this image to HTML.";

/// Mirror of `PROMPT_MAPPING`: resolve a `prompt_type` to the rendered
/// prompt. `None` for unknown types (the handler answers 400 listing
/// the valid ones).
pub fn prompt_for(prompt_type: &str) -> Option<String> {
    match prompt_type {
        "ocr_layout" => Some(format!("{OCR_LAYOUT_HEAD}\n\n{PROMPT_ENDING}")),
        "ocr" => Some(format!("{OCR_HEAD}\n\n{PROMPT_ENDING}")),
        _ => None,
    }
}

/// Valid `prompt_type` values, for error messages.
pub const PROMPT_TYPES: &[&str] = &["ocr_layout", "ocr"];

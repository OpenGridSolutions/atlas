// SPDX-License-Identifier: AGPL-3.0-only

//! HTML→Markdown for the Chandra tag subset — functional port of the
//! `Markdownify` subclass in chandra/output.py (markdownify with
//! heading_style=ATX, bullets "-", escape_underscores/asterisks/dollars,
//! sub/sup kept literal, math → $…$/$$…$$, tables passed through as raw
//! HTML, link text with bracket/paren escapes).
//!
//! Scope note: this is a subset converter for MODEL OUTPUT (flat,
//! whitelisted tags), not a general HTML→MD engine. Where markdownify
//! has conventions the model never exercises (nested lists beyond a few
//! levels, blockquotes, …) we keep the simplest faithful behavior.
//! Byte-parity with the Python lib is validated per-document by the
//! 3.3d golden tests, not guaranteed by construction.

use scraper::{ElementRef, Html, Node};

pub fn convert(html: &str) -> String {
    let fragment = Html::parse_fragment(html);
    let mut out = String::new();
    for child in fragment.root_element().children() {
        render_node(&child, &mut out, &Ctx::default());
    }
    // Collapse 3+ newlines, trim.
    let mut collapsed = String::with_capacity(out.len());
    let mut nl_run = 0usize;
    for c in out.chars() {
        if c == '\n' {
            nl_run += 1;
            if nl_run <= 2 {
                collapsed.push(c);
            }
        } else {
            nl_run = 0;
            collapsed.push(c);
        }
    }
    collapsed.trim().to_string()
}

#[derive(Clone, Default)]
struct Ctx {
    /// Inside pre/code/math — no whitespace normalization, no escapes.
    raw_text: bool,
    /// List nesting depth (for indent).
    list_depth: usize,
}

fn render_children(el: &ElementRef<'_>, out: &mut String, ctx: &Ctx) {
    for child in el.children() {
        render_node(&child, out, ctx);
    }
}

fn children_markdown(el: &ElementRef<'_>, ctx: &Ctx) -> String {
    let mut s = String::new();
    render_children(el, &mut s, ctx);
    s
}

fn render_node(node: &ego_tree::NodeRef<'_, Node>, out: &mut String, ctx: &Ctx) {
    match node.value() {
        Node::Text(t) => {
            let text: &str = t;
            if ctx.raw_text {
                out.push_str(text);
            } else {
                out.push_str(&escape_text(&normalize_ws(text)));
            }
        }
        Node::Element(_) => {
            if let Some(el) = ElementRef::wrap(*node) {
                render_element(&el, out, ctx);
            }
        }
        _ => {}
    }
}

fn normalize_ws(t: &str) -> String {
    // markdownify re_whitespace: runs of [\t \r\n]+ → single space.
    let mut s = String::with_capacity(t.len());
    let mut in_ws = false;
    for c in t.chars() {
        if c == ' ' || c == '\t' || c == '\r' || c == '\n' {
            if !in_ws {
                s.push(' ');
            }
            in_ws = true;
        } else {
            in_ws = false;
            s.push(c);
        }
    }
    s
}

/// escape_underscores + escape_asterisks + escape_dollars
/// (escape_misc=False — everything else passes through).
fn escape_text(t: &str) -> String {
    let mut s = String::with_capacity(t.len());
    for c in t.chars() {
        match c {
            '_' => s.push_str("\\_"),
            '*' => s.push_str("\\*"),
            '$' => s.push_str("\\$"),
            _ => s.push(c),
        }
    }
    s
}

fn wrap_inline(el: &ElementRef<'_>, out: &mut String, ctx: &Ctx, marker: &str) {
    let inner = children_markdown(el, ctx);
    let trimmed = inner.trim();
    if trimmed.is_empty() {
        out.push_str(&inner);
        return;
    }
    // Preserve surrounding whitespace outside the markers.
    let lead_ws = inner.len() - inner.trim_start().len();
    let trail_ws = inner.len() - inner.trim_end().len();
    out.push_str(&inner[..lead_ws]);
    out.push_str(marker);
    out.push_str(trimmed);
    out.push_str(marker);
    out.push_str(&inner[inner.len() - trail_ws..]);
}

fn render_element(el: &ElementRef<'_>, out: &mut String, ctx: &Ctx) {
    let name = el.value().name();
    match name {
        "h1" | "h2" | "h3" | "h4" | "h5" => {
            let level = name[1..].parse::<usize>().unwrap_or(1);
            let text = children_markdown(el, ctx);
            out.push_str("\n\n");
            out.push_str(&"#".repeat(level));
            out.push(' ');
            out.push_str(text.trim());
            out.push_str("\n\n");
        }
        "p" => {
            let text = children_markdown(el, ctx);
            out.push_str("\n\n");
            out.push_str(text.trim());
            out.push_str("\n\n");
        }
        "br" => out.push_str("  \n"),
        "hr" => out.push_str("\n\n---\n\n"),
        "b" | "strong" => wrap_inline(el, out, ctx, "**"),
        "i" | "em" => wrap_inline(el, out, ctx, "*"),
        "del" => wrap_inline(el, out, ctx, "~~"),
        // markdownify options: sub_symbol="<sub>", sup_symbol="<sup>".
        "sub" => {
            out.push_str("<sub>");
            render_children(el, out, ctx);
            out.push_str("</sub>");
        }
        "sup" => {
            out.push_str("<sup>");
            render_children(el, out, ctx);
            out.push_str("</sup>");
        }
        // <u> has no markdownify converter — content passes through.
        "u" | "span" | "small" | "big" | "div" => render_children(el, out, ctx),
        "a" => {
            let inner = children_markdown(el, ctx);
            // Reference convert_a additionally escapes []() in the text.
            let mut text = String::with_capacity(inner.len());
            for c in inner.trim().chars() {
                if matches!(c, '[' | ']' | '(' | ')') {
                    text.push('\\');
                }
                text.push(c);
            }
            let href = el.value().attr("href").unwrap_or("");
            if href.is_empty() {
                out.push_str(&text);
            } else {
                out.push_str(&format!("[{text}]({href})"));
            }
        }
        "img" => {
            let alt = el.value().attr("alt").unwrap_or("");
            let src = el.value().attr("src").unwrap_or("");
            if !src.is_empty() {
                out.push_str(&format!("![{alt}]({src})"));
            }
        }
        "math" => {
            let raw = Ctx {
                raw_text: true,
                ..ctx.clone()
            };
            let body = children_markdown(el, &raw);
            let block = el.value().attr("display") == Some("block");
            if block {
                out.push_str("\n$$");
                out.push_str(body.trim());
                out.push_str("$$\n");
            } else {
                out.push_str(" $");
                out.push_str(body.trim());
                out.push_str("$ ");
            }
        }
        "chem" => {
            // No markdownify converter upstream — content passes through.
            render_children(el, out, ctx);
        }
        "code" => {
            let raw = Ctx {
                raw_text: true,
                ..ctx.clone()
            };
            let body = children_markdown(el, &raw);
            out.push('`');
            out.push_str(&body);
            out.push('`');
        }
        "pre" => {
            let raw = Ctx {
                raw_text: true,
                ..ctx.clone()
            };
            let body = children_markdown(el, &raw);
            out.push_str("\n\n```\n");
            out.push_str(body.trim_end());
            out.push_str("\n```\n\n");
        }
        // Reference convert_table: the whole table passes through as
        // raw HTML (tables with col/rowspan don't round-trip to MD).
        "table" => {
            out.push_str("\n\n");
            out.push_str(&el.html());
            out.push_str("\n\n");
        }
        "ul" | "ol" => {
            let ordered = name == "ol";
            let child_ctx = Ctx {
                list_depth: ctx.list_depth + 1,
                ..ctx.clone()
            };
            out.push('\n');
            let mut index = 0usize;
            for child in el.children() {
                if let Some(li) = ElementRef::wrap(child) {
                    if li.value().name() != "li" {
                        continue;
                    }
                    index += 1;
                    let body = children_markdown(&li, &child_ctx);
                    let indent = "    ".repeat(ctx.list_depth);
                    let marker = if ordered {
                        format!("{index}.")
                    } else {
                        "-".to_string()
                    };
                    out.push_str(&format!("{indent}{marker} {}\n", body.trim()));
                }
            }
            out.push('\n');
        }
        // <input> (checkboxes/radios) has no markdownify converter in
        // the reference subclass either — it renders to nothing outside
        // tables (inside tables the raw HTML passthrough keeps it).
        "input" => {}
        // Table internals only appear under <table> (handled above as
        // raw HTML); stray ones degrade to their content.
        _ => render_children(el, out, ctx),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_blocks() {
        let md = convert("<h1>Title</h1><p>Hello <b>world</b> x_y</p>");
        assert_eq!(md, "# Title\n\nHello **world** x\\_y");
    }

    #[test]
    fn math_delimiters() {
        let md = convert("<p>inline <math>a_1 + b</math> end</p><math display=\"block\">E=mc^2</math>");
        assert!(md.contains("$a_1 + b$"), "{md}");
        assert!(md.contains("$$E=mc^2$$"), "{md}");
    }

    #[test]
    fn table_raw_html() {
        let md = convert("<table><tr><td>a</td><td>b</td></tr></table>");
        assert!(md.starts_with("<table>"), "{md}");
    }

    #[test]
    fn lists_and_links() {
        let md = convert("<ul><li>one</li><li><a href=\"http://x\">a [b]</a></li></ul>");
        assert!(md.contains("- one"), "{md}");
        assert!(md.contains("[a \\[b\\]](http://x)"), "{md}");
    }

    #[test]
    fn img_with_src() {
        let md = convert("<img src=\"abc.webp\" alt=\"chart\"/>");
        assert_eq!(md, "![chart](abc.webp)");
    }
}

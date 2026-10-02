// SPDX-License-Identifier: AGPL-3.0-only

//! Shared multimodal input encoding for the embedding/rerank endpoints
//! (multiclass Plan.md 2.3). Mirrors the chat path's vision handling:
//! server-side base64 decode → patchify (`vision_preprocess`), one
//! `<|vision_start|><|image_pad|><|vision_end|>` frame in the text,
//! pad token replicated per merged patch (`expand_image_pads`), pixel
//! buffers attached to the request for the prefill-time ViT splice.
//!
//! Also hosts the `multipart/form-data` → request-struct parsers
//! (Plan.md Phase U): image files are uploaded the same way audio goes
//! to /v1/audio/transcriptions, bridged to a base64 data URI, and then
//! take the EXACT same path as JSON `image` fields.

use axum::extract::Multipart;
use axum::http::StatusCode;

use crate::main_modules::app_state::{AppState, ModelRuntime};

/// Vision frame inserted before the item text; the single
/// `<|image_pad|>` is expanded to the per-patch count after encoding.
const VISION_FRAME: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// Encoded multimodal item: prompt tokens (pads expanded) + pixel
/// buffers for `prepare_vision_embed`.
pub(crate) struct EncodedItem {
    pub tokens: Vec<u32>,
    pub image_pixels: Vec<(Vec<f32>, usize, usize)>,
}

/// Build the prompt tokens for one (text, optional image) item.
/// `render` receives the vision frame (or "") and must return the FULL
/// prompt text for the item (endpoint-specific framing: instruct
/// prefix + EOS for embeddings, judge prompt for rerank).
pub(crate) fn encode_item(
    state: &AppState,
    rt: &ModelRuntime,
    image: Option<&str>,
    render: impl FnOnce(&str) -> String,
) -> Result<EncodedItem, (StatusCode, String)> {
    let mut image_pixels = Vec::new();
    let mut pad_counts: Vec<usize> = Vec::new();
    let vision_part = if let Some(uri) = image {
        let Some(vcfg) = rt.vision_config.as_ref() else {
            return Err((
                StatusCode::BAD_REQUEST,
                "image input requires a vision-language model (active model serves \
                 text only)"
                    .into(),
            ));
        };
        let (pixels, grid_h, grid_w) =
            spark_model::vision_preprocess::preprocess_image_with_max_pixels(
                uri,
                vcfg,
                state.vision_max_pixels,
            )
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("image decode error: {e}")))?;
        pad_counts.push(spark_model::vision_preprocess::image_pad_count(
            grid_h,
            grid_w,
            vcfg.spatial_merge_size,
        ));
        image_pixels.push((pixels, grid_h, grid_w));
        VISION_FRAME
    } else {
        ""
    };
    let full_text = render(vision_part);
    let tokens = rt
        .tokenizer
        .encode(&full_text)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("tokenizer error: {e:#}")))?;
    let tokens = rt.tokenizer.expand_image_pads(tokens, &pad_counts);
    Ok(EncodedItem {
        tokens,
        image_pixels,
    })
}

/// One input item: plain string, or `{text?, image?}` (image = base64
/// data URI, as in chat `image_url`). Used by both endpoints.
#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
pub enum MultimodalItem {
    Text(String),
    Object {
        #[serde(default)]
        text: Option<String>,
        #[serde(default)]
        image: Option<String>,
    },
}

impl MultimodalItem {
    /// (text, image) view; plain strings have no image.
    pub fn parts(&self) -> (&str, Option<&str>) {
        match self {
            Self::Text(s) => (s.as_str(), None),
            Self::Object { text, image } => (text.as_deref().unwrap_or(""), image.as_deref()),
        }
    }

    /// Text part for `return_documents` echoes.
    pub fn text_owned(&self) -> String {
        self.parts().0.to_string()
    }

    /// An item must carry text and/or an image.
    pub fn is_empty(&self) -> bool {
        let (t, i) = self.parts();
        t.is_empty() && i.is_none()
    }
}

/// `true` when the request body is `multipart/form-data` (with or
/// without the boundary parameter) — the handler then takes the
/// multipart parser instead of the JSON one.
pub(crate) fn is_multipart(req: &axum::extract::Request) -> bool {
    req.headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| {
            ct.trim_start()
                .to_ascii_lowercase()
                .starts_with("multipart/form-data")
        })
}

/// Uploaded file bytes → base64 data URI, the format `encode_item`
/// (and chat `image_url`) already accepts. The mime stays generic —
/// `vision_preprocess::decode_image` sniffs the real format from the
/// magic bytes and ignores the URI mime.
fn bytes_to_data_uri(bytes: &[u8]) -> String {
    use base64::Engine;
    format!(
        "data:image/*;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

type FieldError = (StatusCode, String);

async fn field_text(
    field: axum::extract::multipart::Field<'_>,
    name: &str,
) -> Result<String, FieldError> {
    field.text().await.map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("reading '{name}' field: {e}"),
        )
    })
}

async fn field_image_item(
    field: axum::extract::multipart::Field<'_>,
    name: &str,
) -> Result<MultimodalItem, FieldError> {
    let bytes = field.bytes().await.map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("reading '{name}' file field: {e}"),
        )
    })?;
    if bytes.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("'{name}' file field is empty"),
        ));
    }
    Ok(MultimodalItem::Object {
        text: None,
        image: Some(bytes_to_data_uri(&bytes)),
    })
}

/// Parse a multipart /v1/embeddings request (Plan.md Phase U).
/// Fields: `model` (required); `file` (repeatable → one image item
/// each); `input`/`text` (repeatable → one text item each);
/// `instruction`; `dimensions`; `encoding_format`. Unknown fields are
/// accepted and ignored (parity with the whisper handler).
pub(crate) async fn embeddings_from_multipart(
    mut multipart: Multipart,
) -> Result<crate::openai::embeddings::EmbeddingsRequest, FieldError> {
    let mut model: Option<String> = None;
    let mut items: Vec<MultimodalItem> = Vec::new();
    let mut instruction = None;
    let mut dimensions = None;
    let mut encoding_format = None;
    loop {
        match multipart.next_field().await {
            Ok(Some(field)) => {
                let name = field.name().unwrap_or("").to_string();
                match name.as_str() {
                    "file" => items.push(field_image_item(field, "file").await?),
                    "input" | "text" => {
                        items.push(MultimodalItem::Text(field_text(field, &name).await?))
                    }
                    "model" => model = Some(field_text(field, "model").await?),
                    "instruction" => instruction = Some(field_text(field, "instruction").await?),
                    "dimensions" => {
                        let raw = field_text(field, "dimensions").await?;
                        dimensions = Some(raw.trim().parse::<usize>().map_err(|_| {
                            (
                                StatusCode::BAD_REQUEST,
                                format!("dimensions must be a positive integer (got '{raw}')"),
                            )
                        })?);
                    }
                    "encoding_format" => {
                        encoding_format = Some(field_text(field, "encoding_format").await?)
                    }
                    _ => {}
                }
            }
            Ok(None) => break,
            Err(e) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("multipart parse error: {e}"),
                ));
            }
        }
    }
    let Some(model) = model else {
        return Err((StatusCode::BAD_REQUEST, "missing 'model' field".into()));
    };
    Ok(crate::openai::embeddings::EmbeddingsRequest {
        model,
        input: crate::openai::embeddings::EmbeddingsInput::Batch(items),
        dimensions,
        encoding_format,
        instruction,
    })
}

/// Parse a multipart /v1/rerank request (Plan.md Phase U).
/// Fields: `model`, `query` (required); `documents` (repeatable — a
/// file part (has a filename) becomes an image document, a plain text
/// part a text document; order preserved, mixing allowed);
/// `instruction`; `top_n`; `return_documents`.
pub(crate) async fn rerank_from_multipart(
    mut multipart: Multipart,
) -> Result<crate::openai::rerank::RerankRequest, FieldError> {
    let mut model: Option<String> = None;
    let mut query: Option<String> = None;
    let mut documents: Vec<MultimodalItem> = Vec::new();
    let mut instruction = None;
    let mut top_n = None;
    let mut return_documents = false;
    loop {
        match multipart.next_field().await {
            Ok(Some(field)) => {
                let name = field.name().unwrap_or("").to_string();
                match name.as_str() {
                    "documents" | "document" => {
                        if field.file_name().is_some() {
                            documents.push(field_image_item(field, &name).await?);
                        } else {
                            documents.push(MultimodalItem::Text(field_text(field, &name).await?));
                        }
                    }
                    "model" => model = Some(field_text(field, "model").await?),
                    "query" => query = Some(field_text(field, "query").await?),
                    "instruction" => instruction = Some(field_text(field, "instruction").await?),
                    "top_n" => {
                        let raw = field_text(field, "top_n").await?;
                        top_n = Some(raw.trim().parse::<usize>().map_err(|_| {
                            (
                                StatusCode::BAD_REQUEST,
                                format!("top_n must be a positive integer (got '{raw}')"),
                            )
                        })?);
                    }
                    "return_documents" => {
                        let raw = field_text(field, "return_documents").await?;
                        return_documents = matches!(raw.trim(), "true" | "1" | "yes");
                    }
                    _ => {}
                }
            }
            Ok(None) => break,
            Err(e) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("multipart parse error: {e}"),
                ));
            }
        }
    }
    let Some(model) = model else {
        return Err((StatusCode::BAD_REQUEST, "missing 'model' field".into()));
    };
    let Some(query) = query else {
        return Err((StatusCode::BAD_REQUEST, "missing 'query' field".into()));
    };
    Ok(crate::openai::rerank::RerankRequest {
        model,
        query,
        documents,
        top_n,
        return_documents,
        instruction,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::FromRequest;

    const BOUNDARY: &str = "XATLASX";

    /// Build an axum `Multipart` from (name, filename, value) parts.
    async fn multipart_of(parts: &[(&str, Option<&str>, &str)]) -> Multipart {
        let mut body = String::new();
        for (name, filename, value) in parts {
            body.push_str(&format!("--{BOUNDARY}\r\n"));
            match filename {
                Some(f) => body.push_str(&format!(
                    "Content-Disposition: form-data; name=\"{name}\"; filename=\"{f}\"\r\n\
                     Content-Type: application/octet-stream\r\n\r\n"
                )),
                None => body.push_str(&format!(
                    "Content-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                )),
            }
            body.push_str(value);
            body.push_str("\r\n");
        }
        body.push_str(&format!("--{BOUNDARY}--\r\n"));
        let req = axum::http::Request::builder()
            .header(
                axum::http::header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={BOUNDARY}"),
            )
            .body(axum::body::Body::from(body))
            .unwrap();
        Multipart::from_request(req, &()).await.unwrap()
    }

    #[tokio::test]
    async fn embeddings_files_and_text_keep_order() {
        let mp = multipart_of(&[
            ("model", None, "Qwen/Qwen3-VL-Embedding-8B"),
            ("file", Some("a.png"), "PNGBYTES-A"),
            ("input", None, "hello world"),
            ("file", Some("b.png"), "PNGBYTES-B"),
            ("dimensions", None, "64"),
        ])
        .await;
        let req = embeddings_from_multipart(mp).await.unwrap();
        assert_eq!(req.model, "Qwen/Qwen3-VL-Embedding-8B");
        assert_eq!(req.dimensions, Some(64));
        let items = req.input.into_items();
        assert_eq!(items.len(), 3);
        // item 0 + 2: image-only; item 1: text
        let (t0, i0) = items[0].parts();
        assert!(t0.is_empty() && i0.unwrap().starts_with("data:image/*;base64,"));
        assert_eq!(items[1].parts(), ("hello world", None));
        assert!(items[2].parts().1.is_some());
    }

    #[tokio::test]
    async fn embeddings_missing_model_is_400() {
        let mp = multipart_of(&[("file", Some("a.png"), "PNGBYTES")]).await;
        let (status, msg) = embeddings_from_multipart(mp).await.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("model"));
    }

    #[tokio::test]
    async fn embeddings_bad_dimensions_is_400() {
        let mp = multipart_of(&[("model", None, "m"), ("dimensions", None, "sixty-four")]).await;
        let (status, msg) = embeddings_from_multipart(mp).await.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("dimensions"));
    }

    #[tokio::test]
    async fn rerank_mixed_documents_keep_positions() {
        let mp = multipart_of(&[
            ("model", None, "Qwen/Qwen3-VL-Reranker-8B"),
            ("query", None, "the formula"),
            ("documents", None, "a text document"),
            ("documents", Some("f.png"), "PNGBYTES"),
            ("top_n", None, "2"),
            ("return_documents", None, "true"),
        ])
        .await;
        let req = rerank_from_multipart(mp).await.unwrap();
        assert_eq!(req.query, "the formula");
        assert_eq!(req.top_n, Some(2));
        assert!(req.return_documents);
        assert_eq!(req.documents.len(), 2);
        assert_eq!(req.documents[0].parts(), ("a text document", None));
        let (t1, i1) = req.documents[1].parts();
        assert!(t1.is_empty() && i1.is_some());
    }

    #[tokio::test]
    async fn rerank_missing_query_is_400() {
        let mp = multipart_of(&[("model", None, "m"), ("documents", None, "d")]).await;
        let (status, msg) = rerank_from_multipart(mp).await.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("query"));
    }

    #[tokio::test]
    async fn empty_file_field_is_400() {
        let mp = multipart_of(&[("model", None, "m"), ("file", Some("a.png"), "")]).await;
        let (status, msg) = embeddings_from_multipart(mp).await.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("empty"));
    }

    #[test]
    fn data_uri_roundtrips_through_the_shared_decoder_format() {
        // The decoder strips everything up to the first comma; STANDARD
        // base64 never contains a comma, so the payload survives intact.
        let uri = bytes_to_data_uri(b"\x89PNG");
        let payload = uri.split_once(',').unwrap().1;
        use base64::Engine;
        let back = base64::engine::general_purpose::STANDARD
            .decode(payload)
            .unwrap();
        assert_eq!(back, b"\x89PNG");
    }
}

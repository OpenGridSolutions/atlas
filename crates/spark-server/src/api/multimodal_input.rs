// SPDX-License-Identifier: AGPL-3.0-only

//! Shared multimodal input encoding for the embedding/rerank endpoints
//! (multiclass Plan.md 2.3). Mirrors the chat path's vision handling:
//! server-side base64 decode → patchify (`vision_preprocess`), one
//! `<|vision_start|><|image_pad|><|vision_end|>` frame in the text,
//! pad token replicated per merged patch (`expand_image_pads`), pixel
//! buffers attached to the request for the prefill-time ViT splice.

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
            Self::Object { text, image } => {
                (text.as_deref().unwrap_or(""), image.as_deref())
            }
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

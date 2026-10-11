// SPDX-License-Identifier: AGPL-3.0-only

//! `POST /convert` multipart form → [`ConvertParams`].
//!
//! Accepts every field the Datalab API documents (`file`, `mode`,
//! `output_format`, `include_markdown_in_chunks`, `max_pages`,
//! `page_range`, `save_checkpoint`, `skip_cache`, `extras`) plus the
//! Atlas extension `accepted_processing_time`. Fields Atlas has no use
//! for are accepted and ignored so an unmodified Datalab client works.

use axum::extract::Multipart;

/// Parsed submission. `file` is the raw document (PDF or image).
#[derive(Debug, Default)]
pub struct ConvertParams {
    pub file: Vec<u8>,
    pub filename: String,
    /// Datalab: process only the first N pages of the (selected) range.
    pub max_pages: Option<usize>,
    /// Datalab/Chandra page selection, 0-based ("0,2-5,9").
    pub page_range: Option<String>,
    /// Datalab: bypass the result cache (and refresh the entry).
    pub skip_cache: bool,
    /// Atlas extension: seconds from submit to `complete`. None = no
    /// deadline (behaves like Datalab).
    pub accepted_processing_time: Option<f64>,
    /// Datalab `mode` — accepted, ignored (one model, one quality).
    pub mode: String,
}

#[derive(Debug)]
pub enum ParamError {
    /// Malformed multipart body or unreadable part.
    Multipart(String),
    /// No `file` part.
    MissingFile,
    /// A field has an invalid value.
    Invalid(String),
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::Multipart(m) => write!(f, "multipart parse error: {m}"),
            ParamError::MissingFile => write!(f, "missing 'file' field"),
            ParamError::Invalid(m) => write!(f, "{m}"),
        }
    }
}

fn parse_bool(name: &str, raw: &str) -> Result<bool, ParamError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" | "" => Ok(false),
        other => Err(ParamError::Invalid(format!(
            "field '{name}': expected true/false, got '{other}'"
        ))),
    }
}

/// Apply one text field. Shared by the multipart reader and the unit
/// tests (which feed `(name, value)` pairs directly).
pub fn apply_text_field(p: &mut ConvertParams, name: &str, value: &str) -> Result<(), ParamError> {
    match name {
        "max_pages" => {
            let v = value.trim();
            if !v.is_empty() {
                let n: usize = v.parse().map_err(|_| {
                    ParamError::Invalid(format!("field 'max_pages': '{v}' is not an integer"))
                })?;
                if n == 0 {
                    return Err(ParamError::Invalid("field 'max_pages': must be ≥ 1".into()));
                }
                p.max_pages = Some(n);
            }
        }
        "page_range" => {
            let v = value.trim();
            if !v.is_empty() {
                crate::ocr::input::parse_range_str(v)
                    .map_err(|e| ParamError::Invalid(format!("field 'page_range': {e:#}")))?;
                p.page_range = Some(v.to_string());
            }
        }
        "skip_cache" => p.skip_cache = parse_bool(name, value)?,
        "accepted_processing_time" => {
            let v = value.trim();
            if !v.is_empty() {
                let secs: f64 = v.parse().map_err(|_| {
                    ParamError::Invalid(format!(
                        "field 'accepted_processing_time': '{v}' is not a number of seconds"
                    ))
                })?;
                if !secs.is_finite() || secs <= 0.0 {
                    return Err(ParamError::Invalid(
                        "field 'accepted_processing_time': must be a positive number of seconds"
                            .into(),
                    ));
                }
                p.accepted_processing_time = Some(secs);
            }
        }
        "mode" => p.mode = value.trim().to_string(),
        // output_format, include_markdown_in_chunks, save_checkpoint,
        // extras, paginate, …: Datalab knobs with no Atlas equivalent.
        _ => {}
    }
    Ok(())
}

/// Drain the multipart stream into [`ConvertParams`].
pub async fn parse_multipart(mut mp: Multipart) -> Result<ConvertParams, ParamError> {
    let mut p = ConvertParams::default();
    let mut have_file = false;
    loop {
        match mp.next_field().await {
            Ok(Some(field)) => {
                let name = field.name().unwrap_or("").to_string();
                if name == "file" {
                    p.filename = field.file_name().unwrap_or("").to_string();
                    p.file = field
                        .bytes()
                        .await
                        .map_err(|e| ParamError::Multipart(format!("reading 'file': {e}")))?
                        .to_vec();
                    have_file = true;
                } else {
                    let text = field
                        .text()
                        .await
                        .map_err(|e| ParamError::Multipart(format!("reading '{name}': {e}")))?;
                    apply_text_field(&mut p, &name, &text)?;
                }
            }
            Ok(None) => break,
            Err(e) => return Err(ParamError::Multipart(e.to_string())),
        }
    }
    if !have_file || p.file.is_empty() {
        return Err(ParamError::MissingFile);
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datalab_fields_accepted_and_atlas_fields_parsed() {
        let mut p = ConvertParams::default();
        for (k, v) in [
            ("mode", "accurate"),
            ("output_format", "json,markdown"),
            ("include_markdown_in_chunks", "true"),
            ("max_pages", "12"),
            ("save_checkpoint", "false"),
            ("skip_cache", "true"),
            ("extras", ""),
            ("accepted_processing_time", "90.5"),
            ("page_range", "0,2-3"),
        ] {
            apply_text_field(&mut p, k, v).unwrap();
        }
        assert_eq!(p.mode, "accurate");
        assert_eq!(p.max_pages, Some(12));
        assert!(p.skip_cache);
        assert_eq!(p.accepted_processing_time, Some(90.5));
        assert_eq!(p.page_range.as_deref(), Some("0,2-3"));
    }

    #[test]
    fn invalid_values_rejected() {
        let mut p = ConvertParams::default();
        assert!(apply_text_field(&mut p, "max_pages", "0").is_err());
        assert!(apply_text_field(&mut p, "max_pages", "x").is_err());
        assert!(apply_text_field(&mut p, "accepted_processing_time", "-1").is_err());
        assert!(apply_text_field(&mut p, "accepted_processing_time", "nan").is_err());
        assert!(apply_text_field(&mut p, "skip_cache", "maybe").is_err());
        assert!(apply_text_field(&mut p, "page_range", "5-2").is_err());
        // Empty values mean "unset", like an HTML form posting blanks.
        apply_text_field(&mut p, "max_pages", "").unwrap();
        assert_eq!(p.max_pages, None);
    }
}

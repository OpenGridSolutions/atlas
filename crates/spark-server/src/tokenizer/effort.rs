// SPDX-License-Identifier: AGPL-3.0-only

//! The chat-template variable `reasoning_effort` while thinking is on.
//!
//! Templates differ in the levels they accept and reject unknown ones, so
//! the value a template receives is chosen per model type in
//! `tool_defaults.toml`, `[template_reasoning_effort]`: a default for
//! requests that name no effort, and the template level for each effort a
//! request may name.

use std::collections::HashMap;

/// What templates of model types without an entry receive; the level
/// Mistral's template expects.
const ESTABLISHED_LEVEL: &str = "high";

/// Template levels of one model type.
#[derive(Debug, Clone, PartialEq)]
pub struct TemplateEffort {
    /// Level for requests that name no effort (or one the model type has no
    /// level for).
    default: String,
    /// Request effort -> template level.
    by_request: HashMap<String, String>,
}

impl TemplateEffort {
    /// The entry of `model_type` in `tool_defaults.toml`.
    pub(super) fn for_model_type(model_type: &str) -> Self {
        let defaults: toml::Table = toml::from_str(include_str!("../../tool_defaults.toml"))
            .expect("invalid tool_defaults.toml");
        let entry = defaults
            .get("template_reasoning_effort")
            .and_then(|t| t.get(model_type))
            .and_then(|v| v.as_table());
        let Some(entry) = entry else {
            return Self {
                default: ESTABLISHED_LEVEL.to_string(),
                by_request: HashMap::new(),
            };
        };
        let mut by_request: HashMap<String, String> = entry
            .iter()
            .filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string())))
            .collect();
        let default = by_request
            .remove("default")
            .expect("tool_defaults.toml: [template_reasoning_effort.<type>] needs `default`");
        Self {
            default,
            by_request,
        }
    }

    /// The template level for a request's effort (`None`: none named).
    pub(super) fn level(&self, requested: Option<&str>) -> &str {
        requested
            .and_then(|r| self.by_request.get(r))
            .unwrap_or(&self.default)
    }
}

#[cfg(test)]
mod tests {
    use super::TemplateEffort;

    #[test]
    fn model_types_without_an_entry_keep_the_established_level() {
        for model_type in ["mistral", "qwen3_next"] {
            let e = TemplateEffort::for_model_type(model_type);
            assert_eq!(e.level(None), "high");
            assert_eq!(e.level(Some("low")), "high");
        }
    }

    #[test]
    fn flash_next_thinks_briefly_unless_the_request_names_an_effort() {
        let e = TemplateEffort::for_model_type("qwen4_exp");
        assert_eq!(e.level(None), "low");
        assert_eq!(e.level(Some("minimal")), "low");
        assert_eq!(e.level(Some("low")), "low");
        assert_eq!(e.level(Some("medium")), "medium");
        // The template has no "high"; its top level is "xhigh".
        assert_eq!(e.level(Some("high")), "xhigh");
        assert_eq!(e.level(Some("xhigh")), "xhigh");
        assert_eq!(e.level(Some("max")), "xhigh");
    }

    #[test]
    fn an_effort_the_model_type_has_no_level_for_falls_back_to_the_default() {
        let e = TemplateEffort::for_model_type("qwen4_exp");
        assert_eq!(e.level(Some("unheard-of")), "low");
    }
}

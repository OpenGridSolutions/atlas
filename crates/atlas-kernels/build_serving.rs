// SPDX-License-Identifier: AGPL-3.0-only

//! `[serving]` section of MODEL.toml: what the model's serving block holds.
//!
//! `sessions` — how many sessions the server keeps warm in the KV pool.
//! The context window per session is not stored — the server derives it at
//! load time by splitting the KV pool into `sessions` equal windows (capped
//! at the model's native maximum), so the count stays valid when the pool
//! size changes.
//!
//! `moe_prefill_transpose` — whether the loader builds transposed copies of
//! the MoE expert weights for prefill. They cost memory the KV pool would
//! otherwise get; without the key the loader decides by the free memory it
//! sees at load time.
//!
//! Free of `super::` imports on purpose: `tests/serving_parse.rs` compiles
//! this file directly.

/// The `[serving]` values of one MODEL.toml. Both `None` when the file or
/// the section is absent (models without a token window, targets that keep
/// the server's window flag).
#[derive(Default, Clone, Copy)]
pub(super) struct ServingRaw {
    pub(super) sessions: Option<u32>,
    pub(super) moe_prefill_transpose: Option<bool>,
}

pub(super) fn parse_serving(model_dir: &std::path::Path) -> ServingRaw {
    let model_toml_path = model_dir.join("MODEL.toml");
    if !model_toml_path.exists() {
        return ServingRaw::default();
    }
    let origin = model_toml_path.display().to_string();
    let content =
        std::fs::read_to_string(&model_toml_path).unwrap_or_else(|e| panic!("{origin}: {e}"));
    ServingRaw {
        sessions: serving_sessions(&content, &origin),
        moe_prefill_transpose: serving_moe_prefill_transpose(&content, &origin),
    }
}

const KNOWN_KEYS: [&str; 2] = ["sessions", "moe_prefill_transpose"];

/// The `[serving]` table of MODEL.toml `content`. A section that is present
/// but malformed is a build error, never a silent default; `origin` names
/// the file in the message.
fn serving_table(content: &str, origin: &str) -> Option<toml::Table> {
    let toml: toml::Value = toml::from_str(content).unwrap_or_else(|e| panic!("{origin}: {e}"));
    let table = toml
        .get("serving")?
        .as_table()
        .unwrap_or_else(|| panic!("{origin}: [serving] must be a table"))
        .clone();
    if let Some(key) = table.keys().find(|k| !KNOWN_KEYS.contains(&k.as_str())) {
        panic!(
            "{origin}: unknown key [serving].{key} (known: {})",
            KNOWN_KEYS.join(", ")
        );
    }
    Some(table)
}

/// The session count declared in MODEL.toml `content`; required whenever
/// the `[serving]` section exists.
pub(super) fn serving_sessions(content: &str, origin: &str) -> Option<u32> {
    let table = serving_table(content, origin)?;
    let sessions = table
        .get("sessions")
        .and_then(|v| v.as_integer())
        .unwrap_or_else(|| panic!("{origin}: [serving].sessions must be an integer"));
    assert!(
        (1..=i64::from(u32::MAX)).contains(&sessions),
        "{origin}: [serving].sessions must be at least 1, got {sessions}"
    );
    Some(sessions as u32)
}

/// `[serving].moe_prefill_transpose` of MODEL.toml `content`; `None` when
/// the key is absent.
pub(super) fn serving_moe_prefill_transpose(content: &str, origin: &str) -> Option<bool> {
    let value = serving_table(content, origin)?
        .get("moe_prefill_transpose")?
        .as_bool()
        .unwrap_or_else(|| {
            panic!("{origin}: [serving].moe_prefill_transpose must be true or false")
        });
    Some(value)
}

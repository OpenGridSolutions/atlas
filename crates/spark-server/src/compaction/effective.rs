// SPDX-License-Identifier: AGPL-3.0-only

//! Effective context (CMP-2, CMP-6): head + summary + messages after
//! the cut. The summary is appended to the head's system message as its
//! own section — or a system message is created — so chat templates
//! that allow a single leading system message keep working.

use crate::openai::IncomingMessage;

use super::head::is_head_role;

pub const SUMMARY_OPEN: &str = "<conversation_summary>";
pub const SUMMARY_CLOSE: &str = "</conversation_summary>";

/// The section appended to the system message.
pub fn summary_section(summary: &str) -> String {
    format!(
        "The earlier part of this conversation was compacted to fit the context window. \
         Summary of the messages that were removed:\n{SUMMARY_OPEN}\n{}\n{SUMMARY_CLOSE}",
        summary.trim()
    )
}

/// Build the effective message list. `head_len` messages are kept, the
/// summary (if any) is attached to the last head system/developer
/// message or inserted as a new system message at the front, and the
/// messages after `cut` follow.
pub fn build_effective(
    raw: &[IncomingMessage],
    head_len: usize,
    cut: usize,
    summary: &str,
) -> Vec<IncomingMessage> {
    let head_len = head_len.min(raw.len());
    let mut out: Vec<IncomingMessage> = raw[..head_len].to_vec();
    if !summary.trim().is_empty() {
        let section = summary_section(summary);
        match out.iter().rposition(|m| is_head_role(&m.role)) {
            Some(i) => {
                if !out[i].content.text.trim().is_empty() {
                    out[i].content.text.push_str("\n\n");
                }
                out[i].content.text.push_str(&section);
            }
            None => out.insert(0, IncomingMessage::synthetic_system(section)),
        }
    }
    if cut + 1 < raw.len() {
        out.extend(raw[cut + 1..].iter().cloned());
    }
    out
}

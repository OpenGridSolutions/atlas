// SPDX-License-Identifier: AGPL-3.0-only

//! Head detection (CMP-4): all leading `system` / `developer` messages
//! and, with `compaction_keep_first_user`, the first `user` message that
//! follows them. The head is never summarised.

use crate::openai::IncomingMessage;

pub fn is_head_role(role: &str) -> bool {
    role == "system" || role == "developer"
}

/// Number of leading messages that form the head.
pub fn head_len(msgs: &[IncomingMessage], keep_first_user: bool) -> usize {
    let mut n = msgs.iter().take_while(|m| is_head_role(&m.role)).count();
    if keep_first_user && msgs.get(n).is_some_and(|m| m.role == "user") {
        n += 1;
    }
    n
}

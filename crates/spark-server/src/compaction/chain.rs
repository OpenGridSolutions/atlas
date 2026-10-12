// SPDX-License-Identifier: AGPL-3.0-only

//! The message chain (CMP-1):
//!
//! ```text
//! m_0 = H(domain ‖ model_id)
//! m_j = H(domain ‖ m_{j-1} ‖ canonical(message_j))
//! ```
//!
//! BLAKE3 truncated to 128 bits, the same primitive as the KV block
//! chain, with its own domain tag so the two key spaces never collide.
//! Two conversations with an identical history up to message j share
//! `m_j`; editing a message changes every key after it (CMP-12).

use spark_runtime::block_keys::{BlockKey, chain_seed, chain_step};

use super::canonical::canonical_message;
use crate::openai::IncomingMessage;

/// Domain tag of the message chain.
pub const DOMAIN: &[u8] = b"atlas-msg-v1";

/// `m_0` for a model.
pub fn seed(model_id: &str) -> BlockKey {
    chain_seed(DOMAIN, model_id.as_bytes())
}

/// `m_1..m_n` for `msgs` (one key per message, in order).
pub fn message_chain(model_id: &str, msgs: &[IncomingMessage]) -> Vec<BlockKey> {
    let mut prev = seed(model_id);
    msgs.iter()
        .map(|m| {
            prev = chain_step(DOMAIN, &prev, canonical_message(m).as_bytes());
            prev
        })
        .collect()
}

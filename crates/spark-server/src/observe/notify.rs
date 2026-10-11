// SPDX-License-Identifier: AGPL-3.0-only

//! Push notifier behind the observer (SBIO I/O boundary). ntfy's HTTP
//! API: `POST {url}/{topic}` with the message as body and `Title`,
//! `Priority`, `Tags` headers — the same shape
//! `tests/nvme_session_sweep.py` uses.

use std::time::Duration;

pub trait Notifier: Send + Sync {
    fn notify(&self, title: &str, body: &str, priority: u8, tags: &str) -> Result<(), String>;
}

pub struct NoopNotifier;

impl Notifier for NoopNotifier {
    fn notify(&self, _title: &str, _body: &str, _priority: u8, _tags: &str) -> Result<(), String> {
        Ok(())
    }
}

pub struct NtfyNotifier {
    agent: ureq::Agent,
    endpoint: String,
}

impl NtfyNotifier {
    pub fn new(server_url: &str, topic: &str) -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(8)))
            .build();
        Self {
            agent: config.into(),
            endpoint: format!("{}/{}", server_url.trim_end_matches('/'), topic),
        }
    }
}

impl Notifier for NtfyNotifier {
    fn notify(&self, title: &str, body: &str, priority: u8, tags: &str) -> Result<(), String> {
        // ntfy rejects non-ASCII header values; the body carries the detail.
        let ascii_title: String = title
            .chars()
            .map(|c| if c.is_ascii() { c } else { '?' })
            .collect();
        self.agent
            .post(&self.endpoint)
            .header("Title", &ascii_title)
            .header("Priority", &priority.clamp(1, 5).to_string())
            .header("Tags", tags)
            .send(body)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_join() {
        let n = NtfyNotifier::new("https://ntfy.sh/", "jonas-dgx-spark001");
        assert_eq!(n.endpoint, "https://ntfy.sh/jonas-dgx-spark001");
        assert!(NoopNotifier.notify("t", "b", 3, "warning").is_ok());
    }
}

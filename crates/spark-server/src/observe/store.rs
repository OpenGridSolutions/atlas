// SPDX-License-Identifier: AGPL-3.0-only

//! Embedded SQLite event store (design doc §7): one file on NVMe,
//! queryable after the fact by request id, kind or time window.
//!
//! Tables: `events` (every anomaly, pushed or not), `snapshots`
//! (`/internal/state` + memory every 30 s and on urgent events),
//! `requests` (one row per finished HTTP request). Size is bounded by
//! deleting the oldest rows once the file exceeds the cap.

use std::path::Path;

use rusqlite::{Connection, params};

use super::{Event, RequestRecord};

pub trait EventStore: Send {
    fn insert_event(&mut self, ev: &Event, notified: bool);
    fn insert_request(&mut self, r: &RequestRecord);
    fn insert_snapshot(&mut self, trigger: &str, state: &serde_json::Value);
    /// Trim oldest rows until the database fits `max_bytes`.
    fn enforce_size(&mut self, max_bytes: u64);
}

pub struct NoopStore;

impl EventStore for NoopStore {
    fn insert_event(&mut self, _ev: &Event, _notified: bool) {}
    fn insert_request(&mut self, _r: &RequestRecord) {}
    fn insert_snapshot(&mut self, _trigger: &str, _state: &serde_json::Value) {}
    fn enforce_size(&mut self, _max_bytes: u64) {}
}

pub struct SqliteStore {
    conn: Connection,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    ts_ms INTEGER NOT NULL,
    priority INTEGER NOT NULL,
    kind TEXT NOT NULL,
    model TEXT,
    request_id TEXT,
    message TEXT NOT NULL,
    detail TEXT,
    notified INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts_ms);
CREATE INDEX IF NOT EXISTS events_kind ON events(kind, ts_ms);
CREATE INDEX IF NOT EXISTS events_rid ON events(request_id);
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY,
    ts_ms INTEGER NOT NULL,
    trigger TEXT NOT NULL,
    state TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS snapshots_ts ON snapshots(ts_ms);
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY,
    ts_ms INTEGER NOT NULL,
    request_id TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    status INTEGER NOT NULL,
    elapsed_ms INTEGER NOT NULL,
    model TEXT
);
CREATE INDEX IF NOT EXISTS requests_ts ON requests(ts_ms);
CREATE INDEX IF NOT EXISTS requests_rid ON requests(request_id);
";

impl SqliteStore {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA auto_vacuum=INCREMENTAL;",
        )?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn })
    }

    #[cfg(test)]
    pub fn in_memory() -> anyhow::Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn })
    }

    pub fn file_bytes(&self) -> u64 {
        let q = |pragma: &str| -> u64 {
            self.conn
                .query_row(pragma, [], |r| r.get::<_, i64>(0))
                .unwrap_or(0)
                .max(0) as u64
        };
        q("PRAGMA page_count") * q("PRAGMA page_size")
    }

    pub fn count(&self, table: &str) -> u64 {
        self.conn
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| {
                r.get::<_, i64>(0)
            })
            .unwrap_or(0)
            .max(0) as u64
    }

    /// Delete the oldest fifth of every table.
    fn trim(&mut self) {
        for table in ["events", "snapshots", "requests"] {
            let n = self.count(table);
            if n == 0 {
                continue;
            }
            let drop = (n / 5).max(1);
            let _ = self.conn.execute(
                &format!(
                    "DELETE FROM {table} WHERE id IN (SELECT id FROM {table} ORDER BY id ASC LIMIT ?1)"
                ),
                params![drop as i64],
            );
        }
        let _ = self.conn.execute_batch("PRAGMA incremental_vacuum;");
    }
}

impl EventStore for SqliteStore {
    fn insert_event(&mut self, ev: &Event, notified: bool) {
        let detail = if ev.detail.is_null() {
            None
        } else {
            Some(ev.detail.to_string())
        };
        if let Err(e) = self.conn.execute(
            "INSERT INTO events (ts_ms, priority, kind, model, request_id, message, detail, notified) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                ev.ts_ms as i64,
                ev.priority as i64,
                ev.kind,
                ev.model,
                ev.request_id,
                ev.message,
                detail,
                notified as i64
            ],
        ) {
            tracing::warn!(target: "observe", "event store insert failed: {e}");
        }
    }

    fn insert_request(&mut self, r: &RequestRecord) {
        let _ = self.conn.execute(
            "INSERT INTO requests (ts_ms, request_id, endpoint, status, elapsed_ms, model) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                r.ts_ms as i64,
                r.request_id,
                r.endpoint,
                r.status as i64,
                r.elapsed_ms as i64,
                r.model
            ],
        );
    }

    fn insert_snapshot(&mut self, trigger: &str, state: &serde_json::Value) {
        let _ = self.conn.execute(
            "INSERT INTO snapshots (ts_ms, trigger, state) VALUES (?1, ?2, ?3)",
            params![super::now_ms() as i64, trigger, state.to_string()],
        );
    }

    fn enforce_size(&mut self, max_bytes: u64) {
        let mut guard = 0;
        while self.file_bytes() > max_bytes && guard < 8 {
            self.trim();
            guard += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_trim() {
        let mut s = SqliteStore::in_memory().unwrap();
        for i in 0..50 {
            let ev = Event::new(3, "kv_tier", format!("event {i}"))
                .model("m")
                .request_id(format!("r{i}"))
                .detail(serde_json::json!({"i": i}));
            s.insert_event(&ev, i % 2 == 0);
            s.insert_request(&RequestRecord {
                ts_ms: 1,
                request_id: format!("r{i}"),
                endpoint: "/convert".into(),
                status: if i % 10 == 0 { 503 } else { 200 },
                elapsed_ms: 12,
                model: None,
            });
        }
        s.insert_snapshot("periodic", &serde_json::json!({"ok": true}));
        assert_eq!(s.count("events"), 50);
        assert_eq!(s.count("requests"), 50);
        assert_eq!(s.count("snapshots"), 1);
        let notified: i64 = s
            .conn
            .query_row("SELECT COUNT(*) FROM events WHERE notified = 1", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(notified, 25);
        let by_rid: String = s
            .conn
            .query_row(
                "SELECT message FROM events WHERE request_id = 'r7'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(by_rid, "event 7");
        s.trim();
        assert_eq!(s.count("events"), 40);
        assert_eq!(s.count("requests"), 40);
        assert_eq!(s.count("snapshots"), 0, "1/5 of 1 rounds up to 1");
        // A cap below the current size trims repeatedly but terminates.
        s.enforce_size(1);
        assert!(s.count("events") < 40);
    }
}

// SPDX-License-Identifier: AGPL-3.0-only

//! Pressure-based arena scheduler (requirements §7): per-model two-tier
//! queues ([`queues`]), the pure swap decision ([`decision`]) and — in
//! the dispatcher stage — the orchestration that replaces the synchronous
//! lazy swap. [`estimate`] + [`ledger`] add the deterministic completion-
//! time estimate behind `accepted_processing_time` (PR A).

pub mod decision;
pub mod dispatcher;
pub mod estimate;
pub mod ledger;
pub mod queues;

// Consumed by the dispatcher stage; re-exported for its call sites.
#[allow(unused_imports)]
pub use decision::{ActiveStats, CandidateStats, Decision, DecisionConfig, SwapTrigger, decide};
#[allow(unused_imports)]
pub use dispatcher::{ArenaDispatcher, DispatcherConfig};
pub use estimate::{Estimate, EstimateError, Snapshot, WorkProfile};
pub use ledger::{CapacityLedger, Reservation, Verdict};
#[allow(unused_imports)]
pub use queues::{ArenaQueues, QueueSnapshot, SubmitError, Ticket, TicketOutcome};

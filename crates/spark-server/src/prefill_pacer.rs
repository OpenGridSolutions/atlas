// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill pacer: time-sliced interleaving of chunked prefill and decode
//! (`--prefill-max-pause-ms`, `--prefill-decode-share`).
//!
//! While a prompt is prefilled, every running session waits for the
//! scheduler iteration that carries the prefill chunk. The pacer bounds
//! that wait and splits GPU time between the two sides:
//!
//!  - **Chunk size from time, not tokens.** The chunk is sized so one
//!    iteration (chunk + the decode step riding with it) takes about
//!    `max_pause`. The cost model is `fixed + per_token × tokens`: the
//!    fixed part (one pass over the weights, the decode lanes) is measured,
//!    because a chunk can never be faster than it — shrinking the chunk
//!    below that only multiplies full passes and slows the prefill.
//!  - **Decode share.** After a chunk the pacer owes the running sessions
//!    decode-only steps until they received `decode_share` of the GPU time
//!    of the cycle. At 100 % a prefill waits until nobody decodes.
//!
//! Pure bookkeeping: no clock, no GPU. The scheduler measures the
//! iteration and reports it; the pacer only applies while a decode waits
//! on a prefill — a prefill running alone keeps its full chunk size.

use std::time::Duration;

/// Smallest chunk the regulator shrinks to.
const MIN_CHUNK: usize = 128;
/// Chunk size before the first measurement.
const START_CHUNK: usize = 512;
/// Weight of a new decode-step sample in the running average.
const DECODE_EWMA: f64 = 0.2;
/// Weight of a new chunk sample in the running cost statistics.
const COST_EWMA: f64 = 0.2;
/// The chunk sizes seen must spread by this fraction of their mean before
/// the fixed cost is separated from the per-token cost.
const FLOOR_MIN_SPREAD: f64 = 0.15;
/// Debt below this (seconds) counts as settled.
const DEBT_EPS: f64 = 1e-4;

/// Operator settings of the pacer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PacerConfig {
    /// Longest wait a prefill may impose on running sessions per iteration.
    pub max_pause: Duration,
    /// Share of GPU time the running sessions keep while a prefill is in
    /// progress, `0.0..=1.0`.
    pub decode_share: f64,
}

impl PacerConfig {
    /// `None` when pacing is off (`--prefill-max-pause-ms 0`).
    pub fn from_cli(max_pause_ms: u64, decode_share_pct: u8) -> Option<Self> {
        (max_pause_ms > 0).then(|| Self {
            max_pause: Duration::from_millis(max_pause_ms),
            decode_share: f64::from(decode_share_pct.min(100)) / 100.0,
        })
    }
}

/// One planned prefill chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkPlan {
    pub len: usize,
    pub is_last: bool,
    /// The chunk may ride fused with the decode step (`mixed_forward`).
    /// False for chunk 0 (prefix-cache lookup), chunks inside the cached
    /// prefix (skip / SSM replay) and the last chunk (tail checkpoint,
    /// cache insert) — those need the standard prefill path.
    pub fuse_ok: bool,
}

/// Plan the chunk starting at `offset` of a `total`-token prompt.
///
/// `skip_to` / `cached_prefix` are the sequence's `marconi_skip_to` /
/// `cached_prefix_tokens` (both 0 before the chunk-0 lookup). Tokens below
/// `skip_to` cost nothing, so the chunk spans them plus `budget` tokens of
/// real work, capped at `max_chunk` (the arena limit of one chunk).
///
/// Tail rule: the standard path saves the prompt-tail checkpoint by
/// splitting the LAST chunk at `cut` (the last full block boundary minus
/// one block). A non-last chunk ending in `[cut, total)` would leave a last
/// chunk that starts past `cut` and never splits, so such a chunk becomes
/// the last one (or, when the rest exceeds `max_chunk`, ends before `cut`).
pub fn plan_chunk(
    offset: usize,
    total: usize,
    skip_to: usize,
    cached_prefix: usize,
    budget: usize,
    max_chunk: usize,
    block_size: usize,
) -> ChunkPlan {
    let block = block_size.max(1);
    let remaining = total.saturating_sub(offset);
    let cached = skip_to.saturating_sub(offset);
    let mut len = remaining
        .min(max_chunk.max(1))
        .min(cached.saturating_add(budget.max(1)));
    if len < remaining {
        let aligned_end = (offset + len) / block * block;
        if aligned_end > offset {
            len = aligned_end - offset;
        }
        let cut = (total.saturating_sub(1) / block * block).saturating_sub(block);
        if offset + len >= cut {
            if remaining <= max_chunk {
                len = remaining;
            } else if cut >= block && cut - block > offset {
                len = cut - block - offset;
            }
        }
    }
    let is_last = len >= remaining;
    // Intermediate chunks stay on the GDN WY4 boundary (4 tokens).
    if !is_last && len >= 4 {
        len = len / 4 * 4;
    }
    ChunkPlan {
        len,
        is_last,
        fuse_ok: offset > 0 && !is_last && (skip_to == 0 || offset >= cached_prefix),
    }
}

/// Chunk-size regulator + decode-share ledger. One per scheduler; reset on
/// a model switch (the measured costs belong to the model).
#[derive(Debug)]
pub struct PrefillPacer {
    cfg: PacerConfig,
    /// Largest chunk (the scheduler's `max_prefill_tokens`).
    max_chunk: usize,
    block_size: usize,
    /// Tokens of real work per paced chunk.
    budget: usize,
    /// Running decode-only step time: `(batch size, seconds)`.
    decode: Option<(usize, f64)>,
    /// Running statistics over the measured chunk iterations.
    cost: Option<CostStats>,
    /// Measured fixed cost of a prefill iteration (seconds).
    floor_s: Option<f64>,
    /// Decode time owed to the running sessions (seconds; negative = credit).
    debt_s: f64,
    /// Prefill tokens computed in the iteration being measured. `Some(0)`:
    /// a prefill step ran but its cost is no sample (cached / unpaced).
    pending: Option<usize>,
    warned_floor: bool,
}

impl PrefillPacer {
    pub fn new(cfg: PacerConfig, max_chunk: usize, block_size: usize) -> Self {
        let mut pacer = Self {
            cfg,
            max_chunk: max_chunk.max(1),
            block_size: block_size.max(1),
            budget: 0,
            decode: None,
            cost: None,
            floor_s: None,
            debt_s: 0.0,
            pending: None,
            warned_floor: false,
        };
        pacer.budget = pacer.clamp_budget(START_CHUNK);
        pacer
    }

    pub fn config(&self) -> PacerConfig {
        self.cfg
    }

    /// Forget everything measured (model switch).
    pub fn reset(&mut self) {
        *self = Self::new(self.cfg, self.max_chunk, self.block_size);
    }

    /// No session is decoding: nothing is owed.
    pub fn idle(&mut self) {
        self.debt_s = 0.0;
        self.pending = None;
    }

    /// The running sessions are owed decode-only steps before the next chunk.
    pub fn decode_owed(&self) -> bool {
        self.cfg.decode_share >= 1.0 || self.debt_s > DEBT_EPS
    }

    /// Tokens of real work for the next paced chunk.
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Plan the next chunk with the current budget (see [`plan_chunk`]).
    /// `budget_cap` is the fused-step buffer limit of this iteration.
    pub fn plan(
        &self,
        offset: usize,
        total: usize,
        skip_to: usize,
        cached_prefix: usize,
        budget_cap: usize,
        max_chunk: usize,
    ) -> ChunkPlan {
        plan_chunk(
            offset,
            total,
            skip_to,
            cached_prefix,
            self.budget.min(budget_cap),
            max_chunk,
            self.block_size,
        )
    }

    /// A prefill step ran in this iteration and computed `tokens` prompt
    /// tokens (0 = fully cached, or a chunk the pacer did not size).
    pub fn note_chunk(&mut self, tokens: usize) {
        self.pending = Some(tokens);
    }

    /// Close the measured iteration: `elapsed` spans the prefill step and
    /// the decode step, `waiting` is the number of sessions that were
    /// decoding when it began.
    pub fn end_iteration(&mut self, elapsed: Duration, waiting: usize) {
        let chunk = self.pending.take();
        if waiting == 0 {
            self.debt_s = 0.0;
            return;
        }
        let secs = elapsed.as_secs_f64();
        match chunk {
            None => {
                self.observe_decode(waiting, secs);
                // Only a debt is paid down; decode time while nothing is
                // owed earns no credit against a later prefill.
                if self.debt_s > 0.0 {
                    self.debt_s -= secs;
                }
            }
            Some(0) => {}
            Some(tokens) => {
                self.observe_chunk(tokens, secs);
                self.charge(secs);
            }
        }
    }

    fn decode_step_s(&self) -> f64 {
        self.decode.map_or(0.0, |(_, s)| s)
    }

    fn observe_decode(&mut self, batch: usize, secs: f64) {
        self.decode = Some(match self.decode {
            Some((b, s)) if b == batch => (b, s + DECODE_EWMA * (secs - s)),
            // A different batch size has a different step time: start over.
            _ => (batch, secs),
        });
    }

    /// Book a prefill iteration of `secs`: the sessions got one decode step
    /// in it; to end the cycle at `decode_share` they are owed
    /// `(share × secs − step) / (1 − share)` of decode-only time.
    fn charge(&mut self, secs: f64) {
        let share = self.cfg.decode_share;
        if share >= 1.0 {
            return;
        }
        let step = self.decode_step_s();
        self.debt_s = (self.debt_s + (share * secs - step) / (1.0 - share)).max(-step);
    }

    fn observe_chunk(&mut self, tokens: usize, secs: f64) {
        // Short pieces (prompt tails) say little about the per-token cost.
        if tokens * 2 < self.budget || secs <= 0.0 {
            return;
        }
        let n = tokens as f64;
        let cost = self.cost.get_or_insert(CostStats::first(n, secs));
        cost.add(n, secs);
        let (mean_n, mean_t) = (cost.n, cost.t);
        if let Some(floor) = cost.fixed_cost() {
            self.floor_s = Some(floor);
        }
        // Until the chunk sizes spread enough to measure it, the decode
        // step riding with the chunk is the known part of the fixed cost.
        let floor = self
            .floor_s
            .unwrap_or_else(|| self.decode_step_s())
            .min(0.9 * mean_t);
        let per_token = (mean_t - floor) / mean_n;
        let pause = self.cfg.max_pause.as_secs_f64();
        // A step cannot be shorter than its fixed cost. When the target is
        // out of reach, let the fixed cost be at most half the step instead
        // of collapsing to minimal chunks that are all overhead.
        let step_target = pause.max(2.0 * floor);
        if step_target > pause && self.floor_s.is_some() && !self.warned_floor {
            self.warned_floor = true;
            tracing::warn!(
                "prefill pacer: --prefill-max-pause-ms {} is below what this model can do \
                 (fixed cost per prefill step ≈ {:.0} ms); pacing at ≈ {:.0} ms per step",
                self.cfg.max_pause.as_millis(),
                floor * 1e3,
                step_target * 1e3,
            );
        }
        let mut want = ((step_target - floor) / per_token).round() as usize;
        // The averages lag; a step far over the target is cut at once.
        if secs > 2.0 * step_target {
            want = want.min((n * step_target / secs) as usize);
        }
        // Grow at most 2× per step; shrink at once.
        let next = self.clamp_budget(want.min(self.budget.saturating_mul(2)));
        if next != self.budget {
            tracing::debug!(
                "prefill pacer: chunk {} → {} tokens ({} tokens took {:.0} ms, fixed ≈ {:.0} ms)",
                self.budget,
                next,
                tokens,
                secs * 1e3,
                floor * 1e3,
            );
            self.budget = next;
        }
    }

    fn clamp_budget(&self, tokens: usize) -> usize {
        let clamped = tokens.clamp(MIN_CHUNK.min(self.max_chunk), self.max_chunk);
        let aligned = clamped / self.block_size * self.block_size;
        if aligned >= MIN_CHUNK.min(self.max_chunk) && aligned > 0 {
            aligned
        } else {
            clamped
        }
    }
}

/// Exponentially weighted statistics over `(chunk tokens, iteration
/// seconds)` samples: the line `seconds = fixed + per_token × tokens`.
#[derive(Debug, Clone, Copy)]
struct CostStats {
    /// Mean chunk size and mean iteration time.
    n: f64,
    t: f64,
    /// Variance of the chunk size, covariance of size and time.
    var_n: f64,
    cov: f64,
}

impl CostStats {
    fn first(n: f64, t: f64) -> Self {
        Self {
            n,
            t,
            var_n: 0.0,
            cov: 0.0,
        }
    }

    fn add(&mut self, n: f64, t: f64) {
        let (dn, dt) = (n - self.n, t - self.t);
        self.n += COST_EWMA * dn;
        self.t += COST_EWMA * dt;
        self.var_n = (1.0 - COST_EWMA) * (self.var_n + COST_EWMA * dn * dn);
        self.cov = (1.0 - COST_EWMA) * (self.cov + COST_EWMA * dn * dt);
    }

    /// The intercept of the fitted line — `None` while the chunk sizes
    /// seen are too alike to tell fixed from per-token cost.
    fn fixed_cost(&self) -> Option<f64> {
        let min_spread = FLOOR_MIN_SPREAD * self.n;
        if self.var_n < min_spread * min_spread || self.cov <= 0.0 {
            return None;
        }
        let per_token = self.cov / self.var_n;
        Some((self.t - per_token * self.n).clamp(0.0, 0.9 * self.t))
    }
}

#[cfg(test)]
#[path = "prefill_pacer_tests.rs"]
mod tests;

// SPDX-License-Identifier: AGPL-3.0-only

//! Cut selection (CMP-5): the recent turns are the last messages whose
//! token sum reaches at least `tail_target`; the cut lands only on a
//! message boundary and never between a tool call and its results.
//! When that tail does not fit the budget left by head, summary and
//! output reserve, the tail shrinks one unit at a time; the newest unit
//! always survives (CTX-6 decides what happens if even that is too big).

/// Chosen cut for a new compaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CutPlan {
    /// 0-based index of the last replaced message.
    pub cut: usize,
    /// Tokens of the messages after the cut.
    pub tail_tokens: usize,
    /// Tokens of the messages `first..=cut`.
    pub replaced_tokens: usize,
}

/// A cut after message `i` is allowed unless message `i + 1` is a tool
/// result — it belongs to the assistant turn that produced the call.
pub fn cut_allowed(roles: &[&str], i: usize) -> bool {
    i + 1 < roles.len() && roles[i + 1] != "tool"
}

/// Pick the cut.
///
/// - `roles` / `sizes`: role and token size of every message (raw order);
/// - `first`: smallest index a cut may land on (end of head, or one past
///   the previous cut for a cumulative compaction);
/// - `tail_target`: CMP-5 minimum tail; `tail_budget`: tokens the tail
///   may occupy so the effective context fits.
///
/// `None` when nothing can be folded (fewer than two messages after
/// `first`, or no allowed boundary).
pub fn choose_cut(
    roles: &[&str],
    sizes: &[usize],
    first: usize,
    tail_target: usize,
    tail_budget: usize,
) -> Option<CutPlan> {
    let n = roles.len();
    if n < 2 || first + 1 >= n {
        return None;
    }
    // Ascending cut index == descending tail size.
    let cands: Vec<usize> = (first..n - 1).filter(|&c| cut_allowed(roles, c)).collect();
    let tail = |c: usize| sizes[c + 1..].iter().sum::<usize>();
    // The minimal tail that reaches the target (largest such cut); if no
    // tail reaches it, start from the largest tail.
    let preferred = cands
        .iter()
        .rev()
        .find(|&&c| tail(c) >= tail_target)
        .copied()
        .unwrap_or(*cands.first()?);
    // Shrink until the tail fits the budget; stop at the newest unit.
    let mut chosen = preferred;
    for &c in cands.iter().filter(|&&c| c >= preferred) {
        chosen = c;
        if tail(c) <= tail_budget {
            break;
        }
    }
    Some(CutPlan {
        cut: chosen,
        tail_tokens: tail(chosen),
        replaced_tokens: sizes[first..=chosen].iter().sum(),
    })
}

// SPDX-License-Identifier: AGPL-3.0-only

//! Deterministic completion-time estimate for `accepted_processing_time`
//! (docs/design/deadline-admission.md §Frage 1/3).
//!
//! A PURE function over three inputs — the model's calibrated
//! [`EstimateProfile`] (MODEL.toml `[estimate]`), the request's
//! [`WorkProfile`], and a [`Snapshot`] of the server taken under the
//! ledger lock — so replayed snapshots reproduce every decision in unit
//! tests. No clocks, no I/O, nothing learned at runtime.
//!
//! ```text
//! ETA = T_ahead + T_model + T_work
//! T_ahead = reserved work of admitted-but-unfinished requests that
//!           cannot overlap with this one (batch full), plus queued tickets
//! T_model = 0 if resident; else hysteresis remainder + T_swap (WAIT-3),
//!           plus the running swap's remainder
//! T_work  = pages × page_secs_p90                     (OCR)
//!         | prompt / prefill_tok_s + out × step(batch) (generation)
//! ```
//! The result is multiplied by the operator's safety factor before it is
//! compared with the client's budget.

pub use atlas_kernels::EstimateProfile;

/// What the request will cost, in model-independent units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WorkProfile {
    /// Prompt length (tokens; a byte-based guess before tokenization).
    pub prompt_tokens: usize,
    /// Upper bound on generated tokens (`max_tokens` or the endpoint's
    /// fixed output).
    pub output_tokens: usize,
    /// OCR pages (0 for non-OCR work). When > 0 the page model is used
    /// and the token fields are ignored.
    pub pages: usize,
}

impl WorkProfile {
    /// Conservative token guess from raw prompt bytes (≈3.5 bytes/token
    /// for mixed European text; CJK is denser, so this over-estimates
    /// there, which is the safe direction).
    pub fn tokens_from_bytes(bytes: usize) -> usize {
        bytes.div_ceil(3) * 3 / 4 + 1
    }
}

/// Server state at the decision instant. Built by `api::admission`
/// under the ledger lock.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Snapshot {
    /// Target model is loaded and no swap is running.
    pub target_resident: bool,
    /// WAIT-3 wait for a non-resident target (hysteresis remainder +
    /// T_swap), or the running swap's remainder. 0 when resident.
    pub model_wait_secs: f64,
    /// Sequences active in the GPU batch right now.
    pub n_active: usize,
    /// `--max-batch-size`.
    pub max_batch: usize,
    /// Admitted-but-unfinished work ahead of this request (seconds of
    /// single-sequence time, from the ledger).
    pub reserved_secs: f64,
    /// Requests already queued for the target model (tickets).
    pub queued_ahead: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Estimate {
    pub ahead_secs: f64,
    pub model_secs: f64,
    pub work_secs: f64,
}

impl Estimate {
    pub fn total_secs(&self) -> f64 {
        self.ahead_secs + self.model_secs + self.work_secs
    }
}

/// Why no estimate can be given.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EstimateError {
    /// The model has no `[estimate]` block at all.
    NoProfile,
    /// The block lacks the constants this work needs (e.g. OCR pages on a
    /// profile without `ocr_page_secs_p90`).
    Uncalibrated,
}

/// Decode step time at `batch` by linear interpolation over the
/// calibrated points (1, 2, 4, 8); clamped at the ends.
pub fn decode_step_secs(profile: &EstimateProfile, batch: usize) -> f64 {
    const POINTS: [f64; 4] = [1.0, 2.0, 4.0, 8.0];
    let b = (batch.max(1) as f64).min(8.0);
    let ms = &profile.decode_step_ms;
    let mut i = 0;
    while i + 1 < POINTS.len() && b > POINTS[i + 1] {
        i += 1;
    }
    if i + 1 >= POINTS.len() {
        return ms[3] / 1000.0;
    }
    let t = (b - POINTS[i]) / (POINTS[i + 1] - POINTS[i]);
    (ms[i] + t * (ms[i + 1] - ms[i])) / 1000.0
}

/// The estimate. Pure.
pub fn estimate(
    profile: Option<&EstimateProfile>,
    work: &WorkProfile,
    snap: &Snapshot,
) -> Result<Estimate, EstimateError> {
    let profile = profile.ok_or(EstimateError::NoProfile)?;

    let work_secs = if work.pages > 0 {
        if profile.ocr_page_secs_p90 <= 0.0 {
            return Err(EstimateError::Uncalibrated);
        }
        // Measured 2026-10-11: page fan-out barely shortens wall time on
        // the vision-heavy OCR model, so pages are costed sequentially.
        work.pages as f64 * profile.ocr_page_secs_p90
    } else {
        if profile.prefill_tok_s <= 0.0 || profile.decode_step_ms.iter().any(|&m| m <= 0.0) {
            return Err(EstimateError::Uncalibrated);
        }
        let batch_after = (snap.n_active + snap.queued_ahead + 1).min(snap.max_batch.max(1));
        let prefill = work.prompt_tokens as f64 / profile.prefill_tok_s;
        let decode = work.output_tokens as f64 * decode_step_secs(profile, batch_after);
        prefill + decode
    };

    // Work ahead of us: OCR pages are vision-prefill-bound and the
    // prefills run one after another, so every reserved page second is
    // paid in full (measured 2026-10-11: two parallel 3-page jobs both
    // missed a 55 s budget that each alone met). Token generation shares
    // the batch — reserved work only delays us when no slot is free, and
    // then it is spread across the batch width.
    let slots = snap.max_batch.max(1);
    let ahead_secs = if work.pages > 0 {
        snap.reserved_secs
    } else if snap.n_active + snap.queued_ahead >= slots {
        snap.reserved_secs / slots as f64
    } else {
        0.0
    };

    let model_secs = if snap.target_resident {
        0.0
    } else {
        snap.model_wait_secs.max(0.0)
    };

    Ok(Estimate {
        ahead_secs,
        model_secs,
        work_secs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_profile() -> EstimateProfile {
        EstimateProfile {
            prefill_tok_s: 2000.0,
            decode_step_ms: [15.0, 16.0, 18.0, 22.0],
            ocr_page_secs_p90: 0.0,
            ocr_page_tokens_p90: 0,
        }
    }

    fn ocr_profile() -> EstimateProfile {
        EstimateProfile {
            prefill_tok_s: 0.0,
            decode_step_ms: [0.0; 4],
            ocr_page_secs_p90: 12.5,
            ocr_page_tokens_p90: 0,
        }
    }

    fn idle(resident: bool) -> Snapshot {
        Snapshot {
            target_resident: resident,
            model_wait_secs: if resident { 0.0 } else { 40.0 },
            n_active: 0,
            max_batch: 4,
            reserved_secs: 0.0,
            queued_ahead: 0,
        }
    }

    #[test]
    fn step_interpolation() {
        let p = gen_profile();
        assert!((decode_step_secs(&p, 1) - 0.015).abs() < 1e-9);
        assert!((decode_step_secs(&p, 3) - 0.017).abs() < 1e-9);
        assert!((decode_step_secs(&p, 8) - 0.022).abs() < 1e-9);
        assert!((decode_step_secs(&p, 64) - 0.022).abs() < 1e-9, "clamped");
        assert!(
            (decode_step_secs(&p, 0) - 0.015).abs() < 1e-9,
            "batch 0 → 1"
        );
    }

    #[test]
    fn generation_resident_idle() {
        let work = WorkProfile {
            prompt_tokens: 2000,
            output_tokens: 1000,
            pages: 0,
        };
        let e = estimate(Some(&gen_profile()), &work, &idle(true)).unwrap();
        // 1 s prefill + 1000 × 15 ms
        assert!((e.work_secs - 16.0).abs() < 1e-9, "{e:?}");
        assert_eq!(e.model_secs, 0.0);
        assert_eq!(e.ahead_secs, 0.0);
    }

    #[test]
    fn swap_adds_model_wait_and_batch_slows_decode() {
        let work = WorkProfile {
            prompt_tokens: 0,
            output_tokens: 1000,
            pages: 0,
        };
        let e = estimate(Some(&gen_profile()), &work, &idle(false)).unwrap();
        assert_eq!(e.model_secs, 40.0);
        let busy = Snapshot {
            n_active: 3,
            ..idle(true)
        };
        let e2 = estimate(Some(&gen_profile()), &work, &busy).unwrap();
        // batch after admission = 4 → 18 ms/step
        assert!((e2.work_secs - 18.0).abs() < 1e-9, "{e2:?}");
        assert_eq!(e2.ahead_secs, 0.0, "a slot is free");
    }

    #[test]
    fn full_batch_waits_on_reserved_work() {
        let work = WorkProfile {
            prompt_tokens: 0,
            output_tokens: 100,
            pages: 0,
        };
        let full = Snapshot {
            n_active: 4,
            reserved_secs: 80.0,
            ..idle(true)
        };
        let e = estimate(Some(&gen_profile()), &work, &full).unwrap();
        assert!((e.ahead_secs - 20.0).abs() < 1e-9, "{e:?}");
        let queued = Snapshot {
            n_active: 2,
            queued_ahead: 2,
            reserved_secs: 40.0,
            ..idle(true)
        };
        let e = estimate(Some(&gen_profile()), &work, &queued).unwrap();
        assert!((e.ahead_secs - 10.0).abs() < 1e-9, "{e:?}");
    }

    #[test]
    fn ocr_pages_sequential() {
        let work = WorkProfile {
            pages: 3,
            ..Default::default()
        };
        let e = estimate(Some(&ocr_profile()), &work, &idle(true)).unwrap();
        assert!((e.work_secs - 37.5).abs() < 1e-9);
        assert!((e.total_secs() - 37.5).abs() < 1e-9);
        // Reserved OCR work ahead is paid in full even with free slots.
        let busy = Snapshot {
            reserved_secs: 37.5,
            ..idle(true)
        };
        let e2 = estimate(Some(&ocr_profile()), &work, &busy).unwrap();
        assert!((e2.ahead_secs - 37.5).abs() < 1e-9, "{e2:?}");
        assert!((e2.total_secs() - 75.0).abs() < 1e-9);
    }

    #[test]
    fn missing_or_partial_profiles_are_errors_not_guesses() {
        let work = WorkProfile {
            pages: 1,
            ..Default::default()
        };
        assert_eq!(
            estimate(None, &work, &idle(true)).unwrap_err(),
            EstimateError::NoProfile
        );
        assert_eq!(
            estimate(Some(&gen_profile()), &work, &idle(true)).unwrap_err(),
            EstimateError::Uncalibrated,
            "generation profile has no page cost"
        );
        let gen_work = WorkProfile {
            prompt_tokens: 10,
            output_tokens: 10,
            pages: 0,
        };
        assert_eq!(
            estimate(Some(&ocr_profile()), &gen_work, &idle(true)).unwrap_err(),
            EstimateError::Uncalibrated,
            "OCR profile has no token costs"
        );
    }

    #[test]
    fn monotone_in_pages_prompt_and_queue() {
        let p = gen_profile();
        let w = |prompt, out| WorkProfile {
            prompt_tokens: prompt,
            output_tokens: out,
            pages: 0,
        };
        let a = estimate(Some(&p), &w(100, 100), &idle(true))
            .unwrap()
            .total_secs();
        let b = estimate(Some(&p), &w(1000, 100), &idle(true))
            .unwrap()
            .total_secs();
        let c = estimate(Some(&p), &w(1000, 1000), &idle(true))
            .unwrap()
            .total_secs();
        assert!(a < b && b < c);
        let q0 = estimate(Some(&p), &w(100, 100), &idle(true))
            .unwrap()
            .total_secs();
        let q1 = estimate(
            Some(&p),
            &w(100, 100),
            &Snapshot {
                queued_ahead: 2,
                ..idle(true)
            },
        )
        .unwrap()
        .total_secs();
        assert!(q1 >= q0);
        assert_eq!(WorkProfile::tokens_from_bytes(0), 1);
        assert!(WorkProfile::tokens_from_bytes(3500) >= 875);
    }
}

// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

const BLOCK: usize = 16;
const MAX_CHUNK: usize = 8192;

fn secs(s: f64) -> Duration {
    Duration::from_secs_f64(s)
}

fn pacer(max_pause_ms: u64, share_pct: u8) -> PrefillPacer {
    PrefillPacer::new(
        PacerConfig::from_cli(max_pause_ms, share_pct).expect("pacing on"),
        MAX_CHUNK,
        BLOCK,
    )
}

/// Drive the pacer against a synthetic machine with one waiting session:
/// an iteration carrying an n-token chunk takes `fixed + per_token × n`, a
/// decode-only step takes `decode`. Returns (seconds in chunk iterations,
/// seconds in decode-only steps).
fn drive(
    p: &mut PrefillPacer,
    fixed: f64,
    per_token: f64,
    decode: f64,
    steps: usize,
) -> (f64, f64) {
    let (mut chunk_s, mut decode_s) = (0.0, 0.0);
    for _ in 0..steps {
        if p.decode_owed() {
            p.end_iteration(secs(decode), 1);
            decode_s += decode;
        } else {
            let n = p.budget();
            p.note_chunk(n);
            let t = fixed + per_token * n as f64;
            p.end_iteration(secs(t), 1);
            chunk_s += t;
        }
    }
    (chunk_s, decode_s)
}

// ── Config ──

#[test]
fn zero_pause_turns_pacing_off() {
    assert_eq!(PacerConfig::from_cli(0, 50), None);
}

#[test]
fn share_is_a_percentage() {
    let cfg = PacerConfig::from_cli(500, 50).unwrap();
    assert_eq!(cfg.max_pause, Duration::from_millis(500));
    assert_eq!(cfg.decode_share, 0.5);
    assert_eq!(PacerConfig::from_cli(500, 100).unwrap().decode_share, 1.0);
    assert_eq!(PacerConfig::from_cli(500, 0).unwrap().decode_share, 0.0);
}

// ── Chunk-size regulator ──

#[test]
fn chunk_grows_until_the_iteration_fills_the_pause() {
    // 0.5 ms/token, 20 ms decode riding along: 500 ms ⇒ 960 tokens.
    let mut p = pacer(500, 50);
    drive(&mut p, 0.02, 0.0005, 0.02, 200);
    assert_eq!(p.budget(), 960);
}

#[test]
fn fixed_cost_is_measured_and_left_out_of_the_per_token_cost() {
    // The measured shape of the fused step: 250 ms per pass regardless of
    // size, 0.13 ms/token on top. 500 ms ⇒ (500 − 250) / 0.13 = 1923.
    let mut p = pacer(500, 50);
    drive(&mut p, 0.25, 0.00013, 0.1, 200);
    assert_eq!(p.budget(), 1920);
    assert!((p.floor_s.unwrap() - 0.25).abs() < 0.01);
}

#[test]
fn pause_below_the_fixed_cost_does_not_collapse_the_chunk() {
    // 200 ms asked, 250 ms is the floor: shrinking would only multiply
    // 250 ms passes. The pacer settles where the fixed cost is half the
    // step (2 × 250 ms) instead of at the minimum chunk.
    let mut p = pacer(200, 50);
    drive(&mut p, 0.25, 0.00013, 0.1, 200);
    assert_eq!(p.budget(), 1920);
    assert!(p.warned_floor);
}

#[test]
fn slow_model_shrinks_to_the_minimum_chunk() {
    // 10 ms/token: 500 ms would be 50 tokens — below the minimum.
    let mut p = pacer(500, 50);
    drive(&mut p, 0.0, 0.01, 0.05, 200);
    assert_eq!(p.budget(), MIN_CHUNK);
}

#[test]
fn fast_model_stops_at_the_full_chunk() {
    let mut p = pacer(500, 50);
    drive(&mut p, 0.01, 0.00001, 0.01, 200);
    assert_eq!(p.budget(), MAX_CHUNK);
}

#[test]
fn chunk_grows_at_most_twofold_per_sample() {
    let mut p = pacer(500, 0);
    let before = p.budget();
    p.note_chunk(before);
    p.end_iteration(secs(0.001), 1);
    assert_eq!(p.budget(), before * 2);
}

#[test]
fn budget_is_block_aligned_and_capped_by_a_small_max_chunk() {
    let p = PrefillPacer::new(PacerConfig::from_cli(500, 50).unwrap(), 100, BLOCK);
    assert_eq!(p.budget(), 100);
    let p = PrefillPacer::new(PacerConfig::from_cli(500, 50).unwrap(), 300, BLOCK);
    assert_eq!(p.budget(), 288);
}

#[test]
fn short_pieces_are_no_cost_sample() {
    let mut p = pacer(500, 50);
    let before = p.budget();
    p.note_chunk(before / 4);
    p.end_iteration(secs(3.0), 1);
    assert_eq!(p.budget(), before);
    assert!(p.decode_owed(), "the time is still charged");
}

// ── Decode share ──

#[test]
fn half_share_owes_three_decode_steps_after_a_500ms_chunk() {
    // 500 ms iteration, 100 ms decode step, 50 %:
    // (0.5 × 500 − 100) / 0.5 = 300 ms = 3 decode-only steps, so the cycle
    // is 800 ms with 400 ms of decode.
    let mut p = pacer(500, 50);
    p.end_iteration(secs(0.1), 1);
    assert!(
        !p.decode_owed(),
        "decode alone builds no debt and no credit"
    );
    p.note_chunk(512);
    p.end_iteration(secs(0.5), 1);
    for _ in 0..3 {
        assert!(p.decode_owed());
        p.end_iteration(secs(0.1), 1);
    }
    assert!(!p.decode_owed());
}

#[test]
fn share_holds_over_many_cycles() {
    for share in [25u8, 50, 75] {
        let mut p = pacer(500, share);
        let (chunk_s, decode_s) = drive(&mut p, 0.25, 0.00013, 0.1, 5000);
        // Each chunk iteration carries one decode step (0.1 s) itself.
        let chunks = chunk_s / (0.25 + 0.00013 * p.budget() as f64);
        let got = (decode_s + 0.1 * chunks) / (chunk_s + decode_s);
        let want = f64::from(share) / 100.0;
        assert!((got - want).abs() < 0.03, "share {share}: got {got:.3}");
    }
}

#[test]
fn zero_share_never_owes_decode() {
    let mut p = pacer(500, 0);
    p.end_iteration(secs(0.1), 1);
    p.note_chunk(512);
    p.end_iteration(secs(2.0), 1);
    assert!(!p.decode_owed());
}

#[test]
fn full_share_always_owes_decode() {
    let p = pacer(500, 100);
    assert!(p.decode_owed());
}

#[test]
fn nothing_is_owed_once_nobody_decodes() {
    let mut p = pacer(500, 50);
    p.note_chunk(512);
    p.end_iteration(secs(0.5), 1);
    assert!(p.decode_owed());
    p.idle();
    assert!(!p.decode_owed());

    // An iteration nobody waited on is neither charged nor a sample.
    let before = p.budget();
    p.note_chunk(before);
    p.end_iteration(secs(9.0), 0);
    assert!(!p.decode_owed());
    assert_eq!(p.budget(), before);
    assert_eq!(p.decode, None);
}

#[test]
fn decode_step_time_is_averaged_per_batch_size() {
    let mut p = pacer(500, 50);
    p.end_iteration(secs(0.1), 1);
    p.end_iteration(secs(0.2), 1);
    let (batch, step) = p.decode.unwrap();
    assert_eq!(batch, 1);
    assert!((step - 0.12).abs() < 1e-9, "EWMA 0.2: {step}");
    // Another batch size is another step time: no blending.
    p.end_iteration(secs(0.3), 2);
    assert_eq!(p.decode, Some((2, 0.3)));
}

#[test]
fn cached_or_unsized_step_is_neither_sample_nor_charge() {
    let mut p = pacer(500, 50);
    let before = p.budget();
    p.note_chunk(0);
    p.end_iteration(secs(2.0), 1);
    assert!(!p.decode_owed());
    assert_eq!(p.budget(), before);
    assert_eq!(p.decode, None, "not a decode-only step either");
}

#[test]
fn reset_forgets_the_measurements() {
    let mut p = pacer(500, 50);
    drive(&mut p, 0.25, 0.00013, 0.1, 50);
    p.reset();
    assert_eq!(p.budget(), START_CHUNK);
    assert_eq!(p.decode, None);
    assert_eq!(p.floor_s, None);
    assert!(!p.decode_owed());
    assert_eq!(p.config(), PacerConfig::from_cli(500, 50).unwrap());
}

// ── Chunk plan ──

fn plan(offset: usize, total: usize, skip_to: usize, cached: usize) -> ChunkPlan {
    plan_chunk(offset, total, skip_to, cached, 512, MAX_CHUNK, BLOCK)
}

#[test]
fn middle_chunk_of_a_cold_prompt_rides_with_decode() {
    assert_eq!(
        plan(512, 10_000, 0, 0),
        ChunkPlan {
            len: 512,
            is_last: false,
            fuse_ok: true
        }
    );
}

#[test]
fn chunk_zero_takes_the_standard_path_for_the_cache_lookup() {
    assert_eq!(
        plan(0, 10_000, 0, 0),
        ChunkPlan {
            len: 512,
            is_last: false,
            fuse_ok: false
        }
    );
}

#[test]
fn chunk_end_lands_on_a_block_boundary() {
    // 8196 = an inline chunk 0 of max_batch_tokens.
    let p = plan(8196, 20_000, 0, 0);
    assert_eq!((8196 + p.len) % BLOCK, 0);
    assert_eq!(p.len, 508);
}

#[test]
fn intermediate_chunk_length_is_a_multiple_of_four() {
    let p = plan(8195, 20_000, 0, 0);
    assert_eq!(p.len % 4, 0);
    assert!(!p.is_last);
}

#[test]
fn cached_prefix_is_crossed_in_full_chunks_on_the_standard_path() {
    // 50 000 tokens cached: no work, so no reason for small steps.
    let p = plan(512, 60_000, 50_000, 50_000);
    assert_eq!((p.len, p.fuse_ok), (MAX_CHUNK, false));
    // The chunk that leaves the cache: the cached rest + one budget.
    let p = plan(49_664, 60_000, 50_000, 50_000);
    assert_eq!((p.len, p.fuse_ok), (336 + 512, false));
    // Past the cache the chunks may ride with decode.
    let p = plan(50_512, 60_000, 50_000, 50_000);
    assert_eq!((p.len, p.fuse_ok), (512, true));
}

#[test]
fn ssm_replay_window_stays_on_the_standard_path() {
    // Snapshot at 49 000, cache match 50 000: [49 000, 50 000) is replayed
    // through the SSM layers without rewriting the shared KV blocks.
    let p = plan(49_512, 60_000, 49_000, 50_000);
    assert!(!p.fuse_ok);
    assert_eq!(49_512 + p.len, 50_016);
    assert!(plan(50_016, 60_000, 49_000, 50_000).fuse_ok);
}

#[test]
fn chunk_reaching_the_tail_cut_becomes_the_last_chunk() {
    // total 1560: last full block boundary 1552, cut 1536.
    assert_eq!(
        plan(1024, 1560, 0, 0),
        ChunkPlan {
            len: 536,
            is_last: true,
            fuse_ok: false
        }
    );
    // total 1100: cut 1072, the chunk ends at 1024 — before it.
    assert_eq!(
        plan(512, 1100, 0, 0),
        ChunkPlan {
            len: 512,
            is_last: false,
            fuse_ok: true
        }
    );
    assert!(plan(1024, 1100, 0, 0).is_last);
}

#[test]
fn chunk_that_cannot_reach_the_end_stops_before_the_tail_cut() {
    // 8200 tokens cached of 8220: one max chunk would end at 8192 = cut
    // and the 28-token rest would never split for the tail checkpoint.
    let p = plan(0, 8220, 8200, 8200);
    assert_eq!((p.len, p.is_last), (8176, false));
    assert!(plan(8176, 8220, 8200, 8200).is_last);
}

#[test]
fn short_prompt_is_one_last_chunk() {
    assert_eq!(
        plan(0, 300, 0, 0),
        ChunkPlan {
            len: 300,
            is_last: true,
            fuse_ok: false
        }
    );
    assert!(plan(0, 20, 0, 0).is_last);
}

#[test]
fn pacer_plan_uses_its_budget_under_the_fuse_cap() {
    let p = pacer(500, 50);
    assert_eq!(p.plan(512, 10_000, 0, 0, usize::MAX, MAX_CHUNK).len, 512);
    assert_eq!(p.plan(512, 10_000, 0, 0, 256, MAX_CHUNK).len, 256);
}

#[test]
fn noisy_timings_keep_the_chunk_near_the_target() {
    // ±15 % jitter on every measurement (deterministic LCG). Target chunk
    // is 1920 tokens (see the fixed-cost test); the regulator must neither
    // collapse to the minimum nor run away to the full chunk.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut jitter = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        0.85 + 0.30 * ((state >> 33) as f64 / (1u64 << 31) as f64)
    };
    let mut p = pacer(500, 50);
    let (mut lo, mut hi) = (usize::MAX, 0);
    for step in 0..4000 {
        if p.decode_owed() {
            p.end_iteration(secs(0.1 * jitter()), 1);
        } else {
            let n = p.budget();
            p.note_chunk(n);
            p.end_iteration(secs((0.25 + 0.00013 * n as f64) * jitter()), 1);
        }
        if step >= 200 {
            lo = lo.min(p.budget());
            hi = hi.max(p.budget());
        }
    }
    assert!(lo >= 960 && hi <= 3840, "chunk wandered to {lo}..{hi}");
}

#[test]
fn whole_prompt_is_walked_in_steps_no_longer_than_the_pause() {
    // The scheduler's loop against the measured machine shape: a 40 000
    // token prompt with 20 000 tokens cached, one session decoding.
    let (total, cached) = (40_000usize, 20_000usize);
    let mut p = pacer(500, 50);
    let (mut offset, mut skip_to) = (0usize, 0usize);
    let (mut longest, mut chunk_s, mut decode_s, mut fused) = (0.0f64, 0.0, 0.0, 0);
    while offset < total {
        if p.decode_owed() {
            p.end_iteration(secs(0.1), 1);
            decode_s += 0.1;
            continue;
        }
        p.note_chunk(0);
        let plan = p.plan(offset, total, skip_to, skip_to, MAX_CHUNK, MAX_CHUNK);
        assert!(plan.len > 0);
        assert_eq!(plan.is_last, offset + plan.len == total);
        if offset == 0 {
            skip_to = cached; // the chunk-0 lookup finds the cached prefix
        }
        let computed = (offset + plan.len).saturating_sub(offset.max(skip_to));
        assert!(!plan.fuse_ok || offset >= cached, "fused inside the cache");
        fused += usize::from(plan.fuse_ok);
        offset += plan.len;
        p.note_chunk(computed);
        let t = if computed == 0 {
            0.1
        } else {
            0.25 + 0.00013 * computed as f64
        };
        p.end_iteration(secs(t), 1);
        longest = longest.max(t);
        chunk_s += t;
    }
    assert_eq!(offset, total);
    assert!(longest <= 0.55, "longest step {longest:.3} s");
    // 20 000 computed tokens in ~1920-token steps, not in 128-token crumbs.
    assert!((8..=16).contains(&fused), "{fused} fused steps");
    let share = decode_s / (chunk_s + decode_s);
    assert!((0.3..=0.5).contains(&share), "decode-only share {share:.2}");
}

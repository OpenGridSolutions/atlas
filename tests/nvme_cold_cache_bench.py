#!/usr/bin/env python3
"""
nvme_cold_cache_bench.py — T-NVM-12 / T-BAT-02 on-box: force KV-pool
overcommit so sessions are evicted to the NVMe tier and restored from it,
and measure whether a warm streaming session keeps its tokens/s while the
tier churns in the background.

Phases:
  1. baseline  — one warm stream alone (per-token timings).
  2. fill      — many cold sessions with long prompts until the pool
                 overcommits; every finish persists to the store.
  3. churn     — the warm stream again, with cold sessions cycling in the
                 background (evict LRU → restore from NVMe), sampling
                 /internal/state; then /metrics deltas and store files.

Usage:
  python tests/nvme_cold_cache_bench.py --url http://127.0.0.1:8899 \
      --model <id> --cache-dir /var/tmp/atlas-nvme-bench [--token TOKEN]
"""

import argparse
import json
import os
import statistics
import threading
import time
from urllib.request import Request, urlopen


def headers(token):
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def get(url, path, token, raw=False):
    with urlopen(Request(f"{url}{path}", headers=headers(token)), timeout=20) as r:
        data = r.read()
        return data.decode() if raw else json.loads(data)


def metric(text, name):
    for line in text.splitlines():
        if line.startswith(name + " ") or line.startswith(name + "{"):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def chat(url, model, prompt, max_tokens, token, stream=True):
    """Returns dict: ttft, token_gaps (s), tokens, usage, status, elapsed."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": stream,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    t0 = time.monotonic()
    rec = {"ttft": None, "gaps": [], "tokens": 0, "usage": None, "status": None}
    resp = urlopen(Request(f"{url}/v1/chat/completions", data=body, headers=headers(token)), timeout=900)
    rec["status"] = resp.status
    if not stream:
        data = json.loads(resp.read())
        rec["usage"] = data.get("usage")
        rec["elapsed"] = time.monotonic() - t0
        return rec
    last = None
    for raw in resp:
        line = raw.decode(errors="replace").rstrip("\n")
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        if chunk.get("usage"):
            rec["usage"] = chunk["usage"]
        for ch in chunk.get("choices", []):
            if ch.get("delta", {}).get("content") or ch.get("delta", {}).get("reasoning_content"):
                now = time.monotonic()
                if rec["ttft"] is None:
                    rec["ttft"] = now - t0
                elif last is not None:
                    rec["gaps"].append(now - last)
                last = now
                rec["tokens"] += 1
    rec["elapsed"] = time.monotonic() - t0
    return rec


def gap_stats(gaps):
    if not gaps:
        return {}
    g = sorted(gaps)
    return {
        "tok_s": round(1.0 / statistics.mean(g), 1),
        "tbt_p50_ms": round(statistics.median(g) * 1000, 1),
        "tbt_p95_ms": round(g[int(len(g) * 0.95) - 1] * 1000, 1),
        "tbt_max_ms": round(g[-1] * 1000, 1),
        "stalls_gt_250ms": sum(1 for x in g if x > 0.25),
    }


def cached_tokens(rec):
    u = rec.get("usage") or {}
    return (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0), u.get("prompt_tokens")


def store_files(cache_dir):
    if not cache_dir or not os.path.isdir(cache_dir):
        return None
    counts, size = {}, 0
    for root, _, files in os.walk(cache_dir):
        for f in files:
            ext = f.rsplit(".", 1)[-1]
            counts[ext] = counts.get(ext, 0) + 1
            size += os.path.getsize(os.path.join(root, f))
    return {"files": counts, "bytes": size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8899")
    ap.add_argument("--model", required=True)
    ap.add_argument("--token", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--warm-tokens", type=int, default=300)
    ap.add_argument("--cold-words", type=int, default=2200)
    ap.add_argument("--cold-sessions", type=int, default=0, help="0 = derive from pool size")
    ap.add_argument("--churn-rounds", type=int, default=2)
    args = ap.parse_args()

    state = get(args.url, "/internal/state", args.token)
    profile = state.get("capacity_profile") or {}
    pool_tokens = profile.get("pool_tokens_total", 0)
    print("kv_pool:", json.dumps(state.get("kv_pool")), "pool_tokens:", pool_tokens)
    print("nvme_store:", json.dumps(state.get("nvme_store")))
    print("slots:", json.dumps(state.get("slots")))
    m0 = get(args.url, "/metrics", args.token, raw=True)

    filler = ("The expedition logged temperature, humidity, wind and soil samples every hour; "
              "the field notes describe each site, its geology, flora and the instruments used. ")
    cold_base = " ".join([filler] * (args.cold_words // 25))
    warm_prompt = ("You are a patient tutor. " + " ".join([filler] * 40) +
                   " Now explain, step by step and at length, how a transformer attends over a long context.")

    # ── 1. baseline ──
    print("\n== 1. baseline warm stream (alone)")
    base = chat(args.url, args.model, warm_prompt, args.warm_tokens, args.token)
    print("  ttft_s=%.2f" % base["ttft"], gap_stats(base["gaps"]), "cached/prompt:", cached_tokens(base))
    base2 = chat(args.url, args.model, warm_prompt, args.warm_tokens, args.token)
    print("  repeat (prefix hit): ttft_s=%.2f" % base2["ttft"], gap_stats(base2["gaps"]), "cached/prompt:", cached_tokens(base2))

    # ── 2. fill ──
    approx_cold_tokens = int(args.cold_words * 1.35)
    n_cold = args.cold_sessions or max(6, (pool_tokens // max(approx_cold_tokens, 1)) + 4)
    print(f"\n== 2. fill: {n_cold} cold sessions × ~{approx_cold_tokens} tokens (pool {pool_tokens} tokens)")
    cold_prompts = [f"Site report {i}: {cold_base} Summarise site {i} in one sentence." for i in range(n_cold)]
    t_fill = time.monotonic()
    for i, p in enumerate(cold_prompts):
        r = chat(args.url, args.model, p, 12, args.token, stream=False)
        if i % 4 == 0:
            s = get(args.url, "/internal/state", args.token)
            print(f"  cold {i}: prompt={r['usage']['prompt_tokens']} kv_pool={s['kv_pool']} queue_bytes={s['nvme_store']['write_queue_bytes'] if s.get('nvme_store') else None}")
    print("  fill took %.1f s" % (time.monotonic() - t_fill))
    time.sleep(2)
    s = get(args.url, "/internal/state", args.token)
    print("  after fill: kv_pool", s["kv_pool"], "nvme_store", s.get("nvme_store"))
    print("  store files:", store_files(args.cache_dir))

    # ── 3. churn ──
    print(f"\n== 3. warm stream under NVMe churn ({args.churn_rounds} rounds over {n_cold} cold sessions)")
    samples, churn_log, stop = [], [], threading.Event()

    def churn():
        for rnd in range(args.churn_rounds):
            for i, p in enumerate(cold_prompts):
                if stop.is_set():
                    return
                r = chat(args.url, args.model, p, 12, args.token, stream=False)
                churn_log.append({"round": rnd, "i": i, "cached": cached_tokens(r)[0], "elapsed": round(r["elapsed"], 2)})

    def sampler():
        while not stop.is_set():
            try:
                s = get(args.url, "/internal/state", args.token)
                samples.append({"used": s["kv_pool"]["used_blocks"],
                                "qb": (s.get("nvme_store") or {}).get("write_queue_bytes", 0),
                                "rej": (s.get("nvme_store") or {}).get("write_queue_rejections", 0)})
            except Exception:
                pass
            time.sleep(0.25)

    th_c = threading.Thread(target=churn, daemon=True)
    th_s = threading.Thread(target=sampler, daemon=True)
    th_c.start(); th_s.start()
    time.sleep(1.0)
    warm_prompt2 = warm_prompt + " Begin with a short recap of the field notes."
    churned = chat(args.url, args.model, warm_prompt2, args.warm_tokens, args.token)
    print("  warm under churn: ttft_s=%.2f" % churned["ttft"], gap_stats(churned["gaps"]), "cached/prompt:", cached_tokens(churned))
    churned2 = chat(args.url, args.model, warm_prompt2, args.warm_tokens, args.token)
    print("  warm under churn (2nd): ttft_s=%.2f" % churned2["ttft"], gap_stats(churned2["gaps"]), "cached/prompt:", cached_tokens(churned2))
    th_c.join()
    stop.set(); th_s.join()

    m1 = get(args.url, "/metrics", args.token, raw=True)
    s = get(args.url, "/internal/state", args.token)
    print("\n== results")
    print("  baseline      :", gap_stats(base["gaps"]))
    print("  prefix-hit    :", gap_stats(base2["gaps"]))
    print("  under churn   :", gap_stats(churned["gaps"]))
    print("  under churn 2 :", gap_stats(churned2["gaps"]))
    print("  churn requests:", len(churn_log), "restored-from-cache (cached_tokens>0):",
          sum(1 for c in churn_log if c["cached"] > 0), "mean elapsed s:",
          round(statistics.mean(c["elapsed"] for c in churn_log), 2) if churn_log else None)
    if samples:
        print("  kv_pool used blocks: min=%d max=%d | write_queue_bytes max=%d | rejections=%d" % (
            min(x["used"] for x in samples), max(x["used"] for x in samples),
            max(x["qb"] for x in samples), max(x["rej"] for x in samples)))
    for name in ("atlas_nvme_write_bytes_total", "atlas_nvme_read_bytes_total", "atlas_nvme_restored_tokens_total"):
        print(f"  Δ{name} = {metric(m1, name) - metric(m0, name):.0f}")
    print("  nvme_store:", json.dumps(s.get("nvme_store")))
    print("  store files:", store_files(args.cache_dir))


if __name__ == "__main__":
    main()

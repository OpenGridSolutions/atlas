#!/usr/bin/env python3
"""
slot_concurrency_bench.py — T-SLOT-01 on-box: raise the number of parallel
sessions step by step past `slots_total` and record what happens to the
surplus (reject / queue / drop).

Each level launches N concurrent streaming chat requests with distinct
prompts, samples `/internal/state` while they run, and reports HTTP
status, TTFT and the peak number of admitted sequences.

Usage:
  python tests/slot_concurrency_bench.py --url http://127.0.0.1:8899 \
      --model <id> --levels 1,2,4,6,8,12 [--token TOKEN]
"""

import argparse
import json
import statistics
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def headers(token):
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def get_json(url, path, token):
    with urlopen(Request(f"{url}{path}", headers=headers(token)), timeout=10) as r:
        return json.loads(r.read())


def stream_chat(url, model, prompt, max_tokens, token, out, idx):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    t0 = time.monotonic()
    rec = {"idx": idx, "status": None, "ttft": None, "total": None,
           "tokens": 0, "finish": None, "error": None, "queued_comment": False}
    try:
        resp = urlopen(Request(f"{url}/v1/chat/completions", data=body,
                               headers=headers(token)), timeout=600)
        rec["status"] = resp.status
        for raw in resp:
            line = raw.decode(errors="replace").rstrip("\n")
            if line.startswith(":"):
                rec["queued_comment"] = True
                continue
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if "error" in chunk:
                rec["error"] = chunk["error"].get("code") or chunk["error"].get("message")
                break
            for ch in chunk.get("choices", []):
                if ch.get("delta", {}).get("content"):
                    if rec["ttft"] is None:
                        rec["ttft"] = time.monotonic() - t0
                    rec["tokens"] += 1
                if ch.get("finish_reason"):
                    rec["finish"] = ch["finish_reason"]
        rec["total"] = time.monotonic() - t0
    except HTTPError as e:
        rec["status"] = e.code
        rec["total"] = time.monotonic() - t0
        try:
            rec["error"] = json.loads(e.read())["error"].get("code")
        except Exception:
            rec["error"] = f"http_{e.code}"
    except Exception as e:  # connection drop etc.
        rec["status"] = "drop"
        rec["error"] = repr(e)[:80]
        rec["total"] = time.monotonic() - t0
    out[idx] = rec


def sample_state(url, token, stop, samples):
    while not stop.is_set():
        try:
            s = get_json(url, "/internal/state", token)
            slots = s.get("slots") or {}
            sched = s.get("scheduler") or {}
            samples.append({
                "t": time.monotonic(),
                "occupied": slots.get("occupied", s["batch"]["scheduler_load"]),
                "queues": sum(q["priority_len"] + q["normal_len"] for q in sched.get("queues", [])),
            })
        except Exception:
            pass
        time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8899")
    ap.add_argument("--model", required=True)
    ap.add_argument("--levels", default="1,2,4,6,8,12")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--prompt-words", type=int, default=900)
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    state = get_json(args.url, "/internal/state", args.token)
    slots = state.get("slots") or {}
    print("slots:", json.dumps(slots))
    print("kv_pool:", json.dumps(state.get("kv_pool")))
    slots_total = slots.get("slots_total")

    filler = ("Atlas serves many sessions and every session keeps its own context. "
              "Describe the water cycle, the carbon cycle and plate tectonics in detail. ")
    base_prompt = " ".join([filler] * (args.prompt_words // 20))

    results = {}
    for level in [int(x) for x in args.levels.split(",")]:
        out = {}
        samples = []
        stop = threading.Event()
        sampler = threading.Thread(target=sample_state, args=(args.url, args.token, stop, samples), daemon=True)
        sampler.start()
        threads = []
        t_level = time.monotonic()
        for i in range(level):
            prompt = f"Session {level}-{i}: {base_prompt} Finally, write a {i+3}-item numbered list about rivers."
            th = threading.Thread(target=stream_chat, args=(args.url, args.model, prompt, args.max_tokens, args.token, out, i))
            th.start()
            threads.append(th)
            time.sleep(0.05)
        for th in threads:
            th.join()
        stop.set()
        wall = time.monotonic() - t_level
        recs = [out[i] for i in range(level)]
        statuses = {}
        for r in recs:
            statuses[str(r["status"])] = statuses.get(str(r["status"]), 0) + 1
        ttfts = sorted(r["ttft"] for r in recs if r["ttft"] is not None)
        peak = max((s["occupied"] for s in samples), default=None)
        peak_queue = max((s["queues"] for s in samples), default=None)
        tokens = sum(r["tokens"] for r in recs)
        row = {
            "level": level, "statuses": statuses, "peak_occupied": peak,
            "peak_arena_queue": peak_queue,
            "ttft_min_s": round(ttfts[0], 2) if ttfts else None,
            "ttft_med_s": round(statistics.median(ttfts), 2) if ttfts else None,
            "ttft_max_s": round(ttfts[-1], 2) if ttfts else None,
            "wall_s": round(wall, 1), "agg_tok_s": round(tokens / wall, 1) if wall else None,
            "finish": sorted({str(r["finish"]) for r in recs}),
            "errors": sorted({str(r["error"]) for r in recs if r["error"]}),
            "sse_comments": sum(1 for r in recs if r["queued_comment"]),
        }
        results[level] = row
        print(json.dumps(row))
        if slots_total and level >= 2 * slots_total:
            break

    print("\n== summary")
    print(f"slots_total={slots_total}")
    for level, row in results.items():
        verdict = "all admitted" if (row["peak_occupied"] or 0) >= level else \
            f"queued: peak {row['peak_occupied']} admitted, TTFT {row['ttft_min_s']}..{row['ttft_max_s']} s"
        print(f"N={level:3d}  status={row['statuses']}  {verdict}  errors={row['errors']}")


if __name__ == "__main__":
    sys.exit(main())

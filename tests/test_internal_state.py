#!/usr/bin/env python3
"""
test_internal_state.py — T-TEL-01: `/internal/state` and the session probe.

Checks, against a live server, that the TEL-3 introspection surface is
reachable, bearer-gated, and self-consistent with the slot/pool state,
and that `POST /v1/atlas/session/state` answers for the active model.

Usage:
  python tests/test_internal_state.py --url http://host:port [--token TOKEN]

Exit status is non-zero on the first failed check.
"""

import argparse
import json
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def call(url, path, token=None, body=None, timeout=30):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = Request(f"{url}{path}", data=data, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def check(cond, label):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8888")
    ap.add_argument("--token", default=None, help="bearer token when --require-auth is on")
    args = ap.parse_args()

    print("== /internal/state (TEL-3)")
    status, state = call(args.url, "/internal/state", args.token)
    check(status == 200, f"GET /internal/state → {status}")
    for field in ("arena", "sidecars", "capacity_profile", "kv_pool", "slots",
                  "batch", "scheduler", "swap_costs", "nvme_store"):
        check(field in state, f"field '{field}' present")

    pool = state.get("kv_pool")
    if pool:
        check(pool["used_blocks"] <= pool["total_blocks"], "kv_pool.used ≤ total")
        check(pool["free_blocks"] == pool["total_blocks"] - pool["used_blocks"],
              "kv_pool.free = total − used")

    slots = state.get("slots")
    if slots:
        check(slots["occupied"] <= slots["slots_total"], "slots.occupied ≤ slots_total")
        check(slots["slot_buffer_tokens"] < slots["slot_tokens"], "slot buffer < slot tokens")
        check(slots["priority_reserved"] < slots["slots_total"], "reserved < slots_total")
        check(slots["occupied"] == state["batch"]["scheduler_load"],
              "slots.occupied matches batch.scheduler_load")

    profile = state.get("capacity_profile")
    if profile:
        check(profile["parallel_seqs"]["at_8k"] >= profile["parallel_seqs"]["at_32k"],
              "capacity profile monotone in context length")

    sched = state.get("scheduler")
    if sched:
        check("queues" in sched and "last_decision" in sched, "scheduler fragment complete")
        for q in sched["queues"]:
            check(q["priority_len"] >= 0 and q["normal_len"] >= 0, f"queue '{q['model']}' lengths")

    print("== /v1/atlas/session/state (E2)")
    probe = {"messages": [{"role": "user", "content": "ping"}]}
    status, body = call(args.url, "/v1/atlas/session/state", args.token, probe)
    check(status == 200, f"POST /v1/atlas/session/state → {status}")
    for field in ("slot", "ram_cache", "nvme"):
        check(field in body, f"probe field '{field}' present")

    print("== TEL-7: bearer gate")
    status, _ = call(args.url, "/internal/state", None)
    if args.token:
        check(status == 401, f"no token → {status} (expected 401)")
    else:
        print(f"  SKIP  auth disabled on this server (status {status} without token)")

    print("all checks passed")


if __name__ == "__main__":
    main()

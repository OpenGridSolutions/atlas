#!/usr/bin/env python3
"""On-box integration test for E7 context compaction (Runde B, Teil 1).

Covers T-CMP-01/03/04/07/08/13/14/15/16 plus the streaming keep-alive
path (CMP-10), the Anthropic surface (T-API-01) and /internal/state
(TEL-3). Run against a server started with a SMALL window so the
history overflows quickly, e.g.

    ATLAS_MAX_SEQ_LEN=4096 ./serve.sh
    python3 tests/test_compaction.py --model <id>

The script never relies on the exact token count of the filler: it
grows the history until the server reports a compaction.
"""

import argparse
import json
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def call(url, path, body=None, headers=None, timeout=600, method=None):
    data = json.dumps(body).encode() if body is not None else None
    req = Request(url + path, data=data, method=method or ("POST" if data else "GET"))
    req.add_header("content-type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), json.loads(r.read() or b"{}")
    except HTTPError as e:
        try:
            payload = json.loads(e.read() or b"{}")
        except Exception:
            payload = {}
        return e.code, dict(e.headers), payload


def stream(url, path, body, headers=None, timeout=600):
    data = json.dumps(body).encode()
    req = Request(url + path, data=data, method="POST")
    req.add_header("content-type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urlopen(req, timeout=timeout) as r:
        return r.status, dict(r.headers), r.read().decode(errors="replace")


FAILS = []


def check(cond, label, extra=""):
    print(("  ok   " if cond else "  FAIL ") + label + (f"  [{extra}]" if extra else ""))
    if not cond:
        FAILS.append(label)


FILLER = (
    "This sentence is filler so that every turn occupies a reasonable number of tokens "
    "in the prompt without carrying any information worth keeping. "
)


def user_turn(n, filler_reps):
    return f"Round {n}. The code word for round {n} is ZETA-{n:03d}-KILO. " + FILLER * filler_reps


def assistant_turn(n):
    return f"Noted: round {n}, code word ZETA-{n:03d}-KILO."


def chat(url, model, messages, headers=None, max_tokens=8, stream_=False):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if stream_:
        body["stream"] = True
        return stream(url, "/v1/chat/completions", body, headers)
    return call(url, "/v1/chat/completions", body, headers)


def hdr(headers, name):
    for k, v in headers.items():
        if k.lower() == name.lower():
            return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", required=True)
    ap.add_argument("--filler", type=int, default=12, help="filler repetitions per user turn")
    ap.add_argument("--max-rounds", type=int, default=40)
    a = ap.parse_args()
    url, model = a.url, a.model
    system = {"role": "system", "content": "You are a terse assistant. Answer in one short sentence."}

    print("T-CMP-01: short history → no compaction")
    msgs = [system, {"role": "user", "content": user_turn(1, 1)}]
    st, h, body = chat(url, model, msgs)
    check(st == 200, "short request 200", str(st))
    check(hdr(h, "x-atlas-compacted") is None, "no X-Atlas-Compacted on short history")

    print("T-CMP-03 / CMP-10: grow history until the server compacts synchronously")
    msgs = [system]
    first_id, rounds, prompt_full = None, 0, 0
    for n in range(1, a.max_rounds + 1):
        msgs.append({"role": "user", "content": user_turn(n, a.filler)})
        t0 = time.time()
        st, h, body = chat(url, model, msgs)
        dt = time.time() - t0
        if st != 200:
            check(False, f"round {n} status", f"{st} {body}")
            break
        rounds = n
        msgs.append({"role": "assistant", "content": assistant_turn(n)})
        if hdr(h, "x-atlas-compacted") == "true":
            first_id = hdr(h, "x-atlas-compaction-id")
            prompt_full = body.get("usage", {}).get("prompt_tokens", 0)
            print(f"  compacted at round {n} after {dt:.1f}s, id={first_id}, prompt_tokens={prompt_full}")
            break
    check(first_id is not None and len(first_id) == 32, "compaction happened with a 32-hex id", str(first_id))
    if first_id is None:
        print("no compaction within max rounds; increase --filler or lower ATLAS_MAX_SEQ_LEN")
        sys.exit(1)

    print("T-CMP-04: follow-up with the full history + one new message reuses the entry")
    msgs.append({"role": "user", "content": "What was the code word for round 1? Answer with the code word only."})
    t0 = time.time()
    st, h, body = chat(url, model, msgs, max_tokens=32)
    dt = time.time() - t0
    answer = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    check(st == 200, "follow-up 200", str(st))
    check(hdr(h, "x-atlas-compacted") == "true", "follow-up carries X-Atlas-Compacted")
    check(hdr(h, "x-atlas-compaction-id") == first_id, "same compaction id recognised", str(hdr(h, "x-atlas-compaction-id")))
    check(body.get("usage", {}).get("prompt_tokens", 10**9) <= prompt_full + 256, "prompt stays near the compacted size")
    cached = body.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens")
    print(f"  follow-up {dt:.2f}s, cached_tokens={cached}, answer={answer!r}")
    check("ZETA-001-KILO" in answer, "summary preserved the round-1 code word (quality smoke)", answer.strip()[:80])
    msgs.append({"role": "assistant", "content": answer})

    print("T-CMP-08: editing a message AFTER the cut keeps the entry")
    edited = [dict(m) for m in msgs]
    edited[-2]["content"] = "What was the code word for round 2? Answer with the code word only."
    st, h, body = chat(url, model, edited, max_tokens=32)
    check(st == 200 and hdr(h, "x-atlas-compaction-id") == first_id, "entry still applied", str(hdr(h, "x-atlas-compaction-id")))

    print("T-CMP-07: editing a summarised message invalidates the entry (new compaction)")
    edited = [dict(m) for m in msgs]
    edited[1]["content"] = user_turn(1, a.filler) + " (edited)"
    st, h, body = chat(url, model, edited, max_tokens=8)
    nid = hdr(h, "x-atlas-compaction-id")
    check(st == 200, "edited history 200", str(st))
    check(nid is not None and nid != first_id, "different compaction id after edit", str(nid))

    print("T-CMP-14: X-Atlas-Context-Policy: reject → 400 context_length_exceeded")
    st, h, body = chat(url, model, msgs, headers={"X-Atlas-Context-Policy": "reject"})
    check(st == 400, "reject policy 400", str(st))
    check(body.get("error", {}).get("code") == "context_length_exceeded", "error code", str(body.get("error", {}).get("code")))

    print("T-CMP-15: X-Atlas-Context-Policy: truncate → compaction without summary")
    trunc = [system] + msgs[1:]
    trunc[2] = {"role": "assistant", "content": assistant_turn(1) + " (truncate path)"}
    st, h, body = chat(url, model, trunc, headers={"X-Atlas-Context-Policy": "truncate"})
    tid = hdr(h, "x-atlas-compaction-id")
    check(st == 200 and hdr(h, "x-atlas-compacted") == "true", "truncate compacted", f"{st} {tid}")

    print("T-CMP-16: no header behaves like summarize (covered above); invalid header → 400")
    st, h, body = chat(url, model, msgs[:2], headers={"X-Atlas-Context-Policy": "drop"})
    check(st == 400 and body.get("error", {}).get("code") == "invalid_context_policy", "invalid policy value 400")

    print("T-CMP-13 / CTX-6: a single message larger than the window → 400 regardless of policy")
    huge = [system, {"role": "user", "content": FILLER * 400}]
    st, h, body = chat(url, model, huge)
    check(st == 400 and body.get("error", {}).get("code") == "context_length_exceeded", "huge message 400", str(st))
    check("newest message" in body.get("error", {}).get("message", ""), "CTX-6 message names the sizes")

    print("CMP-10 streaming: keep-alive comments while compacting, then the stream")
    smsgs = [system]
    for n in range(1, rounds + 1):
        smsgs.append({"role": "user", "content": user_turn(n, a.filler) + " (stream)"})
        smsgs.append({"role": "assistant", "content": assistant_turn(n)})
    smsgs.append({"role": "user", "content": "Say OK."})
    st, h, text = chat(url, model, smsgs, stream_=True)
    check(st == 200, "stream 200", str(st))
    check(": compacting" in text, "': compacting' keep-alive seen")
    check(": compacted id=" in text, "': compacted id=' comment seen")
    check("data: [DONE]" in text, "stream completed")

    print("T-API-01: Anthropic /v1/messages carries the compaction headers")
    amsgs = [{"role": m["role"], "content": m["content"]} for m in msgs[1:]]
    st, h, body = call(url, "/v1/messages", {"model": model, "max_tokens": 8, "system": system["content"], "messages": amsgs})
    check(st == 200, "anthropic 200", f"{st} {str(body)[:120]}")
    check(hdr(h, "x-atlas-compacted") == "true", "anthropic response has X-Atlas-Compacted", str(hdr(h, "x-atlas-compaction-id")))

    print("TEL-3: /internal/state reports compaction counters")
    st, h, body = call(url, "/internal/state")
    comp = body.get("compaction", {})
    check(st == 200 and comp.get("created", {}).get("summarize", 0) >= 2, "created counter", json.dumps(comp)[:200])
    check(comp.get("applied_requests", 0) >= 2, "applied counter")
    check(comp.get("rejected", {}).get("context_length_exceeded", 0) >= 1, "ctx6 counter")

    print(f"\n{len(FAILS)} failures" + (": " + ", ".join(FAILS) if FAILS else ""))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

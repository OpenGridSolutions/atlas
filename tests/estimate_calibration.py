#!/usr/bin/env python3
"""Calibrate the `[estimate]` constants of MODEL.toml on the box.

Deterministic admission (docs/design/deadline-admission.md §Frage 3)
needs per-model cost constants that are measured ONCE and committed —
never learned at runtime. This script measures them against a running
server and prints the `[estimate]` block to paste into
`kernels/gb10/<model>/MODEL.toml`.

Generative models (`--model <id>`, repeatable):
  prefill_tok_s   from TTFT of a long prompt with max_tokens=1
  decode_step_ms  at batch 1, 2, 4, 8: concurrent streams of 256 tokens,
                  step time = decode wall time / tokens of one stream
OCR (`--ocr <pdf>`): per-page wall time via /convert, pages one at a
time, p50/p90 over the pages → ocr_page_secs_p90; tokens per page from
/v1/ocr usage (ocr_page_tokens_p90).

Run against 127.0.0.1 (no proxy timeout). The arena swaps models as
requests arrive, so one run covers every preseeded model.

Usage:
  python3 tests/estimate_calibration.py --base-url http://127.0.0.1:11434 \
      --model nvidia/Qwen3.8-27B-NVFP4 --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
      --ocr tests/fixtures/sample.pdf
"""

import argparse
import base64
import concurrent.futures as cf
import json
import os
import statistics
import sys
import time
import urllib.request
import uuid

HDR = {"Content-Type": "application/json", "Accept": "application/json"}
tok = os.environ.get("ATLAS_TOKEN")
if tok:
    HDR["Authorization"] = f"Bearer {tok}"


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, json.dumps(body).encode(), HDR)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def stream_tokens(url, body, timeout=1800):
    """Return (ttft_s, decode_s, n_tokens) for one SSE chat stream."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url, json.dumps(body).encode(), HDR)
    t0 = time.time()
    first = None
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            ev = json.loads(line[6:])
            for ch in ev.get("choices", []):
                if ch.get("delta", {}).get("content") or ch.get("delta", {}).get("reasoning_content"):
                    if first is None:
                        first = time.time()
                    n += 1
            if ev.get("usage"):
                n = ev["usage"].get("completion_tokens", n)
    end = time.time()
    if first is None:
        first = end
    return first - t0, end - first, n


def long_prompt(words):
    base = "Die Deckungsbeitragsrechnung ordnet Kosten und Erloese den Produkten zu. "
    return (base * (words // 10 + 1))[: words * 6]


def calibrate_generative(base, model, repeats):
    chat = f"{base}/v1/chat/completions"
    # Warm the model (swap in) once.
    post(chat, {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
    # Prefill: long prompt, one token out. prompt_tokens from usage.
    rates = []
    for _ in range(repeats):
        body = {"model": model, "messages": [{"role": "user", "content": long_prompt(1500)}],
                "max_tokens": 1, "temperature": 0}
        t0 = time.time()
        r = post(chat, body)
        dt = time.time() - t0
        ptoks = r["usage"]["prompt_tokens"]
        rates.append(ptoks / dt)
    prefill = statistics.median(rates)
    # Decode step time at batch 1/2/4/8.
    steps = []
    for batch in (1, 2, 4, 8):
        body = {"model": model, "messages": [{"role": "user", "content": "Erklaere die Deckungsbeitragsrechnung ausfuehrlich mit Beispielen."}],
                "max_tokens": 256, "temperature": 0.7, "min_tokens": 256}
        with cf.ThreadPoolExecutor(max_workers=batch) as ex:
            res = list(ex.map(lambda _: stream_tokens(chat, body), range(batch)))
        per_stream = [decode / max(n, 1) for _, decode, n in res]
        steps.append(statistics.median(per_stream) * 1000.0)
        print(f"  {model}: batch {batch}: {steps[-1]:.1f} ms/step (p50 of {batch} streams)", file=sys.stderr)
    return prefill, steps


def multipart(fields, name, data):
    b = "----cal" + uuid.uuid4().hex
    body = bytearray()
    for k, v in fields.items():
        body += f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    body += f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\nContent-Type: application/pdf\r\n\r\n".encode()
    body += data + f"\r\n--{b}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={b}"


def calibrate_ocr(base, pdf_path):
    data = open(pdf_path, "rb").read()
    hdr = {k: v for k, v in HDR.items() if k != "Content-Type"}
    hdr["X-API-Key"] = tok or "cal"
    # Page count via a 1-page job's metadata is roundabout; ask /convert
    # for each page separately so each measurement is one page.
    page_times, page_tokens = [], []
    idx = 0
    while True:
        body, ct = multipart({"page_range": str(idx), "skip_cache": "true"}, os.path.basename(pdf_path), data)
        req = urllib.request.Request(f"{base}/convert", body, dict(hdr, **{"Content-Type": ct}))
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                sub = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            err = json.loads(e.read().decode() or "{}")
            if err.get("error_code") == "unprocessable_document":
                break  # page_range beyond the document
            raise
        t0 = time.time()
        while True:
            with urllib.request.urlopen(urllib.request.Request(sub["request_check_url"], headers=hdr), timeout=60) as r:
                st = json.loads(r.read().decode())
            if st["status"] not in ("pending", "processing"):
                break
            time.sleep(1)
        if st["status"] != "complete":
            print(f"  page {idx}: {st.get('error_code')} — skipped", file=sys.stderr)
            idx += 1
            continue
        page_times.append(st["runtime"])
        print(f"  page {idx}: {st['runtime']:.1f} s", file=sys.stderr)
        idx += 1
    # Tokens per page from /v1/ocr (usage) on the whole document.
    r = post(f"{base}/v1/ocr", {"image": base64.b64encode(data).decode()})
    if r.get("pages"):
        page_tokens = [r["usage"]["completion_tokens"] / len(r["pages"])]
    return page_times, page_tokens


def p90(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--ocr", help="PDF for the OCR model")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    base = a.base_url.rstrip("/")
    for m in a.model:
        print(f"== {m}", file=sys.stderr)
        prefill, steps = calibrate_generative(base, m, a.repeats)
        print(f"# {m} — measured {time.strftime('%Y-%m-%d')} with tests/estimate_calibration.py")
        print("[estimate]")
        print(f"prefill_tok_s = {prefill:.0f}")
        print("decode_step_ms = [" + ", ".join(f"{s:.1f}" for s in steps) + "]")
        print()
    if a.ocr:
        print("== OCR", file=sys.stderr)
        times, toks = calibrate_ocr(base, a.ocr)
        if times:
            print(f"# OCR — {len(times)} pages of {os.path.basename(a.ocr)}, p50 {statistics.median(times):.1f} s, p90 {p90(times):.1f} s")
            print("[estimate]")
            print(f"ocr_page_secs_p90 = {p90(times):.1f}")
            if toks:
                print(f"ocr_page_tokens_p90 = {int(toks[0])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

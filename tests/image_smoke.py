#!/usr/bin/env python3
"""/v1/images/generations smoke test for the serve matrix (Todo 3.2).

Drives the Qwen-Image-2.1 sidecar through the OpenAI Images endpoint and
checks the response schema, the PNG header (dimensions, colour type), seed
determinism, the transparent-background path and the request validation /
category errors. Small sizes and few steps keep it to about a minute after
the one-time sidecar load. Exit 0 = all pass.

Usage: python3 tests/image_smoke.py --base-url http://localhost:11434
"""

import argparse
import base64
import json
import struct
import sys
import time
import urllib.error
import urllib.request

MODEL = "Qwen/Qwen-Image-2.1"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
RGB, RGBA = 2, 6


def post(base_url: str, path: str, body: dict, timeout: int = 1800):
    """Returns (status, parsed JSON body)."""
    req = urllib.request.Request(
        f"{base_url}{path}",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def png_header(data: bytes):
    """(width, height, colour type) from the IHDR chunk."""
    if data[:8] != PNG_MAGIC or data[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    width, height, _depth, colour = struct.unpack(">IIBB", data[16:26])
    return width, height, colour


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--chat-model", default="", help="a generative model id for the category check")
    args = ap.parse_args()
    base = args.base_url
    failures = 0

    def record(name: str, problems: list, detail: str = ""):
        nonlocal failures
        if problems:
            print(f"[FAIL] {name}: {'; '.join(problems)}")
            failures += 1
        else:
            print(f"[PASS] {name}{' (' + detail + ')' if detail else ''}")

    def generate(name: str, body: dict, want_dims, want_colour):
        t0 = time.time()
        status, r = post(base, "/v1/images/generations", body)
        elapsed = time.time() - t0
        if status != 200:
            record(name, [f"status {status}: {r.get('error', {}).get('message')}"])
            return None
        problems = []
        data = r.get("data") or []
        if len(data) != body.get("n", 1):
            problems.append(f"{len(data)} images, wanted {body.get('n', 1)}")
        raw = b""
        for item in data:
            raw = base64.b64decode(item.get("b64_json", ""))
            try:
                w, h, colour = png_header(raw)
            except ValueError as e:
                problems.append(str(e))
                continue
            if (w, h) != want_dims:
                problems.append(f"{w}x{h}, wanted {want_dims[0]}x{want_dims[1]}")
            if colour != want_colour:
                problems.append(f"PNG colour type {colour}, wanted {want_colour}")
            # A flat or collapsed image compresses to almost nothing.
            if len(raw) < w * h // 20:
                problems.append(f"suspiciously small PNG ({len(raw)} bytes) — degenerate image?")
        record(name, problems, f"{elapsed:.1f}s, {len(raw) // 1024} KiB")
        return raw

    prompt = 'A neon shop sign that reads "ATLAS", rainy night'
    small = {"model": MODEL, "prompt": prompt, "size": "512x512", "num_inference_steps": 8, "seed": 7}
    first = generate("512x512 text-to-image", small, (512, 512), RGB)
    second = generate("same seed again", small, (512, 512), RGB)
    if first is not None and second is not None:
        record("seed determinism", [] if first == second else ["same seed gave different images"])
    generate(
        "aspect_ratio 16:9",
        {"prompt": prompt, "aspect_ratio": "16:9", "num_inference_steps": 4, "seed": 7},
        (1376, 768),
        RGB,
    )
    generate(
        "transparent background",
        {
            "prompt": "A cute cartoon dragon sticker.",
            "size": "512x512",
            "num_inference_steps": 8,
            "seed": 7,
            "background": "transparent",
        },
        (512, 512),
        RGBA,
    )

    # ── validation and category errors ──
    for name, body, want in [
        ("empty prompt → 400", {"prompt": " "}, 400),
        ("oversized image → 400", {"prompt": prompt, "size": "4096x4096"}, 400),
        ("unknown model → 404", {"prompt": prompt, "model": "nope/none"}, 404),
    ]:
        status, _ = post(base, "/v1/images/generations", body, timeout=60)
        record(name, [] if status == want else [f"status {status}"])
    status, r = post(
        base,
        "/v1/chat/completions",
        {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
        timeout=60,
    )
    code = r.get("error", {}).get("code")
    record(
        "image model on chat endpoint → 400",
        [] if status == 400 and code == "model_wrong_category" else [f"status {status}, code {code}"],
    )
    if args.chat_model:
        status, _ = post(
            base, "/v1/images/generations", {"prompt": prompt, "model": args.chat_model}, timeout=60
        )
        record("chat model on image endpoint → 400", [] if status == 400 else [f"status {status}"])
    status, _ = post(base, "/v1/images/edits", {}, timeout=60)
    record("image edits → 501", [] if status == 501 else [f"status {status}"])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

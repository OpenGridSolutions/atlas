#!/usr/bin/env python3
"""/v1/ocr smoke test for the serve matrix (Todo 3.3d).

Drives the Chandra-2 model-card sample images (shipped in the HF
snapshot) through /v1/ocr and checks content keywords + response-schema
invariants. Exit 0 = all pass.

Usage: python3 tests/ocr_smoke.py --base-url http://localhost:11434
"""

import argparse
import base64
import glob
import json
import os
import sys
import time
import urllib.request


CASES = [
    # (snapshot file, required keywords in markdown, expect_figure_crop)
    ("bench.png", ["olmOCR", "Chandra"], True),
    ("multilingual.png", ["Multilingual", "Chandra"], True),
    ("handwritten_form.png", ["Georgia", "Campaign"], False),
]


def snapshot_dir() -> str:
    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--datalab-to--chandra-ocr-2/snapshots/*"
    )
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        sys.exit("chandra-ocr-2 snapshot not found in the HF cache")
    return dirs[-1]


def ocr(base_url: str, image_b64: str, timeout: int = 1200) -> dict:
    req = urllib.request.Request(
        f"{base_url}/v1/ocr",
        json.dumps({"image": image_b64}).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11434")
    args = ap.parse_args()

    d = snapshot_dir()
    failures = 0
    for name, keywords, expect_crop in CASES:
        path = os.path.join(d, name)
        img = base64.b64encode(open(path, "rb").read()).decode()
        t0 = time.time()
        try:
            r = ocr(args.base_url, img)
        except Exception as e:  # noqa: BLE001 — report, don't crash the matrix
            print(f"[FAIL] {name}: request error {e}")
            failures += 1
            continue
        page = r["pages"][0]
        elapsed = time.time() - t0
        problems = []
        if not page["html"].strip():
            problems.append("empty html")
        md = page["markdown"]
        for kw in keywords:
            if kw not in md and kw not in page["html"]:
                problems.append(f"keyword '{kw}' missing")
        if not page["chunks"]:
            problems.append("no layout chunks")
        if expect_crop and not page["images"]:
            problems.append("expected a Figure/Image WebP crop")
        if problems:
            print(f"[FAIL] {name} ({elapsed:.1f}s): {'; '.join(problems)}")
            failures += 1
        else:
            print(
                f"[PASS] {name} ({elapsed:.1f}s, retries {page['retries']}, "
                f"{len(page['chunks'])} chunks, {len(page['images'])} crops)"
            )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Contract test for the Datalab-compatible `/convert` surface.

Drives Atlas the way the OpenNotes Datalab driver does (submit multipart,
poll `request_check_url`, DELETE) and checks the guarantees from
docs/design/deadline-admission.md §2b.3: page nodes 0-based with a bbox,
block ids `/page/{n}/{Type}/{i}`, polygons in the page pixel space,
Marker block types, `section_hierarchy`, `markdown` on every block,
images under the block id and under the markdown name, `error_code` on
every failure, truncation never reported as success, idempotent DELETE,
stable ids across two runs of the same file (cache).

Needs a running server with an OCR model (ATLAS_PRESEED_OCR) and the
chandra-ocr-2 HF snapshot (sample PNGs) — or `--pdf <file>`.

Usage: python3 tests/test_convert_contract.py --base-url http://127.0.0.1:11434
Exit 0 = all pass.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid

HEADERS = {
    "X-API-Key": os.environ.get("ATLAS_TOKEN", "test"),
    "User-Agent": "OpenNotes/1.0 (+https://opennotes.de)",
    "Accept": "application/json",
}

MARKER_TYPES = {
    "Page", "Caption", "Code", "Figure", "Footnote", "Form", "Equation", "ListGroup",
    "PageFooter", "PageHeader", "Picture", "SectionHeader", "Table", "Text",
    "TableOfContents", "ComplexRegion",
}


def multipart(fields, file_name, file_bytes, content_type="application/pdf"):
    boundary = "----atlas" + uuid.uuid4().hex
    body = bytearray()
    for k, v in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    body += (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{file_name}\"\r\nContent-Type: {content_type}\r\n\r\n"
    ).encode()
    body += file_bytes
    body += f"\r\n--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def request(method, url, body=None, content_type=None, timeout=30):
    headers = dict(HEADERS)
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"_raw": raw}


def submit(base, file_bytes, name, ctype, **fields):
    body, bct = multipart(fields, name, file_bytes, ctype)
    return request("POST", f"{base}/convert", body, bct)


def poll(url, max_s=900):
    t0, delay = time.time(), 2.0
    while time.time() - t0 < max_s:
        status, doc = request("GET", url)
        if status != 200:
            return status, doc
        if doc["status"] in ("pending", "processing"):
            time.sleep(delay)
            delay = min(delay * 1.5, 10.0)
            continue
        return status, doc
    raise TimeoutError(url)


class Check:
    def __init__(self):
        self.failures = []

    def __call__(self, cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            self.failures.append(msg)


def check_document(ck, doc, expect_pages):
    ck(doc["status"] == "complete" and doc["success"] is True, "complete + success")
    ck(doc["error"] == "" and doc.get("error_code") is None, "no error on success")
    ck(isinstance(doc["markdown"], str) and doc["markdown"], "document markdown present")
    ck(isinstance(doc["runtime"], (int, float)), "runtime numeric")
    ck("atlas" in doc["versions"] and "model" in doc["versions"], "versions filled")
    ck(doc["metadata"]["failed_pages"] == [], "no failed pages")
    pages = doc["json"]["children"]
    ck(len(pages) == expect_pages, f"{expect_pages} page node(s), got {len(pages)}")
    for n, page in enumerate(pages):
        ck(page["block_type"] == "Page", f"page {n} is a Page node")
        ck(re.fullmatch(rf"/page/{page['page']}/Page/{page['page']}", page["id"]) is not None, f"page id pattern {page['id']}")
        bbox = page["bbox"]
        ck(len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0, f"page {n} bbox present {bbox}")
        kids = page["children"] or []
        ck(len(kids) > 0, f"page {n} has blocks")
        for i, b in enumerate(kids):
            pid = page["page"]
            ck(b["id"] == f"/page/{pid}/{b['block_type']}/{i}", f"block id {b['id']}")
            ck(b["block_type"] in MARKER_TYPES, f"marker block type {b['block_type']}")
            ck(isinstance(b["markdown"], str), f"{b['id']} markdown is a string")
            ck(isinstance(b["html"], str), f"{b['id']} html is a string")
            poly = b["polygon"]
            ck(len(poly) == 4 and all(0 <= p[0] <= bbox[2] and 0 <= p[1] <= bbox[3] for p in poly), f"{b['id']} polygon inside page")
            ck(isinstance(b["section_hierarchy"], dict), f"{b['id']} section_hierarchy dict")
            for k, v in b["section_hierarchy"].items():
                ck(k.isdigit() and "/SectionHeader/" in v, f"{b['id']} hierarchy entry {k}→{v}")
            for name, data in b["images"].items():
                ck(name == b["id"] and isinstance(data, str) and len(data) > 100, f"{b['id']} block image under its id")
                md_name = f"_page_{pid}_{b['block_type']}_{i}.jpeg"
                ck(md_name in doc["images"], f"doc image {md_name} present")
                ck(md_name in b["markdown"], f"{b['id']} markdown references {md_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--pdf", help="PDF to convert (default: chandra sample PNG as image)")
    ap.add_argument("--max-pages", type=int, default=2)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    ck = Check()

    if args.pdf:
        file_bytes, name, ctype = open(args.pdf, "rb").read(), os.path.basename(args.pdf), "application/pdf"
        expect_pages = args.max_pages
    else:
        dirs = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--datalab-to--chandra-ocr-2/snapshots/*")))
        if not dirs:
            sys.exit("no --pdf and no chandra-ocr-2 snapshot for the sample image")
        file_bytes, name, ctype = open(os.path.join(dirs[-1], "bench.png"), "rb").read(), "bench.png", "image/png"
        expect_pages = 1

    print("1. error_code on bad input")
    status, doc = submit(base, b"not a document", "x.bin", "application/octet-stream")
    ck(status == 400 and doc["success"] is False and doc["error_code"] == "unprocessable_document", f"garbage → 400 unprocessable_document ({status} {doc.get('error_code')})")
    status, doc = request("POST", f"{base}/convert", b"{}", "application/json")
    ck(status == 400 and doc.get("error_code") == "invalid_request", f"json body → 400 invalid_request ({status})")
    status, doc = request("GET", f"{base}/convert/{uuid.uuid4().hex}")
    ck(status == 404 and doc["error_code"] == "unknown_request", "unknown id → 404 unknown_request")

    print("2. submit → poll → complete")
    t0 = time.time()
    status, sub = submit(base, file_bytes, name, ctype, mode="fast", output_format="json,markdown",
                         include_markdown_in_chunks="true", max_pages=str(args.max_pages),
                         save_checkpoint="false", skip_cache="true", extras="")
    ck(status == 200 and sub["success"] is True, f"submit accepted in {time.time() - t0:.2f}s")
    ck("request_id" in sub and sub["request_check_url"].endswith(f"/convert/{sub['request_id']}"), "request_check_url points at the job")
    ck("estimated_seconds" in sub, "estimated_seconds field present")
    status, first = poll(sub["request_check_url"])
    ck(status == 200, f"poll ended with HTTP {status}")
    if first.get("status") == "complete":
        check_document(ck, first, expect_pages)
    else:
        ck(False, f"job failed: {first.get('error_code')} {first.get('error')}")
        print(json.dumps(first, indent=1)[:2000])
        return 1

    print("3. deterministic ids across runs (cache)")
    status, sub2 = submit(base, file_bytes, name, ctype, max_pages=str(args.max_pages))
    status, second = poll(sub2["request_check_url"])
    if second.get("status") == "complete":
        ids1 = [b["id"] for p in first["json"]["children"] for b in p["children"]]
        ids2 = [b["id"] for p in second["json"]["children"] for b in p["children"]]
        ck(ids1 == ids2, f"block ids identical ({len(ids1)} blocks)")
        ck(first["markdown"] == second["markdown"], "markdown identical")
        ck(second["metadata"].get("cached") in (True, False), "cached flag present")
    else:
        ck(False, f"second run failed: {second.get('error_code')}")

    print("4. DELETE is idempotent")
    status, sub3 = submit(base, file_bytes, name, ctype, skip_cache="true")
    status, d1 = request("DELETE", sub3["request_check_url"])
    ck(status == 200 and d1["success"] is True, "first DELETE ok")
    status, d2 = request("DELETE", sub3["request_check_url"])
    ck(status == 200 and d2["success"] is True, "second DELETE ok")
    status, st = request("GET", sub3["request_check_url"])
    ck(status == 200 and st["status"] == "failed" and st["error_code"] == "cancelled", f"cancelled job reads as failed/cancelled ({st.get('error_code')})")

    print("5. deadline enforcement")
    status, sub4 = submit(base, file_bytes, name, ctype, skip_cache="true", accepted_processing_time="0.5")
    if status == 200:
        status, late = poll(sub4["request_check_url"])
        ck(late["status"] == "failed" and late["error_code"] == "deadline_exceeded", f"0.5 s deadline → failed/deadline_exceeded ({late.get('error_code')})")
    else:
        ck(status == 503 and sub4.get("error_code") == "capacity_exceeded", f"rejected up front with capacity_exceeded ({status} {sub4.get('error_code')})")

    print()
    if ck.failures:
        print(f"{len(ck.failures)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

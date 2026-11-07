#!/usr/bin/env python3
"""
switch_bench.py — model-switch and KV-session-reactivation benchmark.

Runs against a live, preseeded `spark serve` (lazy multi-model mode) and
measures, per model, what a client actually sees:

  1. switch   — the request that forces a model change: wall time and the
                `X-Atlas-Model-Swap` header (ms the request waited for the
                unload → snapshot load → rebuild of the target model).
  2. cold     — TTFT of a fresh ~N-token session (nothing cached).
  3. warm     — TTFT of the same session one turn later (RAM prefix cache).
  4. restore  — TTFT of that session after the server has served other
                models in between: its KV blocks were persisted to the NVMe
                tier on release, evicted from RAM, and are restored on
                revisit. Classified by the `/v1/atlas/session/state` probe
                taken right before the turn (ram / nvme / miss).

Generative models get sessions; embed/rerank/translate/ocr models only get
the switch request (they share the arena and swap like everything else).
Sidecars (transcribe, image) are skipped.

Usage:
  python3 tests/switch_bench.py --url http://127.0.0.1:11434 --cycles 2 \
      --new-sessions 2 --prompt-tokens 3000 [--token TOKEN] [--models a,b]

Output: <out>/events.jsonl (every request), summary.json, summary.md.
"""

import argparse
import base64
import json
import os
import random
import statistics
import struct
import time
import zlib
from urllib.error import HTTPError
from urllib.request import Request, urlopen

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
         "omicron pi rho sigma tau upsilon phi chi psi omega anchor bridge canyon "
         "dune ember forest glacier harbor island jungle kettle lantern meadow").split()
FOLLOW_UPS = ["Which word came last?", "How many distinct words were there?",
              "Name the word in the middle of the stream."]
SIDE_CAR_CATEGORIES = {"transcribe", "image"}


def tiny_png():
    """64x32 grey PNG with a dark bar — enough for an OCR request body."""
    w, h = 64, 32
    rows = []
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            row.append(40 if 10 <= y <= 20 and 8 <= x <= 56 else 230)
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


class Bench:
    def __init__(self, a):
        self.a = a
        self.out = a.out or os.path.join("logs", "switch-bench", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.out, exist_ok=True)
        self.events = open(os.path.join(self.out, "events.jsonl"), "a")
        self.sessions = {}  # model -> [session]
        self.png = tiny_png()
        # Prompts are seeded per run: the NVMe store is content-addressed, so
        # a repeated prompt text would be restored from a previous run and
        # the "cold" column would silently measure a restore.
        self.nonce = a.nonce or time.strftime("%Y%m%d-%H%M%S")

    # ── plumbing ─────────────────────────────────────────────────────
    def say(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)

    def headers(self):
        h = {"Content-Type": "application/json"}
        if self.a.token:
            h["Authorization"] = f"Bearer {self.a.token}"
        return h

    def http(self, path, body=None, timeout=60, want_headers=False):
        data = json.dumps(body).encode() if body is not None else None
        req = Request(f"{self.a.url}{path}", data=data, headers=self.headers())
        t0 = time.monotonic()
        try:
            with urlopen(req, timeout=timeout) as r:
                raw = r.read()
                hdrs = dict(r.headers.items())
                status = r.status
        except HTTPError as e:
            raw, hdrs, status = e.read(), dict(e.headers.items()), e.code
        wall = round(time.monotonic() - t0, 3)
        try:
            parsed = json.loads(raw)
        except Exception:  # noqa: BLE001 — non-JSON bodies (e.g. /metrics) are returned as text
            parsed = raw.decode(errors="replace")
        return (status, parsed, hdrs, wall) if want_headers else (status, parsed)

    def event(self, **fields):
        fields["ts"] = time.time()
        self.events.write(json.dumps(fields) + "\n")
        self.events.flush()

    def metric(self, name):
        st, text = self.http("/metrics")
        if st != 200:
            return None
        for line in str(text).splitlines():
            if line.startswith(name + " ") or line.startswith(name + "{"):
                try:
                    return float(line.rsplit(" ", 1)[1])
                except ValueError:
                    return None
        return 0.0

    def active_model(self):
        st, body = self.http("/admin/models")
        return body.get("model") if st == 200 and isinstance(body, dict) else None

    # ── discovery ────────────────────────────────────────────────────
    def discover(self):
        st, stats = self.http("/v1/stats")
        if st != 200:
            raise SystemExit(f"/v1/stats -> {st}: {stats}")
        models = [(m["id"], m["category"]) for m in stats["models"] if m["category"] not in SIDE_CAR_CATEGORIES]
        if self.a.models:
            keep = set(self.a.models.split(","))
            models = [m for m in models if m[0] in keep]
        st, state = self.http("/internal/state")
        self.event(kind="preflight", models=models, internal_state=state if st == 200 else None)
        return models

    # ── requests ─────────────────────────────────────────────────────
    def switch_request(self, model, category):
        """One minimal request at the model's own endpoint; returns a record."""
        before = self.active_model()
        if category == "generative":
            # Streaming, like a real chat client: a non-streaming request is
            # refused with 503 model_wait_too_long once the estimated swap
            # exceeds --nonstream-max-wait-s, and the SSE path opens the
            # stream before the swap, so the cost shows up as TTFT.
            rec = self.chat(model, [{"role": "user", "content": "Reply with OK."}])
            # A thinking model with a tiny budget may emit no delta at all;
            # the wall time of the 4-token request is then the best bound.
            swap_s = rec["ttft_s"] if rec["ttft_s"] is not None else (rec["wall_s"] if rec["status"] == 200 else None)
            ev = {"kind": "switch", "model": model, "category": category, "from": before,
                  "status": rec["status"], "wall_s": rec["wall_s"], "swap_ms": rec["swap_ms"],
                  "swap_s": swap_s, "swapped": before != model, "error": rec["error"]}
            self.event(**ev)
            self.say(f"switch {before and before.split('/')[-1]} -> {model.split('/')[-1]}: "
                     f"status={rec['status']} wall={rec['wall_s']}s ttft={swap_s}s")
            return ev
        if category == "embed":
            path, body = "/v1/embeddings", {"model": model, "input": ["switch bench probe"]}
        elif category == "rerank":
            path, body = "/v1/rerank", {"model": model, "query": "memory",
                                        "documents": ["KV blocks live in the pool", "a sunny beach"]}
        elif category == "translate":
            path, body = "/v1/translate", {"model": model, "source_lang_code": "de",
                                           "target_lang_code": "en", "text": "Der Speicher ist knapp."}
        elif category == "ocr":
            path, body = "/v1/ocr", {"model": model, "image": self.png, "max_tokens": 256, "max_retries": 0}
        else:
            return None
        st, resp, hdrs, wall = self.http(path, body, timeout=self.a.swap_timeout, want_headers=True)
        swap_ms = hdrs.get("x-atlas-model-swap") or hdrs.get("X-Atlas-Model-Swap")
        rec = {"kind": "switch", "model": model, "category": category, "from": before,
               "status": st, "wall_s": wall, "swap_ms": int(swap_ms) if swap_ms else None,
               "swap_s": round(int(swap_ms) / 1000, 3) if swap_ms else None,
               "swapped": before != model}
        if st != 200:
            rec["error"] = str(resp)[:400]
        self.event(**rec)
        self.say(f"switch {before and before.split('/')[-1]} -> {model.split('/')[-1]}: "
                 f"status={st} wall={wall}s header={swap_ms}ms")
        return rec

    def chat(self, model, messages):
        body = {"model": model, "messages": messages, "max_tokens": self.a.max_tokens,
                "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False}}
        if "thinking" in model.lower():
            body["thinking"] = {"type": "enabled", "budget_tokens": 16}
        rec = {"status": None, "ttft_s": None, "content": "", "usage": None, "error": None, "swap_ms": None}
        t0 = time.monotonic()
        try:
            resp = urlopen(Request(f"{self.a.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                   headers=self.headers()), timeout=self.a.chat_timeout)
            rec["status"] = resp.status
            hdr = resp.headers.get("x-atlas-model-swap")
            rec["swap_ms"] = int(hdr) if hdr else None
            for raw in resp:
                line = raw.decode(errors="replace").rstrip("\n")
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    rec["usage"] = chunk["usage"]
                for ch in chunk.get("choices", []):
                    d = ch.get("delta", {})
                    if d.get("content") or d.get("reasoning_content"):
                        if rec["ttft_s"] is None:
                            rec["ttft_s"] = round(time.monotonic() - t0, 3)
                        rec["content"] += d.get("content") or ""
        except HTTPError as e:
            rec["status"], rec["error"] = e.code, e.read().decode(errors="replace")[:500]
        except Exception as e:  # noqa: BLE001 — recorded as the request's outcome
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["wall_s"] = round(time.monotonic() - t0, 3)
        return rec

    def probe(self, model, messages):
        st, body = self.http("/v1/atlas/session/state", {"model": model, "messages": messages,
                                                         "max_tokens": self.a.max_tokens})
        return body if st == 200 and isinstance(body, dict) else {}

    # ── sessions ─────────────────────────────────────────────────────
    def make_session(self, model, idx):
        rng = random.Random(f"{self.nonce}/{model}/{idx}")
        text = " ".join(rng.choice(WORDS) for _ in range(int(self.a.prompt_tokens * 0.75)))
        prompt = (f"Session {idx} of {model}, run {self.nonce}. Memorise the following word stream, then answer in one short "
                  f"sentence: what was the first word and how many words were there?\n\n{text}")
        return {"id": f"{model.split('/')[-1]}#{idx}", "messages": [{"role": "user", "content": prompt}], "turns": 0}

    def turn(self, model, s, phase):
        pre = self.probe(model, s["messages"]) if s["turns"] else {}
        ram = (pre.get("ram_cache") or {}).get("cached_tokens", 0)
        nvme = (pre.get("nvme") or {}).get("persisted_tokens", 0)
        source = "ram" if ram else ("nvme" if nvme else ("none" if s["turns"] else "cold"))
        restored_before = self.metric("atlas_nvme_restored_tokens_total")
        rec = self.chat(model, s["messages"])
        restored_after = self.metric("atlas_nvme_restored_tokens_total")
        usage = rec.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        if source == "none" and cached and not ram:
            # The probe answers from the resident model's caches; while the
            # switch to this model is still in flight it sees nothing, yet
            # the request itself was served from a staged NVMe restore.
            source = "nvme"
        ev = {"kind": "turn", "model": model, "session": s["id"], "phase": phase, "source": source,
              "probe_ram_tokens": ram, "probe_nvme_tokens": nvme, "status": rec["status"],
              "ttft_s": rec["ttft_s"], "wall_s": rec["wall_s"], "prompt_tokens": usage.get("prompt_tokens"),
              "cached_tokens": cached, "swap_ms": rec["swap_ms"],
              "nvme_restored_delta": (restored_after or 0) - (restored_before or 0)
              if restored_before is not None and restored_after is not None else None,
              "error": rec["error"]}
        self.event(**ev)
        self.say(f"  {s['id']} {phase:8s} src={source:4s} ttft={rec['ttft_s']} prompt={usage.get('prompt_tokens')} "
                 f"cached={cached} restored_delta={ev['nvme_restored_delta']}"
                 + (f" ERROR {rec['error'][:80]}" if rec["error"] else ""))
        if rec["status"] == 200:
            s["messages"] = s["messages"] + [{"role": "assistant", "content": rec["content"] or "(no content)"},
                                             {"role": "user", "content": FOLLOW_UPS[s["turns"] % len(FOLLOW_UPS)]}]
            s["turns"] += 1
        return ev

    def run_generative(self, model, cycle):
        old = self.sessions.setdefault(model, [])
        for s in old:
            self.turn(model, s, "revisit")
        start = len(old) + 1
        for idx in range(start, start + self.a.new_sessions):
            s = self.make_session(model, idx)
            self.turn(model, s, "cold")
            self.turn(model, s, "warm")
            old.append(s)

    # ── driver ───────────────────────────────────────────────────────
    def run(self):
        models = self.discover()
        self.say(f"models: {[m for m, _ in models]}; out={self.out}")
        for cycle in range(1, self.a.cycles + 1):
            self.say(f"=== cycle {cycle}/{self.a.cycles}")
            for model, category in models:
                self.switch_request(model, category)
                if category == "generative":
                    self.run_generative(model, cycle)
        self.summarise(models)

    def summarise(self, models):
        rows = [json.loads(l) for l in open(os.path.join(self.out, "events.jsonl"))]
        summary = {"url": self.a.url, "cycles": self.a.cycles, "prompt_tokens": self.a.prompt_tokens, "models": {}}
        for model, category in models:
            sw = [r for r in rows if r["kind"] == "switch" and r["model"] == model and r["swapped"] and r["status"] == 200]
            turns = [r for r in rows if r["kind"] == "turn" and r["model"] == model and r["status"] == 200]
            by = lambda ph, src=None: [r for r in turns if r["phase"] == ph and (src is None or r["source"] == src)]  # noqa: E731
            m = {"category": category, "switch_n": len(sw),
                 "switch_s_med": median([r.get("swap_s") for r in sw]),
                 "switch_s_max": max([r.get("swap_s") for r in sw if r.get("swap_s")], default=None),
                 "switch_wall_s_med": median([r["wall_s"] for r in sw]),
                 "switch_wall_s_max": max([r["wall_s"] for r in sw], default=None)}
            if category == "generative":
                cold, warm = by("cold"), by("warm")
                nv, ram = by("revisit", "nvme"), by("revisit", "ram")
                m.update({
                    "cold_n": len(cold), "cold_ttft_s_med": median([r["ttft_s"] for r in cold]),
                    "cold_prompt_tokens_med": median([r["prompt_tokens"] for r in cold]),
                    "warm_n": len(warm), "warm_ttft_s_med": median([r["ttft_s"] for r in warm]),
                    "warm_cached_tokens_med": median([r["cached_tokens"] for r in warm]),
                    "nvme_n": len(nv), "nvme_ttft_s_med": median([r["ttft_s"] for r in nv]),
                    "nvme_ttft_s_max": max([r["ttft_s"] for r in nv if r["ttft_s"]], default=None),
                    "nvme_cached_tokens_med": median([r["cached_tokens"] for r in nv]),
                    "nvme_prompt_tokens_med": median([r["prompt_tokens"] for r in nv]),
                    "ram_revisit_n": len(ram), "ram_revisit_ttft_s_med": median([r["ttft_s"] for r in ram]),
                    "revisit_miss_n": len(by("revisit", "none")) + len(by("revisit", "cold")),
                    "errors": sum(1 for r in rows if r["kind"] == "turn" and r["model"] == model and r["status"] != 200)})
            summary["models"][model] = m
        with open(os.path.join(self.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        md = ["| Model | Category | Swaps | Switch wait p50 / max | Request wall p50 |",
              "|---|---|---:|---:|---:|"]
        for model, m in summary["models"].items():
            med = f"{m['switch_s_med']:.1f} s" if m["switch_s_med"] else "–"
            mx = f"{m['switch_s_max']:.1f} s" if m["switch_s_max"] else "–"
            md.append(f"| `{model}` | {m['category']} | {m['switch_n']} | {med} / {mx} | "
                      f"{m['switch_wall_s_med'] or '–'} s |")
        md += ["", "| Model | Cold TTFT p50 (prompt tok) | Warm TTFT p50 (RAM) | NVMe restore TTFT p50 / max (restored tok) | Restores | Misses |",
               "|---|---:|---:|---:|---:|---:|"]
        for model, m in summary["models"].items():
            if m["category"] != "generative":
                continue
            md.append(f"| `{model}` | {m['cold_ttft_s_med']} s ({m['cold_prompt_tokens_med']}) | "
                      f"{m['warm_ttft_s_med']} s | {m['nvme_ttft_s_med']} s / {m['nvme_ttft_s_max']} s "
                      f"({m['nvme_cached_tokens_med']}) | {m['nvme_n']} | {m['revisit_miss_n']} |")
        with open(os.path.join(self.out, "summary.md"), "w") as f:
            f.write("\n".join(md) + "\n")
        print("\n".join(md))
        self.say(f"done; summary in {self.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:11434")
    p.add_argument("--token", default=os.environ.get("ATLAS_TOKEN"))
    p.add_argument("--models", default=None, help="comma-separated allowlist of model ids")
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--new-sessions", type=int, default=2, help="fresh sessions per generative model per cycle")
    p.add_argument("--prompt-tokens", type=int, default=3000)
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--chat-timeout", type=float, default=900)
    p.add_argument("--swap-timeout", type=float, default=900)
    p.add_argument("--out", default=None)
    p.add_argument("--nonce", default=None, help="prompt seed; default = start timestamp (fresh prompts per run)")
    Bench(p.parse_args()).run()


if __name__ == "__main__":
    main()

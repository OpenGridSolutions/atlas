#!/usr/bin/env python3
"""
nvme_session_sweep.py — on-box sweep of the NVMe session tier across every
generative model: sessions are created, revisited (RAM prefix hit), pushed
off the GPU by a model switch (persist-on-release), and revisited again
(restore from NVMe). Every step records the memory picture so a failure can
be traced back to the model, the session and the pool state that produced it.

Per step the run directory receives:
  run.log             human-readable timeline (what, which model, which session)
  events.jsonl        one JSON object per request/probe with the full memory
                      snapshot before and after (/internal/state kv_pool +
                      nvme_store, /v1/stats memory + per-model residency,
                      /metrics NVMe + swap + prefix-cache counters)
  server-excerpt.log  the server's own log lines (memdebug per-sequence block
                      accounting with physical block indices, kv-tier /
                      nvme-restore, hot-swap with the weights-arena base
                      DevicePtr, errors) tagged with the step that caused them
  anomalies.jsonl     every check that failed (also pushed to ntfy)
  summary.json        totals per model

Server prerequisites (the tier is OFF without them):
  --nvme-cache-dir <dir>  --enable-prefix-caching  and a KV pool small enough
  to overcommit (--kv-pool-gb 2); for the per-sequence block log the server
  needs ATLAS_MEM_DEBUG=1 RUST_LOG=info,memdebug=debug.

Usage:
  python3 tests/nvme_session_sweep.py --url http://127.0.0.1:11434 \
      --duration-min 30 --ntfy jonas-dgx-spark001 [--token TOKEN]
"""

import argparse
import glob
import json
import os
import random
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ANSI = re.compile(r"\x1b\[[0-9;]*m")
SERVER_LINE = re.compile(
    r"memdebug|nvme|kv-tier|block-store|[Hh]ot-swap|hotswap|arena|DevicePtr|Swap-out|Swap-in"
    r"|Session 0x|Session eviction|Marconi|SSM snapshot|SSM state pool|Decode-time guard"
    r"|prefix-cache evict|KV cache:|KV budget|Lazy swap|Done:|panicked|ERROR|WARN"
)
SERVER_BAD = re.compile(
    r"panicked|ERROR|kv-tier:.*failed|nvme-restore.*failed|write queue full|persist_timeout"
    r"|load failed|unload finished with errors|KV cache exhausted|memory LOW"
)
METRICS = (
    "atlas_nvme_write_bytes_total", "atlas_nvme_read_bytes_total",
    "atlas_nvme_restored_tokens_total", "atlas_prefix_cache_hit_tokens_total",
    "atlas_prefix_cache_hits_total", "atlas_prefix_cache_misses_total",
    "atlas_model_swaps_total", "atlas_requests_total",
)
WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
         "omicron pi rho sigma tau upsilon phi chi psi omega anchor bridge canyon "
         "dune ember forest glacier harbor island jungle kettle lantern meadow").split()


class Sweep:
    def __init__(self, a):
        self.a = a
        self.t0 = time.monotonic()
        self.out = a.out or os.path.join("logs", "nvme-sweep", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.out, exist_ok=True)
        self.run_log = open(os.path.join(self.out, "run.log"), "a", buffering=1)
        self.events = open(os.path.join(self.out, "events.jsonl"), "a", buffering=1)
        self.anoms = open(os.path.join(self.out, "anomalies.jsonl"), "a", buffering=1)
        self.excerpt = open(os.path.join(self.out, "server-excerpt.log"), "a", buffering=1)
        self.step = "preflight"
        self.log_path, self.log_off = None, 0
        self.sessions = {}  # model -> [session dict]
        self.counts = {}    # model -> counters
        self.n_anoms = 0
        self.ntfy_sent = {}
        self.nvme_active = False

    # ── plumbing ─────────────────────────────────────────────────────
    def say(self, msg):
        line = f"{time.strftime('%H:%M:%S')} +{time.monotonic() - self.t0:7.1f}s [{self.step}] {msg}"
        print(line, flush=True)
        self.run_log.write(line + "\n")

    def headers(self, json_body=True):
        h = {"Content-Type": "application/json"} if json_body else {}
        if self.a.token:
            h["Authorization"] = f"Bearer {self.a.token}"
        return h

    def http(self, path, body=None, timeout=30, raw=False):
        data = json.dumps(body).encode() if body is not None else None
        req = Request(f"{self.a.url}{path}", data=data, headers=self.headers())
        try:
            with urlopen(req, timeout=timeout) as r:
                payload = r.read()
                return r.status, (payload.decode(errors="replace") if raw else json.loads(payload))
        except HTTPError as e:
            return e.code, e.read().decode(errors="replace")[:2000]
        except (URLError, TimeoutError, OSError) as e:
            return 0, f"{type(e).__name__}: {e}"

    def ntfy(self, title, msg, priority=4, tags="warning", key=None):
        if not self.a.ntfy:
            return
        if key and self.ntfy_sent.get(key, 0) >= self.a.ntfy_per_key:
            return
        if key:
            self.ntfy_sent[key] = self.ntfy_sent.get(key, 0) + 1
        req = Request(f"https://ntfy.sh/{self.a.ntfy}", data=msg.encode(),
                      headers={"Title": title, "Priority": str(priority), "Tags": tags})
        try:
            urlopen(req, timeout=15).read()
        except Exception as e:  # noqa: BLE001 — notification must never kill the run
            self.say(f"ntfy failed: {e}")

    def anomaly(self, kind, detail, model=None, session=None, notify=True):
        self.n_anoms += 1
        rec = {"ts": time.time(), "step": self.step, "kind": kind, "model": model,
               "session": session, "detail": detail}
        self.anoms.write(json.dumps(rec) + "\n")
        self.say(f"ANOMALY {kind}: {detail}")
        if notify:
            self.ntfy(f"Atlas sweep: {kind}", f"{model or '-'} / {session or '-'} / step {self.step}\n{detail}"[:900],
                      key=f"{kind}:{model}")

    def event(self, kind, **fields):
        rec = {"ts": time.time(), "elapsed_s": round(time.monotonic() - self.t0, 2),
               "step": self.step, "kind": kind}
        rec.update(fields)
        self.events.write(json.dumps(rec) + "\n")

    # ── server log tail ──────────────────────────────────────────────
    def pick_server_log(self):
        cands = [p for p in glob.glob(os.path.join(self.a.log_dir, "atlas-*.log")) if os.path.getsize(p) > 0]
        if not cands:
            cands = glob.glob(os.path.join(self.a.log_dir, "serve-*.log"))
        if not cands:
            return
        newest = max(cands, key=os.path.getmtime)
        if newest != self.log_path:
            self.log_path = newest
            self.log_off = os.path.getsize(newest)
            self.say(f"server log: {newest} (tailing from byte {self.log_off})")

    def drain_server_log(self):
        """Copy new matching server lines into the excerpt; return bad lines."""
        self.pick_server_log()
        if not self.log_path:
            return []
        bad = []
        with open(self.log_path, "rb") as f:
            f.seek(self.log_off)
            chunk = f.read()
            self.log_off = f.tell()
        for raw in chunk.decode(errors="replace").splitlines():
            line = ANSI.sub("", raw)
            if SERVER_LINE.search(line):
                self.excerpt.write(f"[{self.step}] {line}\n")
                if SERVER_BAD.search(line):
                    bad.append(line)
        return bad

    # ── memory snapshot ──────────────────────────────────────────────
    def snapshot(self):
        _, st = self.http("/internal/state")
        _, stats = self.http("/v1/stats")
        _, metrics = self.http("/metrics", raw=True)
        snap = {}
        if isinstance(st, dict):
            snap["arena"] = {k: st.get("arena", {}).get(k) for k in ("state", "model", "resident_s")}
            snap["kv_pool"] = st.get("kv_pool")
            snap["nvme_store"] = st.get("nvme_store")
            snap["scheduler_load"] = (st.get("batch") or {}).get("scheduler_load")
            snap["swap_in_progress"] = (st.get("scheduler") or {}).get("swap_in_progress")
        else:
            snap["internal_state_error"] = st
        if isinstance(stats, dict):
            snap["memory"] = stats.get("memory")
            snap["resident_models"] = [
                {"id": m["id"], "weights_bytes": m["weights_bytes"], "kv_reserved_bytes": m["kv_reserved_bytes"],
                 "kv_used_blocks": m["kv_used_blocks"]}
                for m in stats.get("models", []) if m.get("resident")]
        if isinstance(metrics, str):
            snap["metrics"] = {}
            for line in metrics.splitlines():
                for name in METRICS:
                    if line.startswith(name + " ") or line.startswith(name + "{"):
                        snap["metrics"][line.rsplit(" ", 1)[0]] = float(line.rsplit(" ", 1)[1])
        return snap

    def metric(self, snap, name):
        return sum(v for k, v in (snap.get("metrics") or {}).items() if k.startswith(name))

    # ── requests ─────────────────────────────────────────────────────
    def chat(self, model, messages, max_tokens):
        body = {"model": model, "messages": messages, "max_tokens": max_tokens,
                "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False}}
        if "thinking" in model.lower():
            # A template that seeds <think> (Qwen3-Next-Thinking) ignores
            # enable_thinking=false and runs the model's 4096-token thinking
            # budget, which max_tokens does not cap (~60 s per turn). An
            # explicit small budget bounds it. Only for thinking checkpoints:
            # forcing it on an instruct model makes it answer with "\n".
            body["thinking"] = {"type": "enabled", "budget_tokens": 16}
        rec = {"status": None, "ttft_s": None, "content": "", "reasoning_chars": 0,
               "tokens": 0, "usage": None, "finish_reason": None, "error": None}
        t0 = time.monotonic()
        last = None
        gaps = []
        try:
            resp = urlopen(Request(f"{self.a.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                   headers=self.headers()), timeout=self.a.chat_timeout)
            rec["status"] = resp.status
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
                    d = ch.get("delta", {})
                    if ch.get("finish_reason"):
                        rec["finish_reason"] = ch["finish_reason"]
                    if d.get("content") or d.get("reasoning_content"):
                        now = time.monotonic()
                        if rec["ttft_s"] is None:
                            rec["ttft_s"] = round(now - t0, 3)
                        elif last is not None:
                            gaps.append(now - last)
                        last = now
                        rec["tokens"] += 1
                        rec["content"] += d.get("content") or ""
                        rec["reasoning_chars"] += len(d.get("reasoning_content") or "")
        except HTTPError as e:
            rec["status"], rec["error"] = e.code, e.read().decode(errors="replace")[:1000]
        except Exception as e:  # noqa: BLE001 — recorded as the request's outcome
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["elapsed_s"] = round(time.monotonic() - t0, 3)
        if gaps:
            rec["tbt_max_ms"] = round(max(gaps) * 1000, 1)
            rec["stalls_gt_250ms"] = sum(1 for g in gaps if g > 0.25)
        return rec

    def probe(self, model, messages):
        status, body = self.http("/v1/atlas/session/state", {"model": model, "messages": messages,
                                                             "max_tokens": self.a.max_tokens})
        return body if status == 200 and isinstance(body, dict) else {"error": f"{status} {body}"}

    def aux_request(self, kind, model):
        if kind == "embed":
            return self.http("/v1/embeddings", {"model": model, "input": ["Atlas NVMe tier sweep probe"]},
                             timeout=self.a.swap_timeout)
        if kind == "rerank":
            return self.http("/v1/rerank", {"model": model, "query": "which text is about memory?",
                                            "documents": ["KV cache blocks live in the pool", "a sunny beach"]},
                             timeout=self.a.swap_timeout)
        if kind == "translate":
            return self.http("/v1/translate", {"model": model, "source_lang_code": "de",
                                               "target_lang_code": "en", "text": "Der Speicher ist knapp."},
                             timeout=self.a.swap_timeout)
        return 0, f"unknown aux kind {kind}"

    # ── session helpers ──────────────────────────────────────────────
    def make_prompt(self, model, idx):
        rng = random.Random(f"{model}/{idx}")
        words = [rng.choice(WORDS) for _ in range(int(self.a.prompt_tokens * 0.75))]
        text = " ".join(words)
        return (f"Session {idx} of {model}. Memorise the following word stream, then answer in one short "
                f"sentence: what was the first word and how many words were there?\n\n{text}")

    def new_sessions(self, model):
        return [{"id": f"{model.split('/')[-1]}#{i}", "messages": [{"role": "user", "content": self.make_prompt(model, i)}],
                 "turns": 0} for i in range(1, self.a.sessions + 1)]

    def counter(self, model, key, inc=1):
        c = self.counts.setdefault(model, {})
        c[key] = c.get(key, 0) + inc

    # ── one chat turn with full bookkeeping ──────────────────────────
    def turn(self, model, s, phase, expect):
        """expect: None | 'ram_hit' | 'nvme_restore'"""
        if expect == "nvme_restore" and not self.nvme_active:
            expect = "ram_hit"  # tier off: the best we can check is the RAM prefix cache
        self.step = f"{model.split('/')[-1]}/{s['id']}/{phase}"
        before = self.snapshot()
        pre_probe = self.probe(model, s["messages"]) if s["turns"] else None
        rec = self.chat(model, s["messages"], self.a.max_tokens)
        after = self.snapshot()
        post_probe = self.probe(model, s["messages"] + [{"role": "assistant", "content": rec["content"]}])
        bad = self.drain_server_log()
        usage = rec.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        restored = self.metric(after, "atlas_nvme_restored_tokens_total") - self.metric(before, "atlas_nvme_restored_tokens_total")
        self.event("chat", model=model, session=s["id"], phase=phase, expect=expect, request=rec,
                   cached_tokens=cached, nvme_restored_tokens_delta=restored,
                   probe_before=pre_probe, probe_after=post_probe, mem_before=before, mem_after=after)
        self.say(f"{phase}: status={rec['status']} prompt={usage.get('prompt_tokens')} cached={cached} "
                 f"restored_delta={restored:.0f} ttft={rec['ttft_s']} out={rec['tokens']} "
                 f"pool_used={(after.get('kv_pool') or {}).get('used_blocks')}/{(after.get('kv_pool') or {}).get('total_blocks')} "
                 f"nvme={(after.get('nvme_store') or {}).get('entries')}e/{(after.get('nvme_store') or {}).get('total_bytes')}B "
                 f"nvme_probe={(post_probe.get('nvme') or {}) if isinstance(post_probe, dict) else post_probe}")
        self.counter(model, "requests")
        if rec["status"] != 200 or rec["error"]:
            self.counter(model, "errors")
            self.anomaly("http_error", f"{rec['status']} {rec['error']}", model, s["id"])
            return rec
        if not re.search(r"[A-Za-z]{3,}", rec["content"]):
            self.anomaly("empty_or_garbled_output", f"content={rec['content']!r}", model, s["id"])
        if expect == "ram_hit" and cached == 0:
            self.anomaly("no_prefix_hit", f"turn after warm turn had cached_tokens=0 (probe before: {pre_probe})", model, s["id"])
        if expect == "nvme_restore":
            persisted = ((pre_probe or {}).get("nvme") or {}).get("persisted_tokens")
            # The probe only sees sessions of the LOADED arena model; right
            # after the switch-away the aux model is resident, so the first
            # restore turn's probe answers active=false without an nvme view.
            probe_blind = (pre_probe or {}).get("active") is False
            if persisted in (None, 0) and not probe_blind:
                self.anomaly("not_persisted", f"session probe before restore turn: nvme={(pre_probe or {}).get('nvme')}", model, s["id"])
            if cached == 0 and restored == 0:
                self.anomaly("restore_miss", f"cached_tokens=0 and no atlas_nvme_restored_tokens delta (probe before: {pre_probe})", model, s["id"])
            elif restored > 0:
                self.counter(model, "nvme_restores")
        if cached > 0:
            self.counter(model, "cached_turns")
        for line in bad:
            self.anomaly("server_log", line[:600], model, s["id"])
        pool = after.get("kv_pool") or {}
        if pool and pool.get("used_blocks", 0) > pool.get("total_blocks", 0):
            self.anomaly("pool_invariant", f"used > total: {pool}", model, s["id"])
        # extend the conversation for the next turn
        s["messages"].append({"role": "assistant", "content": rec["content"] or "(no output)"})
        s["messages"].append({"role": "user", "content": f"Turn {s['turns'] + 2}: repeat your previous answer verbatim."})
        s["turns"] += 1
        time.sleep(self.a.pause)
        return rec

    def switch_away(self, model, cycle_idx):
        kinds = self.a.aux
        if not kinds:
            return
        kind = kinds[cycle_idx % len(kinds)]
        aux_model = self.aux_models.get(kind)
        if not aux_model:
            self.say(f"no {kind} model registered — skipping switch-away")
            return
        self.step = f"{model.split('/')[-1]}/switch->{aux_model.split('/')[-1]}"
        before = self.snapshot()
        t0 = time.monotonic()
        status, body = self.aux_request(kind, aux_model)
        el = round(time.monotonic() - t0, 2)
        after = self.snapshot()
        bad = self.drain_server_log()
        # `kind` is event()'s positional name; the aux category goes under its own key.
        self.event("aux", model=aux_model, aux_kind=kind, status=status, elapsed_s=el,
                   body_head=str(body)[:300], mem_before=before, mem_after=after)
        self.say(f"switch-away via {kind} {aux_model}: status={status} in {el}s; resident now "
                 f"{(after.get('arena') or {}).get('model')} nvme={(after.get('nvme_store') or {})}")
        if status != 200:
            self.anomaly("aux_error", f"{kind} {aux_model}: {status} {str(body)[:500]}", aux_model)
        for line in bad:
            self.anomaly("server_log", line[:600], aux_model)
        time.sleep(self.a.pause)

    # ── per-model block ──────────────────────────────────────────────
    def run_model(self, model, cycle):
        self.step = f"{model.split('/')[-1]}/enter"
        self.say(f"=== model {model} (cycle {cycle}) ===")
        fresh = model not in self.sessions
        if fresh:
            self.sessions[model] = self.new_sessions(model)
        sessions = self.sessions[model]
        # turn A: fresh prompts (cycle 1) or cross-switch restore of old sessions
        for s in sessions:
            self.turn(model, s, "A-first" if fresh else "A-cross-switch-restore",
                      None if fresh else "nvme_restore")
        # turn B: immediate revisit → RAM prefix hit expected
        for s in sessions:
            self.turn(model, s, "B-warm-revisit", "ram_hit")
        # push the model out (persist on release + unload) and come back
        self.switch_away(model, cycle)
        for s in sessions:
            self.turn(model, s, "C-after-switch", "nvme_restore")
        self.step = f"{model.split('/')[-1]}/leave"
        snap = self.snapshot()
        self.event("model_done", model=model, cycle=cycle, counts=self.counts.get(model), mem=snap)
        self.say(f"model done: {self.counts.get(model)}; sleeping {self.a.model_pause}s")
        time.sleep(self.a.model_pause)

    # ── main ─────────────────────────────────────────────────────────
    def preflight(self):
        self.step = "preflight"
        checks = {}
        st, health = self.http("/health")
        checks["/health"] = st
        st, models = self.http("/v1/models")
        checks["/v1/models"] = st
        st, istate = self.http("/internal/state")
        checks["/internal/state"] = st
        st, stats = self.http("/v1/stats")
        checks["/v1/stats"] = st
        st, _ = self.http("/metrics", raw=True)
        checks["/metrics"] = st
        # After preseed the arena idles unloaded until the first request, so
        # /health reports model=null; the probe then needs any registered
        # generative id (it answers active=false, 200) instead of null (400).
        active = health.get("model") if isinstance(health, dict) else None
        if not active and isinstance(stats, dict):
            active = next((m["id"] for m in stats.get("models", []) if m.get("category") == "generative"), None)
        st, probe = self.http("/v1/atlas/session/state", {"model": active, "messages": [{"role": "user", "content": "ping"}]})
        checks["/v1/atlas/session/state"] = st
        self.say(f"endpoints: {checks}")
        self.event("preflight", checks=checks, health=health, probe=probe,
                   nvme_store=istate.get("nvme_store") if isinstance(istate, dict) else None,
                   server_args=self.server_args())
        if any(v != 200 for v in checks.values()):
            raise SystemExit(f"preflight failed: {checks}")
        self.gen_models = [m["id"] for m in stats["models"] if m["category"] == "generative"]
        if self.a.models:
            self.gen_models = [m for m in self.a.models if m in self.gen_models] or self.a.models
        self.aux_models = {}
        for m in stats["models"]:
            self.aux_models.setdefault(m["category"], m["id"])
        self.say(f"generative models: {self.gen_models}; aux: {self.aux_models}")
        nvme = istate.get("nvme_store") if isinstance(istate, dict) else None
        self.nvme_active = nvme is not None
        if nvme is None:
            msg = "server runs WITHOUT --nvme-cache-dir (nvme_store=null): persist/restore cannot be tested"
            if self.a.allow_no_nvme:
                self.say("WARNING: " + msg)
            else:
                raise SystemExit(msg)
        self.pick_server_log()
        self.ntfy("Atlas sweep started", f"{len(self.gen_models)} generative models, {self.a.sessions} sessions each, "
                  f"{self.a.duration_min} min, nvme_store={nvme}", priority=2, tags="rocket")

    def server_args(self):
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    argv = f.read().split(b"\0")
            except OSError:
                continue
            if len(argv) > 1 and argv[1] == b"serve" and b"spark" in argv[0]:
                return [x.decode() for x in argv if x]
        return None

    def run(self):
        self.preflight()
        deadline = self.t0 + self.a.duration_min * 60
        cycle = 0
        try:
            while True:
                cycle += 1
                for model in self.gen_models:
                    if time.monotonic() > deadline - self.a.reserve_min * 60:
                        raise StopIteration
                    self.run_model(model, cycle)
                    st, _ = self.http("/health", timeout=10)
                    if st != 200:
                        self.anomaly("server_down", f"/health → {st} after {model}", model)
                        self.ntfy("Atlas sweep: server down", f"/health → {st}", priority=5, tags="rotating_light")
                        raise StopIteration
        except StopIteration:
            pass
        except KeyboardInterrupt:
            self.say("interrupted")
        except Exception as e:  # noqa: BLE001 — report, then summarise what we have
            self.anomaly("sweep_crash", f"{type(e).__name__}: {e}")
            raise
        finally:
            self.finish(cycle)

    def finish(self, cycles):
        self.step = "summary"
        snap = self.snapshot()
        summary = {"elapsed_min": round((time.monotonic() - self.t0) / 60, 1), "cycles": cycles,
                   "anomalies": self.n_anoms, "per_model": self.counts, "final_mem": snap, "out": self.out}
        with open(os.path.join(self.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        self.say(f"SUMMARY {json.dumps({k: v for k, v in summary.items() if k != 'final_mem'})}")
        self.ntfy("Atlas sweep finished", f"{summary['elapsed_min']} min, {cycles} cycle(s), {self.n_anoms} anomalies\n"
                  f"{json.dumps(self.counts)}"[:900], priority=4 if self.n_anoms else 2,
                  tags="warning" if self.n_anoms else "white_check_mark")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=os.environ.get("ATLAS_BASE_URL", "http://127.0.0.1:11434"))
    p.add_argument("--token", default=os.environ.get("ATLAS_TOKEN"))
    p.add_argument("--models", nargs="*", help="generative model ids (default: all from /v1/stats)")
    p.add_argument("--aux", nargs="*", default=["embed", "rerank", "translate"],
                   help="categories used to push the chat model out (rotated per model)")
    p.add_argument("--sessions", type=int, default=4)
    p.add_argument("--prompt-tokens", type=int, default=3000, help="approx. prompt length per session")
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--duration-min", type=float, default=30)
    p.add_argument("--reserve-min", type=float, default=4, help="don't start a model block with less time left")
    p.add_argument("--pause", type=float, default=2.5, help="seconds between requests")
    p.add_argument("--model-pause", type=float, default=15)
    p.add_argument("--chat-timeout", type=float, default=600)
    p.add_argument("--swap-timeout", type=float, default=600)
    p.add_argument("--ntfy", default=None, help="ntfy.sh topic for anomalies and start/end")
    p.add_argument("--ntfy-per-key", type=int, default=2, help="max notifications per (kind, model)")
    p.add_argument("--log-dir", default="logs", help="where the server writes atlas-*.log")
    p.add_argument("--out", default=None)
    p.add_argument("--allow-no-nvme", action="store_true")
    Sweep(p.parse_args()).run()


if __name__ == "__main__":
    main()

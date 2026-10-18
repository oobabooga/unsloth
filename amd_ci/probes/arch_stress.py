#!/usr/bin/env python3
"""KV preemption stress scenarios against one llama-server binary and one GGUF.

    stress.py --bin B --model M --scenario S1 --arm swap --ctx 512 --np 4 [--ngl 0] [--out runs.jsonl]

Arms: base (merge-base binary, no flag), off (PR binary, no flag), swap (ample --preempt-ram),
recomp (--preempt-ram 1, forces recompute parks). --policy fcfs|cost.
Scenarios (N = ctx, M = np):
  S0 idle          60 s with no requests: CPU seconds, parks, park bytes
  S1 equal         M requests, prompt 0.4N + gen 0.3N, simultaneous
  S2 mixed         1 long (0.7N + 0.2N) and M-1 short (0.05N + 32), long first then short first
  S3 fit           M requests that fit together; returns token ids for identity checks
  S4 flood         open-loop Poisson at --rate-mult x M/T1 for --duration s, then drain
  S5 full          one 0.9N request while M-1 short ones keep arriving
  S6 prefix        M requests sharing a 90% prefix, cache_prompt on
  S7 cancel        M streams over capacity, half cancelled mid-run, then a health request
  S8 overlong      one prompt longer than N: must be a clean 4xx
  S9 multiturn     M users x 3 growing turns, cache_prompt on
  S10 exact        each request alone vs the same request with forced parks every 8 tokens
Prompts are token ids drawn below the model's vocab, greedy, ignore_eos, so every arch (including
random-weight fixtures) runs the same shapes. One JSON line per run goes to --out.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request


def http(method, url, data=None, timeout=1800):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # noqa: BLE001
        return -1, repr(e).encode()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Server:
    def __init__(self, a, extra_env=None):
        self.a = a
        self.port = a.port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        os.makedirs(a.logdir, exist_ok=True)
        self.log = os.path.join(a.logdir, f"{a.tag}_{a.scenario}_{a.arm}_{int(time.time() * 1000)}.log")
        args = [a.bin, "-m", a.model, "--host", "127.0.0.1", "--port", str(self.port), "-c", str(a.ctx),
                "-np", str(a.np), "--kv-unified", "--metrics", "--slots", "-ngl", str(a.ngl),
                "-b", str(a.batch), "-ub", str(a.ubatch), "--no-warmup", "-t", str(a.threads), "-tb", str(a.threads)]
        if a.fa:
            args += ["-fa", a.fa]
        if a.cache_type:
            args += ["-ctk", a.cache_type, "-ctv", a.cache_type]
        if a.arm == "swap":
            args += ["--preempt-ram", str(a.preempt_ram), "--preempt-policy", a.policy]
        elif a.arm == "recomp":
            args += ["--preempt-ram", "1", "--preempt-policy", a.policy]
        args += a.extra
        self.args = args
        env = dict(os.environ, **(extra_env or {}))
        env["LD_LIBRARY_PATH"] = os.path.dirname(a.bin) + ":" + env.get("LD_LIBRARY_PATH", "")
        for k in ("LLAMA_ARG_PREEMPT_RAM", "LLAMA_ARG_PREEMPT_POLICY"):
            env.pop(k, None)
        self.env = env

    def __enter__(self):
        self.lf = open(self.log, "w")
        self.p = subprocess.Popen(self.args, stdout=self.lf, stderr=subprocess.STDOUT, env=self.env,
                                  start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < self.a.start_timeout:
            if self.p.poll() is not None:
                raise RuntimeError(f"server died rc={self.p.returncode}: {self.text()[-2500:]}")
            st, _ = http("GET", self.url + "/health", timeout=2)
            if st == 200:
                return self
            time.sleep(0.3)
        raise RuntimeError("server did not become healthy: " + self.text()[-1500:])

    def __exit__(self, *exc):
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGINT)
            try:
                self.p.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid, signal.SIGKILL)
                self.p.wait()
        self.lf.close()

    def text(self):
        try:
            return open(self.log, errors="replace").read()
        except OSError:
            return ""

    def alive(self):
        return self.p.poll() is None

    def json(self, path):
        st, raw = http("GET", self.url + path, timeout=30)
        try:
            return json.loads(raw) if st == 200 else {}
        except Exception:  # noqa: BLE001
            return {}

    def metrics(self):
        st, raw = http("GET", self.url + "/metrics", timeout=30)
        out = {}
        if st == 200:
            for line in raw.decode(errors="replace").splitlines():
                m = re.match(r"^llamacpp:(\w+)\s+([0-9.eE+-]+)$", line)
                if m:
                    out[m.group(1)] = float(m.group(2))
        return out

    def cpu_seconds(self):
        with open(f"/proc/{self.p.pid}/stat") as f:
            parts = f.read().rsplit(")", 1)[1].split()
        return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")

    def n_vocab(self):
        meta = (self.json("/v1/models").get("data") or [{}])[0].get("meta") or {}
        return int(meta.get("n_vocab") or 0)


def toks(n, seed, n_vocab):
    rng = random.Random(seed)
    hi = max(4, n_vocab - 1)
    return [rng.randint(3, hi) for _ in range(max(1, n))]


def stream(url, prompt, n_predict, extra=None, cancel_after=None):
    """Streams one completion. Returns per-chunk stamps, the final object, an error and the SSE comments."""
    data = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "ignore_eos": True,
            "cache_prompt": False, "seed": 42, "stream": True, "return_tokens": True}
    data.update(extra or {})
    req = urllib.request.Request(url + "/completion", data=json.dumps(data).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    stamps, toks_out, final, err, comments, status = [], [], None, None, [], None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            status = r.status
            for line in r:
                line = line.decode(errors="replace").strip()
                if line.startswith(":"):
                    comments.append(line)
                    continue
                if not line.startswith("data:"):
                    continue
                try:
                    j = json.loads(line[5:].strip())
                except Exception:  # noqa: BLE001
                    continue
                if "error" in j:
                    err = j["error"]
                    break
                got = j.get("tokens") or []
                if got or j.get("content"):
                    stamps.append(time.time())
                    toks_out.extend(got)
                if j.get("stop"):
                    final = j
                if cancel_after is not None and len(toks_out) >= cancel_after:
                    err = "cancelled"
                    break
    except urllib.error.HTTPError as e:
        status = e.code
        err = e.read()[:400].decode(errors="replace")
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    return {"t0": t0, "t1": time.time(), "stamps": stamps, "tokens": toks_out, "final": final, "err": err,
            "status": status, "comments": comments, "n_predict": n_predict}


def concurrent(jobs):
    """jobs: list of (delay_s, fn, args). Starts each after its delay, returns results in order."""
    res = [None] * len(jobs)
    t0 = time.time()

    def run(i, delay, fn, args):
        wait = t0 + delay - time.time()
        if wait > 0:
            time.sleep(wait)
        res[i] = fn(*args)

    th = [threading.Thread(target=run, args=(i, *j)) for i, j in enumerate(jobs)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return res, time.time() - t0


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, max(0, int(math.ceil(p / 100 * len(xs))) - 1))]


def is_ctx_error(r):
    e = json.dumps(r.get("err") or "").lower()
    return "exceed" in e or "context size" in e or "kv cache" in e


def summarize(results, srv, wall):
    done = [r for r in results if r and r["final"] is not None and not r["err"]]
    gaps, ttft, per_req = [], [], []
    for r in done:
        if r["stamps"]:
            ttft.append(r["stamps"][0] - r["t0"])
            gaps += [b - a for a, b in zip(r["stamps"], r["stamps"][1:])]
            n = len(r["tokens"]) or (r["final"].get("timings") or {}).get("predicted_n") or 0
            if r["t1"] > r["stamps"][0] and n > 1:
                per_req.append(n / (r["t1"] - r["t0"]))
    parks = sum(((r["final"] or {}).get("preempt") or {}).get("parks", 0) for r in done)
    recomputes = sum(((r["final"] or {}).get("preempt") or {}).get("recomputes", 0) for r in done)
    short_out = [r for r in done if (r["final"].get("timings") or {}).get("predicted_n", r["n_predict"]) < r["n_predict"]]
    out_tokens = sum((r["final"].get("timings") or {}).get("predicted_n", 0) for r in done)
    text = srv.text()
    m = srv.metrics() if srv.alive() else {}
    return {
        "submitted": len(results),
        "completed": len(done),
        "cancelled": sum(1 for r in results if r and r["err"] == "cancelled"),
        "ctx_errors": sum(1 for r in results if r and is_ctx_error(r)),
        "other_errors": sum(1 for r in results if r and r["err"] and r["err"] != "cancelled" and not is_ctx_error(r)),
        "short_outputs": len(short_out),
        "log_ctx_exceeded": text.count("Context size has been exceeded"),
        "log_failed_free_space": text.count("failed to find free space"),
        "parks": parks,
        "recomputes": recomputes,
        "metric_preempt": m.get("n_preempt_total"),
        "metric_recompute": m.get("preempt_recompute_total"),
        "park_ram_bytes_after": m.get("preempt_ram_bytes"),
        "rotations": text.count("rotat"),
        "asserts": text.count("GGML_ASSERT") + text.count("Segmentation") + text.count("terminate called"),
        "wall_s": round(wall, 3),
        "out_tokens": out_tokens,
        "agg_tok_s": round(out_tokens / wall, 2) if wall > 0 else None,
        "ttft_p50": pct(ttft, 50), "ttft_p99": pct(ttft, 99),
        "gap_p50": pct(gaps, 50), "gap_p99": pct(gaps, 99), "gap_max": max(gaps) if gaps else None,
        "per_req_tok_s_p50": pct(per_req, 50),
        "server_alive": srv.alive(),
    }


def emit(a, rec):
    rec = {"tag": a.tag, "model": os.path.basename(a.model), "scenario": a.scenario, "arm": a.arm,
           "policy": a.policy if a.arm in ("swap", "recomp") else None, "ctx": a.ctx, "np": a.np, "ngl": a.ngl,
           "bin": a.bin, "ts": time.time(), **rec}
    line = json.dumps(rec)
    print("RESULT " + line, flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "a") as f:
            f.write(line + "\n")


def scenario(a, srv):
    N, M, V = a.ctx, a.np, srv.n_vocab() or 32000
    u = srv.url
    seed0 = a.seed

    def P(n, s):
        return toks(n, seed0 * 1000 + s, V)

    if a.scenario == "S0":
        m0 = srv.metrics()
        c0 = srv.cpu_seconds()
        time.sleep(a.idle_s)
        c1 = srv.cpu_seconds()
        m1 = srv.metrics()
        return {"idle_s": a.idle_s, "cpu_s": round(c1 - c0, 3), "cpu_frac": round((c1 - c0) / a.idle_s, 4),
                "decode_delta": (m1.get("tokens_predicted_total") or 0) - (m0.get("tokens_predicted_total") or 0),
                "metric_preempt": m1.get("n_preempt_total"), "park_ram_bytes": m1.get("preempt_ram_bytes"),
                "props_preempt": srv.json("/props").get("preempt"), "server_alive": srv.alive()}

    if a.scenario == "S1":
        lp, lg = int(0.4 * N), int(0.3 * N)
        res, wall = concurrent([(0.0, stream, (u, P(lp, i), lg)) for i in range(M)])
        return summarize(res, srv, wall)

    if a.scenario == "S2":
        out = {}
        for order in ("long_first", "short_first"):
            long_job = (0.0 if order == "long_first" else 0.5, stream, (u, P(int(0.7 * N), 100), int(0.2 * N)))
            shorts = [(0.5 if order == "long_first" else 0.0, stream, (u, P(int(0.05 * N), 200 + i), 32)) for i in range(M - 1)]
            res, wall = concurrent([long_job] + shorts)
            s = summarize(res, srv, wall)
            s["short_ttft_p50"] = pct([r["stamps"][0] - r["t0"] for r in res[1:] if r and r["stamps"]], 50)
            s["long_wall"] = round(res[0]["t1"] - res[0]["t0"], 3) if res[0] else None
            out[order] = s
        flat = {"completed": sum(v["completed"] for v in out.values()), "submitted": sum(v["submitted"] for v in out.values()),
                "ctx_errors": sum(v["ctx_errors"] for v in out.values()), "other_errors": sum(v["other_errors"] for v in out.values()),
                "parks": sum(v["parks"] for v in out.values()), "asserts": max(v["asserts"] for v in out.values()),
                "server_alive": srv.alive()}
        return {**flat, "orders": out}

    if a.scenario == "S3":
        lp, lg = max(8, int(0.6 * N / M) - 40), 32
        res, wall = concurrent([(0.0, stream, (u, P(lp, i), lg)) for i in range(M)])
        s = summarize(res, srv, wall)
        s["tokens"] = [r["tokens"] for r in res]
        return s

    if a.scenario == "S4":
        # T1: one mid-size request alone, so the arrival rate is a multiple of what M slots could sustain
        probe = stream(u, P(int(0.2 * N), 1), int(0.1 * N))
        t1 = max(0.05, probe["t1"] - probe["t0"])
        rate = a.rate_mult * M / t1
        rng = random.Random(seed0)
        jobs, t = [], 0.0
        i = 0
        while t < a.duration:
            t += rng.expovariate(rate)
            lp = min(int(0.8 * N), max(8, int(rng.lognormvariate(math.log(0.2 * N), 0.6))))
            lg = min(int(0.95 * N) - lp, max(8, int(rng.lognormvariate(math.log(0.1 * N), 0.6))))
            jobs.append((t, stream, (u, P(lp, 1000 + i), max(4, lg))))
            i += 1
        res, wall = concurrent(jobs)
        s = summarize(res, srv, wall)
        s.update({"rate_req_s": round(rate, 4), "t1_s": round(t1, 3), "rate_mult": a.rate_mult,
                  "arrival_window_s": a.duration, "drain_s": round(wall - a.duration, 3)})
        return s

    if a.scenario == "S5":
        big = (0.0, stream, (u, P(int(0.9 * N) - 40, 7), 32))
        shorts = [(0.2 + 0.3 * k, stream, (u, P(int(0.08 * N), 300 + k), 24)) for k in range(3 * (M - 1))]
        res, wall = concurrent([big] + shorts)
        s = summarize(res, srv, wall)
        s["full_window_ok"] = bool(res[0] and res[0]["final"] and not res[0]["err"])
        return s

    if a.scenario == "S6":
        prefix = P(int(0.36 * N), 9)
        jobs = [(0.05 * i, stream, (u, prefix + P(int(0.04 * N), 400 + i), int(0.25 * N), {"cache_prompt": True})) for i in range(M)]
        res, wall = concurrent(jobs)
        return summarize(res, srv, wall)

    if a.scenario == "S7":
        lp, lg = int(0.4 * N), int(0.3 * N)
        jobs = []
        for i in range(M + 2):
            cancel = (int(0.05 * N) * (1 + i % 3)) if i % 2 else None
            jobs.append((0.1 * i, stream, (u, P(lp, 500 + i), lg, None, cancel)))
        res, wall = concurrent(jobs)
        time.sleep(2.0)
        s = summarize(res, srv, wall)
        after = stream(u, P(16, 999), 8)
        time.sleep(1.0)
        m = srv.metrics() if srv.alive() else {}
        s.update({"after_ok": bool(after["final"] and not after["err"]), "park_ram_bytes_settled": m.get("preempt_ram_bytes"),
                  "slots_processing_settled": m.get("requests_processing")})
        return s

    if a.scenario == "S8":
        r = stream(u, P(N + 64, 11), 8)
        after = stream(u, P(16, 12), 8)
        return {"status": r["status"], "err": str(r["err"])[:300], "clean_4xx": bool(r["status"] and 400 <= r["status"] < 500) or ("exceed" in str(r["err"]).lower() and r["final"] is None),
                "after_ok": bool(after["final"] and not after["err"]), "server_alive": srv.alive(),
                "asserts": srv.text().count("GGML_ASSERT")}

    if a.scenario == "S9":
        hist = [P(int(0.08 * N), 600 + i) for i in range(M)]
        allres, t0 = [], time.time()
        for turn in range(3):
            res, _ = concurrent([(0.0, stream, (u, hist[i], int(0.08 * N), {"cache_prompt": True})) for i in range(M)])
            for i, r in enumerate(res):
                hist[i] = hist[i] + (r["tokens"] if r else []) + P(int(0.04 * N), 700 + 10 * turn + i)
            allres += res
        s = summarize(allres, srv, time.time() - t0)
        s["final_history_tokens"] = [len(h) for h in hist]
        return s

    if a.scenario == "S10":
        lp, lg = int(0.3 * N), int(0.3 * N)
        refs = [stream(u, P(lp, 800 + i), lg) for i in range(min(M, 3))]
        return {"refs": [r["tokens"] for r in refs], "ref_err": [r["err"] for r in refs]}

    raise SystemExit(f"unknown scenario {a.scenario}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--arm", required=True, choices=["base", "off", "swap", "recomp"])
    ap.add_argument("--policy", default="fcfs", choices=["fcfs", "cost"])
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--np", type=int, default=4)
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ubatch", type=int, default=64)
    ap.add_argument("--fa", default="auto")
    ap.add_argument("--cache-type", default="")
    ap.add_argument("--preempt-ram", type=int, default=4096)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rate-mult", type=float, default=0.9)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--idle-s", type=float, default=30.0)
    ap.add_argument("--start-timeout", type=float, default=600.0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--logdir", default="stress_logs")
    ap.add_argument("--out", default="")
    ap.add_argument("--env", action="append", default=[], help="K=V for the server")
    ap.add_argument("extra", nargs="*", help="extra llama-server args after --")
    a = ap.parse_args()
    env = dict(e.split("=", 1) for e in a.env)
    if a.scenario == "S10":
        a.arm = a.arm if a.arm in ("swap", "recomp") else "swap"
        with Server(a, env) as srv:
            ref = scenario(a, srv)
        env2 = dict(env, LLAMA_SERVER_PREEMPT_EVERY="8")
        with Server(a, env2) as srv:
            got = scenario(a, srv)
            parks = srv.metrics().get("n_preempt_total")
        same = [x == y for x, y in zip(ref["refs"], got["refs"])]
        first_div = []
        for x, y in zip(ref["refs"], got["refs"]):
            d = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), None)
            first_div.append(d if d is not None else (None if len(x) == len(y) else min(len(x), len(y))))
        emit(a, {"identical": same, "first_divergence": first_div, "forced_parks": parks,
                 "ref_len": [len(x) for x in ref["refs"]], "got_len": [len(x) for x in got["refs"]],
                 "errors": ref["ref_err"] + got["ref_err"]})
        return
    try:
        with Server(a, env) as srv:
            props = srv.json("/props").get("preempt")
            rec = scenario(a, srv)
            rec["props_preempt"] = props
            rec["server_alive"] = srv.alive()
            rec["log"] = srv.log
    except RuntimeError as e:
        rec = {"start_error": str(e)[-1500:]}
    emit(a, rec)


if __name__ == "__main__":
    main()

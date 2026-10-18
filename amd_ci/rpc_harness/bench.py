#!/usr/bin/env python3
"""llama.cpp RPC / pipeline-groups / no-regression harness.

Runs on any box with built llama.cpp trees (CUDA, HIP or CPU). Stdlib only (pynvml optional).

  bench.py --bin base=DIR --bin h187=DIR ... --model qwen3=PATH ... --suite single,correct,stress,groups,load,teardown
           --out results.json [--gpus 0,1] [--scale quick|full]

Topologies (by available GPU count):
  single : coordinator on GPU g0 only.
  native2: one process, two local GPUs, -sm layer (llama.cpp's own multi-GPU split, the reference).
  rpc2   : ggml-rpc-server on GPU g1, coordinator on g0, --device RPC0,<local0> -sm layer.
  rpcN   : N-1 rpc-servers (one per extra GPU), coordinator on g0.
Every suite writes into one JSON document, flushed after each case.
"""

import argparse
import json
import math
import os
import random
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import http.client

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush = True)


# ---------------------------------------------------------------- GPU sampling

class Sampler:
    """Per-device utilisation % and used memory (MiB), sampled every `period` seconds."""

    def __init__(self, gpus, vendor, period = 0.1):
        self.gpus, self.vendor, self.period = list(gpus), vendor, period
        self.samples = []  # (t, [util], [mem])
        self._stop = threading.Event()
        self._thr = None
        self._nvml = None
        if vendor == "nvidia":
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml = pynvml
                self._handles = [pynvml.nvmlDeviceGetHandleByIndex(int(g)) for g in self.gpus]
            except Exception:
                self._nvml = None

    def _read(self):
        if self._nvml is not None:
            u, m = [], []
            for h in self._handles:
                u.append(float(self._nvml.nvmlDeviceGetUtilizationRates(h).gpu))
                m.append(self._nvml.nvmlDeviceGetMemoryInfo(h).used / 2**20)
            return u, m
        if self.vendor == "nvidia":
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output = True, text = True, timeout = 5).stdout
            rows = {r.split(",")[0].strip(): r.split(",") for r in out.strip().splitlines()}
            return ([float(rows[str(g)][1]) for g in self.gpus], [float(rows[str(g)][2]) for g in self.gpus])
        if self.vendor == "amd":
            out = subprocess.run(["rocm-smi", "--showuse", "--showmemuse", "--showmeminfo", "vram", "--json"],
                                 capture_output = True, text = True, timeout = 5).stdout
            d = json.loads(out or "{}")
            u, m = [], []
            for g in self.gpus:
                card = d.get(f"card{g}", {})
                u.append(float(next((v for k, v in card.items() if "GPU use" in k), 0) or 0))
                used = next((v for k, v in card.items() if "VRAM Total Used" in k), 0)
                m.append(float(used or 0) / 2**20)
            return u, m
        return [0.0] * len(self.gpus), [0.0] * len(self.gpus)

    def _loop(self):
        while not self._stop.is_set():
            t = time.time()
            try:
                u, m = self._read()
                self.samples.append((t, u, m))
            except Exception:
                pass
            self._stop.wait(max(0.0, self.period - (time.time() - t)))

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thr = threading.Thread(target = self._loop, daemon = True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout = 5)

    def window(self, t0, t1, busy = 30.0):
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if not s:
            return {"n": 0}
        n = len(self.gpus)
        mean_u = [round(statistics.mean(x[1][i] for x in s), 1) for i in range(n)]
        peak_m = [round(max(x[2][i] for x in s)) for i in range(n)]
        any_busy = sum(1 for x in s if max(x[1]) >= busy)
        all_busy = sum(1 for x in s if min(x[1]) >= busy)
        return {"n": len(s), "mean_util": mean_u, "peak_mem_mib": peak_m,
                "busy_frac": [round(sum(1 for x in s if x[1][i] >= busy) / len(s), 3) for i in range(n)],
                # Of the time any device was busy, how much had EVERY device busy: 1/n = a pure relay, 1 = full overlap.
                "overlap": round(all_busy / any_busy, 3) if any_busy else None}


# ---------------------------------------------------------------- processes

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Proc:
    def __init__(self, argv, env, logpath):
        self.argv, self.logpath = argv, logpath
        self.log = open(logpath, "w")
        self.p = subprocess.Popen(argv, env = env, stdout = self.log, stderr = subprocess.STDOUT,
                                  preexec_fn = os.setsid)

    def alive(self):
        return self.p.poll() is None

    def stop(self, sig = signal.SIGTERM, wait = 30):
        if self.p.poll() is None:
            try:
                os.killpg(os.getpgid(self.p.pid), sig)
            except ProcessLookupError:
                pass
            try:
                self.p.wait(timeout = wait)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.p.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.p.wait(timeout = 10)
        self.log.close()
        return self.p.returncode

    def tail(self, n = 40):
        try:
            with open(self.logpath, errors = "replace") as f:
                return "".join(f.readlines()[-n:])
        except OSError:
            return ""


def wait_port(port, timeout):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout = 1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def http_json(port, path, body = None, timeout = 600, method = None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout = timeout)
    data = json.dumps(body).encode() if body is not None else None
    c.request(method or ("POST" if data else "GET"), path, body = data,
              headers = {"Content-Type": "application/json"} if data else {})
    r = c.getresponse()
    raw = r.read()
    c.close()
    try:
        return r.status, json.loads(raw or b"{}")
    except ValueError:
        return r.status, {"raw": raw[:500].decode(errors = "replace")}


def wait_health(port, proc, timeout):
    end = time.time() + timeout
    while time.time() < end:
        if not proc.alive():
            return False
        try:
            st, _ = http_json(port, "/health", timeout = 5)
            if st == 200:
                return True
        except OSError:
            pass
        time.sleep(0.3)
    return False


# ---------------------------------------------------------------- environment

class Env:
    def __init__(self, gpus, vendor, threads):
        self.gpus, self.vendor, self.threads = list(gpus), vendor, threads

    def visible(self, gpus):
        env = dict(os.environ)
        val = ",".join(str(g) for g in gpus) if gpus else ""
        if self.vendor == "amd":
            env["HIP_VISIBLE_DEVICES"] = val
            env.pop("CUDA_VISIBLE_DEVICES", None)
        elif self.vendor == "nvidia":
            env["CUDA_VISIBLE_DEVICES"] = val if gpus else "-1"
        return env

    def local_name(self, bindir, gpus):
        """The first local accelerator name llama.cpp gives the visible device(s) (CUDA0, ROCm0, Vulkan0)."""
        out = subprocess.run([os.path.join(bindir, "llama-server"), "--list-devices"], env = self.visible(gpus),
                             capture_output = True, text = True, timeout = 120)
        names = re.findall(r"^\s+([A-Za-z]+\d+):", out.stdout + out.stderr, re.M)
        names = [n for n in names if not n.startswith("RPC")]
        return names


def server_argv(bindir, model, port, ctx, parallel, extra, threads):
    return [os.path.join(bindir, "llama-server"), "-m", model, "--host", "127.0.0.1", "--port", str(port),
            "-c", str(ctx), "-np", str(parallel), "-ngl", "999", "-t", str(threads), "--cache-ram", "0",
            "--no-warmup"] + list(extra)


class Topology:
    """Start / stop one serving topology; `port` is the llama-server port."""

    def __init__(self, env, bindir, model, kind, ctx, parallel, extra, workdir, n_rpc = 1, rpc_cache = False,
                 order = "rpc_first", label = ""):
        self.env, self.bindir, self.model, self.kind = env, bindir, model, kind
        self.ctx, self.parallel, self.extra, self.workdir = ctx, parallel, list(extra), workdir
        self.n_rpc, self.rpc_cache, self.order, self.label = n_rpc, rpc_cache, order, label
        self.rpcs, self.server, self.port = [], None, None
        self.load_s = None

    def devices_used(self):
        if self.kind == "single":
            return self.env.gpus[:1]
        if self.kind == "native2":
            return self.env.gpus[:2]
        return self.env.gpus[: 1 + self.n_rpc]

    def start(self, timeout = 1800):
        os.makedirs(self.workdir, exist_ok = True)
        g = self.env.gpus
        extra = list(self.extra)
        if self.kind == "single":
            env = self.env.visible(g[:1])
        elif self.kind == "native2":
            env = self.env.visible(g[:2])
            names = self.env.local_name(self.bindir, g[:2])
            extra += ["--device", ",".join(names[:2]), "-sm", "layer", "-ts", "1,1"]
        else:  # rpcN
            endpoints = []
            for i in range(self.n_rpc):
                port = free_port()
                argv = [os.path.join(self.bindir, "ggml-rpc-server"), "-H", "127.0.0.1", "-p", str(port)]
                if self.rpc_cache:
                    argv.append("-c")
                rpc = Proc(argv, self.env.visible([g[1 + i]]), os.path.join(self.workdir, f"rpc{i}.log"))
                if not wait_port(port, 120):
                    raise RuntimeError(f"rpc-server {i} did not listen:\n{rpc.tail()}")
                self.rpcs.append(rpc)
                endpoints.append(f"127.0.0.1:{port}")
            env = self.env.visible(g[:1])
            local = self.env.local_name(self.bindir, g[:1])[:1]
            remote = [f"RPC{i}" for i in range(self.n_rpc)]
            devs = remote + local if self.order == "rpc_first" else local + remote
            extra += ["--rpc", ",".join(endpoints), "--device", ",".join(devs), "-sm", "layer",
                      "-ts", ",".join(["1"] * len(devs))]
        self.port = free_port()
        argv = server_argv(self.bindir, self.model, self.port, self.ctx, self.parallel, extra, self.env.threads)
        t = time.time()
        self.server = Proc(argv, env, os.path.join(self.workdir, "server.log"))
        ok = wait_health(self.port, self.server, timeout)
        self.load_s = round(time.time() - t, 2)
        if not ok:
            tail = self.server.tail(60)
            self.stop()
            raise RuntimeError(f"llama-server not healthy ({self.kind} {self.label}):\n{tail}")
        return self

    def stop(self):
        rc = None
        if self.server:
            rc = self.server.stop()
        for r in self.rpcs:
            r.stop()
        self.rpcs = []
        return rc


# ---------------------------------------------------------------- client

WORDS = ("river mountain quiet lantern copper orbit garden signal velvet harbor winter meadow cipher "
         "falcon marble thunder ember willow prism canyon atlas beacon cobalt drift echo fable glacier "
         "horizon ivory jasper kernel lagoon mosaic nectar onyx pepper quartz raven saffron tundra").split()


def make_prompt(n_tokens, seed):
    rnd = random.Random(seed)
    # ~1.3 tokens per word for these words; a fixed instruction keeps greedy output comparable.
    n = max(4, int(n_tokens / 1.3))
    return "Continue this list of words with more words: " + " ".join(rnd.choice(WORDS) for _ in range(n))


def complete(port, prompt, n_predict, stream = True, timeout = 900, extra = None):
    """One /completion request. Returns dict(ok, ttft, e2e, n, content, err)."""
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "top_k": 1, "seed": 1,
            "ignore_eos": True, "cache_prompt": False, "stream": stream}
    if extra:
        body.update(extra)
    t0 = time.time()
    out = {"ok": False, "ttft": None, "e2e": None, "n": 0, "content": "", "err": None}
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout = timeout)
        c.request("POST", "/completion", body = json.dumps(body).encode(), headers = {"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            out["err"] = f"HTTP {r.status}: {r.read()[:300].decode(errors = 'replace')}"
            return out
        if not stream:
            d = json.loads(r.read())
            out.update(ok = True, e2e = time.time() - t0, content = d.get("content", ""),
                       n = int(d.get("tokens_predicted") or 0))
            return out
        parts = []
        buf = b""
        while True:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                ev, buf = buf.split(b"\n\n", 1)
                for line in ev.split(b"\n"):
                    if not line.startswith(b"data: "):
                        continue
                    d = json.loads(line[6:])
                    if d.get("content"):
                        if out["ttft"] is None:
                            out["ttft"] = time.time() - t0
                        parts.append(d["content"])
                    if d.get("stop"):
                        out["n"] = int(d.get("tokens_predicted") or len(parts))
        c.close()
        out.update(ok = True, e2e = time.time() - t0, content = "".join(parts))
        if not out["n"]:
            out["n"] = len(parts)
    except Exception as e:  # noqa: BLE001 - every failure is a recorded outcome
        out["err"] = f"{type(e).__name__}: {e}"
    return out


def chat(port, messages, n_predict, timeout = 900):
    t0 = time.time()
    try:
        st, d = http_json(port, "/v1/chat/completions",
                          {"messages": messages, "max_tokens": n_predict, "temperature": 0, "seed": 1,
                           "ignore_eos": True}, timeout = timeout)
        if st != 200:
            return {"ok": False, "err": f"HTTP {st}: {str(d)[:300]}", "e2e": time.time() - t0, "n": 0}
        msg = d["choices"][0]["message"]["content"] or ""
        n = int((d.get("usage") or {}).get("completion_tokens") or 0)
        return {"ok": True, "e2e": time.time() - t0, "n": n, "content": msg, "err": None}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "err": f"{type(e).__name__}: {e}", "e2e": time.time() - t0, "n": 0}


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)


def run_requests(port, specs, concurrency, arrivals = None):
    """specs: list of (prompt, n_predict). arrivals: optional start offsets (s). Returns per-request results."""
    results = [None] * len(specs)
    sem = threading.Semaphore(concurrency)
    t0 = time.time()

    def one(i):
        if arrivals is not None:
            delay = arrivals[i] - (time.time() - t0)
            if delay > 0:
                time.sleep(delay)
        with sem:
            p, n = specs[i]
            results[i] = complete(port, p, n)

    threads = [threading.Thread(target = one, args = (i,), daemon = True) for i in range(len(specs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, time.time() - t0


def summarize(results, wall):
    ok = [r for r in results if r and r["ok"]]
    toks = sum(r["n"] for r in ok)
    errs = [r["err"] for r in results if r and not r["ok"]]
    return {"requests": len(results), "ok": len(ok), "failed": len(errs), "errors": sorted(set(errs))[:5],
            "wall_s": round(wall, 2), "gen_tokens": toks, "agg_tok_s": round(toks / wall, 2) if wall else None,
            "ttft_p50": pct([r["ttft"] for r in ok], 0.5), "ttft_p95": pct([r["ttft"] for r in ok], 0.95),
            "e2e_p50": pct([r["e2e"] for r in ok], 0.5), "e2e_p95": pct([r["e2e"] for r in ok], 0.95),
            "per_req_tok_s_p50": pct([r["n"] / (r["e2e"] - (r["ttft"] or 0)) for r in ok
                                      if r["e2e"] and r["e2e"] > (r["ttft"] or 0)], 0.5)}


def proc_cpu_seconds(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            v = f.read().rsplit(")", 1)[1].split()
        return (int(v[11]) + int(v[12])) / os.sysconf("SC_CLK_TCK")
    except OSError:
        return None


# ---------------------------------------------------------------- stress patterns

def patterns(scale, parallel):
    q = scale == "quick"
    P = parallel
    return {
        "idle": {"idle_s": 10 if q else 30},
        "single": {"specs": [(make_prompt(256, 1), 128)], "conc": 1},
        "equal": {"specs": [(make_prompt(256, 100 + i), 128) for i in range(P)], "conc": P},
        "mixed": {"specs": [(make_prompt(32, 200 + i), 32) if i % 2 == 0 else (make_prompt(1500 if not q else 800, 300 + i), 192)
                            for i in range(2 * P)], "conc": P},
        "burst": {"idle_s": 5, "specs": [(make_prompt(128, 400 + i), 64) for i in range(3 * P)], "conc": 3 * P},
        "flood": {"flood": True, "rate_end": 2.0 * P / 8, "dur": 30 if q else 60, "prompt": 128, "n": 64},
        "long_chat": {"users": 4 if q else 8, "turns": 4 if q else 6, "user_tokens": 300, "n": 96},
    }


def run_pattern(top, sampler, name, spec, out):
    port = top.port
    res = {"pattern": name}
    t0 = time.time()
    if "idle_s" in spec and "specs" not in spec:
        c0 = proc_cpu_seconds(top.server.p.pid)
        time.sleep(spec["idle_s"])
        c1 = proc_cpu_seconds(top.server.p.pid)
        res["server_cpu_frac_idle"] = round((c1 - c0) / spec["idle_s"], 3) if c0 is not None and c1 is not None else None
        st, _ = http_json(port, "/health", timeout = 10)
        res["healthy_after_idle"] = st == 200
    elif spec.get("flood"):
        rnd = random.Random(7)
        arrivals, t = [], 0.0
        # Rate ramps linearly from 0.25/s to rate_end/s over dur seconds (Poisson within).
        while t < spec["dur"]:
            rate = 0.25 + (spec["rate_end"] - 0.25) * (t / spec["dur"])
            t += rnd.expovariate(max(rate, 0.05))
            arrivals.append(t)
        specs = [(make_prompt(spec["prompt"], 500 + i), spec["n"]) for i in range(len(arrivals))]
        results, wall = run_requests(port, specs, concurrency = 10**6, arrivals = arrivals)
        res.update(summarize(results, wall))
    elif "users" in spec:
        def user(u, acc):
            msgs = [{"role": "system", "content": "You are a helpful assistant."}]
            for turn in range(spec["turns"]):
                msgs.append({"role": "user", "content": make_prompt(spec["user_tokens"], 1000 * u + turn)})
                r = chat(port, msgs, spec["n"])
                acc.append(r)
                if not r["ok"]:
                    return
                msgs.append({"role": "assistant", "content": r["content"]})

        accs = [[] for _ in range(spec["users"])]
        threads = [threading.Thread(target = user, args = (u, accs[u]), daemon = True) for u in range(spec["users"])]
        ts = time.time()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        flat = [r for a in accs for r in a]
        for r in flat:
            r.setdefault("ttft", None)
        res.update(summarize(flat, time.time() - ts))
        res["turns_completed"] = [len([r for r in a if r["ok"]]) for a in accs]
    else:
        if spec.get("idle_s"):
            time.sleep(spec["idle_s"])
        results, wall = run_requests(port, spec["specs"], spec["conc"])
        res.update(summarize(results, wall))
    t1 = time.time()
    res["gpu"] = sampler.window(t0, t1) if sampler else None
    res["server_alive"] = top.server.alive()
    out.append(res)
    log(f"  {name}: " + json.dumps({k: res.get(k) for k in ("ok", "failed", "agg_tok_s", "ttft_p50", "e2e_p95")})
        + f" gpu={res['gpu'] and {k: res['gpu'].get(k) for k in ('mean_util', 'overlap')}}")
    return res


# ---------------------------------------------------------------- suites

class Results:
    def __init__(self, path, meta):
        self.path = path
        self.doc = {"meta": meta, "cases": []}
        self.flush()

    def add(self, case):
        case["t"] = round(time.time() - T0, 1)
        self.doc["cases"].append(case)
        self.flush()

    def flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.doc, f, indent = 1)
        os.replace(tmp, self.path)


def suite_single(a, env, bins, models, R, sampler):
    """llama-bench pp/tg per build, builds interleaved per round; plus server-path A/B on `single`."""
    names = list(bins)
    for mname in a.single_models:
        mpath = models[mname]
        for rnd in range(a.rounds):
            order = names[rnd % len(names):] + names[: rnd % len(names)]
            for b in order:
                cmd = [os.path.join(bins[b], "llama-bench"), "-m", mpath, "-ngl", "999", "-p", "512", "-n", "128",
                       "-r", "3", "-o", "json", "-t", str(env.threads)]
                t = time.time()
                p = subprocess.run(cmd, env = env.visible(env.gpus[:1]), capture_output = True, text = True, timeout = 1800)
                try:
                    rows = json.loads(p.stdout)
                    got = {("pp" if r.get("n_prompt") else "tg"): round(r.get("avg_ts", 0), 2) for r in rows}
                except ValueError:
                    got = {"error": (p.stderr or p.stdout)[-600:]}
                R.add({"suite": "single_bench", "model": mname, "build": b, "round": rnd, **got,
                       "gpu": sampler.window(t, time.time()) if sampler else None})
                log(f"bench {mname} {b} r{rnd}: {got}")
        # Server path (slot loop, sampling, HTTP): same workload per build, interleaved.
        for rnd in range(a.rounds):
            order = names[rnd % len(names):] + names[: rnd % len(names)]
            for b in order:
                top = Topology(env, bins[b], mpath, "single", a.ctx, a.parallel, [], os.path.join(a.work, f"srv_{mname}_{b}_{rnd}"))
                try:
                    top.start()
                except Exception as e:  # noqa: BLE001
                    R.add({"suite": "single_server", "model": mname, "build": b, "round": rnd, "error": str(e)[-1500:]})
                    continue
                pats = patterns(a.scale, a.parallel)
                rows = []
                for pn in ("single", "equal", "mixed"):
                    run_pattern(top, sampler, pn, pats[pn], rows)
                mem = sampler.window(time.time() - 1, time.time()) if sampler else None
                top.stop()
                R.add({"suite": "single_server", "model": mname, "build": b, "round": rnd, "load_s": top.load_s,
                       "patterns": rows, "mem_after": mem})


def suite_correct(a, env, bins, models, R, sampler):
    """Greedy outputs: single vs RPC split (same build), and every build vs the first (same topology)."""
    prompts = [make_prompt(64, 9000 + i) for i in range(3)]
    kinds = ["single"] + (["rpc2"] if len(env.gpus) >= 2 else []) + (["native2"] if len(env.gpus) >= 2 and a.native else [])
    for mname in a.correct_models:
        mpath = models[mname]
        ref = {}
        for b in bins:
            for kind in kinds:
                if kind == "single" and mname in a.no_single:
                    continue
                top = Topology(env, bins[b], mpath, "rpcN" if kind == "rpc2" else kind, 4096, 1, [],
                               os.path.join(a.work, f"cor_{mname}_{b}_{kind}"), n_rpc = 1)
                case = {"suite": "correct", "model": mname, "build": b, "topology": kind}
                try:
                    top.start()
                    outs = [complete(top.port, p, 48, stream = False) for p in prompts]
                    case["ok"] = all(o["ok"] for o in outs)
                    case["errors"] = [o["err"] for o in outs if not o["ok"]][:2]
                    texts = [o["content"] for o in outs]
                    case["texts"] = [t[:160] for t in texts]
                    key = ("first",)
                    if key not in ref:
                        ref[key] = (b, kind, texts)
                    rb, rk, rt = ref[key]
                    case["vs"] = f"{rb}/{rk}"
                    case["match"] = [t == r for t, r in zip(texts, rt)]
                except Exception as e:  # noqa: BLE001
                    case["ok"] = False
                    case["errors"] = [str(e)[-1500:]]
                finally:
                    top.stop()
                R.add(case)
                log(f"correct {mname} {b} {kind}: ok={case.get('ok')} match={case.get('match')}")


def suite_stress(a, env, bins, models, R, sampler):
    for mname in a.stress_models:
        for b in a.stress_builds:
            for kind in a.stress_topologies:
                n_rpc = int(kind[3:]) - 1 if kind.startswith("rpc") else 0
                if kind.startswith("rpc") and len(env.gpus) < n_rpc + 1:
                    continue
                top = Topology(env, bins[b], models[mname], "rpcN" if n_rpc else kind, a.ctx, a.parallel, [],
                               os.path.join(a.work, f"st_{mname}_{b}_{kind}"), n_rpc = n_rpc)
                case = {"suite": "stress", "model": mname, "build": b, "topology": kind, "parallel": a.parallel}
                try:
                    top.start()
                    case["load_s"] = top.load_s
                    rows = []
                    for pn, spec in patterns(a.scale, a.parallel).items():
                        run_pattern(top, sampler, pn, spec, rows)
                    case["patterns"] = rows
                except Exception as e:  # noqa: BLE001
                    case["error"] = str(e)[-1500:]
                finally:
                    case["exit"] = top.stop()
                R.add(case)


def suite_groups(a, env, bins, models, R, sampler):
    """--pipeline-groups 1 vs G on the same split: throughput, latency and per-device overlap."""
    for mname in a.groups_models:
        for b in a.groups_builds:
            for kind in a.groups_topologies:
                n_rpc = int(kind[3:]) - 1 if kind.startswith("rpc") else 0
                if len(env.gpus) < max(2, n_rpc + 1):
                    continue
                for rnd in range(a.rounds):
                    for g in a.groups:
                        for order in (a.orders if n_rpc else ["-"]):
                            extra = ["--pipeline-groups", str(g)] if g > 1 else []
                            par = max(a.parallel, g) // g * g
                            top = Topology(env, bins[b], models[mname], "rpcN" if n_rpc else kind, a.ctx, par, extra,
                                           os.path.join(a.work, f"gr_{mname}_{b}_{kind}_{g}_{order}"), n_rpc = n_rpc,
                                           order = order if order != "-" else "rpc_first")
                            case = {"suite": "groups", "model": mname, "build": b, "topology": kind, "groups": g,
                                    "order": order, "round": rnd, "parallel": par}
                            try:
                                top.start()
                                rows = []
                                pats = patterns(a.scale, par)
                                for pn in ("single", "equal", "mixed", "flood"):
                                    run_pattern(top, sampler, pn, pats[pn], rows)
                                case["patterns"] = rows
                            except Exception as e:  # noqa: BLE001
                                case["error"] = str(e)[-1500:]
                            finally:
                                case["exit"] = top.stop()
                            R.add(case)


def suite_load(a, env, bins, models, R, sampler):
    """Time to ready for an RPC split, per build; with and without the rpc-server file cache."""
    if len(env.gpus) < 2:
        return
    for mname in a.load_models:
        for rnd in range(a.rounds):
            for b in bins:
                for cache in (False, True):
                    top = Topology(env, bins[b], models[mname], "rpcN", 4096, 1, [],
                                   os.path.join(a.work, f"ld_{mname}_{b}_{cache}"), n_rpc = 1, rpc_cache = cache)
                    case = {"suite": "load", "model": mname, "build": b, "rpc_cache": cache, "round": rnd}
                    try:
                        top.start()
                        case["load_s"] = top.load_s
                        r = complete(top.port, make_prompt(32, 1), 16, stream = False)
                        case["first_ok"] = r["ok"]
                    except Exception as e:  # noqa: BLE001
                        case["error"] = str(e)[-800:]
                    finally:
                        top.stop()
                    R.add(case)
                    log(f"load {mname} {b} cache={cache}: {case.get('load_s')} s")


def suite_teardown(a, env, bins, models, R, sampler):
    """Kill the rpc-server mid-generation: does llama-server abort, hang, or report and carry on?"""
    if len(env.gpus) < 2:
        return
    mname = a.teardown_model
    for b in bins:
        top = Topology(env, bins[b], models[mname], "rpcN", 4096, 2, [], os.path.join(a.work, f"td_{b}"), n_rpc = 1)
        case = {"suite": "teardown", "model": mname, "build": b}
        try:
            top.start()
            holder = {}
            th = threading.Thread(target = lambda: holder.update(r = complete(top.port, make_prompt(64, 3), 2000, timeout = 120)),
                                  daemon = True)
            th.start()
            time.sleep(3)
            top.rpcs[0].stop(sig = signal.SIGKILL)
            th.join(timeout = 150)
            case["inflight"] = {k: holder.get("r", {}).get(k) for k in ("ok", "err", "n")}
            case["inflight_returned"] = not th.is_alive()
            time.sleep(5)
            case["server_alive_after_kill"] = top.server.alive()
            case["server_rc_after_kill"] = top.server.p.returncode
            if top.server.alive():
                t = time.time()
                r = complete(top.port, make_prompt(16, 4), 8, stream = False, timeout = 60)
                case["next_request"] = {"ok": r["ok"], "err": r["err"], "s": round(time.time() - t, 1)}
            t = time.time()
            rc = top.server.stop(wait = 60)
            case["shutdown_rc"] = rc
            case["shutdown_s"] = round(time.time() - t, 1)
            case["aborted"] = rc in (-6, 134)
            case["log_tail"] = top.server.tail(25)
        except Exception as e:  # noqa: BLE001
            case["error"] = str(e)[-1500:]
        finally:
            top.stop()
        R.add(case)
        log(f"teardown {b}: alive={case.get('server_alive_after_kill')} rc={case.get('shutdown_rc')} aborted={case.get('aborted')}")
        # The worker dies while the server is IDLE, then the server is shut down (#203's claim).
        top = Topology(env, bins[b], models[mname], "rpcN", 4096, 2, [], os.path.join(a.work, f"tdi_{b}"), n_rpc = 1)
        case = {"suite": "teardown_idle", "model": mname, "build": b}
        try:
            top.start()
            r = complete(top.port, make_prompt(32, 5), 8, stream = False)
            case["warm_ok"] = r["ok"]
            top.rpcs[0].stop(sig = signal.SIGKILL)
            time.sleep(3)
            case["server_alive_after_kill"] = top.server.alive()
            t = time.time()
            rc = top.server.stop(wait = 60)
            case["shutdown_rc"] = rc
            case["shutdown_s"] = round(time.time() - t, 1)
            case["aborted"] = rc in (-6, 134)
            case["log_tail"] = top.server.tail(12)
        except Exception as e:  # noqa: BLE001
            case["error"] = str(e)[-1500:]
        finally:
            top.stop()
        R.add(case)
        log(f"teardown_idle {b}: rc={case.get('shutdown_rc')} aborted={case.get('aborted')} in {case.get('shutdown_s')}s")


SUITES = {"single": suite_single, "correct": suite_correct, "stress": suite_stress, "groups": suite_groups,
          "load": suite_load, "teardown": suite_teardown}


def kv_list(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", action = "append", required = True, help = "NAME=DIR with llama-server, llama-bench, ggml-rpc-server")
    ap.add_argument("--model", action = "append", required = True, help = "NAME=PATH.gguf")
    ap.add_argument("--suite", default = "single,correct,stress,groups,load,teardown")
    ap.add_argument("--out", required = True)
    ap.add_argument("--work", default = "/tmp/rpc_harness_work")
    ap.add_argument("--gpus", default = "0,1")
    ap.add_argument("--vendor", default = "nvidia", choices = ["nvidia", "amd", "cpu"])
    ap.add_argument("--threads", type = int, default = max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--scale", default = "quick", choices = ["quick", "full"])
    ap.add_argument("--rounds", type = int, default = 2)
    ap.add_argument("--ctx", type = int, default = 16384)
    ap.add_argument("--parallel", type = int, default = 8)
    ap.add_argument("--single-models", default = "")
    ap.add_argument("--correct-models", default = "")
    ap.add_argument("--no-single", default = "", help = "models too large for one device (skip single in correct)")
    ap.add_argument("--native", action = "store_true", help = "also run the native one-process two-GPU split")
    ap.add_argument("--stress-models", default = "")
    ap.add_argument("--stress-builds", default = "")
    ap.add_argument("--stress-topologies", default = "single,rpc2")
    ap.add_argument("--groups-models", default = "")
    ap.add_argument("--groups-builds", default = "")
    ap.add_argument("--groups-topologies", default = "rpc2")
    ap.add_argument("--groups", default = "1,2")
    ap.add_argument("--orders", default = "rpc_first,local_first")
    ap.add_argument("--load-models", default = "")
    ap.add_argument("--teardown-model", default = "")
    a = ap.parse_args()
    bins, models = kv_list(a.bin), kv_list(a.model)
    for k in ("single_models", "correct_models", "no_single", "stress_models", "stress_builds", "stress_topologies",
              "groups_models", "groups_builds", "groups_topologies", "load_models", "orders"):
        setattr(a, k, [x for x in getattr(a, k).split(",") if x])
    a.groups = [int(x) for x in a.groups.split(",") if x]
    gpus = [int(x) for x in a.gpus.split(",") if x != ""]
    env = Env(gpus, a.vendor, a.threads)
    os.makedirs(a.work, exist_ok = True)
    meta = {"argv": sys.argv, "bins": bins, "models": models, "gpus": gpus, "vendor": a.vendor,
            "host": socket.gethostname(), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for b, d in bins.items():
        try:
            meta.setdefault("devices", {})[b] = env.local_name(d, gpus)
        except Exception as e:  # noqa: BLE001
            meta.setdefault("devices", {})[b] = f"error: {e}"
    if a.vendor == "nvidia" and shutil.which("nvidia-smi"):
        meta["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap",
                                             "--format=csv,noheader"], capture_output = True, text = True).stdout.strip()
    R = Results(a.out, meta)
    sampler = Sampler(gpus, a.vendor) if a.vendor != "cpu" and gpus else None
    if sampler:
        sampler.start()
    try:
        for s in [x for x in a.suite.split(",") if x]:
            log(f"=== suite {s}")
            try:
                SUITES[s](a, env, bins, models, R, sampler)
            except Exception as e:  # noqa: BLE001
                R.add({"suite": s, "fatal": f"{type(e).__name__}: {e}"})
                log(f"suite {s} failed: {e}")
    finally:
        if sampler:
            sampler.stop()
        R.doc["meta"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        R.flush()
    log("=== done")


if __name__ == "__main__":
    main()

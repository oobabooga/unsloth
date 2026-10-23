"""Drive real llama-server loads and sample the PR's load_progress() (and main's, for contrast) every 25 ms.

    probe.py --server PATH --model GGUF --out results.json --run NAME="ARGS" [--run ...] [--flush CMD]

pr_progress.py / main_progress.py are load_progress() and its helpers, extracted verbatim from each tree's
llama_cpp.py (extract.py). A run ends when /health answers 200; the timeline shows whether the bar moves
while the weights load or only jumps at the end.
"""

import argparse
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_progress  # noqa: E402
import pr_progress  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--server", required = True)
ap.add_argument("--model", required = True)
ap.add_argument("--out", required = True)
ap.add_argument("--run", action = "append", default = [])
ap.add_argument("--flush", default = "", help = "shell command run before each load to drop the file cache")
ap.add_argument("--timeout", type = float, default = 300)
args = ap.parse_args()
GB = 1e9


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def health(port: int) -> int:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout = 1) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def backend(module, proc):
    inst = module.LlamaCppBackend.__new__(module.LlamaCppBackend)
    inst._process = proc
    inst._gguf_path = args.model
    inst._healthy = False
    return inst


def one(name: str, extra: str) -> dict:
    flushed = None
    if args.flush:
        try:
            flushed = subprocess.run(args.flush, shell = True, capture_output = True, text = True, timeout = 120).returncode
        except subprocess.TimeoutExpired:
            flushed = "timeout"
    port = free_port()
    argv = [args.server, "-m", args.model, "--host", "127.0.0.1", "--port", str(port), "-c", "2048"] + shlex.split(extra)
    log = open(Path(args.out).with_name(f"server-{name}.log"), "w", encoding = "utf-8", errors = "replace")
    t0 = time.monotonic()
    proc = subprocess.Popen(argv, stdout = log, stderr = subprocess.STDOUT)
    pr, main = backend(pr_progress, proc), backend(main_progress, proc)
    rows, ready, last_health = [], None, 0.0
    while time.monotonic() - t0 < args.timeout:
        t = time.monotonic() - t0
        p, m = pr.load_progress(), main.load_progress()
        rows.append({
            "t": round(t, 3),
            "pr": p["fraction"] if p else None,
            "main": m["fraction"] if m else None,
            "read": pr_progress.LlamaCppBackend._read_io_bytes(proc.pid),
            "rss": pr_progress.LlamaCppBackend._read_rss_bytes(proc.pid),
            "total": p["bytes_total"] if p else None,
        })
        if proc.poll() is not None:
            break
        if t - last_health >= 0.1:
            last_health = t
            if health(port) == 200:
                ready = t
                break
        time.sleep(0.025)
    proc.terminate()
    try:
        proc.wait(timeout = 20)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()
    loading = [r for r in rows if ready is None or r["t"] < ready]
    def at(frac):
        if not loading or ready is None:
            return None
        target = ready * frac
        row = min(loading, key = lambda r: abs(r["t"] - target))
        return row["pr"]
    prs = [r["pr"] for r in loading if r["pr"] is not None]
    return {
        "name": name,
        "args": extra,
        "flush_rc": flushed,
        "ready_s": None if ready is None else round(ready, 2),
        "exit_code": proc.returncode if ready is None else None,
        "samples": len(loading),
        "pr_at_25_50_75_pct_of_load": [at(0.25), at(0.5), at(0.75)],
        "pr_max_before_ready": max(prs) if prs else None,
        "pr_monotonic": all(b >= a for a, b in zip(prs, prs[1:])) if prs else None,
        "main_values": sorted({r["main"] for r in loading}, key = lambda v: (v is None, v or 0))[:6],
        "read_gb_max": round(max((r["read"] or 0) for r in rows) / GB, 2) if rows else None,
        "rss_gb_max": round(max((r["rss"] or 0) for r in rows) / GB, 2) if rows else None,
        "total_gb": round(rows[-1]["total"] / GB, 2) if rows and rows[-1]["total"] else None,
        "timeline": rows[:: max(1, len(rows) // 40)],
    }


info = {"platform": platform.platform(), "python": sys.version.split()[0], "has_proc": os.path.isdir("/proc")}
try:
    import psutil

    info["psutil"] = psutil.__version__
    info["psutil_has_io_counters"] = hasattr(psutil.Process(), "io_counters")
except ImportError:
    info["psutil"] = None
print(json.dumps(info), flush = True)
results = {"info": info, "model_bytes": os.path.getsize(args.model), "runs": []}
for spec in args.run:
    name, _, extra = spec.partition("=")
    r = one(name, extra)
    results["runs"].append(r)
    print(
        f"{name:14s} ready {r['ready_s']}s  samples {r['samples']}  PR at 25/50/75% of load {r['pr_at_25_50_75_pct_of_load']}"
        f"  PR max before ready {r['pr_max_before_ready']}  monotonic {r['pr_monotonic']}  main {r['main_values']}"
        f"  read {r['read_gb_max']} GB  rss {r['rss_gb_max']} GB  total {r['total_gb']} GB  exit {r['exit_code']}  flush {r['flush_rc']}",
        flush = True,
    )
    Path(args.out).write_text(json.dumps(results, indent = 1), encoding = "utf-8")

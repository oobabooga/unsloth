#!/usr/bin/env python3
"""Probe: drive a prebuilt llama-server through corruption cells and record what it said.

`--checkout` is a directory holding `llama-server`. Observes only: each
completion's text and the corruption signatures it matches go to --out as JSON;
the criteria module judges.

Cells (--cells):
  single     one short prompt, `-np 1`; the negative control
  multiseg   three multi-segment prompts, `-np 1`
  unified4   four concurrent 6k to 12k character prompts on
             `-np 4 --kv-unified -b 2048 -ub 512`

A known-good sentinel model runs before and after the cells, because a poisoned
GPU answers `/` to everything until a reset and that must read as INCONCLUSIVE.
The binary is fingerprinted (sha256, --version, resolved backend libraries).

  python amd_ci/probes/llamacpp_server_probe.py --state head --checkout "$HEAD_BIN" \\
      --out out/obs_head.json --model model-00001-of-00003.gguf \\
      --sentinel-model sentinel.gguf --prompts wikitext2_test.txt \\
      --env GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 --gpu-var NONE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

SIGS = {
    "slash_run":   re.compile(r"/{6,}"),
    "bang_run":    re.compile(r"!{6,}"),
    "word_x4":     re.compile(r"(\b\S{2,}\b)(?:[ _]\1){3,}"),
    "replacement": re.compile("�"),
    "char_run":    re.compile(r"(.)\1{24,}"),
}
LOG_SIGS = ("inconsistent sequence positions", "GGML_ASSERT", "failed to decode",
            "nan", "NaN", "out of memory", "Memory access fault", "HSA_STATUS_ERROR")

MULTISEG = [
    "Context: the following is a short quiz.\n\nQuestion: What is the capital of France?\nAnswer:",
    "User: My name is Bob.\nAssistant: Noted, Bob.\nUser: What is my name?\nAssistant:",
    "Previous conversation: the user said hi and the assistant replied hello.\n\n"
    "User: What is the capital of France?\nAssistant:",
]
UNIFIED_LENGTHS = (6000, 8000, 10000, 12000)


def server_exe(bin_dir: str | Path) -> str:
    for n in ("llama-server", "llama-server.exe"):
        for d in (Path(bin_dir), Path(bin_dir) / "build" / "bin"):
            if (d / n).is_file():
                return str(d / n)
    return str(Path(bin_dir) / "llama-server")


def signatures(text: str) -> list[str]:
    return sorted(k for k, rx in SIGS.items() if rx.search(text or ""))


def cross_slot_shared(texts: list[str], window: int = 60, min_len: int = 80) -> list[list]:
    """Pairs of slots whose outputs share a `window`-character run.

    Two independent prompts answered with the same 60 characters is what a slot
    reading another slot's cache looks like; short outputs are skipped because
    a shared stop phrase is not evidence.
    """
    hits: list[list] = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            ti, tj = texts[i] or "", texts[j] or ""
            if len(ti) <= min_len or len(tj) <= min_len:
                continue
            for k in range(0, len(ti) - window, 20):
                if ti[k:k + window] in tj:
                    hits.append([i, j, ti[k:k + window]])
                    break
    return hits


def post(port: int, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                 data = json.dumps(payload).encode("utf-8"),
                                 headers = {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout = timeout) as r:
        return json.loads(r.read().decode("utf-8", errors = "replace"))


def completion(port: int, prompt: str, n: int, timeout: int = 1800) -> dict:
    t0 = time.time()
    try:
        r = post(port, {"prompt": prompt, "n_predict": n, "temperature": 0, "seed": 1,
                        "cache_prompt": False, "top_k": 1}, timeout)
        timings = r.get("timings") or {}
        out = {"ok": True, "text": r.get("content", ""), "n_tokens": r.get("tokens_predicted"),
               "prompt_tokens": r.get("tokens_evaluated"),
               "stop": r.get("stop_type") or r.get("stopped_eos"),
               "draft_n": timings.get("draft_n"), "draft_n_accepted": timings.get("draft_n_accepted"),
               "tg_tps": timings.get("predicted_per_second")}
    except Exception as e:  # noqa: BLE001
        out = {"ok": False, "error": f"{type(e).__name__}: {e}", "text": ""}
    out["secs"] = round(time.time() - t0, 1)
    out["signatures"] = signatures(out["text"])
    return out


def fingerprint(bin_dir: str) -> dict:
    fp: dict = {}
    exe = Path(server_exe(bin_dir))
    if exe.is_file():
        fp["llama-server_sha256"] = hashlib.sha256(exe.read_bytes()).hexdigest()
    for lib in ("libggml-hip.so", "libggml-cuda.so", "libggml-vulkan.so",
                "ggml-hip.dll", "ggml-cuda.dll", "ggml-vulkan.dll"):
        for cand in (Path(bin_dir) / lib, Path(bin_dir) / "build" / "bin" / lib):
            if not cand.is_file():
                continue
            fp.setdefault("backend_libs", []).append(lib)
            if not shutil.which("ldd"):
                continue
            r = subprocess.run(["ldd", str(cand)], capture_output = True, text = True,
                               encoding = "utf-8", errors = "replace")
            fp[lib] = [l.strip() for l in r.stdout.splitlines()
                       if any(k in l for k in ("hip", "roc", "cuda", "vulkan", "not found"))][:12]
    return fp


class Server:
    """One llama-server, started with the arm's env and stopped by PID."""

    def __init__(self, a, extra: list[str], log: Path):
        self.a, self.extra, self.log = a, extra, log
        self.p = None
        self.fh = None

    def __enter__(self):
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{self.a.bin}{os.pathsep}{self.a.bin}/build/bin{os.pathsep}" \
            + env.get("LD_LIBRARY_PATH", "")
        for var in self.a.gpu_var:
            if var != "NONE":
                env[var] = str(self.a.gpu)
        for kv in self.a.env:
            k, _, v = kv.partition("=")
            env[k] = v
        for k in self.a.unset_env:
            env.pop(k, None)
        cmd = [server_exe(self.a.bin), "-m", self.a.model, "--host", "127.0.0.1",
               "--port", str(self.a.port), *self.a.base_args, *self.extra, *self.a.server_arg]
        self.fh = open(self.log, "w", encoding = "utf-8", errors = "replace")
        watched = {k: env.get(k) for k in self.a.watch_env}
        self.fh.write("CMD: " + " ".join(cmd) + "\nENV: " + json.dumps(watched) + "\n")
        self.fh.flush()
        self.p = subprocess.Popen(cmd, stdout = self.fh, stderr = subprocess.STDOUT, env = env)
        for _ in range(max(1, self.a.load_timeout // 2)):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.a.port}/health", timeout = 2) as r:
                    if b'"ok"' in r.read():
                        return self
            except Exception:  # noqa: BLE001
                pass
            if self.p.poll() is not None:
                raise RuntimeError(f"server died rc={self.p.returncode}")
            time.sleep(2)
        raise RuntimeError("server load timeout")

    def __exit__(self, *_):
        # By PID, never by pattern: `pkill -f` matches the shell running it.
        if self.p and self.p.poll() is None:
            self.p.send_signal(signal.SIGINT if os.name != "nt" else signal.SIGTERM)
            try:
                self.p.wait(30)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        if self.fh:
            self.fh.close()

    def log_hits(self) -> dict:
        txt = Path(self.log).read_text(encoding = "utf-8", errors = "replace")
        hits = {s: txt.count(s) for s in LOG_SIGS}
        return {"log_sig_counts": {k: v for k, v in hits.items() if v},
                "buffers": [l.strip() for l in txt.splitlines() if "model buffer size" in l][:6],
                "version_lines": [l.strip() for l in txt.splitlines()
                                  if "build:" in l or "version:" in l][:2],
                "rc": self.p.returncode if self.p else None}


def run_cells(a, res: dict, log_dir: Path) -> None:
    cells = [c.strip() for c in a.cells.split(",") if c.strip()]
    unknown = [c for c in cells if c not in ("single", "multiseg", "unified4")]
    if unknown:
        raise SystemExit(f"unknown cell(s) {unknown}; choose from single, multiseg, unified4")

    if "single" in cells or "multiseg" in cells:
        with Server(a, ["-c", str(a.ctx_single), "-np", "1"], log_dir / f"srv_{a.state}_np1.log") as s:
            if "single" in cells:
                res["cells"]["single"] = {"prompts": [completion(a.port, a.sentinel_prompt, 64)]}
            if "multiseg" in cells:
                prompts = MULTISEG
                if a.multiseg_file:
                    prompts = json.loads(Path(a.multiseg_file).read_text(encoding = "utf-8"))
                res["cells"]["multiseg"] = {"prompts": [completion(a.port, p, 64) for p in prompts]}
            hits = s.log_hits()
        for c in ("single", "multiseg"):
            if c in res["cells"]:
                res["cells"][c].update(hits)

    if "unified4" in cells:
        if not a.prompts:
            raise SystemExit("unified4 needs --prompts (a long UTF-8 text file)")
        # UTF-8 explicitly: the Windows default is cp1252.
        text = Path(a.prompts).read_text(encoding = "utf-8", errors = "replace")
        need = 15000 * (len(UNIFIED_LENGTHS) - 1) + UNIFIED_LENGTHS[-1]
        if len(text) < need:
            raise SystemExit(f"--prompts holds {len(text)} characters; unified4 needs {need}")
        prompts = [text[i * 15000: i * 15000 + n] + "\n\nSummarize the passage above in three sentences."
                   for i, n in enumerate(UNIFIED_LENGTHS)]
        extra = ["-c", str(a.ctx_unified), "-np", "4", "--kv-unified", "-b", "2048", "-ub", "512"]
        with Server(a, extra, log_dir / f"srv_{a.state}_np4.log") as s:
            results: list = [None] * 4

            def run(i):
                results[i] = completion(a.port, prompts[i], a.n_predict)

            # Arrival order decides slot assignment and batch packing.
            order = [int(x) for x in a.arrival_order.split(",")] if a.arrival_order else list(range(4))
            th = {i: threading.Thread(target = run, args = (i,)) for i in range(4)}
            for k, i in enumerate(order):
                if k and a.stagger > 0:
                    time.sleep(a.stagger)
                th[i].start()
            for i in range(4):
                th[i].join()
            res["cells"]["unified4"] = {
                "prompts": results,
                "cross_slot_shared_60char": cross_slot_shared([r["text"] for r in results]),
                "order": order, "stagger": a.stagger, **s.log_hits()}


def sentinel(a, res: dict, log_dir: Path, tag: str) -> None:
    if not a.sentinel_model:
        return
    model = a.model
    a.model = str(Path(a.sentinel_model).resolve())
    try:
        with Server(a, ["-c", "2048", "-np", "1"], log_dir / f"srv_{a.state}_sentinel_{tag}.log"):
            r = completion(a.port, a.sentinel_prompt, 8, timeout = 300)
        r["clean"] = a.sentinel_expect in r["text"] and not r["signatures"]
    except Exception as e:  # noqa: BLE001
        r = {"ok": False, "error": f"{type(e).__name__}: {e}", "text": "", "signatures": [],
             "clean": False}
    finally:
        a.model = model
    res.setdefault("sentinel", {})[tag] = r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path,
                    help = "directory holding llama-server (a fetched release)")
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--model", required = True, help = "GGUF path; the first shard of a split file")
    ap.add_argument("--sentinel-model", default = "",
                    help = "small known-good GGUF run before and after the cells")
    ap.add_argument("--sentinel-prompt", default = "The capital of France is")
    ap.add_argument("--sentinel-expect", default = "Paris")
    ap.add_argument("--prompts", default = "", help = "long UTF-8 text for the unified4 cell")
    ap.add_argument("--multiseg-file", default = "", help = "JSON list of prompts replacing the built-in multiseg set")
    ap.add_argument("--cells", default = "single,multiseg,unified4")
    ap.add_argument("--env", action = "append", default = [], metavar = "K=V")
    ap.add_argument("--unset-env", action = "append", default = [], metavar = "K")
    ap.add_argument("--watch-env", action = "append", default = ["GGML_CUDA_ENABLE_UNIFIED_MEMORY"],
                    help = "env names echoed into the server log header")
    ap.add_argument("--gpu-var", action = "append", default = None,
                    help = "env var(s) that pin the GPU; NONE (the default) to leave selection alone")
    ap.add_argument("--gpu", default = "0")
    ap.add_argument("--port", type = int, default = 8650)
    ap.add_argument("--load-timeout", type = int, default = 1800)
    ap.add_argument("--n-predict", type = int, default = 128)
    ap.add_argument("--ctx-single", type = int, default = 8192)
    ap.add_argument("--ctx-unified", type = int, default = 32768)
    ap.add_argument("--base-arg", action = "append", default = None,
                    help = "server flags every cell gets; default -ngl 999 --fit off -fa on --no-warmup")
    ap.add_argument("--server-arg", action = "append", default = [],
                    help = "extra server flag appended to every launch")
    ap.add_argument("--arrival-order", default = "", help = "e.g. 1,0,2,3 for the unified4 prompts")
    ap.add_argument("--stagger", type = float, default = 0.0, help = "seconds between unified4 starts")
    a = ap.parse_args()

    a.gpu_var = a.gpu_var or ["NONE"]
    a.base_args = a.base_arg if a.base_arg is not None else \
        ["-ngl", "999", "--fit", "off", "-fa", "on", "--no-warmup"]
    a.bin = str(a.checkout.resolve())
    a.model = str(Path(a.model).resolve())
    log_dir = a.out.parent / f"cells_{a.state}"
    log_dir.mkdir(parents = True, exist_ok = True)

    res: dict = {"state": a.state, "checkout": a.bin, "env": a.env, "unset_env": a.unset_env,
                 "gpu_var": a.gpu_var, "cells": {}}
    exe = server_exe(a.bin)
    if not Path(exe).is_file():
        res["setup_error"] = f"no llama-server under {a.bin}"
    else:
        ver = subprocess.run([exe, "--version"], capture_output = True, text = True,
                             encoding = "utf-8", errors = "replace",
                             env = {**os.environ, "LD_LIBRARY_PATH": f"{a.bin}{os.pathsep}{a.bin}/build/bin"})
        res["version"] = (ver.stdout + ver.stderr).strip().splitlines()[:2]
        res["fingerprint"] = fingerprint(a.bin)
        sentinel(a, res, log_dir, "pre")
        try:
            run_cells(a, res, log_dir)
        except Exception as e:  # noqa: BLE001
            res["error"] = f"{type(e).__name__}: {e}"
        sentinel(a, res, log_dir, "post")

    res["any_signature"] = sorted({s for c in res["cells"].values()
                                   for p in c.get("prompts") or [] for s in p["signatures"]})
    a.out.parent.mkdir(parents = True, exist_ok = True)
    a.out.write_text(json.dumps(res, indent = 1), encoding = "utf-8")
    print(json.dumps({"state": a.state, "version": res.get("version"),
                      "any_signature": res["any_signature"],
                      "cells": {k: [(p.get("n_tokens"), p["signatures"], p.get("error"))
                                    for p in v.get("prompts") or []] for k, v in res["cells"].items()},
                      "error": res.get("error") or res.get("setup_error")}, indent = 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

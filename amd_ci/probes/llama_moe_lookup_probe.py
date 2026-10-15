#!/usr/bin/env python3
"""Probe: llama.cpp MoE cache device lookup under each load mode.

Observes only. For one state (a directory holding that state's llama-completion
binary in bin/), runs the arms given by --arms, each --reps times, and records:
decode tok/s from the `eval time` timing line, the md5 of the generated text,
the MoE cache / host_map / materialize / load_mode log lines, the device and
buffer lines, and `free -m` before and after each run. Judging is left to
criteria/llama_moe_lookup.py.

Arm spec: label:load_mode[:ENV=VAL,ENV=VAL][:reps]   e.g. mmap_mat0:mmap:LLAMA_MOE_CACHE_MATERIALIZE=0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROMPT = ("Write a short, factual explanation of how a mixture-of-experts transformer "
          "routes each token to a subset of experts, and why that reduces compute.")

KEEP = re.compile(
    r"(init_device_lookup|materialize_mapped|ggml_cuda_host_map|host_map|cannot map the host experts|"
    r"run on the device|MoE cache|moe_cache|common_moe_cache_load_mode|load_mode|load_tensors:|"
    r"ggml_cuda_init|Device \d+:|found \d+ ROCm|system_info|llama_model_load|buffer size|"
    r"hit rate|failed to|error|warning: |using device|offloaded|prompt eval time|eval time|total time)",
    re.IGNORECASE)
EVAL = re.compile(r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) runs.*?([\d.]+) tokens per second")
PEVAL = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second")


def free_m() -> str:
    try:
        return subprocess.run(["free", "-m"], capture_output = True, text = True, timeout = 10).stdout
    except Exception as e:  # noqa: BLE001
        return f"free failed: {e}"


def parse_arm(spec: str) -> dict:
    parts = spec.split(":")
    label, mode = parts[0], parts[1]
    env: dict[str, str] = {}
    reps = None
    for p in parts[2:]:
        if p.isdigit():
            reps = int(p)
        elif p:
            for kv in p.split(","):
                k, _, v = kv.partition("=")
                env[k] = v
    return {"label": label, "mode": mode, "env": env, "reps": reps}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--model", required = True)
    ap.add_argument("--cache-mib", type = int, required = True)
    ap.add_argument("--n-predict", type = int, default = 128)
    ap.add_argument("--reps", type = int, default = 3)
    ap.add_argument("--timeout", type = int, default = 1200)
    ap.add_argument("--arms", nargs = "+", required = True)
    args = ap.parse_args()

    exe = args.checkout / "bin" / "llama-completion"
    logdir = args.out.parent / f"runs_{args.state}"
    logdir.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "exe": str(exe), "model": args.model,
                 "cache_mib": args.cache_mib, "n_predict": args.n_predict, "prompt": PROMPT,
                 "arms": {}}
    if not exe.is_file():
        obs["error"] = f"missing {exe}"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 1

    for spec in args.arms:
        arm = parse_arm(spec)
        reps = arm["reps"] or args.reps
        runs = []
        for r in range(reps):
            env = dict(os.environ)
            env.update(arm["env"])
            cmd = [str(exe), "-m", args.model, "-ngl", "99", "-cmoe", "--moe-cache-mib", str(args.cache_mib),
                   "-fa", "on", "-c", "2048", "-p", PROMPT, "-n", str(args.n_predict), "--temp", "0",
                   "--seed", "42", "-no-cnv", "--no-display-prompt", "--load-mode", arm["mode"]]
            before = free_m()
            t0 = time.time()
            try:
                p = subprocess.run(cmd, env = env, capture_output = True, timeout = args.timeout)
                rc, out_b, err_b = p.returncode, p.stdout, p.stderr
            except subprocess.TimeoutExpired as e:
                rc, out_b, err_b = "timeout", e.stdout or b"", e.stderr or b""
            wall = time.time() - t0
            after = free_m()
            text = out_b.decode("utf-8", "replace")
            err = err_b.decode("utf-8", "replace")
            stem = f"{arm['label']}_r{r}"
            (logdir / f"{stem}.stdout.txt").write_text(text, encoding = "utf-8")
            (logdir / f"{stem}.stderr.log").write_text(err, encoding = "utf-8")
            ev = [m for m in EVAL.finditer(err)]
            pe = [m for m in PEVAL.finditer(err)]
            run = {
                "rep": r, "rc": rc, "wall_s": round(wall, 2), "cmd": cmd, "env": arm["env"],
                "md5": hashlib.md5(out_b).hexdigest(), "text_len": len(text),
                "text_head": text[:300],
                "decode_tps": float(ev[-1].group(3)) if ev else None,
                "decode_runs": int(ev[-1].group(2)) if ev else None,
                "prompt_tps": float(pe[-1].group(3)) if pe else None,
                "free_before": before, "free_after": after,
                "lines": [ln for ln in err.splitlines() if KEEP.search(ln)][:400],
                "n_fallback": err.count("cannot map the host experts"),
                "n_materialize": err.count("materialize_mapped"),
                "n_device_lookup": err.count("run on the device"),
                "n_host_map_fail": err.count("failed to map"),
            }
            m = re.search(r"load_mode = (\S+?)\)", err)
            run["load_mode_logged"] = m.group(1) if m else None
            print(f"{args.state} {stem}: rc={rc} tps={run['decode_tps']} md5={run['md5'][:8]} "
                  f"fallback={run['n_fallback']} mat={run['n_materialize']} dev={run['n_device_lookup']} "
                  f"load_mode={run['load_mode_logged']} wall={wall:.1f}s", flush = True)
            runs.append(run)
        obs["arms"][arm["label"]] = {"mode": arm["mode"], "env": arm["env"], "runs": runs}
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

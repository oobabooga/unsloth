#!/usr/bin/env python3
"""Probe: fast_qk_norm_rope eager vs torch.compile bytes at one state (an unsloth checkout). Observes only.

1. qknorm_fp16_run.py with a fresh TRITON_CACHE_DIR: per dtype x packed x backend (inductor, aot_eager)
   per-output diff counts, max abs / ulp, error vs fp64, plus every Triton build of the qk kernels
   (options, ttir signature, asm instruction counts). The qk kernel cache files are kept in the out dir.
2. pytest tests/test_qk_norm_rope.py, per-test outcomes from junit XML.
The state's checkout is first on PYTHONPATH so its unsloth shadows the installed one.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pr13383_probe import run_pytest  # noqa: E402

TESTS = ["tests/test_qk_norm_rope.py"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 1800)
    args = ap.parse_args()
    args.out = args.out.resolve()
    out_dir = args.out.parent
    obs: dict = {"state": args.state, "checkout": args.checkout}
    obs["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = args.checkout,
                                     capture_output = True, text = True).stdout.strip()
    cache = out_dir / f"triton_cache_{args.state}"
    env = dict(os.environ)
    env["PYTHONPATH"] = args.checkout + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop("UNSLOTH_FUSED_QKNORM_ROPE", None)
    env["TRITON_CACHE_DIR"] = str(cache)
    env["QK_STATE"] = args.state
    env["TORCHINDUCTOR_CACHE_DIR"] = str(out_dir / f"inductor_cache_{args.state}")
    env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    env["UNSLOTH_COMPILE_LOCATION"] = str(out_dir / f"compiled_{args.state}_run")
    run: dict = {"results": [], "builds": []}
    try:
        p = subprocess.run([args.python, str(HERE / "qknorm_fp16_run.py")], cwd = str(out_dir), env = env,
                           capture_output = True, text = True, timeout = 1200)
        run["rc"], out, err = p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        run["rc"], out, err = "timeout", "", "TimeoutExpired"
    (out_dir / f"run_{args.state}.log").write_text(out + "\n--- stderr ---\n" + err, encoding = "utf-8")
    for line in out.splitlines():
        for tag, key in (("RESULT ", "results"), ("BUILD ", "builds")):
            if line.startswith(tag):
                run[key].append(json.loads(line[len(tag):]))
        if line.startswith("HEADER "):
            run["header"] = json.loads(line[len("HEADER "):])
        if line.strip() == "DONE":
            run["done"] = True
    run["stderr_tail"] = err[-3000:]
    run["exps"] = []
    try:
        p = subprocess.run([args.python, str(HERE / "qknorm_fp16_run2.py")], cwd = str(out_dir), env = env,
                           capture_output = True, text = True, timeout = 1200)
        run["rc2"], out2, err2 = p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        run["rc2"], out2, err2 = "timeout", "", "TimeoutExpired"
    (out_dir / f"run2_{args.state}.log").write_text(out2 + "\n--- stderr ---\n" + err2, encoding = "utf-8")
    for line in out2.splitlines():
        if line.startswith("EXP "):
            run["exps"].append(json.loads(line[len("EXP "):]))
    run["stderr2_tail"] = err2[-2000:]
    obs["run"] = run
    # Inductor output code that launches the qk kernels.
    code = out_dir / f"inductor_code_{args.state}"
    ind = out_dir / f"inductor_cache_{args.state}"
    if ind.is_dir():
        for f in ind.rglob("*.py"):
            try:
                if "qk_norm_rope" in f.read_text(encoding = "utf-8", errors = "replace"):
                    code.mkdir(parents = True, exist_ok = True)
                    shutil.copy2(f, code / f.name)
            except OSError:
                pass
    # Keep only the qk kernel builds (artifact size).
    keep = out_dir / f"qk_builds_{args.state}"
    if cache.is_dir():
        for meta in cache.rglob("_qk_norm_rope*.json"):
            dst = keep / meta.parent.name
            dst.mkdir(parents = True, exist_ok = True)
            for f in meta.parent.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst / f.name)
        shutil.rmtree(cache, ignore_errors = True)
    shutil.rmtree(out_dir / f"inductor_cache_{args.state}", ignore_errors = True)
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    obs["pytest"] = run_pytest(args.checkout, TESTS, out_dir, args.state, args.python, args.timeout)
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

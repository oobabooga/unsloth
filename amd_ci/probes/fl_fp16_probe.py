#!/usr/bin/env python3
"""Probe: zoo main fused linear losses (CE, log-probs, GKD JSD) in fp16 on gfx1151. Observes only
(criteria/fl_fp16.py judges).

Every arm runs in the HEAD state's invocation from the head checkout (zoo main); the base invocation records "skipped".
Arms (fresh process each): acc (fl_check.py --mode acc: fused vs HF / TRL fp16 math vs fp64 oracle),
perf_fused_1, perf_ref_1, perf_ref_2, perf_fused_2 (fl_check.py --mode perf, A B B A), train (fl_train.py: tiny Qwen3
fp16 LoRA SFT 3 steps + GRPO 2 steps through Unsloth)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--perf-shape", default = "4096,2048,151936")
    ap.add_argument("--arm-timeout", type = int, default = 2400)
    args = ap.parse_args()
    co = Path(args.checkout)
    obs: dict = {"state": args.state, "checkout": str(co),
                 "zoo_sha": subprocess.run(["git", "-C", str(co), "rev-parse", "HEAD"], capture_output = True,
                                           encoding = "utf-8").stdout.strip()}
    if args.state != "head":
        obs["skipped"] = "all arms run in the head state's invocation (fused vs plain-torch reference on zoo main)"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0
    work = args.out.parent / "fl_fp16"
    work.mkdir(parents = True, exist_ok = True)
    env0 = dict(os.environ, UNSLOTH_DISABLE_AUTO_UPDATES = "1", PYTHONUTF8 = "1", UNSLOTH_IS_PRESENT = "1")
    env0.pop("CI", None)  # torch turns custom-op aliasing warnings into errors under CI; users do not set it
    env0.setdefault("HF_HOME", str(Path(os.environ.get("AMD_CI_WORK", str(work))) / "hf"))
    n, h, v = args.perf_shape.split(",")
    chk, trn = str(HERE / "fl_check.py"), str(HERE / "fl_train.py")
    perf = lambda arm: [chk, "--mode", "perf", "--arm", arm, "--n", n, "--h", h, "--vocabs", v]  # noqa: E731
    arms = [("acc", [chk, "--mode", "acc"]), ("perf_fused_1", perf("fused")), ("perf_ref_1", perf("ref")),
            ("perf_ref_2", perf("ref")), ("perf_fused_2", perf("fused")), ("train", [trn])]
    obs["arms"] = {}
    for name, argv in arms:
        d = work / name
        d.mkdir(parents = True, exist_ok = True)
        env = dict(env0, PYTHONPATH = os.pathsep.join([str(co)] + ([env0["PYTHONPATH"]] if env0.get("PYTHONPATH") else [])),
                   UNSLOTH_COMPILE_LOCATION = str(d / "cc"), TORCHINDUCTOR_CACHE_DIR = str(d / "ind"))
        t = time.time()
        try:
            p = subprocess.run([args.python, "-u"] + argv + ["--out", str(d / "result.json")], cwd = d, env = env,
                               capture_output = True, encoding = "utf-8", errors = "replace", timeout = args.arm_timeout)
            rc, text = p.returncode, (p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or "")
        except subprocess.TimeoutExpired as exc:
            rc, text = -1, "TimeoutExpired\n" + str(exc.stdout)[-3000:]
        (d / "log.txt").write_text(text, encoding = "utf-8")
        rec = {"rc": rc, "s": round(time.time() - t, 1), "tail": text[-2500:]}
        if (d / "result.json").is_file():
            rec["result"] = json.loads((d / "result.json").read_text(encoding = "utf-8"))
        obs["arms"][name] = rec
        print(f"arm {name} rc={rc} {rec['s']}s", flush = True)
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        for sub in ("cc", "ind", "sft_out", "grpo_out"):
            subprocess.run(["rm", "-rf", "--", str(d / sub)])
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Probe (unslothai/unsloth#17738): on gfx1151, packed vs padded SFT on a fake Qwen3.5-4B-shaped hybrid
model (4 layers, random weights), plus an attention-kernel microbench. Observes only."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd, env, log, timeout):
    with open(log, "w", encoding = "utf-8") as fh:
        try:
            return subprocess.run(cmd, env = env, stdout = fh, stderr = subprocess.STDOUT, timeout = timeout).returncode
        except subprocess.TimeoutExpired:
            return "timeout"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--bench-states", default = "base")
    ap.add_argument("--mode-states", default = "base")
    ap.add_argument("--modes", default = "padded,packed,packed_varlen")
    a = ap.parse_args()
    logs = a.out.parent
    work = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "i17738"
    work.mkdir(parents = True, exist_ok = True)
    base_env = dict(os.environ, PYTHONFAULTHANDLER = "1", UNSLOTH_DISABLE_AUTO_UPDATES = "1",
                    PYTHONPATH = a.checkout + os.pathsep + os.environ.get("PYTHONPATH", ""))
    base_env.pop("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", None)
    obs = {"state": a.state, "checkout": a.checkout}
    if a.state in a.bench_states.split(","):
        bench = work / f"bench_{a.state}.json"
        obs["bench_rc"] = run([a.python, str(HERE / "attn_bench_child.py"), str(bench)], base_env,
                              logs / f"bench_{a.state}.log", 2400)
        if bench.is_file():
            obs["bench"] = json.loads(bench.read_text(encoding = "utf-8"))
    if a.state in a.mode_states.split(","):
        model_dir = work / "fake_qwen35"
        if not (model_dir / "config.json").is_file():
            obs["build_rc"] = run([a.python, str(HERE / "build_fake_child.py"), str(model_dir)], base_env,
                                  logs / "build_fake.log", 1800)
        obs["modes"] = {}
        for mode in a.modes.split(","):
            # The forced-varlen arm measures main with the kernel swapped in; other states have the fix itself.
            if mode == "packed_varlen" and a.state != "base":
                continue
            env = dict(base_env, UNSLOTH_COMPILE_LOCATION = str(work / f"cc_{a.state}_{mode}"))
            if mode != "padded":
                env["UNSLOTH_EXPERIMENTAL_HYBRID_PACKING"] = "1"
            out = work / f"mode_{a.state}_{mode}.json"
            rc = run([a.python, str(HERE / "train_mode_child.py"), str(model_dir), str(out), mode], env,
                     logs / f"mode_{a.state}_{mode}.log", 2400)
            rec = json.loads(out.read_text(encoding = "utf-8")) if out.is_file() else {}
            rec["rc"] = rc
            obs["modes"][mode] = rec
            a.out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
    a.out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

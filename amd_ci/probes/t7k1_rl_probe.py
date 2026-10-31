#!/usr/bin/env python3
"""Probe (T7k.1): DPO + GRPO step time / memory / loss on gfx1151, BASE checkout vs HEAD checkout.

All measurement happens in the 'head' state: it runs the base and head worktrees (states/base, states/head) in
alternating order for ROUNDS rounds, each arm a fresh process of the shipped jobs/{dpo,grpo}.py with that worktree
first on PYTHONPATH. The base / merge states only record their SHA. Observes only (criteria/t7k1_rl.py judges).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = HERE.parent / "jobs"
PLANS = {
    "dpo": ["dpo.py", "--backend", "unsloth", "--max-steps", "15"],
    "grpo": ["grpo.py", "--backend", "unsloth", "--max-steps", "8", "--max-completion-length", "128"],
}


def _sha(co: Path) -> str:
    return subprocess.run(["git", "-C", str(co), "rev-parse", "HEAD"], capture_output = True,
                          encoding = "utf-8", errors = "replace").stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--rounds", type = int, default = 3)
    ap.add_argument("--wall", type = int, default = 1500)
    args = ap.parse_args()
    co = Path(args.checkout)
    obs: dict = {"state": args.state, "checkout": str(co), "sha": _sha(co), "runs": []}
    if args.state != "head":
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0
    arms = {"base": co.parent / "base", "head": co}
    obs["arm_sha"] = {k: _sha(v) for k, v in arms.items()}
    work = args.out.parent / "t7k1_runs"
    work.mkdir(parents = True, exist_ok = True)
    for r in range(args.rounds):
        order = ("base", "head") if r % 2 == 0 else ("head", "base")
        for job, argv in PLANS.items():
            for arm in order:
                out = work / f"{job}_r{r}_{arm}.json"
                env = dict(os.environ)
                env["PYTHONPATH"] = os.pathsep.join([str(arms[arm])] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
                env.setdefault("PYTHONUTF8", "1")
                env.pop("UNSLOTH_LORA_FUSED_PASSES", None)
                cmd = [args.python, str(JOBS / argv[0]), *argv[1:], "--out", str(out)]
                try:
                    p = subprocess.run(cmd, cwd = str(arms[arm]), env = env, capture_output = True, encoding = "utf-8",
                                       errors = "replace", timeout = args.wall)
                    rc, tail = p.returncode, ((p.stdout or "") + (p.stderr or ""))[-2500:]
                except subprocess.TimeoutExpired:
                    rc, tail = -1, "TimeoutExpired"
                rec = json.loads(out.read_text(encoding = "utf-8")) if out.exists() else {}
                obs["runs"].append({"job": job, "round": r, "arm": arm, "rc": rc, "checkout": str(arms[arm]),
                                    "metrics": rec, "tail": tail if rc else ""})
                print(f"[t7k1] {job} r{r} {arm} rc={rc} steady={(rec.get('summary') or {}).get('steady_step_ms')}",
                      flush = True)
                args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

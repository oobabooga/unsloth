#!/usr/bin/env python3
"""Probe: tiny Unsloth LoRA SFT and DPO (jobs/sft.py, jobs/dpo.py, copied under probes/jobs/) against the state's
unsloth-zoo checkout. Observes only (criteria/zoo_train_parity.py judges).

The zoo checkout goes first on PYTHONPATH, so `import unsloth_zoo` is this state's tree; unsloth is whatever the venv
has (the workflow installs one pinned commit, identical for every state). Each job runs --repeats times, each in its
own process with a fresh compile cache, so the base repeats give the A/A spread. Per run it records the jobs/ metrics
JSON (per-step loss, time_ms, peak_mem_gb; summary) plus zoo_calls_runner.py's record of what really ran.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = HERE / "jobs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--jobs", default = "sft,dpo")
    ap.add_argument("--max-steps", type = int, default = 10)
    ap.add_argument("--repeats", type = int, default = 2)
    ap.add_argument("--model", default = None, help = "default: the job's --tiny model (tiny Qwen3)")
    ap.add_argument("--only-states", default = "base,head", help = "other states record a skip (time)")
    ap.add_argument("--timeout", type = int, default = 1800, help = "per run")
    args = ap.parse_args()

    co = Path(args.checkout)
    obs: dict = {"state": args.state, "checkout": str(co)}
    obs["zoo_sha"] = subprocess.run(["git", "-C", str(co), "rev-parse", "HEAD"], capture_output = True,
                                    text = True).stdout.strip()
    if args.state not in args.only_states.split(","):
        obs["skipped"] = f"state {args.state} not in --only-states {args.only_states}"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0
    work = args.out.parent / f"train_{args.state}"
    work.mkdir(parents = True, exist_ok = True)
    runs: dict = {}
    for job in [j for j in args.jobs.split(",") if j]:
        runs[job] = []
        for rep in range(args.repeats):
            d = work / f"{job}_r{rep}"
            d.mkdir(parents = True, exist_ok = True)
            metrics, calls = d / "metrics.json", d / "calls.json"
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([str(co), str(JOBS)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
            env["WORKSPACE"] = str(d)  # jobs' fresh compile caches land under $WORKSPACE/temp/jobs_compile
            env["ZOO_CALLS_OUT"] = str(calls)
            env.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")
            cmd = [args.python, str(HERE / "zoo_calls_runner.py"), str(JOBS / f"{job}.py"), "--tiny",
                   "--max-steps", str(args.max_steps), "--backend", "unsloth", "--out", str(metrics)]
            if args.model:
                cmd += ["--model", args.model]
            rec: dict = {"rep": rep, "cmd": " ".join(cmd)}
            try:
                p = subprocess.run(cmd, cwd = d, env = env, capture_output = True, text = True, timeout = args.timeout)
                rec["rc"] = p.returncode
                (d / "run.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding = "utf-8")
                rec["stderr_tail"] = p.stderr[-1500:]
            except subprocess.TimeoutExpired:
                rec["rc"] = -1
                rec["error"] = "TimeoutExpired"
            for key, path in (("metrics", metrics), ("calls", calls)):
                if path.is_file():
                    try:
                        rec[key] = json.loads(path.read_text(encoding = "utf-8"))
                    except Exception as e:  # noqa: BLE001
                        rec[f"{key}_parse_error"] = f"{type(e).__name__}: {e}"
            m = rec.get("metrics") or {}
            # Keep the observation small: steps + summary + checks + identity, not the config dump.
            rec["metrics"] = {k: m.get(k) for k in ("job", "model", "backend", "device", "passed", "error", "finished",
                                                     "steps", "summary", "checks", "revisions", "versions")}
            runs[job].append(rec)
    obs["runs"] = runs
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

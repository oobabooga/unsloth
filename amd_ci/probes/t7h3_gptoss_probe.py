#!/usr/bin/env python3
"""Probe (T7h.3): GPT-OSS flex vs eager inference, load_in_4bit generate + 3 SFT steps, and Triton facts, in one state's
zoo checkout. Observes only (criteria/t7h3_gptoss.py judges). Runs probes/t7h3_gptoss_scenario.py in a child process with
the checkout first on PYTHONPATH; states outside --only-states record a skip.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--wall", type = int, default = 3000)
    ap.add_argument("--only-states", default = "head")
    args = ap.parse_args()
    co = Path(args.checkout)
    obs: dict = {"state": args.state, "checkout": str(co)}
    obs["zoo_sha"] = subprocess.run(["git", "-C", str(co), "rev-parse", "HEAD"], capture_output = True,
                                    encoding = "utf-8", errors = "replace").stdout.strip()
    if args.state not in args.only_states.split(","):
        obs["skipped"] = f"state {args.state} not in --only-states {args.only_states} (head-only check)"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0
    res = args.out.with_name(args.out.stem + "_scenario.json")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(co)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("UNSLOTH_IS_PRESENT", "1")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("TORCH_LOGS", "recompiles,graph_breaks")
    env.setdefault("UNSLOTH_COMPILE_LOCATION", str(args.out.parent / f"compiled_cache_{args.state}"))
    try:
        p = subprocess.run([args.python, "-u", str(HERE / "t7h3_gptoss_scenario.py"), str(res)], cwd = str(args.out.parent),
                           env = env, capture_output = True, encoding = "utf-8", errors = "replace", timeout = args.wall)
        obs["rc"], out, errt = p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as exc:
        obs["rc"], errt = -1, "TimeoutExpired"
        out = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
    (args.out.with_name(args.out.stem + "_scenario.log")).write_text((out or "") + "\n--- stderr ---\n" + (errt or ""),
                                                                      encoding = "utf-8")
    obs["stderr_tail"] = (errt or "")[-3000:]
    if res.is_file():
        try:
            obs["scenario"] = json.loads(res.read_text(encoding = "utf-8"))
        except Exception as e:  # noqa: BLE001
            obs["scenario_parse_error"] = f"{type(e).__name__}: {e}"
    else:
        obs["scenario_missing"] = True
    obs["unsloth_zoo_file"] = (obs.get("scenario") or {}).get("unsloth_zoo_file")
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Probe for unsloth PR 15238 at one state (an unsloth checkout). Observes only.

1. pr15238_run.py: the per-expert MoE dW kernel with the heuristic 64x64x256 / 8 warps / 4 stages bf16
   tile, permute_x on and off, compared with the legacy kernel. RESULT lines are recorded per case.
2. pytest over the dW test files (files absent at a state are recorded as such).
The state's checkout is first on PYTHONPATH so its unsloth shadows the installed one.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pr13383_probe import run_pytest  # noqa: E402

TESTS = ["tests/test_moe_dw_rocm_lds_fallback.py", "tests/test_moe_dw_per_expert.py",
         "tests/test_moe_dw_accumulate.py"]


def _env(checkout: str, out_dir: Path, state: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = checkout + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop("UNSLOTH_MOE_DW_KERNEL", None)
    env["UNSLOTH_COMPILE_LOCATION"] = str(out_dir / f"compiled_{state}_run")
    return env


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
    run: dict = {"results": []}
    try:
        p = subprocess.run([args.python, str(HERE / "pr15238_run.py")], cwd = str(out_dir),
                           env = _env(args.checkout, out_dir, args.state), capture_output = True,
                           text = True, timeout = 900)
        run["rc"], out, err = p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        run["rc"], out, err = "timeout", "", "TimeoutExpired"
    (out_dir / f"run_{args.state}.log").write_text(out + "\n--- stderr ---\n" + err, encoding = "utf-8")
    for line in out.splitlines():
        if line.startswith("RESULT "):
            run["results"].append(json.loads(line[len("RESULT "):]))
        elif line.startswith("VERDICT "):
            run["verdict"] = line.split()[1]
        elif line.startswith("torch "):
            run["header"] = line
        elif line.startswith("legacy raised"):
            run.setdefault("legacy_errors", []).append(line)
    run["stderr_tail"] = err[-3000:]
    obs["run"] = run
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    obs["pytest"] = run_pytest(args.checkout, TESTS, out_dir, args.state, args.python, args.timeout)
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

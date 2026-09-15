#!/usr/bin/env python3
"""Probe: run the repository's shell installer suites (tests/sh/test_*.sh) in a
checkout, the way studio-backend-ci.yml does (`bash "$s"` from the repo root), and
record which suites failed BY NAME.

Observes only. Suites absent at a state (added by the PR) are listed, not run, so
the criteria can compare by set. Pairs with criteria/shell_suite_no_regression.py.
Writes JSON via --out, never stdout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 600, help = "per suite, seconds")
    ap.add_argument("--suites", nargs = "*", default = [],
                    help = "tests/sh/test_*.sh paths relative to the checkout; default: all")
    ap.add_argument("--shells", nargs = "*", default = ["bash"],
                    help = "interpreters to run each suite under (bash, dash, sh, powershell, pwsh)")
    args = ap.parse_args()

    root = Path(args.checkout)
    obs: dict = {"state": args.state, "checkout": args.checkout, "results": {}, "absent": []}
    wanted = args.suites or sorted(str(p.relative_to(root)) for p in (root / "tests" / "sh").glob("test_*.sh"))
    for rel in wanted:
        if rel.endswith("test_install_rollback_lifecycle.sh"):
            continue
        p = root / rel
        if not p.exists():
            obs["absent"].append(rel)
            continue
        for sh in args.shells:
            key = f"{rel}@{sh}"
            try:
                if sh in ("powershell", "pwsh"):
                    cmd = [sh, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", rel]
                else:
                    cmd = [sh, rel]
                r = subprocess.run(cmd, cwd = root, capture_output = True, text = True,
                                   timeout = args.timeout)
                obs["results"][key] = {"rc": r.returncode,
                                       "tail": ((r.stdout or "") + (r.stderr or ""))[-1500:]}
            except subprocess.TimeoutExpired:
                obs["results"][key] = {"rc": -1, "tail": "TimeoutExpired"}
            except FileNotFoundError as e:
                obs["results"][key] = {"rc": -2, "tail": f"interpreter missing: {e}"}
    obs["failed"] = sorted(k for k, v in obs["results"].items() if v["rc"] != 0)
    obs["n_run"] = len(obs["results"])
    obs["n_failed"] = len(obs["failed"])
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

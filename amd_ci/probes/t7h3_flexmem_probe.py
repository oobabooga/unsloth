#!/usr/bin/env python3
"""Probe (T7h.3 follow-up): run probes/t7h3_flexmem.py in three fresh processes (default / static / alternate) with the
state's zoo checkout first on PYTHONPATH. Observes only (criteria/t7h3_flexmem.py judges)."""

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
    ap.add_argument("--only-states", default = "head")
    args = ap.parse_args()
    co = Path(args.checkout)
    obs: dict = {"state": args.state, "checkout": str(co), "runs": {}}
    obs["zoo_sha"] = subprocess.run(["git", "-C", str(co), "rev-parse", "HEAD"], capture_output = True,
                                    encoding = "utf-8", errors = "replace").stdout.strip()
    if args.state not in args.only_states.split(","):
        obs["skipped"] = f"state {args.state} not in --only-states {args.only_states} (head-only check)"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(co)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("UNSLOTH_IS_PRESENT", "1")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    for mode in os.environ.get("T7H3_FLEXMEM_MODES", "default,static,alternate").split(","):
        res = args.out.with_name(f"{args.out.stem}_{mode}.json")
        e = dict(env, UNSLOTH_COMPILE_LOCATION = str(args.out.parent / f"cc_{args.state}_{mode}"))
        try:
            p = subprocess.run([args.python, "-u", str(HERE / "t7h3_flexmem.py"), str(res), mode], cwd = str(args.out.parent),
                               env = e, capture_output = True, encoding = "utf-8", errors = "replace", timeout = 1500)
            rc, out, err = p.returncode, p.stdout or "", p.stderr or ""
        except subprocess.TimeoutExpired:
            rc, out, err = -1, "", "TimeoutExpired"
        (args.out.with_name(f"{args.out.stem}_{mode}.log")).write_text(out + "\n--- stderr ---\n" + err, encoding = "utf-8")
        run = {"rc": rc, "stderr_tail": err[-1500:]}
        if res.is_file():
            run.update(json.loads(res.read_text(encoding = "utf-8")))
        obs["runs"][mode] = run
    obs["unsloth_zoo_file"] = (obs["runs"].get("default") or {}).get("unsloth_zoo_file")
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

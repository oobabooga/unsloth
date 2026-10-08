#!/usr/bin/env python3
"""Probe: after fast resizes of the Desktop window, does the WebView end at the
window's client size?

Observes only. Finds the debug Desktop exe prebuilt for this checkout's commit
(`$WINRESIZE_EXE_DIR/<sha>.exe`), runs winresize_core.py for N fresh launches,
and writes the per-launch records. criteria/winresize_settles.py judges.
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
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--launches", type = int, default = 3)
    args = ap.parse_args()

    obs: dict = {"state": args.state}
    sha = subprocess.run(["git", "-C", str(args.checkout), "rev-parse", "HEAD"],
                         capture_output = True, text = True).stdout.strip()
    obs["sha"] = sha
    exe = Path(os.environ.get("WINRESIZE_EXE_DIR", "")) / f"{sha}.exe"
    obs["exe"] = str(exe)
    if not sha or not exe.is_file():
        obs["error"] = f"no prebuilt exe for {sha or 'unknown commit'} at {exe}"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    work = args.out.parent / f"winresize_{args.state}"
    work.mkdir(parents = True, exist_ok = True)
    launches = []
    for i in range(1, args.launches + 1):
        rc = subprocess.run([sys.executable, str(HERE / "winresize_core.py"), "--exe", str(exe),
                             "--label", args.state, "--launch", str(i), "--out", str(work)]).returncode
        rec_path = work / f"{args.state}_l{i}.json"
        if rec_path.is_file():
            rec = json.loads(rec_path.read_text(encoding = "utf-8"))
        else:
            rec = {"error": f"core exited {rc} without a record"}
        rec["rc"] = rc
        launches.append(rec)
    obs["launches"] = launches
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

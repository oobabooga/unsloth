#!/usr/bin/env python3
"""Probe: run dwacc_run.py at one state, record DIAG lines. Observes only."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True); ap.add_argument("--checkout", required=True)
    ap.add_argument("--out", required=True, type=Path); ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args(); a.out = a.out.resolve()
    env = dict(os.environ); env["PYTHONPATH"] = a.checkout + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["UNSLOTH_COMPILE_LOCATION"] = str(a.out.parent / f"ucc_{a.state}")
    obs = {"state": a.state, "checkout": a.checkout, "diag": []}
    try:
        p = subprocess.run([a.python, str(HERE / "dwacc_run.py")], cwd=str(a.out.parent), env=env,
                           capture_output=True, text=True, timeout=1200)
        obs["rc"], out, err = p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        obs["rc"], out, err = "timeout", "", "TimeoutExpired"
    for line in out.splitlines():
        if line.startswith("DIAG "):
            obs["diag"].append(json.loads(line[5:]))
        elif line.startswith("HDR "):
            obs["header"] = json.loads(line[4:])
    obs["stderr_tail"] = err[-3000:]
    a.out.write_text(json.dumps(obs, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

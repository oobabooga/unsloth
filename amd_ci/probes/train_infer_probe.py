#!/usr/bin/env python3
"""Probe: Unsloth LoRA / QLoRA SFT and 16-bit / 4-bit inference at this checkout, plus REAP if present.

Observes only. Each job runs in its own process from jobs/ (shipped next to this file under
../jobs) with PYTHONPATH set to THIS checkout, so the state's unsloth package is the one imported.
Writes the jobs' metrics JSON plus a summary per job; the criteria module compares base and head.

Pairs with criteria/train_infer_regression.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = HERE.parent / "jobs"
RUNS = {
    "sft16": ("sft.py", ["--max-steps", "10"]),
    "sft4": ("sft.py", ["--max-steps", "10", "--load-in-4bit"]),
    "infer16": ("inference_smoke.py", []),
    "infer4": ("inference_smoke.py", ["--load-in-4bit"]),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    obs: dict = {"state": args.state, "jobs": {}}
    for name, (script, extra) in RUNS.items():
        out = args.out.with_name(f"{args.out.stem}_{name}.json")
        env = {
            **os.environ,
            "PYTHONPATH": f"{args.checkout}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
            "UNSLOTH_COMPILE_LOCATION": tempfile.mkdtemp(prefix = f"uc_{args.state}_{name}_"),
        }
        cmd = [sys.executable, str(JOBS / script), "--backend", "unsloth", "--compile-cache", "fresh", "--out", str(out), *extra]
        try:
            p = subprocess.run(cmd, env = env, capture_output = True, text = True, timeout = 2400, cwd = tempfile.mkdtemp())
            rc, tail = p.returncode, (p.stdout + p.stderr)[-2000:]
        except subprocess.TimeoutExpired:
            rc, tail = "timeout", ""
        entry: dict = {"rc": rc, "metrics": str(out)}
        try:
            d = json.loads(out.read_text(encoding = "utf-8"))
            steps = d.get("steps") or []
            times = sorted(s["time_ms"] for s in steps[2:] if s.get("time_ms"))
            entry.update(
                passed = d.get("passed"),
                error = (d.get("error") or "")[-400:] or None,
                loss = [s.get("loss") for s in steps],
                median_step_ms = times[len(times) // 2] if times else None,
                peak_mem_gb = max((s.get("peak_mem_gb") or 0) for s in steps) if steps else None,
                summary = {k: v for k, v in (d.get("summary") or {}).items() if isinstance(v, (int, float, str, bool)) or v is None},
                compile = d.get("compile"),
                checks_failed = [k for k, v in (d.get("checks") or {}).items() if not v.get("ok")],
                versions = d.get("versions"),
            )
        except Exception as e:  # noqa: BLE001
            entry["error"] = f"no metrics ({type(e).__name__}); tail: {tail}"
        obs["jobs"][name] = entry
    reap = Path(args.checkout) / "unsloth" / "models" / "reap.py"
    if reap.is_file():
        out = args.out.with_name(f"{args.out.stem}_reap.json")
        p = subprocess.run([sys.executable, str(JOBS / "reap_gpu_check.py"), "--tree", args.checkout, "--out", str(out)],
                           capture_output = True, text = True, timeout = 2400)
        try:
            obs["reap"] = json.loads(out.read_text(encoding = "utf-8"))
        except Exception:  # noqa: BLE001
            obs["reap"] = {"ok": False, "error": (p.stdout + p.stderr)[-2000:]}
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Runs bench.py steps one after another on this card (each a fresh process), all ROW lines into --out.

  python driver.py --steps acc,profile,abba:2,cold,sweep --out rows.jsonl [--shapes a,b] [--python PY]

abba:N = N rounds of base, head, head, base (one process per arm, all shapes each, shape order rotated per round);
cold = per package an empty TRITON_CACHE_DIR, one process (cold), then a second process on the same dir (warm)."""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(py, args, out, env = None):
    t0 = time.time()
    cmd = [py, "-u", str(HERE / "bench.py"), *args, "--out", out]
    print("STEP", " ".join(args), flush = True)
    r = subprocess.run(cmd, env = env, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True)
    for line in r.stdout.splitlines():
        if line.startswith("ROW ") or "Error" in line or "error" in line or "Traceback" in line or line.startswith("  File"):
            print(line[:4000], flush = True)
    print(f"STEP_DONE rc={r.returncode} {time.time() - t0:.0f}s", flush = True)
    if r.returncode:
        print(r.stdout[-6000:], flush = True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--shapes", default = "")
    ap.add_argument("--acc-shapes", default = "")
    ap.add_argument("--profile-shapes", default = "")
    ap.add_argument("--sweep-shapes", default = "")
    ap.add_argument("--cold-shapes", default = "s1k_h24,s1k_d64")
    ap.add_argument("--configs", default = str(HERE / "sweep.json"))
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--work", default = "/tmp/i8ts_work")
    a = ap.parse_args()
    Path(a.work).mkdir(parents = True, exist_ok = True)
    sh = ["--shapes", a.shapes] if a.shapes else []
    for step in a.steps.split(","):
        name, _, arg = step.partition(":")
        if name == "acc":
            run(a.python, ["--mode", "acc", *(["--shapes", a.acc_shapes] if a.acc_shapes else sh)], a.out)
        elif name == "profile":
            for pkg in ("i8_base", "i8_head"):
                run(a.python, ["--mode", "profile", "--pkg", pkg, *(["--shapes", a.profile_shapes] if a.profile_shapes else sh)], a.out)
        elif name == "abba":
            for rep in range(int(arg or 2)):
                for slot, pkg in (("A1", "i8_base"), ("B1", "i8_head"), ("B2", "i8_head"), ("A2", "i8_base")):
                    run(a.python, ["--mode", "time", "--pkg", pkg, "--rep", str(rep), "--slot", slot, *sh], a.out)
        elif name == "cold":
            for pkg in ("i8_base", "i8_head"):
                d = Path(a.work) / f"triton_cache_{pkg}"
                shutil.rmtree(d, ignore_errors = True)
                env = dict(os.environ, TRITON_CACHE_DIR = str(d))
                for phase in ("cold", "warm"):
                    run(a.python, ["--mode", "cold", "--pkg", pkg, "--slot", phase, "--shapes", a.cold_shapes], a.out, env)
        elif name == "sweep":
            run(a.python, ["--mode", "sweep", "--configs", a.configs,
                           *(["--shapes", a.sweep_shapes] if a.sweep_shapes else sh)], a.out)
    print("DRIVER_DONE", flush = True)


if __name__ == "__main__":
    main()

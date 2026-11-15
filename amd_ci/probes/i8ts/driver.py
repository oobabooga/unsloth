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
    ap.add_argument("--head-extra", default = "", help = "JSON merged into the head's Triton configs for acc / profile / abba")
    a = ap.parse_args()
    Path(a.work).mkdir(parents = True, exist_ok = True)
    sh = ["--shapes", a.shapes] if a.shapes else []
    hx = dict(os.environ, I8TS_HEAD_EXTRA = a.head_extra) if a.head_extra else None
    for step in a.steps.split(","):
        name, _, arg = step.partition(":")
        hx0 = {k: v for k, v in os.environ.items() if k != "I8TS_HEAD_EXTRA"}
        if name in ("acc", "accx"):
            run(a.python, ["--mode", "acc", "--arm", "i8_headx" if name == "accx" else "i8_head",
                           *(["--shapes", a.acc_shapes] if a.acc_shapes else sh)], a.out, hx if name == "accx" else hx0)
        elif name == "profile":
            arms = [("i8_base", "i8_base", hx0), ("i8_head", "i8_head", hx0)] + ([("i8_head", "i8_headx", hx)] if hx else [])
            for pkg, arm, env in arms:
                run(a.python, ["--mode", "profile", "--pkg", pkg, "--arm", arm,
                               *(["--shapes", a.profile_shapes] if a.profile_shapes else sh)], a.out, env)
        elif name == "abba":
            for rep in range(int(arg or 2)):
                for slot, pkg in (("A1", "i8_base"), ("B1", "i8_head"), ("B2", "i8_head"), ("A2", "i8_base")):
                    run(a.python, ["--mode", "time", "--pkg", pkg, "--rep", str(rep), "--slot", slot, *sh], a.out, hx)
        elif name == "abc":
            # three arms: base, head as committed, head with --head-extra
            for rep in range(int(arg or 2)):
                for slot, pkg, arm, env in (("A1", "i8_base", "i8_base", hx0), ("B1", "i8_head", "i8_head", hx0),
                                            ("C1", "i8_head", "i8_headx", hx), ("C2", "i8_head", "i8_headx", hx),
                                            ("B2", "i8_head", "i8_head", hx0), ("A2", "i8_base", "i8_base", hx0)):
                    run(a.python, ["--mode", "time", "--pkg", pkg, "--arm", arm, "--rep", str(rep), "--slot", slot, *sh],
                        a.out, env)
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

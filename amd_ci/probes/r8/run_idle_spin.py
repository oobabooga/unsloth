#!/usr/bin/env python3
"""C15-6: process CPU while idle after a GPU sync (ROCr AsyncEventsLoop spin, TheRock #9283). Observes only.

Initialises the device, runs a few GEMMs, synchronises, waits 1 s, then measures os.times() user + system over
--idle seconds of time.sleep (the main thread does nothing), plus per-thread utime / stime from /proc/self/task so a
spinning runtime thread is named. HSA_TOOLS_DISABLE_REGISTER is set (or not) by the caller's environment.

  run_idle_spin.py --out F [--idle 10]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def tasks() -> dict:
    out = {}
    base = Path("/proc/self/task")
    if not base.is_dir():
        return out
    tick = os.sysconf("SC_CLK_TCK")
    for t in base.iterdir():
        try:
            comm = (t / "comm").read_text(encoding = "utf-8").strip()
            f = (t / "stat").read_text(encoding = "utf-8").rsplit(")", 1)[1].split()
            out[t.name] = (comm, (int(f[11]) + int(f[12])) / tick)
        except Exception:  # noqa: BLE001
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--idle", type = float, default = 10.0)
    a = ap.parse_args()
    obs = {"HSA_TOOLS_DISABLE_REGISTER": os.environ.get("HSA_TOOLS_DISABLE_REGISTER"), "idle_s": a.idle}
    try:
        import torch
        obs["torch"], obs["hip"] = torch.__version__, torch.version.hip
        for p in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
            if os.path.isfile(p):
                obs["opt_rocm_version"] = open(p, encoding = "utf-8").read().strip()
                break
        x = torch.randn(2048, 2048, device = "cuda")
        for _ in range(5):
            x = (x @ x).tanh()
        torch.cuda.synchronize()
        time.sleep(1.0)
        t0, w0, k0 = os.times(), time.perf_counter(), tasks()
        time.sleep(a.idle)
        t1, w1, k1 = os.times(), time.perf_counter(), tasks()
        wall = w1 - w0
        cpu = (t1.user - t0.user) + (t1.system - t0.system)
        obs["wall_s"] = round(wall, 3)
        obs["cpu_s"] = round(cpu, 3)
        obs["core_frac"] = round(cpu / wall, 4)
        th = []
        for tid, (comm, s1) in k1.items():
            s0 = k0.get(tid, (comm, 0.0))[1]
            if s1 - s0 > 0:
                th.append({"tid": tid, "comm": comm, "core_frac": round((s1 - s0) / wall, 4)})
        obs["threads"] = sorted(th, key = lambda r: -r["core_frac"])[:8]
        obs["n_threads"] = len(k1)
        obs["ok"] = True
    except Exception as exc:  # noqa: BLE001
        obs["error"] = repr(exc)[:400]
    Path(a.out).write_text(json.dumps(obs), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

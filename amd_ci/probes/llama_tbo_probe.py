#!/usr/bin/env python3
"""Probe: test-backend-ops on one llama.cpp state (ROCm build). Observes only.

The SAME test-backend-ops binary (the head build's, passed via --tbo) runs for every
state, with LD_LIBRARY_PATH pointing at the state's own build. That keeps the test
list identical (the head commit adds the IQ1_M cases) while the ggml libraries, and
so the kernels under test, are the state's. The resolved libggml-hip path is
recorded so the criteria can check the swap really happened.

Writes JSON to --out, never stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

EVAL = [("MUL_MAT", "iq1_m"), ("MUL_MAT_ID", "iq1_m"), ("MUL_MAT", "iq1_s"), ("MUL_MAT_ID", "iq1_s")]
PERF = {
    # 64 experts, top-8, m 512, k 1024: cheap to set up on a 16-core host
    "mmid": ("MUL_MAT_ID", "type_a={t},type_b=f32,n_mats=64,n_used=8,b=0,m=512,n=(1|32|128|512),k=1024,"),
    "dense": ("MUL_MAT", "type_a={t},type_b=f32,m=4096,n=(16|128|512),k=4096,"),
}
PERF_TYPES = ["iq1_m", "iq1_s"]


def sh(cmd: list[str], env: dict, log: Path, timeout: int) -> tuple[int, str, float]:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, env = env, capture_output = True, text = True, timeout = timeout)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        out += "\n<<TIMEOUT>>"
        rc = -999
    log.write_text(" ".join(cmd) + "\n" + out, encoding = "utf-8")
    return rc, out, time.time() - t0


def parse_eval(out: str) -> dict:
    m = re.findall(r"(\d+)/(\d+) tests passed", out)
    passed, total = (int(m[-1][0]), int(m[-1][1])) if m else (0, 0)
    fails = [l.strip() for l in out.splitlines() if re.search(r"\bFAIL\b", l)]
    return {"passed": passed, "total": total, "fail_lines": fails[:20], "n_fail_lines": len(fails),
            "backend_ok": bool(re.search(r"Backend \w+0: .*OK", out))}


def parse_perf(out: str) -> dict:
    res = {}
    for line in out.splitlines():
        m = re.search(r"\((type_a=\w+,.*?)\):\s+\d+ runs -\s+([\d.]+) us/run", line)
        if m:
            vars_ = m.group(1)
            n = re.search(r",n=(\d+),", vars_)
            res[n.group(1) if n else vars_] = float(m.group(2))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--tbo", required = True, type = Path, help = "test-backend-ops binary (head build)")
    ap.add_argument("--build-dir", default = "build")
    ap.add_argument("--timeout", type = int, default = 3600)
    ap.add_argument("--backend", default = "ROCm0")
    args = ap.parse_args()

    libdir = (args.checkout / args.build_dir / "bin").resolve()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{libdir}:{env.get('LD_LIBRARY_PATH', '')}"
    logs = args.out.parent / f"logs_{args.state}"
    logs.mkdir(parents = True, exist_ok = True)

    obs: dict = {"state": args.state, "libdir": str(libdir), "tbo": str(args.tbo)}
    try:
        ldd = subprocess.run(["ldd", str(args.tbo)], env = env, capture_output = True, text = True).stdout
        hip = [l.split("=>")[1].split("(")[0].strip() for l in ldd.splitlines() if re.search(r"libggml-(hip|cuda|musa)\.so", l) and "=>" in l]
        base = [l.split("=>")[1].split("(")[0].strip() for l in ldd.splitlines() if "libggml-base" in l and "=>" in l]
        obs["hip_lib"] = hip[0] if hip else None
        obs["base_lib"] = base[0] if base else None
    except Exception as e:  # noqa: BLE001
        obs["ldd_error"] = f"{type(e).__name__}: {e}"

    obs["eval"] = {}
    for op, t in EVAL:
        rc, out, secs = sh([str(args.tbo), "-o", op, "-b", args.backend, "-p", f"type_a={t},"],
                           env, logs / f"eval_{op}_{t}.log", args.timeout)
        if "device_lines" not in obs:
            # the backend's init banner names the device and its arch, e.g. "Device 0: ..., gfx1151 (0x1151)"
            obs["device_lines"] = [l.strip() for l in out.splitlines() if re.search(r"Device \d+:", l)][:4]
            obs["gfx1151"] = any("gfx1151" in l for l in obs["device_lines"])
        r = parse_eval(out)
        r.update({"rc": rc, "secs": round(secs, 1)})
        obs["eval"][f"{op}:{t}"] = r

    obs["perf"] = {}
    for kind, (op, flt) in PERF.items():
        for t in PERF_TYPES:
            rc, out, secs = sh([str(args.tbo), "perf", "-o", op, "-b", args.backend, "-p", flt.format(t = t)],
                               env, logs / f"perf_{kind}_{t}.log", args.timeout)
            obs["perf"][f"{kind}:{t}"] = {"us_per_run": parse_perf(out), "rc": rc, "secs": round(secs, 1)}

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

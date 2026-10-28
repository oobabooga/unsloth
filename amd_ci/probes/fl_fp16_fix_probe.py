#!/usr/bin/env python3
"""Probe: fp16 fused linear losses on gfx1151, zoo main (base) vs main + the HIP fp16 bounded half-result GEMM fix
(head). Observes only (criteria/fl_fp16_fix.py judges). All arms run in the head invocation, each a fresh process with
PYTHONPATH = that state's checkout: perf fp16 base / head / head / base, bf16 base, fp16 plain-torch reference;
accuracy base / head at GradScaler-like scale 1024 and 65536; the fused-linear pytest files on the head tree."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--perf-shape", default = "4096,2048,151936")
    args = ap.parse_args()
    co = Path(args.checkout)
    obs: dict = {"state": args.state, "checkout": str(co)}
    if args.state != "head":
        obs["skipped"] = "all arms (base and head) run in the head invocation"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0
    cos = {"head": co, "base": co.parent / "base"}
    git = lambda c, *a: subprocess.run(["git", "-C", str(c), *a], capture_output = True, encoding = "utf-8").stdout.strip()  # noqa: E731
    obs["base_sha"], obs["head_status"] = git(cos["base"], "rev-parse", "HEAD"), git(co, "status", "--short")
    obs["head_has_fix"] = "_half_mm_bounded" in (co / "unsloth_zoo/fused_losses/fused_linear.py").read_text(encoding = "utf-8")
    obs["base_has_fix"] = "_half_mm_bounded" in (cos["base"] / "unsloth_zoo/fused_losses/fused_linear.py").read_text(encoding = "utf-8")
    work = args.out.parent / "fl_fix"
    work.mkdir(parents = True, exist_ok = True)
    env0 = dict(os.environ, UNSLOTH_DISABLE_AUTO_UPDATES = "1", PYTHONUTF8 = "1", UNSLOTH_IS_PRESENT = "1")
    env0.pop("CI", None)
    n, h, v = args.perf_shape.split(",")
    chk = str(HERE / "fl_check.py")
    perf = lambda arm, dt: [chk, "--mode", "perf", "--arm", arm, "--dtype", dt, "--n", n, "--h", h, "--vocabs", v]  # noqa: E731
    acc = lambda scale: [chk, "--mode", "acc", "--vocabs", "151936,32001", "--scale", str(scale)]  # noqa: E731
    arms = [("perf_fp16_base_1", "base", perf("fused", "fp16")), ("perf_fp16_head_1", "head", perf("fused", "fp16")),
            ("perf_fp16_head_2", "head", perf("fused", "fp16")), ("perf_fp16_base_2", "base", perf("fused", "fp16")),
            ("perf_bf16_base", "base", perf("fused", "bf16")), ("perf_fp16_ref", "base", perf("ref", "fp16")),
            ("acc_base_s1024", "base", acc(1024)), ("acc_head_s1024", "head", acc(1024)),
            ("acc_base_s65536", "base", acc(65536)), ("acc_head_s65536", "head", acc(65536)),
            ("pytest_head", "head", ["-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfE", "--tb=line",
                                     "tests/test_fused_linear_hip_fp16_gemm.py", "tests/test_fused_linear.py",
                                     "tests/test_fused_linear_jsd.py", "tests/test_fused_linear_routing.py",
                                     "tests/test_fused_linear_collapse.py", "tests/test_fused_linear_memory.py"])]
    pi = subprocess.run([args.python, "-m", "pip", "install", "-q", "pytest"], capture_output = True, encoding = "utf-8")
    obs["pytest_install_rc"] = pi.returncode
    obs["arms"] = {}
    for name, st, argv in arms:
        d = work / name
        d.mkdir(parents = True, exist_ok = True)
        env = dict(env0, PYTHONPATH = os.pathsep.join([str(cos[st])] + ([env0["PYTHONPATH"]] if env0.get("PYTHONPATH") else [])),
                   UNSLOTH_COMPILE_LOCATION = str(d / "cc"), TORCHINDUCTOR_CACHE_DIR = str(d / "ind"))
        is_pytest = argv[0] == "-m"
        cmd = [args.python, "-u"] + argv + ([] if is_pytest else ["--out", str(d / "result.json")])
        t = time.time()
        try:
            p = subprocess.run(cmd, cwd = cos[st] if is_pytest else d, env = env, capture_output = True,
                               encoding = "utf-8", errors = "replace", timeout = 3000)
            rc, text = p.returncode, (p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or "")
        except subprocess.TimeoutExpired as exc:
            rc, text = -1, "TimeoutExpired\n" + str(exc.stdout)[-3000:]
        (d / "log.txt").write_text(text, encoding = "utf-8")
        rec = {"state": st, "rc": rc, "s": round(time.time() - t, 1), "tail": text[-3000:]}
        if (d / "result.json").is_file():
            rec["result"] = json.loads((d / "result.json").read_text(encoding = "utf-8"))
        if is_pytest:
            rec["summary"] = [l for l in text.splitlines() if " passed" in l or " failed" in l or l.startswith(("FAILED", "ERROR"))][-40:]
        obs["arms"][name] = rec
        print(f"arm {name} ({st}) rc={rc} {rec['s']}s", flush = True)
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        for sub in ("cc", "ind"):
            subprocess.run(["rm", "-rf", "--", str(d / sub)])
    return 0


if __name__ == "__main__":
    sys.exit(main())

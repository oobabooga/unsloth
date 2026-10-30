#!/usr/bin/env python3
"""Probe: why the fused linear losses are 3-5.6x slower than plain torch in fp16 on gfx1151 (bf16 is faster than plain
torch). Observes only (criteria/fl_fp16_diag.py judges). All arms run in the head invocation on zoo main, each a fresh
process: perf fused fp16 default / fp16 with HIP half-result GEMMs (fl_check.py --patch halfout) / bf16 default, in
A B C B A order, plus fp16 plain-torch reference; accuracy fp16 default vs halfout at GradScaler-like scale 1024 and
65536 (overflow)."""

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
        obs["skipped"] = "all arms run in the head invocation"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0
    work = args.out.parent / "fl_diag"
    work.mkdir(parents = True, exist_ok = True)
    env0 = dict(os.environ, UNSLOTH_DISABLE_AUTO_UPDATES = "1", PYTHONUTF8 = "1", UNSLOTH_IS_PRESENT = "1")
    env0.pop("CI", None)
    n, h, v = args.perf_shape.split(",")
    chk = str(HERE / "fl_check.py")
    perf = lambda arm, dt, patch: [chk, "--mode", "perf", "--arm", arm, "--dtype", dt, "--n", n, "--h", h, "--vocabs", v] + (  # noqa: E731
        ["--patch", patch] if patch else [])
    acc = lambda patch, scale: [chk, "--mode", "acc", "--vocabs", "151936", "--scale", str(scale)] + (["--patch", patch] if patch else [])  # noqa: E731
    arms = [("perf_fp16_default_1", perf("fused", "fp16", "")), ("perf_fp16_halfout_1", perf("fused", "fp16", "halfout")),
            ("perf_bf16_default_1", perf("fused", "bf16", "")), ("perf_fp16_halfout_2", perf("fused", "fp16", "halfout")),
            ("perf_fp16_default_2", perf("fused", "fp16", "")), ("perf_fp16_ref", perf("ref", "fp16", "")),
            ("acc_default_s1024", acc("", 1024)), ("acc_halfout_s1024", acc("halfout", 1024)),
            ("acc_default_s65536", acc("", 65536)), ("acc_halfout_s65536", acc("halfout", 65536))]
    obs["arms"] = {}
    for name, argv in arms:
        d = work / name
        d.mkdir(parents = True, exist_ok = True)
        env = dict(env0, PYTHONPATH = os.pathsep.join([str(co)] + ([env0["PYTHONPATH"]] if env0.get("PYTHONPATH") else [])),
                   UNSLOTH_COMPILE_LOCATION = str(d / "cc"), TORCHINDUCTOR_CACHE_DIR = str(d / "ind"))
        t = time.time()
        try:
            p = subprocess.run([args.python, "-u"] + argv + ["--out", str(d / "result.json")], cwd = d, env = env,
                               capture_output = True, encoding = "utf-8", errors = "replace", timeout = 2400)
            rc, text = p.returncode, (p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or "")
        except subprocess.TimeoutExpired as exc:
            rc, text = -1, "TimeoutExpired\n" + str(exc.stdout)[-3000:]
        (d / "log.txt").write_text(text, encoding = "utf-8")
        rec = {"rc": rc, "s": round(time.time() - t, 1), "tail": text[-2000:]}
        if (d / "result.json").is_file():
            rec["result"] = json.loads((d / "result.json").read_text(encoding = "utf-8"))
        obs["arms"][name] = rec
        print(f"arm {name} rc={rc} {rec['s']}s", flush = True)
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        for sub in ("cc", "ind"):
            subprocess.run(["rm", "-rf", "--", str(d / sub)])
    return 0


if __name__ == "__main__":
    sys.exit(main())

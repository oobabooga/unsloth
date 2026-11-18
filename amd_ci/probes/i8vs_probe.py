#!/usr/bin/env python3
"""Probe: Studio INT8 attention vs SageAttention vs torch SDPA (flash / AOTriton, efficient, math) on this AMD GPU,
through probes/i8vs/bench_attn.py, the same batched benchmark run on the NVIDIA machines (observes only; judged by
criteria/i8vs_bench.py).

Head state only (head = the measured tree): kernel grid (dtype bf16/fp16, head_dim 64/128/256, heads 24/56, tokens
1k-32k, probe + sink inputs, interleaved timing, fp32 reference, every arm OK / NOT_RUN / ERROR), then optionally one
public ungated model end to end (--models). Base state: nothing measured (the run measures one tree, it is not a
before / after). Synthetic inputs only; no token is used."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--models", default = "")
    ap.add_argument("--kb-args", default = "")
    ap.add_argument("--kb-subset", action = "store_true", help = "bf16, head_dim 64/128, heads 24/56, 4k-16k tokens")
    ap.add_argument("--model-steps", type = int, default = 0)
    ap.add_argument("--model-arms", default = "")
    a = ap.parse_args()
    out = Path(a.out)
    if a.state != "head":
        out.write_text(json.dumps({"state": a.state, "skipped": "measured at head only"}), encoding = "utf-8")
        return 0
    # the work root (HF cache, model weights) sits OUTSIDE the uploaded out/ tree; only the results go in out/
    root = (out.resolve().parents[2] if len(out.resolve().parents) > 2 else out.parent) / "i8vs_root"
    (root / "ext").mkdir(parents = True, exist_ok = True)
    bout = out.parent / "bench_head"
    if a.models:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lpips"], check = False)
    cmd = [sys.executable, "-u", str(HERE / "i8vs" / "bench_attn.py"), "--root", str(root), "--out", str(bout),
           "--backend", str(Path(a.checkout) / "studio" / "backend"), "--python", sys.executable, "--mgpu", "off",
           "--prompts", "1"]
    kb = a.kb_args or ("--dtypes bfloat16 --head-dims 64,128 --heads 24,56 --tokens 4096,8192,16384" if a.kb_subset else "")
    if kb:
        cmd += [f"--kb-args={kb}"]
    if a.model_steps:
        cmd += ["--model-gen", json.dumps({"steps": a.model_steps})]
    if a.model_arms:
        cmd += ["--model-arms", a.model_arms]
    if a.models:
        cmd += ["--models", a.models]
    rc = subprocess.run(cmd).returncode
    res = {"state": "head", "bench_rc": rc, "bench_dir": str(bout)}
    j = bout / "bench_attn.json"
    if j.is_file():
        b = json.loads(j.read_text(encoding = "utf-8"))
        res["machine"] = b.get("machine")
        res["kernels"] = b.get("kernels")
        res["models"] = {"rc": (b.get("models") or {}).get("rc"),
                         "records": {t: {k: r.get(k) for k in ("verdict", "status", "renders", "profile", "error", "params")}
                                     for t, r in ((b.get("models") or {}).get("records") or {}).items()},
                         "scores": (b.get("models") or {}).get("scores")}
    out.write_text(json.dumps(res, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

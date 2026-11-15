#!/usr/bin/env python3
"""Probe: does a returning process re-benchmark the Triton reductions a first process already tuned?

Observes only. Runs the same child twice on one fresh inductor cache dir (a first start, then a returning start),
each a new interpreter with the tag Unsloth Zoo adds to torch.compiler.config.cache_key_tag, the checkout's Studio
backend on sys.path and, when the checkout has it, diffusion_aot_cache.pin_autotune_key() called (absent on base).
Each child compiles a block of reductions (GroupNorm + SiLU, LayerNorm, softmax, a row sum) through the stock
compile-worker pool and records how many times inductor benchmarked configs. Pairs with
criteria/autotune_key_reuse.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

CHILD = r"""
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
out = {"python": sys.version.split()[0]}
import torch
import torch.compiler.config as cc
cc.cache_key_tag = "unsloth/inductor-rand4x-off"
out["torch"] = torch.__version__
out["hip"] = getattr(torch.version, "hip", None)
out["gpu"] = torch.cuda.is_available()
if out["gpu"]:
    p = torch.cuda.get_device_properties(0)
    out["device"] = torch.cuda.get_device_name(0)
    out["arch"] = getattr(p, "gcnArchName", None)
import torch._inductor.config as ic
out["compile_threads"] = ic.compile_threads
try:
    from core.inference import diffusion_aot_cache as aot
    fn = getattr(aot, "pin_autotune_key", None)
    out["helper_present"] = fn is not None
    out["pinned"] = bool(fn()) if fn is not None else None
except Exception as e:
    out["helper_error"] = f"{type(e).__name__}: {e}"
from torch._inductor.runtime import triton_heuristics as th
counts = {"bench": 0}
orig = th.CachingAutotuner.benchmark_all_configs
def bench(self, *a, **k):
    counts["bench"] += 1
    return orig(self, *a, **k)
th.CachingAutotuner.benchmark_all_configs = bench

class Block(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gn = torch.nn.GroupNorm(32, c)
        self.ln = torch.nn.LayerNorm(c)
    def forward(self, x):
        h = torch.nn.functional.silu(self.gn(x))
        t = h.flatten(2).transpose(1, 2)
        t = self.ln(t)
        a = torch.softmax(t @ t.transpose(1, 2) / 8.0, dim = -1)
        return (a @ t).sum(dim = 1) + h.mean(dim = (2, 3))

t0 = time.time()
try:
    for c, s in ((320, 64), (640, 32), (1280, 16)):
        m = Block(c).cuda().to(torch.bfloat16)
        f = torch.compile(m, dynamic = False)
        x = torch.randn(2, c, s, s, device = "cuda", dtype = torch.bfloat16)
        with torch.no_grad():
            y = f(x)
        torch.cuda.synchronize()
    out["ok"] = True
except Exception as e:
    out["ok"] = False
    out["error"] = f"{type(e).__name__}: {e}"[:600]
out["seconds"] = round(time.time() - t0, 2)
out["bench"] = counts["bench"]
with open(sys.argv[2], "w") as fh:
    json.dump(out, fh)
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    args = ap.parse_args()

    obs: dict = {"state": args.state}
    backend = str(args.checkout / "studio" / "backend")
    root = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()) / f"autotune_key_{args.state}"
    env = dict(os.environ)
    env.pop("TORCH_COMPILE_CACHE_KEY_TAG", None)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(root / "inductor")
    env["TRITON_CACHE_DIR"] = str(root / "triton")
    root.mkdir(parents = True, exist_ok = True)
    for phase in ("first", "returning"):
        res = root / f"{phase}.json"
        p = subprocess.run([args.python, "-c", CHILD, backend, str(res)], env = env, capture_output = True,
                           text = True, timeout = 1800)
        rec: dict = {"rc": p.returncode}
        if res.exists():
            rec.update(json.loads(res.read_text()))
        else:
            rec["stderr_tail"] = p.stderr[-3000:]
        obs[phase] = rec
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

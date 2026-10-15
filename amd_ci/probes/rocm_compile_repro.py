#!/usr/bin/env python3
"""Observation only: reproduce the ROCm regional-compile failure Studio gated on, and check the head's probe.

Compiles one transformer-like block exactly as the default tier does (dynamic=True, fullgraph=True, no cudagraphs)
under inductor memory_planning True (the setting Unsloth's training config leaked, which hits
inductor::_alloc_from_pool) and False (what Studio pins since #13227), and records error / max rel err / time.
Then imports the head checkout's diffusion_rocm_compile and records its probe answer. JSON to --out.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

CHILD = r'''
import json, sys, time, traceback
import torch, torch.nn.functional as F
from torch import nn
mp = sys.argv[1] == "1"
class B(nn.Module):
    def __init__(s, d=1024, h=16):
        super().__init__(); s.h=h; s.n1=nn.LayerNorm(d, elementwise_affine=False); s.qkv=nn.Linear(d,3*d)
        s.o=nn.Linear(d,d); s.n2=nn.LayerNorm(d); s.m=nn.Sequential(nn.Linear(d,4*d), nn.GELU(approximate="tanh"), nn.Linear(4*d,d))
    def forward(s, x, sc):
        b,n,d=x.shape; y=s.n1(x)*(1+sc)
        q,k,v=s.qkv(y).view(b,n,3,s.h,d//s.h).permute(2,0,3,1,4)
        a=F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,n,d)
        x=x+s.o(a); return x+s.m(s.n2(x))
r={"memory_planning": mp, "torch": torch.__version__, "hip": torch.version.hip}
try:
    dev=torch.device("cuda"); blk=B().to(dev, torch.bfloat16).eval()
    torch._inductor.config.memory_planning = mp
    c=torch.compile(blk, dynamic=True, fullgraph=True)
    worst=0.0; t0=time.perf_counter(); times={}
    with torch.no_grad():
        for n in (1024, 4096, 4352):
            x=torch.randn(1,n,1024,device=dev,dtype=torch.bfloat16); sc=torch.randn(1,1,1024,device=dev,dtype=torch.bfloat16)*0.1
            ref=blk(x,sc).float(); got=c(x,sc).float()
            worst=max(worst, float((got-ref).abs().max()/ref.abs().max()))
            for name, fn in (("eager", blk), ("compiled", c)):
                for _ in range(3): fn(x,sc)
                torch.cuda.synchronize(); t=time.perf_counter()
                for _ in range(20): fn(x,sc)
                torch.cuda.synchronize(); times[f"{name}_{n}_ms"]=(time.perf_counter()-t)/20*1e3
    r.update(ok=True, rel_err=worst, first_compile_and_check_s=time.perf_counter()-t0, block_ms=times)
except Exception as e:
    r.update(ok=False, error=f"{type(e).__name__}: {e}"[:2000], alloc_from_pool="_alloc_from_pool" in (str(e)+traceback.format_exc()), tb=traceback.format_exc()[-3000:])
print("RESULT " + json.dumps(r))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    obs: dict = {"python": sys.executable, "runs": {}}
    for mp in ("1", "0"):
        t0 = time.time()
        p = subprocess.run([sys.executable, "-c", CHILD, mp], capture_output = True, text = True, timeout = 1800)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
        rec = json.loads(line[7:]) if line else {"ok": False, "error": "no result", "rc": p.returncode,
                                                  "tail": (p.stdout + p.stderr)[-3000:]}
        rec["wall_s"] = round(time.time() - t0, 1)
        obs["runs"][f"memory_planning={mp}"] = rec
    try:
        sys.path.insert(0, str(args.head / "studio" / "backend"))
        from core.inference import diffusion_rocm_compile as rc  # noqa: PLC0415

        t0 = time.time()
        obs["head_probe"] = list(rc.rocm_block_compile_probe())
        obs["head_probe_s"] = round(time.time() - t0, 1)
    except Exception:  # noqa: BLE001
        obs["head_probe_error"] = traceback.format_exc()[-2000:]
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

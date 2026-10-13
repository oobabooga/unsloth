#!/usr/bin/env python3
"""Probe: run unsloth's public FP8 block-quant forward, unmocked, at one state.

OBSERVES only (verdicts live in criteria/fp8_forward_runs.py). Runs in a child
process with the state's checkout first on PYTHONPATH, so an fp8e4nv Triton
compile failure (or a hard crash) is recorded rather than taking the probe down.
Result JSON goes to --out, never stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SNIPPET = r"""
import json, torch
res = {"torch": torch.__version__, "hip": torch.version.hip}
try:
    import triton
    res["triton"] = triton.__version__
except Exception as e:
    res["triton"] = f"unavailable: {e}"
res["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
from unsloth.kernels import fp8
res["unsloth_file"] = fp8.__file__
try:
    res["lacks_kernel"] = bool(fp8._fp8_device_lacks_kernel(torch.device("cuda", 0)))
except Exception as e:
    res["lacks_kernel"] = f"error: {type(e).__name__}: {e}"
torch.manual_seed(0)
bs, m, k = 128, 256, 512
w = (torch.randn(m, k, device = "cuda") * 0.3).to(torch.float8_e4m3fn)
s = torch.rand(m // bs, k // bs, device = "cuda", dtype = torch.float32) + 0.5
s.block_size = [bs, bs]
X = torch.randn(16, k, device = "cuda", dtype = torch.bfloat16) * 0.5
try:
    out = fp8.FP8BlockQuantLinear.apply(X, w, s)
    torch.cuda.synchronize()
    full = s.repeat_interleave(bs, 0).repeat_interleave(bs, 1)
    ref = X.float() @ (w.float() * full).t()
    res["finite"] = bool(torch.isfinite(out.float()).all())
    res["rel_err"] = float((out.float() - ref).norm() / ref.norm())
    res["ok"] = True
except Exception as e:
    res["ok"] = False
    res["error"] = f"{type(e).__name__}: {str(e)[:600]}"
print("PROBE_JSON " + json.dumps(res))
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 900)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "checkout": args.checkout}
    env = dict(os.environ)
    env["PYTHONPATH"] = args.checkout + os.pathsep + env.get("PYTHONPATH", "")
    env["UNSLOTH_COMPILE_LOCATION"] = tempfile.mkdtemp(prefix = f"ucc_{args.state}_")
    try:
        p = subprocess.run([args.python, "-c", SNIPPET], cwd = tempfile.gettempdir(), env = env,
                           capture_output = True, text = True, timeout = args.timeout)
        obs["rc"] = p.returncode
        line = next((l for l in p.stdout.splitlines() if l.startswith("PROBE_JSON ")), None)
        if line:
            obs.update(json.loads(line[len("PROBE_JSON "):]))
        else:
            obs["ok"] = False
            obs["error"] = "child produced no result (crash?)"
        obs["stderr_tail"] = p.stderr[-2000:]
    except subprocess.TimeoutExpired:
        obs["ok"] = False
        obs["error"] = f"timed out after {args.timeout}s"
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

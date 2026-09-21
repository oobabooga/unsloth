#!/usr/bin/env python3
"""Probe: does bitsandbytes dequantisation follow the LIVE torch stream?

unsloth#10563 / PR 10745: unsloth/kernels/utils.py builds CUDA_STREAMS once at import and
hands `CUDA_STREAMS[device_index]` to every bitsandbytes call, so work issued under a
non-default stream is dequantised against a stream nothing is synchronising with, and the
WEIGHT_BUFFERS / ABSMAX_BUFFERS scratch is shared across streams.

Observes only: structure (is the stream captured at import), capability (does this host have
usable 4-bit bitsandbytes kernels at all) and, when it does, a numeric comparison of the same
dequantisation on the default stream and on a side stream, repeated to give a race a chance.

Pairs with criteria/bnb_live_stream.py.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

_MEASURE = r'''
import json, os, sys
out = {}
checkout = sys.argv[1]
sys.path.insert(0, checkout)
try:
    import torch
    out["torch_version"] = torch.__version__
    out["hip"] = getattr(torch.version, "hip", None)
    out["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        out["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
except BaseException as e:
    out["torch_error"] = f"{type(e).__name__}: {e}"[:300]
    print(json.dumps(out)); raise SystemExit(0)
try:
    import bitsandbytes as bnb
    out["bnb_version"] = getattr(bnb, "__version__", None)
    out["bnb_file"] = getattr(bnb, "__file__", None)
except BaseException as e:
    out["bnb_error"] = f"{type(e).__name__}: {e}"[:300]
    print(json.dumps(out)); raise SystemExit(0)
try:
    import unsloth.kernels.utils as ku
    out["kernels_utils_file"] = ku.__file__
    out["has_CUDA_STREAMS"] = hasattr(ku, "CUDA_STREAMS")
    out["has_get_tensor_stream"] = hasattr(ku, "_get_tensor_stream")
except BaseException as e:
    out["unsloth_error"] = f"{type(e).__name__}: {e}"[:300]

# The measurement: quantise once, then dequantise on the default stream and on a side
# stream, and compare. A stream the caller is not synchronising against shows up as a
# mismatch or as NaN/garbage, not as an exception.
try:
    from bitsandbytes.functional import quantize_4bit, dequantize_4bit
    torch.manual_seed(0)
    w = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    q, state = quantize_4bit(w, quant_type="nf4")
    ref = dequantize_4bit(q, state).float()
    out["ref_norm"] = float(ref.norm())
    mismatches = 0
    max_abs = 0.0
    for _ in range(25):
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            # deliberately NO synchronize: that is exactly the omission under test
            other = dequantize_4bit(q, state).float()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        d = float((other - ref).abs().max())
        max_abs = max(max_abs, d)
        if d > 0.0 or not torch.isfinite(other).all():
            mismatches += 1
    out["side_stream_iters"] = 25
    out["side_stream_mismatches"] = mismatches
    out["side_stream_max_abs_diff"] = max_abs
except BaseException as e:
    out["measure_error"] = f"{type(e).__name__}: {e}"[:400]
print(json.dumps(out))
'''


def _structure(checkout: str) -> dict:
    """Where the stream comes from, read off the source. No GPU needed."""
    src = Path(checkout) / "unsloth" / "kernels" / "utils.py"
    info: dict = {"file": str(src), "exists": src.is_file()}
    if not src.is_file():
        return info
    text = src.read_text(encoding = "utf-8")
    info["builds_CUDA_STREAMS_at_import"] = any(
        isinstance(n, ast.Assign)
        and any(getattr(t, "id", "") == "CUDA_STREAMS" for t in n.targets)
        for n in ast.parse(text).body
    )
    info["n_CUDA_STREAM_uses"] = text.count("CUDA_STREAMS[")
    info["mentions_current_stream"] = "current_stream" in text
    info["mentions_get_tensor_stream"] = "_get_tensor_stream" in text
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "source": _structure(args.checkout)}
    proc = subprocess.run([sys.executable, "-c", _MEASURE, args.checkout],
                          capture_output = True, text = True, timeout = 2700)
    obs["_rc"] = proc.returncode
    obs["_stderr_tail"] = proc.stderr[-1200:]
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obs.update(json.loads(line))
                break
            except ValueError:
                continue
    else:
        obs["child_failed"] = True
        obs["_stdout_tail"] = proc.stdout[-1200:]
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

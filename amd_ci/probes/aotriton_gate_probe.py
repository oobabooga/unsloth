#!/usr/bin/env python3
"""Probe: does `import unsloth` from THIS checkout open PyTorch's ROCm AOTriton gate,
and what do the SDPA backends do afterwards?

Observes only. Runs a FRESH interpreter (the gate is read once by torch and then
latched, so the observer must not have touched SDPA before the import) with
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL scrubbed from the environment, imports the
checkout's `unsloth`, then records:

  gate_after_import           the variable's value once the import returned
  can_use_flash / efficient   torch's own eligibility answer for a fp16 SDPA shape
  flash_ran / efficient_ran   whether forcing each backend ran or raised
  *_max_abs_diff_vs_math      agreement of the unlocked kernel with the MATH backend
  default_peak_gib            peak memory of an unforced SDPA call
  default_matches_math_exactly  bit-identical output to MATH (the silent-fallback signature)
  stderr_gate_warnings        torch's own "Enable it with TORCH_ROCM_AOTRITON..." lines

Pairs with criteria/aotriton_gate_opens.py. Writes JSON via --out, never stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

KEY = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"

BODY = r'''
import json, os, sys, math
checkout, out_path = sys.argv[1], sys.argv[2]
KEY = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"
obs = {"gate_before_import": os.environ.get(KEY)}
sys.path.insert(0, checkout)
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
try:
    import unsloth  # noqa: F401
    obs["unsloth_file"] = getattr(unsloth, "__file__", None)
except BaseException as e:  # noqa: BLE001
    obs["unsloth_import_error"] = f"{type(e).__name__}: {e}"[:500]
finally:
    obs["gate_after_import"] = os.environ.get(KEY)
try:
    import torch
    import torch.nn.functional as F
    obs["torch"] = torch.__version__
    obs["hip"] = getattr(torch.version, "hip", None)
    obs["cuda_available"] = torch.cuda.is_available()
    if not torch.cuda.is_available():
        raise RuntimeError("no device")
    props = torch.cuda.get_device_properties(0)
    obs["arch"] = getattr(props, "gcnArchName", None) or props.name
    obs["device_name"] = props.name
    torch.manual_seed(0)
    dev = "cuda"
    q = torch.randn(2, 8, 512, 64, device = dev, dtype = torch.float16)
    k = torch.randn(2, 8, 512, 64, device = dev, dtype = torch.float16)
    v = torch.randn(2, 8, 512, 64, device = dev, dtype = torch.float16)
    from torch.nn.attention import sdpa_kernel, SDPBackend
    try:
        from torch.backends.cuda import SDPAParams, can_use_flash_attention, can_use_efficient_attention
        params = SDPAParams(q, k, v, None, 0.0, False, False)
        obs["can_use_flash"] = bool(can_use_flash_attention(params, False))
        obs["can_use_efficient"] = bool(can_use_efficient_attention(params, False))
    except Exception as e:  # noqa: BLE001
        obs["can_use_error"] = f"{type(e).__name__}: {e}"[:300]
    with sdpa_kernel([SDPBackend.MATH]):
        torch.cuda.reset_peak_memory_stats()
        ref = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        obs["math_peak_gib"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    for name, backend in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION)):
        try:
            with sdpa_kernel([backend]):
                o = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
            obs[f"{name}_ran"] = True
            obs[f"{name}_finite"] = bool(torch.isfinite(o).all())
            obs[f"{name}_max_abs_diff_vs_math"] = float((o.float() - ref.float()).abs().max())
        except Exception as e:  # noqa: BLE001
            obs[f"{name}_ran"] = False
            obs[f"{name}_error"] = f"{type(e).__name__}: {e}"[:300]
    torch.cuda.reset_peak_memory_stats()
    d = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    obs["default_peak_gib"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    obs["default_matches_math_exactly"] = bool(torch.equal(d, ref))
    obs["default_max_abs_diff_vs_math"] = float((d.float() - ref.float()).abs().max())
except Exception as e:  # noqa: BLE001
    obs["torch_error"] = f"{type(e).__name__}: {e}"[:500]
with open(out_path, "w", encoding = "utf-8") as f:
    json.dump(obs, f, indent = 2)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 900)
    args = ap.parse_args()

    env = dict(os.environ)
    env.pop(KEY, None)
    env.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
    obs: dict = {"state": args.state, "checkout": args.checkout, "python": args.python}
    with tempfile.TemporaryDirectory() as td:
        body = Path(td) / "body.py"
        body.write_text(BODY, encoding = "utf-8")
        inner = Path(td) / "inner.json"
        try:
            p = subprocess.run([args.python, str(body), args.checkout, str(inner)],
                               env = env, capture_output = True, text = True,
                               timeout = args.timeout)
            obs["rc"] = p.returncode
            err = p.stderr or ""
            obs["stderr_tail"] = err[-3000:]
            obs["stderr_gate_warnings"] = [l.strip() for l in err.splitlines() if KEY in l]
            obs["stdout_tail"] = (p.stdout or "")[-1000:]
        except subprocess.TimeoutExpired:
            obs["rc"] = -1
            obs["error"] = "TimeoutExpired"
        if inner.exists():
            obs.update(json.loads(inner.read_text(encoding = "utf-8")))
        else:
            obs["error"] = obs.get("error") or "inner probe wrote nothing"
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

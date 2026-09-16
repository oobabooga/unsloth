#!/usr/bin/env python3
"""Probe: which SDPA backends does ROCm expose for this checkout, and what does attention cost?

unsloth#8819 / #8225 / #9404: TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL is only exported by
install.sh's WSL drop-in, so on a plain ROCm host PyTorch leaves the fused SDPA kernels shut
and attention falls back to the quadratic MATH path. PR 8821 moves the gate to
`import unsloth` / studio's main.py.

Observes only. Three groups of readings per state:
  * what the state's own code does to the gate (`import unsloth` in a child process)
  * what torch then exposes, and what a long-context attention call costs
  * a CONTROL: the same measurement with the gate forced to "0" and to "1", independent of
    the state. Without it a base and head that differ only in an environment variable cannot
    be told apart from a flag that does nothing at all on this chip.

Pairs with criteria/rocm_sdpa_gate_open.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

GIB = 1024 ** 3
GATE = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"

# Run in a CHILD so each reading gets a fresh process: torch caches the gate in a
# function-local `static const bool` at the first ROCm SDPA capability probe, so a
# second reading in the same process would report the first one's answer.
_MEASURE = r'''
import json, os, sys, time
out = {"gate_env": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")}
checkout = sys.argv[1]
mode = sys.argv[2]
if mode == "import_unsloth":
    sys.path.insert(0, checkout)
    try:
        import unsloth  # noqa: F401
        out["unsloth_file"] = getattr(unsloth, "__file__", None)
        out["unsloth_imported"] = True
    except BaseException as e:
        out["unsloth_imported"] = False
        out["unsloth_error"] = f"{type(e).__name__}: {e}"[:400]
    out["gate_after_import"] = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")
try:
    import torch
    out["torch_version"] = torch.__version__
    out["torch_hip"] = getattr(torch.version, "hip", None)
    out["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out["arch"] = getattr(p, "gcnArchName", None)
        out["device_name"] = p.name
        out["total_gib"] = p.total_memory / (1024 ** 3)
    b = torch.backends.cuda
    out["flash_sdp_enabled"] = b.flash_sdp_enabled()
    out["mem_efficient_sdp_enabled"] = b.mem_efficient_sdp_enabled()
    out["math_sdp_enabled"] = b.math_sdp_enabled()
    try:
        out["cudnn_sdp_enabled"] = b.cudnn_sdp_enabled()
    except Exception:
        out["cudnn_sdp_enabled"] = None
except BaseException as e:
    out["torch_error"] = f"{type(e).__name__}: {e}"[:400]
    print(json.dumps(out)); raise SystemExit(0)

# Which backends will torch actually ACCEPT for a real long-context shape? Asking
# can_use_* is what torch itself asks; running under sdpa_kernel is what proves it.
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import torch.nn.functional as F
    B, H, S, D = 1, 16, 8192, 64
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    usable = {}
    for name, backend in (("flash", SDPBackend.FLASH_ATTENTION),
                          ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
                          ("math", SDPBackend.MATH)):
        torch.cuda.empty_cache() if dev == "cuda" else None
        try:
            q = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
            k = torch.randn_like(q); v = torch.randn_like(q)
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            with sdpa_kernel([backend]):
                o = F.scaled_dot_product_attention(q, k, v)
            if dev == "cuda":
                torch.cuda.synchronize()
            usable[name] = {
                "ok": True,
                "seconds": round(time.time() - t0, 3),
                "peak_gib": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3) if dev == "cuda" else None,
            }
            del q, k, v, o
        except BaseException as e:
            usable[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
        finally:
            if dev == "cuda":
                torch.cuda.empty_cache()
    out["sdpa_usable"] = usable
    out["available_backends"] = sorted(n for n, r in usable.items() if r.get("ok"))
except BaseException as e:
    out["sdpa_probe_error"] = f"{type(e).__name__}: {e}"[:400]

# Studio's own math-only detector -- the exact signal unsloth#9404's reporter saw
# logged as `diffusion.attention.math_only: ... kernels=math` on a Radeon 8060S.
try:
    sys.path.insert(0, os.path.join(checkout, "studio", "backend"))
    for stale in [m for m in list(sys.modules) if m.startswith(("core.", "utils."))]:
        del sys.modules[stale]
    from core.inference import diffusion_attention as da
    kernels = da._probe_sdpa_kernels("cuda" if torch.cuda.is_available() else "cpu", torch.bfloat16)
    out["studio_available_kernels"] = list(kernels)
    out["studio_sdpa_math_only"] = bool(
        kernels and da.SDPA_MATH in kernels
        and not any(k in kernels for k in da._SDPA_SUBQUADRATIC)
    )
except BaseException as e:
    out["studio_error"] = f"{type(e).__name__}: {e}"[:300]

print(json.dumps(out))
'''


def _child(checkout: str, mode: str, env_overrides: dict, log: list) -> dict:
    env = dict(os.environ)
    env.pop(GATE, None)
    env.update({k: v for k, v in env_overrides.items() if v is not None})
    proc = subprocess.run([sys.executable, "-c", _MEASURE, checkout, mode],
                          capture_output = True, text = True, timeout = 1800, env = env)
    log.append({"mode": mode, "env": env_overrides, "rc": proc.returncode,
                "stderr_tail": proc.stderr[-1500:]})
    # stdout carries import banners as well, so take the LAST JSON object on it.
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return {"child_failed": True, "rc": proc.returncode,
            "stdout_tail": proc.stdout[-1500:], "stderr_tail": proc.stderr[-1500:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    log: list = []
    obs: dict = {"state": args.state, "checkout": args.checkout,
                 "inherited_gate": os.environ.get(GATE)}

    # 1. what THIS state's code does, imported the way a user gets it
    obs["as_shipped"] = _child(args.checkout, "import_unsloth", {}, log)
    # 2. controls: is the flag load-bearing on this chip at all?
    obs["control_gate_0"] = _child(args.checkout, "torch_only", {GATE: "0"}, log)
    obs["control_gate_1"] = _child(args.checkout, "torch_only", {GATE: "1"}, log)
    obs["_child_log"] = log

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

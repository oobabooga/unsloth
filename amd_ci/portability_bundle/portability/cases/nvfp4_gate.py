"""Item 2b: NVFP4 only where supported, a clean refusal everywhere else.

  resolver        PR 10731's ``_resolve_backend`` with NVFP4 switched on: flashinfer only on sm_100/103/120 with a
                  passing preflight, otherwise torchao with a reason, and never an exception.
  forced_flashinfer  same with UNSLOTH_NVFP4_BACKEND=flashinfer: must warn and fall back, not raise.
  bias_eligible   whether the fused bias add would engage for a real NVFP4 output shape at the base dtype, and that
                  it is bit-identical to ``add_`` (the kernel's numerics are covered by h3vae_triton/nvfp4_bias_add).
  torchao_nvfp4   torchao's NVFP4 linear in a throwaway child: runs on Blackwell, must raise (not crash) elsewhere.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import types
import warnings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from portlib import FAILS, REFUSES, RUNS, SKIPPED, WRONG  # noqa: E402

INF = os.path.join("studio", "backend", "core", "inference")
CAPS = {(10, 0), (10, 3), (12, 0)}

_TORCHAO_PROBE = r"""
import json, sys, torch
res = {}
try:
    from torchao.quantization import quantize_
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig as C
    lin = torch.nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x.float(), lin.weight.float())
    try:
        cfg = C(use_triton_kernel=False)
    except TypeError:
        cfg = C()
    quantize_(lin, cfg)
    with torch.no_grad():
        y = lin(x)
    torch.cuda.synchronize()
    res = {"ok": True, "rel_rms": float(((y.float() - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())}
except Exception as exc:
    res = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
print("NVFP4_PROBE " + json.dumps(res))
"""


def _load_pkg(path: str):
    import importlib

    pkg = types.ModuleType("nvfp4snap")
    pkg.__path__ = [path]
    sys.modules["nvfp4snap"] = pkg
    return (importlib.import_module("nvfp4snap.diffusion_nvfp4_ops"),
            importlib.import_module("nvfp4snap.diffusion_nvfp4_bias"))


def run(ctx):
    import torch

    ws = os.environ.get("WORKSPACE") or os.path.abspath(os.path.join(HERE, "..", ".."))
    base = ctx.bundle_path("trees", "bias", INF)
    if not base:
        ctx.row("resolver", status = SKIPPED, note = "bundle has no PR 10731 sources")
        return ctx.rows
    os.environ["UNSLOTH_NVFP4_DIFFUSION"] = "1"
    os.environ.pop("UNSLOTH_NVFP4_BACKEND", None)
    try:
        ops, bias = _load_pkg(base)
    except Exception as exc:  # noqa: BLE001
        ctx.fail("import", exc)
        return ctx.rows
    supported = ctx.cap in CAPS and not ctx.is_rocm
    try:
        fi = ops._flashinfer_available()
    except Exception as exc:  # noqa: BLE001
        fi = (False, repr(exc))
    try:
        backend, reason = ops._resolve_backend(0)
        if backend == "flashinfer":
            st = RUNS if supported else WRONG
        else:
            st = REFUSES
        ctx.row("resolver", status = st, engaged = backend == "flashinfer",
                note = f"backend={backend}; {reason}; flashinfer={fi}; cap={ctx.cap} in NVFP4 set={supported}")
    except Exception as exc:  # noqa: BLE001
        ctx.fail("resolver", exc, note = "resolver RAISED instead of refusing")

    os.environ["UNSLOTH_NVFP4_BACKEND"] = "flashinfer"
    try:
        ops.reset_preflight_cache()
        with warnings.catch_warnings(record = True) as w:
            warnings.simplefilter("always")
            b = ops.select_nvfp4_backend(0)
        msg = "; ".join(str(x.message) for x in w)[:300]
        ctx.row("forced_flashinfer", status = RUNS if b == "flashinfer" else REFUSES, engaged = b == "flashinfer",
                note = f"backend={b}; warning={msg or 'none'}")
    except Exception as exc:  # noqa: BLE001
        ctx.fail("forced_flashinfer", exc, note = "forced request RAISED instead of falling back")
    finally:
        os.environ.pop("UNSLOTH_NVFP4_BACKEND", None)

    for dt in (torch.bfloat16, torch.float16):
        name = {torch.bfloat16: "bf16", torch.float16: "fp16"}[dt]
        try:
            out = torch.randn(4608, 3072, device = "cuda", dtype = dt)
            bb = torch.randn(3072, device = "cuda", dtype = dt)
            elig = bool(bias._eligible(out, bb))
            ref = out.clone().add_(bb)
            got = bias.fused_bias_add_(out.clone(), bb)
            torch.cuda.synchronize()
            same = bool(torch.equal(ref, got))
            ms = ctx.bench(lambda: bias.fused_bias_add_(out, bb))
            ms0 = ctx.bench(lambda: out.add_(bb))
            ctx.row("bias_add_4608x3072", dtype = name, status = RUNS if same else WRONG, engaged = elig,
                    err = {"max_abs": float((ref.float() - got.float()).abs().max())}, ms = ms, eager_ms = ms0,
                    note = f"bit-identical to add_={same}")
        except Exception as exc:  # noqa: BLE001
            ctx.fail("bias_add_4608x3072", exc, dtype = name)

    if not ctx.info.get("torchao"):
        ctx.row("torchao_nvfp4", status = SKIPPED, note = f"torchao unavailable: {ctx.info.get('torchao_error')}")
        return ctx.rows
    try:
        p = subprocess.run([sys.executable, "-c", _TORCHAO_PROBE], capture_output = True, text = True, timeout = 600,
                           encoding = "utf-8", errors = "replace")
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("NVFP4_PROBE ")), None)
        if line is None:
            ctx.row("torchao_nvfp4", status = FAILS, note = f"child died rc={p.returncode}: "
                    f"{(p.stderr or '').strip().splitlines()[-1:] }"[:300])
        else:
            res = json.loads(line[len("NVFP4_PROBE "):])
            if res.get("ok"):
                st = RUNS if (res.get("rel_rms", 1) < 0.2) else WRONG
                ctx.row("torchao_nvfp4", status = st, engaged = True, err = {"rel_rms": res.get("rel_rms")},
                        note = "256x256 bf16 linear")
            else:
                ctx.row("torchao_nvfp4", status = REFUSES if not supported else FAILS, engaged = False,
                        note = res.get("error", ""))
    except subprocess.TimeoutExpired:
        ctx.row("torchao_nvfp4", status = FAILS, note = "timeout")
    return ctx.rows

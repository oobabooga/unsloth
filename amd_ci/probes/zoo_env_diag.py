#!/usr/bin/env python3
"""Observe (outside pytest) what the fused linear engine will do on this host. Writes JSON to --out, never stdout.

torch / HIP / CUDA versions and device; triton importable; torch_directml present; unsloth_zoo fused_linear
_HAS_TRITON, _TRITON, _use_triton(cuda:0) and _use_triton(cpu); a tiny torch.compile on cpu and on the GPU with the
default (inductor) backend and with aot_eager; one fused_linear_cross_entropy forward + backward on each device with the
CALLS delta (torch vs triton). Run with the state's zoo checkout first on PYTHONPATH and UNSLOTH_IS_PRESENT=1.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:600]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    args = ap.parse_args()
    rec: dict = {"platform": platform.platform(), "python": sys.version.split()[0]}

    def save():
        with open(args.out, "w", encoding = "utf-8") as fh:
            json.dump(rec, fh, indent = 1)

    try:
        import torch
    except Exception as e:  # noqa: BLE001
        rec["torch_error"] = _err(e)
        save()
        return 0
    t = rec["torch"] = {"version": torch.__version__, "hip": getattr(torch.version, "hip", None),
                        "cuda": getattr(torch.version, "cuda", None), "is_available": bool(torch.cuda.is_available())}
    if t["is_available"]:
        try:
            t["device_count"] = torch.cuda.device_count()
            t["device_name"] = torch.cuda.get_device_name(0)
            t["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
        except Exception as e:  # noqa: BLE001
            t["device_error"] = _err(e)
    try:
        import triton

        rec["triton"] = {"version": triton.__version__, "file": triton.__file__}
    except Exception as e:  # noqa: BLE001
        rec["triton"] = {"version": None, "error": _err(e)}
    try:
        import torch_directml  # noqa: F401

        rec["torch_directml"] = True
    except Exception:  # noqa: BLE001
        rec["torch_directml"] = False
    save()

    devices = ["cpu"] + (["cuda"] if t["is_available"] else [])
    fl = None
    try:
        try:
            import unsloth_zoo  # noqa: F401
        except Exception as e:  # noqa: BLE001
            # No accelerator: zoo refuses to import unless UNSLOTH_ALLOW_CPU=1 (what the zoo conftest does). Record it.
            rec["zoo_import_error_without_allow_cpu"] = _err(e)
            os.environ["UNSLOTH_ALLOW_CPU"] = "1"
            for m in [m for m in sys.modules if m == "unsloth_zoo" or m.startswith("unsloth_zoo.")]:
                del sys.modules[m]
        import unsloth_zoo
        import unsloth_zoo.fused_losses.fused_linear as fl

        rec["unsloth_zoo_file"] = unsloth_zoo.__file__
        rec["fused_linear"] = {"_HAS_TRITON": getattr(fl, "_HAS_TRITON", None), "_TRITON": getattr(fl, "_TRITON", None),
                               "_IS_HIP": getattr(fl, "_IS_HIP", None),
                               "use_triton": {d: bool(fl._use_triton(torch.device(d, 0) if d == "cuda" else torch.device(d)))
                                              for d in devices}}
    except Exception as e:  # noqa: BLE001
        rec["fused_linear_error"] = _err(e)
        rec["fused_linear_tb"] = traceback.format_exc()[-2000:]
    save()

    comp = rec["compile"] = {}
    for d in devices:
        for backend in ("inductor", "aot_eager"):
            key = f"{d}/{backend}"
            t0 = time.perf_counter()
            try:
                torch._dynamo.reset()
                f = torch.compile(lambda x: (x * 2).sin().sum(), backend = backend)
                x = torch.randn(64, device = d)
                v = float(f(x))
                ok = abs(v - float((x * 2).sin().sum())) < 1e-3
                comp[key] = {"ok": ok, "s": round(time.perf_counter() - t0, 2)}
            except Exception as e:  # noqa: BLE001
                comp[key] = {"ok": False, "error": _err(e), "s": round(time.perf_counter() - t0, 2)}
            save()

    if fl is not None and hasattr(fl, "fused_linear_cross_entropy"):
        runs = rec["fused_ce_smoke"] = {}
        for d in devices:
            try:
                before = dict(fl.CALLS)
                torch.manual_seed(0)
                h = torch.randn(64, 128, device = d, requires_grad = True)
                w = torch.randn(1000, 128, device = d, requires_grad = True)
                y = torch.randint(0, 1000, (64,), device = d)
                loss = fl.fused_linear_cross_entropy(h, w, None, y, divisor = y.numel())
                loss.backward()
                ref = torch.nn.functional.cross_entropy((h.detach() @ w.detach().T).float(), y)
                runs[d] = {"loss": loss.item(), "ref": ref.item(), "abs_err": abs(loss.item() - ref.item()),
                           "calls_delta": {k: fl.CALLS[k] - before.get(k, 0) for k in fl.CALLS},
                           "triton_broken": [str(x) for x in fl._TRITON_BROKEN]}
            except Exception as e:  # noqa: BLE001
                runs[d] = {"error": _err(e), "tb": traceback.format_exc()[-1500:]}
            save()
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Supplementary cell (zoo PR 2858): which hipHostRegister refusals exist on this HIP runtime, whether each leaves a
sticky error for the next launch, and (tree mode) whether this tree's _Registered / _HostBuf consume it.

kinds: overlap  register [b+512K, b+1.5M) while [b, b+1M) of a 2 MiB buffer is registered
       badflags flags 0x7fffffff
       size0    nbytes 0
       dup      the exact range twice (the PR's own test shape)
mode raw:  the runtime call itself, then a launch.
mode tree: <target> (arena = _Registered, disk = _HostBuf) built from the test module of --checkout with
           torch.cuda.cudart redirected so its registration becomes that refusal; then the real cudart is restored
           and a kernel is launched. Observes only; JSON via --out.
"""

from __future__ import annotations

import argparse
import json
import runpy
import traceback
import types
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--kind", required = True, choices = ("overlap", "badflags", "size0", "dup"))
ap.add_argument("--mode", required = True, choices = ("raw", "tree"))
ap.add_argument("--target", default = "arena", choices = ("arena", "disk"))
ap.add_argument("--test-file", default = None)
ap.add_argument("--out", required = True, type = Path)
a = ap.parse_args()
rec: dict = {"kind": a.kind, "mode": a.mode, "target": a.target}


def save():
    a.out.write_text(json.dumps(rec, indent = 1, default = str), encoding = "utf-8")


g = runpy.run_path(a.test_file) if a.mode == "tree" else None
import torch  # noqa: E402

real = torch.cuda.cudart()
torch.ones(1, device = "cuda").sum().item()
MB = 1 << 20
hold = torch.empty(2 * MB, dtype = torch.uint8)
pre = None
if a.kind == "overlap":
    pre = int(real.cudaHostRegister(hold.data_ptr(), MB, 1))
    rec["pre_register_rc"] = pre
elif a.kind == "dup":
    pre = int(real.cudaHostRegister(hold.data_ptr(), MB, 1))
    rec["pre_register_rc"] = pre


def refused(ptr, n, flags):
    if a.kind == "overlap":
        return real.cudaHostRegister(hold.data_ptr() + MB // 2, MB, flags)
    if a.kind == "badflags":
        return real.cudaHostRegister(ptr, n, 0x7FFFFFFF)
    if a.kind == "size0":
        return real.cudaHostRegister(ptr, 0, flags)
    return real.cudaHostRegister(hold.data_ptr(), MB, flags)


def launch():
    try:
        (torch.ones(16, device = "cuda") * 2).sum().item()
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"


save()
if a.mode == "raw":
    buf = torch.empty(MB, dtype = torch.uint8)
    rc = int(refused(buf.data_ptr(), MB, 1))
    rec["refused_rc"] = rc
    rec["launch"] = launch()
    if rc == 0 and a.kind not in ("dup", "overlap"):
        real.cudaHostUnregister(buf.data_ptr())
else:
    mod = g["_mod"]
    disk = mod._disk()
    rec["block_swap_file"] = mod.__file__
    rec["has_consume"] = hasattr(disk, "consume_cuda_error")
    seen = []

    def reg(ptr, n, flags):
        rc = refused(ptr, n, flags)
        seen.append(int(rc))
        return rc
    redirect = types.SimpleNamespace(cudaHostRegister = reg, cudaHostUnregister = real.cudaHostUnregister)
    orig = torch.cuda.cudart
    torch.cuda.cudart = lambda: redirect
    obj = None
    try:
        if a.target == "arena":
            buf = torch.empty(MB, dtype = torch.uint8)
            try:
                obj = mod._Registered(buf, MB)
                rec["construct"] = "accepted"
            except RuntimeError as e:
                rec["construct"] = f"RuntimeError: {str(e)[:200]}"
        else:
            obj = disk._HostBuf(MB, pin = True)
            rec["construct"] = f"pinned={obj.pinned}"
    except Exception:  # noqa: BLE001
        rec["construct"] = "unexpected: " + traceback.format_exc()[-800:]
    finally:
        torch.cuda.cudart = orig
    rec["register_rcs"] = seen
    rec["launch"] = launch()
rec["done"] = True
save()

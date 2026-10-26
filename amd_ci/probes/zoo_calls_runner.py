#!/usr/bin/env python3
"""Run a jobs/*.py script unchanged and, at interpreter exit, record what it really used to $ZOO_CALLS_OUT (JSON):
fused_linear CALLS (the head's fused path engaged?), _TRITON_BROKEN, chunked_logprob loaded, the unsloth / unsloth_zoo
files imported, and torch's GPU identity. Usage: zoo_calls_runner.py <script.py> [script args...]
"""

from __future__ import annotations

import atexit
import json
import os
import runpy
import sys


def _dump():
    path = os.environ.get("ZOO_CALLS_OUT")
    if not path:
        return
    rec: dict = {}
    fl = sys.modules.get("unsloth_zoo.fused_losses.fused_linear")
    rec["fused_linear_loaded"] = fl is not None
    if fl is not None:
        rec["calls"] = dict(getattr(fl, "CALLS", {}) or {})
        rec["triton_broken"] = [str(d) for d in getattr(fl, "_TRITON_BROKEN", ())]
    rec["chunked_logprob_loaded"] = "unsloth_zoo.fused_losses.chunked_logprob" in sys.modules
    for name in ("unsloth", "unsloth_zoo"):
        rec[f"{name}_file"] = getattr(sys.modules.get(name), "__file__", None)
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            rec["torch"] = {"version": torch.__version__, "hip": getattr(torch.version, "hip", None),
                            "is_available": bool(torch.cuda.is_available())}
            if rec["torch"]["is_available"]:
                rec["torch"]["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
                rec["torch"]["device_name"] = torch.cuda.get_device_name(0)
        except Exception as e:  # noqa: BLE001
            rec["torch_error"] = f"{type(e).__name__}: {e}"
    with open(path, "w", encoding = "utf-8") as fh:
        json.dump(rec, fh, indent = 1)


atexit.register(_dump)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name = "__main__")

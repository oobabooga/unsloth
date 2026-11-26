#!/usr/bin/env python3
"""Cell (one process, one torch venv, one unsloth tree): run that tree's `_triton_probe(0)` and `triton_runs_on_device(0)`
on the real GPU without importing unsloth (whose import needs the full stack).

The three functions are taken verbatim (ast source segments) from <checkout>/unsloth/models/_utils.py and exec'd in
a namespace standing in for that module's globals: torch, a logger, an empty _TRITON_RUNS_ON_DEVICE,
__package__ = 'unsloth.models'. `unsloth` / `unsloth.models` are bare package shells whose __path__ is the tree's
directories, so the head's `from ._triton_probe_kernel import _add_one` loads the tree's own file and no
__init__.py runs. The real module has no module-level `tl` (checked: no star import, no `tl` binding), so the
base kernel sees the same globals it would in unsloth. Observes only; JSON via --out.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
import traceback
import types
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--checkout", required = True)
ap.add_argument("--out", required = True, type = Path)
a = ap.parse_args()
ck = Path(a.checkout).resolve()
rec: dict = {"checkout": str(ck)}


def save():
    a.out.write_text(json.dumps(rec, indent = 1, default = str), encoding = "utf-8")


import torch  # noqa: E402

p = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
rec["env"] = {"torch": torch.__version__, "hip": torch.version.hip, "cuda_available": torch.cuda.is_available(),
              "device": p.name if p else None, "arch": getattr(p, "gcnArchName", None) if p else None,
              "python": sys.version.split()[0]}
try:
    import triton
    rec["env"]["triton"] = triton.__version__
    rec["env"]["triton_file"] = triton.__file__
except Exception as e:  # noqa: BLE001
    rec["env"]["triton"] = "import failed: " + repr(e)[:300]
save()

src_path = ck / "unsloth" / "models" / "_utils.py"
src = src_path.read_text(encoding = "utf-8")
tree = ast.parse(src)
want = ("_is_hip_build", "_triton_probe", "triton_runs_on_device")
# Padded to the original line numbers: triton.jit reads the kernel's source back through inspect / linecache.
segs = {n.name: "\n" * (n.lineno - 1) + ast.get_source_segment(src, n) for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in want and not n.decorator_list}
rec["utils_file"] = str(src_path)
rec["functions_found"] = sorted(segs)
rec["module_level_tl"] = any(isinstance(n, (ast.Import, ast.ImportFrom)) and any((al.asname or al.name) == "tl" for al in n.names)
                             for n in tree.body)
rec["kernel_file_exists"] = (ck / "unsloth" / "models" / "_triton_probe_kernel.py").is_file()
for name, path in (("unsloth", ck / "unsloth"), ("unsloth.models", ck / "unsloth" / "models")):
    m = types.ModuleType(name)
    m.__path__ = [str(path)]
    sys.modules[name] = m
ns = {"__name__": "unsloth.models._utils", "__package__": "unsloth.models", "__file__": str(src_path),
      "__builtins__": __builtins__, "torch": torch, "logger": logging.getLogger("unsloth_probe"),
      "_TRITON_RUNS_ON_DEVICE": {}}
for name in want:
    if name in segs:
        exec(compile(segs[name], str(src_path), "exec"), ns)
save()


def attempt(fn):
    try:
        return {"ok": True, "value": fn()}
    except BaseException as e:  # noqa: BLE001
        return {"ok": False, "type": type(e).__name__, "msg": str(e)[-1500:], "tb": traceback.format_exc()[-3000:]}


rec["triton_probe"] = attempt(lambda: ns["_triton_probe"](0))
km = sys.modules.get("unsloth.models._triton_probe_kernel")
rec["kernel_module_file"] = getattr(km, "__file__", None)
save()
ns["_TRITON_RUNS_ON_DEVICE"].clear()
rec["triton_runs_on_device"] = attempt(lambda: ns["triton_runs_on_device"](0))
rec["done"] = True
save()

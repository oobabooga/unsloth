#!/usr/bin/env python3
"""Cell (one process, one torch venv, one unsloth tree, one --stage): the tree's ROCm Triton probe kernel on the real
GPU without importing unsloth (whose import needs the full stack). The caller runs every stage in its own process
(under -X faulthandler), so a crash in one stage (a segfault in the HIP runtime / Triton driver) does not hide the
others.

Stages:
  torch_launch    a torch allocation + kernel on cuda:0 (no Triton)
  triton_driver   triton.runtime.driver.active: current target (Triton's HIP driver load, no compile)
  compile_only    the tree's kernel object compiled for GPUTarget('hip', <arch>, 32) with triton.compile (no
                  driver, no launch): the base's function-local-`tl` NameError surfaces here
  probe           the tree's `_triton_probe(0)` as unsloth calls it (compile + launch)
  runs_on_device  the tree's `triton_runs_on_device(0)` (fresh cache)

The three functions are taken verbatim (ast source segments, padded to their real line numbers so triton.jit can
re-read the source) from <checkout>/unsloth/models/_utils.py and exec'd in a namespace standing in for that
module's globals: torch, a logger, an empty _TRITON_RUNS_ON_DEVICE, __package__ = 'unsloth.models'. `unsloth` /
`unsloth.models` are bare package shells whose __path__ is the tree's directories, so the head's
`from ._triton_probe_kernel import _add_one` loads the tree's own file and no __init__.py runs. The real module has
no module-level `tl` (recorded as module_level_tl), so the base kernel sees the globals it would in unsloth.
Observes only; JSON via --out (written before each risky step, so a crash leaves what was reached).
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
ap.add_argument("--stage", required = True,
                choices = ("torch_launch", "triton_driver", "compile_only", "probe", "runs_on_device"))
ap.add_argument("--out", required = True, type = Path)
a = ap.parse_args()
ck = Path(a.checkout).resolve()
rec: dict = {"checkout": str(ck), "stage": a.stage, "reached": []}


def save():
    a.out.write_text(json.dumps(rec, indent = 1, default = str), encoding = "utf-8")


def reach(tag):
    rec["reached"].append(tag)
    save()


def attempt(fn):
    try:
        return {"ok": True, "value": fn()}
    except BaseException as e:  # noqa: BLE001
        return {"ok": False, "type": type(e).__name__, "msg": str(e)[-1500:], "tb": traceback.format_exc()[-3000:]}


import torch  # noqa: E402

reach("torch_imported")
ok = torch.cuda.is_available()
p = torch.cuda.get_device_properties(0) if ok else None
rec["env"] = {"torch": torch.__version__, "hip": torch.version.hip, "cuda_available": ok,
              "device": p.name if p else None, "arch": getattr(p, "gcnArchName", None) if p else None,
              "python": sys.version.split()[0]}
try:
    import triton
    rec["env"]["triton"] = triton.__version__
    rec["env"]["triton_file"] = triton.__file__
except Exception as e:  # noqa: BLE001
    triton = None
    rec["env"]["triton"] = "import failed: " + repr(e)[:300]
reach("env")

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


class _Stop(Exception):
    pass


class _TorchStub:
    """compile_only: stands in for torch inside the exec'd probe so it stops right after defining / importing the
    kernel (its first torch call is torch.zeros)."""

    def __getattr__(self, name):
        raise _Stop(name)


ns = {"__name__": "unsloth.models._utils", "__package__": "unsloth.models", "__file__": str(src_path),
      "__builtins__": __builtins__, "torch": _TorchStub() if a.stage == "compile_only" else torch,
      "logger": logging.getLogger("unsloth_probe"), "_TRITON_RUNS_ON_DEVICE": {}}
for name in want:
    if name in segs:
        exec(compile(segs[name], str(src_path), "exec"), ns)
reach("functions_loaded")

if a.stage == "torch_launch":
    def _t():
        x = torch.zeros(16, device = "cuda:0")
        x += 1
        torch.cuda.synchronize(0)
        return bool((x == 1).all().item())
    rec["result"] = attempt(_t)
elif a.stage == "triton_driver":
    def _d():
        from triton.runtime import driver
        t = driver.active.get_current_target()
        return {"backend": t.backend, "arch": t.arch, "warp_size": t.warp_size}
    rec["result"] = attempt(_d)
elif a.stage == "compile_only":
    captured = []
    real_jit = triton.jit

    def jit(*args, **kw):
        out = real_jit(*args, **kw)
        captured.append(out)
        return out
    triton.jit = jit
    stop = attempt(lambda: ns["_triton_probe"](0))
    triton.jit = real_jit
    rec["define_step"] = {k: v for k, v in stop.items() if k != "tb" or stop.get("type") != "_Stop"}
    km = sys.modules.get("unsloth.models._triton_probe_kernel")
    rec["kernel_module_file"] = getattr(km, "__file__", None)
    rec["captured_kernels"] = [getattr(getattr(c, "fn", c), "__qualname__", repr(c)) for c in captured]
    if captured:
        def _c():
            from triton.backends.compiler import GPUTarget
            from triton.compiler import ASTSource
            arch = str(rec["env"]["arch"] or "gfx1151").split(":")[0]
            src_ = ASTSource(fn = captured[-1], signature = {"X": "*fp32", "N": "i32", "BLOCK": "constexpr"},
                             constexprs = {"BLOCK": 16})
            k = triton.compile(src_, target = GPUTarget("hip", arch, 32))
            return {"compiled": True, "asm_keys": sorted(getattr(k, "asm", {}) or {})}
        reach("compile_start")
        rec["result"] = attempt(_c)
    else:
        rec["result"] = {"ok": False, "type": "NoKernel", "msg": "no triton.jit kernel was defined / imported"}
elif a.stage == "probe":
    reach("probe_start")
    rec["result"] = attempt(lambda: ns["_triton_probe"](0))
    rec["kernel_module_file"] = getattr(sys.modules.get("unsloth.models._triton_probe_kernel"), "__file__", None)
elif a.stage == "runs_on_device":
    reach("probe_start")
    rec["result"] = attempt(lambda: ns["triton_runs_on_device"](0))
    rec["kernel_module_file"] = getattr(sys.modules.get("unsloth.models._triton_probe_kernel"), "__file__", None)
rec["done"] = True
save()

#!/usr/bin/env python3
"""Diagnostic: can real torchao import, quantize and export through Unsloth on a torch without
torch.distributed (AMD Windows ROCm), given the import-window shim in _torchao_nodist_patch.py?

Each case runs in a fresh interpreter with PYTHONPATH=<torchao target dir>; results -> --out JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
_MARK = "AMDCI_JSON="

_COMMON = r'''
import json, os, sys, tempfile, traceback
sys.path.insert(0, HERE)
r = {}
def err(e):
    return f"{type(e).__name__}: {e}"[:700]
import torch
r["torch"] = torch.__version__
r["dist_available"] = bool(torch.distributed.is_available())
def finite_logits(model_dir):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_dir, device_map = "cuda")
    with torch.no_grad():
        out = m(torch.tensor([[1, 2, 3, 4]], device = "cuda")).logits
    return bool(torch.isfinite(out).all()), str(type(next(iter(m.state_dict().values()))).__name__)
'''

_NO_PATCH = _COMMON + r'''
try:
    import torchao
    r["import"] = "ok " + torchao.__version__
except BaseException as e:
    r["import"] = err(e)
print("AMDCI_JSON=" + json.dumps(r))
'''

_PATCH_QUANT = _COMMON + r'''
from _torchao_nodist_patch import fix_torchao_without_torch_distributed
try:
    r["patched"] = fix_torchao_without_torch_distributed()
    import torchao
    r["import"] = "ok " + torchao.__version__
    r["dist_tensor_importable_after"] = True
    try:
        import torch.distributed.tensor  # noqa: F401
    except Exception as e:
        r["dist_tensor_importable_after"] = err(e)
except BaseException as e:
    r["import"] = err(e); r["trace"] = traceback.format_exc()[-2500:]
    print("AMDCI_JSON=" + json.dumps(r)); raise SystemExit
from torchao.quantization import Int8WeightOnlyConfig, Float8WeightOnlyConfig, quantize_
for name, cfg in (("int8", Int8WeightOnlyConfig()), ("fp8", Float8WeightOnlyConfig())):
    try:
        lin = torch.nn.Sequential(torch.nn.Linear(256, 256, dtype = torch.bfloat16, device = "cuda"))
        x = torch.randn(8, 256, dtype = torch.bfloat16, device = "cuda")
        ref = lin(x)
        quantize_(lin, cfg)
        out = lin(x)
        r[f"quantize_{name}"] = {"tensor": type(lin[0].weight).__name__, "max_abs_err": float((out - ref).abs().max())}
    except BaseException as e:
        r[f"quantize_{name}"] = err(e); r[f"quantize_{name}_trace"] = traceback.format_exc()[-2000:]
from transformers import AutoModelForCausalLM, TorchAoConfig
for name, cfg, safe in (("int8", Int8WeightOnlyConfig(), False), ("fp8", Float8WeightOnlyConfig(), True)):
    try:
        m = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM", dtype = torch.bfloat16,
                                                 device_map = "cuda", quantization_config = TorchAoConfig(quant_type = cfg))
        d = tempfile.mkdtemp()
        m.save_pretrained(d, safe_serialization = safe)
        ok, t = finite_logits(d)
        r[f"hf_save_reload_{name}"] = {"finite_logits": ok, "reloaded_weight_type": t}
    except BaseException as e:
        r[f"hf_save_reload_{name}"] = err(e); r[f"hf_{name}_trace"] = traceback.format_exc()[-2000:]
print("AMDCI_JSON=" + json.dumps(r))
'''

_PATCH_UNSLOTH = _COMMON + r'''
from _torchao_nodist_patch import fix_torchao_without_torch_distributed
r["patched"] = fix_torchao_without_torch_distributed()
try:
    import unsloth
    from unsloth import FastLanguageModel
    import torchao
    r["unsloth"] = unsloth.__version__
    r["torchao_file"] = str(getattr(torchao, "__file__", None))
    r["torchao_version"] = getattr(torchao, "__version__", None)
except BaseException as e:
    r["unsloth_import"] = err(e); r["trace"] = traceback.format_exc()[-2500:]
    print("AMDCI_JSON=" + json.dumps(r)); raise SystemExit
model, tok = FastLanguageModel.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM", max_seq_length = 128, load_in_4bit = False)
model = FastLanguageModel.get_peft_model(model, r = 8, target_modules = ["q_proj", "v_proj"], lora_alpha = 8)
for method in ("merged_16bit", "torchao_int8", "torchao_fp8"):
    base = os.path.join(tempfile.mkdtemp(), "m")
    try:
        model.save_pretrained_merged(base, tok, save_method = method)
        outs = [p for p in os.listdir(os.path.dirname(base)) if p.startswith("m-")] or ["m"]
        out = os.path.join(os.path.dirname(base), outs[0])
        ok, t = finite_logits(out)
        r[f"unsloth_{method}"] = {"output": outs[0], "files": sorted(os.listdir(out))[:12], "finite_logits": ok, "reloaded_weight_type": t}
    except BaseException as e:
        r[f"unsloth_{method}"] = err(e); r[f"unsloth_{method}_trace"] = "".join(l for l in traceback.format_exc().splitlines(True) if "site-packages" in l or "Error" in l)[-6000:]
print("AMDCI_JSON=" + json.dumps(r))
'''

_BASELINE_TORCH = _COMMON + r'''
for stmt in ("import torch._inductor.lowering", "import torch.distributed.rpc", "import torch.distributed.device_mesh"):
    try:
        exec(stmt); r[stmt] = "ok"
    except BaseException as e:
        r[stmt] = err(e)
print("AMDCI_JSON=" + json.dumps(r))
'''

_PATCH_SWEEP = _COMMON + r'''
import pkgutil, importlib
import _torchao_nodist_patch as P
r["patched"] = P.fix_torchao_without_torch_distributed()
import torchao
fails = {}
n = 0
for m in pkgutil.walk_packages(torchao.__path__, "torchao.", onerror = lambda x: None):
    name = m.name
    if any(t in name for t in (".test", "_models", "benchmarks", "examples", "mps")):
        continue
    n += 1
    try:
        importlib.import_module(name)
    except BaseException as e:
        fails[name] = err(e)[:200]
r["modules_tried"] = n
r["module_failures"] = fails
r["failure_causes"] = sorted(set(fails.values()))
for stmt in ("from transformers.processing_utils import Unpack", "import transformers.quantizers.quantizer_torchao", "import peft.tuners.lora.torchao"):
    try:
        exec(stmt); r[stmt] = "ok"
    except BaseException as e:
        r[stmt] = err(e)
r["stand_in_requests_refused_outside_torchao"] = sorted({f"{m} <- {imp}" for m, imp in P.STUB_IMPORTERS if not str(imp).startswith("torchao")})
poisoned = []
for mname, mod in list(sys.modules.items()):
    if mname == "torchao" or mname.startswith("torchao.") or mname.startswith("_torchao_nodist") or mod is None:
        continue
    try:
        items = list(vars(mod).items())
    except Exception:
        continue
    for k, v in items:
        try:
            hit = (isinstance(v, (P._NeverMeta, P._InertPacket))
                   or (isinstance(v, type(sys)) and v.__dict__.get("__unsloth_nodist_stub__", False))
                   or (callable(v) and getattr(v, "__qualname__", "") == "_unavailable.<locals>.fn"))
        except Exception:
            hit = False
        if hit:
            poisoned.append(f"{mname}.{k}")
r["stand_ins_held_outside_torchao"] = poisoned
r["stub_count"] = len(P.STUB_IMPORTERS)
try:
    import torch.distributed.tensor  # noqa: F401
    r["dist_tensor_outside_torchao"] = "importable (LEAK)"
except Exception as e:
    r["dist_tensor_outside_torchao"] = err(e)
print("AMDCI_JSON=" + json.dumps(r))
'''


def _run(py, code, tao_dir, timeout):
    with tempfile.NamedTemporaryFile("w", suffix = ".py", delete = False, encoding = "utf-8") as f:
        f.write(f"HERE = {str(HERE)!r}\n" + code)
        script = f.name
    env = dict(os.environ, PYTHONPATH = tao_dir, PYTHONIOENCODING = "utf-8", UNSLOTH_DISABLE_AUTO_UPDATES = "1")
    try:
        p = subprocess.run([py, script], capture_output = True, text = True, encoding = "utf-8",
                           errors = "replace", timeout = timeout, env = env, cwd = tempfile.gettempdir())
    except subprocess.TimeoutExpired:
        return {"error": f"timeout {timeout}s"}
    finally:
        os.unlink(script)
    lines = [l for l in (p.stdout or "").splitlines() if l.startswith(_MARK)]
    if not lines:
        return {"rc": p.returncode, "error": "no result", "stdout_tail": (p.stdout or "")[-2500:], "stderr_tail": (p.stderr or "")[-4000:]}
    res = json.loads(lines[-1][len(_MARK):])
    res["rc"] = p.returncode
    res["stderr_tail"] = (p.stderr or "")[-1500:]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--tao-dirs", nargs = "+", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--timeout", type = int, default = 1500)
    a = ap.parse_args()
    res = {}
    for d in a.tao_dirs:
        key = Path(d).name
        res[key] = {
            "baseline_torch": _run(a.python, _BASELINE_TORCH, d, a.timeout),
            "no_patch": _run(a.python, _NO_PATCH, d, a.timeout),
            "patch_sweep": _run(a.python, _PATCH_SWEEP, d, a.timeout),
            "patch_quant": _run(a.python, _PATCH_QUANT, d, a.timeout),
            "patch_unsloth": _run(a.python, _PATCH_UNSLOTH, d, a.timeout),
        }
    a.out.parent.mkdir(parents = True, exist_ok = True)
    a.out.write_text(json.dumps(res, indent = 2), encoding = "utf-8")
    for k, v in res.items():
        print("=====", k)
        for case, r in v.items():
            print(case, json.dumps({x: y for x, y in r.items() if "trace" not in x and x != "stderr_tail"})[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())

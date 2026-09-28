#!/usr/bin/env python3
"""Probe for unslothai/unsloth#7102 (real torchao on Windows ROCm). Observes only.

Per state, with that state's unsloth + Studio backend first on PYTHONPATH:
1. premise: torch facts, and whether stock `import torchao` works.
2. worker: what the Studio export worker does in that state (real-or-stub loader if the state has
   one, else the stub), then Unsloth, a tiny LoRA and the real torchao_int8 / torchao_fp8 exports
   through ExportBackend.export_merged_model, plus the gate and export_capability().
3. verify: each successful export reloaded in a process that never imports Unsloth.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_MARK = "AMDCI_JSON="

_PREMISE = r'''
import json, sys
r = {"platform": sys.platform}
import torch
r.update(torch = torch.__version__, hip = getattr(torch.version, "hip", None),
         dist_available = bool(torch.distributed.is_available()), cuda_available = bool(torch.cuda.is_available()))
if r["cuda_available"]:
    r["device"] = torch.cuda.get_device_name(0)
    r["gcn_arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
try:
    import torchao
    r["stock_torchao"] = "imported " + torchao.__version__
except BaseException as e:
    r["stock_torchao"] = f"{type(e).__name__}: {e}"[:400]
print("AMDCI_JSON=" + json.dumps(r))
'''

_WORKER = r'''
import json, os, sys, tempfile, traceback
r = {}
def err(e):
    return f"{type(e).__name__}: {e}"[:700]
# What Studio's spawn child does before the worker target: re-run run.py as __mp_main__.
import runpy
sys.argv = ["run.py"]
try:
    runpy.run_path("run.py", run_name = "__mp_main__")
    r["mp_main"] = "ran"
except BaseException as e:
    r["mp_main"] = err(e)
import core._torchao_stub as S
r["stubbed_before_loader"] = S.is_stubbed("torchao")
loader = getattr(S, "install_torchao_windows_rocm_real_or_stub", None)
r["worker_loader"] = "real_or_stub" if loader else "stub"
(loader or S.install_torchao_windows_rocm_stub)()
import torchao
r["torchao_is_stub"] = S.is_stubbed("torchao")
r["torchao_version"] = getattr(torchao, "__version__", None)
try:
    import unsloth
    from unsloth import FastLanguageModel
    r["unsloth_file"] = unsloth.__file__
except BaseException as e:
    r["unsloth_import_error"] = err(e); r["trace"] = traceback.format_exc()[-2500:]
    print("AMDCI_JSON=" + json.dumps(r)); raise SystemExit
import core.export.export as ex
r["gate_torchao_export_supported"] = bool(ex._torchao_export_supported())
try:
    from utils.hardware import hardware as hw
    hw.detect_hardware()
    cap = hw.export_capability()
    r["export_capability"] = {k: cap.get(k) for k in ("export_supported", "torchao_export_supported", "win32_rocm")}
except BaseException as e:
    r["capability_error"] = err(e)
try:
    model, tok = FastLanguageModel.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM", max_seq_length = 128, load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, r = 8, target_modules = ["q_proj", "v_proj"], lora_alpha = 8)
except BaseException as e:
    r["model_error"] = err(e); r["trace"] = traceback.format_exc()[-2500:]
    print("AMDCI_JSON=" + json.dumps(r)); raise SystemExit
r["exports"] = {}
for alias in ("torchao_int8", "torchao_fp8"):
    be = ex.ExportBackend.__new__(ex.ExportBackend)
    be.current_model, be.current_tokenizer, be._audio_type, be.is_peft = model, tok, None, True
    be.current_checkpoint = None
    try:
        ok, message, out = be.export_merged_model(f"amdci_{alias}", compressed_method = alias)
        r["exports"][alias] = {"ok": bool(ok), "message": str(message)[:600], "output": out}
    except BaseException as e:
        r["exports"][alias] = {"ok": False, "message": err(e), "trace": traceback.format_exc()[-2000:]}
print("AMDCI_JSON=" + json.dumps(r))
'''

_CORE = r'''
import json, sys
r = {}
try:
    import unsloth
    r["unsloth_file"] = unsloth.__file__
except BaseException as e:
    r["unsloth_import_error"] = f"{type(e).__name__}: {e}"[:600]
t = sys.modules.get("torchao")
r["torchao_loaded"] = t is not None
r["torchao_file"] = getattr(t, "__file__", None)
r["torchao_version"] = getattr(t, "__version__", None)
try:
    from torchao.quantization import Int8WeightOnlyConfig, Float8WeightOnlyConfig
    r["configs"] = [Int8WeightOnlyConfig.__module__, Float8WeightOnlyConfig.__module__]
except BaseException as e:
    r["configs"] = f"{type(e).__name__}: {e}"[:400]
print("AMDCI_JSON=" + json.dumps(r))
'''

_VERIFY = r'''
import importlib.util, json, os, sys, traceback
r = {}
shim = os.path.join(sys.argv[1], "unsloth", "_torchao_nodist.py")
spec = importlib.util.spec_from_file_location("_nodist", shim)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
r["patched"] = m.fix_torchao_without_torch_distributed()
import torch
from transformers import AutoModelForCausalLM
ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device = "cuda")
for alias, out in json.loads(os.environ["AMDCI_EXPORTS"]).items():
    try:
        model = AutoModelForCausalLM.from_pretrained(out, device_map = "cuda")
        with torch.no_grad():
            logits = model(ids).logits.float()
        r[alias] = {"finite": bool(torch.isfinite(logits).all()),
                    "param_types": sorted({type(p.data).__name__ for p in model.parameters()}),
                    "files": sorted(os.listdir(out))}
    except BaseException as e:
        r[alias] = f"{type(e).__name__}: {e}"[:600]
print("AMDCI_JSON=" + json.dumps(r))
'''


def _run(py, code, cwd, pythonpath, timeout, args = (), extra_env = None):
    with tempfile.NamedTemporaryFile("w", suffix = ".py", delete = False, encoding = "utf-8") as f:
        f.write(code)
        script = f.name
    home = os.path.join(tempfile.gettempdir(), "amdci_studio_home")
    os.makedirs(home, exist_ok = True)
    env = dict(os.environ, PYTHONPATH = os.pathsep.join(pythonpath), PYTHONIOENCODING = "utf-8",
               UNSLOTH_DISABLE_AUTO_UPDATES = "1", UNSLOTH_STUDIO_HOME = home, **(extra_env or {}))
    try:
        p = subprocess.run([py, script, *args], cwd = cwd, capture_output = True, text = True,
                           encoding = "utf-8", errors = "replace", timeout = timeout, env = env)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout {timeout}s"}
    finally:
        os.unlink(script)
    lines = [l for l in (p.stdout or "").splitlines() if l.startswith(_MARK)]
    if not lines:
        return {"rc": p.returncode, "error": "no result", "stderr_tail": (p.stderr or "")[-3000:]}
    res = json.loads(lines[-1][len(_MARK):])
    res["rc"] = p.returncode
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 1500)
    a = ap.parse_args()
    root = Path(a.checkout).resolve()
    backend = root / "studio" / "backend"
    obs: dict = {"state": a.state}
    path = [str(root), str(backend)]
    obs["premise"] = _run(a.python, _PREMISE, tempfile.gettempdir(), [], a.timeout)
    obs["core"] = _run(a.python, _CORE, tempfile.gettempdir(), [str(root)], a.timeout)
    obs["worker"] = _run(a.python, _WORKER, str(backend), path, a.timeout)
    exports = {k: v["output"] for k, v in (obs["worker"].get("exports") or {}).items() if v.get("ok") and v.get("output")}
    if exports and (root / "unsloth" / "_torchao_nodist.py").is_file():
        obs["verify"] = _run(a.python, _VERIFY, tempfile.gettempdir(), [], a.timeout, [str(root)],
                             {"AMDCI_EXPORTS": json.dumps(exports)})
    a.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

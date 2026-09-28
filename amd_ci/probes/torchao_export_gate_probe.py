#!/usr/bin/env python3
"""Probe for unslothai/unsloth#7102: the Studio torchao export gate on Windows ROCm.

Observes only. Two fresh interpreters per state, both with cwd = <checkout>/studio/backend:

1. premise: torch build facts and whether REAL torchao imports (from an optional
   --torchao-dir target, never installed into the venv, since Studio skips torchao on
   Windows ROCm and the stub path is what users actually hit).
2. worker: what a Studio export worker does: install the Windows-ROCm torchao stub, import
   unsloth, then record the stubbed config value, what transformers says about
   TorchAoConfig(quant_type=<that>), the export gate, a forced torchao export request, and
   the export_capability() payload.
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
tao_dir = sys.argv[1]
r = {"platform": sys.platform}
try:
    import torch
    r.update(torch = torch.__version__, hip = getattr(torch.version, "hip", None),
             dist_available = bool(torch.distributed.is_available()),
             has_c10d_ext = hasattr(torch._C, "_distributed_c10d"),
             cuda_available = bool(torch.cuda.is_available()))
    if r["cuda_available"]:
        r["device"] = torch.cuda.get_device_name(0)
        r["gcn_arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
except BaseException as e:
    r["torch_error"] = f"{type(e).__name__}: {e}"[:600]
if tao_dir:
    sys.path.insert(0, tao_dir)
    try:
        import torchao
        r["real_torchao"] = "imported " + getattr(torchao, "__version__", "?")
    except BaseException as e:
        r["real_torchao"] = f"{type(e).__name__}: {e}"[:600]
print("AMDCI_JSON=" + json.dumps(r))
'''

_WORKER = r'''
import json, os, sys, tempfile, traceback
backend = os.getcwd()
sys.path.insert(0, backend)
r = {}
def err(e):
    return f"{type(e).__name__}: {e}"[:600]
try:
    from core._torchao_stub import install_torchao_windows_rocm_stub
    install_torchao_windows_rocm_stub()
    import torchao
    r["torchao_is_stub"] = getattr(torchao, "_unsloth_stub", None) is not None
    from torchao.quantization import Int8WeightOnlyConfig, Float8WeightOnlyConfig
    int8, fp8 = Int8WeightOnlyConfig(), Float8WeightOnlyConfig()
    r["int8_config"], r["fp8_config"] = repr(int8), repr(fp8)
except BaseException as e:
    r["stub_error"] = err(e)
    int8 = None
try:
    import unsloth.save as us
    r["unsloth_version"] = getattr(sys.modules.get("unsloth"), "__version__", None)
    r["unsloth_has_torchao_method"] = hasattr(us, "_normalize_torchao_method")
except BaseException as e:
    r["unsloth_import_error"] = err(e)
    r["unsloth_import_trace"] = traceback.format_exc()[-1500:]
try:
    from transformers import TorchAoConfig
    try:
        TorchAoConfig(quant_type = int8)
        r["torchaoconfig_error"] = None
    except BaseException as e:
        r["torchaoconfig_error"] = err(e)
except BaseException as e:
    r["transformers_error"] = err(e)
try:
    import core.export.export as ex
    r["gate_torchao_export_supported"] = bool(ex._torchao_export_supported())
    be = ex.ExportBackend.__new__(ex.ExportBackend)
    be.current_model = object()
    be.current_tokenizer = object()
    be._audio_type = None
    be.is_peft = True
    out = be.export_merged_model(os.path.join(tempfile.mkdtemp(), "x"), compressed_method = "torchao_int8")
    r["forced_torchao_export"] = {"ok": bool(out[0]), "message": str(out[1])[:600]}
except BaseException as e:
    r["export_error"] = err(e)
    r["export_trace"] = traceback.format_exc()[-1500:]
# End to end, as the Studio export worker runs it: a real (tiny) LoRA model through the real export call.
try:
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        "trl-internal-testing/tiny-Qwen3ForCausalLM", max_seq_length = 128,
        load_in_4bit = False, dtype = None)
    model = FastLanguageModel.get_peft_model(model, r = 8, target_modules = ["q_proj", "v_proj"], lora_alpha = 8)
    r["e2e_model_device"] = str(next(model.parameters()).device)
    be2 = ex.ExportBackend.__new__(ex.ExportBackend)
    be2.current_model, be2.current_tokenizer, be2._audio_type, be2.is_peft = model, tok, None, True
    be2.current_checkpoint = None
    out = be2.export_merged_model("amdci_torchao_e2e", compressed_method = "torchao_int8")
    r["e2e_torchao_export"] = {"ok": bool(out[0]), "message": str(out[1])[:800], "output": str(out[2])[:300]}
except BaseException as e:
    r["e2e_error"] = err(e)
    r["e2e_trace"] = traceback.format_exc()[-2500:]
try:
    from utils.hardware import hardware as hw
    hw.detect_hardware()
    cap = hw.export_capability()
    r["export_capability"] = {k: cap.get(k) for k in ("export_supported", "export_unsupported_reason", "win32_rocm")}
    r["export_capability_has_win32_rocm"] = "win32_rocm" in cap
except BaseException as e:
    r["capability_error"] = err(e)
print("AMDCI_JSON=" + json.dumps(r))
'''


def _run(py: str, code: str, cwd: Path, args: list[str], timeout: int) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix = ".py", delete = False, encoding = "utf-8") as f:
        f.write(code)
        script = f.name
    studio_home = os.path.join(tempfile.gettempdir(), "amdci_studio_home")
    os.makedirs(studio_home, exist_ok = True)
    env = dict(os.environ, PYTHONPATH = str(cwd), PYTHONIOENCODING = "utf-8", UNSLOTH_DISABLE_AUTO_UPDATES = "1", UNSLOTH_STUDIO_HOME = studio_home)
    try:
        p = subprocess.run([py, script, *args], cwd = cwd, capture_output = True, text = True,
                           encoding = "utf-8", errors = "replace", timeout = timeout, env = env)
        rc, out, errout = p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return {"rc": -1, "error": f"timeout after {timeout}s"}
    finally:
        os.unlink(script)
    lines = [l for l in out.splitlines() if l.startswith(_MARK)]
    if not lines:
        return {"rc": rc, "error": "no result line", "stdout_tail": out[-2000:], "stderr_tail": errout[-3000:]}
    res = json.loads(lines[-1][len(_MARK):])
    res["rc"] = rc
    res["stderr_tail"] = errout[-1500:]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--torchao-dir", default = "")
    ap.add_argument("--timeout", type = int, default = 1200)
    args = ap.parse_args()

    backend = Path(args.checkout) / "studio" / "backend"
    obs: dict = {"state": args.state, "backend": str(backend)}
    if not backend.is_dir():
        obs["error"] = f"no such directory: {backend}"
    else:
        obs["premise"] = _run(args.python, _PREMISE, backend, [args.torchao_dir], args.timeout)
        obs["worker"] = _run(args.python, _WORKER, backend, [], args.timeout)
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

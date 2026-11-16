#!/usr/bin/env python3
"""Probe (PR 14655): does checking a candidate layer-split secondary flip the process-wide ROCm SDPA switches, and
at which dtype does the planner probe a MiniMax-H3 denoiser (fp32 stem, bf16 attention)?

Observes only. Each scenario runs in a fresh interpreter (fresh probe caches, fresh torch switches) with the
state's studio/backend first on sys.path, on the runner's real ROCm torch. The real probe children
(`_run_rocm_sdpa_children`, one subprocess per backend on the real card) answer for cuda:0. In the "mixed"
scenarios cuda:1's child answer is replaced by math-only (the only fake: this host has one gfx1151 APU, so a
real mixed pair does not exist); the probe, the guard and torch.backends.cuda switches are real. The "real_spoof"
scenario runs real children on cuda:1 too (the HIP device multiplier's renumbered copy of the same card).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_CHILD = r'''
import json, os, sys, traceback
backend, scenario, out_path, h3_path = sys.argv[1:5]
sys.path.insert(0, backend)
os.chdir(backend)
res = {"scenario": scenario}
SMOKE = os.environ.get("PR14655_CPU_SMOKE") == "1"
def dump():
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
try:
    import torch
    from types import SimpleNamespace
    from core.inference import diffusion_attention as da
    from core.inference import diffusion_layer_split_plan as lsp
    os.environ.pop(da.ROCM_FUSED_SDPA_ALLOW_ENV, None)
    if SMOKE:  # local CPU wiring check only: never set on the runner
        from core import _torchao_stub
        _torchao_stub._module_is_rocm = lambda t: True
        da._run_rocm_sdpa_children = lambda d, dt: ((da.SDPA_MATH,) if str(dt) == "torch.float32"
                                                    else (da.SDPA_FLASH, da.SDPA_MEM_EFFICIENT, da.SDPA_MATH))
        torch.cuda.device_count = lambda: 1
        torch.cuda.get_device_properties = lambda i: SimpleNamespace(gcnArchName="smoke")
        torch.cuda.synchronize = lambda *a: None
    res["smoke"] = SMOKE
    res["imported_from"] = {"diffusion_attention": da.__file__, "diffusion_layer_split_plan": lsp.__file__}
    res["torch"] = torch.__version__
    res["hip"] = getattr(torch.version, "hip", None)
    res["device_count"] = torch.cuda.device_count()
    res["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
    res["has_read_only_api"] = hasattr(da, "read_only_sdpa_probe")
    res["guard_env"] = da.ROCM_FUSED_SDPA_ALLOW_ENV
    flags = lambda: [torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled()]
    real_children = da._run_rocm_sdpa_children
    calls = []
    fake_secondary = scenario.startswith("mixed")
    def children(device, dtype):
        if fake_secondary and device != "cuda:0":
            ans = (da.SDPA_MATH,)
            calls.append({"device": device, "dtype": str(dtype), "answer": list(ans), "source": "faked math-only"})
            return ans
        ans = real_children(device, dtype)
        calls.append({"device": device, "dtype": str(dtype), "answer": list(ans), "source": "real child probe"})
        return ans
    da._run_rocm_sdpa_children = children
    class Log:
        def __init__(self): self.lines = []
        def info(self, *a): self.lines.append(a[0] % a[1:] if len(a) > 1 else a[0])
        warning = debug = info
    log = Log()

    if scenario == "kernels":
        res["real_kernels"] = {}
        for dt in (torch.float16, torch.bfloat16, torch.float32):
            res["real_kernels"][str(dt)] = list(real_children("cuda:0", dt))
        res["flash_attn_runs"] = {}
        for dt in (torch.bfloat16, torch.float32):
            try:
                res["flash_attn_runs"][str(dt)] = bool(da._rocm_flash_attn_runs(
                    SimpleNamespace(device="cuda", torch_device="cuda:0", dtype=dt)))
            except Exception as e:
                res["flash_attn_runs"][str(dt)] = f"error: {e!r}"
    elif scenario in ("mixed_switches", "real_spoof_switches"):
        pipe = SimpleNamespace(transformer=torch.nn.Linear(64, 64).to(torch.bfloat16), dtype=torch.bfloat16)
        res["flags_start"] = flags()
        res["probe_dtype"] = str(lsp._rocm_attention_dtype(pipe))
        res["primary_math_only"] = bool(da.sdpa_math_only(
            SimpleNamespace(device="cuda", torch_device="cuda:0", dtype=torch.bfloat16)))
        res["flags_after_primary_probe"] = flags()
        res["kept_first"] = list(lsp._rocm_drop_math_only(pipe, 0, (1,), log))
        res["flags_after_first_secondary_check"] = flags()
        res["kept_second"] = list(lsp._rocm_drop_math_only(pipe, 0, (1,), log))
        res["flags_after_second_secondary_check"] = flags()
        res["primary_math_only_after"] = bool(da.sdpa_math_only(
            SimpleNamespace(device="cuda", torch_device="cuda:0", dtype=torch.bfloat16)))
        # Does the primary's real attention still pick a fused kernel? A real bf16 SDPA call under the profiler.
        q = torch.randn(1, 4, 256, 64, device="cpu" if SMOKE else "cuda:0", dtype=torch.bfloat16)
        torch.nn.functional.scaled_dot_product_attention(q, q, q)
        torch.cuda.synchronize()
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
            torch.cuda.synchronize()
        names = sorted({e.key for e in prof.key_averages() if "attention" in e.key.lower() or "sdpa" in e.key.lower()
                        or "softmax" in e.key.lower() or "bmm" in e.key.lower()})
        res["primary_sdpa_ops_after"] = names
    elif scenario == "mixed_h3_dtype":
        sys.path.insert(0, h3_path)
        try:
            import pr14655_tiny_h3
            model = pr14655_tiny_h3.build()
            res["model"] = type(model).__module__ + "." + type(model).__name__
        except Exception as e:
            res["model_error"] = repr(e)
            class Attn(torch.nn.Module):
                def __init__(self):
                    super().__init__(); self.to_q = torch.nn.Linear(32, 32)
            class H3Shaped(torch.nn.Module):
                _keep_in_fp32_modules = ["proj_in"]
                def __init__(self):
                    super().__init__(); self.proj_in = torch.nn.Linear(4, 32)
                    self.blocks = torch.nn.ModuleList([torch.nn.Module()]); self.blocks[0].attn1 = Attn()
            model = H3Shaped().to(torch.bfloat16); model.proj_in.float()
            res["model"] = "synthetic H3-shaped (diffusers class unavailable)"
        first_name, first_param = next(model.named_parameters())
        res["first_param"] = [first_name, str(first_param.dtype)]
        attn = next(m for m in model.modules() if getattr(m, "to_q", None) is not None or getattr(m, "to_qkv", None) is not None)
        proj = getattr(attn, "to_q", None) or getattr(attn, "to_qkv")
        res["attention_projection_dtype"] = str(proj.weight.dtype)
        pipe = SimpleNamespace(transformer=model, dtype=torch.bfloat16)
        dt = lsp._rocm_attention_dtype(pipe)
        res["probe_dtype"] = str(dt)
        res["flash_branch_dtype"] = str(lsp._module_compute_dtype(model) or lsp._denoiser_dtype(model)) \
            if hasattr(lsp, "_module_compute_dtype") else str(lsp._denoiser_dtype(model))
        res["primary_math_only_at_probe_dtype"] = bool(da.sdpa_math_only(
            SimpleNamespace(device="cuda", torch_device="cuda:0", dtype=dt)))
        res["kept"] = list(lsp._rocm_drop_math_only(pipe, 0, (1,), log))
        res["flash_runs_at_flash_branch_dtype"] = None
        try:
            fdt = getattr(torch, res["flash_branch_dtype"].removeprefix("torch."))
            res["flash_runs_at_flash_branch_dtype"] = bool(lsp._rocm_flash_runs_on(0, fdt))
        except Exception as e:
            res["flash_runs_at_flash_branch_dtype"] = f"error: {e!r}"
        res["flags_end"] = flags()
    res["child_calls"] = calls
    res["log"] = log.lines
    res["ok"] = True
except Exception:
    res["ok"] = False
    res["error"] = traceback.format_exc()[-3000:]
dump()
'''

SCENARIOS = ("kernels", "mixed_switches", "real_spoof_switches", "mixed_h3_dtype")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--python", default = sys.executable)
    args, _ = ap.parse_known_args()
    backend = str(Path(args.checkout) / "studio" / "backend")
    here = str(Path(__file__).resolve().parent)
    obs: dict = {"state": args.state, "checkout": args.checkout, "scenarios": {}}
    with tempfile.TemporaryDirectory(dir = os.environ.get("RUNNER_TEMP")) as tmp:
        child = Path(tmp) / "child.py"
        child.write_text(_CHILD, encoding = "utf-8")
        for sc in SCENARIOS:
            out = Path(tmp) / f"{sc}.json"
            try:
                p = subprocess.run([args.python, str(child), backend, sc, str(out), here],
                                   capture_output = True, text = True, timeout = 900, cwd = backend)
                rec = json.loads(out.read_text(encoding = "utf-8")) if out.exists() else {"ok": False}
                rec["rc"] = p.returncode
                if not rec.get("ok"):
                    rec["stderr_tail"] = p.stderr[-2000:]
            except Exception as e:  # noqa: BLE001
                rec = {"ok": False, "error": repr(e)}
            obs["scenarios"][sc] = rec
    Path(args.out).write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

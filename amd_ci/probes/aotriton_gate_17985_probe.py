#!/usr/bin/env python3
"""Probe (#17985 / PR 17997): which AOTriton experimental switch the ROCm fused-SDPA probe children run under, what
the guard leaves on, and which aten SDPA op the parent then dispatches. Observes only (criteria/aotriton_gate_17985.py).

Arms per state, each scenario in a fresh interpreter on the runner's real ROCm torch and real gfx1151:
  tree   the state's own studio/backend.
  fix    (head state only) the state's studio/backend with core/inference/diffusion_attention.py replaced by the
         candidate fix shipped beside this probe (amd_ci/patches/diffusion_attention_17985.py).
Scenarios:
  real        the card as it is (gfx1151).
  as_gfx1103  rocm_bf16._device_gfx_arch answers "gfx1103" (exists in every tree). Only the arch Studio READS is
              faked; children, torch switches and kernels are real.
  truth       (base tree only) the stock rocm_sdpa_probe.py child per fused backend with the experimental switch
              "0" vs "1": does this torch gate gfx1151 behind it at all.
JSON to --out only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

GATE = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"

_CHILD = r'''
import json, os, sys, traceback
backend, scenario, out_path = sys.argv[1:4]
sys.path.insert(0, backend)
os.chdir(backend)
GATE = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"
os.environ.setdefault(GATE, "1")  # what Studio's main.py does before torch is imported
res = {"scenario": scenario, "parent_gate": os.environ.get(GATE)}
def dump():
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
try:
    import subprocess
    import torch
    from types import SimpleNamespace
    from core.inference import rocm_bf16
    if scenario == "as_gfx1103":
        rocm_bf16._device_gfx_arch = lambda t, o: "gfx1103"
    from core.inference import diffusion_attention as da
    from core.inference import host_pinning
    os.environ.pop(da.ROCM_FUSED_SDPA_ALLOW_ENV, None)
    res["imported_from"] = da.__file__
    res["torch"] = torch.__version__
    res["hip"] = getattr(torch.version, "hip", None)
    res["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
    res["integrated"] = bool(host_pinning.device_is_integrated("cuda:0"))
    res["has_fix"] = hasattr(da, "ROCM_AOTRITON_EXPERIMENTAL_VERIFIED_ARCHS")
    calls = []
    real_run = subprocess.run
    def run(cmd, **kw):
        r = real_run(cmd, **kw)
        if any("rocm_sdpa_probe" in str(c) for c in cmd):
            env = kw.get("env")
            calls.append({"backend": cmd[-1], "gate": (env if env is not None else os.environ).get(GATE),
                          "rc": r.returncode, "stdout": (r.stdout or "")[-200:]})
        return r
    subprocess.run = run
    flags = lambda: {"flash": torch.backends.cuda.flash_sdp_enabled(),
                     "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                     "math": torch.backends.cuda.math_sdp_enabled()}
    lines = []
    class Log:
        def info(self, *a): lines.append(a[0] % a[1:] if len(a) > 1 else a[0])
        warning = debug = info
    target = SimpleNamespace(device="cuda", torch_device="cuda:0", dtype=torch.bfloat16)
    res["guard_disabled"] = list(da.guard_rocm_fused_sdpa(target, Log()))
    res["kernels"] = list(da._SDPA_PROBE_CACHE.get(("cuda:0", "torch.bfloat16"), ()))
    res["math_only"] = bool(da.sdpa_math_only(target))
    res["flags"] = flags()
    res["child_calls"] = calls
    res["log"] = lines
    try:
        res["auto_flash"] = bool(da._rocm_auto_flash(target))
        res["flash_attn_installed"] = bool(da._flash_attn_installed())
    except Exception as e:
        res["auto_flash"] = f"error: {e!r}"
    from torch.profiler import profile, ProfilerActivity
    SDPA = ("aten::_scaled_dot_product_flash_attention", "aten::_scaled_dot_product_efficient_attention",
            "aten::_scaled_dot_product_attention_math")
    def op_and_err(shape, causal, mask):
        g = torch.Generator(device="cuda:0").manual_seed(0)
        q, k, v = (torch.randn(*shape, device="cuda:0", dtype=torch.bfloat16, generator=g) for _ in range(3))
        m = None
        if mask:
            m = torch.ones(shape[0], 1, shape[2], shape[2], device="cuda:0", dtype=torch.bool).tril()
            m[..., -8:] = False
            m[..., 0] = True
        sdpa = torch.nn.functional.scaled_dot_product_attention
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            out = sdpa(q, k, v, attn_mask=m, is_causal=causal)
            torch.cuda.synchronize()
        ops = sorted({e.key for e in prof.key_averages() if e.key in SDPA})
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.double(), k.double(), v.double(), attn_mask=m, is_causal=causal)
        err = ((out.double() - ref).abs().sum() / ref.abs().sum()).item()
        return {"ops": ops, "rel_l1": err, "finite": bool(torch.isfinite(out).all().item())}
    res["dispatch"] = {
        "text_encoder_causal": op_and_err((1, 32, 64, 128), True, False),
        "text_encoder_masked": op_and_err((1, 32, 64, 128), False, True),
        "dit_1024": op_and_err((1, 30, 1024, 128), False, False),
    }
    res["ok"] = True
except Exception:
    res["ok"] = False
    res["error"] = traceback.format_exc()[-3000:]
dump()
'''


def _truth(backend: str, python: str) -> dict:
    out: dict = {}
    script = str(Path(backend) / "core" / "inference" / "rocm_sdpa_probe.py")
    for gate in ("0", "1"):
        for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION"):
            env = dict(os.environ, **{GATE: gate})
            try:
                p = subprocess.run([python, script, "cuda:0", "bfloat16", name], capture_output = True,
                                   text = True, timeout = 300, env = env, cwd = backend)
                line = [x for x in p.stdout.splitlines() if x.startswith("UNSLOTH_SDPA_PROBE ")]
                out[f"{name}@{gate}"] = json.loads(line[-1].split(" ", 1)[1]) if line else f"rc={p.returncode}"
            except Exception as e:  # noqa: BLE001
                out[f"{name}@{gate}"] = f"error: {e!r}"
    return out


def _run(python: str, child: Path, backend: str, scenario: str, tmp: Path) -> dict:
    out = tmp / f"{scenario}_{abs(hash(backend))}.json"
    try:
        p = subprocess.run([python, str(child), backend, scenario, str(out)], capture_output = True, text = True,
                           timeout = 1200, cwd = backend)
        rec = json.loads(out.read_text(encoding = "utf-8")) if out.exists() else {"ok": False}
        rec["rc"] = p.returncode
        if not rec.get("ok"):
            rec["stderr_tail"] = p.stderr[-2000:]
    except Exception as e:  # noqa: BLE001
        rec = {"ok": False, "error": repr(e)}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--python", default = sys.executable)
    args, _ = ap.parse_known_args()
    backend = str(Path(args.checkout) / "studio" / "backend")
    fix_file = Path(__file__).resolve().parent.parent / "patches" / "diffusion_attention_17985.py"
    obs: dict = {"state": args.state, "checkout": args.checkout, "arms": {}}
    with tempfile.TemporaryDirectory(dir = os.environ.get("RUNNER_TEMP")) as tmp_s:
        tmp = Path(tmp_s)
        child = tmp / "child.py"
        child.write_text(_CHILD, encoding = "utf-8")
        arms = {"tree": backend}
        if args.state == "head":
            fix_backend = tmp / "fix_backend"
            shutil.copytree(backend, fix_backend, ignore = shutil.ignore_patterns("tests", "__pycache__", "*.pyc"))
            shutil.copy(fix_file, fix_backend / "core" / "inference" / "diffusion_attention.py")
            arms["fix"] = str(fix_backend)
        for arm, root in arms.items():
            obs["arms"][arm] = {"root": root, "scenarios": {
                sc: _run(args.python, child, root, sc, tmp) for sc in ("real", "as_gfx1103")}}
        if args.state == "base":
            obs["truth"] = _truth(backend, args.python)
    Path(args.out).write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

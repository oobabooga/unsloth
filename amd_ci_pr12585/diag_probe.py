#!/usr/bin/env python3
"""PR 12585 diagnostics on gfx1151 (observes only).

ga_*      test_gradient_accumulation_matches_one_large_batch under variants, plus where the
          parameters live and whether autocast is on during the training forward.
kernels   proposed transformers-version-proof kernel-binding test (copied into tests/).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

PLUGIN = r'''
import json, os, torch
_seen = []
def pytest_configure(config):
    if os.environ.get("DIAG_HIGHEST"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    from unsloth.models import decision
    orig = decision.DecisionTrainer.compute_loss
    def compute_loss(self, model, inputs, *a, **k):
        p = next(model.parameters())
        x = inputs.get("input_ids")
        _seen.append({"param_device": str(p.device), "param_dtype": str(p.dtype),
                      "input_device": str(getattr(x, "device", None)),
                      "autocast_cuda": torch.is_autocast_enabled("cuda"),
                      "autocast_cpu": torch.is_autocast_enabled("cpu"),
                      "batch": int(x.shape[0]) if x is not None else None})
        return orig(self, model, inputs, *a, **k)
    decision.DecisionTrainer.compute_loss = compute_loss
def pytest_unconfigure(config):
    with open(os.environ["DIAG_SEEN"], "w") as fh:
        json.dump(_seen, fh)
'''


def run(python, cwd, args, env_extra, timeout, seen):
    env = dict(os.environ, **env_extra)
    env["PYTHONPATH"] = os.pathsep.join([str(HERE), str(cwd), str(Path(cwd) / "tests"), env.get("PYTHONPATH", "")])
    env["DIAG_SEEN"] = str(seen)
    p = subprocess.run([python, "-m", "pytest", "-q", "-rf", "-s", "-p", "diag_plugin", *args], cwd = cwd,
                       env = env, capture_output = True, text = True, timeout = timeout)
    out = (p.stdout or "") + (p.stderr or "")
    rec = {"rc": p.returncode, "summary": (re.findall(r"^=*\s*(\d+ \w+.*in [\d.]+s.*)$", out, re.M) or [""])[-1]}
    m = re.search(r"Greatest absolute difference: ([\d.e+-]+).*?\n.*?Greatest relative difference: ([\d.e+-]+)", out)
    if m:
        rec["max_abs"], rec["max_rel"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"Mismatched elements: (\d+) / (\d+)", out)
    if m:
        rec["mismatched"] = f"{m.group(1)}/{m.group(2)}"
    rec["kernels"] = re.findall(r"KERNELS (.*)", out)
    try:
        s = json.loads(Path(seen).read_text())
        rec["calls"] = len(s)
        rec["seen"] = sorted({json.dumps(x, sort_keys = True) for x in s})
    except Exception as e:  # noqa: BLE001
        rec["seen_error"] = str(e)
    rec["tail"] = out[-5000:]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--timeout", type = int, default = 1800)
    args = ap.parse_args()
    co = Path(args.checkout).resolve()
    (HERE / "diag_plugin.py").write_text(PLUGIN, encoding = "utf-8")
    obs: dict = {"state": args.state, "runs": {}}
    if not (co / "tests" / "test_decision_model.py").exists():
        obs["absent"] = "no decision tests at this state"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0
    shutil.copy(HERE / "test_amd_diag_kernels.py", co / "tests" / "test_amd_diag_kernels.py")
    ga = "tests/test_decision_model.py::test_gradient_accumulation_matches_one_large_batch"
    seen = Path(os.environ.get("RUNNER_TEMP") or co) / "diag_seen.json"
    variants = {
        "ga_default": {},
        "ga_highest_precision": {"DIAG_HIGHEST": "1", "HIPBLASLT_ALLOW_TF32": "0"},
        "ga_no_gpu": {"HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": ""},
        "ga_one_thread": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
    }
    for name, env in variants.items():
        try:
            obs["runs"][name] = run(sys.executable, co, [ga], env, args.timeout, seen)
        except subprocess.TimeoutExpired:
            obs["runs"][name] = {"rc": -1, "summary": "timeout"}
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    for name, sel in (("kernels_original", "tests/test_decision_model.py::test_clef_backbone_runs_the_compiled_gated_delta_and_conv_kernels"),
                      ("kernels_proposed", "tests/test_amd_diag_kernels.py")):
        try:
            obs["runs"][name] = run(sys.executable, co, [sel], {}, args.timeout, seen)
        except subprocess.TimeoutExpired:
            obs["runs"][name] = {"rc": -1, "summary": "timeout"}
    import transformers
    obs["transformers"] = transformers.__version__
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

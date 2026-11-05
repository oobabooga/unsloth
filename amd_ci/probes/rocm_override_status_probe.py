#!/usr/bin/env python3
"""Probe: the vLLM status a cold (no torch) poll gives under HSA_OVERRIDE_GFX_VERSION, beside
what this checkout says once torch has run and the gcnArchName torch really reports.

One subprocess per (override, mode); observes only. Mode `cold` is the real poll. `cold_prereq`
and `hot` stub only the /opt/rocm version + library-file checks (the runner may lack a full
ROCm 7 install), so the GPU-target decision is reached on the real KFD topology and torch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

OVERRIDES = ["", "11.0.0", "10.3.0", "11.5.1", "9.0.10", "11.0.0 "]

CHILD = r"""
import json, sys
backend, mode = sys.argv[1], sys.argv[2]
sys.path.insert(0, backend)
out = {"mode": mode}
import core.inference.engine_install as ei
from utils.hardware import amd
out["kfd_targets"] = amd.amd_kfd_gpu_gfx_targets()
out["rocm_version"] = ei.rocm_version()
out["platform"] = ei.gpu_platform()
if mode != "cold":
    want = tuple(ei.profile("vllm").get("rocm") or (7, 0))
    ei.rocm_version = lambda: want
    ei.ROCM_LIBRARIES = ()
    ei.ROCM_SYSTEM_LIBRARIES = {}
wait = mode == "hot"
out["answers"] = [ei.support_reason("vllm", wait=wait) for _ in range(3)]
out["answers_gpu0"] = ei.support_reason("vllm", 0, wait=wait)
out["torch_imported_by_poll"] = "torch" in sys.modules
if mode == "hot":
    import torch
    out["torch_arches"] = [torch.cuda.get_device_properties(i).gcnArchName
                           for i in range(torch.cuda.device_count())]
print("PROBE_JSON " + json.dumps(out))
"""


def run(backend: Path, python: str, override: str, mode: str) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("HSA_OVERRIDE_GFX_VERSION")}
    if override:
        env["HSA_OVERRIDE_GFX_VERSION"] = override
    try:
        p = subprocess.run([python, "-c", CHILD, str(backend), mode], env=env, cwd=str(backend),
                           capture_output=True, text=True, timeout=300, encoding="utf-8")
    except subprocess.TimeoutExpired:
        return {"mode": mode, "error": "timeout"}
    for line in p.stdout.splitlines():
        if line.startswith("PROBE_JSON "):
            return json.loads(line[len("PROBE_JSON "):])
    return {"mode": mode, "error": f"rc={p.returncode}", "stderr_tail": p.stderr[-1500:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--checkout", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    backend = args.checkout / "studio" / "backend"
    obs: dict = {"state": args.state, "cases": {}}
    for override in OVERRIDES:
        obs["cases"][repr(override)] = {
            mode: run(backend, sys.executable, override, mode)
            for mode in ("cold", "cold_prereq", "hot")
        }
    args.out.write_text(json.dumps(obs, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

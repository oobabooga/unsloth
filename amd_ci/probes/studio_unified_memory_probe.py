#!/usr/bin/env python3
"""Probe: what does Studio decide about GGML_CUDA_ENABLE_UNIFIED_MEMORY here?

unsloth#7449 is a Windows Strix Halo (Radeon 8060S, 128 GB unified) where models
load into system RAM instead of the GPU pool. #9884 narrowed the setting to
"only when host RAM is the larger pool"; the issue's last comment says that on
Windows the comparison is *expected* to leave it off, and that nobody has run it
on a Windows Strix Halo. This probe is that run.

It observes three things per state and judges none of them:

  * what the ROCm runtime reports for this device: the arch, whether the driver
    calls it integrated, and `total_memory`, which on an APU is the BIOS
    carve-out and is the number the whole decision turns on
  * how much host RAM Studio's own helper sees
  * what the state's code DECIDES, for a range of model sizes either side of the
    carve-out

The decision helper was renamed by #9884, so the ladder below asks for each name
in turn and records WHICH one answered. A probe that hardcoded the new name would
report the old state as "no decision" and turn a real differential into a VOID.

Pairs with criteria/unified_memory_only_when_it_gains.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

GIB = 1024 ** 3

# Run in a CHILD so each state gets a fresh interpreter: the backend module
# caches its device classification, and two states imported into one process
# would have the first one's answer.
_MEASURE = r'''
import json, os, sys

out = {}
checkout = sys.argv[1]
needs = [float(x) for x in sys.argv[2].split(",") if x]
backend_dir = os.path.join(checkout, "studio", "backend")
sys.path.insert(0, backend_dir)

try:
    import torch
    out["torch_version"] = torch.__version__
    out["torch_hip"] = getattr(torch.version, "hip", None)
    out["torch_cuda"] = getattr(torch.version, "cuda", None)
    out["cuda_available"] = bool(torch.cuda.is_available())
    devices = []
    if out["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            devices.append({
                "ordinal": i,
                "name": getattr(p, "name", None),
                "gcnArchName": getattr(p, "gcnArchName", None),
                "is_integrated": getattr(p, "is_integrated", None),
                "total_memory": int(getattr(p, "total_memory", 0) or 0),
                "total_mib": int(getattr(p, "total_memory", 0) or 0) // (1024 * 1024),
            })
    out["devices"] = devices
except BaseException as e:
    out["torch_error"] = f"{type(e).__name__}: {e}"[:400]
    print(json.dumps(out)); raise SystemExit(0)

try:
    from core.inference.llama_cpp import LlamaCppBackend as B
    out["backend_file"] = getattr(B, "__module__", None)
except BaseException as e:
    out["backend_error"] = f"{type(e).__name__}: {e}"[:600]
    print(json.dumps(out)); raise SystemExit(0)


def call(name, *args, **kw):
    fn = getattr(B, name, None)
    if fn is None:
        return {"present": False}
    try:
        return {"present": True, "value": fn(*args, **kw)}
    except BaseException as e:
        return {"present": True, "error": f"{type(e).__name__}: {e}"[:300]}


out["rocm_unified_memory_gpu_ids"] = call("_rocm_unified_memory_gpu_ids")
ids = out["rocm_unified_memory_gpu_ids"].get("value")
out["rocm_unified_memory_gpu_ids"]["value"] = sorted(ids) if isinstance(ids, set) else ids
out["rocm_classification_answered"] = call("_rocm_classification_answered")
out["rocm_selected_pool_mib"] = call("_rocm_selected_pool_mib", [0])
out["available_system_memory_mib"] = call("_available_system_memory_mib")
out["amd_apu_wants_unified_memory"] = call("_amd_apu_wants_unified_memory", [0])
out["unified_memory_opted_out"] = call("_unified_memory_opted_out", {})

# The decision, for model sizes either side of whatever the carve-out turns out
# to be. Each entry records the function that answered, since the name changed.
decisions = []
for gib in needs:
    need_bytes = int(gib * (1024 ** 3))
    answer = {"need_gib": gib, "need_bytes": need_bytes}
    for name, args, kw in (
        ("_unified_memory_for_launch", ([0], need_bytes), {}),
        ("_unified_memory_would_help", ([0],), {"need_bytes": need_bytes}),
        ("_unified_memory_would_help", ([0],), {}),
        ("_amd_apu_wants_unified_memory", ([0],), {}),
    ):
        fn = getattr(B, name, None)
        if fn is None:
            continue
        try:
            answer["decided_by"] = name
            answer["kwargs"] = sorted(kw)
            answer["unified_memory"] = bool(fn(*args, **kw))
            break
        except TypeError as e:
            answer["last_type_error"] = f"{name}: {e}"[:200]
            continue
        except BaseException as e:
            answer["error"] = f"{name}: {type(e).__name__}: {e}"[:300]
            break
    decisions.append(answer)
out["decisions"] = decisions

# #7449's other half: the VRAM readout Studio shows in Settings.
try:
    from utils.hardware.hardware import get_gpu_summary, get_gpu_memory_info
    out["gpu_summary"] = get_gpu_summary()
    out["gpu_memory_info"] = get_gpu_memory_info()
except BaseException as e:
    out["gpu_readout_error"] = f"{type(e).__name__}: {e}"[:400]

print(json.dumps(out, default = str))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--need-gib", default = "1,4,16,40,90",
                    help = "model sizes, in GiB, to ask the decision about")
    ap.add_argument("--timeout", type = int, default = 1800)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "checkout": args.checkout,
                 "platform": sys.platform, "need_gib": args.need_gib}
    env = dict(os.environ)
    # The decision reads these; an inherited value would answer for the harness
    # rather than for a user's launch.
    for name in ("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "UNSLOTH_DISABLE_UNIFIED_MEMORY",
                 "UNSLOTH_ENABLE_UNIFIED_MEMORY"):
        env.pop(name, None)

    proc = subprocess.run(
        [sys.executable, "-c", _MEASURE, args.checkout, args.need_gib],
        capture_output = True, text = True, timeout = args.timeout, env = env,
        encoding = "utf-8", errors = "replace", stdin = subprocess.DEVNULL)
    obs["rc"] = proc.returncode
    obs["stderr_tail"] = proc.stderr[-4000:]

    # Import banners share stdout, so take the LAST JSON object on it.
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obs.update(json.loads(line))
                break
            except ValueError:
                continue
    else:
        obs["child_failed"] = True
        obs["stdout_tail"] = proc.stdout[-4000:]

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

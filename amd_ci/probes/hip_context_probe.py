#!/usr/bin/env python3
"""Probe: which calls (plain torch, then this checkout's Studio entry points) make a process a KFD process,
and how much GPU memory it then holds. Each step runs in a fresh child that reads its own
/sys/class/kfd/kfd/proc/<pid> entry and DRM fdinfo; reading sysfs opens no GPU state."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_CHILD = r'''
import glob, json, os, sys, time
pid = os.getpid()
def drm():
    """This process's amdgpu memory from DRM fdinfo (KiB keys), one reading per DRM client: duplicated
    fds of one client (same drm-pdev + drm-client-id) report the same memory and are counted once."""
    clients = {}
    for fd in os.listdir("/proc/self/fd"):
        try:
            if not os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/dri/"):
                continue
            ident, mem = {}, {}
            for line in open(f"/proc/self/fdinfo/{fd}"):
                k, _, v = line.partition(":")
                v = v.split()
                if k in ("drm-pdev", "drm-client-id") and v:
                    ident[k] = v[0]
                elif k.startswith("drm-") and k.split("-")[1] in ("memory", "resident", "total") \
                        and len(v) == 2 and v[1] == "KiB":
                    mem[k] = int(v[0]) * 1024
            key = (ident.get("drm-pdev"), ident["drm-client-id"]) if "drm-client-id" in ident else ("fd", fd)
            clients[key] = mem
        except (OSError, ValueError):
            pass
    tot = {}
    for mem in clients.values():
        for k, v in mem.items():
            tot[k] = tot.get(k, 0) + v
    return tot
def kfd():
    d = f"/sys/class/kfd/kfd/proc/{pid}"
    res = {"kfd_process": os.path.isdir(d), "vram_bytes": 0}
    for f in glob.glob(d + "/vram_*"):
        try:
            res["vram_bytes"] += int(open(f).read().split()[0])
        except (OSError, ValueError, IndexError):
            pass
    m = drm()
    res["drm"] = m
    # Resident GPU memory by either account: KFD's VRAM, or DRM's VRAM + GTT (an APU may
    # place allocations in GTT, which the KFD vram files do not count).
    best = 0
    for kind in ("memory", "resident", "total"):
        best = max(best, m.get(f"drm-{kind}-vram", 0) + m.get(f"drm-{kind}-gtt", 0))
    res["gpu_bytes"] = max(res["vram_bytes"], best)
    return res
out = {"start": kfd()}
sys.path.insert(0, sys.argv[1])
step = sys.argv[2]
try:
    if step.startswith("studio_"):
        import utils.hardware.hardware as hw
    import torch
    out["after_import"] = kfd()
    out["torch_hip"] = getattr(torch.version, "hip", None)
    if step == "import_torch":
        pass
    elif step == "is_available":
        out["value"] = torch.cuda.is_available()
    elif step == "device_count":
        out["value"] = torch.cuda.device_count()
    elif step == "get_device_properties":
        p = torch.cuda.get_device_properties(0)
        out["value"] = [p.name, p.total_memory, getattr(p, "gcnArchName", None)]
    elif step == "properties_then_mem_get_info":
        torch.cuda.get_device_properties(0)
        out["after_properties"] = kfd()
        out["value"] = list(torch.cuda.mem_get_info(0))
    elif step == "allocate_256mib":
        torch.cuda.mem_get_info(0)  # context first, so the reading below isolates the allocation
        out["before_alloc"] = kfd()
        x = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
        x.fill_(1)
        torch.cuda.synchronize()
        out["value"] = x.numel()
    elif step == "studio_detect_hardware":
        out["value"] = str(hw.detect_hardware())
    elif step == "studio_gpu_memory_info":
        hw.detect_hardware()
        out["after_detect"] = kfd()
        out["value"] = {k: v for k, v in hw.get_gpu_memory_info().items() if k in ("available", "backend", "total_gb")}
    elif step == "studio_device_inventory":
        hw.detect_hardware()
        out["after_detect"] = kfd()
        out["value"] = [{k: d.get(k) for k in ("index", "name", "total_gb")} for d in hw._torch_get_device_inventory([0])]
    elif step == "studio_llama_placement_helpers":
        hw.detect_hardware()
        out["after_detect"] = kfd()
        from core.inference.llama_cpp import LlamaCppBackend as L
        out["value"] = [sorted(L._integrated_cuda_gpu_ids()), L._all_selected_gpus_match(L._DATACENTER_GPU_RE, [0])]
    else:
        raise SystemExit("unknown step " + step)
except BaseException as e:
    out["error"] = f"{type(e).__name__}: {e}"[:400]
out["end"] = kfd()
# A file, not the last stdout line: whatever the code under test prints at exit cannot displace it.
with open(sys.argv[3] + ".tmp", "w", encoding="utf-8") as f:
    json.dump(out, f, default=str)
os.replace(sys.argv[3] + ".tmp", sys.argv[3])
'''

STEPS = (
    "import_torch",
    "is_available",
    "device_count",
    "get_device_properties",
    "properties_then_mem_get_info",
    "allocate_256mib",
    "studio_detect_hardware",
    "studio_gpu_memory_info",
    "studio_device_inventory",
    "studio_llama_placement_helpers",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--subdir", default = "studio/backend")
    ap.add_argument("--repeats", type = int, default = 2)
    args = ap.parse_args()

    backend = args.checkout / args.subdir
    obs: dict = {"state": args.state, "kfd_sysfs": os.path.isdir("/sys/class/kfd/kfd/proc"),
                 "python": sys.executable, "steps": {}}
    with tempfile.TemporaryDirectory() as td:
        for step in STEPS:
            runs = []
            for i in range(args.repeats):
                res = Path(td) / f"{step}.{i}.json"
                try:
                    r = subprocess.run([sys.executable, "-c", _CHILD, str(backend), step, str(res)],
                                       cwd = backend, capture_output = True, text = True, timeout = 300)
                    run = json.loads(res.read_text(encoding = "utf-8")) if res.is_file() else {}
                    if r.returncode != 0 or not run:  # a crash after writing (at exit) is still a failure
                        run.setdefault("error", f"rc={r.returncode} stderr={r.stderr[-400:]} stdout={r.stdout[-200:]}")
                        run["rc"] = r.returncode
                    runs.append(run)
                except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
                    runs.append({"error": f"{type(e).__name__}: {e}"[:400]})
            obs["steps"][step] = runs
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

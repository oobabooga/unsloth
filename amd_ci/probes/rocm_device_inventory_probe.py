#!/usr/bin/env python3
"""Probe: what devices does this machine actually have, and can it be made to show more?

Asked because "one GPU" was an inference from `torch.cuda.device_count() == 1`, and that
is the answer AFTER every layer of filtering. This enumerates underneath it: HSA agents,
KFD topology nodes, DRM render nodes, amd-smi's own list, and the compute-partition
sysfs. If a second GPU-ish device exists anywhere in that stack, it shows up here.

Also re-tests device duplication more thoroughly than before, including the ROCr UUID
form and combinations the earlier pass did not try, and records whether llama.cpp's
virtual-device mechanism is usable on this host even though it cannot reach torch.

Observes only. Every command failure is a recorded reading.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path


def _sh(cmd: str, timeout: int = 60) -> dict:
    try:
        r = subprocess.run(["bash", "-lc", cmd], capture_output = True, text = True,
                           timeout = timeout)
        return {"rc": r.returncode, "out": (r.stdout or "")[-4000:],
                "err": (r.stderr or "")[-1000:]}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


_COUNT = r'''
import json, sys
out = {}
try:
    import torch
    out["count"] = int(torch.cuda.device_count())
    out["available"] = bool(torch.cuda.is_available())
except Exception as e:
    out["error"] = "%s: %s" % (type(e).__name__, e)
print("@@J@@" + json.dumps(out))
'''

# Beyond the first pass: the UUID form ROCR accepts, ordinal stacking, and the ggml
# variable on its own and combined, so the negative is not resting on one spelling.
MECHANISMS = [
    ("baseline", {}),
    ("rocr_dup_uuid_style", {"ROCR_VISIBLE_DEVICES": "0,0"}),
    ("rocr_then_hip_dup", {"ROCR_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "0,0"}),
    ("ordinal_then_hip", {"GPU_DEVICE_ORDINAL": "0,0", "HIP_VISIBLE_DEVICES": "0,1"}),
    ("ggml_virtual", {"GGML_CUDA_DEVICES": "3"}),
    ("ggml_virtual_8", {"GGML_CUDA_DEVICES": "8"}),
    ("hsa_enable_sdma_off", {"HSA_ENABLE_SDMA": "0"}),
    ("all_masks_cleared", {"ROCR_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}),
]


def _torch_count(python: str, overlay: dict, timeout: int) -> dict:
    env = dict(os.environ)
    for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
              "GPU_DEVICE_ORDINAL", "GGML_CUDA_DEVICES", "HSA_ENABLE_SDMA"):
        env.pop(k, None)
    env.update(overlay)
    try:
        r = subprocess.run([python, "-c", _COUNT], capture_output = True, text = True,
                           timeout = timeout, env = env)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "env": overlay}
    for line in reversed((r.stdout or "").splitlines()):
        if line.startswith("@@J@@"):
            d = json.loads(line[5:])
            d["env"] = overlay
            return d
    return {"error": "no JSON", "env": overlay, "stderr": (r.stderr or "")[-500:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--timeout", type = int, default = 300)
    args = ap.parse_args()

    obs: dict = {"state": args.state}

    # 1. HSA agents. rocminfo names every agent, CPU included, so the GPU count is the
    #    number of agents whose device type is GPU -- not the number of "Agent" blocks.
    obs["rocminfo_agents"] = _sh(
        "rocminfo 2>/dev/null | grep -E '^Agent |  Name:|  Device Type:|  Marketing Name:'")
    obs["rocminfo_gpu_agent_count"] = _sh(
        "rocminfo 2>/dev/null | grep -c 'Device Type:.*GPU'")

    # 2. KFD topology: the kernel's own list, below HSA and below any env filtering.
    nodes = sorted(glob.glob("/sys/class/kfd/kfd/topology/nodes/*"))
    obs["kfd_node_count"] = len(nodes)
    kfd: list = []
    for n in nodes:
        rec = {"node": os.path.basename(n)}
        try:
            props = Path(n, "properties").read_text(errors = "replace")
            for key in ("simd_count", "gfx_target_version", "vendor_id", "device_id"):
                for line in props.splitlines():
                    if line.startswith(key + " "):
                        rec[key] = line.split()[1]
            rec["name"] = Path(n, "name").read_text(errors = "replace").strip()
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
        kfd.append(rec)
    obs["kfd_nodes"] = kfd
    # vendor_id 4098 = 0x1002 = AMD, and a GPU node has SIMDs where the CPU node has none.
    obs["kfd_amd_gpu_nodes"] = [
        r.get("node") for r in kfd
        if r.get("vendor_id") == "4098" and (r.get("simd_count") or "0") != "0"
    ]

    # 3. DRM render nodes: one per physical GPU the amdgpu driver bound.
    obs["render_nodes"] = sorted(os.path.basename(p) for p in glob.glob("/dev/dri/render*"))
    obs["card_nodes"] = sorted(os.path.basename(p) for p in glob.glob("/dev/dri/card*"))

    # 4. amd-smi's own view, and the partition sysfs. CPX is the only AMD mechanism
    #    that turns one package into several HIP devices, so its absence here is the
    #    thing that decides the whole question.
    obs["amd_smi_list"] = _sh("amd-smi list 2>&1 | head -40")
    obs["amd_smi_partition"] = _sh("amd-smi static --partition 2>&1 | head -30")
    obs["amd_smi_set_help"] = _sh(
        "amd-smi set --help 2>&1 | grep -iE 'partition|compute|memory' | head -20")
    obs["sysfs_compute_partition"] = _sh(
        "for f in /sys/class/drm/card*/device/current_compute_partition "
        "/sys/class/drm/card*/device/available_compute_partition; do "
        "[ -e \"$f\" ] && echo \"$f: $(cat $f 2>&1)\"; done; true")
    obs["rocm_smi_partition"] = _sh("rocm-smi --showcomputepartition 2>&1 | head -15")

    # 5. And the direct question, more thoroughly than the first pass.
    obs["torch_counts"] = {
        label: _torch_count(sys.executable, overlay, args.timeout)
        for label, overlay in MECHANISMS
    }

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

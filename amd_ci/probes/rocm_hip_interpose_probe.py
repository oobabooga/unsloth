#!/usr/bin/env python3
"""Probe: can torch be forced to see two REAL devices, below Python?

Every previous attempt patched Python. This tries the layer underneath: an LD_PRELOAD
shim over the HIP runtime that reports one more device than exists and remaps the extra
ordinal onto the real one. If it works, `torch.cuda.device_count()` is 2 because the C++
runtime says so, and a tensor can actually live on `cuda:1` -- which is the thing Python
patching provably cannot do.

Two facts found by reading torch and shape this:

  1. On ROCm, `torch.cuda.device_count()` prefers `_device_count_amdsmi()`, which calls
     the amdsmi PYTHON library, and only falls back to `torch._C._cuda_getDeviceCount()`
     (the HIP path) when that returns < 0. So a HIP shim alone gets overridden. amdsmi's
     Python binding is ctypes over libamd_smi.so, and ctypes resolves through a dlopen
     handle, so LD_PRELOAD cannot reach it either. The lever is to make the amdsmi
     import fail -- a real configuration, not a patch -- so torch uses the HIP count.
  2. Faking only the count yields hipErrorInvalidDevice as soon as anything selects the
     extra ordinal. The ordinal has to be remapped too, which is why the shim keeps a
     virtual current device rather than just inflating a number.

Known risk, recorded rather than assumed away: on ROCm 7.x, LD_PRELOAD interposition of
HIP is reported to recurse through HIP-internal calls, and LD_AUDIT is the suggested
alternative. This probe records the ROCm version and tests the shim standalone BEFORE
letting torch near it, so a failure is attributable.

Observes only. Every failure is a recorded reading, and the layered design means a
partial result still says exactly which layer stopped it.
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

# The SHIPPED shim, not a copy. A private duplicate here would drift from what
# scaffold.py --spoof-devices actually preloads, and this probe would then validate
# something nobody runs.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import device_multiplier as _dm  # noqa: E402

SHIM_C = _dm.SHIM_C

# A real interposition check. The first run used ctypes for this and read count == 1,
# which looked like the shim had failed. It had not: ctypes resolves through a dlopen
# handle, so it bypasses LD_PRELOAD entirely -- the same reason LD_PRELOAD cannot reach
# amdsmi's ctypes binding. Reasoning that applied to amdsmi and not to my own probe.
# A compiled binary binds through the PLT and is interposed.
PROBE_C = r'''
#include <stdio.h>
extern int hipGetDeviceCount(int *);
int main(void) {
    int n = -1;
    int rc = hipGetDeviceCount(&n);
    printf("{\"rc\": %d, \"count\": %d}\n", rc, n);
    return 0;
}
'''

_TORCH = r'''
import json, os, sys
out = {"ld_preload": os.environ.get("LD_PRELOAD"),
       "blocked_amdsmi": os.environ.get("SHIM_BLOCK_AMDSMI") == "1"}

if os.environ.get("SHIM_BLOCK_AMDSMI") == "1":
    # Make `import amdsmi` fail, so torch.cuda.device_count() falls back to the HIP
    # count the shim controls. This is a real configuration (amdsmi not installed),
    # not a patch of torch.
    import importlib.abc, importlib.machinery
    class _Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path = None, target = None):
            if name == "amdsmi" or name.startswith("amdsmi."):
                raise ImportError("amdsmi blocked by probe")
            return None
    sys.meta_path.insert(0, _Block())

try:
    import torch
    out["torch_version"] = torch.__version__
    out["hip"] = torch.version.hip
    out["has_pynvml"] = getattr(__import__("torch.cuda", fromlist = ["_HAS_PYNVML"]),
                                "_HAS_PYNVML", None)
    out["device_count"] = int(torch.cuda.device_count())
    try:
        out["cpp_device_count"] = int(torch._C._cuda_getDeviceCount())
    except Exception as e:
        out["cpp_device_count_error"] = "%s: %s" % (type(e).__name__, e)
    out["is_available"] = bool(torch.cuda.is_available())
    out["arch_list"] = list(torch.cuda.get_arch_list())

    archs = {}
    for i in range(out["device_count"]):
        try:
            archs[str(i)] = str(torch.cuda.get_device_properties(i).gcnArchName)
        except Exception as e:
            archs[str(i)] = "ERROR %s: %s" % (type(e).__name__, e)
    out["device_archs"] = archs

    # THE question. Python patching can never get here. Broken into steps so a
    # failure names the operation that failed rather than just "device 1".
    import traceback as _tb
    for i in range(min(2, out["device_count"])):
        key = "device_%d" % i
        dev = "cuda:%d" % i
        steps = {}
        try:
            steps["set_device"] = (torch.cuda.set_device(i), "ok")[1]
        except Exception as e:
            steps["set_device"] = "ERROR %s: %s" % (type(e).__name__, str(e)[:200])
        for label, fn in (
            ("empty",   lambda: torch.empty(4, device = dev)),
            ("zeros",   lambda: torch.zeros(256, 256, device = dev)),
            ("randn",   lambda: torch.randn(256, 256, device = dev)),
            ("copy_h2d", lambda: torch.ones(256, 256).to(dev)),
        ):
            try:
                t = fn()
                torch.cuda.synchronize()
                steps[label] = "ok"
                steps[label + "_dev"] = str(t.device)
            except Exception as e:
                steps[label] = "ERROR %s: %s" % (type(e).__name__, str(e)[:200])
        try:
            a = torch.randn(256, 256, device = dev)
            b = torch.randn(256, 256, device = dev)
            c = (a @ b).sum().item()
            torch.cuda.synchronize()
            steps["matmul"] = "ok" if c == c else "nan"
        except Exception as e:
            steps["matmul"] = "ERROR %s: %s" % (type(e).__name__, str(e)[:200])
            steps["matmul_tb"] = _tb.format_exc()[-900:]
        out[key] = steps
        out[key + "_matmul"] = steps.get("matmul")
except Exception as e:
    import traceback
    out["error"] = "%s: %s" % (type(e).__name__, e)
    out["tb"] = traceback.format_exc()[-1200:]

print("@@J@@" + json.dumps(out))
'''

# Standalone check: does the shim move hipGetDeviceCount at all, with no torch in the
# picture? Run FIRST, so a torch failure can be blamed on the right layer.
_CTYPES = r'''
import ctypes, json, os
out = {"ld_preload": os.environ.get("LD_PRELOAD")}
try:
    lib = ctypes.CDLL("libamdhip64.so")
    n = ctypes.c_int(-1)
    rc = lib.hipGetDeviceCount(ctypes.byref(n))
    out["rc"], out["count"] = rc, n.value
except Exception as e:
    out["error"] = "%s: %s" % (type(e).__name__, e)
print("@@J@@" + json.dumps(out))
'''


def _sh(cmd: str, timeout: int = 120) -> dict:
    try:
        r = subprocess.run(["bash", "-lc", cmd], capture_output = True, text = True,
                           timeout = timeout)
        return {"rc": r.returncode, "out": (r.stdout or "")[-2000:],
                "err": (r.stderr or "")[-800:]}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _run(script: str, env: dict, timeout: int, tag: str) -> dict:
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output = True,
                           text = True, timeout = timeout, env = env)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    for line in reversed((r.stdout or "").splitlines()):
        if line.startswith("@@J@@"):
            d = json.loads(line[5:])
            d["_rc"] = r.returncode
            return d
    return {"no_json": True, "_rc": r.returncode, "stderr": (r.stderr or "")[-1500:],
            "stdout": (r.stdout or "")[-800:], "_tag": tag}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--timeout", type = int, default = 900)
    args = ap.parse_args()

    obs: dict = {"state": args.state}

    # Context first. The ROCm 7.x LD_PRELOAD recursion report makes the version a
    # material reading rather than trivia.
    obs["rocm_version"] = _sh("cat /opt/rocm/.info/version 2>&1")
    obs["hip_lib"] = _sh("ldconfig -p 2>/dev/null | grep -i amdhip64 | head -3")
    obs["cc"] = _sh("which gcc cc 2>&1 | head -2")
    obs["amdsmi_python"] = _sh(
        f"{sys.executable} -c 'import amdsmi; print(getattr(amdsmi, \"__file__\", \"?\"))' 2>&1 | tail -2")

    with tempfile.TemporaryDirectory() as td:
        src, so = Path(td) / "shim.c", Path(td) / "libhipshim.so"
        src.write_text(SHIM_C, encoding = "utf-8")
        obs["build"] = _sh(f"gcc -shared -fPIC -O2 -o {so} {src} -ldl 2>&1")
        obs["shim_source"] = "amd_ci/lib/device_multiplier.py (shipped)"
        obs["built"] = so.is_file()

        # The compiled interposition check (see PROBE_C on why ctypes cannot do this).
        csrc, cbin = Path(td) / "count.c", Path(td) / "count"
        csrc.write_text(PROBE_C, encoding = "utf-8")
        hipdir = "/opt/rocm/lib"
        obs["cprobe_build"] = _sh(
            f"gcc -O2 -o {cbin} {csrc} -L{hipdir} -lamdhip64 -Wl,-rpath,{hipdir} 2>&1")
        obs["cprobe_built"] = cbin.is_file()

        base_env = dict(os.environ)
        for k in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
                  "HIP_VISIBLE_DEVICES"):
            base_env.pop(k, None)

        def _cprobe(env: dict) -> dict:
            if not obs.get("cprobe_built"):
                return {"skipped": "c probe did not build"}
            try:
                r = subprocess.run([str(cbin)], capture_output = True, text = True,
                                   timeout = 120, env = env)
                return {"rc": r.returncode, "out": (r.stdout or "").strip(),
                        "err": (r.stderr or "")[-400:]}
            except Exception as e:  # noqa: BLE001
                return {"error": f"{type(e).__name__}: {e}"}

        # Layer 0: no shim, for the control.
        obs["cprobe_clean"] = _cprobe(base_env)
        obs["ctypes_clean"] = _run(_CTYPES, base_env, args.timeout, "ctypes_clean")
        obs["torch_clean"] = _run(_TORCH, base_env, args.timeout, "torch_clean")

        if obs["built"]:
            shim_env = dict(base_env)
            shim_env["LD_PRELOAD"] = str(so)
            shim_env["SHIM_EXTRA_DEVICES"] = "1"

            # Layer 1: does the shim move HIP at all? Compiled binary, no torch.
            obs["cprobe_shimmed"] = _cprobe(shim_env)
            # Kept only to record that ctypes is NOT interposed, so a future reader
            # does not repeat my mistake of treating it as the interposition check.
            obs["ctypes_shimmed"] = _run(_CTYPES, shim_env, args.timeout,
                                         "ctypes_shimmed")
            # Layer 2: torch under the shim, LD_PRELOAD binding only. The previous run
            # reached device_count 2 and then died with "CUDA driver error: 101" on the
            # first allocation.
            obs["torch_shimmed"] = _run(_TORCH, shim_env, args.timeout,
                                        "torch_shimmed")

            # Layer 3: the expandable-segments allocator disabled, the path leaning
            # hardest on the driver API.
            noexp = dict(shim_env)
            noexp["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
            obs["torch_shimmed_noexp"] = _run(_TORCH, noexp, args.timeout,
                                              "torch_shimmed_noexp")
            # Control: amdsmi is not installed on this runner (has_pynvml False), so
            # torch already uses the HIP count and the block is expected to be inert.
            # Kept as a control that the count moves because of the shim.
            only_blocked = dict(base_env)
            only_blocked["SHIM_BLOCK_AMDSMI"] = "1"
            obs["torch_no_amdsmi_only"] = _run(_TORCH, only_blocked, args.timeout,
                                               "torch_no_amdsmi_only")

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

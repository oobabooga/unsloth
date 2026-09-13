#!/usr/bin/env python3
"""Probe: on THIS host, would the no-reserve policy hand DirectIO to the GPU?

Observes only. It records what each classifier actually answers on the machine
it is running on, and what `_gpu_offload_confirmed` concludes from those answers.
It does not decide whether any of that is acceptable -- criteria do.

Why this needs real hardware: the gfx1151 in these runners is a Strix Halo, a
unified-memory APU whose "VRAM" is carved out of system RAM. Under
`--load-mode dio` llama.cpp does not map the GGUF (src/llama-model-loader.cpp:559),
so every byte it leaves off a real discrete GPU becomes an allocated host buffer.
Choosing dio for an APU therefore reserves the whole model in RAM -- the opposite
of what "Don't reserve system RAM" promises, and worse than the mmap it replaces.

Whether that happens turns on `_rocm_unified_memory_gpu_ids`, which reads the
driver through a ROCm-enabled torch. No simulation can tell you whether such a
torch exists on a given machine; that is the fact this probe is here to read.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

#: What a HIP build of llama-server prints for this host under `--list-devices`.
#: Stubbed rather than executed: the runner has no llama.cpp install, and the id
#: format is fixed by ggml (`common/arg.cpp:1138-1161`). The stub is the INPUT to
#: the decision under test, not the decision.
ASSUMED_DEVICE_IDS = ["ROCm0"]


def _amd_smi() -> dict:
    exe = shutil.which("amd-smi")
    if not exe:
        return {"present": False}
    try:
        p = subprocess.run([exe, "static", "--json"], capture_output = True,
                           text = True, timeout = 120)
        return {"present": True, "rc": p.returncode, "stdout_head": p.stdout[:4000]}
    except Exception as e:  # noqa: BLE001
        return {"present": True, "error": repr(e)}


def _video_controllers() -> dict:
    """What Windows itself says is in the box.

    Needed because the two signals the rest of this probe reads are BOTH absent
    on exactly the host that matters: there is no ROCm torch on Windows, and
    amd-smi is present but answers `Error LoadLibraryA` there. Without a third
    source the run cannot corroborate that it is talking about an APU, which is
    the gate the criteria refuses to pass on faith.

    Reuses the toolkit's own mapping rather than a second copy of it.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
        import capability  # type: ignore[import-not-found]

        names = capability.windows_video_controllers()
        return {"names": names, "archs": capability.archs_from_windows_video_controllers(names)}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def _torch_facts() -> dict:
    out: dict = {"importable": False}
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        out["import_error"] = repr(e)
        return out
    out["importable"] = True
    out["version"] = getattr(torch, "__version__", None)
    out["hip_version"] = getattr(getattr(torch, "version", None), "hip", None)
    out["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        out["cuda_is_available"] = bool(torch.cuda.is_available())
        out["device_count"] = int(torch.cuda.device_count()) if out["cuda_is_available"] else 0
        out["device_names"] = [
            torch.cuda.get_device_name(i) for i in range(out["device_count"])
        ]
    except Exception as e:  # noqa: BLE001
        out["query_error"] = repr(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--subdir", default = "studio/backend")
    args = ap.parse_args()
    # Resolved BEFORE the chdir below, or a relative --out lands in the checkout
    # and the runner finds no observations where it left them.
    args.out = args.out.resolve()

    obs: dict = {
        "state": args.state,
        "platform": sys.platform,
        "machine": platform.machine(),
        "node": platform.node(),
        "assumed_device_ids": ASSUMED_DEVICE_IDS,
    }
    obs["amd_smi"] = _amd_smi()
    obs["video_controllers"] = _video_controllers()
    obs["torch"] = _torch_facts()

    backend_root = Path(args.checkout) / args.subdir
    obs["backend_root"] = str(backend_root)
    if not backend_root.is_dir():
        obs["error"] = f"no such directory: {backend_root}"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    sys.path.insert(0, str(backend_root))
    os.chdir(backend_root)
    try:
        from core.inference.llama_cpp import LlamaCppBackend as B
    except Exception as e:  # noqa: BLE001
        obs["import_error"] = repr(e)
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    # Does this revision even have the feature? Absent at the merge base, which
    # is a fact the criteria needs rather than an error.
    obs["feature_present"] = hasattr(B, "_gpu_offload_confirmed")
    obs["classification_helper_present"] = hasattr(B, "_rocm_classification_answered")

    # The three answers the decision is built from, read on this machine.
    try:
        obs["rocm_unified_memory_gpu_ids"] = sorted(B._rocm_unified_memory_gpu_ids())
    except Exception as e:  # noqa: BLE001
        obs["rocm_unified_memory_gpu_ids_error"] = repr(e)
    try:
        obs["amd_apu_wants_unified_memory"] = bool(B._amd_apu_wants_unified_memory([0]))
    except Exception as e:  # noqa: BLE001
        obs["amd_apu_wants_unified_memory_error"] = repr(e)
    if obs["classification_helper_present"]:
        try:
            obs["rocm_classification_answered"] = bool(B._rocm_classification_answered())
        except Exception as e:  # noqa: BLE001
            obs["rocm_classification_answered_error"] = repr(e)

    if not obs["feature_present"]:
        obs["note"] = "this revision has no _gpu_offload_confirmed; nothing can emit dio"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    # `_weights_in_host_memory` returns exactly this for the non-Vulkan,
    # every-layer-offloaded case, which is the only shape that reaches the dio
    # gate. Taken from the same helper rather than hand-set, so the probe cannot
    # quietly feed the decision a residency answer the launch would not.
    host_resident = bool(obs.get("amd_apu_wants_unified_memory", False))
    obs["host_resident_fed_to_the_gate"] = host_resident

    original = B._enumerated_gpu_devices
    try:
        B._enumerated_gpu_devices = classmethod(  # type: ignore[assignment]
            lambda cls, binary = None, env = None: list(ASSUMED_DEVICE_IDS)
        )
        obs["gpu_offload_confirmed"] = bool(
            B._gpu_offload_confirmed("llama-server", {}, [0], host_resident, True, None)
        )
    except Exception as e:  # noqa: BLE001
        obs["gpu_offload_confirmed_error"] = repr(e)
    finally:
        B._enumerated_gpu_devices = original  # type: ignore[assignment]

    # And the end of the chain: the flags a Windows no-reserve launch would carry.
    try:
        from core.inference.llama_server_args import resolve_launch_load_mode

        real_platform = sys.platform
        sys.platform = "win32"          # native on the Windows leg; declared on Linux
        try:
            emitted, effective = resolve_launch_load_mode(
                None,
                supports_load_mode = True,
                weights_in_host_memory = host_resident,
                gpu_offload_confirmed = bool(obs.get("gpu_offload_confirmed", False)),
                requested_load_mode = None,
                env = {},
                settings = (False, True),   # keep_resident off, no-reserve ON
            )
        finally:
            sys.platform = real_platform
        obs["policy_emitted_dio"] = bool(emitted)
        obs["effective_dio"] = bool(effective)
        obs["decision_platform"] = "win32"
        obs["decision_platform_is_native"] = real_platform == "win32"
    except Exception as e:  # noqa: BLE001
        obs["policy_error"] = repr(e)

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

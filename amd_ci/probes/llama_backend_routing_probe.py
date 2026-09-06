#!/usr/bin/env python3
"""Probe: which llama.cpp bundle does this checkout pick on THIS machine?

Observes only. It reports what `detect_host()` saw and which asset
`direct_upstream_release_plan()` put first, against a canned release payload so
the reading never depends on the GitHub API.

The canned payload matters twice over. It keeps the probe offline, and it keeps
base and head reading the SAME asset list: if one state fetched a release that
had since gained or lost an asset, a routing difference would be the release
moving rather than the change under test.

`--container IMAGE` re-executes the whole probe inside docker with the checkout
bind-mounted and the DRM/KFD device nodes passed through. That is how an AMD GPU
with no usable ROCm is reached on a runner that has ROCm installed: the silicon
and its `/sys/class/drm/card*/device/vendor` are the real ones, while rocminfo,
amd-smi and /opt/rocm are genuinely absent from the image rather than hidden by
an environment variable the code under test is documented to ignore.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# The real ggml-org/llama.cpp b6100 asset set, trimmed to the names any Linux or
# Windows branch looks up. Nothing here is invented: an asset the probe omits
# would read as "upstream does not publish it", which is a different answer.
RELEASE_TAG = "b6100"
_ASSET_NAMES = [
    f"llama-{RELEASE_TAG}-bin-ubuntu-x64.tar.gz",
    f"llama-{RELEASE_TAG}-bin-ubuntu-vulkan-x64.tar.gz",
    f"llama-{RELEASE_TAG}-bin-ubuntu-arm64.tar.gz",
    f"llama-{RELEASE_TAG}-bin-ubuntu-vulkan-arm64.tar.gz",
    f"llama-{RELEASE_TAG}-bin-ubuntu-rocm-6.4-x64.tar.gz",
    f"llama-{RELEASE_TAG}-bin-win-cpu-x64.zip",
    f"llama-{RELEASE_TAG}-bin-win-vulkan-x64.zip",
    f"llama-{RELEASE_TAG}-bin-macos-arm64.tar.gz",
]


def _canned_release() -> dict:
    return {
        "tag_name": RELEASE_TAG,
        "assets": [
            {"name": n, "browser_download_url": f"https://example.invalid/{n}"}
            for n in _ASSET_NAMES
        ],
    }


def _drm_vendors() -> list[str]:
    """The vendor ids sysfs reports, read the same way the code under test reads them.

    Recorded whatever the verdict, because "no 0x1002 in sysfs" and "0x1002 present
    but routed to CPU anyway" are different findings and the table has to tell them
    apart.
    """
    out = []
    for path in sorted(glob.glob("/sys/class/drm/card*/device/vendor")):
        try:
            with open(path, encoding = "utf-8") as handle:
                out.append(f"{path}={handle.read().strip().lower()}")
        except (OSError, UnicodeDecodeError):
            out.append(f"{path}=<unreadable>")
    return out


def _rocm_tools() -> dict:
    return {
        "rocminfo_on_path": shutil.which("rocminfo"),
        "amd_smi_on_path": shutil.which("amd-smi"),
        "opt_rocm_rocminfo": os.access("/opt/rocm/bin/rocminfo", os.X_OK),
        "opt_rocm_exists": os.path.isdir("/opt/rocm"),
    }


def _in_container(args) -> int:
    checkout = Path(args.checkout).resolve()
    # The container sees the checkout at the same path, so nothing in the probe
    # needs to know it is containerised.
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{checkout}:{checkout}:ro",
        "-v", f"{Path(__file__).resolve().parent}:/probe:ro",
        "-w", str(checkout),
        "-e", "PYTHONDONTWRITEBYTECODE=1",
    ]
    for dev in ("/dev/dri", "/dev/kfd"):
        if os.path.exists(dev):
            cmd += ["--device", dev]
    # /sys is mounted read-only in a container by default, which is all the
    # vendor-id read needs.
    cmd += [
        args.container, "python3", f"/probe/{Path(__file__).name}",
        "--state", args.state, "--checkout", str(checkout), "--out", "/dev/stdout",
        "--emit-json-only",
    ]
    proc = subprocess.run(cmd, capture_output = True, text = True, timeout = 900)
    obs: dict = {"state": args.state, "container": args.container}
    # The container's stdout carries the JSON document and nothing else, but a
    # docker warning on stderr must not be parsed as part of it (E004 in spirit).
    text = (proc.stdout or "").strip()
    try:
        obs.update(json.loads(text[text.index("{"):text.rindex("}") + 1]))
    except (ValueError, json.JSONDecodeError):
        obs["error"] = "container probe produced no JSON"
        obs["docker_rc"] = proc.returncode
        obs["stdout_tail"] = text[-2000:]
        obs["stderr_tail"] = (proc.stderr or "")[-2000:]
    obs["containerised"] = True
    Path(args.out).write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--container", default = "")
    ap.add_argument("--emit-json-only", action = "store_true")
    args = ap.parse_args()

    if args.container:
        return _in_container(args)

    obs: dict = {
        "state": args.state,
        "platform": sys.platform,
        "drm_vendors": _drm_vendors(),
        "rocm_tools": _rocm_tools(),
        "containerised": False,
    }

    sys.path.insert(0, str(Path(args.checkout) / "studio"))
    try:
        import install_llama_prebuilt as ilp
    except Exception as exc:                      # noqa: BLE001
        obs["error"] = f"import failed: {type(exc).__name__}: {exc}"
        Path(args.out).write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    try:
        host = ilp.detect_host()
    except Exception as exc:                      # noqa: BLE001
        obs["error"] = f"detect_host failed: {type(exc).__name__}: {exc}"
        Path(args.out).write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    # getattr with a default, not attribute access: `has_amd_gpu_without_rocm` is
    # the field this change ADDS, so it does not exist at the base state and a
    # bare read would turn the base leg into a probe error instead of a reading.
    obs["host"] = {
        name: getattr(host, name, None)
        for name in (
            "is_linux", "is_windows", "is_macos", "is_x86_64", "is_arm64",
            "has_physical_nvidia", "has_usable_nvidia", "has_rocm",
            "has_intel_gpu", "has_amd_gpu_without_rocm",
            "rocm_gfx_target", "rocm_gfx_targets",
        )
    }
    obs["has_amd_gpu_without_rocm_field_exists"] = hasattr(host, "has_amd_gpu_without_rocm")

    try:
        plan = ilp.direct_upstream_release_plan(
            _canned_release(), host, "ggml-org/llama.cpp", RELEASE_TAG
        )
    except Exception as exc:                      # noqa: BLE001
        obs["plan_error"] = f"{type(exc).__name__}: {exc}"
        plan = None
    if plan is None:
        obs["attempts"] = []
        obs["first_install_kind"] = None
    else:
        obs["attempts"] = [
            {"name": a.name, "install_kind": a.install_kind} for a in plan.attempts
        ]
        obs["first_install_kind"] = plan.attempts[0].install_kind if plan.attempts else None

    Path(args.out).write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

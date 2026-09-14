#!/usr/bin/env python3
"""Run this checkout's install.sh AMD architecture routing against the REAL host.

PR 9672 replaces `rocminfo | grep -oE 'gfx...'` with per-device records
(`_rocminfo_gpu_records | _gfx_arch_slots`) and adds an amd-smi HIP_ID
translation. Every test for that is a stub. This probe feeds the block the
actual `rocminfo`, `amd-smi list`, `amd-smi list -e` and `amd-smi static --asic`
output of the machine it runs on, and records what the routing decides.

It observes and does not judge: the same block is lifted from whichever checkout
the state names, so base and head can be compared on one host's real output.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

_HELPERS = (
    "_rocm_leaf_below",
    "_rocminfo_gpu_records",
    "_amd_smi_gpu_records",
    "_amd_smi_hip_order",
    "_gfx_arch_slots",
    "_infer_linux_amd_gfx_arch",
    "_hsa_spoofed_physical_gfx",
    "_run_bounded",
)


def _function_body(source: str, name: str) -> str:
    """The shell function `name`, by brace matching. Empty when absent."""
    marker = f"\n{name}() {{"
    start = source.find(marker)
    if start < 0:
        return ""
    start += 1
    depth, i = 0, start
    while i < len(source):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
        i += 1
    return ""


def _capture(cmd, timeout=60):
    exe = shutil.which(cmd[0])
    if not exe:
        return {"present": False}
    try:
        run = subprocess.run(
            [exe, *cmd[1:]], capture_output=True, text=True, timeout=timeout
        )
        return {
            "present": True,
            "rc": run.returncode,
            "stdout": run.stdout[-20000:],
            "stderr_tail": run.stderr[-500:],
        }
    except subprocess.TimeoutExpired:
        return {"present": True, "timeout": True}
    except Exception as e:  # pragma: no cover - defensive
        return {"present": True, "error": f"{type(e).__name__}: {e}"}


def _route(source: str, leaf: str, env_extra: dict, timeout=180):
    """Run the architecture-routing arm with `leaf` as the resolved index leaf."""
    start = source.find('case "$_torch_index_leaf" in\n    rocm[0-9]*)')
    end = source.find("\nfi  # _torch_index_pinned guard", start)
    if start < 0 or end < 0:
        return {"block_not_found": True}
    helpers = "\n".join(_function_body(source, n) for n in _HELPERS)
    script = (
        "set -euo pipefail\n"
        + helpers
        + "\n"
        + f'TORCH_INDEX_URL="https://download.pytorch.org/whl/{leaf}"\n'
        + f'_torch_index_leaf="{leaf}"\n'
        + "_torch_index_pinned=false\nSKIP_TORCH=false\n_amd_gpu_radeon=false\n"
        + "_gfx_rocm64_target=false\n"
        + source[start:end]
        + '\nprintf "RESULT %s|%s|%s|%s\\n" "$_torch_index_leaf" "$TORCH_INDEX_URL"'
        ' "${_runtime_gfx:-}" "$_amd_gpu_radeon"\n'
    )
    env = dict(os.environ, **env_extra)
    for name in (
        "HSA_OVERRIDE_GFX_VERSION",
        "UNSLOTH_ROCM_GFX_ARCH",
        "UNSLOTH_PYTORCH_MIRROR",
        "UNSLOTH_AMD_ROCM_MIRROR",
    ):
        if name not in env_extra:
            env.pop(name, None)
    for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if name not in env_extra:
            env.pop(name, None)
    try:
        run = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True,
            env=env, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"timeout": True}
    out = {"rc": run.returncode, "stderr_tail": run.stderr[-1500:]}
    m = re.search(r"^RESULT (.*)$", run.stdout, re.MULTILINE)
    if m:
        leaf_out, url, runtime_gfx, radeon = m.group(1).split("|")
        out.update(
            leaf=leaf_out, url=url, runtime_gfx=runtime_gfx, radeon=radeon,
        )
    else:
        out["no_result"] = True
        out["stdout_tail"] = run.stdout[-1500:]
    return out


def _pipe(source: str, fn: str, text: str, timeout=60):
    """Feed `text` through one of the checkout's awk helpers."""
    body = _function_body(source, fn)
    if not body:
        return {"absent_at_this_state": True}
    try:
        run = subprocess.run(
            ["bash", "-c", body + f'\n{fn}\n'],
            input=text, capture_output=True, text=True, timeout=timeout,
        )
        return {"rc": run.returncode, "stdout": run.stdout}
    except subprocess.TimeoutExpired:
        return {"timeout": True}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--checkout", required=True, type=Path)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    obs: dict = {"state": args.state}
    install_sh = args.checkout / "install.sh"
    try:
        source = install_sh.read_text(encoding="utf-8")
    except Exception as e:
        obs["install_sh_error"] = f"{type(e).__name__}: {e}"
        Path(args.out).write_text(json.dumps(obs, indent=2), encoding="utf-8")
        return 0

    obs["helpers_present"] = {n: bool(_function_body(source, n)) for n in _HELPERS}

    tools = {
        "rocminfo": _capture(["rocminfo"]),
        "amd_smi_list": _capture(["amd-smi", "list"]),
        "amd_smi_list_e": _capture(["amd-smi", "list", "-e"]),
        "amd_smi_static_asic": _capture(["amd-smi", "static", "--asic"]),
    }
    # Raw output is the same on both states; keep it once, under a key the
    # differential treats as an ordinary field.
    obs["tools_present"] = {k: v.get("present") for k, v in tools.items()}
    obs["rocminfo_head"] = (tools["rocminfo"].get("stdout") or "")[:4000]
    obs["amd_smi_list_head"] = (tools["amd_smi_list"].get("stdout") or "")[:4000]

    obs["records"] = {
        "rocminfo_gpu_records": _pipe(
            source, "_rocminfo_gpu_records", tools["rocminfo"].get("stdout") or ""
        ),
        "amd_smi_gpu_records": _pipe(
            source, "_amd_smi_gpu_records", tools["amd_smi_list"].get("stdout") or ""
        ),
        "amd_smi_static_records": _pipe(
            source,
            "_amd_smi_gpu_records",
            tools["amd_smi_static_asic"].get("stdout") or "",
        ),
    }

    # The real question: on this host's real probe output, where does each state
    # route? Several leaves, because the reroutes are leaf-conditional.
    obs["routing"] = {}
    for leaf in ("rocm6.1", "rocm6.3", "rocm6.4", "rocm7.0", "rocm7.2"):
        obs["routing"][leaf] = _route(source, leaf, {})
    # And with the masks a user would actually set on a one-GPU box.
    for label, extra in (
        ("hip0", {"HIP_VISIBLE_DEVICES": "0"}),
        ("rocr0", {"ROCR_VISIBLE_DEVICES": "0"}),
        ("cuda0", {"CUDA_VISIBLE_DEVICES": "0"}),
        ("hip_empty", {"HIP_VISIBLE_DEVICES": ""}),
        ("override_gfx1102", {"UNSLOTH_ROCM_GFX_ARCH": "gfx1102"}),
        ("override_gfx1200", {"UNSLOTH_ROCM_GFX_ARCH": "gfx1200"}),
        ("override_gfx906", {"UNSLOTH_ROCM_GFX_ARCH": "gfx906"}),
    ):
        obs["routing"][f"rocm6.1+{label}"] = _route(source, "rocm6.1", extra)

    Path(args.out).write_text(json.dumps(obs, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

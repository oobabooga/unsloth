#!/usr/bin/env python3
"""Probe: which torch index does install.sh's TOP-LEVEL reroute actually resolve?

The existing rocm_wheel_route_probe drives `get_torch_index_url` and nothing else, so it
cannot see PR 9152 at all: that function still returns */cpu on the Fedora shape by design
and DEFERS to a reroute in install.sh's top-level code. This probe splices that block out
by the same stable markers the repo's own e2e harness uses and EXECUTES it, so the decision
is observed rather than grepped for.

  start marker  ^_ROCM_TAG_MEMO_DIR=$(mktemp   (head) or ^TORCH_INDEX_URL=$(get_torch_index_url)$
  end marker    ^fi  # _torch_index_pinned guard

Both markers exist at the base as well as at the head, which is what makes a differential
possible: at the base the reported Fedora host resolves */cpu (the defect), at the head it
resolves the per-arch AMD index.

Every host shape is faked -- stub rocminfo / amd-smi / hipconfig / rpm / dpkg-query / lspci /
uname on a hermetic PATH, and /opt/rocm, /dev/kfd, /sys/class/kfd, /sys/bus/pci/devices,
/proc/driver/nvidia, /proc/cpuinfo, /proc/version redirected into a scratch tree. On this
runner that shadowing is the whole point: it is a real ROCm gfx1151 box, so without it every
AMD scenario would measure the machine instead of the layout. `real_host` is the one scenario
left unstubbed, as the control that says the stubs did not simply break routing for everyone.

Observes only. Whether a routing is right is the criteria's call.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

_FUNC = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*\(\)[ \t]*\{[ \t]*$")

_BLOCK_START = re.compile(
    r"^(_ROCM_TAG_MEMO_DIR=\$\(mktemp|TORCH_INDEX_URL=\$\(get_torch_index_url\)$)")
_BLOCK_END = "fi  # _torch_index_pinned guard"

# Extract by NAME, not "every column-0 function": install.sh embeds whole scripts as
# heredocs whose payloads carry a bare `}` at column 0, and a naive scan runs past the end of
# a function and drags top-level code in with it. A name absent from a state is recorded, not
# an error -- 9152's helpers legitimately do not exist at its base.
WANTED = [
    "_run_bounded", "_cvd_hides_nvidia", "_has_amd_rocm_gpu", "_has_usable_nvidia_gpu",
    "_ensure_rocm_probe_env", "_probe_amd_gfx_arch", "_amd_gpu_present_via_pci",
    "_infer_amd_gfx_arch_from_gpu_name", "_infer_linux_amd_gfx_arch",
    "_amd_arch_index_family_for_gfx", "_trim_index_path_slashes",
    "_nvidia_cu126_verdict", "_cap_cuda_family_for_pre_turing",
    "_rocm_tag_from_amd_smi", "_rocm_tag_from_version_file", "_rocm_tag_from_hipconfig",
    "_rocm_tag_from_dpkg", "_rocm_tag_from_rpm", "_highest_rocm_tag",
    "_detect_rocm_version_tag", "get_torch_index_url",
    "_strip_index_url_credentials", "_expected_torch_flavor_tag",
    "_is_pip_rocm_family_leaf", "_hsa_spoofed_physical_gfx",
    "_radeon_host_ver_not_older", "get_radeon_wheel_url",
    # added by 9152
    "_amd_probe_arches", "_amd_agreed_index_family", "_amd_sole_index_arch",
    "_rocm_sdk_install_hint",
]

# Must exist in EVERY state. A helper the block calls but the splice never defined makes the
# branch die and the whole thing resolve to "cpu" -- indistinguishable from a routing
# decision, and exactly the bug 9152 fixes, so it must fail the run rather than score.
REQUIRED = [
    "_run_bounded", "_cvd_hides_nvidia", "_has_amd_rocm_gpu", "_has_usable_nvidia_gpu",
    "_probe_amd_gfx_arch", "_infer_linux_amd_gfx_arch", "_amd_arch_index_family_for_gfx",
    "_rocm_tag_from_amd_smi", "_rocm_tag_from_version_file", "_rocm_tag_from_hipconfig",
    "_rocm_tag_from_dpkg", "_rocm_tag_from_rpm", "_highest_rocm_tag",
    "_detect_rocm_version_tag", "get_torch_index_url", "_strip_index_url_credentials",
    "_is_pip_rocm_family_leaf",
]

_REDIRECT = ["/usr/bin/nvidia-smi", "/proc/driver/nvidia", "/opt/rocm", "/sys/class/kfd",
             "/sys/bus/pci/devices", "/dev/kfd", "/dev/dxg", "/proc/cpuinfo",
             "/proc/version"]

_TOOLS = ("uname", "grep", "sed", "awk", "head", "tail", "tr", "ls", "sort", "cat", "cut",
          "wc", "mktemp", "rm", "mkdir", "chmod", "dirname", "basename", "expr", "readlink",
          "id", "date", "find", "stat", "env", "timeout", "sleep", "sh", "bash", "printf",
          "echo", "test", "true", "false", "touch", "cp", "ln")


def _rocminfo(arches: list[str]) -> str:
    body = ["ROCk module is loaded", "Agent 1",
            "  Name:                    AMD Ryzen 9 9950X", "  Device Type:             CPU"]
    for i, a in enumerate(arches, start = 2):
        body += [f"Agent {i}",
                 f"  Name:                    {a}",
                 "  Marketing Name:          AMD Radeon Graphics",
                 "  Device Type:             GPU",
                 f"  Name:                    {a}"]
    payload = "\\n".join(body)
    return f"#!/bin/sh\nprintf '{payload}\\n'\n"


def _amdsmi(arches: list[str], rocm_version: str = "N/A") -> str:
    lst = "\\n".join(f"GPU: {i}\\n    BDF: 0000:0{i}:00.0" for i in range(len(arches)))
    asic = "\\n".join(
        "ASIC:\\n    MARKET_NAME: AMD Radeon Graphics\\n"
        f"    TARGET_GRAPHICS_VERSION: {a}" for a in arches)
    return (
        "#!/bin/sh\n"
        "case \"${1:-}\" in\n"
        f"  list) printf '{lst}\\n' ;;\n"
        f"  static) printf '{asic}\\n' ;;\n"
        "  *) printf 'AMDSMI Tool: 25.0.1 | AMDSMI Library version: 25.0.1.0 | "
        f"ROCm version: {rocm_version} | amdgpu version: 6.10.10\\n' ;;\n"
        "esac\n"
    )


def _lspci(name: str) -> str:
    return ("#!/bin/sh\nprintf '03:00.0 VGA compatible controller [0300]: Advanced Micro "
            f"Devices, Inc. [AMD/ATI] {name} [1002:7550]\\n'\n")


def _uname(kernel: str, real: str) -> str:
    return (f"#!/bin/sh\nif [ $# -eq 0 ] || [ \"$1\" = \"-s\" ]; then printf '%s\\n' "
            f"'{kernel}'; exit 0; fi\nexec {real} \"$@\"\n")


_NVIDIA_SMI = """#!/bin/sh
for _a in "$@"; do
  case "$_a" in
    -L) echo "GPU 0: NVIDIA B200 (UUID: GPU-deadbeef)"; exit 0 ;;
    --query-gpu=compute_cap) echo "9.0"; exit 0 ;;
  esac
done
printf 'NVIDIA-SMI 580.65.06   Driver Version: 580.65.06   CUDA Version: 12.8\\n'
"""

_HIPCONFIG_SILENT = "#!/bin/sh\nexit 0\n"        # on PATH (Fedora owns /usr/bin/hipconfig)
_HIPCONFIG_64 = "#!/bin/sh\necho '6.4.43483-0'\n"
_RPM_NONE = ("#!/bin/sh\nfor _a in \"$@\"; do case \"$_a\" in -*) ;; *) "
             "printf 'package %s is not installed\\n' \"$_a\" ;; esac; done\nexit 1\n")
_RPM_ROCM_CORE = """#!/bin/sh
_hit=0; _skip=0
for _a in "$@"; do
  if [ "$_skip" = 1 ]; then _skip=0; continue; fi
  case "$_a" in
    --qf|--queryformat) _skip=1 ;;
    -*) ;;
    rocm-core) echo '6.4.1'; _hit=1 ;;
    *) printf 'package %s is not installed\\n' "$_a" ;;
  esac
done
[ "$_hit" = 1 ] || exit 1
"""
_DPKG_ROCM_CORE = ("#!/bin/sh\nprintf 'rocm-core install ok installed 6.4.1-1\\n'\n")


def _fedora(arches: list[str], **kw) -> dict:
    """The reported host: an arch is readable, no version source is."""
    sc = {
        "stubs": {
            "rocminfo": _rocminfo(arches),
            "amd-smi": _amdsmi(arches, "N/A"),
            "hipconfig": _HIPCONFIG_SILENT,
            "rpm": _RPM_NONE,
            # dpkg-query deliberately absent, /opt/rocm/.info/version never created.
        },
        "kfd": True,
        "pci_amd": True,
    }
    sc.update(kw)
    return sc


SCENARIOS: dict[str, dict] = {
    # No stubs, no redirection: this runner's own ROCm stack, through the same block.
    "real_host": {"stubs": {}, "hermetic": False},

    # ---- issue 8731, the host in the report -----------------------------------------
    "fedora_no_version_gfx1201": _fedora(["gfx1201"],
                                         lspci = "Navi 48 [Radeon RX 9070 XT]"),
    # ---- safety controls ------------------------------------------------------------
    "strix_gfx1151_alone": _fedora(["gfx1151"], lspci = "Strix Halo [Radeon 8060S]"),
    "strix_beside_discrete_gfx1201": _fedora(["gfx1151", "gfx1201"],
                                             lspci = "Navi 48 [Radeon RX 9070 XT]"),
    "gfx906_no_version": _fedora(["gfx906"]),
    "fedora_no_version_aarch64": _fedora(["gfx1201"], arch = "aarch64"),
    "fedora_no_version_on_darwin": _fedora(["gfx1201"], uname_s = "Darwin"),
    "nvidia_with_stale_rocm": {
        "stubs": {"nvidia-smi": _NVIDIA_SMI, "hipconfig": _HIPCONFIG_64,
                  "rpm": _RPM_ROCM_CORE, "dpkg-query": _DPKG_ROCM_CORE},
        "nvidia_tree": True,
    },
    "cpu_only_with_stale_rocm": {
        "stubs": {"hipconfig": _HIPCONFIG_64, "rpm": _RPM_ROCM_CORE,
                  "dpkg-query": _DPKG_ROCM_CORE},
    },
}


def _splice_functions(lines: list[str]) -> tuple[str, list[str]]:
    starts: dict[str, int] = {}
    for i, line in enumerate(lines):
        if _FUNC.match(line):
            starts.setdefault(line.split("(")[0], i)
    out: list[str] = []
    found: list[str] = []
    for name in WANTED:
        i = starts.get(name)
        if i is None:
            continue
        for j in range(i, len(lines)):
            if lines[j] == "}":
                out.extend(lines[i:j + 1])
                out.append("")
                found.append(name)
                break
    return "\n".join(out) + "\n", found


def _splice_block(lines: list[str]) -> tuple[str, bool]:
    started = False
    out: list[str] = []
    for line in lines:
        if not started and _BLOCK_START.match(line):
            started = True
        if started:
            out.append(line)
            if line.startswith(_BLOCK_END):
                return "\n".join(out) + "\n", True
    return "\n".join(out) + "\n", False


def _write_stub(d: Path, name: str, body: str) -> None:
    p = d / name
    p.write_text(body, encoding = "utf-8")
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run_scenario(shell: str, funcs_src: str, block_src: str, label: str,
                  spec: dict, timeout: int) -> dict:
    rec: dict = {"scenario": label, "shell": Path(shell).name}
    hermetic = spec.get("hermetic", True)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        env = dict(os.environ)
        for k in ("UNSLOTH_TORCH_INDEX_URL", "UNSLOTH_TORCH_INDEX_FAMILY",
                  "UNSLOTH_ROCM_GFX_ARCH", "UNSLOTH_AMD_ROCM_MIRROR",
                  "UNSLOTH_PYTORCH_MIRROR", "UNSLOTH_TORCH_BACKEND", "ROCM_PATH",
                  "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                  "HSA_OVERRIDE_GFX_VERSION"):
            env.pop(k, None)
        # The spliced block runs its own `mktemp -d` with no cleanup trap.
        tmp = root / "tmp"
        tmp.mkdir()
        env["TMPDIR"] = str(tmp)

        funcs_here, block_here = funcs_src, block_src
        if hermetic:
            binroot = root / "bin"
            toolroot = root / "tools"
            binroot.mkdir()
            toolroot.mkdir()
            for tool in _TOOLS:
                real = shutil.which(tool)
                if real:
                    try:
                        os.symlink(real, toolroot / tool)
                    except OSError:
                        pass
            for name, body in spec.get("stubs", {}).items():
                _write_stub(binroot, name, body)
            if spec.get("lspci"):
                _write_stub(binroot, "lspci", _lspci(spec["lspci"]))
            if spec.get("uname_s"):
                _write_stub(binroot, "uname",
                            _uname(spec["uname_s"], shutil.which("uname") or "/bin/uname"))
            env["PATH"] = f"{binroot}:{toolroot}"

            # Absolute paths cannot be stubbed through PATH, and several of them decide the
            # whole branch on a machine that really is an AMD ROCm box.
            fake = root / "fake"
            targets = {p: str(fake / p.strip("/").replace("/", "-")) for p in _REDIRECT}
            (fake / "opt-rocm").mkdir(parents = True)
            (fake / "proc-driver-nvidia").mkdir(parents = True)
            (fake / "sys-bus-pci-devices").mkdir(parents = True)
            (fake / "proc-cpuinfo").write_text("", encoding = "utf-8")
            (fake / "proc-version").write_text(
                "Linux version 7.1.5 (builder) #1 SMP\n", encoding = "utf-8")
            if spec.get("kfd"):
                (fake / "dev-kfd").write_text("", encoding = "utf-8")
                node = fake / "sys-class-kfd" / "kfd" / "topology" / "nodes" / "1"
                node.mkdir(parents = True)
                (node / "properties").write_text(
                    "cpu_cores_count 0\nsimd_count 128\nvendor_id 4098\n"
                    "device_id 29824\n", encoding = "utf-8")
            if spec.get("pci_amd"):
                dev = fake / "sys-bus-pci-devices" / "0000:03:00.0"
                dev.mkdir(parents = True)
                (dev / "vendor").write_text("0x1002\n", encoding = "utf-8")
                (dev / "class").write_text("0x030000\n", encoding = "utf-8")
            if spec.get("nvidia_tree"):
                (fake / "proc-driver-nvidia" / "gpus" / "0000:01:00.0").mkdir(parents = True)
                _write_stub(fake, "usr-bin-nvidia-smi", _NVIDIA_SMI)
            for real_path, fake_path in targets.items():
                funcs_here = funcs_here.replace(real_path, fake_path)
                block_here = block_here.replace(real_path, fake_path)

        fp = root / "funcs.sh"
        bp = root / "block.sh"
        fp.write_text(funcs_here, encoding = "utf-8")
        bp.write_text(block_here, encoding = "utf-8")

        script = (
            f"_ARCH={spec.get('arch', 'x86_64')}\n"
            "_torch_index_pinned=false\n"
            "SKIP_TORCH=false\n"
            "TORCH_CONSTRAINT=''\nTORCHVISION_CONSTRAINT=''\nTORCHAUDIO_CONSTRAINT=''\n"
            f". '{fp}'\n"
            "printf 'TAG=%s\\n' \"$(_detect_rocm_version_tag 2>/dev/null)\"\n"
            f". '{bp}'\n"
            "printf 'URL=%s\\n' \"$TORCH_INDEX_URL\"\n"
            "printf 'GFX=%s\\n' \"${UNSLOTH_ROCM_GFX_ARCH:-}\"\n"
            "printf 'RADEON=%s\\n' \"${_amd_gpu_radeon:-}\"\n"
            "printf 'CONSTRAINT=%s\\n' \"$TORCH_CONSTRAINT\"\n"
            "printf 'BACKEND=%s\\n' \"${UNSLOTH_TORCH_BACKEND:-}\"\n"
        )
        try:
            r = subprocess.run([shell, "-c", script], capture_output = True, text = True,
                               timeout = timeout, env = env)
        except subprocess.TimeoutExpired:
            rec["error"] = f"timed out after {timeout}s"
            return rec
        rec["rc"] = r.returncode
        fields = {"TAG": "rocm_tag", "URL": "index_url", "GFX": "gfx_arch",
                  "RADEON": "radeon", "CONSTRAINT": "torch_constraint",
                  "BACKEND": "backend"}
        for line in (r.stdout or "").splitlines():
            key, _, val = line.partition("=")
            if key in fields:
                rec[fields[key]] = val.strip()
        rec["stderr_tail"] = (r.stderr or "")[-800:]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--timeout", type = int, default = 240)
    args = ap.parse_args()

    obs: dict = {"state": args.state}
    install_sh = args.checkout / "install.sh"
    if not install_sh.is_file():
        obs["error"] = f"no install.sh at {install_sh}"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    src = install_sh.read_text(encoding = "utf-8", errors = "replace")
    obs["has_agreed_index_family"] = "_amd_agreed_index_family" in src
    obs["has_no_version_reroute"] = "_amd_no_rocm_version_reroute" in src

    lines = src.splitlines()
    funcs_src, found = _splice_functions(lines)
    block_src, reached_end = _splice_block(lines)
    obs["spliced_functions"] = sorted(found)
    obs["missing_functions"] = sorted(set(WANTED) - set(found))
    obs["missing_required"] = sorted(set(REQUIRED) - set(found))
    obs["block_reached_end_marker"] = reached_end
    obs["block_has_index_call"] = "TORCH_INDEX_URL=$(get_torch_index_url)" in block_src
    obs["block_lines"] = len(block_src.splitlines())

    obs["scenarios"] = {}
    shells = [s for s in (shutil.which("bash"), shutil.which("dash")) if s]
    obs["shells"] = [Path(s).name for s in shells]
    for shell in shells:
        for label, spec in SCENARIOS.items():
            obs["scenarios"][f"{Path(shell).name}:{label}"] = _run_scenario(
                shell, funcs_src, block_src, label, spec, args.timeout)

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

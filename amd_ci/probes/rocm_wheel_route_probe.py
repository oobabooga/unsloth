#!/usr/bin/env python3
"""Probe: which torch wheel index does this checkout's install.sh choose?

Drives `get_torch_index_url` from the checkout, once against the runner's own real
ROCm stack and once against each stubbed host layout. The stubs exist because the
layouts these PRs fix -- Debian 13 with a split ROCm stack (PR 8886) and Fedora
packaging where no source reports a version (PR 9152) -- are not this machine, and
faking the probe binaries is the only way to reach them. The real-host reading is the
control that says the stubs did not simply break routing for everyone.

Functions are sourced by splicing EVERY column-0 function out of install.sh, rather
than naming them. The repo's own `tests/sh` harnesses keep a hand-maintained list, and
a helper missing from it fails silently as "cpu" -- which is exactly the failure these
two PRs would produce if mis-tested. Taking all of them removes that failure mode.

Observes only. Whether a given routing is right is the criteria's call.
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

# Column-0 `name() {` ... `}` , the same shape the repo's own tests/sh harnesses assume.
_FUNC = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*\(\) \{$")

# Taking EVERY function is tempting and wrong: install.sh embeds whole scripts as
# heredocs whose payloads contain a bare `}` at column 0, so a naive scan runs past the
# end of a function and drags the installer's top-level code in with it -- which then
# actually tries to install PyTorch. Extract by NAME instead, exactly as the repo's
# harnesses do, and fail loudly when a name is missing rather than silently routing to
# the empty string.
#
# The union across both PRs and their bases. A function absent from a given state is
# recorded in `missing_functions`, not treated as an error: 9152's helpers legitimately
# do not exist at its base.
WANTED = [
    "_run_bounded", "_cvd_hides_nvidia", "_has_amd_rocm_gpu", "_has_usable_nvidia_gpu",
    "_ensure_rocm_probe_env", "_probe_amd_gfx_arch", "_amd_gpu_present_via_pci",
    "_infer_amd_gfx_arch_from_gpu_name", "_infer_linux_amd_gfx_arch",
    "_amd_arch_index_family_for_gfx", "_trim_index_path_slashes",
    "_nvidia_cu126_verdict", "_cap_cuda_family_for_pre_turing",
    "_rocm_tag_from_amd_smi", "_rocm_tag_from_version_file", "_rocm_tag_from_hipconfig",
    "_rocm_tag_from_dpkg", "_rocm_tag_from_rpm", "_highest_rocm_tag",
    "_detect_rocm_version_tag", "get_torch_index_url",
    # added by 9152
    "_amd_probe_arches", "_amd_agreed_index_family", "_amd_sole_index_arch",
    "_rocm_sdk_install_hint",
    # added by 8886
    "_radeon_host_ver_not_older", "get_radeon_wheel_url",
]

# These must exist in EVERY state or the reading is meaningless rather than negative.
REQUIRED = [
    "_run_bounded", "_rocm_tag_from_amd_smi", "_rocm_tag_from_version_file",
    "_rocm_tag_from_hipconfig", "_rocm_tag_from_dpkg", "_rocm_tag_from_rpm",
    "_highest_rocm_tag", "_detect_rocm_version_tag", "get_torch_index_url",
    "_probe_amd_gfx_arch", "_amd_arch_index_family_for_gfx",
]

# Stub scripts. Each stands in for a probe binary install.sh shells out to.
STUBS = {
    "absent": "#!/bin/sh\nexit 127\n",
    "hipconfig_57": "#!/bin/sh\necho '5.7.31921-0'\n",
    "hipconfig_64": "#!/bin/sh\necho '6.4.43483-0'\n",
    "amd_smi_arch_only": "#!/bin/sh\necho 'ASIC: gfx1201'\n",
    "rocminfo_1201": "#!/bin/sh\nprintf 'Name: gfx1201\\nName: gfx1201\\n'\n",
    "rocminfo_1151": "#!/bin/sh\nprintf 'Name: gfx1151\\nName: gfx1151\\n'\n",
    "rocminfo_mixed": "#!/bin/sh\nprintf 'Name: gfx1201\\nName: gfx1036\\n'\n",
    "nvidia_smi_ok": "#!/bin/sh\necho 'GPU 0: NVIDIA GeForce RTX 4090'\n",
}

def _dpkg_stub(table: dict) -> str:
    """A dpkg-query that RENDERS the --showformat it was handed.

    This matters for fairness. The base asks for `${Version}` and the head asks for
    `${Package} ${Status} ${Version}`; a stub that printed one fixed shape would make
    the base look like it found nothing, and the resulting "difference" would be an
    artifact of the harness rather than of the PR. Same approach as the repo's own
    _DPKG_QUERY_STUB. Packages absent from `table` exit non-zero, as dpkg-query does.

    `table` maps package -> (status, version).
    """
    cases = "\n".join(
        f"    {pkg}) _st='{st}'; _ver='{ver}' ;;" for pkg, (st, ver) in table.items()
    )
    return f"""#!/bin/sh
_fmt=''
_pkgs=''
while [ $# -gt 0 ]; do
  case "$1" in
    -W|--show) ;;
    -f=*|--showformat=*) _fmt="${{1#*=}}" ;;
    -f|--showformat) shift; _fmt="$1" ;;
    -*) ;;
    *) _pkgs="$_pkgs $1" ;;
  esac
  shift
done
[ -n "$_fmt" ] || _fmt='${{Package}} ${{Status}} ${{Version}}\\n'
_any=0
for _p in $_pkgs; do
  _st=''; _ver=''
  case "$_p" in
{cases}
  esac
  if [ -z "$_ver" ]; then
    echo "dpkg-query: no packages found matching $_p" >&2
    continue
  fi
  _any=1
  _out=$(printf '%s' "$_fmt" \\
    | sed -e "s/\\${{Package}}/$_p/g" -e "s/\\${{Status}}/$_st/g" -e "s/\\${{Version}}/$_ver/g")
  printf '%b' "$_out"
done
[ "$_any" = 1 ] || exit 1
exit 0
"""


# Debian 13 layout from issue 8402: hipconfig 5.7, HSA runtime 6.1, no rocm-core.
DPKG_DEBIAN_SPLIT = _dpkg_stub({
    "libhsa-runtime64-1": ("install ok installed", "6.1.2-3"),
})
# Ubuntu with AMD's repo: rocm-core 7.2.1 beside Ubuntu's much older HSA package.
# Both readable, and rocm-core must win rather than the two being voted as peers.
DPKG_UBUNTU_BOTH = _dpkg_stub({
    "rocm-core": ("install ok installed", "7.2.1-1"),
    "libhsa-runtime64-1": ("install ok installed", "5.7.1-2build1"),
})
# Removed-but-not-purged must not vote.
DPKG_DECONFIGURED = _dpkg_stub({
    "rocm-core": ("deinstall ok config-files", "7.2.1-1"),
    "libhsa-runtime64-1": ("install ok installed", "6.1.2-3"),
})
DPKG_NONE = "#!/bin/sh\nexit 1\n"
RPM_NONE = "#!/bin/sh\nexit 1\n"
# Fedora: rpm knows the packages but not under the name install.sh used to ask for.
RPM_FEDORA = """#!/bin/sh
printf 'package rocm-core is not installed\\n' >&2
printf '6.4.0\\n'
exit 0
"""

# (label, {binary: stub-body or None-to-omit}, extra env)
SCENARIOS = {
    # The control: nothing stubbed, the runner's own ROCm stack answers.
    "real_host": ({}, {}),
    # PR 8886, issue 8402.
    "debian_split_stack": ({
        "hipconfig": STUBS["hipconfig_57"],
        "dpkg-query": DPKG_DEBIAN_SPLIT,
        "rpm": RPM_NONE,
        "amd-smi": STUBS["absent"],
        "rocminfo": STUBS["rocminfo_1151"],
    }, {}),
    # PR 8886's own regression risk: rocm-core must outrank the distro HSA package.
    "ubuntu_rocm_core_and_hsa": ({
        "hipconfig": STUBS["absent"],
        "dpkg-query": DPKG_UBUNTU_BOTH,
        "rpm": RPM_NONE,
        "amd-smi": STUBS["absent"],
        "rocminfo": STUBS["rocminfo_1151"],
    }, {}),
    "dpkg_deconfigured_does_not_vote": ({
        "hipconfig": STUBS["absent"],
        "dpkg-query": DPKG_DECONFIGURED,
        "rpm": RPM_NONE,
        "amd-smi": STUBS["absent"],
        "rocminfo": STUBS["rocminfo_1151"],
    }, {}),
    # PR 9152, issue 8731: arch readable, no version anywhere.
    "fedora_no_version_gfx1201": ({
        "hipconfig": STUBS["absent"],
        "dpkg-query": DPKG_NONE,
        "rpm": RPM_NONE,
        "amd-smi": STUBS["amd_smi_arch_only"],
        "rocminfo": STUBS["rocminfo_1201"],
    }, {}),
    # An APU beside a discrete card must NOT route on whichever enumerated first.
    "mixed_arch_no_version": ({
        "hipconfig": STUBS["absent"],
        "dpkg-query": DPKG_NONE,
        "rpm": RPM_NONE,
        "amd-smi": STUBS["amd_smi_arch_only"],
        "rocminfo": STUBS["rocminfo_mixed"],
    }, {}),
    # The two that must never move: a working NVIDIA box and a CPU-only box, each
    # carrying stale ROCm packaging. Neither may be routed to AMD wheels.
    "nvidia_with_stale_rocm": ({
        "nvidia-smi": STUBS["nvidia_smi_ok"],
        "hipconfig": STUBS["hipconfig_64"],
        "dpkg-query": DPKG_UBUNTU_BOTH,
        "rpm": RPM_NONE,
        "amd-smi": STUBS["absent"],
        "rocminfo": STUBS["absent"],
    }, {}),
    "cpu_only_with_stale_rocm": ({
        "nvidia-smi": STUBS["absent"],
        "hipconfig": STUBS["hipconfig_64"],
        "dpkg-query": DPKG_UBUNTU_BOTH,
        "rpm": RPM_NONE,
        "amd-smi": STUBS["absent"],
        "rocminfo": STUBS["absent"],
    }, {}),
}


def _splice(install_sh: Path) -> str:
    """Every column-0 function definition, in order. No hand-maintained name list."""
    lines = install_sh.read_text(encoding = "utf-8", errors = "replace").splitlines()
    starts = {}
    for i, line in enumerate(lines):
        if _FUNC.match(line):
            starts.setdefault(line.split("(")[0], i)

    out, found = [], []
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


def _write_stub(d: Path, name: str, body: str) -> None:
    p = d / name
    p.write_text(body, encoding = "utf-8")
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run_scenario(shell: str, funcs: Path, label: str, stubs: dict,
                  extra_env: dict, timeout: int) -> dict:
    rec: dict = {"scenario": label, "shell": Path(shell).name}
    with tempfile.TemporaryDirectory() as td:
        binroot = Path(td) / "bin"
        binroot.mkdir()
        funcs_here = funcs
        env = dict(os.environ)
        for k in ("UNSLOTH_TORCH_INDEX_URL", "UNSLOTH_TORCH_INDEX_FAMILY",
                  "UNSLOTH_ROCM_GFX_ARCH", "ROCM_PATH",
                  "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
            env.pop(k, None)

        if stubs:
            # A hermetic PATH, so nothing the runner happens to have installed can
            # answer a probe this scenario meant to stub out.
            for tool in ("uname", "grep", "sed", "head", "sh", "bash", "cat", "awk",
                         "tr", "sort", "cut", "timeout", "sleep", "printf", "expr",
                         "mktemp", "rm", "dirname", "basename", "command", "ls"):
                real = shutil.which(tool)
                if real:
                    try:
                        os.symlink(real, binroot / tool)
                    except OSError:
                        pass
            for name, body in stubs.items():
                _write_stub(binroot, name, body)
            env["PATH"] = str(binroot)

            # Absolute paths cannot be stubbed through PATH, and two of them decide
            # the whole branch. `/proc/driver/nvidia/gpus` is the NVIDIA fallback
            # detector: on any host that really has an NVIDIA card -- like the box
            # these scenarios were authored on -- it returns first and every AMD
            # scenario silently measures the CUDA path instead. `/opt/rocm` is the
            # version file, which must miss for the layouts these PRs target.
            # Redirected the same way the repo's tests/sh harnesses do.
            fake = Path(td) / "fake"
            (fake / "rocm").mkdir(parents = True)
            src = funcs.read_text(encoding = "utf-8")
            src = src.replace("/usr/bin/nvidia-smi", str(fake / "nvidia-smi-absent"))
            src = src.replace("/proc/driver/nvidia", str(fake / "proc-nvidia"))
            src = src.replace("/opt/rocm", str(fake / "rocm"))
            src = src.replace("/dev/kfd", str(fake / "kfd-absent"))
            funcs_here = Path(td) / "funcs_scoped.sh"
            funcs_here.write_text(src, encoding = "utf-8")
        env.update(extra_env)

        script = (
            f". '{funcs_here}'\n"
            "printf 'TAG=%s\\n' \"$(_detect_rocm_version_tag 2>/dev/null)\"\n"
            "printf 'URL=%s\\n' \"$(get_torch_index_url 2>/dev/null)\"\n"
        )
        try:
            r = subprocess.run([shell, "-c", script], capture_output = True,
                               text = True, timeout = timeout, env = env)
        except subprocess.TimeoutExpired:
            rec["error"] = f"timed out after {timeout}s"
            return rec
        rec["rc"] = r.returncode
        for line in (r.stdout or "").splitlines():
            if line.startswith("TAG="):
                rec["rocm_tag"] = line[4:].strip()
            elif line.startswith("URL="):
                rec["index_url"] = line[4:].strip()
        rec["stderr_tail"] = (r.stderr or "")[-800:]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--timeout", type = int, default = 180)
    args = ap.parse_args()

    obs: dict = {"state": args.state}
    install_sh = args.checkout / "install.sh"
    if not install_sh.is_file():
        obs["error"] = f"no install.sh at {install_sh}"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    src = install_sh.read_text(encoding = "utf-8", errors = "replace")
    # Which helpers each PR adds, recorded so the criteria can tell the states apart
    # without guessing from behaviour.
    obs["has_hsa_runtime_source"] = "libhsa-runtime64-1" in src
    obs["has_agreed_index_family"] = "_amd_agreed_index_family" in src
    obs["has_sdk_install_hint"] = "_rocm_sdk_install_hint" in src

    with tempfile.TemporaryDirectory() as td:
        funcs = Path(td) / "funcs.sh"
        body, found = _splice(install_sh)
        funcs.write_text(body, encoding = "utf-8")
        obs["spliced_functions"] = sorted(found)
        obs["missing_functions"] = sorted(set(WANTED) - set(found))
        # The failure this guard exists for is silent: a helper that install.sh calls
        # but the splice never defined makes the ROCm branch die and the whole thing
        # resolve to "cpu", which looks like a routing decision rather than a broken
        # harness. The criteria refuses to score a state that trips this.
        obs["missing_required"] = sorted(set(REQUIRED) - set(found))

        obs["scenarios"] = {}
        # dash as well as bash: install.sh is #!/bin/sh and POSIX by contract, and a
        # bashism would only show up under a real POSIX shell.
        shells = [s for s in (shutil.which("bash"), shutil.which("dash")) if s]
        obs["shells"] = [Path(s).name for s in shells]
        for shell in shells:
            for label, (stubs, extra) in SCENARIOS.items():
                key = f"{Path(shell).name}:{label}"
                obs["scenarios"][key] = _run_scenario(
                    shell, funcs, label, stubs, extra, args.timeout
                )

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Feed a real Windows host's GPU identity through THIS checkout's AMD routing.

PR 9672 rewrites AMD architecture detection and ROCm wheel-index routing. On
Linux that lives in install.sh. The Windows equivalents are:

  * install.ps1                      -- $nameArchTable (adapter name -> gfx),
                                        $unsupportedNameArchTable (#8529 wording)
                                        and $archFamilyMap (gfx -> repo.amd.com leaf)
  * studio/setup.ps1                 -- Test-RocmGfx211Leaf / Test-PipRocmFamilyLeaf /
                                        Test-CudaFamilyLeaf / Test-RocmKnown211Version
  * studio/install_python_stack.py   -- the CROSS-PLATFORM half the PR actually edits:
                                        _GENERIC_WHEEL_GFX_MIN_ROCM, the per-arch
                                        generic-wheel floor, and _detect_windows_gfx_arch

Every unit test for those is a stub. This probe lifts the tables and helpers out
of whichever checkout the state names and runs them against the machine it is on:
the real Win32_VideoController adapter name, the real amd-smi output, the real
Windows arch detection. Base and head can then be compared on one real box.

It OBSERVES and never judges. No pass/fail decision is made here; see
criteria/windows_amd_routing_no_regression.py.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Adapter names fed through the checkout's name -> gfx table alongside whatever
# this box really reports. Fixed, so base and head see identical input, and
# chosen to cover every arch the PR's floor table names plus the RDNA 1 / Polaris
# parts the unsupported table exists to word a message for.
SYNTHETIC_ADAPTERS = [
    "AMD Radeon(TM) 8060S Graphics",
    "AMD Radeon RX 7600",
    "AMD Radeon RX 7700S",
    "AMD Radeon RX 7900 XTX",
    "AMD Radeon RX 9060 XT",
    "AMD Radeon RX 9070 XT",
    "AMD Radeon 890M",
    "AMD Radeon 860M",
    "AMD Radeon RX 6700 XT",
    "AMD Radeon RX 5700 XT",
    "AMD Radeon RX 580",
    "NVIDIA GeForce RTX 4090",
]

# Leaves fed through setup.ps1's leaf classifiers.
LEAVES = [
    "cpu", "cu128", "cu126", "xpu", "rocm6.1", "rocm6.3", "rocm6.4", "rocm7.0",
    "gfx1151", "gfx1150", "gfx1152", "gfx120x-all", "gfx110X-all", "gfx103X-all",
]

# (gfx, ROCm version) pairs put through the Python floor logic. gfx1102 /
# gfx1200 / gfx1201 are the arches the PR's floor table names; the rest are
# controls that must not move.
FLOOR_ARCHES = [
    "gfx1102", "gfx1200", "gfx1201", "gfx1151", "gfx1150", "gfx1152",
    "gfx1100", "gfx1101", "gfx1103", "gfx1030", "gfx950", "gfx906", "gfx1010",
]
FLOOR_VERSIONS = [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (7, 0), (7, 1), (7, 2)]

PS_HELPERS = (
    "Test-RocmGfx211Leaf",
    "Test-RocmKnown211Version",
    "Test-CudaFamilyLeaf",
    "Test-PipRocmFamilyLeaf",
    "Get-TorchIndexLeaf",
    "Trim-IndexPathSlashes",
)


# --------------------------------------------------------------------------- #
# text extraction


def _read(path: Path) -> str:
    """Name the encoding. Path.read_text() is cp1252 on Windows, and a UTF-8
    source read that way is mojibake, which has already been mistaken for a
    defect once on these runners."""
    return path.read_text(encoding = "utf-8", errors = "replace")


def _balanced(source: str, marker: str, opener: str, closer: str) -> str:
    """The text of the literal that starts at `marker`, by bracket matching."""
    start = source.find(marker)
    if start < 0:
        return ""
    i = source.find(opener, start)
    if i < 0:
        return ""
    depth = 0
    j = i
    while j < len(source):
        ch = source[j]
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return source[start : j + 1]
        j += 1
    return ""


def _ps_function(source: str, name: str) -> str:
    """A PowerShell function definition, by brace matching. Empty when absent."""
    m = re.search(r"^function\s+" + re.escape(name) + r"\b", source, re.M)
    if not m:
        return ""
    start = m.start()
    i = source.find("{", start)
    if i < 0:
        return ""
    depth, j = 0, i
    while j < len(source):
        if source[j] == "{":
            depth += 1
        elif source[j] == "}":
            depth -= 1
            if depth == 0:
                return source[start : j + 1]
        j += 1
    return ""


# --------------------------------------------------------------------------- #
# PowerShell execution


def _powershell() -> str | None:
    return shutil.which("powershell") or shutil.which("powershell.exe")


def _run_ps(script: str, timeout: int = 180) -> dict:
    """Run a PowerShell script that writes its JSON result to $ResultPath.

    The result travels through a FILE, never stdout: a profile banner or a
    progress record concatenated into the JSON has cost a run before. Written
    without a BOM by the script, read back with utf-8-sig regardless.
    """
    exe = _powershell()
    if not exe:
        return {"powershell_absent": True}
    tmpdir = Path(tempfile.mkdtemp(prefix = "amdci_ps_"))
    ps1 = tmpdir / "probe.ps1"
    res = tmpdir / "result.json"
    prologue = (
        "$ErrorActionPreference = 'Continue'\n"
        "$ProgressPreference = 'SilentlyContinue'\n"
        f"$ResultPath = '{res}'\n"
        "function Write-Result($obj) {\n"
        "  $json = $obj | ConvertTo-Json -Depth 8 -Compress\n"
        "  [System.IO.File]::WriteAllText($ResultPath, $json,"
        " (New-Object System.Text.UTF8Encoding($false)))\n"
        "}\n"
    )
    ps1.write_text(prologue + script, encoding = "utf-8")
    out: dict = {}
    try:
        p = subprocess.run(
            [exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(ps1)],
            capture_output = True, text = True, timeout = timeout)
        out["rc"] = p.returncode
        out["stdout_tail"] = (p.stdout or "")[-1500:]
        out["stderr_tail"] = (p.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        out["timeout"] = True
        return out
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    if res.is_file():
        try:
            out["result"] = json.loads(res.read_text(encoding = "utf-8-sig"))
        except Exception as e:  # noqa: BLE001
            out["parse_error"] = f"{type(e).__name__}: {e}"
    else:
        out["no_result_file"] = True
    return out


def _json_literal(value) -> str:
    """A PowerShell string literal for a value, via a here-string-free encoding."""
    return "'" + str(value).replace("'", "''") + "'"


# --------------------------------------------------------------------------- #
# observations


def host_facts() -> dict:
    facts: dict = {
        "sys_platform": sys.platform,
        "python": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "node": platform.node(),
        "computername": os.environ.get("COMPUTERNAME", ""),
        "runner_os_env": os.environ.get("RUNNER_OS", ""),
        "powershell_path": _powershell() or "",
        "amd_smi_path": shutil.which("amd-smi") or shutil.which("amd-smi.exe") or "",
        "bash_path": shutil.which("bash") or "",
    }
    ps = _run_ps(
        "$v = $PSVersionTable\n"
        "$os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue\n"
        "$vc = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |\n"
        "        ForEach-Object { [pscustomobject]@{ Name = $_.Name;"
        " DriverVersion = $_.DriverVersion; PNPDeviceID = $_.PNPDeviceID;"
        " Status = $_.Status } })\n"
        "Write-Result ([pscustomobject]@{\n"
        "  PSVersion = [string]$v.PSVersion; PSEdition = [string]$v.PSEdition;\n"
        "  CLRVersion = [string]$v.CLRVersion;\n"
        "  OSCaption = [string]$os.Caption; OSBuild = [string]$os.BuildNumber;\n"
        "  ComputerName = $env:COMPUTERNAME;\n"
        "  VideoControllers = $vc\n"
        "})\n")
    facts["powershell"] = ps
    return facts


def amd_smi_facts() -> dict:
    exe = shutil.which("amd-smi") or shutil.which("amd-smi.exe")
    out: dict = {"present": bool(exe), "path": exe or ""}
    if not exe:
        return out
    for key, argv in (("list", ["list"]),
                      ("list_e", ["list", "-e"]),
                      ("static_asic", ["static", "--asic"])):
        try:
            p = subprocess.run([exe, *argv], capture_output = True, text = True,
                               timeout = 120)
            out[key] = {"rc": p.returncode, "stdout": (p.stdout or "")[:8000],
                        "stderr_tail": (p.stderr or "")[-500:]}
        except subprocess.TimeoutExpired:
            out[key] = {"timeout": True}
        except Exception as e:  # noqa: BLE001
            out[key] = {"error": f"{type(e).__name__}: {e}"}
    # The gfx tokens install.ps1 scrapes out of amd-smi, applied to the real text.
    tokens: dict = {}
    for key in ("list", "list_e", "static_asic"):
        text = ((out.get(key) or {}).get("stdout") or "")
        tokens[key] = sorted({t.lower() for t in re.findall(r"\b(gfx\d+[a-z]?)\b", text, re.I)})
    out["gfx_tokens"] = tokens
    return out


def install_ps1_routing(checkout: Path, real_names: list[str]) -> dict:
    """Run the checkout's adapter-name -> gfx -> index-family tables for real."""
    path = checkout / "install.ps1"
    obs: dict = {"present": path.is_file()}
    if not path.is_file():
        return obs
    source = _read(path)
    name_table = _balanced(source, "$nameArchTable = @(", "(", ")")
    unsup_table = _balanced(source, "$unsupportedNameArchTable = @(", "(", ")")
    family_map = _balanced(source, "$archFamilyMap = @{", "{", "}")
    obs["tables_found"] = {
        "nameArchTable": bool(name_table),
        "unsupportedNameArchTable": bool(unsup_table),
        "archFamilyMap": bool(family_map),
    }
    # Recorded verbatim so a diff in the DECISIONS can be traced to a diff in the
    # table rather than to this probe.
    obs["nameArchTable_sha"] = _sha(name_table)
    obs["unsupportedNameArchTable_sha"] = _sha(unsup_table)
    obs["archFamilyMap_sha"] = _sha(family_map)
    obs["nameArchTable_rows"] = re.findall(r'A\s*=\s*"(gfx[0-9a-z]+)"', name_table)
    obs["archFamilyMap_pairs"] = dict(
        re.findall(r'"(gfx[0-9a-z]+)"\s*=\s*"([^"]+)"', family_map))
    if not (name_table and unsup_table and family_map):
        obs["note"] = "a table could not be lifted out of install.ps1 at this state"
        return obs

    names = list(dict.fromkeys(list(real_names) + SYNTHETIC_ADAPTERS))
    names_ps = "@(" + ",".join(_json_literal(n) for n in names) + ")"
    script = (
        name_table + "\n" + unsup_table + "\n" + family_map + "\n"
        + f"$names = {names_ps}\n"
        + "$rows = @()\n"
        "foreach ($label in $names) {\n"
        "  $arch = $null; $pat = $null\n"
        "  foreach ($row in $nameArchTable) {\n"
        "    if ($label -match $row.P) { $arch = $row.A; $pat = $row.P; break }\n"
        "  }\n"
        "  $unsup = $null\n"
        "  foreach ($row in $unsupportedNameArchTable) {\n"
        "    if ($label -match $row.P) { $unsup = $row.A; break }\n"
        "  }\n"
        "  $fam = $null\n"
        "  if ($arch -and $archFamilyMap.ContainsKey($arch)) { $fam = $archFamilyMap[$arch] }\n"
        "  $rows += [pscustomobject]@{\n"
        "    name = $label; arch = $arch; matched = $pat; unsupported = $unsup;\n"
        "    family = $fam; hasGpuWheels = [bool]$fam\n"
        "  }\n"
        "}\n"
        "Write-Result ([pscustomobject]@{ rows = $rows })\n"
    )
    run = _run_ps(script)
    obs["run"] = {k: v for k, v in run.items() if k != "result"}
    rows = ((run.get("result") or {}).get("rows")) or []
    if isinstance(rows, dict):
        rows = [rows]
    obs["decisions"] = {
        r.get("name"): {
            "arch": r.get("arch"),
            "matched": r.get("matched"),
            "unsupported": r.get("unsupported"),
            "family": r.get("family"),
            "has_gpu_wheels": bool(r.get("hasGpuWheels")),
        }
        for r in rows if isinstance(r, dict)
    }
    return obs


def setup_ps1_helpers(checkout: Path) -> dict:
    path = checkout / "studio" / "setup.ps1"
    obs: dict = {"present": path.is_file()}
    if not path.is_file():
        return obs
    source = _read(path)
    bodies = {name: _ps_function(source, name) for name in PS_HELPERS}
    obs["helpers_present"] = {k: bool(v) for k, v in bodies.items()}
    obs["helpers_sha"] = {k: _sha(v) for k, v in bodies.items()}
    usable = [b for b in bodies.values() if b]
    if not usable:
        obs["note"] = "none of the leaf helpers could be lifted out of setup.ps1"
        return obs
    leaves_ps = "@(" + ",".join(_json_literal(v) for v in LEAVES) + ")"
    script = (
        "\n".join(usable) + "\n"
        + f"$leaves = {leaves_ps}\n"
        + "$rows = @()\n"
        "foreach ($leaf in $leaves) {\n"
        "  $row = @{ leaf = $leaf }\n"
        "  foreach ($fn in @('Test-RocmGfx211Leaf','Test-CudaFamilyLeaf',"
        "'Test-PipRocmFamilyLeaf')) {\n"
        "    if (Get-Command $fn -ErrorAction SilentlyContinue) {\n"
        "      try { $row[$fn] = [string]([bool](& $fn $leaf)) }\n"
        "      catch { $row[$fn] = 'THREW: ' + $_.Exception.Message }\n"
        "    } else { $row[$fn] = 'ABSENT' }\n"
        "  }\n"
        "  $rows += [pscustomobject]$row\n"
        "}\n"
        "Write-Result ([pscustomobject]@{ rows = $rows })\n"
    )
    run = _run_ps(script)
    obs["run"] = {k: v for k, v in run.items() if k != "result"}
    rows = ((run.get("result") or {}).get("rows")) or []
    if isinstance(rows, dict):
        rows = [rows]
    obs["leaf_classification"] = {
        r.get("leaf"): {k: v for k, v in r.items() if k != "leaf"}
        for r in rows if isinstance(r, dict)
    }
    return obs


_PY_SNIPPET = r'''
import importlib.util, json, os, platform, sys
from pathlib import Path
out = {}
src = Path(sys.argv[1])
dest = Path(sys.argv[2])
try:
    spec = importlib.util.spec_from_file_location("amdci_ips", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["amdci_ips"] = mod
    spec.loader.exec_module(mod)
    out["imported"] = True
except BaseException as e:
    out["imported"] = False
    out["import_error"] = "%s: %s" % (type(e).__name__, e)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    raise SystemExit(0)

def g(name):
    return getattr(mod, name, None)

table = g("_GENERIC_WHEEL_GFX_MIN_ROCM") or {}
out["generic_wheel_floor"] = {k: list(v) for k, v in table.items()}
out["rocm_torch_index_tags"] = sorted(["%d.%d" % k for k in (g("_ROCM_TORCH_INDEX") or {})])
amd_index = g("_GFX_TO_AMD_INDEX_ARCH") or {}
out["gfx_to_amd_index"] = {k: amd_index.get(k) for k in ARCHES if k in amd_index}
out["generic_rocm_wheel_gfx"] = sorted(g("_GENERIC_ROCM_WHEEL_GFX") or [])

def call(fn, *a):
    f = g(fn)
    if f is None:
        return "ABSENT"
    try:
        return f(*a)
    except BaseException as e:
        return "THREW: %s: %s" % (type(e).__name__, e)

decisions = {}
for gfx in ARCHES:
    per = {
        "has_wheel_route": call("_gfx_has_a_wheel_route", gfx),
        "route_on_host_single": call("_gfx_route_on_host", gfx, [gfx]),
        "route_on_host_mixed": call("_gfx_route_on_host", gfx, [gfx, "gfx1151"]),
        "tag_lacks_kernels": {},
        "reroute_lacks_kernels": {},
        "generic_only_below_floor": {},
    }
    for ver in VERSIONS:
        key = "%d.%d" % tuple(ver)
        per["tag_lacks_kernels"][key] = call("_generic_tag_lacks_kernels", gfx, tuple(ver))
        per["reroute_lacks_kernels"][key] = call("_generic_rocm_wheel_lacks_kernels", gfx, tuple(ver))
        per["generic_only_below_floor"][key] = call("_generic_only_target_below_floor", gfx, tuple(ver))
    per["reroute_no_version"] = call("_generic_rocm_wheel_lacks_kernels", gfx, None)
    per["generic_only_below_floor_no_version"] = call("_generic_only_target_below_floor", gfx, None)
    decisions[gfx] = per
out["decisions"] = decisions

# The real host, through the checkout's own Windows detection.
out["detect_windows_gfx_arch"] = call("_detect_windows_gfx_arch")
out["detect_amd_gfx_codes"] = call("_detect_amd_gfx_codes")
out["is_windows_flag"] = bool(g("IS_WINDOWS"))
out["sys_platform"] = sys.platform
dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
'''


def python_stack_decisions(checkout: Path) -> dict:
    """Import THIS checkout's install_python_stack and record what it decides.

    In a subprocess: the module mutates os.environ on Windows and pulls in the
    Studio backend package, and an import that dies must produce an observation
    rather than kill the probe.
    """
    src = checkout / "studio" / "install_python_stack.py"
    obs: dict = {"present": src.is_file()}
    if not src.is_file():
        return obs
    tmpdir = Path(tempfile.mkdtemp(prefix = "amdci_py_"))
    runner = tmpdir / "run.py"
    dest = tmpdir / "out.json"
    header = (
        "ARCHES = " + json.dumps(FLOOR_ARCHES) + "\n"
        "VERSIONS = " + json.dumps([list(v) for v in FLOOR_VERSIONS]) + "\n"
    )
    runner.write_text(header + _PY_SNIPPET, encoding = "utf-8")
    try:
        p = subprocess.run([sys.executable, str(runner), str(src), str(dest)],
                           capture_output = True, text = True, timeout = 600)
        obs["rc"] = p.returncode
        obs["stdout_tail"] = (p.stdout or "")[-1500:]
        obs["stderr_tail"] = (p.stderr or "")[-2500:]
    except subprocess.TimeoutExpired:
        obs["timeout"] = True
        return obs
    if dest.is_file():
        try:
            obs.update(json.loads(dest.read_text(encoding = "utf-8")))
        except Exception as e:  # noqa: BLE001
            obs["parse_error"] = f"{type(e).__name__}: {e}"
    else:
        obs["no_output_file"] = True
    return obs


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "checkout": str(args.checkout)}
    try:
        obs["host"] = host_facts()
        real_names = [
            str(v.get("Name") or "")
            for v in _controllers(obs["host"])
            if str(v.get("Name") or "").strip()
        ]
        obs["real_adapter_names"] = real_names
        obs["amd_smi"] = amd_smi_facts()
        obs["install_ps1"] = install_ps1_routing(args.checkout, real_names)
        obs["setup_ps1"] = setup_ps1_helpers(args.checkout)
        obs["python_stack"] = python_stack_decisions(args.checkout)
    except Exception as e:  # noqa: BLE001
        import traceback
        obs["probe_error"] = f"{type(e).__name__}: {e}"
        obs["probe_traceback"] = traceback.format_exc()[-3000:]
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    # A probe reports; it does not decide. Always 0.
    return 0


def _controllers(host: dict) -> list:
    result = ((host.get("powershell") or {}).get("result")) or {}
    vc = result.get("VideoControllers")
    if vc is None:
        return []
    if isinstance(vc, dict):
        return [vc]
    return [v for v in vc if isinstance(v, dict)]


if __name__ == "__main__":
    sys.exit(main())

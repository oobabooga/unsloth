#!/usr/bin/env python3
"""Probe: what the installer's NVIDIA driver-library inventory reports with ZLUDA installed.

Observes only. For each ZLUDA build under --zluda-root (one subdirectory per build, holding
the libraries at its top level) it puts that directory where a ZLUDA user has it (Windows:
PATH; Linux: LD_LIBRARY_PATH) and reads:
  ps  Get-NvidiaLibraryInventory from the state's studio/setup.ps1 (Windows PowerShell 5.1 on
      Windows; pwsh on Linux when installed)
  py  studio/nvidia_probe.py --json, the Studio backend's twin
plus a control run with nothing added. Each read is a fresh process: the emitted P/Invoke type
and the loaded DLL live for the life of the process.
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

INV_PS1 = r"""
param([string]$Setup)
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Setup, [ref]$null, [ref]$null)
$fns = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Parent.Parent -eq $ast }, $false)
foreach ($f in $fns) { Invoke-Expression $f.Extent.Text }
$inv = Get-NvidiaLibraryInventory
if ($null -eq $inv) { "RESULT null" } else { "RESULT " + ($inv | ConvertTo-Json -Compress) }
"""

WINDOWS = sys.platform == "win32"


def _env(extra_dir: str | None) -> dict:
    env = dict(os.environ)
    env.pop("UNSLOTH_NVIDIA_LIBRARY_PROBE", None)
    if extra_dir:
        key = "PATH" if WINDOWS else "LD_LIBRARY_PATH"
        env[key] = extra_dir + os.pathsep + env.get(key, "")
    return env


def _run(cmd: list[str], env: dict) -> tuple[int | None, str, str]:
    try:
        p = subprocess.run(cmd, env = env, capture_output = True, text = True,
                           encoding = "utf-8", errors = "replace", timeout = 120)
        return p.returncode, p.stdout or "", p.stderr or ""
    except Exception as exc:  # a hang or a missing shell is an observation, not a crash
        return None, "", repr(exc)


def read_ps(shell: str, script: Path, setup: Path, env: dict) -> dict:
    cmd = [shell, "-NoProfile"]
    if WINDOWS:
        cmd += ["-ExecutionPolicy", "Bypass"]
    rc, out, err = _run(cmd + ["-File", str(script), "-Setup", str(setup)], env)
    lines = [l.strip() for l in out.splitlines() if l.strip().startswith("RESULT ")]
    if not lines:
        return {"ran": False, "rc": rc, "stderr": err[-600:], "stdout": out[-600:]}
    body = lines[-1][len("RESULT "):]
    return {"ran": True, "rc": rc, "inventory": None if body == "null" else json.loads(body)}


def read_py(checkout: Path, env: dict) -> dict:
    rc, out, err = _run([sys.executable, str(checkout / "studio" / "nvidia_probe.py"), "--json"], env)
    try:
        payload = json.loads(out.strip().splitlines()[-1]) if out.strip() else None
    except ValueError:
        return {"ran": False, "rc": rc, "stderr": err[-600:], "stdout": out[-600:]}
    if payload is not None:
        payload = {"source": payload.get("source"), "cuda_driver_version": payload.get("cuda_driver_version"),
                   "devices": [d.get("name") for d in payload.get("devices") or []]}
    return {"ran": rc in (0, 1), "rc": rc, "inventory": payload}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--zluda-root", required = True, type = Path)
    args = ap.parse_args()

    checkout = Path(args.checkout)
    setup = checkout / "studio" / "setup.ps1"
    obs: dict = {"state": args.state, "platform": sys.platform, "scenarios": {}}
    if WINDOWS:
        sysroot = os.environ.get("SystemRoot", r"C:\Windows")
        obs["system32_nvcuda"] = os.path.exists(os.path.join(sysroot, "System32", "nvcuda.dll"))
        obs["system32_nvml"] = os.path.exists(os.path.join(sysroot, "System32", "nvml.dll"))
        shell = "powershell"
    else:
        shell = shutil.which("pwsh")
    obs["ps_shell"] = shell

    builds = sorted(p for p in args.zluda_root.iterdir() if p.is_dir()) if args.zluda_root.is_dir() else []
    obs["builds"] = {p.name: sorted(f.name for f in p.iterdir() if f.is_file())[:40] for p in builds}
    scenarios = [("control", None)] + [(p.name, str(p)) for p in builds]

    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "inv.ps1"
        script.write_text(INV_PS1, encoding = "utf-8")
        for name, d in scenarios:
            env = _env(d)
            row: dict = {"dir": d}
            row["ps"] = read_ps(shell, script, setup, env) if shell else {"ran": False, "note": "no PowerShell"}
            row["py"] = read_py(checkout, env)
            obs["scenarios"][name] = row

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

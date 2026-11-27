#!/usr/bin/env python3
"""Probe: does each Windows installer detection entry point spawn amd-smi, and what does it detect?
A logging amd-smi.bat leads PATH (never forwards to the real one, which can raise UAC); each scenario
runs in a fresh child so memoized probes start cold."""

from __future__ import annotations

import argparse, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

_CHILD = r'''
import importlib.util, json, os, sys
co = sys.argv[1]
def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(co, rel))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
out = {}
def call(key, fn):
    try:
        v = fn(); out[key] = list(v) if isinstance(v, tuple) else v
    except Exception as e:
        out[key] = "ERR " + type(e).__name__ + ": " + str(e)[:200]
try:
    st = load("install_python_stack", "studio/install_python_stack.py")
    call("stack.amd_smi_allowed", st._amd_smi_allowed)
    call("stack.has_rocm_gpu", st._has_rocm_gpu)
    call("stack.windows_gfx_arch", st._detect_windows_gfx_arch)
    call("stack.rocm_version", st._detect_rocm_version_uncached)
    call("stack.amd_gfx_codes", st._detect_amd_gfx_codes)
except Exception as e:
    out["stack_load"] = "ERR " + repr(e)[:300]
try:
    pb = load("install_llama_prebuilt", "studio/install_llama_prebuilt.py")
    call("prebuilt.amd_smi_allowed", pb._amd_smi_allowed)
    call("prebuilt.host_rocm_version", pb._detect_host_rocm_version)
    def _host():
        h = pb.detect_host()
        return {k: getattr(h, k) for k in ("has_rocm", "rocm_gfx_target", "has_usable_nvidia") if hasattr(h, k)}
    call("prebuilt.detect_host", _host)
except Exception as e:
    out["prebuilt_load"] = "ERR " + repr(e)[:300]
print("PROBE_JSON " + json.dumps(out, default=str))
'''


def scenario(checkout: str, name: str, shim: Path, env_extra: dict, hide_hip: bool) -> dict:
    log = shim / f"amd_smi_calls_{name}.log"
    log.unlink(missing_ok = True)
    env = dict(os.environ)
    path = env.get("PATH", "")
    if hide_hip:
        for v in ("HIP_PATH", "HIP_PATH_57", "ROCM_PATH"):
            env.pop(v, None)
        path = os.pathsep.join(p for p in path.split(os.pathsep)
                               if not (p.strip('"') and os.path.isfile(os.path.join(p.strip('"'), "hipinfo.exe"))))
    env["PATH"] = str(shim) + os.pathsep + path
    env["AMD_SMI_SHIM_LOG"] = str(log)
    env.update(env_extra)
    r = subprocess.run([sys.executable, "-I", "-c", _CHILD, checkout], capture_output = True, text = True,
                       encoding = "utf-8", errors = "replace", env = env, timeout = 600)
    res = {}
    for line in r.stdout.splitlines():
        if line.startswith("PROBE_JSON "):
            res = json.loads(line[len("PROBE_JSON "):])
    calls = log.read_text(encoding = "utf-8", errors = "replace").splitlines() if log.exists() else []
    return {"results": res, "amd_smi_calls": [c.strip() for c in calls if c.strip()],
            "rc": r.returncode, "stderr_tail": r.stderr[-1500:],
            "hipinfo_on_path": shutil.which("hipinfo", path = env["PATH"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    a = ap.parse_args()
    shim = Path(tempfile.mkdtemp(prefix = "amdsmi_shim_"))
    (shim / "amd-smi.bat").write_text('@echo off\r\necho %* >> "%AMD_SMI_SHIM_LOG%"\r\nexit /b 1\r\n', encoding = "ascii")
    obs = {"os": sys.platform, "real_amd_smi": shutil.which("amd-smi"), "real_hipinfo": shutil.which("hipinfo"),
           "HIP_PATH": os.environ.get("HIP_PATH"), "shim_resolves": shutil.which("amd-smi", path = str(shim))}
    sc = {}
    sc["default"] = scenario(a.checkout, "default", shim, {}, hide_hip = False)
    sc["opt_in"] = scenario(a.checkout, "opt_in", shim, {"UNSLOTH_ENABLE_AMD_SMI": "1"}, hide_hip = False)
    sc["opt_in_no_hipinfo"] = scenario(a.checkout, "opt_in_no_hipinfo", shim, {"UNSLOTH_ENABLE_AMD_SMI": "1"}, hide_hip = True)
    obs["scenarios"] = sc
    a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

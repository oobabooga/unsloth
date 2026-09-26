#!/usr/bin/env python3
"""Probe: which torch index and constraints does this checkout route THIS host to?

Observes only, on the host's REAL detection: real rocminfo / hipconfig / amd-smi on
PATH, the real /opt/rocm, real KFD sysfs. Nothing is faked and nothing is installed.
The comparison between states is criteria/rocm_routing_same.py's job.

Two observations per checkout:

  install_sh   install.sh's top-level functions plus the source span from the torch
               ceiling through "fi  # _torch_index_pinned guard" (the index choice and
               the Radeon / Strix / RDNA 4 reroutes), lifted into a standalone script and
               run in its own bash. Prints the resolved TORCH_INDEX_URL, leaf and the three
               constraints. _maybe_bootstrap_rocm_wsl is replaced by a no-op because it can
               install a ROCm userland on WSL; everything else is the checkout's own code.

  python_stack studio/install_python_stack.py imported from the checkout. Its detection
               helpers are called as-is, and _ensure_rocm_torch() is run with ONLY the
               installers (pip_install, pip_install_try, the bnb URL fetch and provenance
               write) replaced by recorders, so its routing decision is captured as the
               pip call it would have made. Run twice: against the venv's real installed
               torch, and against a "no ROCm torch installed yet" state (the fresh-install
               case, where the routing decision always emits an index). Only the installed
               TORCH is counterfactual there; hardware detection is never touched.

Writes JSON to --out, never stdout (import banners corrupt stdout).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

SPAN_START = "_TORCH_CEILING="
SPAN_END = "fi  # _torch_index_pinned guard"
FN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\(\)")
HEREDOC = re.compile(r"<<(-?)\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")
MARK = "__AMD_CI_ROUTING__"
SH_FIELDS = (
    "TORCH_INDEX_URL", "_torch_index_leaf", "TORCH_CONSTRAINT", "TORCHVISION_CONSTRAINT",
    "TORCHAUDIO_CONSTRAINT", "_torch_index_pinned", "_amd_gpu_radeon", "_gfx_rocm64_target",
    "_runtime_gfx", "_gfx906_env", "_rdna4_gfx",
)


def _run(cmd: list[str], timeout: int = 30) -> dict:
    exe = shutil.which(cmd[0])
    if exe is None:
        return {"present": False}
    try:
        p = subprocess.run(cmd, capture_output = True, text = True, timeout = timeout)
        return {"present": True, "path": exe, "rc": p.returncode,
                "stdout_tail": (p.stdout or "")[-1500:], "stderr_tail": (p.stderr or "")[-500:]}
    except Exception as e:  # noqa: BLE001
        return {"present": True, "path": exe, "error": f"{type(e).__name__}: {e}"}


def host_evidence() -> dict:
    """What the detection sources on this host say, recorded raw for the report."""
    ev: dict = {"machine": platform.machine(), "system": platform.system(),
                "env": {k: os.environ.get(k) for k in (
                    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                    "HSA_OVERRIDE_GFX_VERSION", "UNSLOTH_ROCM_GFX_ARCH", "UNSLOTH_TORCH_BACKEND",
                    "UNSLOTH_TORCH_INDEX_URL", "UNSLOTH_TORCH_INDEX_FAMILY",
                    "UNSLOTH_AMD_ROCM_MIRROR")}}
    ri = _run(["rocminfo"])
    if ri.get("present"):
        try:
            out = subprocess.run(["rocminfo"], capture_output = True, text = True,
                                 timeout = 30).stdout or ""
            ri["gfx"] = sorted(set(re.findall(r"\bgfx[0-9a-f]+\b", out)))
        except Exception as e:  # noqa: BLE001
            ri["gfx_error"] = str(e)
        ri.pop("stdout_tail", None)
    ev["rocminfo"] = ri
    ev["hipconfig_version"] = _run(["hipconfig", "--version"])
    ev["amd_smi_present"] = shutil.which("amd-smi") is not None
    ver = Path("/opt/rocm/.info/version")
    ev["opt_rocm"] = {"exists": Path("/opt/rocm").exists(),
                      "version_file": ver.read_text(encoding = "utf-8").strip()
                      if ver.is_file() else None}
    kfd = Path("/sys/class/kfd/kfd/topology/nodes")
    targets = []
    if kfd.is_dir():
        for node in sorted(kfd.iterdir()):
            props = node / "properties"
            if props.is_file():
                m = re.search(r"gfx_target_version\s+(\d+)",
                              props.read_text(encoding = "utf-8", errors = "replace"))
                if m and m.group(1) != "0":
                    targets.append(m.group(1))
    ev["kfd_gfx_target_versions"] = targets
    return ev


# ---------------------------------------------------------------- install.sh side
def lift_install_sh(checkout: Path, dest: Path) -> Path:
    lines = (checkout / "install.sh").read_text(encoding = "utf-8").splitlines()

    def skip_heredoc(k: int) -> int:
        m = HEREDOC.search(lines[k]) if "<<" in lines[k] and "<<<" not in lines[k] else None
        if not m:
            return k + 1
        j = k + 1
        while (lines[j].strip() if m.group(1) else lines[j]) != m.group(2):
            j += 1
        return j + 1

    funcs, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if FN.match(ln) and not ln.startswith("_unsloth_main()"):
            if ln.rstrip().endswith("}") and ln.count("{") <= ln.count("}"):
                funcs.append(ln)
                i += 1
                continue
            j = i
            while lines[j] != "}":
                j = skip_heredoc(j)
            funcs.extend(lines[i:j + 1])
            i = j + 1
            continue
        i = skip_heredoc(i)
    s = next(k for k, l in enumerate(lines) if l.startswith(SPAN_START))
    e = next(k for k, l in enumerate(lines) if k > s and l.startswith(SPAN_END))
    # The WSL bootstrap installs a ROCm userland; this probe installs nothing.
    no_install = "_maybe_bootstrap_rocm_wsl() { return 0; }\n"
    body = "\n".join(funcs) + "\n" + no_install + "\n".join(lines[s:e + 1]) + "\n"
    dest.write_text(body, encoding = "utf-8")
    return dest


def probe_install_sh(checkout: Path, work: Path) -> dict:
    out: dict = {}
    try:
        lifted = lift_install_sh(checkout, work / "lifted_install.sh")
    except Exception as e:  # noqa: BLE001
        return {"error": f"lift failed: {type(e).__name__}: {e}"}
    syn = subprocess.run(["bash", "-n", str(lifted)], capture_output = True, text = True)
    if syn.returncode != 0:
        return {"error": f"bash -n failed: {syn.stderr[-500:]}"}
    osname = "linux" if platform.system() == "Linux" else platform.system().lower()
    prints = "".join(f'printf "%s=%s\\n" "{f}" "${{{f}:-}}"\n' for f in SH_FIELDS)
    script = (
        "set -euo pipefail\n"
        f"OS={osname}\n_ARCH={platform.machine()}\nSKIP_TORCH=false\n"
        f"VENV_DIR='{work}/novenv'\n"
        f". '{lifted}'\n"
        f"echo {MARK}\n" + prints)
    env = dict(os.environ)
    env["TMPDIR"] = str(work)
    p = subprocess.run(["bash", "-c", script], env = env, capture_output = True, text = True,
                       timeout = 300)
    out["rc"] = p.returncode
    out["stderr_tail"] = (p.stderr or "")[-3000:]
    stdout = p.stdout or ""
    if MARK not in stdout:
        out["error"] = f"no result marker (rc={p.returncode})"
        out["stdout_tail"] = stdout[-1500:]
        return out
    fields = {}
    for line in stdout.split(MARK, 1)[1].splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k] = v
    out["fields"] = fields
    return out


# ------------------------------------------------------------- python stack side
def load_stack(checkout: Path):
    path = checkout / "studio" / "install_python_stack.py"
    spec = importlib.util.spec_from_file_location("amd_ci_install_python_stack", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def _safe(fn, *a):
    try:
        v = fn(*a)
        return list(v) if isinstance(v, tuple) else v
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def run_ensure(mod, counterfactual_no_rocm_torch: bool) -> dict:
    calls: list[dict] = []

    def rec(kind):
        def _f(label, *args, **kw):
            calls.append({"kind": kind, "label": str(label), "args": [str(a) for a in args],
                          "constrain": kw.get("constrain")})
            return True
        return _f

    patches = {
        "pip_install": rec("pip_install"),
        "pip_install_try": rec("pip_install_try"),
        "_bnb_rocm_install_is_current": lambda *a, **k: True,
        "_record_bnb_rocm_provenance": lambda *a, **k: None,
        "_bnb_asset_identity": lambda *a, **k: None,
    }
    if counterfactual_no_rocm_torch:
        patches.update({
            "_probe_torch_runtime": lambda: (True, True, "2.10.0+cpu", "", ""),
            "_installed_rocm_wheel_family": lambda: None,
            "_torch_requires_rocm_sdk": lambda: False,
            "_installed_generic_rocm_tag": lambda: None,
        })
    saved = {k: getattr(mod, k) for k in patches if hasattr(mod, k)}
    saved_env = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    buf = io.StringIO()
    res: dict = {}
    try:
        for k, v in patches.items():
            if hasattr(mod, k):
                setattr(mod, k, v)
        for k in ("_invalidate_rocm_version_probe",):
            if hasattr(mod, k):
                getattr(mod, k)()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            mod._ensure_rocm_torch()
    except BaseException as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-2000:]
    finally:
        for k, v in saved.items():
            setattr(mod, k, v)
        if saved_env is not None:
            os.environ["HSA_OVERRIDE_GFX_VERSION"] = saved_env
    res["calls"] = calls
    res["torch_calls"] = [c for c in calls if "torch" in c["label"].lower()]
    idx = None
    for c in res["torch_calls"]:
        if "--index-url" in c["args"]:
            idx = c["args"][c["args"].index("--index-url") + 1]
    res["torch_index_url"] = idx
    res["log_tail"] = buf.getvalue()[-3000:]
    return res


def probe_python_stack(checkout: Path) -> dict:
    out: dict = {}
    try:
        mod = load_stack(checkout)
    except BaseException as e:  # noqa: BLE001
        return {"error": f"import failed: {type(e).__name__}: {e}"}
    out["torch_backend_global"] = getattr(mod, "_TORCH_BACKEND", None)
    ver = _safe(mod._detect_rocm_version)
    out["rocm_version"] = ver
    verq = tuple(ver) if isinstance(ver, list) else (0, 0)
    out["has_rocm_gpu"] = _safe(mod._has_rocm_gpu)
    out["has_usable_nvidia_gpu"] = _safe(mod._has_usable_nvidia_gpu)
    out["inferred_linux_gfx"] = _safe(mod._infer_linux_amd_gfx_arch)
    rt = _safe(mod._runtime_gfx_target, None)
    out["runtime_gfx_target"] = rt
    runtime_gfx = rt[0] if isinstance(rt, list) and rt else None
    out["runtime_gfx"] = runtime_gfx
    out["strix_needs_amd_arch_index"] = _safe(mod._strix_needs_amd_arch_index, verq)
    out["amd_arch_index_url"] = _safe(mod._amd_arch_index_url, runtime_gfx)
    floor_set = getattr(mod, "_AMD_ARCH_INDEX_FLOOR_GFX", None)
    out["floor_set_name"] = "_AMD_ARCH_INDEX_FLOOR_GFX" if floor_set is not None \
        else "_HSA_SPOOFABLE_PHYSICAL_GFX"
    floor_set = floor_set if floor_set is not None else getattr(mod, "_HSA_SPOOFABLE_PHYSICAL_GFX", ())
    out["runtime_gfx_in_floor_set"] = runtime_gfx in floor_set
    ran, imp, tver, hip, cuda = mod._probe_torch_runtime()
    out["installed_torch"] = {"ran": ran, "importable": imp, "version": tver, "hip": hip}
    inst = (tver or "").lower() if (ran and imp) else ""
    out["reroute_pending_installed"] = _safe(mod._rocm_compat_reroute_pending,
                                             runtime_gfx, verq, inst)
    out["reroute_pending_fresh"] = _safe(mod._rocm_compat_reroute_pending,
                                         runtime_gfx, verq, "")
    out["ensure_installed"] = run_ensure(mod, counterfactual_no_rocm_torch = False)
    out["ensure_fresh"] = run_ensure(mod, counterfactual_no_rocm_torch = True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    checkout = args.checkout.resolve()

    obs: dict = {"state": args.state, "checkout": str(checkout)}
    try:
        obs["commit"] = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                       capture_output = True, text = True).stdout.strip()
    except Exception:  # noqa: BLE001
        obs["commit"] = None
    obs["host"] = host_evidence()
    with tempfile.TemporaryDirectory(prefix = f"routing_{args.state}_",
                                     dir = os.environ.get("RUNNER_TEMP") or None) as d:
        try:
            obs["install_sh"] = probe_install_sh(checkout, Path(d))
        except Exception as e:  # noqa: BLE001
            obs["install_sh"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        obs["python_stack"] = probe_python_stack(checkout)
    except BaseException as e:  # noqa: BLE001
        obs["python_stack"] = {"error": f"{type(e).__name__}: {e}",
                               "traceback": traceback.format_exc()[-2000:]}
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

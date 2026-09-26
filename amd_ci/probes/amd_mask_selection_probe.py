#!/usr/bin/env python3
"""Probe: which AMD gfx target does this checkout pick for THIS host, per visibility mask?

Observes only, on the host's REAL detection: real rocminfo and amd-smi on PATH, the
real ROCr runtime applying the mask to rocminfo. Nothing is installed; the comparison
between states is criteria/amd_mask_selection_same.py's job.

Two observations per (arm, mask cell), each in its own subprocess so no mask leaks:

  setup_sh   studio/setup.sh's AMD detection-through-selection block, lifted the way
             tests/sh/test_setup_gpu_summary_probe_sources.sh lifts it: the probe
             helpers, the real initialiser group (_setup_amd_detected=false through
             _setup_amd_records=""), and the awk range from the detection `if` up to the
             UNSLOTH_ROCM_GFX_ARCH override, closed with `fi`. NVIDIA is pinned false
             by the initialiser group. Records _setup_gfx, _setup_mkt, _setup_amd_probe
             (head only; the variable does not exist at the base) and
             _setup_amd_detected.

  detect     studio/install_llama_prebuilt.py imported from the checkout, detect_host()
             called as-is (no network is touched by it). Records has_rocm,
             rocm_gfx_target, rocm_gfx_targets.

Arms:
  rocminfo   the real PATH.
  amdsmi     the real PATH with a `rocminfo` that exits 1 put in front of it, so both
             consumers fall through to the REAL amd-smi (the arm whose ROCr survivor
             filter the PR changes). Only rocminfo is hidden; amd-smi is the host's.

Writes JSON to --out, never stdout (import banners corrupt stdout).
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
import traceback
from pathlib import Path

MASK_VARS = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
# Variables that would short-circuit or steer selection independently of the mask.
SCRUB = MASK_VARS + ("UNSLOTH_ROCM_GFX_ARCH",)

# (cell id, {var: value}). An absent key means UNSET; "" means set-but-empty.
CELLS = (
    ("none", {}),
    ("ROCR=0", {"ROCR_VISIBLE_DEVICES": "0"}),
    ("HIP=0", {"HIP_VISIBLE_DEVICES": "0"}),
    ("CUDA=0", {"CUDA_VISIBLE_DEVICES": "0"}),
    ("ROCR=0,0", {"ROCR_VISIBLE_DEVICES": "0,0"}),
    ("ROCR=0,99", {"ROCR_VISIBLE_DEVICES": "0,99"}),
    ("HIP=''", {"HIP_VISIBLE_DEVICES": ""}),
    ("ROCR=-1", {"ROCR_VISIBLE_DEVICES": "-1"}),
)
ARMS = ("rocminfo", "amdsmi")

MARK = "__AMD_CI_SETUP__"
GUARD = r"""command_not_found_handle() {
    printf '%s\n' "FATAL: the extracted block called '$1', which the lift did not pull in" >&2
    exit 127
}
"""
HELPERS = ("_setup_run_smi", "_setup_rocminfo_gpu_records", "_setup_amd_smi_gpu_records",
           "_setup_amd_smi_hip_order", "_amd_gfx_is_shadowing_integrated",
           "_amd_prefer_discrete_gfx")


# ------------------------------------------------------------------ lift setup.sh
def _function(lines: list[str], name: str) -> list[str]:
    """sed -n '/^name()/,/^}/p', as the shell test does."""
    for i, ln in enumerate(lines):
        if ln.startswith(f"{name}()"):
            for j in range(i, len(lines)):
                if lines[j].startswith("}"):
                    return lines[i:j + 1]
            raise ValueError(f"{name}() has no closing brace")
    raise ValueError(f"{name}() not found")


def lift_setup_sh(checkout: Path, dest: Path) -> dict:
    lines = (checkout / "studio" / "setup.sh").read_text(encoding = "utf-8").splitlines()
    body: list[str] = [GUARD]
    # The printers are display-only; route them to stderr so the block can reach them.
    body.append("step() { printf 'step: %s %s\\n' \"$1\" \"$2\" >&2; }")
    body.append("substep() { printf 'substep: %s\\n' \"$1\" >&2; }")
    for h in HELPERS:
        body.extend(_function(lines, h))
    s = next(i for i, l in enumerate(lines) if l == "_setup_amd_detected=false")
    e = next(i for i, l in enumerate(lines) if i > s and l == '_setup_amd_records=""')
    init = lines[s:e + 1]
    body.extend(init)
    start = next(i for i, l in enumerate(lines)
                 if l.startswith('if [ "$_setup_nvidia_usable" != true ]; then'))
    stop = next(i for i, l in enumerate(lines)
                if i > start and "UNSLOTH_ROCM_GFX_ARCH env override" in l)
    body.extend(lines[start:stop])
    body.append("fi")
    text = "\n".join(body) + "\n"
    dest.write_text(text, encoding = "utf-8")
    return {"path": str(dest), "init_lines": len(init), "block_lines": stop - start,
            "nvidia_pinned_false": "_setup_nvidia_usable=false" in init,
            "has_amd_probe_var": "_setup_amd_probe" in text}


def run_setup_cell(block: Path, env: dict) -> dict:
    fields = ("_setup_gfx", "_setup_mkt", "_setup_amd_probe", "_setup_amd_detected",
              "_setup_gfx_all")
    prints = "".join(f'printf "%s=%s\\n" "{f}" "${{{f}-}}"\n' for f in fields)
    # setup.sh runs under `set -euo pipefail`; match it.
    script = f"set -euo pipefail\n. '{block}'\necho {MARK}\n" + prints
    try:
        p = subprocess.run(["bash", "-c", script], env = env, capture_output = True,
                           text = True, timeout = 180)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    res: dict = {"rc": p.returncode, "stderr_tail": (p.stderr or "")[-800:]}
    out = p.stdout or ""
    if MARK not in out:
        res["error"] = f"no result marker (rc={p.returncode})"
        return res
    for line in out.split(MARK, 1)[1].splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            res[k.lstrip("_").replace("setup_", "")] = v.replace("\n", ",")
    # _setup_gfx_all is multi-line; re-read it cleanly.
    res["gfx_all"] = ",".join(x for x in out.split(MARK, 1)[1]
                               .split("_setup_gfx_all=", 1)[-1].splitlines() if x)
    return res


# ---------------------------------------------------- install_llama_prebuilt side
def detect_worker(checkout: Path, out: Path) -> int:
    """Runs in its own process, under the cell's env. Imports the checkout's module."""
    import contextlib
    import importlib.util
    import io
    res: dict = {}
    buf = io.StringIO()
    try:
        path = checkout / "studio" / "install_llama_prebuilt.py"
        spec = importlib.util.spec_from_file_location("amd_ci_install_llama_prebuilt", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            spec.loader.exec_module(mod)
            host = mod.detect_host()
        res = {"has_rocm": host.has_rocm, "rocm_gfx_target": host.rocm_gfx_target,
               "rocm_gfx_targets": list(host.rocm_gfx_targets),
               "has_usable_nvidia": host.has_usable_nvidia}
    except BaseException as e:  # noqa: BLE001
        res = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-1500:]}
    res["log_tail"] = buf.getvalue()[-800:]
    out.write_text(json.dumps(res, default = str), encoding = "utf-8")
    return 0


def run_detect_cell(checkout: Path, env: dict, work: Path, tag: str) -> dict:
    out = work / f"detect_{tag}.json"
    try:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--detect-worker",
                            "--checkout", str(checkout), "--out", str(out), "--state", "-"],
                           env = env, capture_output = True, text = True, timeout = 180)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    if not out.is_file():
        return {"error": f"worker wrote nothing (rc={p.returncode})",
                "stderr_tail": (p.stderr or "")[-800:]}
    return json.loads(out.read_text(encoding = "utf-8"))


# -------------------------------------------------------------- host evidence
def _cap(cmd: list[str], env: dict) -> dict:
    exe = shutil.which(cmd[0], path = env.get("PATH"))
    if exe is None:
        return {"present": False}
    try:
        p = subprocess.run([exe, *cmd[1:]], env = env, capture_output = True, text = True,
                           timeout = 60)
        return {"present": True, "path": exe, "rc": p.returncode, "stdout": p.stdout or ""}
    except Exception as e:  # noqa: BLE001
        return {"present": True, "path": exe, "error": f"{type(e).__name__}: {e}"}


def rocminfo_view(env: dict) -> dict:
    """What the real ROCr runtime lets rocminfo see under this env."""
    r = _cap(["rocminfo"], env)
    if "stdout" in r:
        out = r.pop("stdout")
        r["gpu_names"] = re.findall(r"^\s*Name:\s*(gfx[1-9][0-9a-z]{2,3})\s*$", out, re.M)
    return r


def base_env(arm: str, stub_dir: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in SCRUB}
    if arm == "amdsmi":
        env["PATH"] = f"{stub_dir}{os.pathsep}{env.get('PATH', '')}"
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--detect-worker", action = "store_true", help = argparse.SUPPRESS)
    args = ap.parse_args()
    checkout = args.checkout.resolve()
    if args.detect_worker:
        return detect_worker(checkout, args.out)

    obs: dict = {"state": args.state, "checkout": str(checkout)}
    try:
        obs["commit"] = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                       capture_output = True, text = True).stdout.strip()
    except Exception:  # noqa: BLE001
        obs["commit"] = None
    obs["host"] = {"system": platform.system(), "machine": platform.machine(),
                   "scrubbed_env": {k: os.environ.get(k) for k in SCRUB},
                   "HSA_OVERRIDE_GFX_VERSION": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
                   "amd_smi_present": shutil.which("amd-smi") is not None,
                   "rocminfo_present": shutil.which("rocminfo") is not None,
                   "nvidia_smi_present": shutil.which("nvidia-smi") is not None}

    with tempfile.TemporaryDirectory(prefix = f"masksel_{args.state}_",
                                     dir = os.environ.get("RUNNER_TEMP") or None) as d:
        work = Path(d)
        stub_dir = work / "hide_rocminfo"
        stub_dir.mkdir()
        stub = stub_dir / "rocminfo"
        stub.write_text("#!/bin/sh\nexit 1\n", encoding = "utf-8")
        stub.chmod(0o755)

        smi = _cap(["amd-smi", "list"], base_env("rocminfo", stub_dir))
        if "stdout" in smi:
            smi["stdout_tail"] = smi.pop("stdout")[-1200:]
        obs["host"]["amd_smi_list"] = smi

        try:
            obs["lift"] = lift_setup_sh(checkout, work / "setup_block.sh")
            syn = subprocess.run(["bash", "-n", obs["lift"]["path"]], capture_output = True,
                                 text = True)
            obs["lift"]["bash_n_rc"] = syn.returncode
            if syn.returncode != 0:
                obs["lift"]["error"] = f"bash -n failed: {syn.stderr[-500:]}"
        except Exception as e:  # noqa: BLE001
            obs["lift"] = {"error": f"lift failed: {type(e).__name__}: {e}"}

        cells: dict = {}
        for arm in ARMS:
            for cid, masks in CELLS:
                env = base_env(arm, stub_dir)
                env.update(masks)
                env["TMPDIR"] = str(work)
                key = f"{arm}|{cid}"
                c: dict = {"arm": arm, "cell": cid, "env": masks}
                if arm == "rocminfo":
                    c["rocminfo_view"] = rocminfo_view(env)
                if not obs["lift"].get("error"):
                    try:
                        c["setup_sh"] = run_setup_cell(Path(obs["lift"]["path"]), env)
                    except Exception as e:  # noqa: BLE001
                        c["setup_sh"] = {"error": f"{type(e).__name__}: {e}"}
                else:
                    c["setup_sh"] = {"error": "lift failed"}
                tag = re.sub(r"[^A-Za-z0-9]+", "_", key)
                try:
                    c["detect"] = run_detect_cell(checkout, env, work, tag)
                except Exception as e:  # noqa: BLE001
                    c["detect"] = {"error": f"{type(e).__name__}: {e}"}
                cells[key] = c
        obs["cells"] = cells
        if isinstance(obs.get("lift"), dict):
            obs["lift"].pop("path", None)

    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build comfy-kitchen's HIP extension for gfx1151 twice, at pinned commits, each in its own source tree:
  k338   PR #338 head (da1ebe9154, dual-M INT8 GEMM)
  kmain  its merge base on main (7d6a22ae82), the "generic" kernel #338 is measured against
Each tree is then an importable package root (comfy_kitchen/backends/hip/_C*.so built in place). Build tools
(nanobind, cmake, ninja) are pip-installed into a separate --target dir, never into the Studio venv.
Every step has a hard timeout; a failure is recorded, never raised. Writes one JSON via --out.

  kern_build.py --dest DIR --out F [--only k338,kmain] [--timeout 2700]"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

REPO = "https://github.com/Comfy-Org/comfy-kitchen"
SHAS = {"k338": "da1ebe9154e5473d91640b5aecffb40207bb1143", "kmain": "7d6a22ae8281cd1053a8449530fa0e764a2e51cb"}


def sh(cmd, timeout, cwd = None, env = None, log = None):
    t = time.time()
    try:
        r = subprocess.run(cmd, cwd = cwd, env = env, capture_output = True, text = True, timeout = timeout,
                           encoding = "utf-8", errors = "replace")
        rc, outp = r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as exc:
        rc, outp = "timeout", str(exc)[-2000:]
    except Exception as exc:  # noqa: BLE001
        rc, outp = "error", repr(exc)
    if log:
        with open(log, "a", encoding = "utf-8") as fh:
            fh.write(f"\n=== {' '.join(map(str, cmd))} rc={rc}\n{outp[-20000:]}\n")
    return {"rc": rc, "s": round(time.time() - t, 1), "tail": outp[-1500:]}


def pick_root(py):
    """First ROCm root that has both the HIP CMake config and a clang driver: the venv's rocm-sdk (torch's own
    runtime) first, then /opt/rocm."""
    cands = []
    try:
        r = subprocess.run([py, "-m", "rocm_sdk", "path", "--root"], capture_output = True, text = True, timeout = 60)
        if r.returncode == 0 and r.stdout.strip():
            cands.append(r.stdout.strip())
    except Exception:  # noqa: BLE001
        pass
    for s in [p for p in sys.path if p.endswith("site-packages")]:
        cands += [os.path.join(s, "_rocm_sdk_devel"), os.path.join(s, "_rocm_sdk_core")]
    cands += sorted(glob.glob("/opt/rocm*"), key = lambda p: (p != "/opt/rocm", p))
    rows = []
    for c in cands:
        if not os.path.isdir(c):
            continue
        cm = os.path.isfile(os.path.join(c, "lib/cmake/hip/hip-config.cmake"))
        cl = next((p for p in ("lib/llvm/bin/clang++", "bin/amdclang++", "bin/hipcc") if os.path.isfile(os.path.join(c, p))),
                  None)
        rows.append({"root": c, "hip_cmake": cm, "clang": cl})
        if cm and cl:
            return c, rows
    return None, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--only", default = "k338,kmain")
    ap.add_argument("--timeout", type = int, default = 2700)
    a = ap.parse_args()
    obs: dict = {"t0": time.time(), "shas": SHAS, "builds": {}}

    def save():
        open(a.out, "w", encoding = "utf-8").write(json.dumps(obs, default = str))
    dest = os.path.abspath(a.dest)
    os.makedirs(dest, exist_ok = True)
    log = os.path.join(os.path.dirname(os.path.abspath(a.out)), "kern_build.log")
    src = os.path.join(dest, "src")
    try:
        if not os.path.isdir(os.path.join(src, ".git")):
            obs["clone"] = sh(["git", "clone", "-q", "--filter=blob:none", REPO, src], 900, log = log)
        obs["fetch"] = sh(["git", "-C", src, "fetch", "-q", "origin", "pull/338/head:pr338"], 600, log = log)
        obs["pr338_head"] = sh(["git", "-C", src, "rev-parse", "pr338"], 60)["tail"].strip()
        tools = os.path.join(dest, "buildtools")
        pkgs = ["nanobind>=2.0.0", "cmake>=3.26", "ninja", "setuptools>=61", "wheel"]
        obs["tools"] = sh([sys.executable, "-m", "pip", "install", "-q", "--target", tools, *pkgs], 900, log = log)
        if obs["tools"]["rc"] != 0 and shutil.which("uv"):  # uv-made venvs may have no pip
            obs["tools_uv"] = sh(["uv", "pip", "install", "-q", "--python", sys.executable, "--target", tools, *pkgs],
                                 900, log = log)
        root, rows = pick_root(sys.executable)
        obs["rocm_root"], obs["rocm_candidates"] = root, rows
        save()
        if root is None:
            obs["error"] = "no ROCm root with hip-config.cmake and a clang driver"
            save()
            return 0
        env = {k: v for k, v in os.environ.items() if "TOKEN" not in k.upper()}
        env.update(ROCM_HOME = root, ROCM_PATH = root, COMFY_KITCHEN_BUILD_HIP = "1", COMFY_HIP_ARCHS = "gfx1151",
                   CMAKE_BUILD_PARALLEL_LEVEL = str(os.cpu_count() or 8),
                   PYTHONPATH = tools + os.pathsep + env.get("PYTHONPATH", ""),
                   PATH = os.path.join(tools, "bin") + os.pathsep + env.get("PATH", ""))
        for name in a.only.split(","):
            tree = os.path.join(dest, name)
            b: dict = {"sha": SHAS[name], "tree": tree}
            obs["builds"][name] = b
            if not os.path.isdir(tree):
                b["worktree"] = sh(["git", "-C", src, "worktree", "add", "-f", "--detach", tree, SHAS[name]], 600, log = log)
            b["head"] = sh(["git", "-C", tree, "rev-parse", "HEAD"], 60)["tail"].strip()
            b["build"] = sh([sys.executable, "setup.py", "build_ext", "--inplace", "--hip-archs=gfx1151"], a.timeout,
                            cwd = tree, env = env, log = log)
            sos = glob.glob(os.path.join(tree, "comfy_kitchen", "backends", "hip", "_C*.so"))
            b["so"] = sos[0] if sos else None
            b["ok"] = bool(sos) and b["build"]["rc"] == 0 and b["head"] == SHAS[name]
            save()
    except Exception as exc:  # noqa: BLE001
        import traceback
        obs["error"] = traceback.format_exc()[-1500:]
    obs["seconds"] = round(time.time() - obs["t0"], 1)
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())

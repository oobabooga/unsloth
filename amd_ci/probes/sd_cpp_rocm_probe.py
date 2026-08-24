#!/usr/bin/env python3
"""Probe: install the real ROCm sd.cpp asset with this state's extractor, then ask the loader.

Observes only. The defect in #9268 is not a tree shape, it is `sd-cli` refusing to start
because `zipfile.extractall` wrote each symlink member's target text as a regular file. Only
the ROCm asset ships symlink members, and only an AMD host has the ROCm libraries needed for
`sd-cli` to get past them, so this reading is meaningful here and nowhere else.

Three consecutive installs into the same root, because `install()` merges rather than wipes:
a fix that works once and destroys the tree on the second pass is not a fix.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ASSET = "sd-master-bfbef5b-bin-Linux-Ubuntu-24.04-x86_64-rocm-7.14.0.zip"
URL = f"https://github.com/leejet/stable-diffusion.cpp/releases/download/master-813-bfbef5b/{ASSET}"


def load_installer(checkout: str):
    path = Path(checkout) / "studio" / "install_sd_cpp_prebuilt.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("sdmod_probe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sdmod_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def fetch(dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return
    req = urllib.request.Request(URL, headers = {"User-Agent": "unsloth-sd-cpp-installer"})
    with urllib.request.urlopen(req, timeout = 600) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def find_cli(root: Path):
    for p in root.rglob("sd-cli"):
        if p.is_file():
            return p
    return None


def inspect(root: Path, archive: Path) -> dict:
    with zipfile.ZipFile(archive) as zf:
        want = sum(1 for i in zf.infolist()
                   if i.create_system in (3, 19) and stat.S_ISLNK(i.external_attr >> 16))
    restored, flattened, dangling = 0, 0, 0
    for p in root.rglob("*"):
        if p.is_symlink():
            restored += 1
            if not p.exists():
                dangling += 1
        elif p.is_file() and ".so" in p.name and p.stat().st_size < 64:
            flattened += 1
    out = {"archive_symlinks": want, "restored": restored,
           "flattened": flattened, "dangling": dangling}

    exe = find_cli(root)
    if exe is None:
        out["error"] = "no sd-cli in the extracted tree"
        return out
    exe.chmod(0o755)
    env = {**os.environ, "LD_LIBRARY_PATH": str(exe.parent)}
    ldd = subprocess.run(["ldd", str(exe)], capture_output = True, text = True, env = env)
    blob = (ldd.stdout or "") + (ldd.stderr or "")
    out["file_too_short"] = "file too short" in blob
    out["not_found"] = blob.count("not found")
    try:
        run = subprocess.run([str(exe), "--help"], capture_output = True, text = True,
                             env = env, timeout = 300)
        out["rc"] = run.returncode
        err = (run.stderr or "").strip().splitlines()
        out["stderr0"] = err[0][:200] if err else ""
    except subprocess.TimeoutExpired:
        out["rc"] = None
        out["stderr0"] = "sd-cli --help timed out"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--passes", type = int, default = 3)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "asset": ASSET, "passes": args.passes}
    mod = load_installer(args.checkout)
    if mod is None:
        obs["error"] = "no studio/install_sd_cpp_prebuilt.py at this state"
        args.out.write_text(json.dumps(obs, indent = 2))
        return 0

    scratch = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "sdcpp_rocm_probe"
    scratch.mkdir(parents = True, exist_ok = True)
    archive = scratch / ASSET
    try:
        fetch(archive)
    except Exception as exc:  # noqa: BLE001 -- a download failure is an absent reading
        obs["error"] = f"download failed: {type(exc).__name__}: {exc}"
        args.out.write_text(json.dumps(obs, indent = 2))
        return 0

    root = scratch / f"install_{args.state}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents = True)

    per_pass = []
    for n in range(1, args.passes + 1):
        try:
            with zipfile.ZipFile(archive) as zf:
                mod._safe_extractall(zf, root)
            raised = None
        except Exception as exc:  # noqa: BLE001 -- a refusal is an observation, not a crash
            raised = f"{type(exc).__name__}: {exc}"
        rec = {"pass": n, "raised": raised}
        rec.update(inspect(root, archive))
        per_pass.append(rec)

    obs["per_pass"] = per_pass
    obs.update({k: v for k, v in per_pass[-1].items() if k != "pass"})
    args.out.write_text(json.dumps(obs, indent = 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

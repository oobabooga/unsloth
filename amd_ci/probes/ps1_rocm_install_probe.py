#!/usr/bin/env python3
"""Probe: what torch Studio's install.ps1 leaves on this Windows gfx1151 box (observes only; judged by
criteria/ps1_rocm_install.py).

Per state (the tree's own install.ps1 --local, each arm in its own UNSLOTH_STUDIO_HOME):
  normal  -- a plain install: installed torch.__version__, torch.version.hip, torch.cuda.is_available(), the device
             arch, a real matmul on the GPU, and the installer's detection / torch lines.
  broken  -- the same install with the AMD multi-arch ROCm index made unreachable
             (UNSLOTH_ROCM_WINDOWS_MULTIARCH_MIRROR -> a closed local port): what the installer does when ROCm torch
             genuinely cannot be installed (exit code, what torch is left, which lines say so).
  unit    -- tests/studio/test_rocm_torch_install_failure.ps1 (passed in with --ps1-test) under Windows
             PowerShell 5.1 against this tree's setup.ps1.
Base state only, state-independent:
  lpips   -- the shape of run 44279815226: a bare venv, then `pip install lpips` (no install.ps1 at all), to name
             where that run's torch 2.15.0+cpu came from.
Writes JSON via --out, never stdout. No token is used."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUERY = HERE / "ps1rocm" / "torch_query.py"
BROKEN_MIRROR = "http://127.0.0.1:9/rocm/whl-multi-arch"
LINE_RX = re.compile(
    r"(^\s*(gpu|rocm|torch)\s{2,}|HIP SDK|gfx\d+[a-z]? --|installing PyTorch|ROCm PyTorch|CPU-only|CPU PyTorch|CPU base|"
    r"\[ERROR\]|\[FAILED\]|falling back|NOT a GPU|UNSLOTH_ALLOW_CPU|torch check|torch flavor|flavor mismatch|"
    r"index:|wanted:|reason:|setup stops|unsloth studio setup failed|torch \d+\.\d+\.\d+\S* detected)",
    re.IGNORECASE)


def _decode(b: bytes) -> str:
    if b[:2] in (b"\xff\xfe", b"\xfe\xff") or (len(b) > 3 and b[1:2] == b"\x00"):
        try:
            return b.decode("utf-16", errors = "replace")
        except Exception:  # noqa: BLE001
            pass
    return b.decode("utf-8", errors = "replace")


def _torch(py: Path, out: Path) -> dict:
    if not py.is_file():
        return {"venv": False}
    out.unlink(missing_ok = True)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    t0 = time.time()
    p = subprocess.run([str(py), "-I", str(QUERY), "--out", str(out)], capture_output = True, env = env, timeout = 900)
    res = {"venv": True, "rc": p.returncode, "secs": round(time.time() - t0, 1),
           "stderr_tail": _decode(p.stderr)[-1500:]}
    if out.is_file():
        res.update(json.loads(out.read_text(encoding = "utf-8")))
    return res


def _install(checkout: Path, home: Path, log: Path, extra_env: dict) -> dict:
    if home.exists():
        shutil.rmtree(home, ignore_errors = True)
    env = dict(os.environ)
    env.update({"UNSLOTH_STUDIO_HOME": str(home)})
    env.update(extra_env)
    t0 = time.time()
    try:
        p = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ".\\install.ps1",
                            "--local", "--skip-autostart", "--no-rollback"],
                           cwd = str(checkout), env = env, stdout = subprocess.PIPE, stderr = subprocess.STDOUT,
                           timeout = 45 * 60)
        rc, raw = p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        rc, raw = "timeout", (e.stdout or b"")
    text = _decode(raw)
    log.write_text(text, encoding = "utf-8")
    lines = [ln.rstrip() for ln in text.splitlines()]
    picked = [ln for ln in lines if LINE_RX.search(ln)][:80]
    return {"rc": rc, "secs": round(time.time() - t0, 1), "log": log.name, "lines": picked,
            "tail": lines[-25:],
            "says_rocm_error": any("[ERROR] AMD ROCm PyTorch could not be installed" in ln for ln in lines),
            "says_gpu_rocm_installed": any("GPU ROCm PyTorch installed" in ln for ln in lines),
            "says_falling_back_cpu": any("falling back to CPU" in ln for ln in lines),
            "setup_installed_cpu": any("installing PyTorch (CPU-only)" in ln for ln in lines),
            "setup_failed_line": any("unsloth studio setup failed" in ln for ln in lines),
            "torch": _torch(home / "unsloth_studio" / "Scripts" / "python.exe", log.with_suffix(".torch.json"))}


def _unit(checkout: Path, ps1_test: Path, log: Path) -> dict:
    dst = checkout / "tests" / "studio" / ps1_test.name
    dst.parent.mkdir(parents = True, exist_ok = True)
    shutil.copyfile(ps1_test, dst)
    p = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(dst)],
                       cwd = str(checkout), stdout = subprocess.PIPE, stderr = subprocess.STDOUT, timeout = 600)
    text = _decode(p.stdout)
    log.write_text(text, encoding = "utf-8")
    return {"rc": p.returncode, "pass": len(re.findall(r"^\s*PASS\s", text, re.M)),
            "fail": len(re.findall(r"^\s*FAIL\s", text, re.M)),
            "failed": [ln.strip() for ln in text.splitlines() if re.match(r"^\s*FAIL\s", ln)][:40],
            "psversion": "5.1 (powershell.exe)"}


def _lpips(root: Path) -> dict:
    venv = root / "lv"
    shutil.rmtree(venv, ignore_errors = True)
    r = subprocess.run([sys.executable, "-m", "venv", str(venv)], capture_output = True)
    py = venv / "Scripts" / "python.exe"
    if r.returncode or not py.is_file():
        return {"error": "venv failed", "stderr": _decode(r.stderr)[-800:]}
    p = subprocess.run([str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check", "lpips"],
                       capture_output = True, timeout = 1800)
    res = {"pip_rc": p.returncode, "pip_tail": _decode(p.stdout + p.stderr)[-800:]}
    res["torch"] = _torch(py, root / "lpips_torch.json")
    shutil.rmtree(venv, ignore_errors = True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--ps1-test", required = True)
    a = ap.parse_args()
    out = Path(a.out)
    co = Path(a.checkout)
    root = (out.resolve().parents[2] if len(out.resolve().parents) > 2 else out.parent) / "s"  # short: the Studio venv nests deep under it (MAX_PATH)
    root.mkdir(parents = True, exist_ok = True)
    logs = out.parent / f"logs_{a.state}"
    logs.mkdir(parents = True, exist_ok = True)
    res: dict = {"state": a.state, "platform": sys.platform}
    try:
        res["head_commit"] = subprocess.run(["git", "-C", str(co), "rev-parse", "HEAD"], capture_output = True,
                                            text = True).stdout.strip()
        # Unit first: it is cheap and does not depend on the network.
        res["unit"] = _unit(co, Path(a.ps1_test), logs / "unit.log")
        res["normal"] = _install(co, root / f"{a.state[0]}n", logs / "install_normal.log", {})
        res["broken"] = _install(co, root / f"{a.state[0]}b", logs / "install_broken.log",
                                 {"UNSLOTH_ROCM_WINDOWS_MULTIARCH_MIRROR": BROKEN_MIRROR})
        if a.state == "base":
            res["lpips"] = _lpips(root)
        for arm in ("normal", "broken"):
            shutil.rmtree(root / f"{a.state[0]}{arm[0]}", ignore_errors = True)
    except Exception as e:  # noqa: BLE001
        res["probe_error"] = f"{type(e).__name__}: {e}"
    out.write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

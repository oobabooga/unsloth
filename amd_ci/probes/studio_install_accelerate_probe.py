#!/usr/bin/env python3
"""Probe: run the REAL installer the way install.ps1 does, and see what accelerate survives.

This is the one link the constraints-resolution probe cannot reach. `constraints.txt` is
only passed as `-c` by the core-packages step, and install.ps1 resolves accelerate itself
without it and then sets SKIP_STUDIO_BASE=1, which makes install_python_stack.py skip that
step entirely (`if skip_base: pass`). So the question "does the cap actually reach a fresh
Windows install" is not answered by resolving a constraints file. It is answered by
running the installer with skip_base set, from a venv seeded the way install.ps1 leaves
one, and reading the version back afterwards.

Per state:

1. A private venv, seeded with the AMD Windows ROCm torch and accelerate 1.15.0 with its
   dependencies. That is exactly the state install.ps1 hands over: a ROCm torch it
   installed from the AMD index, and a 1.15.0 it resolved with no -c.
2. `python <checkout>/studio/install_python_stack.py` with SKIP_STUDIO_BASE=1, which is
   the handoff install.ps1 performs.
3. Read accelerate back. The base checkout has no repair step and should still be on
   1.15.0; the head should be on 1.14.x, and it has to have SURVIVED every later
   with-deps step, which is the part that was never checked.

The torch is seeded first on purpose: `_ensure_rocm_torch` skips when a ROCm build is
already present, so this measures the dependency phase rather than re-downloading torch.

Pairs with criteria/studio_install_caps_accelerate.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROCM_INDEX = "https://repo.amd.com/rocm/whl/gfx1151/"
ROCM_TORCH = "torch==2.9.1+rocm7.13.0"
_UTF8_ENV = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def _run(cmd, timeout = 5400, **kw):
    env = dict(kw.pop("env", None) or os.environ, **_UTF8_ENV)
    return subprocess.run(cmd, capture_output = True, timeout = timeout, env = env,
                          encoding = "utf-8", errors = "replace", **kw)


def _version(py: Path, package: str) -> str | None:
    r = _run([str(py), "-c",
              f"import importlib.metadata as m;print(m.version({package!r}))"], timeout = 300)
    out = (r.stdout or "").strip().splitlines()
    return out[-1] if r.returncode == 0 and out else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "platform": sys.platform}
    work = Path(os.environ.get("RUNNER_TEMP", ".")) / f"studioinstall_{args.state}"
    venv = work / "venv"
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    try:
        work.mkdir(parents = True, exist_ok = True)
        r = _run([sys.executable, "-m", "venv", str(venv)], timeout = 600)
        if r.returncode != 0:
            obs["setup_error"] = f"venv: {(r.stderr or r.stdout)[-400:]}"
            raise SystemExit(0)

        # The state install.ps1 hands over: ROCm torch from the AMD index, and an
        # accelerate 1.15.0 that was resolved with no constraints file in sight.
        for step, cmd in (
            ("pip", [str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"]),
            ("torch", [str(py), "-m", "pip", "install", "-q", "--index-url", ROCM_INDEX,
                       "--extra-index-url", "https://pypi.org/simple", ROCM_TORCH]),
            ("accelerate", [str(py), "-m", "pip", "install", "-q", "accelerate==1.15.0"]),
        ):
            r = _run(cmd)
            if r.returncode != 0:
                obs["setup_error"] = f"{step}: {(r.stderr or r.stdout)[-500:]}"
                raise SystemExit(0)

        obs["seeded_torch"] = _version(py, "torch")
        obs["seeded_accelerate"] = _version(py, "accelerate")

        stack = args.checkout / "studio" / "install_python_stack.py"
        obs["installer_present"] = stack.is_file()
        if not stack.is_file():
            obs["setup_error"] = f"no installer at {stack}"
            raise SystemExit(0)

        # SKIP_STUDIO_BASE=1 is the whole point: it is what install.ps1 sets, and what
        # makes the constrained core step a no-op.
        env = dict(os.environ,
                   SKIP_STUDIO_BASE = "1",
                   UNSLOTH_STUDIO_HOME = str(work / "studio"),
                   UNSLOTH_VERBOSE = "1")
        r = _run([str(py), str(stack)], env = env)
        obs["installer_rc"] = r.returncode
        obs["installer_tail"] = (r.stdout or r.stderr or "")[-1500:]
        obs["final_accelerate"] = _version(py, "accelerate")
        obs["final_torch"] = _version(py, "torch")
    except SystemExit:
        pass
    except Exception as e:  # noqa: BLE001
        obs["probe_error"] = f"{type(e).__name__}: {e}"

    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

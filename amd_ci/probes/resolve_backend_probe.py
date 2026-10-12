#!/usr/bin/env python3
"""Probe: which llama.cpp backend does this checkout's installer pick for "auto" on this host?

Observes only: runs the checkout's install_llama_prebuilt.py --resolve-backends (no download)
with and without the --rocm-gfx forward setup.ps1 makes, and records the auto row plus the
host facts behind the route.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    script = args.checkout / "studio" / "install_llama_prebuilt.py"
    obs: dict = {"state": args.state, "runs": {}}
    for label, extra in (("detected", []), ("forwarded_gfx1151", ["--rocm-gfx", "gfx1151"])):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [sys.executable, str(script), "--resolve-backends", "--output-format", "json",
                   "--install-dir", tmp, *extra]
            p = subprocess.run(cmd, capture_output = True, text = True, encoding = "utf-8",
                               errors = "replace", timeout = 600)
        run = {"rc": p.returncode, "stderr_tail": p.stderr[-1500:]}
        try:
            data = json.loads(p.stdout[p.stdout.index("{"):])
            rows = {row["backend"]: row for row in data.get("backends", [])}
            run["auto"] = rows.get("auto")
            run["rocm"] = rows.get("rocm")
            run["vulkan"] = rows.get("vulkan")
            run["platform"] = data.get("platform")
        except Exception as exc:
            run["error"] = f"{type(exc).__name__}: {exc}; stdout head {p.stdout[:400]!r}"
        obs["runs"][label] = run
    try:
        sys.path.insert(0, str(script.parent))
        import install_llama_prebuilt as ilp  # noqa: PLC0415
        host = ilp.detect_host()
        obs["host"] = {
            "is_windows": host.is_windows,
            "has_rocm": host.has_rocm,
            "rocm_gfx_target": host.rocm_gfx_target,
            "rocm_gfx_targets": list(host.rocm_gfx_targets or []),
            "has_physical_nvidia": host.has_physical_nvidia,
            "amd_vulkan_icd_present": ilp._amd_vulkan_icd_present(),
        }
    except Exception as exc:
        obs["host_error"] = f"{type(exc).__name__}: {exc}"
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Probe: rocm_routing_probe.py's observations PLUS install.sh's migrated-venv ROCm repair
decision, evaluated against the REAL venv torch.

Observes only. Everything rocm_routing_probe.py records is recorded unchanged (same
keys), and one more key is added:

  migrated_repair  install.sh's top-level functions and its torch-routing span, lifted the
                   same way rocm_routing_probe.py lifts them, sourced so every routing
                   variable (_torch_index_leaf, _gfx_rocm64_target, and at the head
                   _amd_arch_index_routed / _amd_arch_index_family) comes from real host
                   detection; then the checkout's OWN "Repair ROCm torch if overwritten
                   during migrated install" block, up to _gfx906_bnb_prune, run verbatim
                   with _VENV_PY pointing at THIS interpreter (the job runs the probe under
                   the Studio venv python, so that is the runner's real torch and real
                   `rocm` metadata). Only substep (prints its message) and
                   _install_torch_default_index (prints REINSTALL) are stubbed, the same
                   seam tests/studio/install/test_rocm_support.py
                   `_run_migrated_rocm_repair` uses. Nothing is installed.

                   Records: _venv_torch_amd_family output (or "undefined" where the
                   checkout has no such function), _venv_torch_rocm_below <venv> 7 13 exit
                   status, the routed family, which substep fired, and whether it would
                   REINSTALL. Plus the venv's torch version / hip / `rocm` requirements read
                   directly, so criteria can check the venv really is the gfx1151 7.13 wheel.

Writes JSON to --out, never stdout (import banners corrupt stdout).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rocm_routing_probe as base_probe  # noqa: E402

BLOCK_START = "        # Repair ROCm torch if overwritten during migrated install"
BLOCK_END = "        _gfx906_bnb_prune"
MARK = "__AMD_CI_REPAIR__"
FIELDS = ("_torch_index_leaf", "_torch_index_is_rocm_family", "TORCH_INDEX_URL",
          "_gfx_rocm64_target", "_gfx_rocm64_floor_maj", "_gfx_rocm64_floor_min",
          "_amd_arch_index_routed", "_amd_arch_index_family")


def venv_torch_facts() -> dict:
    """The interpreter's own torch + rocm metadata, read in a child so a torch import
    failure cannot take the probe down."""
    code = r"""
import json, re
out = {}
try:
    import torch
    out["torch_version"] = getattr(torch, "__version__", None)
    out["torch_hip"] = getattr(getattr(torch, "version", None), "hip", None)
except Exception as e:
    out["torch_error"] = "%s: %s" % (type(e).__name__, e)
from importlib import metadata
for name in ("torch", "rocm"):
    try:
        out[name + "_dist_version"] = metadata.version(name)
    except Exception:
        out[name + "_dist_version"] = None
try:
    out["rocm_requires"] = metadata.requires("rocm")
except Exception as e:
    out["rocm_requires"] = None
out["rocm_sdk_dists"] = sorted({(d.metadata["Name"] or "") + "==" + (d.version or "")
    for d in metadata.distributions()
    if re.match(r"rocm[-_]sdk", (d.metadata["Name"] or ""), re.I)})
print(json.dumps(out))
"""
    p = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True,
                       timeout = 300)
    last = [l for l in (p.stdout or "").splitlines() if l.startswith("{")]
    if not last:
        return {"error": f"rc={p.returncode}", "stderr_tail": (p.stderr or "")[-1500:]}
    return json.loads(last[-1])


def probe_repair(checkout: Path, work: Path) -> dict:
    out: dict = {"venv_python": sys.executable}
    src = (checkout / "install.sh").read_text(encoding = "utf-8")
    s = src.find(BLOCK_START)
    e = src.find(BLOCK_END, s)
    if s < 0 or e < 0:
        return {**out, "error": "migrated repair block not found"}
    block = src[s:e]
    out["block_has_family_check"] = "_venv_torch_amd_family" in block
    try:
        lifted = base_probe.lift_install_sh(checkout, work / "lifted_install_repair.sh")
    except Exception as ex:  # noqa: BLE001
        return {**out, "error": f"lift failed: {type(ex).__name__}: {ex}"}
    blk = work / "repair_block.sh"
    blk.write_text(block + "\n", encoding = "utf-8")
    osname = "linux" if platform.system() == "Linux" else platform.system().lower()
    prints = "".join(f'printf "%s=%s\\n" "{f}" "${{{f}:-}}"\n' for f in FIELDS)
    vpy = sys.executable
    script = (
        "set -euo pipefail\n"
        f"OS={osname}\n_ARCH={platform.machine()}\nSKIP_TORCH=false\n"
        f"VENV_DIR='{work}/novenv'\n"
        f". '{lifted}'\n"
        f"_VENV_PY='{vpy}'\n"
        "substep() { printf 'SUBSTEP %s\\n' \"$1\"; }\n"
        "_install_torch_default_index() { printf 'REINSTALL %s\\n' \"$*\"; }\n"
        f"echo {MARK}\n" + prints +
        # The helpers as install.sh defines them at this checkout, on the real venv python.
        "if declare -F _venv_torch_amd_family >/dev/null; then\n"
        "  printf 'venv_family=%s\\n' \"$(_venv_torch_amd_family \"$_VENV_PY\")\"\n"
        "else printf 'venv_family=undefined\\n'; fi\n"
        "if declare -F _venv_torch_rocm_below >/dev/null; then\n"
        "  if _venv_torch_rocm_below \"$_VENV_PY\" 7 13; then printf 'below_7_13=yes\\n';"
        " else printf 'below_7_13=no\\n'; fi\n"
        "else printf 'below_7_13=undefined\\n'; fi\n"
        "echo __BLOCK__\n"
        f". '{blk}'\n"
        "echo __DONE__\n")
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
    tail = stdout.split(MARK, 1)[1]
    head_part, _, block_part = tail.partition("__BLOCK__")
    fields = {}
    for line in head_part.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k] = v
    out["fields"] = fields
    out["venv_family"] = fields.pop("venv_family", None)
    out["below_7_13"] = fields.pop("below_7_13", None)
    out["block_completed"] = "__DONE__" in block_part
    out["substeps"] = [l[len("SUBSTEP "):] for l in block_part.splitlines()
                       if l.startswith("SUBSTEP ")]
    out["reinstall"] = any(l.startswith("REINSTALL") for l in block_part.splitlines())
    if not out["block_completed"]:
        out["error"] = f"repair block did not complete (rc={p.returncode})"
        out["stdout_tail"] = stdout[-1500:]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    checkout = args.checkout.resolve()

    # Everything the routing probe records, unchanged.
    routing_out = args.out.with_name(args.out.stem + ".routing.json")
    rc = subprocess.run([sys.executable, str(Path(base_probe.__file__).resolve()),
                         "--state", args.state, "--checkout", str(checkout),
                         "--out", str(routing_out)],
                        capture_output = True, text = True, timeout = 1800)
    obs: dict
    try:
        obs = json.loads(routing_out.read_text(encoding = "utf-8"))
    except Exception as ex:  # noqa: BLE001
        obs = {"state": args.state, "checkout": str(checkout),
               "routing_probe_error": f"rc={rc.returncode}; {type(ex).__name__}: {ex}",
               "routing_probe_stderr": (rc.stderr or "")[-2000:]}
    try:
        obs["venv_torch"] = venv_torch_facts()
    except Exception as ex:  # noqa: BLE001
        obs["venv_torch"] = {"error": f"{type(ex).__name__}: {ex}"}
    with tempfile.TemporaryDirectory(prefix = f"repair_{args.state}_",
                                     dir = os.environ.get("RUNNER_TEMP") or None) as d:
        try:
            obs["migrated_repair"] = probe_repair(checkout, Path(d))
        except Exception as ex:  # noqa: BLE001
            obs["migrated_repair"] = {"error": f"{type(ex).__name__}: {ex}",
                                      "traceback": traceback.format_exc()[-2000:]}
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

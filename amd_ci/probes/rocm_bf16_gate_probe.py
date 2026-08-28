#!/usr/bin/env python3
"""Probe: what does this checkout decide about bf16 on a ROCm host, real and spoofed?

Two readings from the same real ROCm install, each in its own subprocess because
`unsloth/_gpu_init.py` settles SUPPORTS_BFLOAT16 at import and never revisits it:

  real    -- the host's own gcnArchName, untouched. gfx1151 here, so bf16 must stay on.
  spoofed -- get_device_properties patched to present gfx1032, the arch from issue 7922.

The spoof is the only way this host can reach the defect at all: the bug needs RDNA2
silicon and the runner is an RDNA3.5 APU. It is honest because the code under test
reads the arch STRING and nothing else, so presenting a different string exercises the
real decision path against the real ROCm runtime. It does not, and cannot, prove that a
physical gfx1032 then trains -- criteria/rocm_bf16_gate.py says so in its own words.

Observes only. Whether "bf16 off under the spoof" is the fix is the criteria's call.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Runs in a child so the import-time decision can be taken twice. Prints one JSON
# object on the last line; the probe reads that, never this script's other chatter,
# because importing unsloth prints a banner.
RUNNER = r'''
import json, os, sys

CHECKOUT = sys.argv[1]
SPOOF    = sys.argv[2] or None

out = {"spoof": SPOOF}

try:
    import torch
    out["torch_version"] = torch.__version__
    out["torch_hip"] = getattr(torch.version, "hip", None)
    out["device_count"] = torch.cuda.device_count()

    if SPOOF:
        # Present a different gcnArchName to every reader, leaving the real runtime
        # underneath. Both spellings are patched: device_type.py reads gcnArchName,
        # and a wrapper keeps every other property attribute live.
        _real = torch.cuda.get_device_properties

        class _Props:
            def __init__(self, p):
                self.__dict__["_p"] = p
            def __getattr__(self, name):
                if name in ("gcnArchName", "gcn_arch_name"):
                    return SPOOF
                return getattr(self.__dict__["_p"], name)

        torch.cuda.get_device_properties = lambda i = 0: _Props(_real(i))
        out["spoof_reads_back"] = str(torch.cuda.get_device_properties(0).gcnArchName)

    out["torch_says_bf16"] = bool(torch.cuda.is_bf16_supported())
except Exception as e:
    out["torch_error"] = "%s: %s" % (type(e).__name__, e)

sys.path.insert(0, CHECKOUT)
for stale in [m for m in list(sys.modules) if m == "unsloth" or m.startswith("unsloth.")]:
    del sys.modules[stale]

try:
    import unsloth
    out["unsloth_file"] = getattr(unsloth, "__file__", None)
    import unsloth._gpu_init as gi
    out["gpu_init_bf16"] = bool(getattr(gi, "SUPPORTS_BFLOAT16"))
    import unsloth.models._utils as mu
    out["utils_bf16"] = bool(getattr(mu, "SUPPORTS_BFLOAT16"))
    out["device_type"] = str(getattr(gi, "DEVICE_TYPE", ""))
    # The patched probe is what transformers, TRL and unsloth_zoo actually ask.
    import torch as _t
    out["patched_torch_says_bf16"] = bool(_t.cuda.is_bf16_supported())
    # The CUDA branch keeps `including_emulation`; the HIP closure takes *args. If a
    # caller passes it and the call raises, that is a real break, so record it.
    try:
        out["bf16_including_emulation_false"] = bool(
            _t.cuda.is_bf16_supported(including_emulation = False)
        )
    except Exception as e:
        out["bf16_kwarg_error"] = "%s: %s" % (type(e).__name__, e)
except Exception as e:
    import traceback
    out["unsloth_error"] = "%s: %s" % (type(e).__name__, e)
    out["unsloth_traceback"] = traceback.format_exc()[-2000:]

# A real bf16 tensor op, so the report is not purely a boolean about a boolean.
try:
    import torch as _t
    if _t.cuda.is_available():
        a = _t.randn(64, 64, device = "cuda", dtype = _t.bfloat16)
        out["bf16_matmul_ok"] = bool(_t.isfinite(a @ a).all().item())
except Exception as e:
    out["bf16_matmul_error"] = "%s: %s" % (type(e).__name__, e)

print("@@PROBE_JSON@@" + json.dumps(out))
'''


def _run(python: str, script: Path, checkout: str, spoof: str, timeout: int) -> dict:
    """One reading. Never raises: a failed child is a recorded reading, not a crash."""
    try:
        r = subprocess.run(
            [python, str(script), checkout, spoof],
            capture_output = True, text = True, timeout = timeout,
        )
    except subprocess.TimeoutExpired:
        return {"child_error": f"timed out after {timeout}s", "spoof": spoof or None}
    except Exception as e:  # noqa: BLE001
        return {"child_error": f"{type(e).__name__}: {e}", "spoof": spoof or None}

    for line in reversed((r.stdout or "").splitlines()):
        if line.startswith("@@PROBE_JSON@@"):
            try:
                d = json.loads(line[len("@@PROBE_JSON@@"):])
                d["child_rc"] = r.returncode
                return d
            except Exception as e:  # noqa: BLE001
                return {"child_error": f"bad JSON: {e}", "child_rc": r.returncode}
    return {
        "child_error": "no JSON line from the child",
        "child_rc": r.returncode,
        "stdout_tail": (r.stdout or "")[-1500:],
        "stderr_tail": (r.stderr or "")[-1500:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--spoof-arch", default = "gfx1032")
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 900)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "checkout": str(args.checkout),
                 "spoof_arch": args.spoof_arch}

    # device_type.py is where the PR puts arch_lacks_bf16; its absence is how the
    # base state is recognised, and it is worth recording explicitly rather than
    # inferring it from a failed import later.
    dt = args.checkout / "unsloth" / "device_type.py"
    obs["has_arch_lacks_bf16"] = dt.is_file() and "def arch_lacks_bf16" in dt.read_text(
        encoding = "utf-8", errors = "replace"
    )

    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "read_bf16.py"
        script.write_text(RUNNER, encoding = "utf-8")
        obs["real"] = _run(args.python, script, str(args.checkout), "", args.timeout)
        obs["spoofed"] = _run(args.python, script, str(args.checkout),
                              args.spoof_arch, args.timeout)

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

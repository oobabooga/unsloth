#!/usr/bin/env python3
"""Observe what `import unsloth` does to ROCm SDPA backend selection.

PR 8821 adds `os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")`
to `unsloth/__init__.py`. Torch reads that variable into a function-local
`static const bool` inside `check_flash_attention_hardware_support` /
`check_mem_efficient_hardware_support`, and only inside `#if USE_ROCM`, and only
once `aotriton::isArchExperimentallySupported(stream)` is true. So the variable
can only matter on a ROCm build whose AOTriton flags THIS arch experimental.

This probe never judges. It records, in a fresh process per state:

  * the build and the architecture, so a verdict can say which arch it is about
  * whether importing unsloth left the gate set
  * whether torch will admit the flash / mem-efficient backends afterwards
  * the backend SDPA actually dispatches to, and whether its output agrees with
    the math reference, because an experimental kernel that runs and is wrong is
    the failure mode that matters here
  * peak memory for a score matrix big enough to separate O(N^2) math from flash
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

GATE = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"

# Runs inside the checkout, in its own interpreter, so the import order and the
# once-only env read are the real ones rather than whatever this probe did first.
_INNER = r'''
import json, os, sys
record = {}
record["gate_before_import"] = os.environ.get("GATE_NAME_PLACEHOLDER")
try:
    import unsloth  # noqa: F401
    record["unsloth_imported"] = True
except Exception as e:
    record["unsloth_imported"] = False
    record["unsloth_error"] = f"{type(e).__name__}: {e}"
record["gate_after_import"] = os.environ.get("GATE_NAME_PLACEHOLDER")

import torch
record["torch"] = torch.__version__
record["hip"] = getattr(torch.version, "hip", None)
record["cuda"] = getattr(torch.version, "cuda", None)
record["cuda_available"] = bool(torch.cuda.is_available())
if record["cuda_available"]:
    props = torch.cuda.get_device_properties(0)
    record["arch"] = getattr(props, "gcnArchName", None) or props.name
    record["device_name"] = props.name

    from torch.backends.cuda import (
        SDPAParams, can_use_flash_attention, can_use_efficient_attention,
    )
    torch.manual_seed(0)
    B, H, S, D = 2, 8, 2048, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q); v = torch.randn_like(q)
    p = SDPAParams(q, k, v, None, 0.0, True, False)
    record["can_use_flash"] = bool(can_use_flash_attention(p, False))
    record["can_use_efficient"] = bool(can_use_efficient_attention(p, False))

    from torch.nn.attention import SDPBackend, sdpa_kernel
    ref = torch.nn.functional.scaled_dot_product_attention  # noqa: E501

    def _run(backends):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        with sdpa_kernel(backends):
            out = ref(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        return out, torch.cuda.max_memory_allocated() / 2**30

    try:
        math_out, math_gib = _run([SDPBackend.MATH])
        record["math_peak_gib"] = round(math_gib, 4)
    except Exception as e:
        math_out = None
        record["math_error"] = f"{type(e).__name__}: {e}"

    for name, backend in (
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("efficient", SDPBackend.EFFICIENT_ATTENTION),
    ):
        try:
            out, gib = _run([backend])
            record[f"{name}_ran"] = True
            record[f"{name}_peak_gib"] = round(gib, 4)
            if math_out is not None:
                diff = (out.float() - math_out.float()).abs().max().item()
                record[f"{name}_max_abs_diff_vs_math"] = float(f"{diff:.6g}")
                record[f"{name}_finite"] = bool(out.isfinite().all())
        except Exception as e:
            record[f"{name}_ran"] = False
            record[f"{name}_error"] = f"{type(e).__name__}: {e}"

    # What the default dispatch actually picks, with no explicit backend context.
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    default_out = ref(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    record["default_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 4)
    if math_out is not None:
        record["default_matches_math_exactly"] = bool(
            torch.equal(default_out, math_out)
        )

print("RECORD " + json.dumps(record, sort_keys=True))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--checkout", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "checkout": str(args.checkout)}

    # Whether this state's source even carries the gate, so the verdict can say
    # which side of the change produced which reading without inferring it.
    init = args.checkout / "unsloth" / "__init__.py"
    try:
        obs["init_sets_gate"] = GATE in init.read_text(encoding="utf-8")
    except Exception as e:
        obs["init_read_error"] = f"{type(e).__name__}: {e}"

    env = {k: v for k, v in os.environ.items() if k != GATE}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(args.checkout)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    code = _INNER.replace("GATE_NAME_PLACEHOLDER", GATE)
    try:
        run = subprocess.run(
            [args.python, "-c", code],
            cwd=str(args.checkout),
            capture_output=True,
            text=True,
            env=env,
            timeout=args.timeout,
        )
        obs["rc"] = run.returncode
        obs["stderr_tail"] = run.stderr[-2000:]
        for line in run.stdout.splitlines():
            if line.startswith("RECORD "):
                obs.update(json.loads(line[len("RECORD "):]))
                break
        else:
            obs["no_record"] = True
            obs["stdout_tail"] = run.stdout[-2000:]
    except subprocess.TimeoutExpired:
        obs["timeout"] = True

    args.out.write_text(json.dumps(obs, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

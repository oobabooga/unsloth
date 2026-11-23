#!/usr/bin/env python3
"""Probe: Studio's ROCm flash-attn probe (`_run_rocm_flash_probe` in studio/backend/core/inference/diffusion_attention.py)
asked about a card that is NOT the current device, as the layer split does for a secondary card from the primary's
load scope. Meant for `scaffold.py --spoof-devices 1`: cuda:1 is the real GPU under another ordinal (wiring only).
Observes only (judged by criteria/rk_flash_probe_device.py).

Per state, in a child process with FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE and current device cuda:0:
  raw     flash_attn_func on cuda:1 operands from current device 0 (the error text), and inside torch.cuda.device(1)
  probe   the state's _run_rocm_flash_probe("cuda:0") and ("cuda:1"), current device before / after each
JSON to --out only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent = 2, default = str), encoding = "utf-8")


def _err(e: BaseException, n: int = 700) -> str:
    return f"{type(e).__name__}: {str(e)[:n]}"


def child(args) -> int:
    res: dict = {"state": args.state}
    try:
        sys.path.insert(0, str(Path(args.checkout) / "studio" / "backend"))
        import torch

        res.update(torch = torch.__version__, hip = getattr(torch.version, "hip", None), platform = sys.platform,
                   device_count = torch.cuda.device_count(), spoofed = os.environ.get("AMD_CI_SPOOFED_DEVICES"))
        if torch.cuda.is_available():
            res["archs"] = [getattr(torch.cuda.get_device_properties(i), "gcnArchName", None)
                            for i in range(torch.cuda.device_count())]
        import flash_attn
        from flash_attn import flash_attn_func

        res["flash_attn"] = {"version": getattr(flash_attn, "__version__", None), "file": flash_attn.__file__}
        torch.cuda.set_device(0)
        g = torch.Generator().manual_seed(0)
        q, k, v = (torch.randn((1, 128, 2, 64), generator = g).to("cuda:1", torch.bfloat16) for _ in range(3))
        raw = {}
        try:
            flash_attn_func(q, k, v)
            torch.cuda.synchronize("cuda:1")
            raw["from_current_0"] = {"ok": True}
        except Exception as e:  # noqa: BLE001
            raw["from_current_0"] = {"ok": False, "error": _err(e)}
        try:
            with torch.cuda.device(1):
                flash_attn_func(q, k, v)
                torch.cuda.synchronize("cuda:1")
            raw["inside_device_1"] = {"ok": True}
        except Exception as e:  # noqa: BLE001
            raw["inside_device_1"] = {"ok": False, "error": _err(e)}
        res["raw"] = raw
        res["current_after_raw"] = torch.cuda.current_device()

        import core.inference.diffusion_attention as att

        res["module_from_checkout"] = os.path.abspath(att.__file__).startswith(os.path.abspath(args.checkout))
        res["has_device_scope"] = "torch.cuda.device(index)" in Path(att.__file__).read_text(encoding = "utf-8")
        probe = {}
        for dev in ("cuda:0", "cuda:1"):
            torch.cuda.set_device(0)
            row = {"current_before": torch.cuda.current_device()}
            try:
                row["verdict"] = bool(att._run_rocm_flash_probe(dev, torch.bfloat16))
            except Exception as e:  # noqa: BLE001
                row["error"] = _err(e)
            row["current_after"] = torch.cuda.current_device()
            probe[dev] = row
        res["probe"] = probe
    except Exception as e:  # noqa: BLE001
        res["error"] = _err(e)
        res["traceback"] = traceback.format_exc()[-3000:]
    _write(args.out, res)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--child", action = "store_true", help = argparse.SUPPRESS)
    args = ap.parse_args()
    args.checkout = str(Path(args.checkout).resolve())
    if args.child:
        return child(args)
    env = dict(os.environ, FLASH_ATTENTION_TRITON_AMD_ENABLE = "TRUE", PYTHONIOENCODING = "utf-8")
    out = args.out.with_name(f"{args.out.stem}_child.json")
    t0 = time.time()
    try:
        rc = subprocess.run([sys.executable, os.path.abspath(__file__), "--state", args.state, "--checkout",
                             args.checkout, "--out", str(out), "--child"], env = env, timeout = 1800).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    got = json.loads(out.read_text(encoding = "utf-8")) if out.is_file() else {"missing_output": True}
    got["rc"], got["seconds"] = rc, round(time.time() - t0, 1)
    _write(args.out, got)
    return 0


if __name__ == "__main__":
    sys.exit(main())

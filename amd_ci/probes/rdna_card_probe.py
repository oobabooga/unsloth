#!/usr/bin/env python3
"""Probe: does a native W8A8 linear quantised on card 1 take the RDNA int8 kernels for operands on card 1?

Runs on gfx1151 with one spoofed extra device (the real GPU wearing index 1). Card 0's RDNA readiness is recorded as
unsupported before anything is built (the condition #18199 cannot handle: a build-time probe of the current card,
not the weight's). Case "cross": weight + operand on cuda:1 while cuda:0 is current. Case "same" (a fresh state):
weight + operand on cuda:0 with nothing marked, the ordinary single-card load. Per case: RDNA engaged (calls of
rdna.rotq), and the bf16 output's relative L2 to an fp32 matmul of the dequantised weight. Observes only."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent = 2, default = str), encoding = "utf-8")


def _case(torch, nq, rdna, card: int, current: int, mark_zero: bool) -> dict:
    rdna._STATE.update({"devices": {}, "on": {}, "broken": None, "rotq": None, "mm": None})
    if mark_zero:
        rdna._STATE["devices"][torch.device("cuda", 0)] = "marked unsupported by the probe"
    torch.cuda.set_device(current)
    dev = torch.device("cuda", card)
    torch.manual_seed(0)
    lin = torch.nn.Linear(512, 256, bias = False, dtype = torch.bfloat16, device = dev)
    layer = nq.native_linear_class()(lin, nq.NATIVE_INT8, act_int8 = True, rot_group = 256).to(dev)
    calls = []
    real = rdna.rotq
    rdna.rotq = lambda *a, **k: calls.append(1) or real(*a, **k)
    try:
        x = torch.randn(4096, 512, dtype = torch.bfloat16, device = dev)
        with torch.no_grad():
            out = layer(x)
        torch.cuda.synchronize(dev)
    finally:
        rdna.rotq = real
    w = lin.weight.float()
    ref = x.float() @ w.t()
    rel = float((out.float() - ref).norm() / ref.norm())
    return {"card": card, "current": current, "rdna_flag": bool(layer._rdna), "rdna_calls": len(calls),
            "out_device": str(out.device), "finite": bool(torch.isfinite(out).all()), "rel_l2": rel,
            "state": rdna.state()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    import torch

    res: dict = {"state": args.state, "torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                 "platform": sys.platform, "cuda": torch.cuda.is_available(),
                 "devices": torch.cuda.device_count() if torch.cuda.is_available() else 0}
    if torch.cuda.is_available():
        res["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
    _write(args.out, res)
    try:
        sys.path.insert(0, str(Path(args.checkout) / "studio" / "backend"))
        from core.inference import diffusion_native_quant as nq
        from core.inference import diffusion_native_w8a8_rdna as rdna

        res["same"] = _case(torch, nq, rdna, 0, 0, False)
        if res["devices"] >= 2:
            res["cross"] = _case(torch, nq, rdna, 1, 0, True)
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-3000:]
    _write(args.out, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""PR 18825 mechanism probe 3: does any kernel's result depend on what the caching allocator's recycled memory held
before (an uninitialised read)? Each case runs on fresh allocations after the free pool was filled with zeros, then
with random bits, then with NaN bit patterns; the outputs must match bit for bit. Shapes include token counts that
are not multiples of 64/128 (kernel tail paths). JSON to --out only."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--device", default = "cuda")
    args = ap.parse_args()
    res: dict = {}
    try:
        res = run(args)
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-4000:]
    Path(args.out).write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")
    return 0


def run(args) -> dict:
    import torch
    import torch.nn.functional as F

    dev = args.device
    bf = torch.bfloat16
    sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)

    def fill_pool(kind: str, gb: float = 6.0):
        """Allocate and free many blocks of mixed sizes holding ``kind`` so the next allocations reuse them."""
        held = []
        sizes = [1 << 20, 3 << 20, 17 << 20, 64 << 20, 200 << 20]
        total = 0
        i = 0
        while total < gb * (1 << 30):
            n = sizes[i % len(sizes)]
            t = torch.empty(n, dtype = torch.uint8, device = dev)
            if kind == "zero":
                t.zero_()
            elif kind == "rand":
                t.random_(0, 256)
            else:  # bf16 / fp32 NaN-ish pattern 0xFF
                t.fill_(255)
            held.append(t)
            total += n
            i += 1
        sync()
        del held  # back to the caching allocator, contents intact

    def case(S: int, D: int, H: int, seed: int):
        g = torch.Generator(device = "cpu").manual_seed(seed)

        def rnd(*shape, scale = 1.0):
            return (torch.randn(*shape, generator = g) * scale).to(dev, bf)

        x = rnd(S, D)
        w = rnd(D, D, scale = D ** -0.5)
        wff = rnd(3 * D, D, scale = D ** -0.5)
        a = rnd(16, D, scale = D ** -0.5)
        b = rnd(D, 16, scale = 0.01)
        gout = rnd(S, D)
        q, k, v = (rnd(1, H, S, D // H) for _ in range(3))
        out = {}
        out["linear"] = F.linear(x, w)
        out["linear_ff"] = F.linear(x, wff)
        h = torch.mm(x, a.t())
        out["lora_out"] = torch.mm(h, b.t())
        out["wgrad"] = torch.mm(gout.t(), x)
        out["wgrad_lora_b"] = torch.mm(gout.t(), h)
        out["layer_norm"] = F.layer_norm(x, (D,))
        out["rms_norm"] = F.rms_norm(x, (D,))
        out["gelu"] = F.gelu(x, approximate = "tanh")
        out["softmax"] = torch.softmax(x.float(), -1)
        with torch.enable_grad():
            qq, kk, vv = (t.detach().requires_grad_() for t in (q, k, v))
            o = F.scaled_dot_product_attention(qq, kk, vv)
            dq, dk, dv = torch.autograd.grad(o, (qq, kk, vv), q)
        out["sdpa"], out["sdpa_dq"], out["sdpa_dk"], out["sdpa_dv"] = o.detach(), dq, dk, dv
        with torch.enable_grad():
            xx = x.detach().float().requires_grad_()
            ln = F.layer_norm(xx, (D,))
            (dln,) = torch.autograd.grad(ln, xx, gout.float())
        out["layer_norm_bwd"] = dln
        sync()
        return {k: v.detach().cpu() for k, v in out.items()}

    res: dict = {"torch": torch.__version__, "hip": torch.version.hip, "platform": sys.platform,
                 "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "cases": {}}
    shapes = [(1536, 3072, 24), (1101, 3072, 24), (1029, 3072, 24), (4096 + 512, 3072, 24)]
    for S, D, H in shapes:
        tag = f"S{S}_D{D}"
        ref = None
        rec = {}
        for kind in ("zero", "rand", "nan", "rand", "zero"):
            fill_pool(kind)
            out = case(S, D, H, seed = 7)
            if ref is None:
                ref = out
                continue
            diffs = {}
            for k, v in ref.items():
                if not torch.equal(v, out[k]):
                    diffs[k] = float((v.float() - out[k].float()).abs().nan_to_num(1e30).max())
            rec[kind + f"_{len(rec)}"] = diffs
        res["cases"][tag] = rec
        torch.cuda.empty_cache() if dev == "cuda" else None
    res["any_dependence"] = any(bool(d) for r in res["cases"].values() for d in r.values())
    return res


if __name__ == "__main__":
    sys.exit(main())

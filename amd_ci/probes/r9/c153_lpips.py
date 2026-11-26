#!/usr/bin/env python3
"""Per-frame LPIPS (alex) of decoded videos saved by c153_vendor_check.py (--save: [1, 3, T, H, W] fp16 in [-1, 1])
against one reference decode, with a bootstrap 95% CI of the mean over frames. Observes only.

  python c153_lpips.py --ref A.pt --cmp tag1=B.pt --cmp tag2=C.pt --out o.json
"""
from __future__ import annotations

import argparse
import json
import random


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required = True)
    ap.add_argument("--cmp", action = "append", default = [])
    ap.add_argument("--out", required = True)
    a = ap.parse_args()
    out: dict = {"ref": a.ref, "rows": {}}
    try:
        import lpips
        import torch
        net = lpips.LPIPS(net = "alex", verbose = False).cuda().eval()
        ref = torch.load(a.ref).float()
        for spec in a.cmp:
            tag, path = spec.split("=", 1)
            x = torch.load(path).float()
            if x.shape != ref.shape:
                out["rows"][tag] = {"error": f"shape {list(x.shape)} vs {list(ref.shape)}"}
                continue
            vals = []
            with torch.no_grad():
                for t in range(ref.shape[2]):
                    vals.append(float(net(x[:, :, t].cuda(), ref[:, :, t].cuda()).mean()))
            rng = random.Random(0)
            boots = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(2000))
            out["rows"][tag] = {"mean": sum(vals) / len(vals), "ci95": [boots[49], boots[1949]], "frames": len(vals),
                                "max": max(vals), "bit_equal": bool(torch.equal(x, ref))}
        out["verdict"] = "ok"
    except Exception:  # noqa: BLE001
        import traceback
        out["verdict"] = "error"
        out["error"] = traceback.format_exc()[-1500:]
    with open(a.out, "w", encoding = "utf-8") as fh:
        json.dump(out, fh, indent = 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

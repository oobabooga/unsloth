"""Compare render arms image by image: bit identity, mean / max abs pixel difference, PSNR, and LPIPS (alex) when the
lpips package is importable. usage: score_images.py --out FILE --imgdir D --pairs PAIRS.json
PAIRS.json: [{"name": N, "a": [label, gen], "b": [label, gen]}, ...]"""

from __future__ import annotations

import json
import math
import os
import sys

import _dc


def main() -> None:
    out = _dc.out_path()
    args = sys.argv[3:]
    imgdir = args[args.index("--imgdir") + 1]
    with open(args[args.index("--pairs") + 1], encoding = "utf-8") as fh:
        pairs = json.load(fh)
    rec = {"pairs": {}}
    import numpy as np
    from PIL import Image

    net = None
    dev = "cpu"
    try:
        import lpips
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net = lpips.LPIPS(net = "alex", verbose = False).to(dev).eval()
        rec["lpips"] = "alex"
    except Exception as exc:  # noqa: BLE001
        rec["lpips"] = None
        rec["lpips_unavailable"] = _dc.err(exc, 300)

    def load(label, gen, i):
        p = os.path.join(imgdir, f"{label}__{gen}__{i}.png")
        return np.asarray(Image.open(p).convert("RGB")) if os.path.exists(p) else None

    for pair in pairs:
        per = []
        for i in range(64):
            x, y = load(*pair["a"], i), load(*pair["b"], i)
            if x is None or y is None:
                break
            d = np.abs(x.astype(np.int16) - y.astype(np.int16))
            mse = float((d.astype(np.float64) ** 2).mean())
            r = {"bit_equal": bool(d.max() == 0), "max_abs": int(d.max()), "mean_abs": round(float(d.mean()), 4),
                 "psnr": None if mse == 0 else round(10 * math.log10(255.0**2 / mse), 2)}
            if net is not None:
                import torch

                ta = torch.from_numpy(x).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(dev)
                tb = torch.from_numpy(y).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(dev)
                with torch.no_grad():
                    r["lpips"] = round(float(net(ta, tb)), 5)
            per.append(r)
        agg = {"n": len(per), "a": pair["a"], "b": pair["b"]}
        if per:
            agg["bit_equal_all"] = all(r["bit_equal"] for r in per)
            agg["mean_abs_mean"] = round(sum(r["mean_abs"] for r in per) / len(per), 4)
            psnrs = [r["psnr"] for r in per if r["psnr"] is not None]
            agg["psnr_min"] = min(psnrs) if psnrs else None
            if "lpips" in per[0]:
                vals = [r["lpips"] for r in per]
                agg["lpips_mean"] = round(sum(vals) / len(vals), 5)
                agg["lpips_max"] = round(max(vals), 5)
        agg["per_image"] = per
        rec["pairs"][pair["name"]] = agg
    _dc.write(out, rec)


if __name__ == "__main__":
    main()

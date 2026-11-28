#!/usr/bin/env python3
"""Scorer for sdcpp_dit_i8_probe: per pair (a = raw PNG, b = i8 or raw2 PNG) LPIPS-alex (CPU torch + lpips) when
importable, plus PSNR and max-abs (numpy + PIL, else a pure-python PNG decoder). Usage: score.py pairs.json out.json"""

from __future__ import annotations

import json
import math
import struct
import sys
import zlib


def png_rgb(path: str):
    """Minimal 8-bit non-interlaced PNG decoder (colour types 2 / 6): returns (w, h, flat RGB bytes)."""
    data = open(path, "rb").read()
    pos, idat, w = 8, b"", None
    while pos < len(data):
        n, typ = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        if typ == b"IHDR":
            w, h, bd, ct, _, _, il = struct.unpack(">IIBBBBB", body)
            if bd != 8 or ct not in (2, 6) or il:
                raise ValueError(f"unsupported PNG bd={bd} ct={ct} il={il}")
            ch = 3 if ct == 2 else 4
        elif typ == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    stride = w * ch
    out = bytearray(h * stride)
    prev = bytearray(stride)
    i = 0
    for y in range(h):
        f = raw[i]
        line = bytearray(raw[i + 1:i + 1 + stride])
        i += 1 + stride
        if f == 1:
            for x in range(ch, stride):
                line[x] = (line[x] + line[x - ch]) & 255
        elif f == 2:
            for x in range(stride):
                line[x] = (line[x] + prev[x]) & 255
        elif f == 3:
            for x in range(stride):
                left = line[x - ch] if x >= ch else 0
                line[x] = (line[x] + ((left + prev[x]) >> 1)) & 255
        elif f == 4:
            for x in range(stride):
                a = line[x - ch] if x >= ch else 0
                b = prev[x]
                c = prev[x - ch] if x >= ch else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[x] = (line[x] + pr) & 255
        out[y * stride:(y + 1) * stride] = line
        prev = line
    if ch == 4:
        out = bytearray(v for k, v in enumerate(out) if k % 4 != 3)
    return w, h, bytes(out)


def main() -> int:
    pairs = json.load(open(sys.argv[1], encoding = "utf-8"))
    res: dict = {"method": [], "pairs": []}
    try:
        import numpy as np
        from PIL import Image
        have_np = True
        res["method"].append("numpy+PIL psnr/maxabs")
    except Exception as e:  # noqa: BLE001
        have_np = False
        res["method"].append(f"pure-python psnr/maxabs (numpy/PIL unavailable: {e})")
    lp = None
    try:
        import torch
        import lpips
        torch.set_num_threads(8)
        lp = lpips.LPIPS(net = "alex", verbose = False)
        res["method"].append(f"lpips-alex torch {torch.__version__} cpu")
    except Exception as e:  # noqa: BLE001
        res["lpips_error"] = f"{type(e).__name__}: {e}"[:500]
    for pr in pairs:
        rec = dict(pr)
        try:
            if have_np:
                a = np.asarray(Image.open(pr["a"]).convert("RGB"), dtype = np.float64)
                b = np.asarray(Image.open(pr["b"]).convert("RGB"), dtype = np.float64)
                d = a - b
                mse = float((d * d).mean())
                rec["max_abs"] = float(np.abs(d).max())
            else:
                wa, ha, a = png_rgb(pr["a"])
                wb, hb, b = png_rgb(pr["b"])
                assert (wa, ha) == (wb, hb)
                se, mx = 0, 0
                for x, y in zip(a, b):
                    dd = x - y
                    se += dd * dd
                    mx = max(mx, abs(dd))
                mse = se / len(a)
                rec["max_abs"] = float(mx)
            rec["psnr"] = float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)
            if lp is not None:
                import torch

                def t(x):
                    return torch.from_numpy(x).permute(2, 0, 1)[None].float() / 127.5 - 1.0
                with torch.no_grad():
                    rec["lpips"] = float(lp(t(a), t(b)).item())
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
        if rec.get("psnr") == float("inf"):
            rec["psnr"] = "inf"
        res["pairs"].append(rec)
        print(json.dumps({k: rec.get(k) for k in ("model", "prompt", "pair", "lpips", "psnr", "max_abs", "error")}),
              flush = True)
    json.dump(res, open(sys.argv[2], "w", encoding = "utf-8"), indent = 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""End-of-trajectory distances of one family's arms against its reference arm (same render index = same prompt and
seed): LPIPS (alex, long side 1024) and PSNR for images; mean over the saved frames for videos; relative L2 and a
log-spectrum L1 for audio. Writes OUT/scores_<family>.json and small thumbnails next to each record.

  python score.py --out OUT --family F --ref A1 --pairs A2:A1,B2:B1 --kind image|video
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--family", required = True)
    ap.add_argument("--ref", default = "A1")
    ap.add_argument("--pairs", default = "")
    ap.add_argument("--kind", default = "image")
    a = ap.parse_args()
    import lpips
    import numpy as np
    import torch
    from PIL import Image

    out = Path(a.out)
    net = lpips.LPIPS(net = "alex", verbose = False).cuda().eval()
    tags = sorted(p.parent.name for p in out.glob(f"{a.family}_*/record.json"))

    def frames(tag, i):
        d = out / tag
        if a.kind == "image":
            p = d / f"r{i}.png"
            return [np.asarray(Image.open(p).convert("RGB"))] if p.exists() else None
        p = d / f"r{i}.npz"
        return list(np.load(p)["frames"]) if p.exists() else None

    def tens(x):
        img = Image.fromarray(x)
        s = 1024 / max(img.size)
        if s < 1:
            img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.LANCZOS)
        return torch.from_numpy(np.asarray(img).astype("float32") / 127.5 - 1).permute(2, 0, 1)[None].cuda()

    def psnr(x, y):
        mse = float(((x.astype("float64") - y.astype("float64")) ** 2).mean())
        return 99.0 if mse == 0 else round(10 * np.log10(255.0**2 / mse), 3)

    def audio_dist(ta, tb, i):
        pa, pb = out / ta / f"r{i}_audio.npy", out / tb / f"r{i}_audio.npy"
        if not (pa.exists() and pb.exists()):
            return None
        x, y = np.load(pa).astype("float64").ravel(), np.load(pb).astype("float64").ravel()
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        rel = float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-12))
        win = 1024
        k = n // win
        if k < 2:
            return {"rel_l2": rel}
        fx = np.log1p(np.abs(np.fft.rfft(x[: k * win].reshape(k, win), axis = 1)))
        fy = np.log1p(np.abs(np.fft.rfft(y[: k * win].reshape(k, win), axis = 1)))
        return {"rel_l2": round(rel, 5), "logspec_l1": round(float(np.abs(fx - fy).mean()), 5)}

    def pair(ta, tb):
        rows = []
        for i in range(64):
            fa, fb = frames(ta, i), frames(tb, i)
            if fa is None or fb is None:
                if i > 16:
                    break
                continue
            n = min(len(fa), len(fb))
            lp, ps, same = [], [], True
            for x, y in zip(fa[:n], fb[:n]):
                if x.shape != y.shape:
                    same = False
                    continue
                with torch.no_grad():
                    lp.append(float(net(tens(x), tens(y)).item()))
                ps.append(psnr(x, y))
                same = same and bool(np.array_equal(x, y))
            row = {"i": i, "lpips": round(float(np.mean(lp)), 5) if lp else None,
                   "psnr": round(float(np.mean(ps)), 3) if ps else None, "identical": same, "frames": n}
            au = audio_dist(ta, tb, i)
            if au:
                row["audio"] = au
            rows.append(row)
        lps = [r["lpips"] for r in rows if r["lpips"] is not None]
        pss = [r["psnr"] for r in rows if r["psnr"] is not None]
        return {"a": ta, "b": tb, "n": len(rows), "lpips_mean": round(float(np.mean(lps)), 5) if lps else None,
                "lpips_max": round(float(np.max(lps)), 5) if lps else None,
                "psnr_mean": round(float(np.mean(pss)), 3) if pss else None, "rows": rows}

    ref = f"{a.family}_{a.ref}"
    res = {"family": a.family, "ref": ref, "vs_ref": [], "pairs": []}
    for tag in tags:
        if tag != ref and ref in tags:
            res["vs_ref"].append(pair(tag, ref))
    for pr in [x for x in a.pairs.split(",") if x]:
        x, y = pr.split(":")
        ta, tb = f"{a.family}_{x}", f"{a.family}_{y}"
        if ta in tags and tb in tags:
            res["pairs"].append(pair(ta, tb))
    for tag in tags:
        for i in range(2):
            fr = frames(tag, i)
            if fr:
                img = Image.fromarray(fr[len(fr) // 2])
                img.thumbnail((384, 384))
                img.save(out / tag / f"thumb_{i}.jpg", quality = 85)
    (out / f"scores_{a.family}.json").write_text(json.dumps(res, indent = 1))
    for r in res["vs_ref"] + res["pairs"]:
        print("SCORE", r["a"], "vs", r["b"], "n", r["n"], "lpips_mean", r["lpips_mean"], "lpips_max", r["lpips_max"],
              "psnr_mean", r["psnr_mean"], "audio", [x.get("audio") for x in r["rows"][:2]])


if __name__ == "__main__":
    main()

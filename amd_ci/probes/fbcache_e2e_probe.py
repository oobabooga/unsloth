#!/usr/bin/env python3
"""Qwen-Image-2.1 end to end on gfx1151: does an explicit First-Block-Cache request now engage, and what does it buy?

Three arms, each one fresh process of studio_weight_only_probe.py --single off (bf16 DiT, eager as Studio runs on
ROCm): the head checkout uncached (reference), the head checkout with transformer_cache=fbcache, and the BASE
checkout with transformer_cache=fbcache. On the base the request is refused for a prefix-KV transformer, so its
renders should match the reference and run at the same speed; if the base arm is already faster, the run is VOID.
Every cached render is scored against the head reference render of the same prompt and seed (LPIPS alex, PSNR).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required = True)
    ap.add_argument("--base", required = True)
    ap.add_argument("--out-dir", required = True, type = Path)
    ap.add_argument("--models", required = True)
    ap.add_argument("--size", type = int, default = 1024)
    ap.add_argument("--steps", type = int, default = 20)
    ap.add_argument("--prompts", type = int, default = 3)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents = True, exist_ok = True)
    arms = (("ref", args.head, "off"), ("fb", args.head, "fbcache"), ("base_fb", args.base, "fbcache"))
    summary: dict = {"arms": {}}
    for state, checkout, cache in arms:
        part = out / f"{state}.json"
        log = out / f"{state}.log"
        env = dict(os.environ)
        env.pop("HF_TOKEN", None)
        cmd = [sys.executable, "-u", str(HERE / "studio_weight_only_probe.py"), "--state", state,
               "--checkout", checkout, "--out", str(part), "--models", args.models, "--schemes", "off",
               "--size", str(args.size), "--steps", str(args.steps), "--prompts", str(args.prompts),
               "--single", "off", "--speed", "off", "--cache", cache]
        with open(log, "w", encoding = "utf-8") as fh:
            rc = subprocess.call(cmd, env = env, stdout = fh, stderr = subprocess.STDOUT)
        text = log.read_text(encoding = "utf-8", errors = "replace")
        sub = json.loads(part.read_text(encoding = "utf-8")) if part.is_file() else {}
        arm = (sub.get("arms") or {}).get("off") or {}
        arm["exit_code"] = rc
        arm["log_says_engaged"] = "diffusion.cache: fbcache engaged" in text
        summary["arms"][state] = arm
        for key in ("torch", "hip", "device", "arch"):
            summary.setdefault(key, sub.get(key))
        print(state, rc, json.dumps({k: arm.get(k) for k in ("loaded", "transformer_cache", "cache_resolved",
              "log_says_engaged")}, default = str), flush = True)
        (out / "summary.json").write_text(json.dumps(summary, indent = 2, default = str), encoding = "utf-8")

    import numpy as np
    from PIL import Image

    try:
        import lpips
        import torch

        net = lpips.LPIPS(net = "alex", verbose = False).eval()
    except Exception as exc:  # noqa: BLE001
        net = None
        summary["lpips_error"] = str(exc)[:200]
    for state in ("fb", "base_fb"):
        for item in (summary["arms"].get(state) or {}).get("images") or []:
            ref = out / "images_ref" / str(item.get("path"))
            img = out / f"images_{state}" / str(item.get("path"))
            if "error" in item or not ref.is_file() or not img.is_file():
                continue
            a = np.asarray(Image.open(ref).convert("RGB"), dtype = np.float32) / 255.0
            b = np.asarray(Image.open(img).convert("RGB"), dtype = np.float32) / 255.0
            item["psnr"] = round(float(10 * np.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-12))), 3)
            if net is not None:
                ta = torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1
                tb = torch.from_numpy(b).permute(2, 0, 1)[None] * 2 - 1
                with torch.no_grad():
                    item["lpips"] = round(float(net(ta, tb).item()), 4)
    lines = ["| arm | cache status | engaged (log) | image s (each; first is warm-up) | LPIPS vs ref | PSNR |",
             "|---|---|---|---|---|---|"]
    for state, arm in summary["arms"].items():
        imgs = arm.get("images") or []
        secs = ", ".join(str(i.get("seconds", "err")) for i in imgs)
        lp = ", ".join(str(i.get("lpips", "")) for i in imgs if "lpips" in i)
        ps = ", ".join(str(i.get("psnr", "")) for i in imgs if "psnr" in i)
        res = arm.get("cache_resolved") or {}
        lines.append(f"| {state} | {res.get('value')} ({res.get('reason')}) | {arm.get('log_says_engaged')} | {secs} | {lp} | {ps} |")
    base_fast = None
    try:
        ref_s = [i["seconds"] for i in summary["arms"]["ref"]["images"][1:]]
        base_s = [i["seconds"] for i in summary["arms"]["base_fb"]["images"][1:]]
        fb_s = [i["seconds"] for i in summary["arms"]["fb"]["images"][1:]]
        base_fast = min(base_s) < 0.9 * min(ref_s)
        summary["warm_median"] = {"ref": sorted(ref_s)[len(ref_s) // 2], "fb": sorted(fb_s)[len(fb_s) // 2],
                                  "base_fb": sorted(base_s)[len(base_s) // 2]}
    except Exception:  # noqa: BLE001
        pass
    summary["void"] = bool(base_fast)
    table = "\n".join(lines) + f"\n\nwarm medians: {summary.get('warm_median')}; base already fast (VOID): {base_fast}\n"
    summary["table"] = table
    (out / "summary.json").write_text(json.dumps(summary, indent = 2, default = str), encoding = "utf-8")
    (out / "SUMMARY.md").write_text(table, encoding = "utf-8")
    print(table, flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""End to end through Studio: Qwen-Image-2.1 at bf16, int8 weight-only, int8 W8A8 (torch._int_mm), and W8A8 with ConvRot.

Each arm is one fresh process of studio_weight_only_probe.py --single, as a user starting Studio would.
The W8A8 arm is the same int8 load with UNSLOTH_NATIVE_INT8_ACT=1. Every quantised render is scored
against the bf16 render of the same prompt and seed (LPIPS alex, PSNR). The log of each arm is kept and
searched for Studio's own load line, so a W8A8 arm that silently fell back to weight-only is caught.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = (
    ("ref", "off", {}),
    ("wo", "int8", {}),
    ("w8a8", "int8", {"UNSLOTH_NATIVE_INT8_ACT": "1", "UNSLOTH_NATIVE_INT8_ROT": "0"}),
    ("w8a8_rot", "int8", {"UNSLOTH_NATIVE_INT8_ACT": "1"}),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out-dir", required = True, type = Path)
    ap.add_argument("--models", required = True)
    ap.add_argument("--size", type = int, default = 1024)
    ap.add_argument("--steps", type = int, default = 20)
    ap.add_argument("--prompts", type = int, default = 2)
    ap.add_argument("--speed", default = "off", help = "Studio speed_mode: off = eager, default = compiled")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents = True, exist_ok = True)
    summary: dict = {"arms": {}}
    for state, scheme, extra in ARMS:
        part = out / f"{state}.json"
        log = out / f"{state}.log"
        env = dict(os.environ, **extra)
        env.pop("HF_TOKEN", None)
        cmd = [sys.executable, "-u", str(HERE / "studio_weight_only_probe.py"), "--state", state,
               "--checkout", args.checkout, "--out", str(part), "--models", args.models, "--schemes", scheme,
               "--size", str(args.size), "--steps", str(args.steps), "--prompts", str(args.prompts),
               "--single", scheme, "--speed", args.speed]
        with open(log, "w", encoding = "utf-8") as fh:
            rc = subprocess.call(cmd, env = env, stdout = fh, stderr = subprocess.STDOUT)
        text = log.read_text(encoding = "utf-8", errors = "replace")
        sub = json.loads(part.read_text(encoding = "utf-8")) if part.is_file() else {}
        arm = (sub.get("arms") or {}).get(scheme) or {}
        arm["exit_code"] = rc
        arm["log_says_w8a8"] = "W8A8 torch._int_mm" in text
        arm["log_says_convrot"] = "ConvRot g" in text
        arm["log_says_weight_only"] = "weight-only (torch" in text
        summary["arms"][state] = arm
        for key in ("torch", "hip", "device", "arch"):
            summary.setdefault(key, sub.get(key))
        print(state, rc, json.dumps({k: arm.get(k) for k in ("loaded", "transformer_quant", "transformer_gib",
              "load_seconds", "speed", "log_says_w8a8", "log_says_weight_only")}), flush = True)
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
    for state in ("wo", "w8a8", "w8a8_rot"):
        for item in summary["arms"][state].get("images") or []:
            ref = out / "images_ref" / f"off_p{item.get('prompt_index')}.png"
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
    lines = ["| arm | GiB | load s | image s (each) | LPIPS vs bf16 | PSNR | log |", "|---|---|---|---|---|---|---|"]
    for state, arm in summary["arms"].items():
        imgs = arm.get("images") or []
        secs = ", ".join(str(i.get("seconds", "err")) for i in imgs)
        lp = ", ".join(str(i.get("lpips", "")) for i in imgs if "lpips" in i)
        ps = ", ".join(str(i.get("psnr", "")) for i in imgs if "psnr" in i)
        tag = ("W8A8 ConvRot" if arm.get("log_says_convrot") else "W8A8") if arm.get("log_says_w8a8") else ("weight-only" if arm.get("log_says_weight_only") else "-")
        lines.append(f"| {state} | {arm.get('transformer_gib')} | {arm.get('load_seconds')} | {secs} | {lp} | {ps} | {tag} |")
    table = "\n".join(lines)
    summary["table"] = table
    (out / "summary.json").write_text(json.dumps(summary, indent = 2, default = str), encoding = "utf-8")
    (out / "SUMMARY.md").write_text(table + "\n", encoding = "utf-8")
    print(table, flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

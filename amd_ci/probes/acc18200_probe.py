#!/usr/bin/env python3
"""Probe: does gfx1151 Auto INT8 attention (PR 18200's ROCM_AUTO_DITS) move Wan2.2-TI2V-5B / Qwen-Image-2.1-Turbo
output further from the same platform's bf16 than INT8 attention off does? Observes only; judged by
criteria/acc18200.py.

Head state only (head = current main; base is not measured). Per family, fresh process per arm, same prompts and
seeds (probes/i8vs/run.py + arm.py), in this order:
  A1  Studio default load (int8_attention unset = Auto)
  O1  int8_attention="off" per load
  B1  bf16 reference: int8_attention="off", transformer_quant="off", text_encoder_quant="off"
  A2  Auto again (A/A)
  O2  off again (A/A)
then probes/i8vs/score.py: LPIPS (alex) / PSNR per frame for the listed pairs. Public repos only, no token: every
HF token variable is removed from the arms' environment."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN")
I8_ENV = "UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL"
FAMS = {
    "wan": {"kind": "video", "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "load": {},
            "gen": {"width": 832, "height": 480, "num_frames": 33}},
    "q21t": {"kind": "image", "model": "Qwen/Qwen-Image-2.1-Turbo", "load": {"model_kind": "pipeline"},
             "gen": {"width": 1024, "height": 1024}},
}
ARMS = {"A": {}, "O": {"load": {"int8_attention": "off"}},
        "B": {"load": {"int8_attention": "off", "transformer_quant": "off", "text_encoder_quant": "off"}}}
ORDER = ["A1", "O1", "B1", "A2", "O2"]
PAIRS = ["A2:A1", "O2:O1", "A1:O1", "A2:O2", "A1:B1", "A2:B1", "O1:B1", "O2:B1"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--families", default = "wan,q21t")
    ap.add_argument("--video-prompts", type = int, default = 4)
    ap.add_argument("--image-prompts", type = int, default = 2)
    a = ap.parse_args()
    out = Path(a.out)
    if a.state != "head":
        out.write_text(json.dumps({"state": a.state, "skipped": "measured at head only"}), encoding = "utf-8")
        return 0
    for k in TOKEN_VARS:
        os.environ.pop(k, None)
    os.environ.pop(I8_ENV, None)
    root = out.resolve().parents[2] / "acc_root"
    root.mkdir(parents = True, exist_ok = True)
    arms_out = out.parent / "acc_arms"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "lpips"], check = False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scipy", "tqdm"], check = False)
    fams = []
    for name in [x for x in a.families.split(",") if x]:
        f = dict(FAMS[name])
        n = a.video_prompts if f["kind"] == "video" else a.image_prompts
        arms = []
        for tag in ORDER:
            arm = {"tag": tag, "prompts": n, "warmup": 0, **json.loads(json.dumps(ARMS[tag[0]]))}
            arms.append(arm)
        f.update(name = name, arms = arms, pairs = PAIRS, score_ref = "B1", arm_timeout = 5400, cleanup = True)
        fams.append(f)
    plan = {"run_id": "acc18200", "python": sys.executable,
            "backend": str(Path(a.checkout).resolve() / "studio" / "backend"), "families": fams}
    pj = out.parent / "acc_plan.json"
    pj.write_text(json.dumps(plan, indent = 1), encoding = "utf-8")
    log = out.parent / "acc_run.log"
    with open(log, "w", encoding = "utf-8") as fh:
        rc = subprocess.run([sys.executable, "-u", str(HERE / "i8vs" / "run.py"), "--plan", str(pj), "--root", str(root),
                             "--bench", str(HERE / "i8vs"), "--out", str(arms_out)],
                            stdout = fh, stderr = subprocess.STDOUT, env = dict(os.environ)).returncode
    res: dict = {"state": "head", "run_rc": rc, "families": {}}
    for f in fams:
        fr: dict = {"arms": {}, "scores": None}
        for arm in f["arms"]:
            rp = arms_out / f"{f['name']}_{arm['tag']}" / "record.json"
            if not rp.is_file():
                fr["arms"][arm["tag"]] = {"verdict": "missing"}
                continue
            r = json.loads(rp.read_text(encoding = "utf-8"))
            fr["arms"][arm["tag"]] = {
                "verdict": r.get("verdict"), "error": (r.get("error") or "")[:600], "load": arm.get("load") or {},
                "status": r.get("status"), "dits": r.get("dits"), "params": r.get("params"),
                "dispatch_status_load": r.get("dispatch_status_load"), "dispatch_status": r.get("dispatch_status"),
                "int8_status": r.get("int8_status"), "int8_stats": r.get("int8_stats"), "shapes": r.get("shapes"),
                "renders": [{k: x.get(k) for k in ("i", "seed", "wall_s", "int8_calls", "route_calls", "frames",
                                                    "frame_mean", "nonfinite")} for x in r.get("renders", [])]}
        sp = arms_out / f"scores_{f['name']}.json"
        if sp.is_file():
            s = json.loads(sp.read_text(encoding = "utf-8"))
            fr["scores"] = {f"{p['a']}|{p['b']}": {k: p.get(k) for k in ("n", "lpips_mean", "lpips_max", "psnr_mean")}
                            | {"rows": [{k: x.get(k) for k in ("i", "lpips", "psnr", "identical", "frames")} for x in p["rows"]]}
                            for p in s.get("vs_ref", []) + s.get("pairs", [])}
        res["families"][f["name"]] = fr
    out.write_text(json.dumps(res, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Probe: FLUX.3 Klein explicit int8 vs bf16 through Studio on this AMD GPU (observes only; judged by
criteria/f3win_e2e.py). Written for the Windows gfx1151 runners (no compile tier there: every arm is eager).

Both states: the FLUX.3 / RDNA W8A8 test files of the checkout (pytest, junit ids).
Head state only (its sibling ``base`` checkout ran first): fetch unsloth/flux-3-image-klein once (token-free) into the
job's HF cache, then fresh-process arms (f3win/arm.py), each --warmup untimed renders and --prompts timed renders at
1024^2, 16 steps, with fixed seeds, in --rounds rotating rounds of
   B   base tree, transformer_quant none (bf16), bf16 text encoder
   I   base tree, explicit int8 (what an explicit int8 request runs today), bf16 text encoder
   IA  base tree, explicit int8 with UNSLOTH_NATIVE_INT8_ACT=1 (W8A8 on the RDNA kernels, one act quant per Linear)
   HA  head tree, the same (W8A8 with Q / K / V / MLP input on one shared act quant per block)
   H   head tree, explicit int8 (round 1 only: the head's default on this OS)
Round 1 also profiles one denoiser call per arm (torch.profiler, device time per kernel). Then f3win/score.py:
LPIPS / PSNR of every arm vs B1 (same prompt and seed) and HA1 vs IA1, H1 vs I1 (bit-identity expected).

  python f3win_probe.py --state head --checkout states/head --out obs_head.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from i8auto_probe import arm_env, fetch, pytest as run_pytest  # noqa: E402

TOOLS = HERE / "f3win"
TESTS = ("tests/test_diffusion_flux_convrot_fused.py", "tests/test_diffusion_native_w8a8_rdna.py",
         "tests/test_diffusion_flux3_rdna_shared_quant.py")
TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN")
SPEC = {"repo": "unsloth/flux-3-image-klein", "kind": "image", "ignore": ["*.png", "*.jpg", "variants/*"],
        "gen": {"width": 1024, "height": 1024, "steps": 16}}
DENSE = {"model_kind": "pipeline", "transformer_quant": "none", "text_encoder_quant": "none"}
INT8 = {"model_kind": "pipeline", "transformer_quant": "int8", "text_encoder_quant": "none"}
ACT = {"UNSLOTH_NATIVE_INT8_ACT": "1"}
ARMS = {"B": ("base", DENSE, {}), "I": ("base", INT8, {}), "IA": ("base", INT8, ACT), "HA": ("head", INT8, ACT),
        "H": ("head", INT8, {})}
ORDERS = (("B", "I", "IA", "HA", "H"), ("I", "IA", "HA", "B"), ("IA", "HA", "B", "I"))


def run_arm(tree: Path, tag: str, load: dict, env_extra: dict, outdir: Path, hf_home: Path, a, prompts: int,
            warmup: int, prof: int = -1, timeout: int = 5400) -> dict:
    backend = tree / "studio" / "backend"
    cmd = [sys.executable, "-u", str(TOOLS / "arm.py"), "--root", str(backend), "--kind", "image",
           "--model", SPEC["repo"], "--tag", f"klein_{tag}", "--out", str(outdir), "--load", json.dumps(load),
           "--gen", json.dumps(SPEC["gen"]), "--prompts", str(prompts), "--warmup", str(warmup), "--tf", "none",
           "--prof-step", str(prof)]
    log = outdir / f"klein_{tag}.log"
    t0 = time.time()
    try:
        with open(log, "w", encoding = "utf-8") as fh:
            rc = subprocess.run(cmd, cwd = backend, stdout = fh, stderr = subprocess.STDOUT,
                                env = arm_env(hf_home, env_extra), timeout = timeout).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    try:
        rec = json.loads((outdir / f"klein_{tag}" / "record.json").read_text(encoding = "utf-8"))
    except Exception as exc:  # noqa: BLE001
        rec = {"verdict": "error", "error": f"no record: {exc}"[:300]}
    rec["rc"] = rc
    rec["arm_seconds"] = round(time.time() - t0, 1)
    text = log.read_text(encoding = "utf-8", errors = "replace") if log.is_file() else ""
    rec["log_tail"] = text[-3000:]
    rec["quant_lines"] = [x[:600] for x in text.splitlines()
                          if any(k in x for k in ("native_quant", "flux_convrot_fused", "W8A8", "rocm_compile",
                                                  "transformer_quant", "int8"))][:200]
    rec.pop("proc_load", None)
    rec.pop("proc_end", None)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--rounds", type = int, default = 3)
    ap.add_argument("--prompts", type = int, default = 2)
    ap.add_argument("--warmup", type = int, default = 2)
    ap.add_argument("--prof-step", type = int, default = 5)
    a = ap.parse_args()
    for k in TOKEN_VARS:
        os.environ.pop(k, None)
    out = Path(a.out).resolve()
    checkout = Path(a.checkout).resolve()
    obs: dict = {"state": a.state, "python": sys.executable, "checkout": str(checkout), "platform": sys.platform}

    def save():
        out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")

    try:
        import torch
        obs["torch"] = torch.__version__
        obs["hip"] = getattr(torch.version, "hip", None)
        obs["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        obs["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if obs["gpu"] else None
        import triton
        obs["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        obs["env_error"] = repr(exc)[:500]
    try:
        obs["rev"] = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output = True,
                                    text = True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    import i8auto_probe
    i8auto_probe.TESTS = TESTS
    obs["pytest"] = run_pytest(checkout, sys.executable, out)
    save()
    if a.state != "head":
        return 0
    trees = {"base": checkout.parent / "base", "head": checkout}
    if not (trees["base"] / "studio").is_dir():
        obs["driver_error"] = f"no base checkout at {trees['base']}"
        save()
        return 0
    work = Path(os.environ.get("AMD_CI_WORK") or out.parent) / "f3win_work"
    work.mkdir(parents = True, exist_ok = True)
    hf_home = work / "hf"
    arms_dir = out.parent / "f3win_arms"
    arms_dir.mkdir(parents = True, exist_ok = True)
    obs["lpips_install"] = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-deps", "lpips"],
        capture_output = True, text = True).returncode
    obs["fetch"] = []
    got = fetch(SPEC["repo"], SPEC["ignore"], hf_home, obs["fetch"])
    obs["arms"] = {}
    save()
    if got["rc"] != 0:
        obs["driver_error"] = "fetch failed"
        save()
        return 0
    # throwaway arms first: the job's first processes pay one-off disk caches (2.4x slower first arm on Windows)
    for wtag, side in (("W", "B"), ("WI", "I")):
        tree, load, env = ARMS[side]
        rec = run_arm(trees[tree], wtag, load, env, arms_dir, hf_home, a, prompts = 0, warmup = 2)
        rec.update(side = "W", round = 0)
        obs["arms"][wtag] = rec
        save()
    for r in range(a.rounds):
        for side in ORDERS[r % len(ORDERS)]:
            tree, load, env = ARMS[side]
            tag = f"{side}{r + 1}"
            rec = run_arm(trees[tree], tag, load, env, arms_dir, hf_home, a, prompts = a.prompts, warmup = a.warmup,
                          prof = a.prof_step if r == 0 else -1)
            rec.update(side = side, round = r + 1)
            obs["arms"][tag] = rec
            save()
    t0 = time.time()
    try:
        sc = subprocess.run([sys.executable, "-u", str(TOOLS / "score.py"), "--out", str(arms_dir), "--family",
                             "klein", "--ref", "B1", "--pairs", "HA1:IA1,H1:I1,I2:I1,IA2:IA1", "--kind", "image"],
                            capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                            env = arm_env(hf_home, {}), timeout = 3600)
        obs["score_rc"] = sc.returncode
        obs["score_tail"] = (sc.stdout + sc.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        obs["score_rc"] = "timeout"
    obs["score_seconds"] = round(time.time() - t0, 1)
    try:
        obs["scores"] = json.loads((arms_dir / "scores_klein.json").read_text(encoding = "utf-8"))
    except Exception as exc:  # noqa: BLE001
        obs["scores"] = {"error": repr(exc)[:300]}
    save()
    shutil.rmtree(hf_home / "hub", ignore_errors = True)
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())

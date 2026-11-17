#!/usr/bin/env python3
"""Probe: should Studio's INT8 attention (Triton kernel on ROCm) engage automatically on gfx1151? Observes only;
judged by criteria/i8auto_e2e.py.

Both states: the INT8 attention test files of the checkout (pytest, junit ids).
Head state only (the base ran first; its tree is the sibling ``base`` checkout):
  1. i8auto/shapes.py on the head tree: the module's own accuracy check per head dim, and per call at real shapes
     (synthetic probe inputs) INT8 vs torch SDPA vs flash_attn, accuracy vs fp32 and race timing; MiniMax-H3-Image
     only here (no e2e: 135 GB of weights).
  2. per family (FLUX.3 Klein, Z-Image-Turbo, Qwen-Image-2.1-Turbo, Wan2.2-TI2V-5B): fetch the public weights once
     (token-free) into the job's HF cache, then fresh-process arms (i8auto/arm.py), each one warm-up render and
     --prompts timed renders with fixed seeds, in --rounds rotating rounds of
        N  base tree, switch unset (native attention on ROCm: what users get today)
        A  head tree, switch unset (the automatic path under test)
        F  base tree, UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL=1 (INT8 forced)
     Round 1: N saves teacher-forcing inputs / outputs at 3 denoise steps, A and F compare against them (per-step
     rel L2 of the DiT output vs native). One more arm P (base tree, forced, compile off, i8auto/i8probe.py): per call
     rel L2 vs fp32 of the kernel / SDPA / flash on the real activations, attention-sink share, and a race of each
     distinct shape on its own inputs. Then i8auto/score.py: LPIPS of every arm vs N round 1, same prompt and seed.
     The family's weights are deleted before the next family.

  python i8auto_probe.py --state head --checkout states/head --out obs_head.json [--families klein,zimage,q21t,wan]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE / "i8auto"
TESTS = (
    "tests/test_diffusion_int8_attention.py", "tests/test_diffusion_int8_attention_rocm.py",
    "tests/test_diffusion_int8_attention_dispatch.py", "tests/test_diffusion_int8_attention_auto.py",
    "tests/test_diffusion_int8_attention_triton_fallback.py",
    "tests/test_diffusion_int8_attention_triton_compile_key.py",
    "tests/test_diffusion_int8_attention_triton_device.py",
    "tests/test_diffusion_int8_attention_triton_offsets.py", "tests/test_diffusion_int8_attention_long_probe.py",
)
TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN")
I8_ENV = "UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL"
FAMILIES = {
    "klein": {"repo": "unsloth/flux-3-image-klein", "kind": "image", "ignore": ["*.png", "*.jpg", "variants/*"],
              "load": {"model_kind": "pipeline"}, "gen": {"width": 1024, "height": 1024}},
    "zimage": {"repo": "Tongyi-MAI/Z-Image-Turbo", "kind": "image", "ignore": ["*.png", "*.jpg", "*.webp", "assets/*"],
               "load": {"model_kind": "pipeline"}, "gen": {"width": 1024, "height": 1024}},
    "q21t": {"repo": "Qwen/Qwen-Image-2.1-Turbo", "kind": "image", "ignore": ["*.png", "*.jpg", "assets/*"],
             "load": {"model_kind": "pipeline"}, "gen": {"width": 1024, "height": 1024}},
    # 832x480x33 frames: 9 latent frames x 30 x 52 = 14040 tokens (the earlier gfx1151 Wan run's shape)
    "wan": {"repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "kind": "video",
            "ignore": ["*.png", "*.jpg", "*.gif", "*.mp4", "assets/*", "examples/*"],
            "load": {}, "gen": {"width": 832, "height": 480, "num_frames": 33, "steps": 12}},
}
ORDERS = (("N", "A", "F"), ("A", "F", "N"), ("F", "N", "A"))


def pytest(checkout: Path, python: str, out: Path) -> dict:
    wd = checkout / "studio" / "backend"
    junit = out.with_name(out.stem + ".junit.xml")
    junit.unlink(missing_ok = True)
    tests = [t for t in TESTS if (wd / t).is_file()]
    cmd = [python, "-m", "pytest", "-q", "-rfEs", "--color=no", "-o", "addopts=", "-p", "no:cacheprovider",
           f"--junitxml={junit}", "-o", "junit_family=xunit1", *tests]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd = wd, capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                           timeout = 3600)
        rc, tail = r.returncode, (r.stdout + r.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        rc, tail = "timeout", ""
    res = {"rc": rc, "seconds": round(time.time() - t0, 1), "files": tests, "tail": tail, "cases": {}}
    if junit.is_file():
        for tc in ET.parse(junit).getroot().iter("testcase"):
            name = f"{tc.get('file') or tc.get('classname')}::{tc.get('name')}"
            kind = "passed"
            for child in tc:
                if child.tag in ("failure", "error"):
                    kind = "failed"
                elif child.tag == "skipped":
                    kind = "skipped"
            res["cases"][name] = kind
    for kind in ("passed", "failed", "skipped"):
        res[kind] = sum(v == kind for v in res["cases"].values())
    return res


def arm_env(hf_home: Path, extra: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in TOKEN_VARS and k not in ("CI", I8_ENV, "PYTHONPATH")}
    env.update({"HF_HOME": str(hf_home), "PYTHONUNBUFFERED": "1", "UNSLOTH_DISABLE_AUTO_UPDATES": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_ETAG_TIMEOUT": "60", "HF_HUB_DOWNLOAD_TIMEOUT": "120",
                "UNSLOTH_DIFFUSION_ALLOW_OVERSIZED_LOAD": "1", "PYTHONIOENCODING": "utf-8"})
    env.update(extra)
    return env


def fetch(repo: str, ignore: list, hf_home: Path, log: list) -> dict:
    code = ("import sys, json, time; from huggingface_hub import snapshot_download\n"
            "t=time.time()\n"
            "for i in range(6):\n"
            "    try:\n"
            "        p=snapshot_download(sys.argv[1], ignore_patterns=json.loads(sys.argv[2]), token=False,"
            " max_workers=8); break\n"
            "    except Exception as e:\n"
            "        print('retry', i, repr(e)[:300], flush=True); time.sleep(20)\n"
            "else:\n"
            "    raise SystemExit(3)\n"
            "print('FETCHED', p, round(time.time()-t,1), flush=True)\n")
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", code, repo, json.dumps(ignore)], capture_output = True, text = True,
                       encoding = "utf-8", errors = "replace", env = arm_env(hf_home, {}), timeout = 7200)
    res = {"repo": repo, "rc": r.returncode, "seconds": round(time.time() - t0, 1),
           "tail": (r.stdout + r.stderr)[-1500:]}
    log.append(res)
    return res


def run_arm(tree: Path, fam: str, spec: dict, tag: str, env_extra: dict, outdir: Path, hf_home: Path, a,
            tf: str = "none", tf_ref: Path | None = None, prompts: int | None = None, warmup: int | None = None,
            timeout: int = 5400) -> dict:
    backend = tree / "studio" / "backend"
    cmd = [sys.executable, "-u", str(TOOLS / "arm.py"), "--root", str(backend), "--kind", spec["kind"],
           "--model", spec["repo"], "--tag", f"{fam}_{tag}", "--out", str(outdir),
           "--load", json.dumps(spec["load"]), "--gen", json.dumps(spec["gen"]),
           "--prompts", str(a.prompts if prompts is None else prompts),
           "--warmup", str(a.warmup if warmup is None else warmup), "--tf", tf, "--tf-prompts", "1",
           "--save-frames-every", "4"]
    if tf_ref is not None:
        cmd += ["--tf-ref", str(tf_ref)]
    log = outdir / f"{fam}_{tag}.log"
    t0 = time.time()
    try:
        with open(log, "w", encoding = "utf-8") as fh:
            rc = subprocess.run(cmd, cwd = backend, stdout = fh, stderr = subprocess.STDOUT,
                                env = arm_env(hf_home, env_extra), timeout = timeout).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    rec_path = outdir / f"{fam}_{tag}" / "record.json"
    try:
        rec = json.loads(rec_path.read_text(encoding = "utf-8"))
    except Exception as exc:  # noqa: BLE001
        rec = {"verdict": "error", "error": f"no record: {exc}"[:300]}
    rec["rc"] = rc
    rec["arm_seconds"] = round(time.time() - t0, 1)
    text = log.read_text(encoding = "utf-8", errors = "replace") if log.is_file() else ""
    rec["log_tail"] = text[-3000:]
    # Studio's loggers are structlog (stdout), not stdlib: keep the INT8 attention / compile lines from the log
    rec["i8_lines"] = [x[:600] for x in text.splitlines()
                       if any(k in x for k in ("int8_attention", "rocm_compile", "Triton", "triton"))][:300]
    probe = outdir / f"{fam}_{tag}" / "probe.jsonl"
    if probe.is_file():
        rec["probe_rows"] = [json.loads(x) for x in probe.read_text(encoding = "utf-8").splitlines() if x.strip()]
    rec.pop("proc_load", None)
    rec.pop("proc_end", None)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--families", default = "klein,zimage,q21t,wan")
    ap.add_argument("--rounds", type = int, default = 3)
    ap.add_argument("--prompts", type = int, default = 2)
    ap.add_argument("--warmup", type = int, default = 1)
    ap.add_argument("--no-pytest", action = "store_true")
    ap.add_argument("--keep-models", action = "store_true")
    a = ap.parse_args()
    for k in TOKEN_VARS:
        os.environ.pop(k, None)
    out = Path(a.out).resolve()
    checkout = Path(a.checkout).resolve()
    obs: dict = {"state": a.state, "python": sys.executable, "checkout": str(checkout)}

    def save():
        out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")

    try:
        import torch
        obs["torch"] = torch.__version__
        obs["hip"] = getattr(torch.version, "hip", None)
        obs["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        obs["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if obs["gpu"] else None
        obs["total_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if obs["gpu"] else None
    except Exception as exc:  # noqa: BLE001
        obs["env_error"] = repr(exc)[:500]
    try:
        r = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output = True, text = True)
        obs["rev"] = r.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    if not a.no_pytest:
        obs["pytest"] = pytest(checkout, sys.executable, out)
    save()
    if a.state != "head":
        return 0
    base = checkout.parent / "base"
    if not (base / "studio").is_dir():
        obs["driver_error"] = f"no base checkout at {base}"
        save()
        return 0
    trees = {"N": base, "A": checkout, "F": base, "P": base}
    envs = {"N": {}, "A": {}, "F": {I8_ENV: "1"},
            "P": {I8_ENV: "1", "UNSLOTH_DIFFUSION_ROCM_COMPILE": "0", "I8AUTO_PROBE": "1"}}
    work = Path(os.environ.get("AMD_CI_WORK") or out.parent) / "i8auto_work"
    work.mkdir(parents = True, exist_ok = True)
    hf_home = work / "hf"
    arms_dir = out.parent / "i8auto_arms"
    arms_dir.mkdir(parents = True, exist_ok = True)
    # 1. per call at real shapes, synthetic inputs (+ H3-Image)
    sh_out = arms_dir / "shapes.json"
    t0 = time.time()
    try:
        with open(arms_dir / "shapes.log", "w", encoding = "utf-8") as fh:
            obs["shapes_rc"] = subprocess.run(
                [sys.executable, "-u", str(TOOLS / "shapes.py"), "--root", str(checkout / "studio" / "backend"),
                 "--out", str(sh_out)], stdout = fh, stderr = subprocess.STDOUT,
                env = arm_env(hf_home, {}), cwd = checkout / "studio" / "backend", timeout = 3600).returncode
    except subprocess.TimeoutExpired:
        obs["shapes_rc"] = "timeout"
    obs["shapes_seconds"] = round(time.time() - t0, 1)
    try:
        obs["shapes"] = json.loads(sh_out.read_text(encoding = "utf-8"))
    except Exception as exc:  # noqa: BLE001
        obs["shapes"] = {"error": repr(exc)[:300]}
    save()
    # lpips for the scorer (--no-deps: never replaces the ROCm torch)
    obs["lpips_install"] = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-deps", "lpips"],
        capture_output = True, text = True).returncode
    obs["fetch"] = []
    obs["families"] = {}
    for fam in [f for f in a.families.split(",") if f]:
        spec = FAMILIES[fam]
        fobs: dict = {"repo": spec["repo"], "kind": spec["kind"], "arms": {}}
        obs["families"][fam] = fobs
        got = fetch(spec["repo"], spec["ignore"], hf_home, obs["fetch"])
        if got["rc"] != 0:
            fobs["error"] = "fetch failed"
            save()
            continue
        tf_ref = work / "tf" / fam
        for r in range(a.rounds):
            for side in ORDERS[r % len(ORDERS)]:
                tag = f"{side}{r + 1}"
                tf = "none"
                if r == 0:
                    tf = "save" if side == "N" else "compare"
                rec = run_arm(trees[side], fam, spec, tag, envs[side], arms_dir, hf_home, a, tf = tf, tf_ref = tf_ref)
                rec.update(side = side, round = r + 1)
                fobs["arms"][tag] = rec
                save()
        rec = run_arm(trees["P"], fam, spec, "P", envs["P"], arms_dir, hf_home, a, prompts = 1, warmup = 0)
        rec.update(side = "P", round = 0)
        fobs["arms"]["P"] = rec
        save()
        t0 = time.time()
        try:
            sc = subprocess.run([sys.executable, "-u", str(TOOLS / "score.py"), "--out", str(arms_dir), "--family",
                                 fam, "--ref", "N1", "--kind", spec["kind"]], capture_output = True, text = True,
                                encoding = "utf-8", errors = "replace", env = arm_env(hf_home, {}), timeout = 3600)
            fobs["score_rc"] = sc.returncode
            fobs["score_tail"] = (sc.stdout + sc.stderr)[-2000:]
        except subprocess.TimeoutExpired:
            fobs["score_rc"] = "timeout"
        fobs["score_seconds"] = round(time.time() - t0, 1)
        try:
            fobs["scores"] = json.loads((arms_dir / f"scores_{fam}.json").read_text(encoding = "utf-8"))
        except Exception as exc:  # noqa: BLE001
            fobs["scores"] = {"error": repr(exc)[:300]}
        save()
        if not a.keep_models:
            shutil.rmtree(hf_home / "hub", ignore_errors = True)
            shutil.rmtree(tf_ref, ignore_errors = True)
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build a self-contained bundle: the harness + the exact kernel sources under test + the verifier.

    python snapshot.py [--out DIR] [--h3-ref origin/studio-h3-vae-fast] [--bias-ref origin/studio-nvfp4-kernels]

Layout (what the cases look for, relative to the bundle root):
  portability/                       this harness
  trees/h3/studio/backend/core/inference/video_minimax_h3_vae.py        PR 11801
  trees/bias/studio/backend/core/inference/diffusion_nvfp4_*.py         PR 10731 (bias kernel + backend resolver)
  trees/main/studio/backend/core/inference/diffusion_speed.py (+ cuda_graph, device)   Studio main, for the tier settings
  verify/verify.py                   scripts/triton_vs_inductor/verify.py when it exists
  gemm_shapes.json                   outputs/bottlenecks_q21_h3/gemm_shapes.json when it exists
  MANIFEST.json                      refs and shas
Sources are read with ``git show <ref>:<path>`` from the given repo, so no checkout is needed.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.environ.get("WORKSPACE") or os.path.abspath(os.path.join(HERE, "..", ".."))
INF = "studio/backend/core/inference"
BIAS_FILES = ["diffusion_nvfp4_bias.py", "diffusion_nvfp4_ops.py", "diffusion_nvfp4_flag.py",
              "diffusion_nvfp4_dispatch.py"]
MAIN_FILES = ["diffusion_speed.py", "diffusion_cuda_graph.py", "diffusion_device.py"]


def git(repo, *a) -> str:
    return subprocess.run(["git", "-C", repo, *a], capture_output = True, text = True, check = True).stdout


def put(repo, ref, rel, dst):
    os.makedirs(os.path.dirname(dst), exist_ok = True)
    with open(dst, "w", encoding = "utf-8") as f:
        f.write(git(repo, "show", f"{ref}:{rel}"))


def build(out: str, repo: str, h3_ref: str, bias_ref: str, main_ref: str, fetch: bool) -> dict:
    if fetch:
        for r in (h3_ref, bias_ref, main_ref):
            if r.startswith("origin/"):
                subprocess.run(["git", "-C", repo, "fetch", "-q", "origin", r.split("/", 1)[1]], check = False)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    shutil.copytree(HERE, os.path.join(out, "portability"),
                    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "notebooks", "ci_*"))
    man = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "repo": repo}
    put(repo, h3_ref, f"{INF}/video_minimax_h3_vae.py", os.path.join(out, "trees", "h3", INF, "video_minimax_h3_vae.py"))
    man["h3"] = {"ref": h3_ref, "sha": git(repo, "rev-parse", h3_ref).strip()}
    for f in BIAS_FILES:
        put(repo, bias_ref, f"{INF}/{f}", os.path.join(out, "trees", "bias", INF, f))
    man["bias"] = {"ref": bias_ref, "sha": git(repo, "rev-parse", bias_ref).strip()}
    for f in MAIN_FILES:
        try:
            put(repo, main_ref, f"{INF}/{f}", os.path.join(out, "trees", "main", INF, f))
        except subprocess.CalledProcessError:
            pass
    man["main"] = {"ref": main_ref, "sha": git(repo, "rev-parse", main_ref).strip()}
    v = os.path.join(WS, "scripts", "triton_vs_inductor", "verify.py")
    if os.path.exists(v):
        os.makedirs(os.path.join(out, "verify"), exist_ok = True)
        shutil.copy(v, os.path.join(out, "verify", "verify.py"))
        man["verify"] = {"path": v, "mtime": time.ctime(os.path.getmtime(v))}
    g = os.path.join(WS, "outputs", "bottlenecks_q21_h3", "gemm_shapes.json")
    if os.path.exists(g):
        shutil.copy(g, os.path.join(out, "gemm_shapes.json"))
        man["gemm_shapes"] = g
    with open(os.path.join(out, "MANIFEST.json"), "w", encoding = "utf-8") as f:
        json.dump(man, f, indent = 1)
    return man


def tar_bytes(bundle: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj = buf, mode = "w:gz") as t:
        t.add(bundle, arcname = "bundle")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default = os.path.join(WS, "temp", "portability_bundle"))
    ap.add_argument("--repo", default = os.path.join(WS, "wt_port_h3"), help = "any unslothai/unsloth checkout")
    ap.add_argument("--h3-ref", default = "origin/studio-h3-vae-fast")
    ap.add_argument("--bias-ref", default = "origin/studio-nvfp4-kernels")
    ap.add_argument("--main-ref", default = "origin/main")
    ap.add_argument("--no-fetch", action = "store_true")
    a = ap.parse_args()
    man = build(a.out, a.repo, a.h3_ref, a.bias_ref, a.main_ref, not a.no_fetch)
    print(json.dumps(man, indent = 1))
    print("bundle:", a.out, f"({len(tar_bytes(a.out)) / 1024:.0f} KiB gz)")


if __name__ == "__main__":
    main()

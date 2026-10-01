#!/usr/bin/env python3
"""ComfyUI for the ``comfyui`` backend: a clone at a pinned commit, its own venv, and per-cell model wiring.

  ensure(comfy_commit=None, python=None, torch=None, index=None, kitchen=True, sage=False)
      -> {"python": <venv python>, "comfy_dir": <clone>, "commit": <sha>, ...}

Clone: comfyanonymous/ComfyUI into $WORKSPACE/temp/diffusion_bench/src/ComfyUI@<sha10>. ``comfy_commit`` is a sha, a
tag, a branch, or "latest" (origin/master at call time); None is DEFAULT_COMMIT, the tree the Studio vs ComfyUI
Qwen-Image-2.1 all-optimisations benchmark ran on (it has TextEncodeQwenImage21 and the int8 convrot loader).

Venv: envs.ensure_venv("comfyui", ...) with the clone's requirements.txt minus the torch family (torch / torchvision
come from the detected wheel index; torchaudio is left out because a PyPI torchaudio can pull a torch that does
not match the index build, and ComfyUI only needs it for audio nodes). comfy-kitchen is in ComfyUI's own
requirements (the int8 / fp8 / nvfp4 kernels); ``sage=True`` adds sageattention for --use-sage-attention.

Reuse instead of building: DIFFUSION_BENCH_COMFY_PYTHON=<python> and/or DIFFUSION_BENCH_COMFY_DIR=<clone>. Either
alone is honoured (the other is still resolved normally), so an existing clone can run in a fresh venv and
vice versa.

Models: ``wire_models(files, dest)`` symlinks the cell's files into <dest>/<folder>/ (diffusion_models,
text_encoders, vae, checkpoints, diffusers, loras) and writes <dest>/extra_model_paths.yaml with is_default: true,
passed to the server with --extra-model-paths-config, so the clone's own models/ tree is never written to and a
same-named file there cannot shadow the cell's.

Usage:
  python diffusion_bench/setup_comfyui.py                    # clone + venv at the default commit, print JSON
  python diffusion_bench/setup_comfyui.py --commit latest --plan
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

REPO_URL = "https://github.com/comfyanonymous/ComfyUI"
# master on 2026-09-22 "Port some optimizations to flux model family. (#16488)": the Qwen-Image-2.1 legb / allopt tree.
DEFAULT_COMMIT = "b5cc8830279eae909a59de030af1e50761c36751"
SRC = C.WS / "temp" / "diffusion_bench" / "src"
MIRROR = SRC / "ComfyUI-mirror.git"
MODEL_FOLDERS = ("diffusion_models", "text_encoders", "vae", "checkpoints", "diffusers", "loras", "clip_vision",
                 "model_patches", "upscale_models")
_TORCH_FAMILY = re.compile(r"^(torch|torchvision|torchaudio)\s*([<>=!~;].*)?$", re.I)


def log(msg: str) -> None:
    C.log(f"[setup_comfyui] {msg}")


def git(*args, capture: bool = False) -> str:
    cmd = ["git", *[str(a) for a in args]]
    if not capture:
        log("$ " + " ".join(cmd))
    out = subprocess.run(cmd, check = True, capture_output = capture, text = True)
    return (out.stdout or "").strip() if capture else ""


def resolve_commit(ref: Optional[str]) -> str:
    ref = ref or DEFAULT_COMMIT
    SRC.mkdir(parents = True, exist_ok = True)
    if not MIRROR.exists():
        git("clone", "--bare", "-q", REPO_URL, MIRROR)
    want = "master" if ref in ("latest", "master") else ref
    if ref in ("latest", "master"):
        git("-C", MIRROR, "fetch", "-q", "origin", "+refs/heads/*:refs/heads/*", "--tags")
    for attempt in range(2):
        try:
            return git("-C", MIRROR, "rev-parse", "--verify", f"{want}^{{commit}}", capture = True)
        except subprocess.CalledProcessError:
            if attempt == 0:
                git("-C", MIRROR, "fetch", "-q", "origin", "+refs/heads/*:refs/heads/*", "--tags")
    git("-C", MIRROR, "fetch", "-q", "origin", want)
    return git("-C", MIRROR, "rev-parse", "--verify", "FETCH_HEAD^{commit}", capture = True)


def clone_dir(sha: str) -> Path:
    return SRC / f"ComfyUI@{sha[:10]}"


def ensure_clone(ref: Optional[str]) -> tuple[Path, str]:
    sha = resolve_commit(ref)
    root = clone_dir(sha)
    if not (root / "main.py").exists():
        git("clone", "-q", "--no-checkout", MIRROR, root)
        git("-C", root, "remote", "set-url", "origin", REPO_URL)
        git("-C", root, "checkout", "-q", sha)
    return root, sha


def requirements(comfy_dir: Path, sha: Optional[str] = None) -> list:
    """requirements.txt minus comments and the torch family (torch/torchvision come from the index). Read from the
    bare mirror at ``sha`` when the clone does not exist yet (plan mode)."""
    out = []
    path = comfy_dir / "requirements.txt"
    text = path.read_text() if path.exists() else git("-C", MIRROR, "show", f"{sha}:requirements.txt", capture = True)
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not _TORCH_FAMILY.match(line):
            out.append(line)
    return out


def ensure(comfy_commit: Optional[str] = None, python: Optional[str] = None, torch: Optional[str] = None,
           index: Optional[str] = None, kitchen: bool = True, sage: bool = False, plan: bool = False, **_) -> dict:
    env_py = os.environ.get("DIFFUSION_BENCH_COMFY_PYTHON")
    env_dir = os.environ.get("DIFFUSION_BENCH_COMFY_DIR")
    if env_dir:
        comfy_dir = Path(C.expand_env(env_dir)).resolve()
        if not (comfy_dir / "main.py").exists():
            raise FileNotFoundError(f"DIFFUSION_BENCH_COMFY_DIR={env_dir}: no ComfyUI main.py there")
        sha = C.git_rev(comfy_dir) or "unknown"
        try:
            sha = git("-C", comfy_dir, "rev-parse", "HEAD", capture = True)
        except Exception:  # noqa: BLE001
            pass
    elif plan:
        sha = resolve_commit(comfy_commit)
        comfy_dir = clone_dir(sha)
    else:
        comfy_dir, sha = ensure_clone(comfy_commit)
    info = {"comfy_dir": str(comfy_dir), "commit": sha, "source": "env" if env_dir else "clone"}
    if env_py:
        py = Path(C.expand_env(env_py))
        if not py.exists():
            raise FileNotFoundError(f"DIFFUSION_BENCH_COMFY_PYTHON={env_py} does not exist")
        return {**info, "python": str(py), "venv_source": "env"}
    import envs

    reqs = requirements(comfy_dir, sha)
    if not kitchen:
        reqs = [r for r in reqs if not r.lower().startswith("comfy-kitchen")]
    extra = ["sageattention"] if sage else []
    if plan:
        return {**info, "python": None, "plan": {"profile": "comfyui", "python": python or "3.12",
                                                "torch": torch or "torch", "index": index or envs.detect_torch_index(),
                                                "packages": reqs, "extra": extra}}
    venv = envs.ensure_venv("comfyui", python = python or "3.12", torch_spec = torch or "torch", index = index,
                            packages = reqs, extra = extra)
    return {**info, "python": venv["python"], "venv": venv.get("venv"), "venv_source": "envs"}


# ---------------------------------------------------------------------------------------------- model wiring
def wire_models(files: dict, dest: Path) -> Path:
    """``files`` maps a ComfyUI model folder to paths (a str or a list). Each path is symlinked into
    <dest>/<folder>/<basename> (a directory for ``diffusers``); returns the extra_model_paths.yaml to pass with
    --extra-model-paths-config. Missing sources raise FileNotFoundError naming the folder, before any server starts."""
    dest = Path(dest)
    for folder, paths in files.items():
        if not paths:
            continue
        folder = {"unet": "diffusion_models", "clip": "text_encoders"}.get(folder, folder)
        sub = dest / folder
        sub.mkdir(parents = True, exist_ok = True)
        for p in [paths] if isinstance(paths, (str, Path)) else paths:
            src = Path(C.expand_env(str(p)))
            if not src.exists():
                raise FileNotFoundError(f"ComfyUI {folder} file not found: {src}")
            link = sub / src.name
            if link.is_symlink() or link.exists():
                if link.is_symlink() and Path(os.readlink(link)) == src.resolve():
                    continue
                link.unlink()
            link.symlink_to(src.resolve(), target_is_directory = src.is_dir())
    lines = ["diffusion_bench:", f"    base_path: {dest.resolve()}", "    is_default: true"]
    lines += [f"    {f}: {f}" for f in MODEL_FOLDERS]
    yaml = dest / "extra_model_paths.yaml"
    yaml.write_text("\n".join(lines) + "\n")
    return yaml


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", default = None, help = f"sha / tag / branch / latest (default {DEFAULT_COMMIT[:10]})")
    ap.add_argument("--python", default = None)
    ap.add_argument("--torch", default = None)
    ap.add_argument("--index", default = None)
    ap.add_argument("--no-kitchen", action = "store_true")
    ap.add_argument("--sage", action = "store_true", help = "add sageattention to the venv")
    ap.add_argument("--plan", action = "store_true", help = "resolve the commit and print the venv plan; build nothing")
    args = ap.parse_args()
    print(json.dumps(ensure(args.commit, args.python, args.torch, args.index, kitchen = not args.no_kitchen,
                            sage = args.sage, plan = args.plan), indent = 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

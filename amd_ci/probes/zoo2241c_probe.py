#!/usr/bin/env python3
"""Observe-only probe for check C of unsloth-zoo PR 2241 (rand4x off) on gfx1151, across torch versions.

Runs only at the `head` state (other states record `skipped`): the stock arms never import unsloth_zoo, and the
patched arm loads the head's temporary_patches/inductor_rand4x.py by path. For each torch lane, sequentially, a
FRESH minimal venv (torch + its ROCm Triton + numpy, nothing else) is built, rand4x_c_diag.py runs its arms, and
the venv is deleted before the next lane:
  t212  torch 2.12.1+rocm7.2  (download.pytorch.org/whl/rocm7.2):  no 4x path       arms stock, fallback_random
  t214  torch 2.14.1+rocm7.14 (download.pytorch.org/whl/rocm7.14): 4x path on        arms stock, patched, fallback_random,
        stock_fill_a -> patched_warm (PR head patch), stock_fill_b -> patched_fixed_warm (amd_ci/patches thread-tag fix)
  t215  torch 2.15.0+rocm7.14 (download.pytorch.org/whl/rocm7.14): 4x path opt-in    arms stock, rand4x_on, fallback_random
JSON via --out; Inductor output code per lane / arm under <state>_artifacts/code_<lane>/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zoo_venv  # noqa: E402

LANES = [
    ("t212", "torch==2.12.1+rocm7.2", "https://download.pytorch.org/whl/rocm7.2", "stock,fallback_random"),
    ("t214", "torch==2.14.1+rocm7.14", "https://download.pytorch.org/whl/rocm7.14",
     "stock,patched,fallback_random,stock_fill_a,patched_warm,stock_fill_b,patched_fixed_warm"),
    ("t215", "torch==2.15.0+rocm7.14", "https://download.pytorch.org/whl/rocm7.14", "stock,rand4x_on,fallback_random"),
]
PATCH = "unsloth_zoo/temporary_patches/inductor_rand4x.py"
# PR head's patch plus the cache key tag set in every thread (local branch rand4x-thread-global-tag).
FIXED_PATCH = HERE.parent / "patches" / "zoo2241_inductor_rand4x_threadtag.py"


def build(venv: Path, pin: str, index: str, log: Path) -> dict:
    if venv.exists():
        shutil.rmtree(venv, ignore_errors = True)
    uv = shutil.which("uv")
    py = venv / "bin" / "python"
    if uv:
        steps = [("venv", [uv, "venv", "-q", "--seed", "--python", sys.executable, str(venv)]),
                 ("torch", [uv, "pip", "install", "--python", str(py), "--index-url", index, "--extra-index-url",
                            zoo_venv.PYPI, "--index-strategy", "unsafe-best-match", pin, "numpy"])]
    else:
        steps = [("venv", [sys.executable, "-m", "venv", str(venv)]),
                 ("torch", [str(py), "-m", "pip", "install", "-q", "--index-url", index, "--extra-index-url",
                            zoo_venv.PYPI, pin, "numpy"])]
    info: dict = {"venv": str(venv), "python": str(py), "pin": pin, "index": index, "uv": uv, "steps": {}}
    for name, cmd in steps:
        rc = zoo_venv.sh(cmd, log, timeout = 3000)
        info["steps"][name] = rc
        if rc != 0:
            info["error"] = f"step {name} rc={rc}"
            info["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-3000:]
            return info
    p = subprocess.run([str(py), "-c", zoo_venv.ENV_CODE], capture_output = True, text = True)
    line = next((ln for ln in (p.stdout or "").splitlines() if ln.startswith("AMDCI")), None)
    info["env"] = json.loads(line[5:]) if line else {"error": (p.stderr or "")[-1500:]}
    pl = subprocess.run([str(py), "-m", "pip", "list"], capture_output = True, text = True)
    info["pip_list"] = [ln for ln in (pl.stdout or "").splitlines()
                        if ln.split(" ")[0].lower() in ("torch", "triton", "triton-rocm", "pytorch-triton-rocm")]
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--venv-root", type = Path, required = True)
    ap.add_argument("--lanes", default = ",".join(l[0] for l in LANES))
    args = ap.parse_args()
    args.out = args.out.resolve()
    obs: dict = {"state": args.state, "checkout": args.checkout, "lanes": {}}

    def dump():
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    if args.state != "head":
        obs["skipped"] = "the torch sweep runs once, at head"
        dump()
        return 0
    out_dir = args.out.parent / f"{args.state}_artifacts"
    out_dir.mkdir(parents = True, exist_ok = True)
    patch = Path(args.checkout) / PATCH
    obs["patch_file"] = str(patch)
    obs["has_patch_module"] = patch.is_file()
    want = set(args.lanes.split(","))
    for lane, pin, index, arms in LANES:
        if lane not in want:
            continue
        rec: dict = {}
        obs["lanes"][lane] = rec
        venv = args.venv_root.resolve() / f"venv_{lane}"
        rec["venv"] = build(venv, pin, index, out_dir / f"venv_{lane}.log")
        dump()
        if not rec["venv"].get("error"):
            res_json = out_dir / f"diag_{lane}.json"
            log = out_dir / f"diag_{lane}.log"
            # Clean env: no checkout on PYTHONPATH, no rand4x / cache-tag overrides (the diag clears them too).
            env = zoo_venv.state_env(args.checkout)
            env.pop("PYTHONPATH", None)
            env["TORCHINDUCTOR_CACHE_DIR"] = str(args.venv_root.resolve() / f"inductor_{lane}")
            rc = zoo_venv.sh([rec["venv"]["python"], "-u", str(HERE / "rand4x_c_diag.py"), "--out", str(res_json),
                              "--patch-file", str(patch), "--patch-file-fixed", str(FIXED_PATCH), "--arms", arms, "--artifacts", str(out_dir / f"code_{lane}")],
                             log, timeout = 5400, env = env, cwd = str(args.venv_root.resolve()))
            rec["rc"] = rc
            try:
                rec["arms"] = json.loads(res_json.read_text(encoding = "utf-8"))
            except Exception as e:  # noqa: BLE001
                rec["arms_error"] = repr(e)
                rec["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-3000:] if log.is_file() else ""
        shutil.rmtree(venv, ignore_errors = True)
        shutil.rmtree(args.venv_root.resolve() / f"inductor_{lane}", ignore_errors = True)
        dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())

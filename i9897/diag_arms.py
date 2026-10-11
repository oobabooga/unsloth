"""Diagnosis arms for #9897's torch.compile failures, one fresh process + fresh caches each. Writes <out>/<arm>.json.

Dequant arms run diffusers' GGUF dequant alone (no Studio); studio arms run the instrumented Studio probe.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", required = True)
    ap.add_argument("--tree", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--scratch", required = True)
    ap.add_argument("--arms", default = "")
    ap.add_argument("--timeout", type = int, default = 1500)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok = True)

    gguf = subprocess.check_output(
        [args.py, "-c", "from huggingface_hub import hf_hub_download as h;"
         "print(h('unsloth/FLUX.2-klein-4B-GGUF','flux-2-klein-4b-Q2_K.gguf'))"], text = True).strip().splitlines()[-1]
    tcc = ""
    try:
        tdir = subprocess.check_output([args.py, "-c", "import triton,os;print(os.path.dirname(triton.__file__))"],
                                       text = True).strip().splitlines()[-1]
        hits = glob.glob(os.path.join(tdir, "runtime", "tcc", "tcc*")) + glob.glob(os.path.join(tdir, "**", "tcc.exe"),
                                                                                recursive = True)
        tcc = next((h for h in hits if h.lower().endswith((".exe", "tcc"))), "")
    except Exception:  # noqa: BLE001
        pass

    dq = [os.path.join(HERE, "dequant_alias_repro.py"), "--gguf", gguf]
    st = [os.path.join(HERE, "studio_compile_probe.py"), "--backend-dir", os.path.join(args.tree, "studio", "backend")]
    E1 = {"TORCHINDUCTOR_ERROR_ON_CUSTOM_OP_ALIASING": "1"}
    arms = {
        "dq_mp1_err": (dq + ["--memory-planning", "1"], E1),
        "dq_mp0_err": (dq + ["--memory-planning", "0"], E1),
        "dq_mp1": (dq + ["--memory-planning", "1"], {}),
        "st_default": (st, {}),
        "st_default_err": (st, E1),
        "st_max": (st + ["--speed-mode", "max"], {}),
        "st_max_err": (st + ["--speed-mode", "max"], E1),
    }
    if tcc:
        arms["dq_mp0_tcc"] = (dq + ["--memory-planning", "0"], {"CC": tcc})
    wanted = [a for a in args.arms.split(",") if a] or list(arms)
    summary = {"tcc": tcc, "gguf": gguf}
    for arm in wanted:
        if arm not in arms:
            summary[arm] = {"skipped": "not available here"}
            continue
        argv, extra = arms[arm]
        env = dict(os.environ)
        for k in ("CI", "CC", "TORCHINDUCTOR_ERROR_ON_CUSTOM_OP_ALIASING"):
            env.pop(k, None)
        env.update(extra)
        for name in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "UNSLOTH_COMPILE_LOCATION"):
            d = os.path.join(args.scratch, f"{name.lower()}_{arm}")
            shutil.rmtree(d, ignore_errors = True)
            os.makedirs(d, exist_ok = True)
            env[name] = d
        env["PYTHONUNBUFFERED"] = "1"
        out_json = os.path.join(args.out, f"{arm}.json")
        t0 = time.time()
        with open(os.path.join(args.out, f"{arm}.log"), "w", encoding = "utf-8") as fh:
            try:
                rc = subprocess.call([args.py, "-u", *argv, "--out", out_json], env = env, stdout = fh,
                                     stderr = subprocess.STDOUT, timeout = args.timeout)
            except subprocess.TimeoutExpired:
                rc = "timeout"
        summary[arm] = {"rc": rc, "wall_s": round(time.time() - t0, 1), "json": os.path.exists(out_json)}
        print(arm, summary[arm], flush = True)
        with open(os.path.join(args.out, "summary.json"), "w", encoding = "utf-8") as fh:
            json.dump(summary, fh, indent = 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

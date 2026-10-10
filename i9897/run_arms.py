"""Run gguf_gen_probe.py per arm (tree x Triton C compiler), one fresh process and fresh Triton / inductor caches each.

Arms: {base, head} x {healthy, broken_cc}. broken_cc points CC at a compiler that always fails, so Triton cannot build
its driver helper (hip_utils.c / cuda_utils.c), the #9897 failure mode. Writes <out>/<arm>.json and <out>/summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", required = True)
    ap.add_argument("--base-tree", required = True)
    ap.add_argument("--head-tree", required = True)
    ap.add_argument("--broken-cc", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--scratch", required = True)
    ap.add_argument("--arms", default = "base_healthy,head_healthy,base_injected,head_injected")
    ap.add_argument("--steps", default = "4")
    ap.add_argument("--probe-arg", action = "append", default = [])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok = True)
    probe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gguf_gen_probe.py")
    summary: dict = {}
    for arm in args.arms.split(","):
        tree = args.base_tree if arm.startswith("base") else args.head_tree
        env = dict(os.environ)
        env.pop("CC", None)
        # torch defaults error_on_custom_op_aliasing to bool($CI): a CI runner turns a user-side warning into an error.
        env.pop("CI", None)
        if arm.endswith("broken_cc"):
            env["CC"] = args.broken_cc
        extra = ["--inject-inductor-driver-failure"] if arm.endswith("injected") else []
        for name in ("triton", "inductor", "ucc"):
            d = os.path.join(args.scratch, f"{name}_{arm}")
            shutil.rmtree(d, ignore_errors = True)
            os.makedirs(d, exist_ok = True)
        env["TRITON_CACHE_DIR"] = os.path.join(args.scratch, f"triton_{arm}")
        env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(args.scratch, f"inductor_{arm}")
        env["UNSLOTH_COMPILE_LOCATION"] = os.path.join(args.scratch, f"ucc_{arm}")
        env["PYTHONUNBUFFERED"] = "1"
        out_json = os.path.join(args.out, f"{arm}.json")
        log = os.path.join(args.out, f"{arm}.log")
        t0 = time.time()
        with open(log, "w", encoding = "utf-8") as fh:
            rc = subprocess.call(
                [
                    args.py, "-u", probe,
                    "--backend-dir", os.path.join(tree, "studio", "backend"),
                    "--out", out_json,
                    "--image-out", os.path.join(args.out, f"{arm}.png"),
                    "--steps", args.steps,
                    *args.probe_arg,
                    *extra,
                ],
                env = env,
                stdout = fh,
                stderr = subprocess.STDOUT,
            )
        try:
            with open(out_json, encoding = "utf-8") as fh:
                res = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            res = {"probe_unreadable": repr(exc)}
        keep = (
            "generate_ok", "generate_error_type", "compiled_dequant_installed",
            "torch_compile_runtime_available", "crt_headers_reachable", "toolchain",
            "injected", "torch", "hip", "triton", "device", "load_s", "generate_s", "setup_error",
        )
        summary[arm] = {"rc": rc, "wall_s": round(time.time() - t0, 1), **{k: res.get(k) for k in keep}}
        print(arm, json.dumps(summary[arm]), flush = True)
        with open(os.path.join(args.out, "summary.json"), "w", encoding = "utf-8") as fh:
            json.dump(summary, fh, indent = 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

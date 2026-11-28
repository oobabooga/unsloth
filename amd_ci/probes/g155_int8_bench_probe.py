#!/usr/bin/env python3
"""Probe: int8 versus bf16 GEMM timings for the rotated INT8 frozen-base LoRA kernels (G155, PR 19965).

OBSERVES only. Runs probes/g155_kbench_portable.py (torch + triton, no unsloth import, so every
state runs the same kernels: base and head are an A/A pair) with this interpreter, keeps the full
JSON next to the observation as kbench_<state>.json, and writes a per-shape summary to --out:

  bf16 fwd / dgrad (min of 3 medians), best int8 fwd / dgrad over Triton tiles and
  torch._int_mm + fused epilogue, the activation quant kernels, and int8 speedup ratios with and
  without the activation quant cost.

The criteria module judges. Usage (differential.py passes the first three):
  probe --state S --checkout DIR --out OBS.json [--models qwen3_4b,qwen3_8b] [--ms 4096,8192]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KBENCH = HERE / "g155_kbench_portable.py"


def _f(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _best(r: dict, prefixes: tuple[str, ...]):
    best, name = None, None
    for k, v in r.items():
        if not k.startswith(prefixes) or k.endswith(("_relerr", "_mismatch", "_check")):
            continue
        v = _f(v)
        if v is not None and (best is None or v < best):
            best, name = v, k
    return best, name


def summarise(kb: dict) -> dict:
    shapes = {}
    for key, r in (kb.get("shapes") or {}).items():
        if "cell_error" in r and "bf16_fwd" not in r:
            shapes[key] = {"cell_error": r["cell_error"]}
            continue
        bf = [x for x in (r.get("bf16_fwd") or []) if _f(x) is not None]
        bd = [x for x in (r.get("bf16_dgrad") or []) if _f(x) is not None]
        tri_f, tri_f_k = _best(r, ("tri_fwd_",))
        mm_f, mm_f_k = _best(r, ("intmm_fwd_epi",))
        tri_d, tri_d_k = _best(r, ("tri_dgrad_",))
        mm_d, mm_d_k = _best(r, ("intmm_dgrad_epi",))
        q_x, q_x_k = _best(r, ("rotq2", "rotq1_"))
        q_dy, q_dy_k = _best(r, ("rowq2", "rowq1_"))
        cands_f = [(v, k) for v, k in ((tri_f, tri_f_k), (mm_f, mm_f_k)) if v is not None]
        cands_d = [(v, k) for v, k in ((tri_d, tri_d_k), (mm_d, mm_d_k)) if v is not None]
        i8_f = min(cands_f) if cands_f else (None, None)
        i8_d = min(cands_d) if cands_d else (None, None)
        s = {
            "bf16_fwd_us": min(bf) if bf else None, "bf16_fwd_runs": bf,
            "bf16_dgrad_us": min(bd) if bd else None, "bf16_dgrad_runs": bd,
            "tri_fwd_us": tri_f, "tri_fwd_best": tri_f_k, "intmm_fwd_us": mm_f, "intmm_fwd_best": mm_f_k,
            "tri_dgrad_us": tri_d, "tri_dgrad_best": tri_d_k, "intmm_dgrad_us": mm_d, "intmm_dgrad_best": mm_d_k,
            "int8_fwd_us": i8_f[0], "int8_fwd_best": i8_f[1], "int8_dgrad_us": i8_d[0], "int8_dgrad_best": i8_d[1],
            "quant_x_us": q_x, "quant_x_best": q_x_k, "quant_dy_us": q_dy, "quant_dy_best": q_dy_k,
            "intmm_only_fwd_us": _f(r.get("intmm_only_fwd")), "intmm_only_dgrad_us": _f(r.get("intmm_only_dgrad")),
            "fwd_relerr_vs_bf16": _f(r.get("fwd_relerr_vs_bf16")),
            "dgrad_relerr_vs_bf16": _f(r.get("dgrad_relerr_vs_bf16")),
            "ref_fwd_tile": r.get("ref_fwd_tile"), "ref_dgrad_tile": r.get("ref_dgrad_tile"),
            "cell_error": r.get("cell_error"),
        }
        bfw, bdg = s["bf16_fwd_us"], s["bf16_dgrad_us"]
        if bfw and s["int8_fwd_us"]:
            s["speedup_fwd_gemm"] = bfw / s["int8_fwd_us"]
            if s["quant_x_us"]:
                s["speedup_fwd_with_quant"] = bfw / (s["int8_fwd_us"] + s["quant_x_us"])
        if bdg and s["int8_dgrad_us"]:
            s["speedup_dgrad_gemm"] = bdg / s["int8_dgrad_us"]
            if s["quant_dy_us"]:
                s["speedup_dgrad_with_quant"] = bdg / (s["int8_dgrad_us"] + s["quant_dy_us"])
        shapes[key] = s
    return shapes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--models", default = "qwen3_4b,qwen3_8b")
    ap.add_argument("--ms", default = "4096,8192")
    ap.add_argument("--timeout", type = int, default = 3600, help = "seconds for one kbench run")
    a = ap.parse_args()

    out_dir = a.out.parent
    kb_json = out_dir / f"kbench_{a.state}.json"
    kb_log = out_dir / f"kbench_{a.state}.log"
    obs: dict = {"state": a.state, "python": sys.executable, "kbench_json": kb_json.name,
                 "kbench_log": kb_log.name, "models": a.models, "ms": a.ms}

    # Environment facts, observed in a child so a torch import failure is recorded, not fatal.
    env_code = ("import json, torch\n"
                "d = {'torch': torch.__version__, 'hip': getattr(torch.version, 'hip', None),"
                " 'cuda': torch.version.cuda, 'available': torch.cuda.is_available()}\n"
                "try:\n import triton; d['triton'] = triton.__version__\n"
                "except Exception as e: d['triton_error'] = repr(e)[:300]\n"
                "if d['available']:\n"
                " p = torch.cuda.get_device_properties(0)\n"
                " d.update(name = p.name, arch = getattr(p, 'gcnArchName', None),"
                " total_mib = p.total_memory // 2**20, cap = list(torch.cuda.get_device_capability()))\n"
                "open(__import__('sys').argv[1], 'w', encoding = 'utf-8').write(json.dumps(d))\n")
    env_json = out_dir / f"env_{a.state}.json"
    rc = subprocess.run([sys.executable, "-c", env_code, str(env_json)], capture_output = True, text = True)
    if env_json.is_file():
        obs["env"] = json.loads(env_json.read_text(encoding = "utf-8"))
    else:
        obs["env_error"] = (rc.stderr or rc.stdout)[-1500:]

    t0 = time.time()
    cmd = [sys.executable, "-u", str(KBENCH), "--out", str(kb_json), "--models", a.models, "--ms", a.ms]
    try:
        with open(kb_log, "w", encoding = "utf-8") as fh:
            obs["kbench_rc"] = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT,
                                              timeout = a.timeout).returncode
    except subprocess.TimeoutExpired:
        obs["kbench_rc"] = "timeout"
    obs["kbench_seconds"] = round(time.time() - t0, 1)
    if kb_log.is_file():
        obs["kbench_log_tail"] = kb_log.read_text(encoding = "utf-8", errors = "replace")[-2000:]

    if kb_json.is_file():
        try:
            kb = json.loads(kb_json.read_text(encoding = "utf-8"))
        except Exception as e:  # noqa: BLE001
            obs["kbench_parse_error"] = f"{type(e).__name__}: {e}"
        else:
            for k in ("device", "arch", "cap", "torch", "triton", "hip", "cuda", "dtype", "prefetch_patch",
                      "skip_variants", "probes", "started", "finished"):
                obs[k] = kb.get(k)
            obs["shapes"] = summarise(kb)
    else:
        obs["kbench_missing"] = True

    a.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

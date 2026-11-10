# SPDX-License-Identifier: AGPL-3.0-only
"""fast_qk_norm_rope eager vs torch.compile bytes, per dtype, on one checkout (on PYTHONPATH).

For each dtype x packed: eager vs Inductor (fullgraph) and eager vs aot_eager, per output (q, k, dq, dk):
differing element count, max abs diff, max ulp distance, max error of each against an fp64 reference.
Then the Triton cache is scanned for every build of the qk kernels: options, the ttir signature
(specialisation) and fp16 / fma instruction counts in the amdgcn, so two builds of one kernel can be
diffed. Prints RESULT / BUILD json lines and a header line.
"""
import json
import os
import re
import sys
from pathlib import Path

import torch

import unsloth  # noqa: F401
import torch._dynamo.utils as dynamo_utils
from unsloth.kernels import qk_norm_rope as QK

sys.path.insert(0, os.path.join(os.path.dirname(QK.__file__), "..", "..", "tests"))
import test_qk_norm_rope as T  # noqa: E402

import triton

print("HEADER", json.dumps({
    "torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
    "triton": triton.__version__, "device": torch.cuda.get_device_name(0),
    "arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
    "qk_file": QK.__file__, "triton_cache": os.environ.get("TRITON_CACHE_DIR"),
}), flush = True)


def ulp_dist(x, y):
    # Monotone integer mapping of the float bits: distance in representable steps.
    if x.dtype == torch.float32:
        it, sb = torch.int32, 31
    else:
        it, sb = torch.int16, 15
    def key(t):
        b = t.contiguous().view(it).to(torch.int64)
        return torch.where(b < 0, -(b & ((1 << sb) - 1)), b)
    return (key(x) - key(y)).abs()


def compare(a, b, ref, dtype):
    out = {}
    for name, x, y, r in zip(("q", "k", "dq", "dk"), a, b, ref):
        d = x != y
        n = int(d.sum())
        e = {"n_diff": n, "numel": x.numel()}
        if n:
            u = ulp_dist(x, y)
            e["max_abs"] = (x.double() - y.double()).abs().max().item()
            e["max_ulp"] = int(u.max())
            e["n_ulp1"] = int((u == 1).sum())
            tiny = torch.finfo(dtype).tiny
            e["n_diff_subnormal"] = int((d & ((x.abs() < tiny) | (y.abs() < tiny))).sum())
            e["sample"] = [[x[d][i].item(), y[d][i].item()] for i in range(min(3, n))]
        e["err_eager"] = (x.double() - r).abs().max().item()
        e["err_compiled"] = (y.double() - r).abs().max().item()
        out[name] = e
    return out


results = []
for dtype in (torch.float16, torch.bfloat16, torch.float32):
    for packed in (False, True):
        for backend in ("inductor", "aot_eager"):
            torch._dynamo.reset()
            dynamo_utils.counters.clear()
            case = T._case(2, 96, 16, 4, 128, dtype, packed)
            r = {"dtype": str(dtype).split(".")[-1], "packed": packed, "backend": backend}
            try:
                eager = T._run(QK.fast_qk_norm_rope, *case, 128)
                comp = T._run(torch.compile(QK.fast_qk_norm_rope, fullgraph = True, backend = backend), *case, 128)
                ref = T._ref64(*case, 128)
                r["graph_breaks"] = sum(dynamo_utils.counters["graph_break"].values())
                r["outputs"] = compare(eager, comp, ref, dtype)
                r["equal"] = all(v["n_diff"] == 0 for v in r["outputs"].values())
            except Exception as e:  # recorded, judged by the criteria
                r["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            results.append(r)
            print("RESULT", json.dumps(r), flush = True)
torch._dynamo.reset()

# Every build of the qk kernels in this process's Triton cache.
cache = Path(os.environ.get("TRITON_CACHE_DIR", ""))
pat = re.compile(r"v_pk_fma_f16|v_fma_f16|v_pk_mul_f16|v_mul_f16|v_fma_mix\w*|v_fmac_f16\w*|v_pk_add_f16|v_sub_f16|v_add_f16|v_fma_f32|v_fmac_f32\w*|v_mul_f32|v_pk_fma_f32|fma\.rn\.f16x2|fma\.rn\.f16|mul\.rn\.f16|fma\.rn\.f32")
if cache.is_dir():
    for meta in sorted(cache.rglob("*.json")):
        if not meta.stem.startswith("_qk_norm_rope") or meta.name.startswith("__grp__"):
            continue
        try:
            md = json.loads(meta.read_text(encoding = "utf-8"))
        except Exception:
            continue
        b = {"dir": meta.parent.name, "name": meta.stem}
        for k in ("num_warps", "num_stages", "enable_fp_fusion", "allow_flush_denorm", "arch", "warp_size",
                  "waves_per_eu", "matrix_instr_nonkdim", "kpack", "debug", "sanitize_overflow", "launch_pdl"):
            if k in md:
                b[k] = md[k]
        ttir = meta.with_suffix(".ttir")
        if ttir.exists():
            t = ttir.read_text(encoding = "utf-8")
            m = re.search(r"tt\.func public @\w+\((.*?)\)\s*attributes", t, re.S)
            b["signature"] = re.sub(r"\s+", " ", m.group(1))[:3000] if m else None
            b["ttir_fp16_tensor"] = "xf16" in t
        for ext in (".amdgcn", ".ptx"):
            asm = meta.with_suffix(ext)
            if asm.exists():
                counts = {}
                for ins in pat.findall(asm.read_text(encoding = "utf-8")):
                    counts[ins] = counts.get(ins, 0) + 1
                b["asm_ext"], b["asm_counts"] = ext, counts
        print("BUILD", json.dumps(b), flush = True)
print("DONE", flush = True)

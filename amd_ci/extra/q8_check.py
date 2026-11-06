"""Retro check for unsloth#16884 on one torch: fused Q8_0 dequant+cast (head) vs stock compiled dequant + cast (base).

usage: python q8_check.py <diffusion_gguf_compile.py at the merge commit> <out.json> [device]
Loads only the two kernel functions of the module (the patch backend import is stubbed)."""

import json
import statistics
import sys
import time
import traceback

import numpy as np
import torch
import gguf
from diffusers.quantizers.gguf import utils as gguf_utils

src_path, out_path = sys.argv[1], sys.argv[2]
dev = sys.argv[3] if len(sys.argv) > 3 else "cuda"
src = open(src_path, encoding = "utf-8").read()
src = src.replace(
    "from .diffusion_patch_backend import apply_patch, revert_patch",
    "apply_patch = revert_patch = None",
)
ns: dict = {"__name__": "gc_retro"}
exec(compile(src, src_path, "exec"), ns)
blocks_fn_eager = ns["_q8_0_blocks_dequant_cast"]
dequant_cast = ns["_q8_0_dequant_cast"]
Q8 = gguf.GGMLQuantizationType.Q8_0

res = {
    "torch": torch.__version__,
    "triton": None,
    "device": torch.cuda.get_device_name() if dev == "cuda" else "cpu",
    "cells": [],
}
try:
    import triton

    res["triton"] = triton.__version__
except Exception:  # noqa: BLE001
    pass


def q8_bytes(rows, cols, adversarial, seed = 0):
    rng = np.random.default_rng(seed)
    if not adversarial:
        w = (rng.standard_normal((rows, cols)) * 0.02).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(gguf.quants.quantize(w, Q8)))
    return torch.from_numpy(rng.integers(0, 256, size = (rows, cols // 32 * 34), dtype = np.uint8))


def bits(t):
    return t.view({torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[t.dtype])


def first_line(exc):
    s = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {s[0][:200] if s else ''}"


stock_dequant = gguf_utils.dequantize_gguf_tensor
for emulate in (False, True):
    for adversarial in (False, True):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            torch._dynamo.reset()
            cell = {"emulate": emulate, "adversarial": adversarial, "dtype": str(dtype)}
            raw = q8_bytes(96, 256, adversarial).to(dev)
            param = gguf_utils.GGUFParameter(raw, quant_type = Q8)
            ref = stock_dequant(param).to(dtype)
            with torch._inductor.config.patch(emulate_precision_casts = emulate):
                try:
                    base = torch.compile(stock_dequant, dynamic = True)(param).to(dtype)
                    cell["base_compiled_equal_eager"] = bool(torch.equal(bits(base), bits(ref)))
                except Exception as exc:  # noqa: BLE001
                    cell["base_error"] = first_line(exc)
                try:
                    head = dequant_cast(raw, dtype, torch.compile(blocks_fn_eager, dynamic = True))
                    cell["head_fused_equal_eager"] = bool(torch.equal(bits(head), bits(ref)))
                    cell["head_mismatch"] = int((bits(head) != bits(ref)).sum())
                except Exception as exc:  # noqa: BLE001
                    cell["head_error"] = first_line(exc)
            cell["head_eager_equal"] = bool(torch.equal(bits(dequant_cast(raw, dtype)), bits(ref)))
            res["cells"].append(cell)
            print(json.dumps(cell), flush = True)

# graph breaks / recompiles across layer shapes
try:
    torch._dynamo.reset()
    raw = q8_bytes(32, 128, False).to(dev)
    ex = torch._dynamo.explain(blocks_fn_eager)(raw, torch.bfloat16)
    res["graph_breaks"] = ex.graph_break_count
    res["graph_count"] = ex.graph_count
    from torch._dynamo.testing import CompileCounter

    cnt = CompileCounter()
    fn = torch.compile(blocks_fn_eager, backend = cnt, dynamic = True)
    for r, c in [(32, 128), (48, 256), (64, 4096), (17, 64)]:
        dequant_cast(q8_bytes(r, c, False).to(dev), torch.bfloat16, fn)
    res["frames_over_4_shapes"] = cnt.frame_count
except Exception as exc:  # noqa: BLE001
    res["graph_error"] = first_line(exc)


# per-linear weight materialisation time, base (compiled dequant + eager cast) vs head (fused), shared GPU: indicative
def timeit(f, n = 50):
    for _ in range(5):
        f()
    if dev == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        f()
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return round(statistics.median(ts), 4), round(min(ts), 4)


try:
    torch._dynamo.reset()
    raw = q8_bytes(3072, 3072, False).to(dev)
    param = gguf_utils.GGUFParameter(raw, quant_type = Q8)
    cdeq = torch.compile(stock_dequant, dynamic = True)
    cfused = torch.compile(blocks_fn_eager, dynamic = True)
    res["ms_base_compiled_dequant_then_cast"] = timeit(lambda: cdeq(param).to(torch.bfloat16))
    res["ms_eager_dequant_then_cast"] = timeit(lambda: stock_dequant(param).to(torch.bfloat16))
    try:
        res["ms_head_fused"] = timeit(lambda: dequant_cast(raw, torch.bfloat16, cfused))
    except Exception as exc:  # noqa: BLE001
        res["ms_head_fused_error"] = first_line(exc)
        res["ms_head_eager_fallback"] = timeit(lambda: dequant_cast(raw, torch.bfloat16))
except Exception as exc:  # noqa: BLE001
    res["timing_error"] = first_line(exc)
    traceback.print_exc()

with open(out_path, "w", encoding = "utf-8") as f:
    json.dump(res, f, indent = 1)
print("RESULT", json.dumps({k: v for k, v in res.items() if k != "cells"}), flush = True)

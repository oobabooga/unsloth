"""Does the AOTriton gate do anything on gfx1151 with stable multi-arch ROCm wheels?

Run once per gate state (unset / "0" / "1"); the state is read from the environment this
process was started with, so the caller sets it, never this script.

Part 1 forces each SDPA backend and records whether a kernel exists at all.
Part 2 times MATH against flash at training shapes, forward + backward.
"""

import gc
import json
import os
import sys
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

GATE = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"
STATE = os.environ.get(GATE, "<unset>")

BACKENDS = {"flash": SDPBackend.FLASH_ATTENTION, "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
            "math": SDPBackend.MATH}


def _dev():
    assert torch.cuda.is_available(), "no ROCm device visible"
    return torch.device("cuda")


def _qkv(b, h, n, d, dtype, requires_grad = False):
    g = torch.Generator(device = "cpu").manual_seed(0)
    make = lambda: torch.randn(b, h, n, d, generator = g, dtype = torch.float32).to(
        _dev(), dtype = dtype).requires_grad_(requires_grad)
    return make(), make(), make()


def availability():
    """The 16-cell table from the PR thread, re-run on gfx1151."""
    rows = []
    for dtype_name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        for backend_name in ("flash", "mem_efficient"):
            for causal in (False, True):
                for direction in ("fwd", "bwd"):
                    q, k, v = _qkv(2, 8, 512, 64, dtype, requires_grad = direction == "bwd")
                    try:
                        with sdpa_kernel(BACKENDS[backend_name]):
                            out = torch.nn.functional.scaled_dot_product_attention(
                                q, k, v, is_causal = causal)
                            if direction == "bwd":
                                out.sum().backward()
                        torch.cuda.synchronize()
                        result = "pass"
                    except Exception as e:  # noqa: BLE001 - the message is the datum
                        result = type(e).__name__ + ": " + str(e).strip().splitlines()[0][:90]
                    rows.append({"dtype": dtype_name, "backend": backend_name,
                                 "causal": causal, "direction": direction, "result": result})
                    del q, k, v
                    gc.collect()
                    torch.cuda.empty_cache()
    return rows


def _bench(backend_name, b, h, n, d, dtype, iters = 5):
    q, k, v = _qkv(b, h, n, d, dtype, requires_grad = True)
    torch.cuda.reset_peak_memory_stats()
    try:
        with sdpa_kernel(BACKENDS[backend_name]):
            for _ in range(2):  # warm up the allocator and any autotune
                torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal = True).sum().backward()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal = True).sum().backward()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000 / iters
        peak = torch.cuda.max_memory_allocated() / 2**30
        out = {"ms": round(ms, 2), "peak_gib": round(peak, 3)}
    except Exception as e:  # noqa: BLE001
        out = {"error": type(e).__name__ + ": " + str(e).strip().splitlines()[0][:90]}
    del q, k, v
    gc.collect()
    torch.cuda.empty_cache()
    return out


def benchmark():
    """Llama-3.1-8B attention shape, bf16 causal, forward + backward."""
    rows = []
    for b, n in ((1, 1024), (1, 2048), (1, 4096), (1, 8192), (2, 8192), (1, 16384)):
        row = {"B": b, "N": n, "H": 32, "D": 128}
        for backend_name in ("math", "flash", "mem_efficient"):
            row[backend_name] = _bench(backend_name, b, 32, n, 128, torch.bfloat16)
        rows.append(row)
        print("  " + json.dumps(row), flush = True)
    return rows


def accuracy():
    """flash / mem_efficient against MATH at a shape all three can hold."""
    out = {}
    q, k, v = _qkv(2, 8, 512, 64, torch.float16)
    with sdpa_kernel(BACKENDS["math"]):
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal = True)
    for backend_name in ("flash", "mem_efficient"):
        try:
            with sdpa_kernel(BACKENDS[backend_name]):
                got = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal = True)
            out[backend_name] = float((got.float() - ref.float()).abs().max())
        except Exception as e:  # noqa: BLE001
            out[backend_name] = type(e).__name__ + ": " + str(e).strip().splitlines()[0][:90]
    return out


def main():
    props = torch.cuda.get_device_properties(0)
    report = {
        "gate_state": STATE,
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "device": props.name,
        "gcn_arch": getattr(props, "gcnArchName", None),
        "total_mem_gib": round(props.total_memory / 2**30, 2),
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
    }
    print("== environment ==", flush = True)
    print(json.dumps(report, indent = 2), flush = True)

    print("== backend availability ==", flush = True)
    report["availability"] = availability()
    passed = sum(r["result"] == "pass" for r in report["availability"])
    print(f"  {passed}/{len(report['availability'])} forced calls succeeded", flush = True)
    for r in report["availability"]:
        print("  " + json.dumps(r), flush = True)

    if passed:
        print("== accuracy vs math ==", flush = True)
        report["accuracy"] = accuracy()
        print("  " + json.dumps(report["accuracy"]), flush = True)

    print("== benchmark ==", flush = True)
    report["benchmark"] = benchmark()

    out = os.environ.get("PROBE_OUT")
    if out:
        with open(out, "w", encoding = "utf-8") as f:
            json.dump(report, f, indent = 2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

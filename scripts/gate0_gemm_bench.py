"""Gate 0: are an integrated GPU's GEMMs (through ggml Vulkan) faster than this CPU's, at LoRA training shapes?

For each Linear shape (M tokens, K in, N out) of SmolLM2-135M and Qwen3-0.6B it times the three training GEMMs:
  fwd : Y[M,N]  = X[M,K] @ W[N,K]^T        (W fp16, resident on the device)
  dX  : dX[M,K] = G[M,N] @ W[N,K]          (W^T fp16 resident, i.e. a second orientation)
  dW  : dW[N,K] = G^T @ X                   (full finetune only)
Each device arm includes upload of the streamed operand(s) and download of the result (the real autograd boundary),
and a compute-only number. CPU arms: torch fp32 and bf16 autocast (the CPU backend's two modes). Optionally the ggml
CPU backend too. Writes one JSON with medians; every GEMM result is checked against torch fp32.

Usage: python gate0_gemm_bench.py --lib-dir <dir with libggml*.so|dll incl. ggml-vulkan> --out gate0.json
"""
import argparse
import ctypes as C
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import torch

GGML_TYPE_F32, GGML_TYPE_F16 = 0, 1
DEV_TYPES = {0: "CPU", 1: "GPU", 2: "IGPU", 3: "ACCEL", 4: "META"}

SHAPES = {
    # name: (K, N) per Linear; tokens M from --tokens
    "smollm2_qkv_o": (576, 576),
    "smollm2_kv": (576, 192),
    "smollm2_gate_up": (576, 1536),
    "smollm2_down": (1536, 576),
    "qwen3_q": (1024, 2048),
    "qwen3_o": (2048, 1024),
    "qwen3_kv": (1024, 1024),
    "qwen3_gate_up": (1024, 3072),
    "qwen3_down": (3072, 1024),
}


class InitParams(C.Structure):
    _fields_ = [("mem_size", C.c_size_t), ("mem_buffer", C.c_void_p), ("no_alloc", C.c_bool)]


def load_ggml(lib_dir):
    ext = {"Windows": ".dll", "Darwin": ".dylib"}.get(platform.system(), ".so")
    prefix = "" if platform.system() == "Windows" else "lib"
    if platform.system() == "Windows":
        os.add_dll_directory(lib_dir)
    base = C.CDLL(os.path.join(lib_dir, f"{prefix}ggml-base{ext}"))
    ggml = C.CDLL(os.path.join(lib_dir, f"{prefix}ggml{ext}"))
    P, S, I64 = C.c_void_p, C.c_size_t, C.c_int64
    sig = {
        (base, "ggml_init"): (P, [InitParams]),
        (base, "ggml_free"): (None, [P]),
        (base, "ggml_tensor_overhead"): (S, []),
        (base, "ggml_graph_overhead"): (S, []),
        (base, "ggml_new_tensor_2d"): (P, [P, C.c_int, I64, I64]),
        (base, "ggml_mul_mat"): (P, [P, P, P]),
        (base, "ggml_transpose"): (P, [P, P]),
        (base, "ggml_cont"): (P, [P, P]),
        (base, "ggml_new_graph"): (P, [P]),
        (base, "ggml_build_forward_expand"): (None, [P, P]),
        (base, "ggml_nbytes"): (S, [P]),
        (base, "ggml_backend_alloc_ctx_tensors"): (P, [P, P]),
        (base, "ggml_backend_buffer_free"): (None, [P]),
        (base, "ggml_backend_tensor_set"): (None, [P, P, S, S]),
        (base, "ggml_backend_tensor_get"): (None, [P, P, S, S]),
        (base, "ggml_backend_graph_compute"): (C.c_int, [P, P]),
        (base, "ggml_backend_synchronize"): (None, [P]),
        (base, "ggml_backend_free"): (None, [P]),
        (base, "ggml_backend_dev_name"): (C.c_char_p, [P]),
        (base, "ggml_backend_dev_description"): (C.c_char_p, [P]),
        (base, "ggml_backend_dev_type"): (C.c_int, [P]),
        (base, "ggml_backend_dev_memory"): (None, [P, C.POINTER(S), C.POINTER(S)]),
        (base, "ggml_backend_dev_init"): (P, [P, C.c_char_p]),
        (ggml, "ggml_backend_load_all_from_path"): (None, [C.c_char_p]),
        (ggml, "ggml_backend_dev_count"): (S, []),
        (ggml, "ggml_backend_dev_get"): (P, [S]),
    }
    api = {}
    for (lib, name), (res, args) in sig.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
        api[name] = fn
    api["ggml_backend_load_all_from_path"](lib_dir.encode())
    return api


def list_devices(g):
    out = []
    for i in range(g["ggml_backend_dev_count"]()):
        d = g["ggml_backend_dev_get"](i)
        free, total = C.c_size_t(), C.c_size_t()
        g["ggml_backend_dev_memory"](d, C.byref(free), C.byref(total))
        out.append({"index": i, "handle": d, "name": g["ggml_backend_dev_name"](d).decode(),
                    "description": g["ggml_backend_dev_description"](d).decode(),
                    "type": DEV_TYPES.get(g["ggml_backend_dev_type"](d), "?"),
                    "free_gb": round(free.value / 1e9, 2), "total_gb": round(total.value / 1e9, 2)})
    return out


class GgmlGemm:
    """One cached graph: out = mul_mat(a, b) with a resident (fp16) and b streamed (fp32)."""

    def __init__(self, g, backend, a_np, b_shape, transpose_b = False):
        self.g, self.backend = g, backend
        mem = g["ggml_tensor_overhead"]() * 16 + g["ggml_graph_overhead"]()
        self.ctx = g["ggml_init"](InitParams(mem, None, True))
        rows, cols = a_np.shape  # a: [rows, cols] row-major -> ggml ne0=cols, ne1=rows
        self.a = g["ggml_new_tensor_2d"](self.ctx, GGML_TYPE_F16, cols, rows)
        br, bc = b_shape
        self.b = g["ggml_new_tensor_2d"](self.ctx, GGML_TYPE_F32, bc, br)
        bb = g["ggml_cont"](self.ctx, g["ggml_transpose"](self.ctx, self.b)) if transpose_b else self.b
        self.out = g["ggml_mul_mat"](self.ctx, self.a, bb)
        self.gf = g["ggml_new_graph"](self.ctx)
        g["ggml_build_forward_expand"](self.gf, self.out)
        self.buf = g["ggml_backend_alloc_ctx_tensors"](self.ctx, backend)
        a16 = np.ascontiguousarray(a_np.astype(np.float16))
        g["ggml_backend_tensor_set"](self.a, a16.ctypes.data, 0, a16.nbytes)
        self.out_bytes = g["ggml_nbytes"](self.out)

    def run(self, b_np, out_np, transfer = True):
        g = self.g
        if transfer:
            g["ggml_backend_tensor_set"](self.b, b_np.ctypes.data, 0, b_np.nbytes)
        g["ggml_backend_graph_compute"](self.backend, self.gf)
        g["ggml_backend_synchronize"](self.backend)
        if transfer:
            g["ggml_backend_tensor_get"](self.out, out_np.ctypes.data, 0, self.out_bytes)

    def free(self):
        self.g["ggml_backend_buffer_free"](self.buf)
        self.g["ggml_free"](self.ctx)


def timed(fn, reps):
    fn()  # warm-up (pipeline compile)
    fn()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times), min(times), max(times)


def rel_err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lib-dir", required = True)
    p.add_argument("--out", default = "gate0.json")
    p.add_argument("--tokens", default = "512,1024,2048")
    p.add_argument("--reps", type = int, default = 7)
    p.add_argument("--device", default = None, help = "ggml device name (default: first IGPU, else GPU)")
    p.add_argument("--shapes", default = ",".join(SHAPES))
    p.add_argument("--ggml-cpu", action = "store_true", help = "also time ggml's CPU backend")
    args = p.parse_args()

    g = load_ggml(os.path.abspath(args.lib_dir))
    devices = list_devices(g)
    for d in devices:
        print("DEVICE", {k: v for k, v in d.items() if k != "handle"})
    pick = None
    if args.device:
        pick = next(d for d in devices if d["name"] == args.device)
    else:
        pick = next((d for d in devices if d["type"] == "IGPU"), None) or next((d for d in devices if d["type"] == "GPU"), None)
    arms = []
    if pick is not None:
        arms.append(("dev", pick))
    if args.ggml_cpu:
        cpu = next((d for d in devices if d["type"] == "CPU"), None)
        if cpu:
            arms.append(("ggml_cpu", cpu))
    backends = {name: g["ggml_backend_dev_init"](d["handle"], None) for name, d in arms}

    torch.manual_seed(0)
    rows = []
    for tokens in [int(t) for t in args.tokens.split(",")]:
        for shape_name in args.shapes.split(","):
            K, N = SHAPES[shape_name]
            M = tokens
            X = torch.randn(M, K) * 0.5
            W = (torch.randn(N, K) * 0.02).half().float()  # what the device stores
            G = torch.randn(M, N) * 0.01
            ref = {"fwd": X @ W.t(), "dX": G @ W, "dW": G.t() @ X}
            flops = 2 * M * K * N
            row = {"tokens": M, "shape": shape_name, "K": K, "N": N, "gflop": flops / 1e9}
            # CPU arms (torch, the CPU backend's real modes)
            Wb = W.bfloat16()
            for mode in ("fp32", "bf16"):
                for op in ("fwd", "dX", "dW"):
                    if mode == "fp32":
                        f = {"fwd": lambda: X @ W.t(), "dX": lambda: G @ W, "dW": lambda: G.t() @ X}[op]
                    else:
                        def f(op = op):
                            with torch.autocast("cpu", dtype = torch.bfloat16):
                                return {"fwd": lambda: X @ Wb.t(), "dX": lambda: G @ Wb, "dW": lambda: G.t() @ X}[op]()
                    med, lo, hi = timed(f, args.reps)
                    row[f"cpu_{mode}_{op}_ms"] = round(med * 1e3, 3)
            # device arms
            for name, d in arms:
                be = backends[name]
                Xn, Gn = np.ascontiguousarray(X.numpy()), np.ascontiguousarray(G.numpy())
                # fwd: mul_mat(W [N,K], X [M,K]) -> [M,N]
                fwd = GgmlGemm(g, be, W.numpy(), (M, K))
                # dX: mul_mat(W^T [K,N], G [M,N]) -> [M,K]
                dx = GgmlGemm(g, be, np.ascontiguousarray(W.t().numpy()), (M, N))
                # dW^T[K,N] = mul_mat(X^T as a? ...): use a = G^T [N,M] fp16 would quantise G; instead
                # dW[N,K] = mul_mat(a = X^T [K,M] (fp16), b = G^T [N,M]) -> [N,K]; X^T is streamed, so this arm
                # uploads X as fp16 each call (both operands stream for dW).
                outs = {"fwd": np.empty((M, N), np.float32), "dX": np.empty((M, K), np.float32), "dW": np.empty((N, K), np.float32)}
                fwd.run(Xn, outs["fwd"])
                dx.run(Gn, outs["dX"])
                row[f"{name}_fwd_err"] = rel_err(outs["fwd"], ref["fwd"])
                row[f"{name}_dX_err"] = rel_err(outs["dX"], ref["dX"])
                for op, gem, b in (("fwd", fwd, Xn), ("dX", dx, Gn)):
                    med, lo, hi = timed(lambda: gem.run(b, outs[op]), args.reps)
                    row[f"{name}_{op}_ms"] = round(med * 1e3, 3)
                    med, lo, hi = timed(lambda: gem.run(b, outs[op], transfer = False), args.reps)
                    row[f"{name}_{op}_compute_ms"] = round(med * 1e3, 3)
                fwd.free()
                dx.free()
                # dW: a = X^T [K, M] uploaded per call (fp16), b = G^T streamed fp32 via in-graph transpose
                XT = np.ascontiguousarray(X.t().numpy())
                dw = GgmlGemm(g, be, XT, (M, N), transpose_b = True)

                def run_dw():
                    x16 = XT.astype(np.float16)
                    g["ggml_backend_tensor_set"](dw.a, x16.ctypes.data, 0, x16.nbytes)
                    dw.run(Gn, outs["dW"])
                run_dw()
                row[f"{name}_dW_err"] = rel_err(outs["dW"], ref["dW"].numpy())
                med, lo, hi = timed(run_dw, args.reps)
                row[f"{name}_dW_ms"] = round(med * 1e3, 3)
                dw.free()
            best_cpu = {op: min(row[f"cpu_fp32_{op}_ms"], row[f"cpu_bf16_{op}_ms"]) for op in ("fwd", "dX", "dW")}
            if pick is not None:
                for op in ("fwd", "dX", "dW"):
                    row[f"speedup_{op}"] = round(best_cpu[op] / row[f"dev_{op}_ms"], 2)
            print("ROW", json.dumps(row))
            rows.append(row)

    summary = {}
    if pick is not None:
        for op in ("fwd", "dX", "dW"):
            sp = [r[f"speedup_{op}"] for r in rows]
            summary[op] = {"median_speedup": round(statistics.median(sp), 2), "min": min(sp), "max": max(sp)}
        lora = [r for r in rows]
        tot_cpu = sum(min(r["cpu_fp32_fwd_ms"], r["cpu_bf16_fwd_ms"]) + min(r["cpu_fp32_dX_ms"], r["cpu_bf16_dX_ms"]) for r in lora)
        tot_dev = sum(r["dev_fwd_ms"] + r["dev_dX_ms"] for r in lora)
        summary["lora_fwd_plus_dX_speedup"] = round(tot_cpu / tot_dev, 2)
        summary["max_rel_err"] = max(max(r["dev_fwd_err"], r["dev_dX_err"], r["dev_dW_err"]) for r in rows)
    result = {
        "host": platform.node(), "system": platform.system(), "machine": platform.machine(),
        "cpu": platform.processor(), "torch": torch.__version__, "threads": torch.get_num_threads(),
        "devices": [{k: v for k, v in d.items() if k != "handle"} for d in devices],
        "picked": None if pick is None else pick["name"], "picked_desc": None if pick is None else pick["description"],
        "picked_type": None if pick is None else pick["type"],
        "rows": rows, "summary": summary,
    }
    print("SUMMARY", json.dumps(summary))
    with open(args.out, "w", encoding = "utf-8") as f:
        json.dump(result, f, indent = 1)
    for be in backends.values():
        g["ggml_backend_free"](be)


if __name__ == "__main__":
    main()

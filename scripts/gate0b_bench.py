"""Gate 0b: can the iGPU beat the CPU once the per-op boundary is removed?

Gate 0 (per-GEMM offload) lost to Zen 5 AVX512-BF16 on Strix Halo: device compute was 2-4x faster, but each op
paid an upload, a download and a sync. This measures the two levers on the MLP block (the largest FLOP share):
  per_op    : gate, up, down as three offloaded GEMMs, CPU SwiGLU between (Gate 0 style)
  fused     : one device graph gate+up -> swiglu -> down, one upload, one download
  fused_zc  : the same graph reading / writing page-aligned host memory imported into Vulkan (zero-copy)
against torch CPU fp32 and bf16 autocast on the same box. Forward + backward-input (dX through the MLP) both.

Usage: python gate0b_bench.py --lib-dir <dir with ggml-vulkan> --out gate0b.json
"""
import argparse
import ctypes as C
import json
import os
import platform
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

GGML_TYPE_F32, GGML_TYPE_F16 = 0, 1
MLPS = {"smollm2": (576, 1536), "qwen3_0.6b": (1024, 3072), "qwen3_1.7b": (2048, 6144)}


class InitParams(C.Structure):
    _fields_ = [("mem_size", C.c_size_t), ("mem_buffer", C.c_void_p), ("no_alloc", C.c_bool)]


class Caps(C.Structure):
    _fields_ = [("async_", C.c_bool), ("host_buffer", C.c_bool), ("buffer_from_host_ptr", C.c_bool),
                ("events", C.c_bool), ("mmap_support", C.c_bool)]


class Props(C.Structure):
    _fields_ = [("name", C.c_char_p), ("description", C.c_char_p), ("memory_free", C.c_size_t),
                ("memory_total", C.c_size_t), ("type", C.c_int), ("device_id", C.c_char_p), ("caps", Caps)]


def load(lib_dir):
    system = platform.system()
    ext = {"Windows": ".dll", "Darwin": ".dylib"}.get(system, ".so")
    pre = "" if system == "Windows" else "lib"
    if system == "Windows":
        os.add_dll_directory(lib_dir)
    base = C.CDLL(os.path.join(lib_dir, f"{pre}ggml-base{ext}"))
    ggml = C.CDLL(os.path.join(lib_dir, f"{pre}ggml{ext}"))
    P, S, I64 = C.c_void_p, C.c_size_t, C.c_int64
    sig = {
        (base, "ggml_init"): (P, [InitParams]), (base, "ggml_free"): (None, [P]),
        (base, "ggml_tensor_overhead"): (S, []), (base, "ggml_graph_overhead"): (S, []),
        (base, "ggml_new_tensor_2d"): (P, [P, C.c_int, I64, I64]),
        (base, "ggml_mul_mat"): (P, [P, P, P]), (base, "ggml_swiglu_split"): (P, [P, P, P]),
        (base, "ggml_mul"): (P, [P, P, P]), (base, "ggml_add"): (P, [P, P, P]),
        (base, "ggml_new_graph"): (P, [P]), (base, "ggml_build_forward_expand"): (None, [P, P]),
        (base, "ggml_nbytes"): (S, [P]),
        (base, "ggml_backend_alloc_ctx_tensors"): (P, [P, P]),
        (base, "ggml_backend_tensor_alloc"): (C.c_int, [P, P, P]),
        (base, "ggml_backend_buffer_free"): (None, [P]),
        (base, "ggml_backend_tensor_set"): (None, [P, P, S, S]),
        (base, "ggml_backend_tensor_get"): (None, [P, P, S, S]),
        (base, "ggml_backend_graph_compute"): (C.c_int, [P, P]),
        (base, "ggml_backend_synchronize"): (None, [P]),
        (base, "ggml_backend_dev_init"): (P, [P, C.c_char_p]),
        (base, "ggml_backend_dev_get_props"): (None, [P, C.POINTER(Props)]),
        (base, "ggml_backend_dev_buffer_from_host_ptr"): (P, [P, P, S, S]),
        (ggml, "ggml_backend_load"): (P, [C.c_char_p]),
        (ggml, "ggml_backend_reg_dev_count"): (S, [P]),
        (ggml, "ggml_backend_reg_dev_get"): (P, [P, S]),
    }
    g = {}
    for (lib, name), (res, args) in sig.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
        g[name] = fn
    # Only the Vulkan backend: loading ggml-cpu too clashes with torch's OpenMP runtime on Windows.
    reg = g["ggml_backend_load"](os.path.join(lib_dir, f"{pre}ggml-vulkan{ext}").encode())
    if not reg:
        raise SystemExit("ggml-vulkan failed to load")
    devs = [g["ggml_backend_reg_dev_get"](reg, i) for i in range(g["ggml_backend_reg_dev_count"](reg))]
    return g, devs


def aligned(nbytes, align = 65536):
    size = (nbytes + align - 1) // align * align
    raw = np.empty(size + align, np.uint8)
    off = (-raw.ctypes.data) % align
    return raw[off:off + size], size


class Graph:
    """mul_mat chains over resident fp16 weights; inputs / outputs either copied or imported host memory."""

    def __init__(self, g, dev, backend, build, in_shapes, out_shape, zero_copy):
        self.g, self.backend, self.zero_copy = g, backend, zero_copy
        self.ctx = g["ggml_init"](InitParams(g["ggml_tensor_overhead"]() * 64 + g["ggml_graph_overhead"](), None, True))
        self.inputs = [g["ggml_new_tensor_2d"](self.ctx, GGML_TYPE_F32, cols, rows) for rows, cols in in_shapes]
        self.out = build(self.ctx, *self.inputs)
        self.gf = g["ggml_new_graph"](self.ctx)
        g["ggml_build_forward_expand"](self.gf, self.out)
        self.out_shape = out_shape
        self.host = []
        if zero_copy:
            for t, (rows, cols) in list(zip(self.inputs, in_shapes)) + [(self.out, out_shape)]:
                arr, size = aligned(rows * cols * 4)
                buf = g["ggml_backend_dev_buffer_from_host_ptr"](dev, arr.ctypes.data, size, size)
                if not buf:
                    raise RuntimeError("buffer_from_host_ptr failed")
                g["ggml_backend_tensor_alloc"](buf, t, C.c_void_p(arr.ctypes.data))
                self.host.append((arr[:rows * cols * 4].view(np.float32).reshape(rows, cols), buf))
        self.buf = g["ggml_backend_alloc_ctx_tensors"](self.ctx, backend)

    def run(self, *xs):
        g = self.g
        if self.zero_copy:
            for (view, _), x in zip(self.host, xs):
                np.copyto(view, x)
        else:
            for t, x in zip(self.inputs, xs):
                g["ggml_backend_tensor_set"](t, x.ctypes.data, 0, x.nbytes)
        g["ggml_backend_graph_compute"](self.backend, self.gf)
        g["ggml_backend_synchronize"](self.backend)
        if self.zero_copy:
            return self.host[-1][0]
        y = np.empty(self.out_shape, np.float32)
        g["ggml_backend_tensor_get"](self.out, y.ctypes.data, 0, y.nbytes)
        return y


def resident(g, backend, arrays):
    ctx = g["ggml_init"](InitParams(g["ggml_tensor_overhead"]() * (len(arrays) + 2), None, True))
    ts = [g["ggml_new_tensor_2d"](ctx, GGML_TYPE_F16, a.shape[1], a.shape[0]) for a in arrays]
    buf = g["ggml_backend_alloc_ctx_tensors"](ctx, backend)
    for t, a in zip(ts, arrays):
        a16 = np.ascontiguousarray(a.astype(np.float16))
        g["ggml_backend_tensor_set"](t, a16.ctypes.data, 0, a16.nbytes)
    return ts


def timed(fn, reps):
    fn(); fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lib-dir", required = True)
    p.add_argument("--out", default = "gate0b.json")
    p.add_argument("--tokens", default = "1024,2048,4096")
    p.add_argument("--reps", type = int, default = 9)
    args = p.parse_args()
    g, devs = load(os.path.abspath(args.lib_dir))
    props = Props()
    g["ggml_backend_dev_get_props"](devs[0], C.byref(props))
    info = {"name": props.name.decode(), "description": props.description.decode(), "type": props.type,
            "buffer_from_host_ptr": bool(props.caps.buffer_from_host_ptr), "host_buffer": bool(props.caps.host_buffer)}
    print("DEVICE", info, flush = True)
    dev = devs[0]
    backend = g["ggml_backend_dev_init"](dev, None)
    torch.manual_seed(0)
    rows = []
    for name, (D, I) in MLPS.items():
        Wg, Wu, Wd = [(torch.randn(*s) * 0.02).half().float() for s in ((I, D), (I, D), (D, I))]
        wg, wu, wd, wgT, wuT, wdT = resident(g, backend, [Wg.numpy(), Wu.numpy(), Wd.numpy(),
                                                            Wg.t().contiguous().numpy(), Wu.t().contiguous().numpy(),
                                                            Wd.t().contiguous().numpy()])
        for M in [int(t) for t in args.tokens.split(",")]:
            x = torch.randn(M, D) * 0.5
            gy = torch.randn(M, D) * 0.01
            xn, gyn = x.numpy(), gy.numpy()
            row = {"model": name, "D": D, "I": I, "tokens": M}

            # torch CPU reference fwd + dX through the MLP (input grad only: LoRA base is frozen)
            def cpu(dtype):
                xx = x.clone().requires_grad_()
                ctx = torch.autocast("cpu", dtype = dtype) if dtype else torch.autocast("cpu", enabled = False)
                with ctx:
                    a, b = xx @ Wg.t(), xx @ Wu.t()
                    y = (F.silu(a) * b) @ Wd.t()
                y.backward(gy)
                return y.detach().float(), xx.grad
            y_ref, dx_ref = cpu(None)
            row["cpu_fp32_ms"] = round(timed(lambda: cpu(None), args.reps) * 1e3, 2)
            row["cpu_bf16_ms"] = round(timed(lambda: cpu(torch.bfloat16), args.reps) * 1e3, 2)

            # device: fused forward graph returns y; backward needs a, b again -> recompute on device:
            # dh = gy @ Wd ; da = dh * b * silu'(a) ; db = dh * silu(a) ; dx = da @ Wg + db @ Wu.
            # SwiGLU backward stays on the CPU here (elementwise, cheap); the four GEMMs run on the device.
            for zc in (False, True):
                if zc and not info["buffer_from_host_ptr"]:
                    continue
                tag = "fused_zc" if zc else "fused"
                fwd = Graph(g, dev, backend, lambda c, xi: g["ggml_mul_mat"](c, wd, g["ggml_swiglu_split"](c, g["ggml_mul_mat"](c, wg, xi), g["ggml_mul_mat"](c, wu, xi))),
                            [(M, D)], (M, D), zc)
                ab = Graph(g, dev, backend, lambda c, xi: g["ggml_mul_mat"](c, wg, xi), [(M, D)], (M, I), zc)
                ub = Graph(g, dev, backend, lambda c, xi: g["ggml_mul_mat"](c, wu, xi), [(M, D)], (M, I), zc)
                dh_g = Graph(g, dev, backend, lambda c, gi: g["ggml_mul_mat"](c, wdT, gi), [(M, D)], (M, I), zc)
                dx_g = Graph(g, dev, backend, lambda c, da, db: g["ggml_add"](c, g["ggml_mul_mat"](c, wgT, da), g["ggml_mul_mat"](c, wuT, db)),
                             [(M, I), (M, I)], (M, D), zc)

                def step():
                    y = fwd.run(xn).copy()
                    a = torch.from_numpy(ab.run(xn).copy())
                    b = torch.from_numpy(ub.run(xn).copy())
                    dh = torch.from_numpy(dh_g.run(gyn).copy())
                    s = torch.sigmoid(a)
                    sa = a * s
                    db = (dh * sa).numpy()
                    da = (dh * b * (s * (1 + a * (1 - s)))).numpy()
                    dx = dx_g.run(np.ascontiguousarray(da), np.ascontiguousarray(db)).copy()
                    return y, dx
                y, dx = step()
                row[f"{tag}_y_err"] = float(np.linalg.norm(y - y_ref.numpy()) / np.linalg.norm(y_ref.numpy()))
                row[f"{tag}_dx_err"] = float(np.linalg.norm(dx - dx_ref.numpy()) / np.linalg.norm(dx_ref.numpy()))
                row[f"{tag}_ms"] = round(timed(step, args.reps) * 1e3, 2)
                row[f"{tag}_fwd_only_ms"] = round(timed(lambda: fwd.run(xn), args.reps) * 1e3, 2)
            best = min(row["cpu_fp32_ms"], row["cpu_bf16_ms"])
            for tag in ("fused", "fused_zc"):
                if f"{tag}_ms" in row:
                    row[f"speedup_{tag}"] = round(best / row[f"{tag}_ms"], 2)
            print("ROW", json.dumps(row), flush = True)
            rows.append(row)
    result = {"host": platform.node(), "system": platform.system(), "device": info, "torch": torch.__version__,
              "threads": torch.get_num_threads(), "rows": rows}
    with open(args.out, "w", encoding = "utf-8") as f:
        json.dump(result, f, indent = 1)


if __name__ == "__main__":
    main()

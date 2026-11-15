#!/usr/bin/env python3
"""Studio's Triton INT8 attention, main (package i8_base) vs this branch (i8_head), on one card. One process = one
measurement step; prints ROW {json} lines and appends them to --out.

  python bench.py --mode time    --pkg i8_head --shapes a,b --rep 0 --slot B1   (fresh process per arm, ABBA by driver)
  python bench.py --mode acc     --shapes a,b       (both packages in one process: bit diff + accuracy vs fp64)
  python bench.py --mode profile --pkg i8_base --shapes a,b   (torch.profiler: device time per kernel, host time)
  python bench.py --mode cold    --pkg i8_head      (first call per head dim; the driver sets TRITON_CACHE_DIR)
  python bench.py --mode sweep   --shapes a,b --configs sweep.json   (head kernel, candidate configs per head dim)

The Triton path is forced the way Studio reaches it: ROCm takes its gfx table; NVIDIA runs the op body
(``int8_attention_impl``) with UNSLOTH_DIFFUSION_INT8_ATTENTION_IMPL=triton and the card marked verified.
Inputs: q / k / v chunks of one packed [B, S, 3*H*D] projection (protocol layout), DiT-like statistics, seeded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["UNSLOTH_DIFFUSION_INT8_ATTENTION_IMPL"] = "triton"


def shapes():
    d = json.loads((HERE / "shapes.json").read_text())["shapes"]
    return {k: (int(v["B"]), int(v["Sq"]), int(v["Sk"]), int(v["H"]), int(v["D"])) for k, v in d.items()}


def emit(out, row):
    line = "ROW " + json.dumps(row)
    print(line, flush = True)
    if out:
        with open(out, "a") as f:
            f.write(line + "\n")


def load(pkg):
    M = importlib.import_module(f"{pkg}.diffusion_int8_attention")
    T = importlib.import_module(f"{pkg}.diffusion_int8_attention_triton")
    extra = os.environ.get("I8TS_HEAD_EXTRA")
    if extra and pkg == "i8_head":
        # a candidate table entry for the head, merged into every Triton config of this process (labelled per row)
        upd = json.loads(extra)
        for table in (*M.TRITON_KERNEL_CONFIGS.values(), *M.ROCM_KERNEL_CONFIGS.values()):
            for d in table:
                table[d] = dict(table[d], **upd)
    return M, T


def card():
    import torch

    p = torch.cuda.get_device_properties(0)
    return {"gpu": p.name, "arch": getattr(p, "gcnArchName", "") or "sm%d%d" % torch.cuda.get_device_capability(0),
            "torch": torch.__version__, "triton": __import__("triton").__version__,
            "hip": getattr(torch.version, "hip", None)}


def config_of(M, D):
    import torch

    if M._is_rocm():
        return M.rocm_kernel_config(M._gfx_arch(0), D)
    return M.triton_kernel_config(torch.cuda.get_device_capability(0), D)


def inputs(shape, seed = 0, dtype = None):
    """Packed fused-QKV views: [B, Sk, 3*H*D]; q = the last Sq tokens' Q chunk (Sq < Sk: text prefix keys)."""
    import torch

    B, Sq, Sk, H, D = shape
    dtype = dtype or torch.bfloat16
    g = torch.Generator(device = "cuda").manual_seed(1234 + seed)

    def rn(*s):
        return torch.randn(*s, generator = g, device = "cuda", dtype = torch.float32)

    wq, wk = torch.exp(0.4 * rn(1, 1, H, D)), torch.exp(0.4 * rn(1, 1, H, D))
    kofs, vofs, vsc = 0.6 * rn(1, 1, H, D), 0.5 * rn(1, 1, H, D), 0.5 + rn(1, 1, H, D).abs()
    buf = torch.empty(B, Sk, 3, H, D, device = "cuda", dtype = dtype)
    for i in range(0, Sk, 8192):  # chunked: the fp32 temporaries of a 75k call would be 4 GB
        n = min(8192, Sk - i)
        buf[:, i:i + n, 0] = (rn(B, n, H, D) * wq * 1.6).to(dtype)
        buf[:, i:i + n, 1] = (rn(B, n, H, D) * wk * 1.6 + kofs).to(dtype)
        buf[:, i:i + n, 2] = (rn(B, n, H, D) * vsc + vofs).to(dtype)
    buf = buf.view(B, Sk, 3 * H * D)
    hd = H * D
    q = buf[:, Sk - Sq:, :hd].view(B, Sq, H, D)
    k = buf[:, :, hd:2 * hd].view(B, Sk, H, D)
    v = buf[:, :, 2 * hd:].view(B, Sk, H, D)
    return q, k, v


def runner(M, T, D, dtype, cfg = None):
    """The call Studio makes: the op body on NVIDIA (guards + Triton), the same on ROCm."""
    import torch

    idx = torch.cuda.current_device()
    for dt in ("torch.bfloat16", "torch.float16", "torch.float32"):
        M._VERIFIED.add((idx, D, dt, "triton"))
    if cfg is not None:
        return lambda q, k, v: T.int8_attention(q, k, v, cfg)
    return M.int8_attention_impl


def flush_buf():
    import torch

    return torch.empty(int(256e6), dtype = torch.int8, device = "cuda")


def t_events(fn, reps, flush = None):
    import torch

    ms = []
    for _ in range(reps):
        if flush is not None:
            flush.zero_()
        a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ms.append(a.elapsed_time(b))
    return st.median(ms), ms


def graph_of(fn):
    import torch

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def time_shape(fn, q, k, v, flush, budget_ms = 2500.0, graph = True):
    import torch

    with torch.no_grad():
        call = lambda: fn(q, k, v)  # noqa: E731
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        est, _ = t_events(call, 3, flush)
        reps = int(max(5, min(21, budget_ms / max(est, 1e-3) / 3)))
        if est > 300.0:  # the 75k-token calls: seconds each
            reps = 3
        host = []
        for _ in range(min(reps, 15)):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            call()
            host.append((time.perf_counter() - t0) * 1e6)
        torch.cuda.synchronize()
        n_loop = 3 if est > 300.0 else int(max(10, min(400, budget_ms / max(est, 1e-3) / 3)))
        a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        a.record()
        for _ in range(n_loop):
            call()
        b.record()
        b.synchronize()
        loop_ms = a.elapsed_time(b) / n_loop
        eager_ms, eager_all = t_events(call, reps, flush)
        res = {"eager_ms": eager_ms, "loop_ms": loop_ms, "host_us": st.median(host), "reps": reps, "n_loop": n_loop,
               "eager_spread": (max(eager_all) - min(eager_all)) / eager_ms}
        if graph:
            try:
                g = graph_of(call)
                res["graph_ms"], _ = t_events(g.replay, reps, flush)
                del g
            except Exception as exc:  # noqa: BLE001
                res["graph_error"] = repr(exc)[:300]
    return res


def mode_time(a):
    import torch

    M, T = load(a.pkg)
    info = card()
    flush = flush_buf()
    sh = shapes()
    names = a.shapes.split(",")
    # rotate the shape order per rep so no shape always runs first
    r = a.rep % len(names)
    names = names[r:] + names[:r]
    for name in names:
        B, Sq, Sk, H, D = sh[name]
        cfg = config_of(M, D)
        if cfg is None:
            emit(a.out, {"mode": "time", "shape": name, "pkg": a.pkg, "skip": "no Triton config", **info})
            continue
        q, k, v = inputs(sh[name], seed = 0)
        fn = runner(M, T, D, q.dtype)
        before = dict(M.STATS)
        try:
            res = time_shape(fn, q, k, v, flush)
        except torch.OutOfMemoryError as exc:
            res = {"error": "OOM " + repr(exc)[:200]}
        eng = {x: M.STATS[x] - before[x] for x in ("triton", "native")}
        if "error" not in res:
            # host time of the kernel call alone (what the Triton module costs, without the op's guards)
            direct = runner(M, T, D, q.dtype, cfg)
            with torch.no_grad():
                direct(q, k, v)
                hs = []
                for _ in range(9):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    direct(q, k, v)
                    hs.append((time.perf_counter() - t0) * 1e6)
                torch.cuda.synchronize()
            res["host_us_kernel"] = st.median(hs)
        emit(a.out, {"mode": "time", "shape": name, "pkg": a.pkg, "arm": a.arm or a.pkg, "rep": a.rep, "slot": a.slot, "cfg": cfg,
                     "t": res, "calls": eng, "engaged": eng["triton"] > 0 and eng["native"] == 0, **info})
        del q, k, v
        torch.cuda.empty_cache()


def fp64_ref(q, k, v, heads, rows):
    import torch

    D = q.shape[-1]
    qs = q[0, rows][:, heads].permute(1, 0, 2).double()  # [h, r, D]
    ks = k[0][:, heads].permute(1, 0, 2).double()
    vs = v[0][:, heads].permute(1, 0, 2).double()
    s = torch.matmul(qs, ks.transpose(1, 2)) / (D ** 0.5)
    return torch.matmul(torch.softmax(s, -1), vs)  # [h, r, D]


def metrics(out, ref, heads, rows):
    import torch

    got = out[0, rows][:, heads].permute(1, 0, 2).double()
    rel = float((got - ref).abs().sum() / ref.abs().sum())
    cos = torch.nn.functional.cosine_similarity(got, ref, dim = -1)
    return {"rel_l1": rel, "worst_cos": float(cos.min()), "mean_cos": float(cos.mean()),
            "finite": bool(torch.isfinite(out).all())}


def digest(x):
    return hashlib.sha256(x.contiguous().view(-1).view(__import__("torch").uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def mode_acc(a):
    import torch

    Mb, Tb = load("i8_base")
    Mh, Th = load("i8_head")
    info = card()
    sh = shapes()
    for name in a.shapes.split(","):
        B, Sq, Sk, H, D = sh[name]
        cfg_b, cfg_h = config_of(Mb, D), config_of(Mh, D)
        if cfg_b is None or cfg_h is None:
            emit(a.out, {"mode": "acc", "shape": name, "skip": "no Triton config", **info})
            continue
        for gain in (1.0, 5.0):
            q, k, v = inputs(sh[name], seed = 7)
            if gain != 1.0:
                q, k = q * gain, k * gain
            nb, nh = Mb.STATS["native"], Mh.STATS["native"]
            with torch.no_grad():
                ob = runner(Mb, Tb, D, q.dtype)(q, k, v)
                oh = runner(Mh, Th, D, q.dtype)(q, k, v)
                oh2 = runner(Mh, Th, D, q.dtype)(q, k, v)
            engaged = Mb.STATS["native"] == nb and Mh.STATS["native"] == nh
            with torch.no_grad():
                torch.cuda.synchronize()
                heads = sorted({0, H // 2, H - 1})
                rows = torch.linspace(0, Sq - 1, min(Sq, 256), device = "cuda").long()
                ref = fp64_ref(q, k, v, heads, rows)
                # native SDPA on the reference rows / heads only (all of a 12k x 56-head call is 66 GB in math SDPA)
                nat_sub = torch.nn.functional.scaled_dot_product_attention(
                    q[:, rows][:, :, heads].permute(0, 2, 1, 3), k[:, :, heads].permute(0, 2, 1, 3),
                    v[:, :, heads].permute(0, 2, 1, 3))  # [1, h, r, D]
                nat = torch.zeros_like(q[:1])
                nat[:, rows.unsqueeze(1), torch.tensor(heads, device = "cuda").unsqueeze(0)] = nat_sub[0].permute(1, 0, 2).to(nat.dtype)
                diff = (ob.float() - oh.float()).abs()
                emit(a.out, {
                    "mode": "acc", "arm": a.arm or "i8_head", "shape": name, "gain": gain, "cfg_base": cfg_b, "cfg_head": cfg_h,
                    "engaged": engaged, "bit_identical": bool(torch.equal(ob, oh)), "head_repeat_identical": bool(torch.equal(oh, oh2)),
                    "n_diff": int((diff > 0).sum()), "max_abs_diff": float(diff.max()),
                    "base": metrics(ob, ref, heads, rows), "head": metrics(oh, ref, heads, rows),
                    "native_sdpa": metrics(nat, ref, heads, rows),
                    "sha_base": digest(ob), "sha_head": digest(oh), **info})
            del q, k, v, ob, oh, oh2, nat
            torch.cuda.empty_cache()


def mode_profile(a):
    import torch
    from torch.profiler import ProfilerActivity, profile

    M, T = load(a.pkg)
    info = card()
    sh = shapes()
    for name in a.shapes.split(","):
        B, Sq, Sk, H, D = sh[name]
        if config_of(M, D) is None:
            continue
        q, k, v = inputs(sh[name])
        fn = runner(M, T, D, q.dtype)
        n0 = M.STATS["native"]
        with torch.no_grad():
            for _ in range(3):
                fn(q, k, v)
            torch.cuda.synchronize()
            n = 5
            with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(n):
                    fn(q, k, v)
                torch.cuda.synchronize()
        kern = {}
        for e in prof.events():
            if e.device_type != torch.autograd.DeviceType.CUDA:
                continue
            x = kern.setdefault(e.name[:80], {"us_per_call": 0.0, "launches_per_call": 0.0})
            x["us_per_call"] += e.time_range.elapsed_us() / n
            x["launches_per_call"] += 1.0 / n
        dev = sum(x["us_per_call"] for x in kern.values())
        emit(a.out, {"mode": "profile", "shape": name, "pkg": a.pkg, "arm": a.arm or a.pkg, "device_us_per_call": dev, "kernels": kern,
                     "engaged": M.STATS["native"] == n0,
                     **info})


def mode_cold(a):
    import torch

    t0 = time.perf_counter()
    M, T = load(a.pkg)
    t_import = time.perf_counter() - t0
    info = card()
    sh = shapes()
    for name in a.shapes.split(","):
        B, Sq, Sk, H, D = sh[name]
        if config_of(M, D) is None:
            continue
        q, k, v = inputs(sh[name])
        torch.cuda.synchronize()
        fn = runner(M, T, D, q.dtype)
        res = {}
        for i in range(3):
            t0 = time.perf_counter()
            with torch.no_grad():
                fn(q, k, v)
            torch.cuda.synchronize()
            res[f"call{i + 1}_s"] = time.perf_counter() - t0
        res["engaged"] = M.STATS["native"] == 0 and M.STATS["triton"] >= 3
        cache = os.environ.get("TRITON_CACHE_DIR", "")
        files = sum(len(f) for _, _, f in os.walk(cache)) if cache and os.path.isdir(cache) else None
        emit(a.out, {"mode": "cold", "shape": name, "pkg": a.pkg, "phase": a.slot, "import_s": t_import, **res,
                     "cache_files": files, **info})


def mode_sweep(a):
    import torch

    M, T = load("i8_head")
    info = card()
    sh = shapes()
    flush = flush_buf()
    table = json.loads(Path(a.configs).read_text())
    arch = M._gfx_arch(0) if M._is_rocm() else info["arch"]
    cands = table.get(arch) or table.get("default")
    for name in a.shapes.split(","):
        B, Sq, Sk, H, D = sh[name]
        base_cfg = config_of(M, D)
        if base_cfg is None:
            continue
        q, k, v = inputs(sh[name])
        with torch.no_grad():
            ref = T.int8_attention(q, k, v, base_cfg)
        heads = sorted({0, H // 2, H - 1})
        rows = torch.linspace(0, Sq - 1, min(Sq, 256), device = "cuda").long()
        oracle = fp64_ref(q, k, v, heads, rows)
        for c in [base_cfg] + [dict(base_cfg, **x) for x in cands.get(str(D), [])]:
            c = {kk: vv for kk, vv in c.items() if kk not in ("long", "long_keys")}
            try:
                fn = lambda q, k, v, c = c: T.int8_attention(q, k, v, c)  # noqa: E731
                res = time_shape(fn, q, k, v, flush, budget_ms = 900.0)
                with torch.no_grad():
                    got = fn(q, k, v)
                same = bool(torch.equal(got, ref))
                acc = metrics(got, oracle, heads, rows)
                acc["max_abs_vs_table"] = float((got.float() - ref.float()).abs().max())
                del got
            except Exception as exc:  # noqa: BLE001
                res, same, acc = {"error": repr(exc)[:300]}, None, {}
            emit(a.out, {"mode": "sweep", "shape": name, "cfg": c, "t": res, "bit_identical_to_table": same, "acc": acc,
                         **info})
        del q, k, v, ref
        torch.cuda.empty_cache()


def _isa(ck):
    """Counts of the product instructions in one compiled kernel (PTX on NVIDIA, AMDGCN on ROCm) and its TTGIR dots."""
    import re

    asm = getattr(ck, "asm", {}) or {}
    out = {}
    ptx = asm.get("ptx") or ""
    if isinstance(ptx, bytes):
        ptx = ptx.decode(errors = "replace")
    if ptx:
        out["mma_s8_s32"] = len(re.findall(r"mma\.sync\.aligned\.\S*s32\.s8\.s8\.s32", ptx))
        out["mma_bf16"] = len(re.findall(r"mma\.sync\.aligned\.\S*bf16", ptx))
        out["mma_f16"] = len(re.findall(r"mma\.sync\.aligned\.\S*f32\.f16\.f16", ptx))
        out["local_ld_st"] = len(re.findall(r"\b(ld|st)\.local", ptx))
    gcn = asm.get("amdgcn") or ""
    if isinstance(gcn, bytes):
        gcn = gcn.decode(errors = "replace")
    if gcn:
        out["wmma_iu8"] = len(re.findall(r"v_wmma_i32_16x16x16_iu8", gcn))
        out["wmma_f16"] = len(re.findall(r"v_wmma_f32_16x16x16_f16", gcn))
        out["wmma_bf16"] = len(re.findall(r"v_wmma_f32_16x16x16_bf16", gcn))
        out["scratch"] = len(re.findall(r"scratch_(load|store)", gcn))
    ttgir = asm.get("ttgir") or ""
    if ttgir:
        out["ttgir_dots"] = re.findall(r"tt\.dot[^:\n]*:\s*(tensor<[^>]*>)", ttgir)[:4]
    md = getattr(ck, "metadata", None)
    for key in ("num_warps", "num_stages", "shared"):
        if md is not None and hasattr(md, key):
            out[key] = getattr(md, key)
    n_regs = getattr(ck, "n_regs", None)
    if n_regs is not None:
        out["n_regs"] = n_regs
        out["n_spills"] = getattr(ck, "n_spills", None)
    return out


def mode_variants(a):
    """Every shape through the table config, then the compiled variants per kernel (resources README 13.2) and the
    product instructions of each compiled attention kernel."""
    import torch

    M, T = load(a.pkg)
    info = card()
    sh = shapes()
    for name in a.shapes.split(","):
        B, Sq, Sk, H, D = sh[name]
        cfg = config_of(M, D)
        if cfg is None:
            continue
        q, k, v = inputs(sh[name])
        with torch.no_grad():
            T.int8_attention(q, k, v, cfg)
        del q, k, v
    torch.cuda.synchronize()
    counts, isa = {}, {}
    for kname in ("prep_kernel", "kv_kernel", "attn_kernel"):
        fn = getattr(T, kname)
        n = 0
        for cache in fn.device_caches.values():
            kernels = cache[0]
            n += len(kernels)
            if kname == "attn_kernel":
                for i, ck in enumerate(kernels.values()):
                    isa[f"attn_{i}"] = _isa(ck)
        counts[kname] = n
    emit(a.out, {"mode": "variants", "pkg": a.pkg, "shapes": a.shapes, "compiled_variants": counts, "attn_isa": isa,
                 **info})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required = True, choices = ["time", "acc", "profile", "cold", "sweep", "variants"])
    ap.add_argument("--pkg", default = "i8_head")
    ap.add_argument("--shapes", default = "")
    ap.add_argument("--rep", type = int, default = 0)
    ap.add_argument("--slot", default = "")
    ap.add_argument("--configs", default = "")
    ap.add_argument("--arm", default = "")
    ap.add_argument("--out", default = "")
    a = ap.parse_args()
    if not a.shapes:
        a.shapes = ",".join(shapes())
    {"time": mode_time, "acc": mode_acc, "profile": mode_profile, "cold": mode_cold, "sweep": mode_sweep,
     "variants": mode_variants}[a.mode](a)


if __name__ == "__main__":
    main()

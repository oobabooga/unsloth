# SPDX-License-Identifier: AGPL-3.0-only
"""Localise the fp16 eager vs Inductor byte difference of qk_norm_rope_backward (checkout on PYTHONPATH).

EXP json lines, fp16 and fp32, packed False / True:
  bwd_aa_compiled  Inductor-compiled backward op twice (determinism of the compiled launch)
  bwd_nan_fill     compiled backward op after the caching allocator was filled with NaN: unwritten
                   output elements would stay NaN (grid / coverage)
  bwd_positions    differing (row, head, col) of eager vs compiled, with eager, compiled and the fp64
                   value of the same formula at those positions
  bwd_stages3      eager launch with num_stages=3 (the Inductor build's option) vs default
  code             Inductor wrapper lines that launch the kernel / allocate its outputs
"""
import json
import os
import re
import sys

import torch

import unsloth  # noqa: F401
from unsloth.kernels import qk_norm_rope as QK
from torch._inductor.utils import run_and_get_code

sys.path.insert(0, os.path.join(os.path.dirname(QK.__file__), "..", "..", "tests"))
import test_qk_norm_rope as T  # noqa: E402


def emit(name, dtype, packed, payload):
    print("EXP", json.dumps({"exp": name, "dtype": str(dtype).split(".")[-1], "packed": packed, **payload},
                            default = str), flush = True)


def ndiff(x, y):
    return int((x != y).sum()) if not (x.isnan().any() or y.isnan().any()) else -1


def ref_bwd(dQo, Q, W, r, cos, sin, pos, n_rows_seq):
    # fp64 of the kernel's backward formula with its own roundings of dn (fp16 dY dtype).
    b, s, h, d = Q.shape
    half = d // 2
    idx = pos.reshape(b, s) if pos is not None else torch.arange(s, device = Q.device).expand(b, s)
    c, sn = cos[idx].double()[:, :, None, :half], sin[idx].double()[:, :, None, :half]
    dy = dQo.double()
    dy0, dy1 = dy[..., :half], dy[..., half:]
    dn0 = (dy0 * c + dy1 * sn).to(dQo.dtype).double()
    dn1 = (dy1 * c - dy0 * sn).to(dQo.dtype).double()
    x = Q.double()
    inv = r.double().view(b, s, h, 1)
    n = x * inv
    g = torch.cat((dn0, dn1), -1) * W.double()
    rowsum = (g * n).sum(-1, keepdim = True)
    return inv / d * (d * g - n * rowsum)


for dtype in (torch.float16, torch.float32):
    for packed in (False, True):
        try:
            q_lin, k_lin, cos, sin, pos, grads, norms = T._case(2, 96, 16, 4, 128, dtype, packed)
            Q = q_lin.view(2, 96, -1, 128)
            K = k_lin.view(2, 96, -1, 128)
            has = pos is not None
            idx = pos.reshape(-1).to(torch.int32).contiguous() if has else cos.new_empty(1, dtype = torch.int32)
            Wq, Wk = norms[0].weight.contiguous(), norms[1].weight.contiguous()
            _, _, rq, rk = torch.ops.unsloth.qk_norm_rope(Q, K, Wq, Wk, cos, sin, idx, T.EPS, T.EPS, has)
            dQs, dKs = grads[0].transpose(1, 2), grads[1].transpose(1, 2)
            bop = torch.ops.unsloth.qk_norm_rope_backward
            args = (dQs, dKs, Q, K, Wq, Wk, rq, rk, cos, sin, idx, has)
            eager = bop(*args)
            torch._dynamo.reset()
            cfn = torch.compile(lambda *x: bop(*x), fullgraph = True)
            comp, codes = run_and_get_code(cfn, *args)
            comp2 = cfn(*args)
            emit("bwd_aa_compiled", dtype, packed, {"dQ": ndiff(comp[0], comp2[0]), "dK": ndiff(comp[1], comp2[1]),
                                                     "eager_vs_comp": [ndiff(eager[0], comp[0]), ndiff(eager[1], comp[1])]})
            torch.cuda.synchronize()
            junk = [torch.full((1 << 22,), float("nan"), device = "cuda", dtype = dtype) for _ in range(16)]
            del junk
            comp3 = cfn(*args)
            torch.cuda.synchronize()
            emit("bwd_nan_fill", dtype, packed, {"nan_dQ": int(comp3[0].isnan().sum()), "nan_dK": int(comp3[1].isnan().sum()),
                                                  "vs_comp": [ndiff(comp3[0], comp[0]), ndiff(comp3[1], comp[1])]})
            ref = ref_bwd(dQs, Q, Wq, rq, cos, sin, pos, 96)
            d = (eager[0] != comp[0]).nonzero().tolist()[:12]
            rows = []
            for (bb, ss, hh, cc) in d:
                rows.append({"pos": [bb, ss, hh, cc], "eager": eager[0][bb, ss, hh, cc].item(),
                             "compiled": comp[0][bb, ss, hh, cc].item(), "ref64": ref[bb, ss, hh, cc].item()})
            closer = sum(abs(r["eager"] - r["ref64"]) < abs(r["compiled"] - r["ref64"]) for r in rows)
            emit("bwd_positions", dtype, packed, {"n": len(d), "eager_closer": closer, "rows": rows,
                                                   "max_err_eager": (eager[0].double() - ref).abs().max().item(),
                                                   "max_err_comp": (comp[0].double() - ref).abs().max().item()})
            # Eager launch of the same kernel with num_stages=3.
            batch, seq, nq, hd = Q.shape
            rows_ = batch * seq
            dQ3 = torch.empty_like(Q)
            dK3 = torch.empty_like(K)
            bh = QK._block_h(nq, K.shape[2], hd)
            nb = (nq + bh - 1) // bh + (K.shape[2] + bh - 1) // bh
            Q2, K2 = Q.reshape(rows_, nq, hd), K.reshape(rows_, K.shape[2], hd)
            QK._qk_norm_rope_backward[(rows_, nb)](
                dQ3, dQs, dQs.stride(0), dQs.stride(1), dQs.stride(2), Q2, Q2.stride(0), Q2.stride(1), Wq, rq, nq,
                dK3, dKs, dKs.stride(0), dKs.stride(1), dKs.stride(2), K2, K2.stride(0), K2.stride(1), Wk, rk, K.shape[2],
                cos, cos.stride(0), sin, sin.stride(0), idx, seq,
                HEAD_DIM = hd, BLOCK_H = bh, HAS_ROPE_INDICES = has, num_warps = QK._num_warps(bh, hd),
                enable_fp_fusion = True, num_stages = 3,
            )
            emit("bwd_stages3", dtype, packed, {"vs_eager": [ndiff(dQ3, eager[0]), ndiff(dK3, eager[1])],
                                                 "vs_comp": [ndiff(dQ3, comp[0]), ndiff(dK3, comp[1])]})
            if dtype == torch.float16 and not packed:
                lines = [l for c in codes for l in c.splitlines()
                         if re.search(r"qk_norm_rope|empty_strided|\.run\(|async_compile\.triton|triton_meta|grid|stream|reinterpret", l)]
                emit("code", dtype, packed, {"lines": lines[:80]})
                with open(f"inductor_bwd_code_{os.environ.get('QK_STATE', 'x')}.py", "w", encoding = "utf-8") as fh:
                    fh.write("\n\n#####\n\n".join(codes))
        except Exception as ex:
            import traceback
            emit("error", dtype, packed, {"error": f"{type(ex).__name__}: {str(ex)[:400]}", "tb": traceback.format_exc()[-1500:]})

# Binary vs launcher: the Inductor-built kernel object, launched by Inductor and by Triton's own launcher.
import hashlib, inspect


def _bin(ck):
    for k in ("hsaco", "cubin"):
        try:
            return ck.asm[k]
        except KeyError:
            pass
    return b""

from torch._inductor.codecache import PyCodeCache
for packed in (False, True):
    info = {}
    try:
        dtype = torch.float16
        q_lin, k_lin, cos, sin, pos, grads, norms = T._case(2, 96, 16, 4, 128, dtype, packed)
        Q = q_lin.view(2, 96, -1, 128); K = k_lin.view(2, 96, -1, 128)
        has = pos is not None
        idx = pos.reshape(-1).to(torch.int32).contiguous() if has else cos.new_empty(1, dtype = torch.int32)
        Wq, Wk = norms[0].weight.contiguous(), norms[1].weight.contiguous()
        _, _, rq, rk = torch.ops.unsloth.qk_norm_rope(Q, K, Wq, Wk, cos, sin, idx, T.EPS, T.EPS, has)
        dQs, dKs = grads[0].transpose(1, 2), grads[1].transpose(1, 2)
        bop = torch.ops.unsloth.qk_norm_rope_backward
        args = (dQs, dKs, Q, K, Wq, Wk, rq, rk, cos, sin, idx, has)
        eager = bop(*args)
        torch._dynamo.reset()
        comp = torch.compile(lambda *x: bop(*x), fullgraph = True)(*args)
        info["eager_vs_comp"] = [ndiff(eager[0], comp[0]), ndiff(eager[1], comp[1])]
        import gc
        tuners = [o for o in gc.get_objects() if type(o).__name__ == "CachingAutotuner"
                  and "_qk_norm_rope_backward" in str(getattr(o, "inductor_meta", {}).get("kernel_name", ""))]
        info["n_tuners"] = len(tuners)
        tuners = [o for o in tuners if str(o.triton_meta.get("signature", {}).get("dQ")) == "*fp16"
                  and o.triton_meta.get("constants", {}).get("HAS_ROPE_INDICES") == has]
        info["n_match"] = len(tuners)
        tuner = tuners[-1] if tuners else None
        info["tuner"] = type(tuner).__name__
        L = tuner.launchers[0]
        info["launcher_attrs"] = {k: str(getattr(L, k))[:600] for k in ("def_args", "call_args", "config", "n_regs", "n_spills", "shared", "cache_hash") if hasattr(L, k)}
        cr = getattr(tuner, "compile_results", None) or []
        kern = None
        for c in cr:
            kern = getattr(c, "kernel", None) or kern
        if kern is not None:
            md = kern.metadata
            info["ind_meta"] = {k: getattr(md, k, None) for k in ("num_warps", "num_stages", "warp_size", "shared", "arch", "enable_fp_fusion")}
            info["ind_hsaco_sha"] = hashlib.sha256(_bin(kern)).hexdigest()[:16]
        # Eager build(s) of the same kernel in this process.
        jit = QK._qk_norm_rope_backward
        eb = []
        for dev_cache in jit.device_caches.values():
            for key, ck in dev_cache[0].items():
                eb.append({"sha": hashlib.sha256(_bin(ck)).hexdigest()[:16], "num_warps": ck.metadata.num_warps,
                           "num_stages": ck.metadata.num_stages, "warp_size": getattr(ck.metadata, "warp_size", None)})
        info["eager_builds"] = eb
        # Dump the Inductor binary's IR + metadata and every eager build's, for an offline diff.
        def _md(ck):
            m = ck.metadata
            return {k: str(v)[:300] for k, v in (m._asdict().items() if hasattr(m, "_asdict") else vars(m).items())}
        def _dump(tag, ck):
            st = os.environ.get("QK_STATE", "x")
            for k in ("ttir", "ttgir", "llir", "amdgcn", "ptx"):
                try:
                    v = ck.asm[k]
                except KeyError:
                    continue
                with open(f"dump_{st}_{packed}_{tag}.{k}", "w", encoding = "utf-8") as fh:
                    fh.write(v if isinstance(v, str) else v.decode("utf-8", "replace"))
            with open(f"dump_{st}_{packed}_{tag}.meta.json", "w", encoding = "utf-8") as fh:
                json.dump(_md(ck), fh, indent = 1)
        if kern is not None:
            _dump("inductor", kern)
            info["ind_md"] = _md(kern)
        n = 0
        for dev_cache in jit.device_caches.values():
            for key, ck in dev_cache[0].items():
                _dump(f"eager{n}", ck)
                n += 1
        try:
            src = inspect.getsource(L)
        except Exception:
            src = getattr(L, "__doc__", None)
        info["launcher_src"] = str(src)[:1500]
        # Launch the Inductor-built CompiledKernel through Triton's launcher.
        if kern is not None:
            batch, seq, nq, hd = Q.shape
            rows_ = batch * seq
            Q2, K2 = Q.reshape(rows_, nq, hd), K.reshape(rows_, K.shape[2], hd)
            dQ3 = torch.empty_like(Q); dK3 = torch.empty_like(K)
            kern[(rows_, 5, 1)](dQ3, dQs, dQs.stride(0), dQs.stride(1), dQs.stride(2), Q2, Q2.stride(0), Q2.stride(1), Wq, rq, nq,
                                dK3, dKs, dKs.stride(0), dKs.stride(1), dKs.stride(2), K2, K2.stride(0), K2.stride(1), Wk, rk, K.shape[2],
                                cos, cos.stride(0), sin, sin.stride(0), idx, seq, 128, 4, has)
            torch.cuda.synchronize()
            info["ind_kernel_via_triton_vs_eager"] = [ndiff(dQ3, eager[0]), ndiff(dK3, eager[1])]
            info["ind_kernel_via_triton_vs_comp"] = [ndiff(dQ3, comp[0]), ndiff(dK3, comp[1])]
    except Exception as ex:
        import traceback
        info["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
        info["tb"] = traceback.format_exc()[-1500:]
    emit("launcher", torch.float16, packed, info)
torch._dynamo.reset()
print("DONE2", flush = True)

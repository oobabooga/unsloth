"""16-bit MoE grouped GEMMs (training / prefill), per GEMM of a routed expert MLP: gate_up (K=H, N=2I) and down (K=I, N=H).

Entry points (what models call):
  * zoo moe_grouped_fp16.generic_grouped_mm / generic_grouped_wgrad  <- moe_utils._grouped_mm_with_backward_fix
    (DEFAULT backend "grouped_mm": Triton replaces torch._grouped_mm on sm80 / sm89 / sm120 / sm121 under the
    _TRITON_GROUPED_MM_AUTO_ROWS row limits; sm90 / sm100 / sm75 keep torch._grouped_mm). Qwen3-MoE, gpt-oss (bf16),
    Qwen3.5-MoE, GLM ... LoRA + full FT, train + prefill; 16-bit and 4-bit (after expert dequant).
  * unsloth kernels/moe/grouped_gemm (grouped_gemm fwd / dX / dW)    <- moe_utils.forward_triton_grouped_gemm
    (opt-in UNSLOTH_MOE_BACKEND=unsloth_triton; configs from autotune_cache.get_or_autotune_moe_kernels).
References: torch._grouped_mm (cuBLAS grouped / per-group fallback), per-expert torch.matmul loop (cuBLAS).
No torch.compile arm: a routed per-expert loop over data-dependent counts does not compile to one graph and
torch.compile(torch._grouped_mm) lowers to the same aten op outside max-autotune.
"""
import os, sys, ast, time, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
import moe_common as mc

CASES = [
    dict(name="qwen3_30b_T4096", model="qwen3_30b_a3b", T=4096, autotune=True),
    dict(name="qwen3_30b_T512", model="qwen3_30b_a3b", T=512),
    dict(name="gpt_oss_20b_T4096", model="gpt_oss_20b", T=4096, skip_small=True),
    dict(name="gpt_oss_20b_T512", model="gpt_oss_20b", T=512),
    dict(name="qwen3_30b_T8_host", model="qwen3_30b_a3b", T=8, host=True),
]
BUDGET = dict(budget_ms=150)


def _zoo_gate_table():
    """_TRITON_GROUPED_MM_AUTO_ROWS from moe_utils.py source (no import: moe_utils pulls transformers)."""
    base = sys.modules["unsloth_zoo"].__path__[0]
    src = open(os.path.join(base, "temporary_patches", "moe_utils.py")).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_TRITON_GROUPED_MM_AUTO_ROWS":
            return ast.literal_eval(node.value)
    return {}


def zoo_gate(table, cap, M, E, K, N, kind):
    lim = table.get(cap, {}).get(kind, 0)
    return lim != 0 and (lim < 0 or M <= lim * E)


def import_gg():
    setup_paths()
    try:
        import pandas  # noqa: F401
    except Exception:
        sys.modules["pandas"] = types.ModuleType("pandas")
    import importlib, importlib.util
    spec = importlib.util.find_spec("unsloth")
    from harness import _stub_pkg
    base = os.path.dirname(spec.origin) if "unsloth" not in sys.modules else sys.modules["unsloth"].__path__[0]
    u = _stub_pkg("unsloth", base)
    u.DEVICE_TYPE = "cuda"
    for s in ("unsloth.kernels", "unsloth.kernels.moe"):
        _stub_pkg(s, os.path.join(base, *s.split(".")[1:]))
    gi = importlib.import_module("unsloth.kernels.moe.grouped_gemm.interface")
    ac = importlib.import_module("unsloth.kernels.moe.autotune_cache")
    return gi, ac


def cfg_kwargs(c, kind):
    d = dict(BLOCK_SIZE_M=c.BLOCK_SIZE_M, BLOCK_SIZE_N=c.BLOCK_SIZE_N, BLOCK_SIZE_K=c.BLOCK_SIZE_K,
             num_warps=c.num_warps, num_stages=c.num_stages)
    return d


def run(case, dtype, H):
    global torch
    import torch
    mg = import_zoo("unsloth_zoo.temporary_patches.moe_grouped_fp16")
    gi, ac = import_gg()
    from unsloth.kernels.moe.grouped_gemm.kernels import forward as gfwd
    dev = torch.device("cuda", 0)
    cap = torch.cuda.get_device_capability(0)
    table = _zoo_gate_table()
    zoo_ok = mg.triton_grouped_available(dev)
    Hd, E, k, I = mc.MODELS[case["model"]]
    T = case["T"]
    small = mc.small_gpu()
    mc.tune_budget(H)
    if small and case.get("skip_small"):
        H.emit(kind="skip", why="small GPU (<20 GB): case dropped to fit the T4 time budget")
        return
    if small and T > 2048:
        T = 2048
    H.emit(kind="info", what="shape_used", T=T, small_gpu=small)
    dt = H.dtype
    idx, w = mc.route(T, E, k)
    sorted_idx, tok, counts, offs = mc.sort_routing(idx, E)
    cl = counts.tolist()
    M = T * k
    H.emit(kind="info", what="routing", M=M, E=E, rows_per_expert=M / E, max_count=max(cl), min_count=min(cl),
           zoo_triton_selftest=zoo_ok, has_grouped_mm=mc.has_grouped_mm(), tma=gi.supports_tma())
    heur = ac._get_heuristic_configs(n_dim=2 * I, k_dim=Hd)
    gemms = dict(gate_up=(Hd, 2 * I), down=(I, Hd))
    for gname, (K, N) in gemms.items():
        if case.get("host") and gname == "down":
            continue
        kind = "lora" if min(K, N) <= 32 else ("many" if E >= 64 else "few")
        gate = dict(fwd=zoo_gate(table, cap, M, E, K, N, kind), dX=zoo_gate(table, cap, M, E, N, K, kind),
                    dW=zoo_gate(table, cap, M, E, K, N, "dw") or min(K, N) <= 32)
        H.emit(kind="info", what="zoo_gate_" + gname, cls=kind, uses_triton=gate)
        g = torch.Generator(device="cuda").manual_seed(1)
        W_nk = (torch.randn(E, N, K, device="cuda", generator=g) * 0.02).to(dt)  # HF [E, out, in]
        W_kn = W_nk.transpose(-2, -1)  # what zoo hands torch._grouped_mm (transposed view)
        x = torch.randn(M, K, device="cuda", dtype=dt)
        gy = torch.randn(M, N, device="cuda", dtype=dt)
        shape = [M, K, N, E]
        # unsloth heuristic configs re-fit to this GEMM's dims (forward_triton_grouped_gemm fits per GEMM)
        hc = ac._get_heuristic_configs(n_dim=N, k_dim=K)
        fkw, dxkw, dwkw = (cfg_kwargs(c, None) for c in hc)
        ginds = torch.arange(M, device="cuda", dtype=torch.int32)
        arms = dict(
            zoo_triton=lambda: mg.generic_grouped_mm(x, W_kn, offs),
            torch_gmm=lambda: torch._grouped_mm(x, W_kn, offs=offs),
            unsloth_triton_heur=lambda: gi._grouped_gemm_forward_impl(x, W_nk, k, counts, permute_x=False, permute_y=False, **fkw),
        )
        if M <= 8192 or E <= 32:
            arms["loop_cublas"] = lambda: mc.loop_mm(x, W_kn, cl)
        H.interleave(arms, "fwd_" + gname, shape=shape, host=case.get("host", False), **BUDGET)
        if T >= 512:
            # dX = gy @ W[e]^T  (N -> K)
            arms = dict(
                zoo_triton=lambda: mg.generic_grouped_mm(gy, W_nk, offs),
                torch_gmm=lambda: torch._grouped_mm(gy, W_kn.transpose(-2, -1), offs=offs),
                unsloth_triton_heur=lambda: gi._grouped_gemm_dX_impl(gy, W_nk, ginds, counts, k, permute_x=False, permute_y=False, **dxkw),
            )
            if E <= 32:
                arms["loop_cublas"] = lambda: mc.loop_mm(gy, W_nk, cl)
            H.interleave(arms, "bwd_dX_" + gname, shape=shape, **BUDGET)
            # dW[e] = x_e^T gy_e  (full FT only; LoRA keeps base frozen)
            arms = dict(
                zoo_triton=lambda: mg.generic_grouped_wgrad(x, gy, offs, E),
                torch_gmm=lambda: torch._grouped_mm(x.t(), gy, offs=offs),
                unsloth_triton_heur=lambda: gi._grouped_gemm_dW_impl(x, gy, counts, ginds, k, permute_x=False, permute_y=False, **dwkw),
            )
            if E <= 32:
                arms["loop_cublas"] = lambda: mc.loop_wgrad(x, gy, cl, E)
            H.interleave(arms, "bwd_dW_" + gname, shape=shape, **BUDGET)
        torch.cuda.empty_cache()
        # parity on an expert subset (first experts covering <= 2048 rows, at most 8 experts): keeps the fp32
        # reference under the T4's memory; the kernels run the same tiles on the subset
        csum = offs.tolist()
        ep = max(1, min(8, sum(1 for c in csum if c <= 2048)))
        rows = csum[ep - 1]
        if rows == 0:
            ep, rows = E, M
        xs, gys = x[:rows], gy[:rows]
        Wkn_s, Wnk_s = W_kn[:ep], W_nk[:ep].contiguous()
        cnt_s, off_s, cl_s = counts[:ep].contiguous(), offs[:ep].contiguous(), cl[:ep]
        pshape = [rows, K, N, ep]
        r32 = mc.loop_mm(xs.float(), Wkn_s.float(), cl_s)
        try:
            rl = torch._grouped_mm(xs, Wkn_s, offs=off_s)
        except Exception:
            rl = mc.loop_mm(xs, Wkn_s, cl_s)
        H.parity("zoo_fwd_" + gname, mg.generic_grouped_mm(xs, Wkn_s, off_s), r32, rl, shape=pshape)
        H.parity("unsloth_fwd_" + gname, gi._grouped_gemm_forward_impl(xs, Wnk_s, k, cnt_s, permute_x=False, permute_y=False, **fkw), r32, rl, shape=pshape)
        if T >= 512:
            rw32 = mc.loop_wgrad(xs.float(), gys.float(), cl_s, ep)
            try:
                rwl = torch._grouped_mm(xs.t(), gys, offs=off_s)
            except Exception:
                rwl = None
            H.parity("zoo_dW_" + gname, mg.generic_grouped_wgrad(xs, gys, off_s, ep), rw32, rwl, shape=pshape)
            H.parity("unsloth_dW_" + gname, gi._grouped_gemm_dW_impl(xs, gys, cnt_s, ginds[:rows], k, permute_x=False, permute_y=False, **dwkw).transpose(-2, -1), rw32, rwl, shape=pshape)
            del rw32, rwl
        del r32, rl, Wnk_s
        torch.cuda.empty_cache()
        # config sweeps (is the shipped heuristic near-best?), gate_up only
        if gname == "gate_up" and not case.get("host") and cap >= (8, 0) and not small:
            picked = mg.generic_gemm_config(M, E, N, K, dev, dt)
            cand = {tuple(picked)}
            for bm in (32, 64, 128):
                for bn in (64, 128, 256):
                    for bk in (32, 64):
                        cand.add((bm, bn, bk, 8 if bm * bn >= 128 * 128 else 4, 3, picked[5]))
            res = []
            for c in cand:
                try:
                    c2 = mg._fit_smem(c[0], c[1], c[2], c[4], dt.itemsize, mg._smem_limit(dev), False)
                    c = (c2[0], c[1], c2[1], c[3], c2[2], c[5])
                    med = H.measure(lambda: mg.grouped_gemm(x, W_kn, offs, dt, b_trans=False, num_experts=E, config=c, backend="triton", ends=True), budget_ms=60)[0]
                    res.append((med, c))
                except Exception as ex:
                    pass
            res.sort()
            pm = [m for m, c in res if c == tuple(picked)]
            H.emit(kind="config_sweep", what="zoo_generic_gemm_fwd_" + gname, picked=picked, picked_ms=pm[0] if pm else None,
                   best=res[0][1] if res else None, best_ms=res[0][0] if res else None, n=len(res), top3=res[:3])
            # unsloth fwd sweep
            res = []
            for bm in (32, 64, 128):
                for bn in (64, 128, 256):
                    for bk in (64, 128):
                        for wp, st in ((4, 3), (8, 3)):
                            kw = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, num_warps=wp, num_stages=st)
                            try:
                                med = H.measure(lambda: gi._grouped_gemm_forward_impl(x, W_nk, k, counts, permute_x=False, permute_y=False, **kw), budget_ms=40, warmup=2)[0]
                                res.append((med, (bm, bn, bk, wp, st)))
                            except Exception:
                                pass
            res.sort()
            hp = (fkw["BLOCK_SIZE_M"], fkw["BLOCK_SIZE_N"], fkw["BLOCK_SIZE_K"], fkw["num_warps"], fkw["num_stages"])
            pm = [m for m, c in res if c == hp]
            H.emit(kind="config_sweep", what="unsloth_heuristic_fwd_" + gname, picked=hp, picked_ms=pm[0] if pm else None,
                   best=res[0][1] if res else None, best_ms=res[0][0] if res else None, n=len(res), top3=res[:3])
            # real Triton autotune (the unsloth_triton backend's first-call path), one case only
            if case.get("autotune") and os.environ.get("AUDIT_MOE_AUTOTUNE", "1") == "1":
                t0 = time.time()
                try:
                    gi._grouped_gemm_forward_impl(x, W_nk, k, counts, permute_x=False, permute_y=False, autotune=True)
                    torch.cuda.synchronize()
                    H.emit(kind="info", what="unsloth_autotune_wall_s_fwd_" + gname, secs=time.time() - t0)
                    H.autotune_report("unsloth_fwd_" + gname, gfwd._autotuned_grouped_gemm_forward_kernel)
                    H.time_arm("unsloth_triton_autotuned", "fwd_" + gname,
                               lambda: gi._grouped_gemm_forward_impl(x, W_nk, k, counts, permute_x=False, permute_y=False, autotune=True),
                               shape=shape, **BUDGET)
                except Exception as ex:
                    H.emit(kind="error", arm="unsloth_autotune", phase="fwd_" + gname, err=repr(ex)[:400])
        if gname == "gate_up" and T == 512 and not (small and case["model"] != "qwen3_30b_a3b"):
            x2 = x[: M // 2].contiguous()
            offs2 = torch.clamp(offs, max=M // 2)
            H.compile_check("zoo_generic_grouped_mm", lambda a, o: mg.generic_grouped_mm(a, W_kn, o), [(x, offs), (x2, offs2)])
            c2 = torch.diff(offs2, prepend=offs2.new_zeros(1))
            H.compile_check("unsloth_grouped_gemm_fwd",
                            lambda a, c: gi.grouped_gemm(a, W_nk, c, k, kernel_config_fwd=hc[0], is_first_gemm=True),
                            [(x, counts), (x2, c2)])
        del W_nk, x, gy
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(CASES, run)

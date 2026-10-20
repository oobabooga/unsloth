"""GDN output norm + qk l2norm (Qwen3.5 / Qwen3.6 / Qwen3-Next linear-attention layers).

  FusedRMSNormGated(head_v_dim=128, activation='silu')(x, gate)   o_norm in every GDN layer, train + infer
  l2norm(x)  standalone fla l2norm (in GDN it runs inside chunk_gated_delta_rule when
             use_qk_l2norm_in_kernel=True; standalone use: KDA / Kimi-Linear, Ling remote code)
Arms: ours (zoo-vendored fla 0.5.1) / fla_latest / eager (transformers Qwen3NextRMSNormGated math,
transformers l2norm) / compiled (torch.compile of the same eager math).
Rows = B*T*HV with HV=32, D=128.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
import _fla_dual

HV, D = 32, 128
CASES = [
    dict(name="normgate_T4096", kind="norm", rows=4096 * HV),
    dict(name="normgate_T16384", kind="norm", rows=16384 * HV, big=True),
    dict(name="normgate_decode_B8", kind="norm", rows=8 * HV, host=True),
    dict(name="l2norm_T4096", kind="l2", rows=4096 * HV),
    dict(name="l2norm_T16384", kind="l2", rows=16384 * HV, big=True),
    dict(name="l2norm_tiny", kind="l2", rows=8 * HV, host=True),
]


def _small_gpu():
    import torch
    return os.environ.get("AUDIT_FORCE_SMALL") == "1" or torch.cuda.get_device_properties(0).total_memory < 20 * 2**30


# < 20 GB cards (Colab T4: ~10x B200 per case, 14.5 GB): drop the big=True shapes, keep the rest
if _small_gpu():
    CASES = [c for c in CASES if not c.get("big")]


def ref_norm(x, gate, weight, eps=1e-6):
    input_dtype = x.dtype
    h = x.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    h = weight * h.to(input_dtype)
    h = h * torch.nn.functional.silu(gate.to(torch.float32))
    return h.to(input_dtype)


def ref_l2(x, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)
    return x * inv_norm


def run(case, dtype, H):
    global torch
    import torch
    ns = _fla_dual.load_both()
    impls = ["vendored"] + (["latest"] if "latest" in ns else [])
    if "latest" not in ns:
        H.emit(kind="info", what="fla_latest_missing", err=ns.get("latest_err"))
    an = {"vendored": "ours", "latest": "fla_latest"}
    dt = H.dtype
    R = case["rows"]
    shape = [R, D]
    x = torch.randn(R, D, device="cuda", dtype=dt, requires_grad=True)
    dy = torch.randn(R, D, device="cuda", dtype=dt)
    if case["kind"] == "norm":
        gate = torch.randn(R, D, device="cuda", dtype=dt, requires_grad=True)
        mods = {}
        for i in impls:
            _fla_dual.use(i)
            m = ns[i]["FusedRMSNormGated"](D, eps=1e-6, activation="silu", device="cuda", dtype=dt)
            with torch.no_grad():
                m.weight.copy_(1 + 0.1 * torch.randn(D, device="cuda"))
            mods[i] = m
        w = mods["vendored"].weight.detach().clone().requires_grad_(True)
        for i in impls[1:]:
            with torch.no_grad():
                mods[i].weight.copy_(w)
        fwd = {i: _fla_dual.wrap(i, mods[i]) for i in impls}
        cref = torch.compile(ref_norm, dynamic=False)
        args = (x, gate)

        def mk_f(fn, ref=False):
            def f():
                with torch.no_grad():
                    return fn(x, gate, w) if ref else fn(x, gate)
            return f

        def mk_fb(fn, ref=False):
            def f():
                y = fn(x, gate, w) if ref else fn(x, gate)
                y.backward(dy)
            return f
        arms_f = {an[i]: mk_f(fwd[i]) for i in impls}
        arms_f.update(eager=mk_f(ref_norm, True), compiled=mk_f(cref, True))
        arms_fb = {an[i]: mk_fb(fwd[i]) for i in impls}
        arms_fb.update(eager=mk_fb(ref_norm, True), compiled=mk_fb(cref, True))
        for f in list(arms_f.values()) + list(arms_fb.values()):
            f()
        H.interleave(arms_f, "fwd", shape=shape, host=case.get("host", False))
        H.interleave(arms_fb, "fwdbwd", shape=shape, host=case.get("host", False))
        # parity vs fp64
        x64 = x.detach().double().requires_grad_(True); g64 = gate.detach().double().requires_grad_(True)
        w64 = w.detach().double().requires_grad_(True)
        h = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + 1e-6)
        y64 = w64 * h * torch.nn.functional.silu(g64)
        y64.backward(dy.double())
        for t in (x, gate, w):
            t.grad = None
        yr = ref_norm(x, gate, w); yr.backward(dy)
        gr = (x.grad.clone(), gate.grad.clone(), w.grad.clone())
        for i in impls:
            for t in (x, gate):
                t.grad = None
            mods[i].weight.grad = None
            y = fwd[i](x, gate); y.backward(dy)
            H.parity("%s_y" % an[i], y.detach(), y64.detach(), yr.detach(), shape=shape)
            H.parity("%s_grads(x,gate,w)" % an[i], (x.grad, gate.grad, mods[i].weight.grad),
                     (x64.grad, g64.grad, w64.grad), gr, shape=shape)
        _fla_dual.use("vendored")
        m0 = mods["vendored"]
        x2 = torch.randn(R // 2, D, device="cuda", dtype=dt, requires_grad=True)
        g2 = torch.randn(R // 2, D, device="cuda", dtype=dt, requires_grad=True)
        H.compile_check("ours_FusedRMSNormGated", lambda a, b: m0(a, b), [(x, gate), (x2, g2)])
    else:
        x4 = x.detach().view(R // HV, HV, D).requires_grad_(True)
        dy4 = dy.view(R // HV, HV, D)
        fns = {i: _fla_dual.wrap(i, ns[i]["l2norm"]) for i in impls if ns[i].get("l2norm")}
        cref = torch.compile(ref_l2, dynamic=False)

        def mk_f(fn):
            def f():
                with torch.no_grad():
                    return fn(x4)
            return f

        def mk_fb(fn):
            def f():
                fn(x4).backward(dy4)
            return f
        arms_f = {an[i]: mk_f(f) for i, f in fns.items()}
        arms_f.update(eager=mk_f(ref_l2), compiled=mk_f(cref))
        arms_fb = {an[i]: mk_fb(f) for i, f in fns.items()}
        arms_fb.update(eager=mk_fb(ref_l2), compiled=mk_fb(cref))
        for f in list(arms_f.values()) + list(arms_fb.values()):
            f()
        H.interleave(arms_f, "fwd", shape=shape, host=case.get("host", False))
        H.interleave(arms_fb, "fwdbwd", shape=shape, host=case.get("host", False))
        x64 = x4.detach().double().requires_grad_(True)
        y64 = ref_l2(x64); y64.backward(dy4.double())
        x4.grad = None
        yr = ref_l2(x4); yr.backward(dy4); gr = x4.grad.clone()
        for i, f in fns.items():
            x4.grad = None
            y = f(x4); y.backward(dy4)
            H.parity("%s_l2" % an[i], (y.detach(), x4.grad), (y64.detach(), x64.grad), (yr.detach(), gr), shape=shape)
        _fla_dual.use("vendored")
        f0 = ns["vendored"]["l2norm"]
        x5 = torch.randn(R // HV // 2, HV, D, device="cuda", dtype=dt, requires_grad=True)
        H.compile_check("ours_l2norm", lambda a: f0(a), [(x4,), (x5,)])
    for i in impls:
        import b_fla_gdn
        b_fla_gdn.autotune_summary(H, i)
    _fla_dual.use("vendored")


def _run_with_peak(case, dtype, H):
    import torch
    torch.cuda.reset_peak_memory_stats()
    try:
        run(case, dtype, H)
    finally:
        H.emit(kind="memory", what="case_peak", peak_gib=torch.cuda.max_memory_allocated() / 2**30,
               total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30)


if __name__ == "__main__":
    main(CASES, _run_with_peak)

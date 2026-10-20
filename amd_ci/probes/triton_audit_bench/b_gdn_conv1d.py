"""zoo temporary_patches/gdn_causal_conv1d.py: Triton causal depthwise conv (+SiLU), installed as
causal_conv1d_fn / causal_conv1d_update for Qwen3.5 / Qwen3.6 / Qwen3-Next GDN (and Mamba2 hybrids on
transformers >= 5.15) when the causal-conv1d CUDA package is absent (default on Colab).
  train / prefill: _causal_conv1d_any(x_bdt, weight, bias, 'silu')  fwd + bwd (autograd.Function)
  decode:          _try_fast_update(x_bdt, conv_state, weight, bias, 'silu')  T=1, in-place state
Arms: ours / eager (transformers F.conv1d fallback) / compiled (torch.compile of it) /
fla_latest + fla_vendored (fla's own Triton causal_conv1d, [B,T,D] layout; fla_vendored is the Ling path).
Layout as the modeling code hands it: x = mixed_qkv.transpose(1, 2), a channel-last (B, D, T) view.
Qwen3-Next / Qwen3.5: conv dim = 2*16*128 + 32*128 = 8192, kernel width 4.
Fixed launch configs (no autotune): _FWD_CONFIG / _BWD_CONFIG = (BT 32, BD 64, 2 warps), _UPDATE_CONFIG =
(BD 128, 4 warps); `sweep` cases time alternatives to check the pick.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *

D, W = 8192, 4
CASES = [
    dict(name="conv_B1_T2048", kind="conv", B=1, T=2048, sweep=True),
    dict(name="conv_B1_T8192", kind="conv", B=1, T=8192),
    dict(name="conv_B4_T4096", kind="conv", B=4, T=4096, big=True),
    dict(name="conv_tiny_T16", kind="conv", B=1, T=16, host=True),
    dict(name="update_B1", kind="update", B=1, host=True, sweep=True),
    dict(name="update_B8", kind="update", B=8, host=True),
    dict(name="update_B64", kind="update", B=64, sweep=True),
]


def _small_gpu():
    import torch
    return os.environ.get("AUDIT_FORCE_SMALL") == "1" or torch.cuda.get_device_properties(0).total_memory < 20 * 2**30


# < 20 GB cards (Colab T4: ~10x B200 per case, 14.5 GB): drop the big=True shapes, keep the rest
if _small_gpu():
    CASES = [c for c in CASES if not c.get("big")]


def ref_conv(x, weight, bias=None):
    F = torch.nn.functional
    T = x.shape[2]
    out = F.conv1d(x.to(weight.dtype), weight.unsqueeze(1), bias, padding=weight.shape[-1] - 1, groups=x.shape[1])[:, :, :T]
    return F.silu(out).to(x.dtype)


def ref_update(hidden_states, conv_state, weight, bias=None):
    F = torch.nn.functional
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype)


def run(case, dtype, H):
    global torch
    import torch
    m = import_zoo("unsloth_zoo.temporary_patches.gdn_causal_conv1d", stub=("unsloth_zoo", "unsloth_zoo.temporary_patches"))
    dt = H.dtype
    B = case["B"]
    weight = (torch.randn(D, W, device="cuda", dtype=dt) * 0.5)
    bias = None  # Qwen3-Next / Qwen3.5 conv1d has bias=False
    if case["kind"] == "conv":
        T = case["T"]
        shape = [B, D, T]
        xbtd = torch.randn(B, T, D, device="cuda", dtype=dt)
        x = xbtd.transpose(1, 2).detach().requires_grad_(True)  # (B, D, T) channel-last view, as the model hands it
        wq = weight.clone().requires_grad_(True)
        dy = torch.randn(B, D, T, device="cuda", dtype=dt)
        stats0 = dict(m.GDN_CAUSAL_CONV1D_STATS)
        ours = lambda: m._causal_conv1d_any(x, wq, bias, "silu")
        cref = torch.compile(ref_conv, dynamic=False)
        arms_f = dict(ours=ours, eager=lambda: ref_conv(x, wq), compiled=lambda: cref(x, wq))
        # fla causal_conv1d arms ([B, T, D] layout, returns (y, final_state))
        try:
            import _fla_dual
            ns = _fla_dual.load_both()
            for impl, nm in (("latest", "fla_latest"), ("vendored", "fla_vendored")):
                if impl in ns and ns[impl].get("causal_conv1d"):
                    f = _fla_dual.wrap(impl, ns[impl]["causal_conv1d"])
                    xb = x.transpose(1, 2)
                    arms_f[nm] = (lambda f=f, xb=xb: f(xb, weight=wq, bias=None, activation="silu")[0])
        except Exception as ex:
            H.emit(kind="info", what="fla_conv_unavailable", err=repr(ex)[:300])

        def nog(f):
            def g():
                with torch.no_grad():
                    return f()
            return g

        def fb(f, tr=False):
            # fla arms return y as (B, T, D): backprop the same dy in that layout
            def g():
                y = f()
                y.backward(dy.transpose(1, 2) if tr else dy)
            return g
        fwd_arms = {k: nog(v) for k, v in arms_f.items()}
        fb_arms = {k: fb(v, tr=k.startswith("fla")) for k, v in arms_f.items()}
        for f in list(fwd_arms.values()) + list(fb_arms.values()):
            try:
                f()
            except Exception as ex:
                H.emit(kind="error", arm="warmup", err=repr(ex)[:300])
        H.interleave(fwd_arms, "fwd", shape=shape, host=case.get("host", False))
        H.interleave(fb_arms, "fwdbwd", shape=shape, host=case.get("host", False))
        st = {k: m.GDN_CAUSAL_CONV1D_STATS[k] - stats0.get(k, 0) for k in m.GDN_CAUSAL_CONV1D_STATS}
        H.emit(kind="info", what="engagement", stats=st)
        # parity vs fp64
        x64 = x.detach().double().requires_grad_(True); w64 = wq.detach().double().requires_grad_(True)
        y64 = ref_conv(x64, w64); y64.backward(dy.double())
        x.grad = None; wq.grad = None
        yr = ref_conv(x, wq); yr.backward(dy); gr = (x.grad.clone(), wq.grad.clone())
        x.grad = None; wq.grad = None
        yo = ours(); yo.backward(dy)
        H.parity("ours_y_dx_dw", (yo.detach(), x.grad, wq.grad), (y64.detach(), x64.grad, w64.grad), (yr.detach(),) + gr, shape=shape)
        if "fla_latest" in arms_f:
            x.grad = None; wq.grad = None
            yl = arms_f["fla_latest"](); yl.backward(dy.transpose(1, 2))
            H.parity("fla_latest_y_dx_dw", (yl.transpose(1, 2).detach(), x.grad, wq.grad), (y64.detach(), x64.grad, w64.grad), None, shape=shape)
        if case.get("sweep"):
            # launch-config sweep (fixed configs, no autotune): fwd and fwd+bwd
            base_f, base_b = m._FWD_CONFIG, m._BWD_CONFIG
            for cfg in [(32, 64, 2), (64, 64, 4), (32, 128, 4), (64, 128, 4), (16, 128, 4), (128, 64, 4), (32, 256, 8)]:
                m._FWD_CONFIG = cfg; m._BWD_CONFIG = cfg
                m._conv_plans.clear(); m._conv_fast_ok.clear()
                try:
                    fwd_arms["ours"](); fb_arms["ours"]()
                    H.time_arm("sweep_fwd_%d_%d_%d" % cfg, "fwd", fwd_arms["ours"], shape=shape)
                    H.time_arm("sweep_fb_%d_%d_%d" % cfg, "fwdbwd", fb_arms["ours"], shape=shape)
                except Exception as ex:
                    H.emit(kind="error", arm="sweep_%s" % (cfg,), err=repr(ex)[:300])
            m._FWD_CONFIG, m._BWD_CONFIG = base_f, base_b
            m._conv_plans.clear(); m._conv_fast_ok.clear()
            x2 = torch.randn(B, T // 2, D, device="cuda", dtype=dt).transpose(1, 2).requires_grad_(True)
            H.compile_check("ours_causal_conv1d_any", lambda a, w_: m._causal_conv1d_any(a, w_, None, "silu"), [(x, wq), (x2, wq)])
    else:
        L = W  # transformers Qwen3-Next / Qwen3.5 conv_state = (B, D, conv_kernel_size)
        shape = [B, D, 1]
        xbtd = torch.randn(B, 1, D, device="cuda", dtype=dt)
        x = xbtd.transpose(1, 2)
        state0 = torch.randn(B, D, L, device="cuda", dtype=dt)
        s_ours = state0.clone(); s_ref = state0.clone(); s_c = state0.clone()
        cref = torch.compile(ref_update, dynamic=False)
        with torch.no_grad():
            arms = dict(ours=lambda: m._try_fast_update(x, s_ours, weight, bias, "silu"),
                        eager=lambda: ref_update(x, s_ref, weight),
                        compiled=lambda: cref(x, s_c, weight))
            stats0 = dict(m.GDN_CAUSAL_CONV1D_STATS)
            for f in arms.values():
                f()
            H.interleave(arms, "fwd", shape=shape, host=case.get("host", False))
            st = {k: m.GDN_CAUSAL_CONV1D_STATS[k] - stats0.get(k, 0) for k in m.GDN_CAUSAL_CONV1D_STATS}
            H.emit(kind="info", what="engagement", stats=st)
            # parity: 8 chained steps vs fp64
            sa = state0.clone(); s64 = state0.double().clone(); sr = state0.clone()
            ya, y64, yr = [], [], []
            for _ in range(8):
                xi = torch.randn(B, 1, D, device="cuda", dtype=dt).transpose(1, 2)
                ya.append(m._try_fast_update(xi, sa, weight, bias, "silu"))
                y64.append(ref_update(xi.double(), s64, weight.double()))
                yr.append(ref_update(xi, sr, weight))
            H.parity("ours_update_8step", (torch.cat(ya, -1), sa), (torch.cat(y64, -1), s64), (torch.cat(yr, -1), sr), shape=shape)
            if case.get("sweep"):
                base = m._UPDATE_CONFIG
                for cfg in [(128, 4), (64, 2), (256, 4), (512, 4), (1024, 8), (256, 8)]:
                    m._UPDATE_CONFIG = cfg; m._update_plans.clear()
                    try:
                        arms["ours"]()
                        H.time_arm("sweep_upd_%d_%d" % cfg, "fwd", arms["ours"], shape=shape)
                    except Exception as ex:
                        H.emit(kind="error", arm="sweep_%s" % (cfg,), err=repr(ex)[:300])
                m._UPDATE_CONFIG = base; m._update_plans.clear()
        # the decode dispatcher under torch.compile traces the torch path (by design)
        disp = m._make_update_dispatch(lambda h, c, w_, b_=None, a_=None: ref_update(h, c, w_, b_))
        x2 = torch.randn(B + 1, 1, D, device="cuda", dtype=dt).transpose(1, 2)
        s2 = torch.randn(B + 1, D, L, device="cuda", dtype=dt)
        H.compile_check("ours_update_dispatch", lambda a, s: disp(a, s, weight, None, "silu"), [(x, state0.clone()), (x2, s2)])


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

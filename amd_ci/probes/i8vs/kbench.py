#!/usr/bin/env python3
"""Part 1: Studio INT8 attention vs every SageAttention build that runs on this card vs torch SDPA, one process alone.

Extends scripts/h3v_i8default/vid/attn_micro.py (same INT8 entry, same probe inputs, same fp32 reference, same CUDA-event
interleaved-round timing) to a grid: dtype bf16 / fp16 (fp16 only below sm80), head_dim 64 / 128 / 256, heads 24 / 56,
tokens 1k..32k, inputs [B, S, H, D] strided views of one fused [B, S, 3, H, D] buffer (the layout a strided processor
slices out of a fused QKV projection).

Arms (each only where it loads; a build that cannot load is recorded with its reason):
  int8       diffusion_int8_attention.int8_attention_call, after verdict() passed for (dtype, heads, head_dim);
             engagement = STATS["int8"] delta on every correctness call
  sage_hub   diffusion_attention._load_sage_hub_kernel() (what attention_backend="sage" loads when pip has no Sage 2)
  sage2_pip  SageAttention 2.x built from source into EXT/sage2 (sageattn dispatcher, arch default)
  sage1_pip  PyPI sageattention 1.0.6 (Triton) into EXT/sage1, where Sage 2 cannot run (sm75)
  sage3      SageAttention3 (sageattention3_blackwell) into EXT/sage3, sm120 only; [B,H,S,D] contiguous copies inside
             the timed call (sage3) and kernel-only on pre-made contiguous inputs (sage3_hnd)
  flash / cudnn / efficient   torch SDPA pinned with sdpa_kernel; sdpa = unpinned (torch's own pick)
Inputs: "probe" = diffusion_int8_attention.probe_inputs; "sink" = probe plus H3-like attention sinks (first 4 keys of a
third of the heads take ~half the attention mass, their V 20-60x the channel spread in 80% of channels) and two V
outlier channels (+8 sigma offset on every key). Timing on probe inputs only (the kernels' cost does not depend on
values); accuracy on both, vs fp32 attention: cosine, rel-L1, max-abs.
Timing: CUDA events, interleaved rounds, PASSES passes in alternating arm order (fwd, rev, fwd), 7 rounds per pass up to
8k tokens and 4 above; median over all samples; A/A spread = (max - min) / median of the per-pass medians.

usage: kbench.py --studio BACKEND_DIR --ext EXT_DIR --out kb.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
import traceback


def nvml_handle(index: int = 0):
    """(api, handle) for clock / thermal readings: NVML on NVIDIA, amdsmi on ROCm, (None, None) when neither loads."""
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(index)
    except Exception:  # noqa: BLE001
        pass
    try:
        import amdsmi

        amdsmi.amdsmi_init()
        return amdsmi, amdsmi.amdsmi_get_processor_handles()[index]
    except Exception:  # noqa: BLE001
        return None, None


def clocks(nv, h) -> dict:
    if nv is None:
        return {"available": False}
    out = {}
    if getattr(nv, "__name__", "") == "amdsmi":
        for k, f in (("sm_mhz", lambda: nv.amdsmi_get_clock_info(h, nv.AmdSmiClkType.GFX)["clk"]),
                     ("temp_c", lambda: nv.amdsmi_get_temp_metric(h, nv.AmdSmiTemperatureType.EDGE,
                                                                   nv.AmdSmiTemperatureMetric.CURRENT)),
                     ("power_w", lambda: nv.amdsmi_get_power_info(h).get("current_socket_power")),
                     ("procs", lambda: len(nv.amdsmi_get_gpu_process_list(h)))):
            try:
                out[k] = f()
            except Exception:  # noqa: BLE001
                out[k] = None
        return out
    for k, f in (("sm_mhz", lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)),
                 ("mem_mhz", lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM)),
                 ("temp_c", lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)),
                 ("power_w", lambda: round(nv.nvmlDeviceGetPowerUsage(h) / 1000, 1)),
                 ("throttle", lambda: hex(nv.nvmlDeviceGetCurrentClocksThrottleReasons(h))),
                 ("procs", lambda: len(nv.nvmlDeviceGetComputeRunningProcesses(h)))):
        try:
            out[k] = f()
        except Exception:  # noqa: BLE001
            out[k] = None
    return out


def gate(nv, h, max_temp: float, max_wait: float) -> dict:
    """The same thermal / co-tenant gate before every cell: wait until temp <= max_temp and at most one compute
    process (this one) is on the card, up to max_wait seconds. Never skips a cell: records whether it passed."""
    t0 = time.time()
    while True:
        c = clocks(nv, h)
        temp, procs = c.get("temp_c"), c.get("procs")
        ok = (temp is None or temp <= max_temp) and (procs is None or procs <= 1)
        if ok or time.time() - t0 >= max_wait:
            return {"passed": bool(ok), "wait_s": round(time.time() - t0, 1), "temp_c": temp, "procs": procs,
                    "readable": c.get("available", True)}
        time.sleep(2)


def ext_import(ext: str, name: str, module: str):
    """Import ``module`` from EXT/name only (refuses a copy found elsewhere). Returns (module, info)."""
    import torch

    if getattr(torch.version, "hip", None):
        raise RuntimeError("SageAttention ships CUDA-only kernels (no ROCm / HIP build): not built on ROCm")
    p = os.path.join(ext, name)
    skip = os.path.join(ext, name + ".SKIP")
    if os.path.exists(skip):
        raise RuntimeError(open(skip).read().strip()[:300])
    info = {"dir": p}
    bi = os.path.join(p, "BUILD_INFO.json")
    if os.path.exists(bi):
        info.update(json.load(open(bi)))
    if not os.path.isdir(p):
        raise RuntimeError(f"not built ({p} missing)")
    sys.path.insert(0, p)
    try:
        mod = __import__(module)
    finally:
        sys.path.remove(p)
    if not str(getattr(mod, "__file__", "")).startswith(p):
        raise RuntimeError(f"{module} imported from {getattr(mod, '__file__', None)}, not {p}")
    info["file"] = mod.__file__
    return mod, info


def sink_inputs(I8, torch, dev, dt, S, H, D, seed):
    """probe_inputs plus H3-like sinks / V outlier channels (see module doc)."""
    q, k, v = (x.float() for x in I8.probe_inputs(dev, torch.float32, S, H, D, seed = seed))
    g = torch.Generator(device = "cpu").manual_seed(7000 + seed)
    n_sink = 4
    heads = list(range(0, H, 3))  # a third of the heads (H3: up to 17 of 56)
    u = torch.randn((H, D), generator = g).to(dev)
    q = q + 0.8 * u[None, None]  # a shared query direction per head: what lets a key act as a sink
    # bulk log-sum-exp of the logits over the non-sink keys, from 256 sampled queries, per head
    scale = D ** -0.5
    qi = q[0, torch.linspace(0, S - 1, 256).long().to(dev)]  # [256, H, D]
    lse = torch.logsumexp(torch.einsum("qhd,khd->hqk", qi, k[0, n_sink:]) * scale, dim = -1).mean(dim = 1)  # [H]
    target = lse - math.log(n_sink)  # each sink ~ bulk / n_sink -> sinks take about half of the mass
    un = u / u.norm(dim = -1, keepdim = True)
    for h in heads:
        beta = float(target[h]) / (0.8 * float(u[h].norm()) * scale)
        k[0, :n_sink, h] = beta * un[h] + 0.1 * torch.randn((n_sink, D), generator = g).to(dev)
        std = v[0, :, h].std(dim = 0)  # [D]
        mag = 20 + 40 * torch.rand((n_sink, D), generator = g).to(dev)
        sign = torch.where(torch.rand((n_sink, D), generator = g).to(dev) < 0.5, -1.0, 1.0)
        keep = (torch.rand((n_sink, D), generator = g).to(dev) < 0.8).float()
        v[0, :n_sink, h] = v[0, :n_sink, h] * (1 - keep) + keep * sign * mag * std
    chans = torch.randperm(D, generator = g)[:2].to(dev)
    v[..., chans] += 8 * v.std()
    # measured sink mass (fraction of the softmax on the first n_sink keys), on the sink heads
    qs = q[0, torch.linspace(0, S - 1, 64).long().to(dev)][:, heads]
    p = torch.softmax(torch.einsum("qhd,khd->hqk", qs, k[0][:, heads]) * scale, dim = -1)
    mass = float(p[..., :n_sink].sum(-1).mean())
    return tuple(x.to(dt) for x in (q, k, v)), {"n_sink": n_sink, "sink_heads": len(heads), "sink_mass": round(mass, 3)}


def ref_attention(torch, F, q, k, v):
    """fp32 attention on [B, S, H, D] (what diffusion_int8_attention.reference_attention computes), in head chunks so
    no math fallback ever materialises all heads' S x S scores; query-chunked matmuls where no fused fp32 kernel
    takes the shape."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    B, S, H, D = q.shape
    out = torch.empty((B, S, H, D), device = q.device, dtype = torch.float32)
    hc = 8
    for h0 in range(0, H, hc):
        qh, kh, vh = (x[:, :, h0:h0 + hc].float().permute(0, 2, 1, 3) for x in (q, k, v))
        try:
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                o = F.scaled_dot_product_attention(qh, kh, vh)
        except Exception:  # noqa: BLE001
            o = torch.empty_like(qh)
            qc = max(256, (1 << 29) // max(1, S * qh.shape[1] * 4))
            for s0 in range(0, S, qc):
                sc = torch.matmul(qh[:, :, s0:s0 + qc], kh.transpose(-1, -2)) * D ** -0.5
                o[:, :, s0:s0 + qc] = torch.matmul(torch.softmax(sc, dim = -1), vh)
                del sc
        out[:, :, h0:h0 + hc] = o.permute(0, 2, 1, 3)
        del qh, kh, vh, o
    return out


def fused_views(torch, q, k, v):
    """q / k / v as the [B, S, H, D] slices of one fused [B, S, 3, H, D] buffer (non-contiguous over S)."""
    B, S, H, D = q.shape
    buf = torch.empty((B, S, 3, H, D), device = q.device, dtype = q.dtype)
    buf[:, :, 0].copy_(q)
    buf[:, :, 1].copy_(k)
    buf[:, :, 2].copy_(v)
    return buf[:, :, 0], buf[:, :, 1], buf[:, :, 2]


ARMS = ("int8", "sage_hub", "sage2_pip", "sage1_pip", "sage3", "sage3_hnd", "flash", "cudnn", "efficient", "sdpa")


def arm_status(row: dict, builds: dict) -> dict:
    """Every arm of ARMS gets OK / NOT_RUN: reason / ERROR: reason in every cell: nothing is dropped silently."""
    out = {}
    for arm in ARMS:
        bname = "sage3" if arm == "sage3_hnd" else arm
        if arm in row["ms"]:
            out[arm] = "OK"
        elif bname in builds and not builds[bname].get("ok"):
            out[arm] = "NOT_RUN: build cannot load here: " + str(builds[bname].get("why"))[:200]
        elif arm in row["skip"]:
            out[arm] = "NOT_RUN: " + row["skip"][arm]
        else:
            errs = [v for k, v in row["err"].items() if k.split("/")[0] == arm]
            if errs and all("No available kernel" in e or "PassManager" in e for e in errs):
                out[arm] = "NOT_RUN: no kernel for this card / shape: " + errs[0][:150]
            elif errs:
                out[arm] = "ERROR: " + errs[0][:200]
            elif row.get("fatal"):
                out[arm] = "ERROR: cell failed: " + row["fatal"][:150]
            else:
                out[arm] = "NOT_RUN: not timed in this cell"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--studio", required = True)
    ap.add_argument("--ext", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--tokens", default = "1024,4096,8192,16384,32768")
    ap.add_argument("--heads", default = "24,56")
    ap.add_argument("--head-dims", default = "64,128,256")
    ap.add_argument("--dtypes", default = "auto")
    ap.add_argument("--passes", type = int, default = 3)
    ap.add_argument("--device", type = int, default = 0, help = "CUDA / HIP index to sweep (one card per process)")
    ap.add_argument("--gate-temp", type = float, default = 72.0,
                    help = "thermal gate: before every cell wait (up to --gate-wait s) until the card is at or below "
                           "this temperature and no other compute process is on it; recorded per cell")
    ap.add_argument("--gate-wait", type = float, default = 90.0)
    a = ap.parse_args()
    os.environ.setdefault("UNSLOTH_STUDIO_HOME", tempfile.mkdtemp(prefix = "kb_home_"))
    sys.path.insert(0, a.studio)
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from core.inference import diffusion_attention as DA
    from core.inference import diffusion_int8_attention as I8

    torch.cuda.set_device(a.device)
    dev = torch.device("cuda", a.device)
    IDX = a.device
    cap = tuple(torch.cuda.get_device_capability(a.device))
    nv, nh = nvml_handle(a.device)
    info = {"gpu": torch.cuda.get_device_name(a.device), "cap": list(cap), "torch": torch.__version__,
            "cuda": torch.version.cuda, "hip": getattr(torch.version, "hip", None), "device": a.device,
            "studio": a.studio, "gate_temp": a.gate_temp, "gate_wait": a.gate_wait}
    try:
        import importlib.metadata as md

        info["pkgs"] = {}
        for p in ("diffusers", "transformers", "kernels", "triton", "huggingface_hub", "sageattention", "nvidia-cuda-nvrtc"):
            try:
                info["pkgs"][p] = md.version(p)
            except Exception:  # noqa: BLE001
                info["pkgs"][p] = None
    except Exception:  # noqa: BLE001
        pass
    if nv is not None:
        try:
            info["max_clocks"] = {"sm": nv.nvmlDeviceGetMaxClockInfo(nh, nv.NVML_CLOCK_SM),
                                  "mem": nv.nvmlDeviceGetMaxClockInfo(nh, nv.NVML_CLOCK_MEM)}
            info["power_limit_w"] = nv.nvmlDeviceGetPowerManagementLimit(nh) / 1000
        except Exception:  # noqa: BLE001
            pass
    dtypes = ([torch.bfloat16, torch.float16] if cap >= (8, 0) else [torch.float16]) if a.dtypes == "auto" else \
        [getattr(torch, x) for x in a.dtypes.split(",")]

    # ------------------------------------------------------------------ builds
    builds: dict = {}
    fns: dict = {}
    try:
        fn, why = DA._load_sage_hub_kernel()
        if callable(fn):
            mods = {n: getattr(m, "__file__", None) for n, m in sys.modules.items() if n.startswith("_unsloth_sage_hub_")
                    and n.count(".") == 0}
            builds["sage_hub"] = {"ok": True, "modules": mods, "sm89_no_sync_wrapper": hasattr(fn, "__wrapped__")}
            for f in mods.values():
                try:
                    meta = json.load(open(os.path.join(os.path.dirname(f), "metadata.json")))
                    builds["sage_hub"]["metadata"] = {k: meta.get(k) for k in ("id", "version", "backend", "provenance")}
                except Exception:  # noqa: BLE001
                    pass
            fns["sage_hub"] = lambda q, k, v, _f = fn: _f(q, k, v, tensor_layout = "NHD")
        else:
            builds["sage_hub"] = {"ok": False, "why": why}
    except Exception as exc:  # noqa: BLE001
        builds["sage_hub"] = {"ok": False, "why": f"{type(exc).__name__}: {exc}"[:600]}
    sage_counts: dict = {}
    for name, mod_name in (("sage2_pip", "sageattention"), ("sage1_pip", "sageattention")):
        sub = name.split("_")[0]
        try:
            mod, binfo = ext_import(a.ext, sub, mod_name)
            core = getattr(mod, "core", None)
            if core is not None:
                for n in dir(core):
                    f = getattr(core, n)
                    if callable(f) and n.startswith("sageattn_"):
                        def w(*x, __f = f, __n = n, **kw):
                            sage_counts[__n] = sage_counts.get(__n, 0) + 1
                            return __f(*x, **kw)
                        setattr(core, n, w)
            sa = mod.sageattn
            try:  # a build that imports but cannot run on this card is NOT_RUN, and the next build gets its turn
                _q = torch.randn((1, 256, 2, 64), device = dev, dtype = torch.float16)
                sa(_q, _q, _q, tensor_layout = "NHD")
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001
                builds[name] = {"ok": False, **binfo, "why": f"loads but cannot run here: {type(exc).__name__}: {exc}"[:400]}
                sys.modules.pop("sageattention", None)
                for k in [k for k in sys.modules if k.startswith("sageattention.")]:
                    sys.modules.pop(k, None)
                continue
            builds[name] = {"ok": True, **binfo}
            fns[name] = lambda q, k, v, _f = sa: _f(q, k, v, tensor_layout = "NHD")
            if name == "sage2_pip":
                builds["sage1_pip"] = {"ok": False, "why": "not tried: Sage 2 loaded (one sageattention package per "
                                                           "process; Sage 1 is only built where Sage 2 cannot run)"}
            break  # one sageattention package per process
        except Exception as exc:  # noqa: BLE001
            builds[name] = {"ok": False, "why": f"{type(exc).__name__}: {exc}"[:600]}
    try:
        mod, binfo = ext_import(a.ext, "sage3", "sageattn3")
        s3 = mod.sageattn3_blackwell
        builds["sage3"] = {"ok": True, **binfo}
        fns["sage3"] = lambda q, k, v, _f = s3: _f(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
                                                     v.transpose(1, 2).contiguous()).transpose(1, 2)
    except Exception as exc:  # noqa: BLE001
        builds["sage3"] = {"ok": False, "why": f"{type(exc).__name__}: {exc}"[:600]}
    info["builds"] = builds
    print("KB_BUILDS", json.dumps(builds, default = str)[:4000], flush = True)

    def sdpa(backend):
        def run(q, k, v):
            qh, kh, vh = (x.permute(0, 2, 1, 3) for x in (q, k, v))
            if backend is None:
                return F.scaled_dot_product_attention(qh, kh, vh).permute(0, 2, 1, 3)
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(qh, kh, vh).permute(0, 2, 1, 3)
        return run

    base_fns = {"flash": sdpa(SDPBackend.FLASH_ATTENTION), "cudnn": sdpa(SDPBackend.CUDNN_ATTENTION),
                "efficient": sdpa(SDPBackend.EFFICIENT_ATTENTION), "sdpa": sdpa(None)}

    res = {"info": info, "verdicts": {}, "rows": [], "kernels": {}}

    def dump():
        res["int8_stats"] = dict(I8.STATS)
        res["sage_dispatch_counts"] = dict(sage_counts)
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent = 1, default = str)

    def arm_set(dt, H, D, q, k, v):
        """name -> zero-arg callable on these tensors, plus name -> reason for the arms that cannot take the cell."""
        out, skip = {}, {}
        vkey = f"{str(dt)[6:]}_h{H}_d{D}"
        ver = res["verdicts"].get(vkey)
        reason = I8.unsupported_reason(q, k, v)
        if reason:
            skip["int8"] = f"unsupported: {reason}"
        elif not (ver and ver.get("ok")):
            skip["int8"] = f"verdict not ok: {(ver or {}).get('error')}"
        else:
            out["int8"] = lambda: I8.int8_attention_call(q, k, v)
        for name, f in fns.items():
            if D > 128:
                skip[name] = "head_dim > 128 unsupported"
                continue
            out[name] = (lambda _f = f: _f(q, k, v))
        if "sage3" in out:
            qh, kh, vh = (x.transpose(1, 2).contiguous() for x in (q, k, v))
            s3 = fns["sage3"].__defaults__[0]
            out["sage3_hnd"] = lambda: s3(qh, kh, vh)  # [B,H,S,D] out: accuracy transposes it back below
        for name, f in base_fns.items():
            out[name] = (lambda _f = f: _f(q, k, v))
        return out, skip

    def as_bshd(name, o, S):
        if name == "sage3_hnd":
            return o.transpose(1, 2)
        return o

    toks = [int(x) for x in a.tokens.split(",")]
    heads = [int(x) for x in a.heads.split(",")]
    hds = [int(x) for x in a.head_dims.split(",")]
    profiled = set()
    t_start = time.time()
    for dt in dtypes:
        for D in hds:
            for H in heads:
                vkey = f"{str(dt)[6:]}_h{H}_d{D}"
                if D in I8.HEAD_DIMS:
                    t0 = time.time()
                    try:
                        v0 = I8.verdict(dev, dt, H, D, timed = True)
                        res["verdicts"][vkey] = {k2: (x if isinstance(x, (int, float, str, bool, type(None), list, dict))
                                                      else str(x)) for k2, x in v0.items()}
                    except Exception as exc:  # noqa: BLE001
                        res["verdicts"][vkey] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
                    res["verdicts"][vkey]["impl_now"] = I8.implementation(IDX, D)
                    res["verdicts"][vkey]["s"] = round(time.time() - t0, 2)
                    print("KB_VERDICT", vkey, json.dumps(res["verdicts"][vkey], default = str)[:1500], flush = True)
                for S in toks:
                    row = {"dtype": str(dt)[6:], "D": D, "H": H, "S": S, "ms": {}, "acc": {}, "err": {}, "skip": {},
                           "gate": gate(nv, nh, a.gate_temp, a.gate_wait), "clk0": clocks(nv, nh)}
                    try:
                        seed = (S * 31 + H * 7 + D) % 997
                        sets = {}
                        q0, k0, v0_ = I8.probe_inputs(dev, dt, S, H, D, seed = seed)
                        sets["probe"] = fused_views(torch, q0, k0, v0_)
                        del q0, k0, v0_
                        live: dict = {}
                        for inp in ("probe", "sink"):
                            if inp == "sink":
                                (qs, ks, vs), meta = sink_inputs(I8, torch, dev, dt, S, H, D, seed)
                                row["sink_meta"] = meta
                                sets["sink"] = fused_views(torch, qs, ks, vs)
                                del qs, ks, vs
                            q, k, v = sets[inp]
                            arms, skip = arm_set(dt, H, D, q, k, v)
                            row["skip"].update(skip)
                            ref = ref_attention(torch, F, q, k, v)
                            acc = {}
                            for name, fn in list(arms.items()):
                                try:
                                    n0, nn0 = I8.STATS["int8"], I8.STATS["native"]
                                    o = as_bshd(name, fn(), S)
                                    torch.cuda.synchronize()
                                    if name == "int8":
                                        row.setdefault("int8_engaged", {})[inp] = [I8.STATS["int8"] - n0,
                                                                                   I8.STATS["native"] - nn0]
                                        row["int8_long_cfg"] = bool(I8._uses_long(I8.device_config(IDX, D, I8.implementation(IDX, D)), S))
                                        row["int8_long_ready"] = bool(I8._long_ready(IDX, D, I8.implementation(IDX, D), q, k, v))
                                        if I8.STATS["int8"] - n0 != 1:
                                            raise RuntimeError("INT8 call ran native attention (not engaged)")
                                    acc[name] = {kk: (round(x, 7) if kk == "cos" else float(f"{x:.5g}"))
                                                 for kk, x in I8.accuracy(o, ref).items()}
                                    if not math.isfinite(acc[name]["rel_l1"]):
                                        raise RuntimeError("non-finite output")
                                    del o
                                except Exception as exc:  # noqa: BLE001
                                    row["err"][f"{name}/{inp}"] = f"{type(exc).__name__}: {exc}"[:300]
                                    arms.pop(name)
                            row["acc"][inp] = acc
                            del ref
                            if inp == "probe":
                                live = arms
                            torch.cuda.empty_cache()
                        del sets["sink"]
                        # profile kernel names once per dtype at a mid shape
                        pk = (str(dt), D, H)
                        if S == 4096 and D == 128 and pk not in profiled:
                            profiled.add(pk)
                            from torch.profiler import ProfilerActivity, profile
                            for name, fn in live.items():
                                try:
                                    fn()
                                    torch.cuda.synchronize()
                                    with profile(activities = [ProfilerActivity.CUDA]) as prof:
                                        fn()
                                        torch.cuda.synchronize()
                                    ks = sorted(((e.key, e.device_time_total if hasattr(e, "device_time_total") else
                                                  e.cuda_time_total, e.count) for e in prof.key_averages()),
                                                key = lambda x: -x[1])
                                    res["kernels"][f"{str(dt)[6:]}_h{H}_d{D}_s{S}/{name}"] = [
                                        [n[:160], round(t / 1000, 4), c] for n, t, c in ks[:6]]
                                except Exception as exc:  # noqa: BLE001
                                    res["kernels"][f"{str(dt)[6:]}_h{H}_d{D}_s{S}/{name}"] = repr(exc)[:300]
                        # timing
                        for fn in live.values():
                            for _ in range(2):
                                fn()
                        torch.cuda.synchronize()
                        names = list(live)
                        rounds = 7 if S <= 8192 else 4
                        samples = {n: [] for n in names}
                        per_pass = {n: [] for n in names}
                        for p in range(a.passes):
                            order = names if p % 2 == 0 else names[::-1]
                            got = {n: [] for n in names}
                            for _ in range(rounds):
                                for n in order:
                                    e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
                                    e0.record()
                                    live[n]()
                                    e1.record()
                                    e1.synchronize()
                                    got[n].append(e0.elapsed_time(e1))
                            for n in names:
                                samples[n] += got[n]
                                xs = sorted(got[n])
                                per_pass[n].append(xs[len(xs) // 2])
                        for n in names:
                            xs = sorted(samples[n])
                            med = xs[len(xs) // 2]
                            pp = per_pass[n]
                            row["ms"][n] = {"med": round(med, 4), "min": round(xs[0], 4), "max": round(xs[-1], 4),
                                            "pass_med": [round(x, 4) for x in pp],
                                            "aa": round((max(pp) - min(pp)) / med, 4) if med else None}
                        del live, sets
                    except Exception as exc:  # noqa: BLE001
                        row["fatal"] = f"{type(exc).__name__}: {exc}"[:500]
                        row["tb"] = traceback.format_exc()[-1500:]
                    row["clk1"] = clocks(nv, nh)
                    row["status"] = arm_status(row, builds)
                    torch.cuda.empty_cache()
                    res["rows"].append(row)
                    short = {n: m["med"] for n, m in row["ms"].items()}
                    print("KB_ROW", json.dumps({k2: row.get(k2) for k2 in ("dtype", "D", "H", "S", "skip", "err", "fatal")}
                                               | {"ms": short, "acc_probe_cos": {n: x["cos"] for n, x in row["acc"].get("probe", {}).items()},
                                                  "acc_sink_rel": {n: x["rel_l1"] for n, x in row["acc"].get("sink", {}).items()},
                                                  "sm": [row["clk0"].get("sm_mhz"), row["clk1"].get("sm_mhz")],
                                                  "t": round(time.time() - t_start)}, default = str)[:3000], flush = True)
                    dump()
    dump()
    print("KB_DONE", a.out, round(time.time() - t_start), "s", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

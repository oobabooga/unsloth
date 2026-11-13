#!/usr/bin/env python3
"""Probe: INT8 / 16-bit attention kernels on this AMD GPU, timed and checked against an fp64 reference.

Observes only (judged by criteria/int8_attn_amd_timing.py). Every arm takes the strided ``[B, S, H, D]`` views a
diffusers processor holds (q / k / v made ``[B, S, H, D]``) and returns ``[B, S, H, D]``:

  sdpa / sdpa_aa   torch SDPA on the permuted views (what Studio runs on ROCm today); sdpa_aa is the A/A arm
  sdpa_flash       SDPA pinned to FLASH_ATTENTION (AOTriton flash on ROCm)
  sdpa_eff         SDPA pinned to EFFICIENT_ATTENTION (AOTriton mem-efficient)
  ck_int8          comfy-kitchen ``int8_attention`` (HIP ``sage_int8_sdpa``) on heads-first transposed views
  ck_sol           comfy-kitchen ``sol_attn`` (block-sparse, default tau): informational, not dense attention
  sage_tri         sageattention 1.0.6 (pure Triton, INT8 Q.K^T, fp16 P.V)
  tri_half         the checkout's Triton kernel without quantisation (16-bit Q.K^T and P.V)
  tri_i8h          the checkout's Triton kernel, INT8 Q.K^T, 16-bit P.V
  tri_i8i8         the checkout's Triton kernel, INT8 Q.K^T and INT8 P.V
  studio           the checkout's ``unsloth_studio::int8_attention_nhd`` op after its own install path engaged it
                   (env switch forced), i.e. what a Studio load would run; absent where the checkout has no op

Process layout: one sweep process (block configs of the tri_* arms, best per head dim), one accuracy process
(every arm, every shape, q / k gain 1 / 5 / 8, vs fp64 on 3 heads x 256 query rows), then N fresh timing
processes; within each, arms interleaved with a rotating order, an L2 / MALL flush before every timed call,
CUDA events around the call only. The SCLK reading is sampled per round.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

SHAPES = {
    "h3_1024": (1, 1324, 56, 128),
    "h3_2048": (1, 4396, 56, 128),
    "klein_1024": (1, 4608, 24, 128),
    "klein_2048": (1, 16896, 24, 128),
    "ltx_d64": (1, 6144, 32, 64),
    "wan_480p": (1, 8100, 40, 128),
}
# (BM, BN, warps, stages, waves_per_eu; 0 = unset)
SWEEP = [
    (64, 32, 4, 1, 0), (64, 64, 4, 1, 0), (128, 32, 4, 1, 0), (128, 32, 8, 1, 0), (128, 64, 8, 1, 0),
    (64, 16, 4, 1, 0), (128, 32, 8, 2, 0), (256, 32, 8, 1, 0), (64, 32, 4, 1, 2), (64, 32, 4, 1, 4),
    (128, 32, 8, 1, 2),
]
# Triton arm -> extra config of the checkout's kernel (pv, kscale, acc_in_dot); tri_half is the unquantised kernel
TRI_VARIANTS = {
    "tri_i8h_s": {"pv": "half_strided", "kscale": "key"},
    "tri_i8h_skt": {"pv": "half_strided", "kscale": "tile"},
    "tri_i8h_skt_f": {"pv": "half_strided", "kscale": "tile", "acc_in_dot": True},
}
TRI = ("tri_half", *TRI_VARIANTS)
PKGS = ["comfy-kitchen==0.2.41", "sageattention==1.0.6"]


def _write(path: Path, obs: dict) -> None:
    path.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")


def _sclk() -> str | None:
    for card in sorted(Path("/sys/class/drm").glob("card*/device/pp_dpm_sclk")):
        try:
            for line in card.read_text(encoding = "utf-8").splitlines():
                if line.strip().endswith("*"):
                    return line.split(":", 1)[1].strip(" *")
        except Exception:  # noqa: BLE001
            pass
    return None


# ---------------------------------------------------------------------------------------------------- worker side


def _inputs(torch, shape, seed, gain = 1.0):
    B, S, H, D = shape
    g = torch.Generator(device = "cpu").manual_seed(1000 + seed)
    s = (B, S, H, D)
    q = torch.randn(s, generator = g) * 1.6
    k = torch.randn(s, generator = g) * 1.6 + 0.4 * torch.randn((1, 1, H, D), generator = g)
    v = torch.randn(s, generator = g) * (0.5 + torch.rand((1, 1, H, D), generator = g))
    v = v + 0.5 * torch.randn((1, 1, H, D), generator = g)
    q, k = q * gain, k * gain
    return tuple(x.to("cuda", torch.bfloat16) for x in (q, k, v))


def _arms(torch, checkout_mod, op, configs, info):
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def sdpa(q, k, v):
        return F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3),
                                              v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

    def pinned(kind):
        def run(q, k, v):
            with sdpa_kernel(kind):
                return sdpa(q, k, v)
        return run

    def sdpa_contig(q, k, v):
        return F.scaled_dot_product_attention(q.permute(0, 2, 1, 3).contiguous(), k.permute(0, 2, 1, 3).contiguous(),
                                              v.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3)

    def sdpa_kv(q, k, v):
        # Studio's ROCm native path on gfx1151: k / v heads-major copies, q strided
        return F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3).contiguous(),
                                              v.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3)

    arms = {"sdpa": sdpa, "sdpa_aa": sdpa, "sdpa_contig": sdpa_contig, "sdpa_kv": sdpa_kv}
    if info.get("role") == "accuracy":
        arms["sdpa_flash"] = pinned(SDPBackend.FLASH_ATTENTION)
        arms["sdpa_eff"] = pinned(SDPBackend.EFFICIENT_ATTENTION)
    try:
        import comfy_kitchen as ck
        from comfy_kitchen import sage_attention as cks

        info["ck_version"] = getattr(ck, "__version__", None)
        info["ck_file"] = ck.__file__
        info["ck_int8_available"] = bool(cks.is_available(torch.device("cuda", 0)))
        if info["ck_int8_available"]:
            arms["ck_int8"] = lambda q, k, v: ck.int8_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        try:
            info["ck_sol_available"] = bool(ck.sol_attn_is_available(torch.device("cuda", 0)))
        except Exception as e:  # noqa: BLE001
            info["ck_sol_available"] = f"{type(e).__name__}: {e}"[:200]
        if info["ck_sol_available"] is True:
            arms["ck_sol"] = lambda q, k, v: ck.sol_attn(q, k, v)
    except Exception as e:  # noqa: BLE001
        info["ck_error"] = f"{type(e).__name__}: {e}"[:400]
    try:
        import sageattention

        info["sage_file"] = sageattention.__file__
        arms["sage_tri"] = lambda q, k, v: sageattention.sageattn(q, k, v, tensor_layout = "NHD", is_causal = False)
    except Exception as e:  # noqa: BLE001
        info["sage_error"] = f"{type(e).__name__}: {e}"[:400]
    aiter_root = os.environ.get("AMD_I8A_AITER")
    if aiter_root:
        os.environ["AITER_TRITON_ONLY"] = "1"
        sys.path.insert(0, aiter_root)
        sys.path.insert(0, os.path.join(aiter_root, "aiter", "jit", "utils"))

        def first(x):
            return x[0] if isinstance(x, (tuple, list)) else x
        _aiter_compat(info)
        try:
            from aiter.ops.triton.attention.mha import flash_attn_func

            arms["aiter_fa"] = lambda q, k, v: first(flash_attn_func(q, k, v, causal = False))
        except Exception as e:  # noqa: BLE001
            info["aiter_fa_error"] = f"{type(e).__name__}: {e}"[:400]
        try:
            from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func

            arms["aiter_sage"] = lambda q, k, v: first(fav3_sage_wrapper_func(q, k, v, layout = "bshd"))
            arms["aiter_sage_rot"] = lambda q, k, v: first(fav3_sage_wrapper_func(q, k, v, layout = "bshd",
                                                                                  hadamard_rotation = True))
        except Exception as e:  # noqa: BLE001
            info["aiter_sage_error"] = f"{type(e).__name__}: {e}"[:400]
    if checkout_mod is not None and hasattr(checkout_mod, "PV_MODES"):
        shape_of = info.setdefault("_shape", {})

        def cfg(name):
            return configs.get(f"{name}/{shape_of.get('name')}") or configs.get(f"{name}/{shape_of.get('d')}")

        arms["tri_half"] = lambda q, k, v: checkout_mod.half_attention(q, k, v, cfg("tri_half"))
        for name, extra in TRI_VARIANTS.items():
            def run(q, k, v, name = name, extra = extra):
                return checkout_mod.int8_attention(q, k, v, dict(cfg(name), **extra))
            arms[name] = run
    if op is not None:
        arms["studio"] = op
    return arms


def _aiter_compat(info):
    """AITER main imports gluon names newer than this Triton has and binds ``aiter.dtypes`` only off the Triton-only
    path: stub the missing gluon names (gfx950-only kernels, never launched here) and bind dtypes."""
    import importlib

    stubbed = []
    try:
        import aiter
        from aiter.utility import dtypes

        aiter.dtypes = dtypes
    except Exception as e:  # noqa: BLE001
        info["aiter_compat_error"] = repr(e)[:300]
    for _ in range(20):
        try:
            importlib.import_module("aiter.ops.triton.attention.mha")
            break
        except ImportError as e:
            msg = str(e)
            if "cannot import name '" not in msg or "triton" not in msg:
                break
            name = msg.split("cannot import name '")[1].split("'")[0]
            mod = msg.split("from '")[1].split("'")[0]
            try:
                setattr(importlib.import_module(mod), name, lambda *a, **k: None)
                stubbed.append(f"{mod}.{name}")
            except Exception:  # noqa: BLE001
                break
        except Exception:  # noqa: BLE001
            break
    info["aiter_stubbed"] = stubbed


def _setup(args, info):
    import torch

    props = torch.cuda.get_device_properties(0)
    info.update(arch = (getattr(props, "gcnArchName", "") or "").split(":")[0], device = torch.cuda.get_device_name(0),
                torch = torch.__version__, hip = getattr(torch.version, "hip", None),
                cus = props.multi_processor_count)
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception as e:  # noqa: BLE001
        info["triton_error"] = repr(e)[:200]
    sys.path.insert(0, str(Path(args.checkout) / "studio" / "backend"))
    mod = None
    try:
        from core.inference import diffusion_int8_attention_triton as mod  # noqa: F811
    except Exception as e:  # noqa: BLE001
        info["triton_port_error"] = f"{type(e).__name__}: {e}"[:300]
    op = None
    try:
        os.environ["UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL"] = "1"
        from core.inference import diffusion_int8_attention as dia

        dia.int8_attention_op()
        rocm = getattr(dia, "rocm_status", None)
        found = None
        if hasattr(dia, "verdict"):
            found = dia.verdict(torch.device("cuda", 0), torch.bfloat16, 24, 128, None, "native", timed = False)
            dia.verdict(torch.device("cuda", 0), torch.bfloat16, 32, 64, None, "native", timed = False)
        info["studio_verdict"] = {k: v for k, v in (found or {}).items() if k != "ms"} if found else None
        info["studio_rocm_status"] = rocm() if callable(rocm) else None
        op = torch.ops.unsloth_studio.int8_attention_nhd
        info["studio_stats_attr"] = "STATS"
    except Exception as e:  # noqa: BLE001
        info["studio_error"] = f"{type(e).__name__}: {e}"[:300]
    return torch, mod, op


def _ref64(torch, q, k, v, heads, rows):
    qq = q[0, rows][:, heads].permute(1, 0, 2).double()
    kk = k[0][:, heads].permute(1, 0, 2).double()
    vv = v[0][:, heads].permute(1, 0, 2).double()
    s = (qq @ kk.transpose(1, 2)) / q.shape[-1] ** 0.5
    return torch.softmax(s, -1) @ vv


def _metrics(torch, o, heads, rows, r):
    import torch.nn.functional as F

    nonfinite = int((~torch.isfinite(o)).sum())
    g = o[0, rows][:, heads].permute(1, 0, 2).double()
    g = torch.nan_to_num(g, nan = 0.0, posinf = 0.0, neginf = 0.0)
    d = (g - r).abs()
    return {"rel_l1": float(d.mean() / r.abs().mean()), "cos": float(F.cosine_similarity(g.flatten(), r.flatten(), 0)),
            "max_abs": float(d.max()), "nonfinite": nonfinite}


def _kernels(torch, fn):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities = [ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names: dict = {}
    for ev in prof.key_averages():
        dt = getattr(ev, "device_time_total", None) or getattr(ev, "cuda_time_total", 0)
        if dt and dt > 0:
            names[ev.key] = names.get(ev.key, 0.0) + float(dt)
    return [[k[:100], round(v, 1)] for k, v in sorted(names.items(), key = lambda kv: -kv[1])[:5]]


def _time_arms(torch, arms, q, k, v, reps, budget_s, flush):
    times = {n: [] for n in arms}
    clocks = []
    names = list(arms)
    t0 = time.time()
    rep = 0
    while True:
        order = names[rep % len(names):] + names[:rep % len(names)]
        clocks.append(_sclk())
        for n in order:
            flush.zero_()
            s, e = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            torch.cuda.synchronize()
            s.record()
            out = arms[n](q, k, v)
            e.record()
            torch.cuda.synchronize()
            times[n].append(round(s.elapsed_time(e), 4))
            del out
        rep += 1
        if rep >= reps or (rep >= 3 and time.time() - t0 > budget_s):
            break
    return times, clocks


def worker(args) -> int:
    info: dict = {"role": args.role, "pid": os.getpid()}
    out = Path(args.out)
    try:
        torch, mod, op = _setup(args, info)
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = f"{type(e).__name__}: {e}"
        _write(out, info)
        return 0
    configs = json.loads(args.configs) if args.configs else {}
    info["configs"] = configs
    flush = torch.empty(256 * 1024 * 1024, dtype = torch.uint8, device = "cuda")
    shapes = [s for s in args.shapes.split(",") if s]
    info["cells"] = {}
    with torch.no_grad():
        if args.role == "sweep":
            if mod is None:
                info["skipped"] = "no Triton port in this checkout"
                _write(out, info)
                return 0
            for sname in shapes:
                shape = SHAPES[sname]
                q, k, v = _inputs(torch, shape, seed = shape[1])
                cell: dict = {}
                for arm in TRI:
                    for bm, bn, w, st, wpe in SWEEP:
                        c = {"bm": bm, "bn": bn, "warps": w, "stages": st, "waves_per_eu": wpe}
                        key = f"{arm} {bm}x{bn} w{w} s{st} e{wpe}"
                        try:
                            if arm == "tri_half":
                                def fn():
                                    return mod.half_attention(q, k, v, c)
                            else:
                                extra = TRI_VARIANTS[arm]

                                def fn():
                                    return mod.int8_attention(q, k, v, dict(c, **extra))
                            fn()
                            torch.cuda.synchronize()
                            ts, _ = _time_arms(torch, {"x": lambda q_, k_, v_: fn()}, q, k, v, 5, 30, flush)
                            cell[key] = {"cfg": c, "arm": arm, "median_ms": statistics.median(ts["x"])}
                        except Exception as e:  # noqa: BLE001
                            cell[key] = {"cfg": c, "arm": arm, "error": f"{type(e).__name__}: {e}"[:300]}
                        info["cells"][sname] = cell
                        _write(out, info)
                print(f"sweep {sname} done", flush = True)
            _write(out, info)
            return 0
        arms = _arms(torch, mod, op, configs, info)
        info["arms"] = list(arms)
        # settle the clocks: ~3 s of attention before the first measurement (SCLK ramps from ~900 MHz)
        q, k, v = _inputs(torch, SHAPES["h3_2048"], seed = 1)
        t_end = time.time() + 3.0
        while time.time() < t_end:
            arms["sdpa"](q, k, v)
            torch.cuda.synchronize()
        info["sclk_after_warmup"] = _sclk()
        del q, k, v
        for sname in shapes:
            shape = SHAPES[sname]
            B, S, H, D = shape
            info.setdefault("_shape", {}).update(name = sname, d = D)
            cell = {"shape": list(shape)}
            try:
                if args.role == "accuracy":
                    heads = sorted({0, H // 2, H - 1})
                    rows = torch.arange(0, S, max(1, S // 256), device = "cuda")[:256]
                    acc: dict = {}
                    for gain in (1.0, 5.0, 8.0):
                        q, k, v = _inputs(torch, shape, seed = S, gain = gain)
                        r = _ref64(torch, q, k, v, heads, rows)
                        for n, fn in arms.items():
                            if n == "sdpa_aa":
                                continue
                            try:
                                acc.setdefault(n, {})[f"g{gain:g}"] = _metrics(torch, fn(q, k, v), heads, rows, r)
                            except Exception as e:  # noqa: BLE001
                                acc.setdefault(n, {})[f"g{gain:g}"] = {"error": f"{type(e).__name__}: {e}"[:300]}
                        del q, k, v, r
                        torch.cuda.empty_cache()
                    cell["accuracy"] = acc
                    q, k, v = _inputs(torch, shape, seed = S)
                    kern = {}
                    for n, fn in arms.items():
                        try:
                            kern[n] = _kernels(torch, lambda: fn(q, k, v))
                        except Exception as e:  # noqa: BLE001
                            kern[n] = [[f"error {type(e).__name__}: {e}"[:200], 0]]
                    cell["kernels"] = kern
                else:
                    q, k, v = _inputs(torch, shape, seed = S)
                    live = {}
                    for n, fn in arms.items():
                        try:
                            fn(q, k, v)
                            fn(q, k, v)
                            torch.cuda.synchronize()
                            live[n] = fn
                        except Exception as e:  # noqa: BLE001
                            cell.setdefault("errors", {})[n] = f"{type(e).__name__}: {e}"[:300]
                    times, clocks = _time_arms(torch, live, q, k, v, args.reps, args.cell_budget_s, flush)
                    cell["times_ms"] = times
                    cell["median_ms"] = {n: statistics.median(t) for n, t in times.items()}
                    cell["sclk"] = clocks
            except Exception as e:  # noqa: BLE001
                cell["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}"
            torch.cuda.empty_cache()
            info["cells"][sname] = cell
            _write(out, info)
            print(f"{args.role} {sname}: {cell.get('median_ms') or 'acc'} {cell.get('error', '')[:200]}", flush = True)
    if op is not None:
        try:
            from core.inference import diffusion_int8_attention as dia

            info["studio_calls"] = dict(dia.STATS)
            if hasattr(dia, "STATS_ROCM"):
                info["studio_calls_rocm"] = dict(dia.STATS_ROCM)
        except Exception as e:  # noqa: BLE001
            info["studio_calls_error"] = repr(e)[:200]
    _write(out, info)
    return 0


# ---------------------------------------------------------------------------------------------------- driver side


def _install(pkg_dir: Path, log: list) -> str | None:
    if (pkg_dir / ".done").is_file():
        return None
    pkg_dir.mkdir(parents = True, exist_ok = True)
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(pkg_dir),
           "--index-url", "https://pypi.org/simple", *PKGS]
    r = subprocess.run(cmd, capture_output = True, text = True)
    log.append({"cmd": " ".join(cmd), "rc": r.returncode, "tail": (r.stdout + r.stderr)[-1500:]})
    if r.returncode != 0:
        # uv venvs may carry no pip
        cmd = ["uv", "pip", "install", "-q", "--no-deps", "--target", str(pkg_dir), "--python", sys.executable,
               "--index-url", "https://pypi.org/simple", *PKGS]
        r = subprocess.run(cmd, capture_output = True, text = True)
        log.append({"cmd": " ".join(cmd), "rc": r.returncode, "tail": (r.stdout + r.stderr)[-1500:]})
        if r.returncode != 0:
            return "install failed"
    (pkg_dir / ".done").write_text("ok", encoding = "utf-8")
    return None


def _run_worker(args, role, out: Path, env, configs, timeout, log) -> dict:
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--worker", "--role", role, "--checkout", args.checkout,
           "--out", str(out), "--shapes", args.sweep_shapes if role == "sweep" else args.shapes,
           "--configs", json.dumps(configs), "--reps", str(args.reps), "--cell-budget-s", str(args.cell_budget_s),
           "--state", args.state]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, env = env, capture_output = True, text = True, timeout = timeout)
        rc, tail = r.returncode, (r.stdout + r.stderr)[-3000:]
    except subprocess.TimeoutExpired as e:
        rc, tail = "timeout", str(e.stdout or "")[-2000:]
    log.append({"role": role, "rc": rc, "s": round(time.time() - t0, 1), "tail": tail})
    print(f"[{role}] rc={rc} {time.time() - t0:.0f}s\n{tail[-1500:]}", flush = True)
    try:
        return json.loads(out.read_text(encoding = "utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"read_error": repr(e)}


def _overlay(args, obs) -> None:
    """Head only: copy ``--overlay`` (a tree mirroring the repo) over the checkout, for hosts where states.py cannot
    apply a patch (Windows). The base stays as checked out."""
    if not args.overlay or args.state != "head":
        return
    import shutil

    src = Path(args.overlay)
    copied = []
    for f in sorted(src.rglob("*")):
        if f.is_file():
            dst = Path(args.checkout) / f.relative_to(src)
            dst.parent.mkdir(parents = True, exist_ok = True)
            shutil.copyfile(f, dst)
            copied.append(str(f.relative_to(src)))
    obs["overlay"] = copied


def driver(args) -> int:
    out = Path(args.out)
    work = Path(os.environ.get("RUNNER_TEMP") or out.parent) / f"i8a_{args.state}"
    work.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "log": [], "shapes": args.shapes, "processes": args.processes}
    _overlay(args, obs)
    pkg_dir = Path(os.environ.get("RUNNER_TEMP") or out.parent) / "i8a_pkgs"
    obs["install_error"] = _install(pkg_dir, obs["log"])
    env = dict(os.environ)
    env.pop("CI", None)  # CI=true turns torch's custom-op aliasing warning into an error
    env["PYTHONPATH"] = os.pathsep.join([str(pkg_dir), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    env["TRITON_CACHE_DIR"] = str(work / "triton_cache")
    aiter = Path(os.environ.get("RUNNER_TEMP") or out.parent) / "i8a_aiter"
    if not (aiter / "aiter").is_dir():
        r = subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/ROCm/aiter", str(aiter)],
                           capture_output = True, text = True)
        obs["log"].append({"cmd": "git clone aiter", "rc": r.returncode, "tail": (r.stdout + r.stderr)[-800:]})
    if (aiter / "aiter").is_dir():
        env["AMD_I8A_AITER"] = str(aiter)
        r = subprocess.run(["git", "-C", str(aiter), "rev-parse", "HEAD"], capture_output = True, text = True)
        obs["aiter_commit"] = r.stdout.strip()
    _write(out, obs)

    sweep = _run_worker(args, "sweep", work / "sweep.json", env, {}, args.sweep_timeout, obs["log"])
    obs["sweep"] = sweep
    best: dict = {}
    configs = {}
    for sname, cell in (sweep.get("cells") or {}).items():
        d = SHAPES[sname][3]
        per_shape: dict = {}
        for r in cell.values():
            if "median_ms" not in r:
                continue
            # per head dim: summed relative to the shape's best, so every shape weighs the same
            per_shape.setdefault(r["arm"], {})[json.dumps(r["cfg"], sort_keys = True)] = r["median_ms"]
        for arm, by_cfg in per_shape.items():
            top = min(by_cfg, key = by_cfg.get)
            configs[f"{arm}/{sname}"] = json.loads(top)
            for c, ms in by_cfg.items():
                best.setdefault(f"{arm}/{d}", {}).setdefault(c, []).append(ms / by_cfg[top])
    for key, by_cfg in best.items():
        n = max(len(v) for v in by_cfg.values())
        full = {c: sum(v) for c, v in by_cfg.items() if len(v) == n}
        configs[key] = json.loads(min(full, key = full.get))
    obs["configs"] = configs
    _write(out, obs)

    obs["accuracy"] = _run_worker(args, "accuracy", work / "accuracy.json", env, configs, args.acc_timeout, obs["log"])
    _write(out, obs)
    obs["timing"] = []
    for i in range(args.processes):
        obs["timing"].append(_run_worker(args, "timing", work / f"timing_{i}.json", env, configs, args.timing_timeout,
                                         obs["log"]))
        _write(out, obs)
    # per arm per shape: median of each process, then median / spread across processes
    summary: dict = {}
    for sname in args.shapes.split(","):
        per = {}
        for t in obs["timing"]:
            for n, m in ((t.get("cells") or {}).get(sname, {}).get("median_ms") or {}).items():
                per.setdefault(n, []).append(m)
        summary[sname] = {n: {"per_process": v, "median": statistics.median(v),
                              "spread": (max(v) - min(v)) / statistics.median(v) if v else None}
                          for n, v in per.items()}
    obs["summary"] = summary
    _write(out, obs)
    obs["rocm_tests"] = _run_tests(args, env, work)
    _write(out, obs)
    return 0


def _run_tests(args, env, work: Path) -> dict:
    """The checkout's ROCm INT8 attention tests on this GPU (absent on a checkout without them)."""
    backend = Path(args.checkout) / "studio" / "backend"
    test = backend / "tests" / "test_diffusion_int8_attention_rocm.py"
    if not test.is_file():
        return {"present": False}
    junit = work / "rocm_tests.xml"
    cmd = [sys.executable, "-m", "pytest", "-q", "-rA", "-o", "addopts=", "-p", "no:cacheprovider",
           f"--junitxml={junit}", str(test.relative_to(backend))]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd = backend, env = env, capture_output = True, text = True, timeout = 1800)
        rc, tail = r.returncode, (r.stdout + r.stderr)[-6000:]
    except subprocess.TimeoutExpired as e:
        rc, tail = "timeout", str(e.stdout or "")[-3000:]
    res: dict = {"present": True, "rc": rc, "s": round(time.time() - t0, 1), "tail": tail, "cases": {}}
    try:
        import xml.etree.ElementTree as ET

        for case in ET.parse(junit).getroot().iter("testcase"):
            name = case.get("name")
            outcome = "passed"
            for child in case:
                if child.tag in ("failure", "error"):
                    outcome = "failed"
                elif child.tag == "skipped":
                    outcome = "skipped"
            res["cases"][name] = outcome
    except Exception as e:  # noqa: BLE001
        res["junit_error"] = repr(e)[:300]
    print(f"[rocm tests] rc={rc}\n{tail[-2500:]}", flush = True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--worker", action = "store_true")
    ap.add_argument("--overlay", default = "", help = "head only: a tree copied over the checkout first")
    ap.add_argument("--role", default = "timing")
    ap.add_argument("--shapes", default = ",".join(SHAPES))
    ap.add_argument("--sweep-shapes", default = ",".join(SHAPES))
    ap.add_argument("--configs", default = "")
    ap.add_argument("--reps", type = int, default = 10)
    ap.add_argument("--processes", type = int, default = 3)
    ap.add_argument("--cell-budget-s", type = float, default = 90.0)
    ap.add_argument("--sweep-timeout", type = int, default = 2700)
    ap.add_argument("--acc-timeout", type = int, default = 1200)
    ap.add_argument("--timing-timeout", type = int, default = 1200)
    args = ap.parse_args()
    return worker(args) if args.worker else driver(args)


if __name__ == "__main__":
    sys.exit(main())

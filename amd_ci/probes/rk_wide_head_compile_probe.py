#!/usr/bin/env python3
"""Probe: torch.compile through Studio's Windows ROCm wide-head SDPA guard (studio/backend/core/_rocm_windows_sdpa.py),
installed the way Studio's workers install it: Studio first, then unsloth's own guard (import_fixes, extracted by AST
from the same checkout) which sees the shared marker and keeps Studio's. Observes only (judged by
criteria/rk_wide_head_compile_win.py).

Per state, in a child process (a native crash is an exit code, not a lost probe):
  setup     Studio guard installed? from this checkout? unsloth kept it?
  eager     head-512 bf16 causal attention through F.scaled_dot_product_attention: aten op that ran (profiler),
            error vs a float32 math reference
  compile   for backend in eager / aot_eager / inductor: fullgraph=True compile (ok or error text), output vs the
            guarded eager output, fused SDPA ops in the compiled call, SDPA backend flags restored afterwards
  explain   torch._dynamo.explain: graph count, graph break count, break reasons
  default   fullgraph=False aot_eager compile: output vs eager
  control   a plain function compiled with inductor, so "inductor unusable on this box" is not read as the defect
JSON to --out only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

FUSED_OPS = (
    "aten::_scaled_dot_product_flash_attention",
    "aten::_scaled_dot_product_efficient_attention",
    "aten::_scaled_dot_product_cudnn_attention",
    "aten::_scaled_dot_product_fused_attention_overrideable",
    "aten::_scaled_dot_product_flash_attention_for_cpu",
)
SDPA_OPS = FUSED_OPS + ("aten::_scaled_dot_product_attention_math",)


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent = 2, default = str), encoding = "utf-8")


def _err(e: BaseException, n: int = 900) -> str:
    return f"{type(e).__name__}: {str(e)[:n]}"


def _ops(prof, names) -> list:
    return sorted({ev.name.split("::")[-1] for ev in prof.events() if ev.name in names})


def child(args) -> int:
    res: dict = {"state": args.state}
    try:
        backend_dir = str(Path(args.checkout) / "studio" / "backend")
        sys.path.insert(0, backend_dir)
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, _cur_sdpa_kernel_backends, sdpa_kernel
        from torch.profiler import ProfilerActivity, profile

        res.update(torch = torch.__version__, hip = getattr(torch.version, "hip", None), platform = sys.platform,
                   python = sys.version.split()[0], cuda_available = torch.cuda.is_available())
        fake = os.environ.get("RK_PROBE_FAKE_WINDOWS_ROCM") == "1"  # CPU smoke test of the harness only
        dev = "cpu" if fake else "cuda"
        res["device_type"] = dev
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            res["arch"] = getattr(p, "gcnArchName", None)
            res["device"] = p.name

        import core._rocm_windows_sdpa as G
        res["studio_guard_file"] = G.__file__
        res["studio_guard_from_checkout"] = os.path.normcase(os.path.abspath(G.__file__)).startswith(
            os.path.normcase(os.path.abspath(args.checkout)))
        res["studio_has_allow_in_graph_fn"] = hasattr(G, "_math_only")
        if fake:
            import core._torchao_stub as stub
            stub._is_windows_rocm = lambda: True
            G._wide_cuda_head = lambda q: q.shape[-1] > G.WIDEST_FUSED_HEAD_DIM
            res["faked_windows_rocm"] = True
        stock = F.scaled_dot_product_attention
        res["studio_installed"] = bool(G.install_windows_rocm_wide_head_sdpa_guard())
        studio_fn = F.scaled_dot_product_attention
        res["studio_wrapped"] = studio_fn is not stock and bool(getattr(studio_fn, G._GUARD_ATTR, False))

        # unsloth's guard after Studio's, as `import unsloth` in a Studio worker does.
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from wide_head_sdpa_torch_probe import install_guard

            if fake:
                os.environ["WHS_PROBE_FAKE_WINDOWS_ROCM"] = "1"
            res["unsloth_guard"] = install_guard(args.checkout)
            res["unsloth_kept_studio"] = F.scaled_dot_product_attention is studio_fn
        except Exception as e:  # noqa: BLE001
            res["unsloth_guard_error"] = _err(e)
            res["unsloth_kept_studio"] = None

        g = torch.Generator(device = "cpu").manual_seed(512)
        q, k, v = (torch.randn(1, 8, 256, 512, generator = g).to(dev, torch.bfloat16) for _ in range(3))
        sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)

        def attn(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal = True)

        with sdpa_kernel([SDPBackend.MATH]):
            ref = torch._C._nn.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal = True)
        before = _cur_sdpa_kernel_backends()
        res["backends_before"] = sorted(str(b) for b in before)
        with profile(activities = [ProfilerActivity.CPU]) as prof:
            want = attn(q, k, v)
            sync()
        rel = lambda o, r: round(float((o.float() - r.float()).norm() / r.float().norm()), 6)
        res["eager"] = {"ops": _ops(prof, SDPA_OPS), "rel_l2_vs_fp32_math": rel(want, ref),
                        "finite": bool(torch.isfinite(want).all())}

        comp = {}
        for backend in ("eager", "aot_eager", "inductor"):
            row: dict = {}
            torch._dynamo.reset()
            t0 = time.time()
            try:
                fn = torch.compile(attn, backend = backend, fullgraph = True)
                got = fn(q, k, v)
                sync()
                with profile(activities = [ProfilerActivity.CPU]) as prof:
                    got = fn(q, k, v)
                    sync()
                row.update(ok = True, rel_l2_vs_eager = rel(got, want), rel_l2_vs_fp32_math = rel(got, ref),
                           finite = bool(torch.isfinite(got).all()), fused_ops = _ops(prof, FUSED_OPS),
                           sdpa_ops = _ops(prof, SDPA_OPS))
            except Exception as e:  # noqa: BLE001
                row.update(ok = False, error = _err(e), traceback_tail = traceback.format_exc()[-1500:])
            row["seconds"] = round(time.time() - t0, 1)
            row["backends_restored"] = set(_cur_sdpa_kernel_backends()) == set(before)
            row["held_after"] = list(G._HELD) if hasattr(G, "_HELD") else None
            comp[backend] = row
        res["fullgraph"] = comp

        torch._dynamo.reset()
        try:
            ex = torch._dynamo.explain(attn)(q, k, v)
            res["explain"] = {"graph_count": ex.graph_count, "graph_break_count": ex.graph_break_count,
                              "break_reasons": [str(getattr(r, "reason", r))[:400] for r in ex.break_reasons][:8]}
        except Exception as e:  # noqa: BLE001
            res["explain"] = {"error": _err(e)}

        torch._dynamo.reset()
        try:
            got = torch.compile(attn, backend = "aot_eager")(q, k, v)
            sync()
            res["default_compile"] = {"ok": True, "rel_l2_vs_eager": rel(got, want),
                                      "finite": bool(torch.isfinite(got).all())}
        except Exception as e:  # noqa: BLE001
            res["default_compile"] = {"ok": False, "error": _err(e)}
        res["backends_restored_end"] = set(_cur_sdpa_kernel_backends()) == set(before)
        res["held_end"] = list(G._HELD) if hasattr(G, "_HELD") else None

        torch._dynamo.reset()
        try:
            x = torch.randn(64, 64, device = dev)
            y = torch.compile(lambda t: t * 2 + 1, backend = "inductor", fullgraph = True)(x)
            sync()
            res["inductor_control"] = {"ok": bool(torch.allclose(y, x * 2 + 1))}
        except Exception as e:  # noqa: BLE001
            res["inductor_control"] = {"ok": False, "error": _err(e, 500)}
    except Exception as e:  # noqa: BLE001
        res["error"] = _err(e)
        res["traceback"] = traceback.format_exc()[-3000:]
    _write(args.out, res)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--child", action = "store_true", help = argparse.SUPPRESS)
    args = ap.parse_args()
    args.checkout = str(Path(args.checkout).resolve())
    if args.child:
        return child(args)
    out = args.out.with_name(f"{args.out.stem}_child.json")
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        rc = subprocess.run([sys.executable, __file__, "--state", args.state, "--checkout", args.checkout,
                             "--out", str(out), "--child"], env = env, timeout = 1800).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    got = json.loads(out.read_text(encoding = "utf-8")) if out.is_file() else {"missing_output": True}
    got["rc"], got["seconds"] = rc, round(time.time() - t0, 1)
    _write(args.out, got)
    return 0


if __name__ == "__main__":
    sys.exit(main())

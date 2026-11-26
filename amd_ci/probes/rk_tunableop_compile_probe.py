#!/usr/bin/env python3
"""Probe: Studio's ROCm TunableOp tune hooks (studio/backend/core/inference/diffusion_rocm_tunableop.py) under a
whole-module compile, the way diffusion_speed.py compiles a U-Net (`unet.compile(fullgraph=..., dynamic=False)`).
Observes only (judged by criteria/rk_tunableop_compile.py).

Per state, in a child process with UNSLOTH_DIFFUSION_ROCM_TUNABLEOP=tune, no PYTORCH_TUNABLEOP_* in the environment,
and the compile cache (so the tuning CSV) under the probe's work dir:
  install    install_for_pipe on a small bf16 GEMM "U-Net" on the GPU: eligible? hooks registered? tune mode?
  compile    for backend in eager / aot_eager / inductor: unet.compile(fullgraph=True, dynamic=False), three calls,
             output vs the uncompiled module, TunableOp state afterwards (depth, tuning flag)
  default    unet.compile(fullgraph=False) with aot_eager: output vs eager, and Dynamo's graph-break counters
             (break count and reasons) for that module call, hooks included
  eager_tune after dropping the compiled call, two eager calls at a fresh steady shape: TunableOp results before /
             after, tuning CSV written (rows), hook depth back to 0, tuning switched back off
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
import types
from pathlib import Path


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent = 2, default = str), encoding = "utf-8")


def _err(e: BaseException, n: int = 900) -> str:
    return f"{type(e).__name__}: {str(e)[:n]}"


def child(args) -> int:
    res: dict = {"state": args.state}
    try:
        sys.path.insert(0, str(Path(args.checkout) / "studio" / "backend"))
        import torch
        from torch import nn

        res.update(torch = torch.__version__, hip = getattr(torch.version, "hip", None), platform = sys.platform,
                   cuda_available = torch.cuda.is_available())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            res["arch"] = getattr(p, "gcnArchName", None)
            res["device"] = p.name
        from core.inference import diffusion_rocm_tunableop as rt

        res["module_file"] = rt.__file__
        res["module_from_checkout"] = os.path.abspath(rt.__file__).startswith(os.path.abspath(args.checkout))
        res["head_has_compile_guard"] = "is_compiling()" in Path(rt.__file__).read_text(encoding = "utf-8")
        fake = os.environ.get("RK_PROBE_FAKE_ROCM") == "1"  # CPU smoke test of the harness only
        dev = "cpu" if fake else "cuda"
        res["device_type"] = dev
        if fake:
            rt.eligible = lambda device = None: (True, "gfx1151")
            res["faked_rocm"] = True
        res["eligible"] = list(rt.eligible())
        tunable = torch.cuda.tunable

        class UNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(320, 1280)
                self.b = nn.Linear(1280, 320)

            def forward(self, x):
                return self.b(torch.nn.functional.gelu(self.a(x))) + x

        torch.manual_seed(0)
        unet = UNet().to(dev, torch.bfloat16).eval()
        pipe = types.SimpleNamespace(unet = unet, vae = nn.Linear(4, 4))
        res["installed"] = bool(rt.install_for_pipe(pipe, None if fake else torch.device("cuda", 0)))
        res["hooks"] = [len(unet._forward_pre_hooks), len(unet._forward_hooks)]
        res["state_after_install"] = {k: rt._STATE.get(k) for k in ("active", "tune", "file", "written", "depth")}
        res["tunable_enabled"] = bool(tunable.is_enabled())
        sync = (lambda: None) if fake else torch.cuda.synchronize

        def n_results():
            try:
                return len(tunable.get_results() or ())
            except Exception:  # noqa: BLE001
                return None

        rel = lambda o, r: round(float((o.float() - r.float()).norm() / r.float().norm()), 6)
        x = torch.randn(2, 77, 320, device = dev, dtype = torch.bfloat16)
        with torch.no_grad():
            want = unet._call_impl(x)  # eager, hooks included (tuning off: first call at this shape)
        sync()
        comp = {}
        for backend in ("eager", "aot_eager", "inductor"):
            row: dict = {}
            torch._dynamo.reset()
            unet._compiled_call_impl = None
            t0 = time.time()
            try:
                unet.compile(fullgraph = True, dynamic = False, backend = backend)
                with torch.no_grad():
                    for _ in range(3):
                        got = unet(x)
                sync()
                row.update(ok = True, rel_l2_vs_eager = rel(got, want), finite = bool(torch.isfinite(got).all()))
            except Exception as e:  # noqa: BLE001
                row.update(ok = False, error = _err(e), traceback_tail = traceback.format_exc()[-1500:])
            row["seconds"] = round(time.time() - t0, 1)
            row["depth_after"] = rt._STATE.get("depth")
            row["tuning_on_after"] = bool(tunable.tuning_is_enabled())
            comp[backend] = row
            # a failed trace can leave the hooks mid-call: reset so the next backend starts clean
            rt._STATE["depth"] = 0
            rt._STATE.pop("tuning", None)
            tunable.tuning_enable(False)
        res["fullgraph"] = comp

        # Graph breaks of the module call as Module.compile runs it (hooks included): Dynamo's own counters after a
        # fullgraph=False compile. torch._dynamo.explain(unet._call_impl) does not see the hooks (measured on CPU).
        unet._compiled_call_impl = None
        torch._dynamo.reset()
        from torch._dynamo.utils import counters

        counters.clear()
        try:
            unet.compile(fullgraph = False, dynamic = False, backend = "aot_eager")
            with torch.no_grad():
                for _ in range(3):
                    got = unet(x)
            sync()
            res["default_compile"] = {"ok": True, "rel_l2_vs_eager": rel(got, want), "depth_after": rt._STATE.get("depth")}
        except Exception as e:  # noqa: BLE001
            res["default_compile"] = {"ok": False, "error": _err(e)}
        # A trace that dies in a hook skips the whole frame (counted under "unimplemented", frames ok < total) rather
        # than recording a graph break, so the module call then runs its hooks AND forward eagerly; count both.
        gb = dict(counters.get("graph_break") or {})
        un = dict(counters.get("unimplemented") or {})
        fr = dict(counters.get("frames") or {})
        res["explain"] = {"graph_break_count": sum(gb.values()), "break_reasons": [str(k)[:400] for k in gb][:8],
                          "unimplemented_count": sum(un.values()), "unimplemented": [str(k)[:400] for k in un][:8],
                          "frames_total": fr.get("total"), "frames_ok": fr.get("ok"),
                          "unique_graphs": (counters.get("stats") or {}).get("unique_graphs"), "source": "counters"}
        rt._STATE["depth"] = 0
        rt._STATE.pop("tuning", None)
        tunable.tuning_enable(False)

        # Eager tuning still engages: a fresh shape, called twice (the second call is "steady" and tunes).
        unet._compiled_call_impl = None
        torch._dynamo.reset()
        f = Path(rt._STATE.get("file") or "")
        y = torch.randn(3, 64, 320, device = dev, dtype = torch.bfloat16)
        seen = []
        orig_enable = tunable.tuning_enable

        def spy(v = True):
            seen.append(bool(v))
            return orig_enable(v)

        tunable.tuning_enable = spy
        try:
            n0 = n_results()
            with torch.no_grad():
                unet(y)
                n1 = n_results()
                unet(y)
            sync()
            n2 = n_results()
        finally:
            tunable.tuning_enable = orig_enable
        rows = 0
        if f.is_file():
            rows = sum(1 for line in f.read_text(encoding = "utf-8").splitlines()
                       if line and not line.startswith("Validator,"))
        res["eager_tune"] = {"results_before": n0, "results_after_first": n1, "results_after_steady": n2,
                             "tuning_enable_calls": seen, "file": str(f), "file_exists": f.is_file(),
                             "file_rows": rows, "depth": rt._STATE.get("depth"),
                             "tuning_on_after": bool(tunable.tuning_is_enabled()),
                             "written": rt._STATE.get("written")}
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
    work = args.out.parent / f"tunableop_{args.state}"
    work.mkdir(parents = True, exist_ok = True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTORCH_TUNABLEOP_")}
    env.update(UNSLOTH_DIFFUSION_ROCM_TUNABLEOP = "tune", UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR = str(work),
               TORCHINDUCTOR_CACHE_DIR = str(work / "inductor"), PYTHONIOENCODING = "utf-8")
    out = args.out.with_name(f"{args.out.stem}_child.json")
    t0 = time.time()
    try:
        rc = subprocess.run([sys.executable, __file__, "--state", args.state, "--checkout", args.checkout,
                             "--out", str(out), "--child"], env = env, timeout = 2400).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    got = json.loads(out.read_text(encoding = "utf-8")) if out.is_file() else {"missing_output": True}
    got["rc"], got["seconds"] = rc, round(time.time() - t0, 1)
    _write(args.out, got)
    return 0


if __name__ == "__main__":
    sys.exit(main())

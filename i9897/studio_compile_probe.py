"""Studio GGUF image generation with torch.compile instrumented: which inductor / dynamo config the COMPILING thread
saw, how many graphs compiled, every graph break (reason), suppressed compile errors, aliasing warnings, and first vs
warm generate time. Observes only; JSON to --out.

  python studio_compile_probe.py --backend-dir <tree>/studio/backend --out r.json [--speed-mode default|max|off]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import threading
import time
import traceback
import warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend-dir", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--repo", default = "unsloth/FLUX.2-klein-4B-GGUF")
    ap.add_argument("--gguf", default = "flux-2-klein-4b-Q2_K.gguf")
    ap.add_argument("--speed-mode", default = None)
    ap.add_argument("--steps", type = int, default = 4)
    ap.add_argument("--size", type = int, default = 512)
    ap.add_argument("--repeats", type = int, default = 3)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.backend_dir))
    os.chdir(os.path.abspath(args.backend_dir))
    res: dict = {"platform": platform.platform(), "speed_mode": args.speed_mode,
                 "env": {k: os.environ.get(k) for k in ("CC", "CI", "TORCHINDUCTOR_ERROR_ON_CUSTOM_OP_ALIASING")}}
    compiles: list = []
    alias_warnings: list = []
    suppressed: list = []

    class _Keep(logging.Handler):
        def emit(self, rec):
            msg = rec.getMessage()
            if "WON'T CONVERT" in msg or "suppress" in msg.lower() or "Backend compiler" in msg:
                suppressed.append(f"{rec.name}: {msg[:500]}")

    logging.getLogger("torch._dynamo").addHandler(_Keep())

    orig_showwarning = warnings.showwarning

    def _show(message, category, filename, lineno, file = None, line = None):
        if "alloc_from_pool" in str(message) or "alias" in str(message):
            alias_warnings.append(str(message)[:300])
        return orig_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = _show
    try:
        import torch
        import torch._inductor.compile_fx as cfx
        from torch._dynamo.utils import counters

        res.update(torch = torch.__version__, hip = getattr(torch.version, "hip", None),
                   device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        orig_compile_fx = cfx.compile_fx

        def _compile_fx(gm, *a, **k):
            import torch._dynamo.config as dc
            import torch._inductor.config as ic
            entry = {
                "thread": threading.current_thread().name,
                "nodes": len(list(gm.graph.nodes)),
                "memory_planning": ic.memory_planning,
                "memory_pool": ic.memory_pool,
                "suppress_errors": dc.suppress_errors,
                "max_autotune": ic.max_autotune,
                "triton.cudagraphs": ic.triton.cudagraphs,
            }
            t0 = time.time()
            try:
                out = orig_compile_fx(gm, *a, **k)
                entry["ok"] = True
                return out
            except Exception as exc:  # noqa: BLE001
                entry["ok"] = False
                entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                raise
            finally:
                entry["s"] = round(time.time() - t0, 2)
                compiles.append(entry)

        cfx.compile_fx = _compile_fx

        from core.inference import diffusion_speed
        res["torch_compile_runtime_available"] = diffusion_speed.torch_compile_runtime_available()
        if sys.platform == "win32":
            from core import _msvc_env
            res["crt_headers_reachable"] = _msvc_env.crt_headers_reachable()
            res["toolchain"] = _msvc_env._toolchain_summary()

        from core.inference.diffusion import get_diffusion_backend
        from core.inference import diffusion_gguf_compile

        backend = get_diffusion_backend()
        t0 = time.time()
        kw = {"speed_mode": args.speed_mode} if args.speed_mode else {}
        backend.load_pipeline(args.repo, gguf_filename = args.gguf, **kw)
        res["load_s"] = round(time.time() - t0, 1)
        res["compiled_dequant_installed"] = diffusion_gguf_compile.is_compiled_dequant_installed()
        try:
            res["status_speed"] = {k: v for k, v in (backend.status() or {}).items() if "speed" in k or "compile" in k}
        except Exception as exc:  # noqa: BLE001
            res["status_speed"] = repr(exc)
        import torch._dynamo.config as dc_main
        import torch._inductor.config as ic_main
        res["main_thread_cfg"] = {"memory_planning": ic_main.memory_planning,
                                  "suppress_errors": dc_main.suppress_errors}
        gens = []
        for i in range(args.repeats):
            t0 = time.time()
            try:
                backend.generate(prompt = "a red apple on a wooden table", width = args.size, height = args.size,
                                 steps = args.steps, seed = i)
                gens.append({"ok": True, "s": round(time.time() - t0, 2)})
            except Exception as exc:  # noqa: BLE001
                gens.append({"ok": False, "s": round(time.time() - t0, 2),
                             "error": f"{type(exc).__name__}: {str(exc)[:600]}",
                             "tb": traceback.format_exc()[-2500:]})
                break
        res["generates"] = gens
        res["generate_ok"] = bool(gens) and all(g["ok"] for g in gens)
        res["counters"] = {k: {str(kk): vv for kk, vv in v.items()} for k, v in counters.items()
                           if k in ("stats", "graph_break", "frames", "inductor", "unimplemented")}
    except Exception as exc:  # noqa: BLE001
        res["setup_error"] = f"{type(exc).__name__}: {exc}"[:1500]
        res["setup_tb"] = traceback.format_exc()[-3000:]
    res["compiles"] = compiles
    res["alias_warnings"] = alias_warnings[:20]
    res["suppressed"] = suppressed[:20]
    with open(args.out, "w", encoding = "utf-8") as fh:
        json.dump(res, fh, indent = 1, default = str)
    sys.stdout.flush()
    os._exit(0)  # a lingering render / prefetch thread once held a Windows probe open for 2 h


if __name__ == "__main__":
    main()

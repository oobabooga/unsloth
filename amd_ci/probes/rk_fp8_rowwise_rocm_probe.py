#!/usr/bin/env python3
"""Probe: can FP8 rowwise reach hipBLASLt on this ROCm card, and with which out_dtype does unsloth call it
(unsloth/kernels/fp8.py, #16598 follow-up)? Observes only (judged by criteria/rk_fp8_rowwise_rocm.py).

For every python given (--pythons, comma separated: the Studio venv's torch and a torch >= 2.14 ROCm venv), per state,
in a child process:
  raw       torch._scaled_mm with float8_e4m3fn operands (fnuz on gfx942), rowwise scales (M,1) / (1,N) and
            out_dtype bf16 / fp16 / fp32, plus a tensorwise bf16 call: ok, or the exact error text
  decision  the state's own routing code, extracted by AST from <checkout>/unsloth/kernels/fp8.py (no `import unsloth`,
            so it runs on a torch the rest of the stack may not support): the FP16_OUT_BROKEN / DEQUANT_ONLY flags,
            the rowwise backend the probe picks on this card, and for fp16 / fp32 activations whether the call would go
            to torch._scaled_mm and with which out_dtype
  unsloth   (only where `import unsloth` works: the Studio venv) the same decision from the real imported module,
            whose file must be this checkout's, plus the branch's CPU test file run against this checkout
JSON to --out only.
"""

from __future__ import annotations

import argparse
import ast
import functools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

NAMES = {
    "_has_fbgemm_rowwise", "_rowwise_gemm_works", "_probe_fp8_rowwise_backend", "_scaled_mm_fp16_out_broken",
    "_SCALED_MM_FP16_OUT_BROKEN", "_SCALED_MM_FP16_BIAS_OUT", "_SCALED_MM_FP16_ZERO_BIAS", "_SCALED_MM_FP16_DEQUANT_ONLY",
    "_probe_scaled_mm_float32_out", "_scaled_mm_fp16_dequant_only", "_scaled_mm_dtype_supported", "_scaled_mm_out_dtype",
}
TEST_FILE = "tests/test_fp8_rowwise_rocm_out_dtype.py"


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent = 2, default = str), encoding = "utf-8")


def _err(e: BaseException, n: int = 600) -> str:
    return f"{type(e).__name__}: {str(e)[:n]}"


def _raw(torch, dev) -> dict:
    out = {}
    fnuz = "gfx942" in str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    f8 = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    a = torch.randn(128, 256, device = dev).to(f8)
    b = torch.randn(128, 256, device = dev).to(f8)
    sa = torch.ones(128, 1, device = dev)
    sb = torch.ones(1, 128, device = dev)
    ref = a.float() @ b.float().t()
    for name, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)):
        try:
            y = torch._scaled_mm(a, b.t(), scale_a = sa, scale_b = sb, out_dtype = dt)
            torch.cuda.synchronize()
            out[f"rowwise_{name}"] = {"ok": True, "max_abs_vs_fp32": float((y.float() - ref).abs().max())}
        except Exception as e:  # noqa: BLE001
            out[f"rowwise_{name}"] = {"ok": False, "error": _err(e)}
    try:
        one = torch.ones((), device = dev)
        torch._scaled_mm(a, b.t(), scale_a = one, scale_b = one, out_dtype = torch.bfloat16)
        torch.cuda.synchronize()
        out["tensorwise_bf16"] = {"ok": True}
    except Exception as e:  # noqa: BLE001
        out["tensorwise_bf16"] = {"ok": False, "error": _err(e)}
    out["fp8_dtype"] = str(f8)
    return out


def _decide(ns: dict, torch, dev) -> dict:
    d = {k: ns.get(k) for k in ("_SCALED_MM_FP16_OUT_BROKEN", "_SCALED_MM_FP16_BIAS_OUT", "_SCALED_MM_FP16_ZERO_BIAS",
                                "_SCALED_MM_FP16_DEQUANT_ONLY")}
    d["has_fp16_out_broken_fn"] = "_scaled_mm_fp16_out_broken" in ns
    try:
        d["backend"] = ns["_probe_fp8_rowwise_backend"](torch.device(dev))
    except Exception as e:  # noqa: BLE001
        d["backend_error"] = _err(e)
    for name, dt in (("fp16", torch.float16), ("fp32", torch.float32), ("bf16", torch.bfloat16)):
        x = torch.zeros(4, 4, dtype = dt, device = dev)
        row = {}
        try:
            row["dtype_supported"] = bool(ns["_scaled_mm_dtype_supported"](x))
            row["out_dtype"] = str(ns["_scaled_mm_out_dtype"](dt))
        except Exception as e:  # noqa: BLE001
            row["error"] = _err(e)
        row["to_scaled_mm"] = d.get("backend") == "scaled_mm" and bool(row.get("dtype_supported"))
        d[name] = row
    return d


def _extract(checkout: str, torch) -> dict:
    src = (Path(checkout) / "unsloth" / "kernels" / "fp8.py").read_text(encoding = "utf-8")
    keep = []
    for node in ast.parse(src).body:
        names = set()
        if isinstance(node, ast.FunctionDef):
            names = {node.name}
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if names & NAMES:
            # drop decorators other than functools.lru_cache (keeps the module's caching, nothing else)
            keep.append(node)
    ns = {"torch": torch, "functools": functools, "__name__": "fp8_extract"}
    exec(compile(ast.Module(body = keep, type_ignores = []), "fp8_extract", "exec"), ns)
    # _scaled_mm_dtype_supported calls the assume_constant_result-wrapped name; eager, the probe itself is the same.
    ns.setdefault("_scaled_mm_float32_out", ns.get("_probe_scaled_mm_float32_out"))
    ns["_extracted"] = sorted(n for n in NAMES if n in ns)
    return ns


def child(args) -> int:
    res: dict = {"python": sys.executable}
    try:
        import torch

        dev = "cuda"
        res.update(torch = torch.__version__, hip = getattr(torch.version, "hip", None), platform = sys.platform,
                   cuda_available = torch.cuda.is_available())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            res["arch"] = getattr(p, "gcnArchName", None)
            res["device"] = p.name
        res["raw"] = _raw(torch, dev)
        ns = _extract(args.checkout, torch)
        res["extracted"] = ns["_extracted"]
        res["decision"] = _decide(ns, torch, dev)
        if args.try_unsloth:
            u: dict = {}
            try:
                sys.path.insert(0, args.checkout)
                os.environ.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")
                import unsloth  # noqa: F401
                from unsloth.kernels import fp8 as F8

                u["file"] = F8.__file__
                u["from_checkout"] = os.path.abspath(F8.__file__).startswith(os.path.abspath(args.checkout))
                u["decision"] = _decide(vars(F8), torch, dev)
            except Exception as e:  # noqa: BLE001
                u["error"] = _err(e)
                u["traceback"] = traceback.format_exc()[-1500:]
            res["unsloth"] = u
    except Exception as e:  # noqa: BLE001
        res["error"] = _err(e)
        res["traceback"] = traceback.format_exc()[-3000:]
    _write(args.out, res)
    return 0


def cpu_test(python: str, checkout: str, test_src: Path) -> dict:
    """The branch's CPU test against THIS checkout, from a directory outside both trees (no rootdir pythonpath)."""
    if not test_src.is_file():
        return {"error": f"no test file at {test_src}"}
    tmp = Path(tempfile.mkdtemp(prefix = "rk7_test_"))
    try:
        shutil.copy(test_src, tmp / test_src.name)
        env = dict(os.environ, PYTHONPATH = checkout, UNSLOTH_DISABLE_AUTO_UPDATES = "1")
        p = subprocess.run([python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(tmp),
                            "-c", os.devnull, str(tmp / test_src.name)], cwd = tmp, env = env,
                           capture_output = True, text = True, timeout = 900)
        lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
        return {"rc": p.returncode, "summary": lines[-1] if lines else "", "tail": "\n".join(lines[-15:])[-2500:]}
    except Exception as e:  # noqa: BLE001
        return {"error": _err(e)}
    finally:
        shutil.rmtree(tmp, ignore_errors = True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--pythons", default = "", help = "comma separated interpreters; default: this one")
    ap.add_argument("--unsloth-python", default = "", help = "the interpreter where `import unsloth` is tried")
    ap.add_argument("--test-from", default = "", help = "checkout holding the branch's CPU test file (head)")
    ap.add_argument("--child", action = "store_true", help = argparse.SUPPRESS)
    ap.add_argument("--try-unsloth", action = "store_true", help = argparse.SUPPRESS)
    args = ap.parse_args()
    args.checkout = str(Path(args.checkout).resolve())
    if args.child:
        return child(args)
    pythons = [p for p in (args.pythons or sys.executable).split(",") if p]
    res: dict = {"state": args.state, "checkout": args.checkout, "runs": {}}
    for i, py in enumerate(pythons):
        out = args.out.with_name(f"{args.out.stem}_py{i}.json")
        t0 = time.time()
        cmd = [py, os.path.abspath(__file__), "--state", args.state, "--checkout", args.checkout, "--out", str(out),
               "--child"] + (["--try-unsloth"] if py == args.unsloth_python else [])
        try:
            rc = subprocess.run(cmd, timeout = 1800).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        got = json.loads(out.read_text(encoding = "utf-8")) if out.is_file() else {"missing_output": True}
        got["rc"], got["seconds"] = rc, round(time.time() - t0, 1)
        res["runs"][py] = got
    if args.unsloth_python and args.test_from:
        res["cpu_test"] = cpu_test(args.unsloth_python, args.checkout, Path(args.test_from) / TEST_FILE)
    _write(args.out, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())

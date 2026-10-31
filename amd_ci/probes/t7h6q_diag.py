#!/usr/bin/env python3
"""Diagnostic (T7h.6q): on this host, is a C++ compiler (cl) on PATH, what does zoo #2273's up-front probe report,
and does torch's real message match the probe's classifier. Observes only; writes JSON to --out.

Runs the head checkout's tests/_cpu_compiler_skip.py exactly as the tests do (_probe(dict(os.environ)) in a fresh
process), and separately asks Inductor for its C++ compiler in-process to record torch's own exception text.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required = True, type = Path, help = "zoo head checkout")
    ap.add_argument("--base", type = Path, help = "zoo base checkout (only recorded: it has no probe)")
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    rec: dict = {"python": sys.executable, "platform": sys.platform}
    rec["which"] = {c: shutil.which(c) for c in ("cl", "cl.exe", "clang-cl", "g++", "clang++", "gcc", "cc")}
    rec["env"] = {k: os.environ.get(k) for k in ("CXX", "CC", "VCINSTALLDIR", "VSINSTALLDIR", "INCLUDE",
                                                 "TORCHINDUCTOR_CACHE_DIR", "TORCHDYNAMO_DISABLE",
                                                 "TORCH_COMPILE_DISABLE", "TORCHDYNAMO_SUPPRESS_ERRORS")}
    rec["path_has_msvc"] = [p for p in os.environ.get("PATH", "").split(os.pathsep) if "MSVC" in p or "Visual Studio" in p]
    try:
        p = subprocess.run(["where.exe", "cl"], capture_output = True, encoding = "utf-8", errors = "replace", timeout = 60)
        rec["where_cl"] = {"rc": p.returncode, "out": (p.stdout + p.stderr)[-1000:]}
    except Exception as e:  # noqa: BLE001
        rec["where_cl"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        import torch

        rec["torch"] = {"version": torch.__version__, "hip": getattr(torch.version, "hip", None),
                        "cuda_available": torch.cuda.is_available()}
    except Exception as e:  # noqa: BLE001
        rec["torch"] = {"error": f"{type(e).__name__}: {e}"}
    # torch's own message, in-process.
    try:
        from torch._inductor.cpp_builder import get_cpp_compiler

        rec["get_cpp_compiler"] = {"ok": True, "value": str(get_cpp_compiler())}
    except BaseException as e:  # noqa: BLE001
        chain, seen, x = [], set(), e
        while x is not None and id(x) not in seen:
            seen.add(id(x))
            chain.append([type(x).__name__, str(x)])
            x = x.__cause__ or getattr(x, "inner_exception", None) or x.__context__
        rec["get_cpp_compiler"] = {"ok": False, "chain": chain, "tb": traceback.format_exc()[-1500:]}
    # The PR's probe, as the tests run it.
    tests = args.head / "tests"
    rec["head_has_probe"] = (tests / "_cpu_compiler_skip.py").is_file()
    if rec["head_has_probe"]:
        sys.path.insert(0, str(tests))
        try:
            import _cpu_compiler_skip as s

            rc, out = s._probe(dict(os.environ))
            lines = [ln for ln in (out or "").splitlines() if ln.startswith("PROBE ")]
            rep = s._report(out) if rc == 3 else None
            rec["probe"] = {"returncode": rc, "probe_line": lines[-1] if lines else None, "stdout_tail": (out or "")[-1500:],
                            "report": rep, "missing_compiler": s.missing_compiler(rc, out)}
            msgs = [m for _, m in (rep or {}).get("chain", []) if isinstance(m, str)]
            rec["probe"]["windows_regex_matches"] = [m for m in msgs if s._WINDOWS.match(m.strip())]
            g = rec.get("get_cpp_compiler") or {}
            rec["in_process_message_matches_classifier"] = any(
                (n == "InvalidCxxCompiler") or (n == "RuntimeError" and s._WINDOWS.match(m.strip()))
                for n, m in g.get("chain", []))
        except Exception as e:  # noqa: BLE001
            rec["probe"] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}
    if args.base:
        rec["base_has_probe"] = (args.base / "tests" / "_cpu_compiler_skip.py").is_file()
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(rec, indent = 1, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

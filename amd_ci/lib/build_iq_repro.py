#!/usr/bin/env python3
"""Build the smallest pair of GGUFs that differ only in one quant type.

The hypothesis this exists to test: across the field reports of Qwen3.8-27B,
`IQ4_NL` is present in exactly the files that crash (V3 UD-Q4_K_XL with 6 such
tensors, UD-Q5_K_XL with 1) and absent from exactly the ones that work (V3
UD-Q6_K_XL, V2 UD-Q4_K_XL). `IQ4_XS` appears in a file that works, which makes
it the right control: it is a neighbouring i-quant, so a pair that differs only
`iq4_nl` vs `iq4_xs` isolates one kernel family rather than "i-quants in
general".

Two files, a few hundred MB each, from a 0.6B model, both quantized identically
apart from `ffn_down`. If the iq4_nl side faults and the iq4_xs side does not,
that is a complete reproducer someone upstream can actually run.

Its limits, which belong in the report rather than in a footnote: a 0.6B model's
`ffn_down` is far smaller than the 27B's [17408, 5120], so a size-dependent
shader bug can hide here. And this model is not the `qwen35` SSM hybrid, so it
says nothing at all about the context-checkpoint path. A clean result here does
not clear IQ4_NL; only a dirty one is conclusive.

Writes a states.json the differential runner consumes directly.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def quantize(binary: Path, src: Path, dst: Path, tensor_type: str,
             base_type: str, threads: int, env: dict) -> dict:
    dst.parent.mkdir(parents = True, exist_ok = True)
    cmd = [str(binary),
           # The source is already quantized; quality is irrelevant here, only
           # which kernel the weights end up in.
           "--allow-requantize",
           "--tensor-type", f"ffn_down={tensor_type}",
           str(src), str(dst), base_type, str(threads)]
    p = subprocess.run(cmd, capture_output = True, text = True, timeout = 3600, env = env)
    return {"cmd": cmd, "rc": p.returncode,
            "tail": ((p.stdout or "") + (p.stderr or ""))[-2000:],
            "bytes": dst.stat().st_size if dst.is_file() else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required = True, type = Path)
    ap.add_argument("--source", required = True, type = Path, help = "any small GGUF")
    ap.add_argument("--root", required = True, type = Path, help = "where to put the pair")
    ap.add_argument("--base-type", default = "Q5_K_M")
    ap.add_argument("--control-type", default = "iq4_xs")
    ap.add_argument("--suspect-type", default = "iq4_nl")
    ap.add_argument("--threads", type = int, default = 8)
    ap.add_argument("--out", required = True, type = Path, help = "states.json to write")
    ap.add_argument("--report", type = Path, default = None)
    args = ap.parse_args()

    binary = args.bin_dir / "llama-quantize"
    if not binary.is_file():
        print(f"no llama-quantize at {binary}", file = sys.stderr)
        return 1

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(args.bin_dir), env.get("LD_LIBRARY_PATH", "")]).strip(os.pathsep)

    # base is the CONTROL and head is the SUSPECT, because the criteria module
    # asks "is head worse than base" and the working side has to be base.
    plan = {"base": args.control_type, "head": args.suspect_type}
    report: dict = {"source": str(args.source), "base_type": args.base_type, "runs": {}}
    paths: dict[str, str] = {}
    for state, ttype in plan.items():
        d = args.root / state
        report["runs"][state] = quantize(
            binary, args.source, d / "repro.gguf", ttype, args.base_type, args.threads, env)
        paths[state] = str(d)

    ok = all(r["rc"] == 0 and r["bytes"] for r in report["runs"].values())
    doc = {"paths": paths, "commits": plan}
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(doc, indent = 2))
    if args.report:
        args.report.write_text(json.dumps(report, indent = 2))
    print(json.dumps({"ok": ok, "states": doc}, indent = 2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Probe: does this backend compute each quant type's matmul kernels correctly?

The smallest reproducer available for the hypothesis that one quant type's
kernels are broken on this backend. `test-backend-ops` ships inside every
llama.cpp prebuilt tarball, checks each op against a CPU reference, and needs no
model, no network and about two minutes.

The motivating question: across the field reports, `IQ4_NL` is present in
exactly the GGUFs that crash (V3 UD-Q4_K_XL, UD-Q5_K_XL) and absent from exactly
the ones that work (V3 UD-Q6_K_XL, V2 UD-Q4_K_XL). `IQ4_XS` appears in a working
file, so it serves as an in-run control: if iq4_nl fails and iq4_xs passes in the
same invocation, that is not an artefact of the harness.

This is a CHARACTERISATION, not a differential. It has no base and no head, and
it deliberately has no criteria module - there is nothing here to call a
regression, only a fact about this host. Reported as a table, given no verdict.

Observes only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# test-backend-ops prints one line per case, e.g.
#   MUL_MAT(type_a=iq4_nl,type_b=f32,m=16,n=1,k=256,...): OK
# The trailing verdict word is what matters; the parenthesised parameters carry
# the quant type.
CASE_RE = re.compile(
    r"^\s*(?P<op>[A-Z_0-9]+)\((?P<params>[^)]*)\)\s*:\s*(?P<verdict>OK|FAIL|NOT SUPPORTED|SKIPPED)",
    re.MULTILINE)
TYPE_RE = re.compile(r"type_a=(?P<t>[a-zA-Z0-9_]+)")


def run_ops(binary: Path, backend: str, op: str, env: dict, timeout: int) -> dict:
    cmd = [str(binary), "test", "-b", backend, "-o", op]
    try:
        p = subprocess.run(cmd, capture_output = True, text = True,
                           timeout = timeout, env = env)
    except subprocess.TimeoutExpired:
        return {"cmd": cmd, "timed_out": True}
    except Exception as e:  # noqa: BLE001
        return {"cmd": cmd, "error": f"{type(e).__name__}: {e}"}

    text = (p.stdout or "") + (p.stderr or "")
    by_type: dict[str, dict[str, int]] = {}
    failures: list[str] = []
    for m in CASE_RE.finditer(text):
        tm = TYPE_RE.search(m.group("params"))
        # Cases with no type_a are still counted, under a name that cannot
        # collide with a quant type.
        key = tm.group("t") if tm else "(untyped)"
        verdict = m.group("verdict")
        by_type.setdefault(key, {})
        by_type[key][verdict] = by_type[key].get(verdict, 0) + 1
        if verdict == "FAIL":
            failures.append(m.group(0).strip())

    return {
        "cmd": cmd,
        "rc": p.returncode,
        "by_type": by_type,
        "failing_types": sorted(t for t, v in by_type.items() if v.get("FAIL")),
        "failures": failures[:60],
        "cases_seen": sum(sum(v.values()) for v in by_type.values()),
        "tail": text[-4000:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    # Accepted for harness compatibility; this probe has no per-state input.
    ap.add_argument("--state", default = "single")
    ap.add_argument("--checkout", default = "")
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--bin-dir", required = True, type = Path)
    ap.add_argument("--backend", required = True,
                    help = "device name as test-backend-ops spells it, e.g. Vulkan0, ROCm0")
    ap.add_argument("--op", action = "append", default = [],
                    help = "repeatable; defaults to MUL_MAT and MUL_MAT_ID")
    ap.add_argument("--timeout", type = int, default = 1800)
    args = ap.parse_args()

    ops = args.op or ["MUL_MAT", "MUL_MAT_ID"]
    binary = args.bin_dir / "test-backend-ops"

    obs: dict = {"state": args.state, "backend": args.backend,
                 "binary": str(binary), "ops": ops}

    args.out.parent.mkdir(parents = True, exist_ok = True)
    if not binary.is_file():
        obs["setup_error"] = f"no test-backend-ops at {binary}"
        args.out.write_text(json.dumps(obs, indent = 2))
        return 0

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(args.bin_dir), env.get("LD_LIBRARY_PATH", "")]).strip(os.pathsep)

    obs["results"] = {op: run_ops(binary, args.backend, op, env, args.timeout) for op in ops}

    # The headline, and the control alongside it. iq4_xs is present in a GGUF
    # that works in the field, so "iq4_nl fails, iq4_xs passes" means something,
    # while "everything fails" means the invocation is wrong.
    watch = ("iq4_nl", "iq4_xs", "iq3_s", "q3_K", "q4_K", "q5_K", "q6_K", "q8_0")
    summary: dict[str, dict] = {}
    for t in watch:
        for op, res in obs["results"].items():
            counts = (res.get("by_type") or {}).get(t)
            if counts:
                summary.setdefault(t, {})[op] = counts
    obs["watched_types"] = summary
    obs["any_failure"] = any(r.get("failing_types") for r in obs["results"].values())
    obs["failing_types"] = sorted({t for r in obs["results"].values()
                                   for t in (r.get("failing_types") or [])})

    args.out.write_text(json.dumps(obs, indent = 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

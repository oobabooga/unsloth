#!/usr/bin/env python3
"""Probe: unsloth-zoo GPT-OSS flex inference (zoo #2153) in one state's checkout. Observes only (criteria/gptoss_flex.py
judges).

1. tests/test_gpt_oss_flex_inference.py through zoo_pytest_probe.py (outcomes by id, torch / HIP / arch, which
   unsloth_zoo file was imported). A state without the file (the base) gets the head checkout's copy, so the base
   runs the same tests against its own zoo: they must fail there (no flex inference path).
2. Head only: the eager-vs-flex sanity script (alternating UNSLOTH_GPT_OSS_EAGER_INFERENCE=1 / 0, 3 rounds), inside
   a wrapper that records every Inductor flex lowering decision (flex_decoding vs the main kernel, per query length)
   and zoo's per-VRAM kernel_options.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST = "tests/test_gpt_oss_flex_inference.py"

WRAP = r'''
import atexit, json, os, runpy, sys
out = sys.argv[1]; script = sys.argv[2]
rec = {"decisions": [], "errors": []}
def dump():
    m = sys.modules.get("unsloth_zoo.flex_attention.utils")
    rec["kernel_options"] = getattr(m, "kernel_options", "module not imported") if m else "module not imported"
    rec["prefill_kernel_options"] = getattr(m, "PREFILL_KERNEL_OPTIONS", None) if m else None
    with open(out, "w", encoding = "utf-8") as f:
        json.dump(rec, f, indent = 2, default = str)
atexit.register(dump)
try:
    import torch._inductor.kernel.flex.flex_attention as FA
    _orig = FA._use_flex_decoding
    def _wrapped(*args, **kwargs):
        r = _orig(*args, **kwargs)
        try:
            q = args[0] if args else kwargs.get("query")
            ql = str(q.get_size()[-2])
        except Exception as e:
            ql = f"? {type(e).__name__}"
        opts = args[3] if len(args) > 3 else kwargs.get("kernel_options")
        rec["decisions"].append({"q_len": ql, "flex_decoding": bool(r),
                                 "eager_env": os.environ.get("UNSLOTH_GPT_OSS_EAGER_INFERENCE"),
                                 "kernel_options": {k: str(v) for k, v in dict(opts or {}).items()}})
        return r
    FA._use_flex_decoding = _wrapped
except Exception as e:
    rec["errors"].append(f"wrap: {type(e).__name__}: {e}")
sys.argv = [script]
runpy.run_path(script, run_name = "__main__")
'''


def _parse_sanity(text: str) -> dict:
    import ast
    rows = []
    for line in text.splitlines():
        head, _, rest = line.partition(" ")
        if head in ("eager", "flex", "train") and rest.startswith("{"):
            try:
                rows.append({"mode": head, **ast.literal_eval(rest)})
            except Exception:  # noqa: BLE001
                pass
        elif head == "dtype":
            rows.append({"mode": "header", "line": line})
    return {"rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--only-states", default = "base,head")
    ap.add_argument("--sanity", default = str(HERE / "gptoss_flex_sanity.py"))
    ap.add_argument("--timeout", type = int, default = 1800)
    ap.add_argument("--wall", type = int, default = 5400)
    args = ap.parse_args()

    co = Path(args.checkout)
    obs: dict = {"state": args.state, "checkout": str(co)}
    obs["zoo_sha"] = subprocess.run(["git", "-C", str(co), "rev-parse", "HEAD"], capture_output = True,
                                    encoding = "utf-8", errors = "replace").stdout.strip()
    if args.state not in args.only_states.split(","):
        obs["skipped"] = f"state {args.state} not in --only-states {args.only_states}"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0
    test = co / TEST
    if not test.exists():
        src = co.parent / "head" / TEST
        if src.exists():
            shutil.copyfile(src, test)
            obs["test_copied_from_head"] = str(src)
        else:
            obs["test_missing"] = f"{test} absent and no head copy at {src}"

    py_out = args.out.with_name(args.out.stem + "_tests.json")
    p = subprocess.run([args.python, str(HERE / "zoo_pytest_probe.py"), "--state", args.state, "--checkout", str(co),
                        "--out", str(py_out), "--python", args.python, "--timeout", str(args.timeout),
                        "--wall", str(args.wall), "--only-states", args.only_states, "--tests", TEST],
                       capture_output = True, encoding = "utf-8", errors = "replace")
    obs["pytest_probe_rc"] = p.returncode
    obs["pytest_probe_tail"] = ((p.stdout or "") + (p.stderr or ""))[-2000:]
    if py_out.is_file():
        obs.update({k: v for k, v in json.loads(py_out.read_text(encoding = "utf-8")).items()
                    if k not in ("state", "checkout", "zoo_sha")})

    if args.state == "head":
        work = args.out.parent / f"sanity_{args.state}"
        work.mkdir(parents = True, exist_ok = True)
        wrap = work / "wrap.py"
        wrap.write_text(WRAP, encoding = "utf-8")
        rec = work / "kernels.json"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(co)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        env.setdefault("UNSLOTH_IS_PRESENT", "1")
        env["TORCHINDUCTOR_CACHE_DIR"] = str(work / "inductor")
        env["UNSLOTH_COMPILE_LOCATION"] = str(work / "unsloth_compiled_cache")
        env.setdefault("PYTHONUTF8", "1")
        try:
            s = subprocess.run([args.python, str(wrap), str(rec), args.sanity], cwd = co, env = env,
                               capture_output = True, encoding = "utf-8", errors = "replace", timeout = args.wall)
            rc, sout, serr = s.returncode, s.stdout or "", s.stderr or ""
        except subprocess.TimeoutExpired as exc:
            rc, serr = -1, "TimeoutExpired"
            sout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
        (work / "sanity.log").write_text(sout + "\n--- stderr ---\n" + serr, encoding = "utf-8")
        sanity = {"rc": rc, "stdout_tail": sout[-3000:], "stderr_tail": serr[-3000:], **_parse_sanity(sout)}
        if rec.is_file():
            sanity["kernels"] = json.loads(rec.read_text(encoding = "utf-8"))
        obs["sanity"] = sanity
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

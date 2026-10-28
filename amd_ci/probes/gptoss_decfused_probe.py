#!/usr/bin/env python3
"""Probe: GPT-OSS fused decode kernels (zoo unsloth_zoo/gpt_oss_decode_fused.py) on gfx1151.
Observes only (criteria/gptoss_decfused.py judges).

base = zoo main (fused path gated off on ROCm), head = main + UNSLOTH_GPT_OSS_FUSED_DECODE=1 forcing it on ROCm.
All work runs in the HEAD state's invocation (the base invocation records "skipped"):
  backend    libdevice exp / rsqrt and tl.math.div_rn on this Triton; _ENABLED for env unset / 0 / 1, head and base
  pytest     tests/test_gpt_oss_decode_fused.py + tests/test_gpt_oss_flex_inference.py on head, forced on
  bench      real gpt-oss-20b decode at 1k x1 and x4, arms E F F E (head tree, kill switch 0 vs forced 1) plus
             one base arm (default env), each a fresh process with its own compile caches
  corr       same process, fused on / off toggled: greedy tokens + teacher-forced logits (step 0 bit identity)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

BACKEND = r'''
import json, os, sys
r = {}
try:
    import torch, triton
    import triton.language as tl
    r["torch"], r["triton"], r["hip"] = torch.__version__, triton.__version__, torch.version.hip
    try:
        from triton.language.extra import libdevice
        r["libdevice"] = True
        r["libdevice_exp"] = hasattr(libdevice, "exp")
        r["libdevice_rsqrt"] = hasattr(libdevice, "rsqrt")
    except Exception as e:
        r["libdevice"] = False
        r["libdevice_error"] = repr(e)[:300]
    r["div_rn"] = hasattr(tl.math, "div_rn")
    from unsloth_zoo import gpt_oss_decode_fused as F
    r["enabled"] = bool(F._ENABLED)
    r["module_libdevice"] = F._libdevice is not None
except Exception as e:
    r["error"] = repr(e)[:500]
print("RESULT " + json.dumps(r))
'''


def _git(co: Path, *a: str) -> str:
    return subprocess.run(["git", "-C", str(co), *a], capture_output = True, encoding = "utf-8",
                          errors = "replace").stdout.strip()


def _run(cmd, cwd, env, timeout):
    t = time.time()
    try:
        p = subprocess.run(cmd, cwd = cwd, env = env, capture_output = True, encoding = "utf-8", errors = "replace",
                           timeout = timeout)
        rc, text = p.returncode, (p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or "")
    except subprocess.TimeoutExpired as exc:
        rc, text = -1, "TimeoutExpired\n" + (exc.stdout if isinstance(exc.stdout, str) else "")
    return rc, text, round(time.time() - t, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--model", default = "unsloth/gpt-oss-20b-unsloth-bnb-4bit")
    ap.add_argument("--arm-timeout", type = int, default = 2700)
    args = ap.parse_args()

    co = Path(args.checkout)
    obs: dict = {"state": args.state, "checkout": str(co), "zoo_sha": _git(co, "rev-parse", "HEAD")}
    if args.state != "head":
        obs["skipped"] = "all arms run in the head state's invocation"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0
    cos = {"head": co, "base": co.parent / "base"}
    obs["base_sha"] = _git(cos["base"], "rev-parse", "HEAD")
    obs["model"] = args.model
    work = args.out.parent / "decfused"
    work.mkdir(parents = True, exist_ok = True)
    base_env = dict(os.environ)
    base_env.setdefault("HF_HOME", str(Path(os.environ.get("AMD_CI_WORK", str(work))) / "hf"))
    base_env.update(UNSLOTH_DISABLE_AUTO_UPDATES = "1", PYTHONUTF8 = "1", UNSLOTH_IS_PRESENT = "1")
    base_env.pop("UNSLOTH_GPT_OSS_FUSED_DECODE", None)
    obs["ci_env_unset"] = base_env.pop("CI", None)

    def env_for(state, d, flag):
        env = dict(base_env, PYTHONPATH = os.pathsep.join([str(cos[state])] + (
            [base_env["PYTHONPATH"]] if base_env.get("PYTHONPATH") else [])))
        env["UNSLOTH_COMPILE_LOCATION"] = str(d / "unsloth_compiled_cache")
        env["TORCHINDUCTOR_CACHE_DIR"] = str(d / "inductor")
        env["TRITON_CACHE_DIR"] = str(d / "triton")
        if flag is not None:
            env["UNSLOTH_GPT_OSS_FUSED_DECODE"] = flag
        return env

    def save():
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")

    # 1. backend capabilities and the gate, per state and env value
    obs["backend"] = {}
    for st in ("base", "head"):
        for flag in (None, "0", "1"):
            d = work / f"backend_{st}_{flag}"
            d.mkdir(parents = True, exist_ok = True)
            (d / "b.py").write_text(BACKEND, encoding = "utf-8")
            rc, text, s = _run([args.python, str(d / "b.py")], d, env_for(st, d, flag), 600)
            line = [x for x in text.splitlines() if x.startswith("RESULT ")]
            obs["backend"][f"{st}:{flag}"] = json.loads(line[-1][7:]) if line else {"rc": rc, "tail": text[-800:]}
    save()

    # 2. unit suites on head, forced on
    d = work / "pytest"
    d.mkdir(parents = True, exist_ok = True)
    rc, text, s = _run([args.python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rA",
                        "tests/test_gpt_oss_decode_fused.py", "tests/test_gpt_oss_flex_inference.py"],
                       str(co), env_for("head", d, "1"), 2400)
    (d / "log.txt").write_text(text, encoding = "utf-8")
    summary = [x for x in text.splitlines() if (" passed" in x or " failed" in x) and x.strip().startswith(("=", "1", "2",
               "3", "4", "5", "6", "7", "8", "9"))]
    obs["pytest"] = {"rc": rc, "s": s, "summary": summary[-1:] if summary else [],
                     "failed": [x for x in text.splitlines() if x.startswith(("FAILED", "ERROR"))][:40],
                     "passed": [x for x in text.splitlines() if x.startswith("PASSED")],
                     "tail": text[-2500:]}
    save()

    # 3. model download
    t = time.time()
    rc, text, s = _run([args.python, "-c", "import sys; from huggingface_hub import snapshot_download; "
                        "print(snapshot_download(sys.argv[1]))", args.model], str(work), base_env, 3600)
    obs["download"] = {"rc": rc, "s": s, "tail": text[-800:]}
    save()

    # 4. decode timing: head E F F E (same code, kill switch vs forced), then one base arm (shipped default)
    arms = [("E1", "head", "0"), ("F1", "head", "1"), ("F2", "head", "1"), ("E2", "head", "0"), ("B1", "base", None)]
    obs["arms"] = {}
    for name, st, flag in arms:
        d = work / f"bench_{name}"
        d.mkdir(parents = True, exist_ok = True)
        rc, text, s = _run([args.python, "-u", str(HERE / "gptoss_decfused_bench.py"), str(d / "result.json"),
                            args.model], d, env_for(st, d, flag), args.arm_timeout)
        (d / "log.txt").write_text(text, encoding = "utf-8")
        rec = {"state": st, "flag": flag, "rc": rc, "s": s, "tail": text[-2000:]}
        if (d / "result.json").is_file():
            res = json.loads((d / "result.json").read_text(encoding = "utf-8"))
            for b in ("b1", "b4"):
                if b in res:
                    res[b].pop("scores_first8_top", None)
            rec["result"] = res
        obs["arms"][name] = rec
        print(f"arm {name} rc={rc} {s}s", flush = True)
        save()
        for sub in ("unsloth_compiled_cache", "inductor", "triton"):
            subprocess.run(["rm", "-rf", "--", str(d / sub)])
        for f in d.glob("*.pt"):
            f.unlink()

    # 5. same-process correctness, head tree, forced on at import
    d = work / "corr"
    d.mkdir(parents = True, exist_ok = True)
    rc, text, s = _run([args.python, "-u", str(HERE / "gptoss_decfused_corr.py"), str(d / "corr.json"), args.model],
                       d, env_for("head", d, "1"), args.arm_timeout)
    (d / "log.txt").write_text(text, encoding = "utf-8")
    obs["corr"] = {"rc": rc, "s": s, "tail": text[-2000:]}
    if (d / "corr.json").is_file():
        obs["corr"]["result"] = json.loads((d / "corr.json").read_text(encoding = "utf-8"))
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())

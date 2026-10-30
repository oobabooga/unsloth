#!/usr/bin/env python3
"""Probe: zoo main after the Round 7f GPT-OSS decode merges (#2233 cached Triton launches, #2236 one-launch RoPE,
#2246 bnb direct cgemm_4bit) on gfx1151, base = the commit before #2233. Observes only (criteria/gptoss7f.py judges).

All work happens in the HEAD state's invocation so base and head interleave (A B B A); the base invocation records
"skipped". Fresh process per arm, PYTHONPATH = that state's zoo checkout, its own compile caches:
  bench_base_1 (+ greedy tokens), bench_head_1 (+ greedy tokens), bench_head_2, bench_base_2,
  then pytest_base, pytest_head over --tests (junit XML, failing test IDs compared by the criteria).
gptoss20b_bench.py records each arm's engagement (fast-launch / rope_decode / bnb-direct gates and call counts).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _git(co: Path, *a: str) -> str:
    return subprocess.run(["git", "-C", str(co), *a], capture_output = True, encoding = "utf-8",
                          errors = "replace").stdout.strip()


def _junit(path: Path) -> dict:
    out = {"passed": [], "failed": [], "skipped": [], "errors": []}
    if not path.is_file():
        return out
    for tc in ET.parse(path).getroot().iter("testcase"):
        tid = f"{tc.get('classname')}::{tc.get('name')}"
        kinds = {c.tag for c in tc}
        if "failure" in kinds:
            out["failed"].append(tid)
        elif "error" in kinds:
            out["errors"].append(tid)
        elif "skipped" in kinds:
            out["skipped"].append(tid)
        else:
            out["passed"].append(tid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--model", default = "unsloth/gpt-oss-20b-unsloth-bnb-4bit")
    ap.add_argument("--cases", default = "1024x1,1024x4")
    ap.add_argument("--new-tokens", type = int, default = 128)
    ap.add_argument("--repeats", type = int, default = 3)
    ap.add_argument("--arm-timeout", type = int, default = 3000)
    ap.add_argument("--tests", default = "tests/test_triton_fast_launch.py,tests/test_gpt_oss_rope_decode.py,"
                    "tests/test_bnb_4bit_direct.py,tests/test_moe_routed.py,tests/test_gpt_oss_decode_fused.py,"
                    "tests/test_gpt_oss_routed_nf4.py")
    ap.add_argument("--expect-head-tree-of", default = "4a74d204548a3d61968ccae84e07f459468aa5ef")
    args = ap.parse_args()

    co = Path(args.checkout)
    obs: dict = {"state": args.state, "checkout": str(co), "zoo_sha": _git(co, "rev-parse", "HEAD")}
    if args.state != "head":
        obs["skipped"] = "all arms (base and head, interleaved) run in the head state's invocation"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0
    cos = {"head": co, "base": co.parent / "base"}
    obs["base_sha"] = _git(cos["base"], "rev-parse", "HEAD")
    obs["head_tree"] = _git(co, "rev-parse", "HEAD^{tree}")
    obs["expected_tree"] = os.environ.get("AMD_CI_EXPECT_TREE") or _git(co, "rev-parse",
                                                                         f"{args.expect_head_tree_of}^{{tree}}")
    obs["model"] = args.model
    work = args.out.parent / "gptoss7f"
    work.mkdir(parents = True, exist_ok = True)
    base_env = dict(os.environ)
    base_env.setdefault("HF_HOME", str(Path(os.environ.get("AMD_CI_WORK", str(work))) / "hf"))
    base_env.update(UNSLOTH_DISABLE_AUTO_UPDATES = "1", PYTHONUTF8 = "1", UNSLOTH_IS_PRESENT = "1")
    obs["ci_env_unset"] = base_env.pop("CI", None)

    t = time.time()
    dl = subprocess.run([args.python, "-c", "import sys; from huggingface_hub import snapshot_download; "
                         "print(snapshot_download(sys.argv[1]))", args.model], env = base_env, capture_output = True,
                        encoding = "utf-8", errors = "replace")
    obs["download"] = {"rc": dl.returncode, "s": round(time.time() - t, 1), "tail": (dl.stdout + dl.stderr)[-800:]}
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")

    bench = [str(HERE / "gptoss20b_bench.py"), "--cases", args.cases, "--new-tokens", str(args.new_tokens),
             "--repeats", str(args.repeats), "--no-profile", "--model", args.model]
    arms = [
        ("bench_base_1", "base", bench + ["--gen-out", "{dir}/gen.json"]),
        ("bench_head_1", "head", bench + ["--gen-out", "{dir}/gen.json"]),
        ("bench_head_2", "head", bench),
        ("bench_base_2", "base", bench),
    ]
    obs["arms"] = {}

    def env_for(st, d):
        env = dict(base_env)
        env["PYTHONPATH"] = os.pathsep.join([str(cos[st])] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        env["UNSLOTH_COMPILE_LOCATION"] = str(d / "unsloth_compiled_cache")
        env["TORCHINDUCTOR_CACHE_DIR"] = str(d / "inductor")
        env["TRITON_CACHE_DIR"] = str(d / "triton")
        return env

    for name, st, argv in arms:
        d = work / name
        d.mkdir(parents = True, exist_ok = True)
        av = [x.replace("{dir}", str(d)) for x in argv] + ["--out", str(d / "result.json")]
        t = time.time()
        try:
            p = subprocess.run([args.python, "-u"] + av, cwd = d, env = env_for(st, d), capture_output = True,
                               encoding = "utf-8", errors = "replace", timeout = args.arm_timeout)
            rc, text = p.returncode, (p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or "")
        except subprocess.TimeoutExpired as exc:
            rc = -1
            text = "TimeoutExpired\n" + (exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(
                "utf-8", "replace"))
        (d / "log.txt").write_text(text, encoding = "utf-8")
        rec = {"state": st, "rc": rc, "s": round(time.time() - t, 1), "tail": text[-2500:]}
        for f in ("result.json", "gen.json"):
            if (d / f).is_file():
                rec[f.split(".")[0]] = json.loads((d / f).read_text(encoding = "utf-8"))
        obs["arms"][name] = rec
        print(f"arm {name} ({st}) rc={rc} {rec['s']}s", flush = True)
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        for sub in ("unsloth_compiled_cache", "inductor", "triton"):
            subprocess.run(["rm", "-rf", "--", str(d / sub)])

    tests = [x for x in args.tests.split(",") if x]
    obs["pytest"] = {}
    for st in ("base", "head"):
        d = work / f"pytest_{st}"
        d.mkdir(parents = True, exist_ok = True)
        present = [x for x in tests if (cos[st] / x).is_file()]
        xml = d / "junit.xml"
        t = time.time()
        rec = {"present": present, "absent": [x for x in tests if x not in present]}
        probe_py = ("import json, sys, torch, unsloth_zoo; print('ENV', json.dumps({'zoo': unsloth_zoo.__file__, "
                    "'hip': torch.version.hip, 'avail': torch.cuda.is_available(), "
                    "'arch': torch.cuda.get_device_properties(0).gcnArchName if torch.cuda.is_available() else None}))")
        e = subprocess.run([args.python, "-c", probe_py], cwd = cos[st], env = env_for(st, d), capture_output = True,
                           encoding = "utf-8", errors = "replace")
        rec["env"] = next((json.loads(line[4:]) for line in (e.stdout or "").splitlines() if line.startswith("ENV ")),
                          {"error": (e.stdout + e.stderr)[-800:]})
        try:
            p = subprocess.run([args.python, "-m", "pytest", "-o", "addopts=", "-p", "no:cacheprovider", "-q", "-rfE",
                                f"--junitxml={xml}", *present], cwd = cos[st], env = env_for(st, d),
                               capture_output = True, encoding = "utf-8", errors = "replace",
                               timeout = args.arm_timeout)
            rc, text = p.returncode, (p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or "")
        except subprocess.TimeoutExpired:
            rc, text = -1, "TimeoutExpired"
        (d / "log.txt").write_text(text, encoding = "utf-8")
        rec.update(rc = rc, s = round(time.time() - t, 1), tail = text[-3000:], **_junit(xml))
        obs["pytest"][st] = rec
        print(f"pytest {st} rc={rc} passed={len(rec['passed'])} failed={len(rec['failed'])} "
              f"errors={len(rec['errors'])} skipped={len(rec['skipped'])} {rec['s']}s", flush = True)
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        for sub in ("unsloth_compiled_cache", "inductor", "triton"):
            subprocess.run(["rm", "-rf", "--", str(d / sub)])
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

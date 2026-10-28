#!/usr/bin/env python3
"""Probe: real GPT-OSS 20b on gfx1151, zoo base vs head (the merged flex-inference series #2153 / #2161 / #2163).
Observes only (criteria/gptoss20b.py judges).

All work happens in the HEAD state's invocation, so base and head arms can interleave (A B B A); the base invocation
records "skipped". Every arm is a fresh process with PYTHONPATH = that state's zoo checkout and its own compile caches:
  bench_base_1 (+ greedy tokens), bench_head_1 (+ greedy tokens), gen_head_eager (UNSLOTH_GPT_OSS_EAGER_INFERENCE=1),
  bench_head_2, bench_base_2, sft_base, sft_head
Model: --model (default from $AMD_CI_GPTOSS_MODEL), downloaded once before the arms into $AMD_CI_WORK/hf.
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


def _git(co: Path, *a: str) -> str:
    return subprocess.run(["git", "-C", str(co), *a], capture_output = True, encoding = "utf-8",
                          errors = "replace").stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--model", default = os.environ.get("AMD_CI_GPTOSS_MODEL", "unsloth/gpt-oss-20b-unsloth-bnb-4bit"))
    ap.add_argument("--no-4bit", action = "store_true", default = os.environ.get("AMD_CI_GPTOSS_NO4BIT") == "1")
    ap.add_argument("--cases", default = "1024x1,1024x4,4096x1")
    ap.add_argument("--new-tokens", type = int, default = 128)
    ap.add_argument("--repeats", type = int, default = 3)
    ap.add_argument("--arm-timeout", type = int, default = 3000)
    ap.add_argument("--expect-head-tree-of", default = "af7f8ecd2b220fc1c814c47573de6aac161d4116")
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
    obs["expected_tree"] = _git(co, "rev-parse", f"{args.expect_head_tree_of}^{{tree}}")
    obs["model"], obs["load_in_4bit"] = args.model, not args.no_4bit
    work = args.out.parent / "gptoss20b"
    work.mkdir(parents = True, exist_ok = True)
    base_env = dict(os.environ)
    base_env.setdefault("HF_HOME", str(Path(os.environ.get("AMD_CI_WORK", str(work))) / "hf"))
    base_env.update(UNSLOTH_DISABLE_AUTO_UPDATES = "1", PYTHONUTF8 = "1", UNSLOTH_IS_PRESENT = "1")
    base_env.pop("UNSLOTH_GPT_OSS_EAGER_INFERENCE", None)
    # GitHub sets CI=true, and torch then turns its custom-op aliasing DEPRECATION WARNING into an error
    # (functorch config error_on_custom_op_aliasing defaults to bool($CI)); users' processes do not have CI set, so
    # the arms run without it and the warnings are counted instead (alias_warnings per arm).
    obs["ci_env_unset"] = base_env.pop("CI", None)

    def pick_check(model, no4):
        d = work / ("pick_" + model.replace("/", "_"))
        d.mkdir(parents = True, exist_ok = True)
        env = dict(base_env, PYTHONPATH = str(co), UNSLOTH_COMPILE_LOCATION = str(d / "cc"),
                   TORCHINDUCTOR_CACHE_DIR = str(d / "ind"))
        try:
            p = subprocess.run([args.python, "-u", str(HERE / "gptoss20b_bench.py"), "--model", model, "--cases", "none",
                                "--gen-tokens", "8", "--gen-out", str(d / "gen.json"), "--no-profile",
                                "--out", str(d / "r.json")] + (["--no-4bit"] if no4 else []), cwd = d, env = env,
                               capture_output = True, encoding = "utf-8", errors = "replace", timeout = 2400)
            rc, text = p.returncode, (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired:
            rc, text = -1, "TimeoutExpired"
        subprocess.run(["rm", "-rf", "--", str(d / "cc"), str(d / "ind")])
        return {"model": model, "no_4bit": no4, "rc": rc, "tail": text[-1500:]}

    # Prefer the bnb 4-bit checkpoint (what the T4 / G4 runs used); fall back to BF16 if bitsandbytes cannot run here.
    obs["model_pick"] = [pick_check(args.model, args.no_4bit)]
    if obs["model_pick"][0]["rc"] != 0 and not args.no_4bit:
        args.model, args.no_4bit = "unsloth/gpt-oss-20b-BF16", True
        obs["model_pick"].append(pick_check(args.model, True))
    obs["model"], obs["load_in_4bit"] = args.model, not args.no_4bit
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")

    t = time.time()
    dl = subprocess.run([args.python, "-c", "import sys; from huggingface_hub import snapshot_download; "
                         "print(snapshot_download(sys.argv[1]))", args.model], env = base_env, capture_output = True,
                        encoding = "utf-8", errors = "replace")
    obs["download"] = {"rc": dl.returncode, "s": round(time.time() - t, 1), "tail": (dl.stdout + dl.stderr)[-800:]}

    common = ["--model", args.model] + (["--no-4bit"] if args.no_4bit else [])
    bench = [str(HERE / "gptoss20b_bench.py"), "--cases", args.cases, "--new-tokens", str(args.new_tokens),
             "--repeats", str(args.repeats), "--no-profile"] + common
    arms = [
        ("bench_base_1", "base", bench + ["--gen-out", "{dir}/gen.json"], {}),
        ("bench_head_1", "head", bench + ["--gen-out", "{dir}/gen.json"], {}),
        ("gen_head_eager", "head", [str(HERE / "gptoss20b_bench.py"), "--cases", "none", "--gen-out", "{dir}/gen.json",
                                    "--no-profile"] + common, {"UNSLOTH_GPT_OSS_EAGER_INFERENCE": "1"}),
        ("bench_head_2", "head", bench, {}),
        ("bench_base_2", "base", bench, {}),
        ("sft_base", "base", [str(HERE / "gptoss20b_sft.py")] + common, {}),
        ("sft_head", "head", [str(HERE / "gptoss20b_sft.py")] + common, {}),
    ]
    obs["arms"] = {}
    for name, st, argv, extra in arms:
        d = work / name
        d.mkdir(parents = True, exist_ok = True)
        env = dict(base_env, **extra)
        env["PYTHONPATH"] = os.pathsep.join([str(cos[st])] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        env["UNSLOTH_COMPILE_LOCATION"] = str(d / "unsloth_compiled_cache")
        env["TORCHINDUCTOR_CACHE_DIR"] = str(d / "inductor")
        av = [x.replace("{dir}", str(d)) for x in argv]
        av += ["--out", str(d / "result.json")]
        t = time.time()
        try:
            p = subprocess.run([args.python, "-u"] + av, cwd = d, env = env, capture_output = True, encoding = "utf-8",
                               errors = "replace", timeout = args.arm_timeout)
            rc, text = p.returncode, (p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or "")
        except subprocess.TimeoutExpired as exc:
            rc = -1
            text = "TimeoutExpired\n" + (exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(
                "utf-8", "replace"))
        (d / "log.txt").write_text(text, encoding = "utf-8")
        rec = {"state": st, "rc": rc, "s": round(time.time() - t, 1), "env": extra,
               "tail": text[-2500:], "dynamic_cache_fallback": "dynamic cache instead" in text,
               "alias_warnings": text.count("will become an error in a future version of PyTorch")}
        for f in ("result.json", "gen.json"):
            if (d / f).is_file():
                rec[f.split(".")[0]] = json.loads((d / f).read_text(encoding = "utf-8"))
        obs["arms"][name] = rec
        print(f"arm {name} ({st}) rc={rc} {rec['s']}s", flush = True)
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        for sub in ("unsloth_compiled_cache", "inductor", "sft_out"):
            subprocess.run(["rm", "-rf", "--", str(d / sub)])
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

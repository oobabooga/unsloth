#!/usr/bin/env python3
"""Probe: Inkling 2 banded attention on one llama.cpp state (ROCm build). Observes only.

Per state, with that state's OWN binaries (the head adds ggml API, so a head
binary cannot load base libraries):
  1. test-backend-ops eval vs CPU for the banded op and the ops the head fuses
  2. llama-perplexity on Inkling-2-Mini: short (-c 512, one sequence per batch,
     banded path, no SWA wrap) and long (-c 4096 -ub 512, SWA cache wraps past
     1536 cells). The base state writes the KLD reference logits; every later
     state (and a second base run, the A/A floor) is scored against them.
  3. llama-bench pp512 / pp2048 / tg128, flash attention on, twice (A/A spread).

Writes JSON to --out, never stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

TBO_OPS = ["FLASH_ATTN_EXT_BANDED", "SSM_CONV", "SSM_CONV_RES_SCALE", "TANH_GATE_MIX", "DEPTH_MIX",
           "MUL_MAT_SHARED_SRC1", "CONCAT"]
PPL = {
    "short": ["-c", "512", "-b", "512", "-ub", "512", "--chunks", "8"],
    "long":  ["-c", "4096", "-b", "4096", "-ub", "512", "--chunks", "2"],
}


def sh(cmd: list[str], env: dict, log: Path, timeout: int) -> tuple[int, str, float]:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, env = env, capture_output = True, text = True, timeout = timeout)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        out += "\n<<TIMEOUT>>"
        rc = -999
    log.write_text(" ".join(cmd) + "\n" + out, encoding = "utf-8")
    return rc, out, time.time() - t0


def parse_eval(out: str) -> dict:
    m = re.findall(r"(\d+)/(\d+) tests passed", out)
    passed, total = (int(m[-1][0]), int(m[-1][1])) if m else (0, 0)
    fails = [l.strip() for l in out.splitlines() if re.search(r"\bFAIL\b", l)]
    return {"passed": passed, "total": total, "fail_lines": fails[:20], "n_fail_lines": len(fails)}


def parse_ppl(out: str) -> dict:
    r: dict = {}
    m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", out)
    if m:
        r["ppl"], r["ppl_err"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"Mean\s+KLD:\s+(-?[\d.e+-]+)", out)
    if m:
        r["kld_mean"] = float(m.group(1))
    m = re.search(r"Maximum KLD:\s+(-?[\d.e+-]+)", out)
    if m:
        r["kld_max"] = float(m.group(1))
    m = re.search(r"Same top p:\s+([\d.]+)", out)
    if m:
        r["top1_pct"] = float(m.group(1))
    return r


def parse_bench(out: str) -> dict:
    res: dict = {}
    try:
        start = out.index("[")
        rows = json.loads(out[start:out.rindex("]") + 1])
    except Exception:  # noqa: BLE001
        return res
    for row in rows:
        key = f"pp{row['n_prompt']}" if row.get("n_prompt") else f"tg{row['n_gen']}"
        res[key] = row.get("avg_ts")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--model", required = True, type = Path)
    ap.add_argument("--text", required = True, type = Path)
    ap.add_argument("--kld-dir", required = True, type = Path)
    ap.add_argument("--build-dir", default = "build")
    ap.add_argument("--timeout", type = int, default = 3600)
    ap.add_argument("--backend", default = "ROCm0")
    args = ap.parse_args()

    bindir = (args.checkout / args.build_dir / "bin").resolve()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{bindir}:{env.get('LD_LIBRARY_PATH', '')}"
    logs = args.out.parent / f"logs_{args.state}"
    logs.mkdir(parents = True, exist_ok = True)
    args.kld_dir.mkdir(parents = True, exist_ok = True)

    obs: dict = {"state": args.state, "libdir": str(bindir)}
    tbo = bindir / "test-backend-ops"
    try:
        ldd = subprocess.run(["ldd", str(tbo)], env = env, capture_output = True, text = True).stdout
        hip = [l.split("=>")[1].split("(")[0].strip() for l in ldd.splitlines()
               if re.search(r"libggml-hip\.so", l) and "=>" in l]
        obs["hip_lib"] = hip[0] if hip else None
    except Exception as e:  # noqa: BLE001
        obs["ldd_error"] = f"{type(e).__name__}: {e}"

    obs["eval"] = {}
    for op in TBO_OPS:
        rc, out, secs = sh([str(tbo), "-o", op, "-b", args.backend], env, logs / f"eval_{op}.log", args.timeout)
        if "device_lines" not in obs:
            obs["device_lines"] = [l.strip() for l in out.splitlines() if re.search(r"Device \d+:", l)][:4]
            obs["gfx1151"] = any("gfx1151" in l for l in obs["device_lines"])
        r = parse_eval(out)
        r.update({"rc": rc, "secs": round(secs, 1)})
        obs["eval"][op] = r

    ppl_bin = str(bindir / "llama-perplexity")
    common = ["-m", str(args.model), "-f", str(args.text), "-ngl", "99", "-fa", "on", "-t", "8"]
    obs["ppl"] = {}
    for name, extra in PPL.items():
        ref = args.kld_dir / f"ref_{name}.bin"
        runs = []
        if args.state == "base" and not ref.exists():
            runs.append(("ref", ["--kl-divergence-base", str(ref)]))
        runs.append(("score", ["--kl-divergence-base", str(ref), "--kl-divergence"]))
        for tag, kargs in runs:
            if tag == "score" and not ref.exists():
                obs["ppl"][f"{name}:{tag}"] = {"error": "no reference logits"}
                continue
            rc, out, secs = sh([ppl_bin, *common, *extra, *kargs], env, logs / f"ppl_{name}_{tag}.log", args.timeout)
            r = parse_ppl(out)
            r.update({"rc": rc, "secs": round(secs, 1)})
            obs["ppl"][f"{name}:{tag}"] = r

    obs["bench"] = []
    for rep in range(2):
        rc, out, secs = sh([str(bindir / "llama-bench"), "-m", str(args.model), "-ngl", "99", "-fa", "1",
                            "-p", "512,2048", "-n", "128", "-r", "3", "-o", "json"],
                           env, logs / f"bench_{rep}.log", args.timeout)
        r = parse_bench(out)
        r.update({"rc": rc, "secs": round(secs, 1)})
        obs["bench"].append(r)

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

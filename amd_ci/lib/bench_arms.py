#!/usr/bin/env python3
"""llama-bench over several binary directories, arms alternated inside one job.

Two rules this exists to enforce, both learned the hard way on this hardware:

  * Arms alternate. A block of one arm followed much later by a block of another
    measures the box drifting as much as the change; the same binary has read
    11.0 and 13.1 ms/token hours apart.
  * The spread is reported next to the mean. A difference smaller than the
    within-arm spread is not a difference, and the only way a reader can see
    that is if both numbers are printed.

Each arm is a directory holding llama-bench plus its backend libraries, so
"ROCm stock", "ROCm patched" and "Vulkan" are three arms of the same shape.
Standard library only; llama-bench's own JSON output is parsed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def run_arm(name: str, bin_dir: str, model: str, extra: list[str], env_extra: dict,
            timeout: int, log: Path) -> dict:
    exe = str(Path(bin_dir) / ("llama-bench.exe" if os.name == "nt" else "llama-bench"))
    env = dict(os.environ)
    if os.name != "nt":
        env["LD_LIBRARY_PATH"] = f"{bin_dir}{os.pathsep}" + env.get("LD_LIBRARY_PATH", "")
    env.update(env_extra)
    cmd = [exe, "-m", model, "-o", "json", *extra]
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, capture_output = True, text = True, timeout = timeout,
                           encoding = "utf-8", errors = "replace", env = env)
    except subprocess.TimeoutExpired:
        return {"arm": name, "error": f"timed out after {timeout}s", "cmd": cmd}
    log.write_text(f"CMD: {' '.join(cmd)}\n\n{r.stdout}\n\n--- stderr ---\n{r.stderr}",
                   encoding = "utf-8")
    out: dict = {"arm": name, "bin": bin_dir, "rc": r.returncode,
                 "seconds": round(time.monotonic() - t0, 1), "cmd": cmd}
    if r.returncode != 0:
        # A failed arm is a failed arm. Reporting no rows for it would read as
        # "nothing to say" next to arms that produced numbers.
        out["error"] = (r.stderr or r.stdout or "")[-1200:]
        return out
    try:
        rows = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        out["error"] = f"unparsable llama-bench json: {e}"
        out["stdout"] = r.stdout[:2000]
        return out
    out["rows"] = [{k: row.get(k) for k in
                    ("model_type", "n_prompt", "n_gen", "n_depth", "avg_ts", "stddev_ts",
                     "backends", "gpu_info", "build_commit")} for row in rows]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action = "append", required = True,
                    help = "NAME=BINDIR, repeatable; the order given is the cycle order")
    ap.add_argument("--arm-env", action = "append", default = [],
                    help = "NAME=KEY=VALUE, environment for one arm only")
    ap.add_argument("--model", required = True)
    ap.add_argument("--cycles", type = int, default = 3,
                    help = "passes over every arm; arms alternate within a pass")
    # default=None, then filled in below: argparse APPENDS to a list default, so
    # a caller passing one flag would silently get the defaults as well.
    ap.add_argument("--bench-arg", action = "append", default = None,
                    help = "llama-bench flag, repeatable; pass as --bench-arg=-p to keep "
                           "argparse from reading the value as an option")
    ap.add_argument("--timeout", type = int, default = 3600)
    ap.add_argument("--out", required = True)
    ap.add_argument("--log-dir", default = "")
    a = ap.parse_args()
    if a.bench_arg is None:
        a.bench_arg = ["-p", "512", "-n", "128", "-r", "2", "-ngl", "999"]

    arms = []
    for item in a.arm:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--arm wants NAME=BINDIR, got {item!r}")
        arms.append((name, path))
    env_for: dict[str, dict] = {}
    for item in a.arm_env:
        name, _, kv = item.partition("=")
        k, _, v = kv.partition("=")
        env_for.setdefault(name, {})[k] = v
    known = {n for n, _ in arms}
    unknown = [n for n in env_for if n not in known]
    if unknown:
        raise SystemExit(f"--arm-env names no such arm: {unknown}")

    log_dir = Path(a.log_dir or Path(a.out).parent)
    log_dir.mkdir(parents = True, exist_ok = True)
    res: dict = {"model": a.model, "cycles": a.cycles, "bench_args": a.bench_arg,
                 "arms": [n for n, _ in arms], "arm_env": env_for, "results": []}
    for cycle in range(a.cycles):
        for name, path in arms:
            print(f"== cycle {cycle} arm {name}", flush = True)
            r = run_arm(name, path, a.model, a.bench_arg, env_for.get(name, {}), a.timeout,
                        log_dir / f"bench_{name}_c{cycle}.log")
            r["cycle"] = cycle
            res["results"].append(r)
            for row in r.get("rows") or []:
                print(f"   {row['n_prompt'] or 0}p {row['n_gen'] or 0}g "
                      f"d{row.get('n_depth') or 0}: {row['avg_ts']} t/s", flush = True)

    summary: dict = {}
    for r in res["results"]:
        for row in r.get("rows") or []:
            key = f"pp{row['n_prompt'] or 0}_tg{row['n_gen'] or 0}_d{row.get('n_depth') or 0}"
            summary.setdefault(key, {}).setdefault(r["arm"], []).append(row["avg_ts"])
    res["summary"] = {
        key: {arm: {"n": len(v), "mean": round(statistics.fmean(v), 2),
                    "min": round(min(v), 2), "max": round(max(v), 2),
                    "spread": round(max(v) - min(v), 2)}
              for arm, v in per_arm.items()}
        for key, per_arm in summary.items()}

    lines = ["", "### llama-bench, arms alternated", "",
             "| workload | arm | mean t/s | min | max | spread | reps |", "|---|---|---|---|---|---|---|"]
    for key, per_arm in res["summary"].items():
        for arm, s in per_arm.items():
            lines.append(f"| {key} | {arm} | {s['mean']} | {s['min']} | {s['max']} | "
                         f"{s['spread']} | {s['n']} |")
        best = max(per_arm.items(), key = lambda kv: kv[1]["mean"])
        worst = min(per_arm.items(), key = lambda kv: kv[1]["mean"])
        if len(per_arm) > 1:
            gap = best[1]["mean"] - worst[1]["mean"]
            widest = max(s["spread"] for s in per_arm.values())
            verdict = ("separated" if gap > widest else
                       "NOT separated: the gap is inside the within-arm spread")
            lines.append(f"| {key} | {best[0]} over {worst[0]} | "
                         f"{round(100 * gap / worst[1]['mean'], 1)}% | | | {verdict} | |")
    res["table"] = "\n".join(lines)
    Path(a.out).write_text(json.dumps(res, indent = 2), encoding = "utf-8")
    print(res["table"])
    failed = [r["arm"] for r in res["results"] if r.get("error")]
    if failed:
        print(f"\n**arms that produced no rows: {sorted(set(failed))}** - "
              f"a missing arm is not a slow arm.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

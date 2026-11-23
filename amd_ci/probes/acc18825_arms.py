#!/usr/bin/env python3
"""PR 18825 accuracy recheck: run diffusion_lora_train_probe.py once per ARM (own processes, same seed), each arm a
(commit, env) pair, then compare per-step losses / grad norms and the saved adapters bit for bit.

Arms run sequentially on one GPU. The first arm is a throwaway warm-up (downloads, first-process slowness). Results:
<out-dir>/arms.json (every arm's numbers) and <out-dir>/SUMMARY.md. Observes only; the report judges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

KILL_ENVS = (
    "UNSLOTH_DIFFUSION_GC_MODE",
    "UNSLOTH_DIFFUSION_DEQUANT_CACHE",
    "UNSLOTH_DIFFUSION_GC_SAVE_ATTENTION",
    "UNSLOTH_DIFFUSION_FAST_LORA",
    "UNSLOTH_DIFFUSION_DEQUANT_CACHE_GB",
)


def sh(cmd, cwd = None):
    p = subprocess.run(cmd, cwd = cwd, capture_output = True, text = True)
    return p.returncode, (p.stdout + p.stderr).strip()


def adapter_stats(path: Path, ref: Path | None) -> dict:
    from safetensors.torch import load_file

    sd = load_file(str(path))
    out = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16], "tensors": len(sd)}
    if ref is None or not ref.is_file():
        return out
    r = load_file(str(ref))
    num = den = 0.0
    mx = 0.0
    ndiff = 0
    nel = 0
    for k, v in sd.items():
        if k not in r:
            continue
        a, b = v.float(), r[k].float()
        d = (a - b)
        num += float(d.pow(2).sum())
        den += float(b.pow(2).sum())
        mx = max(mx, float(d.abs().max()))
        ndiff += int((d != 0).sum())
        nel += d.numel()
    out.update({"vs_ref_rel_l2": (num ** 0.5) / max(den ** 0.5, 1e-30), "vs_ref_max_abs": mx,
                "vs_ref_frac_elems_differ": ndiff / max(nel, 1)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required = True)
    ap.add_argument("--extra-commit", action = "append", default = [], help = "name=sha worktree added from the clone")
    ap.add_argument("--arms", required = True, help = "JSON list: {name, state, env, precisions}")
    ap.add_argument("--probe", required = True)
    ap.add_argument("--out-dir", required = True)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--steps", type = int, default = 20)
    ap.add_argument("--ref-arm", default = "base_a")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents = True, exist_ok = True)
    states = json.loads(Path(args.states).read_text(encoding = "utf-8"))
    paths = dict(states["paths"])
    commits = dict(states["commits"])
    repo = Path(paths["base"]).parent.parent / "repo"
    for spec in args.extra_commit:
        name, sha = spec.split("=", 1)
        wt = Path(paths["base"]).parent / name
        if not wt.exists():
            rc, msg = sh(["git", "-c", "core.autocrlf=false", "worktree", "add", "-q", "--detach", str(wt), sha],
                         cwd = repo)
            if rc != 0:
                sh(["git", "fetch", "-q", "origin", sha], cwd = repo)
                rc, msg = sh(["git", "-c", "core.autocrlf=false", "worktree", "add", "-q", "--detach", str(wt), sha],
                             cwd = repo)
            if rc != 0:
                print(f"FATAL: worktree {name} {sha}: {msg}", flush = True)
                return 2
        paths[name] = str(wt)
        commits[name] = sh(["git", "rev-parse", "HEAD"], cwd = wt)[1]
    print("states:", json.dumps(commits, indent = 1), flush = True)

    arms = json.loads(Path(args.arms).read_text(encoding = "utf-8"))
    work = Path(os.environ.get("AMD_CI_WORK", out)) / "acc_probe"
    work.mkdir(parents = True, exist_ok = True)
    adapters = work / "adapters"
    adapters.mkdir(exist_ok = True)
    results: dict = {"commits": commits, "arms": {}}
    failed = 0
    for arm in arms:
        name, state = arm["name"], arm["state"]
        env = {k: v for k, v in os.environ.items() if k not in KILL_ENVS}
        env.update({k: str(v) for k, v in (arm.get("env") or {}).items()})
        obs = out / f"obs_{name}.json"
        cmd = [args.python, "-u", args.probe, "--state", name, "--checkout", paths[state], "--out", str(obs),
               "--suite", "flux3", "--control-state", "none", "--precisions", arm.get("precisions", "bf16,nf4"),
               "--steps", str(args.steps), "--no-lora-render", "--budget-min", "900", "--work", str(work)]
        t0 = time.time()
        with open(out / f"probe_{name}.log", "wb") as fh:
            rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, env = env).returncode
        rec = {"state": state, "commit": commits.get(state), "env": arm.get("env") or {}, "rc": rc,
               "wall_s": round(time.time() - t0, 1), "cells": {}}
        stage_dir = out / f"stages_{name}"
        for prec in arm.get("precisions", "bf16,nf4").split(","):
            f = stage_dir / f"train_target_{prec}.json"
            if not f.is_file():
                rec["cells"][prec] = {"missing": True}
                failed += 1
                continue
            d = json.loads(f.read_text(encoding = "utf-8"))
            cell = {k: d.get(k) for k in ("losses", "grad_norms", "samples_per_second", "trainer_peak_memory_gb",
                                          "max_reserved_gb", "compiled", "base_precision_engaged", "errors",
                                          "lora_norms", "wall_s")}
            ev = stage_dir / f"train_target_{prec}.events.jsonl"
            if ev.is_file():
                for line in ev.read_text(encoding = "utf-8").splitlines():
                    try:
                        e = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    if e.get("type") in ("dequant_cache_planned", "fast_lora_stats"):
                        cell.setdefault("events", []).append({k: v for k, v in e.items() if k != "ts"})
                    if e.get("type") == "warning":
                        cell.setdefault("warnings", []).append(str(e.get("message"))[:300])
            lp = ((d.get("complete") or {}).get("lora_path"))
            if lp and Path(lp).is_file():
                keep = adapters / f"{name}_{prec}.safetensors"
                shutil.copy(lp, keep)
                ref = adapters / f"{args.ref_arm}_{prec}.safetensors"
                try:
                    cell["adapter"] = adapter_stats(keep, ref if ref != keep else None)
                except Exception as exc:  # noqa: BLE001
                    cell["adapter"] = {"error": f"{type(exc).__name__}: {exc}"}
            if not cell.get("losses") or len(cell["losses"]) != args.steps:
                failed += 1
            rec["cells"][prec] = cell
        results["arms"][name] = rec
        (out / "arms.json").write_text(json.dumps(results, indent = 1, default = str), encoding = "utf-8")
        print(f"arm {name} ({state} {arm.get('env')}) rc={rc} {rec['wall_s']}s", flush = True)

    # Pairwise table against the reference arm.
    lines = ["| arm | state | env | prec | max abs dloss vs ref | first differing step | max rel dgrad_norm | "
             "adapter rel L2 vs ref | samples/s | peak GB |", "|" + "---|" * 10]
    ref = results["arms"].get(args.ref_arm, {}).get("cells", {})
    for name, rec in results["arms"].items():
        for prec, c in rec["cells"].items():
            rl = (ref.get(prec) or {}).get("losses") or []
            rg = (ref.get(prec) or {}).get("grad_norms") or []
            ls = c.get("losses") or []
            gs = c.get("grad_norms") or []
            dl = [abs(a - b) for a, b in zip(ls, rl) if a is not None and b is not None]
            first = next((i + 1 for i, x in enumerate(dl) if x != 0), "-")
            dg = [abs(a - b) / max(abs(b), 1e-12) for a, b in zip(gs, rg) if a is not None and b is not None]
            ad = (c.get("adapter") or {}).get("vs_ref_rel_l2")
            lines.append(f"| {name} | {rec['state']} | {rec['env'] or '-'} | {prec} | "
                         f"{max(dl) if dl else '-'} | {first} | {max(dg) if dg else '-'} | {ad} | "
                         f"{c.get('samples_per_second')} | {c.get('trainer_peak_memory_gb')} |")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding = "utf-8")
    print("\n".join(lines), flush = True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

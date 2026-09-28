#!/usr/bin/env python3
"""Run every cell of a spec, one process per cell, in the cell's own venv, and resume where it stopped.

  python diffusion_bench/matrix.py specs/smoke_small.json --out outputs/dbench/smoke
  python diffusion_bench/matrix.py SPEC --out DIR --only 'zimg_*' --pass quick --gpus 5,6 --dry-run

Spec layout (JSON):
  {
    "name": "...", "description": "...",
    "defaults": {<cell fields>},                    # merged under every cell
    "passes": [{"name": "p1", "n": 24}, {"name": "p2", "n": 8}],   # optional; each pass reruns every cell
                                                                   # with these overrides into <out>/<pass>/
    "gate": {"max_util": 10, "min_free_mib": 60000, "timeout_s": 1800},
    "reference": {"studio": "s_bf16", "comfyui": "c_bf16"},        # scorer: reference cell per backend
    "cells": [{"tag": "...", "backend": "studio", "venv": "studio" | {"profile": ...} | path, ...}]
  }

Rules the driver enforces, all learned the hard way:
  - one cell per process, in spec order, so Studio / ComfyUI cells can be interleaved and drift over the run
    hits every arm equally rather than whichever ran last;
  - a GPU gate before each cell (a neighbour's job on the same device makes the timing meaningless; a
    timed-out gate still runs, and the record carries gate=timeout);
  - with several --gpus, cells run in parallel, one per device; only compare timings from the same device;
  - a cell with an ok record is skipped unless --force, so an interrupted matrix resumes;
  - a per-cell timeout turns a hang into a BROKEN record rather than a stuck run.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402
import envs as V  # noqa: E402


def plan(spec: dict, only: list, pass_names: list) -> list:
    passes = spec.get("passes") or [{"name": ""}]
    if pass_names:
        passes = [p for p in passes if p["name"] in pass_names]
    jobs = []
    for p in passes:
        overrides = {k: v for k, v in p.items() if k != "name"}
        for cell in spec["cells"]:
            if only and not any(fnmatch.fnmatch(cell["tag"], pat) for pat in only):
                continue
            if cell.get("skip"):
                continue
            merged = C.merge_cell(spec["defaults"], {**cell, **overrides})
            jobs.append((p["name"], merged))
    return jobs


def run_job(pass_name: str, cell: dict, out_root: Path, gpu: str, gate: dict, timeout_s: float, force: bool,
            dry: bool) -> str:
    out = out_root / pass_name if pass_name else out_root
    cell_dir = out / cell["tag"]
    if not force and C.record_ok(C.read_record(cell_dir)):
        C.log(f"skip {pass_name or '-'} / {cell['tag']}: ok record exists")
        return "skipped"
    python = V.resolve_python(cell.get("venv"), **(cell.get("venv_args") or {})) if not dry else str(cell.get("venv"))
    env = {**os.environ}
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["HIP_VISIBLE_DEVICES"] = gpu
    cmd = [python, "-u", str(HERE / "run_cell.py"), "--cell", json.dumps(cell), "--out", str(out)]
    if dry:
        C.log(f"[dry] gpu={gpu or '-'} {pass_name or '-'} / {cell['tag']} ({cell['backend']}) python={python}")
        return "dry"
    g = None
    if gate is not False and C.gpu_vendor() != "none":
        g = C.gpu_gate(device = gpu or None, **(gate or {}))
        C.log(f"gate {gpu or C.visible_gpu()}: {g.get('gate')} util={g.get('util')} used={g.get('used_mib')} MiB")
    cell_dir.mkdir(parents = True, exist_ok = True)
    log_path = cell_dir / "cell.log"
    C.log(f"=== {pass_name or '-'} / {cell['tag']} on gpu {gpu or C.visible_gpu()} -> {log_path}")
    with open(log_path, "a") as logf:
        proc = subprocess.Popen(cmd, stdout = logf, stderr = subprocess.STDOUT, env = env, cwd = str(C.WS),
                                start_new_session = True)
        try:
            code = proc.wait(timeout = timeout_s)
        except subprocess.TimeoutExpired:
            if hasattr(os, "killpg"):
                import signal

                os.killpg(proc.pid, signal.SIGKILL)
            else:  # Windows (the AMD Windows runners): no process groups via setsid, no SIGKILL; kill the tree
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output = True)
                proc.kill()
            proc.wait()
            rec = C.read_record(cell_dir) or {"tag": cell["tag"], "cell": cell}
            rec.update({"verdict": "BROKEN", "error": f"timeout after {timeout_s}s"})
            C.write_record(cell_dir, rec)
            code = 124
    rec = C.read_record(cell_dir) or {}
    if rec and g is not None:
        rec["gate"] = g
        C.write_record(cell_dir, rec)
    C.log(f"--- {cell['tag']}: exit {code}, {rec.get('verdict')}, median {rec.get('wall_s_median')}s, "
          f"s/step {rec.get('step_s_derived')}, peak {rec.get('peak_smi_gib')} GiB")
    return rec.get("verdict") or f"exit{code}"


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("--out", required = True)
    ap.add_argument("--only", action = "append", default = [], help = "fnmatch pattern on cell tags (repeatable)")
    ap.add_argument("--pass", dest = "passes", action = "append", default = [], help = "run only these passes")
    ap.add_argument("--gpus", default = "",
                    help = "comma list of devices, one worker each (default: inherit the current mask, one worker)")
    ap.add_argument("--timeout", type = float, default = 5400, help = "per cell, seconds")
    ap.add_argument("--no-gate", action = "store_true")
    ap.add_argument("--force", action = "store_true", help = "rerun cells that already have an ok record")
    ap.add_argument("--dry-run", action = "store_true")
    ap.add_argument("--set", action = "append", default = [], metavar = "KEY=JSON",
                    help = "override a default for every cell, e.g. --set n=2 --set warmup=0")
    args = ap.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.exists():
        spec_path = HERE / "specs" / args.spec
    spec = C.load_spec(spec_path)
    for item in args.set:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        # Dotted keys reach into the defaults (options.speed_mode=..., env.X=...), same as run_cell.py --set;
        # setting "options" whole would drop every other default option.
        target = spec["defaults"]
        *path, leaf = key.split(".")
        for part in path:
            target = target.setdefault(part, {})
        target[leaf] = value
    jobs = plan(spec, args.only, args.passes)
    out_root = Path(C.expand_env(args.out))
    out_root.mkdir(parents = True, exist_ok = True)
    (out_root / "spec.json").write_text(json.dumps(spec, indent = 1))
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    gate = False if args.no_gate else spec.get("gate", {})
    C.log(f"{spec['name']}: {len(jobs)} cell runs on gpus {gpus or ['(current)']} -> {out_root}")
    results: dict = {}
    work: queue.Queue = queue.Queue()
    for job in jobs:
        work.put(job)

    def worker(gpu: str) -> None:
        while True:
            try:
                pass_name, cell = work.get_nowait()
            except queue.Empty:
                return
            try:
                results[(pass_name, cell["tag"])] = run_job(pass_name, cell, out_root, gpu, gate, args.timeout,
                                                            args.force, args.dry_run)
            except Exception as exc:  # noqa: BLE001 - one broken cell never stops the matrix
                C.log(f"{cell['tag']}: driver error {type(exc).__name__}: {exc}")
                results[(pass_name, cell["tag"])] = "driver_error"

    threads = [threading.Thread(target = worker, args = (g,)) for g in (gpus or [""])]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    summary = {f"{p}/{t}" if p else t: v for (p, t), v in results.items()}
    (out_root / "matrix_summary.json").write_text(json.dumps({"elapsed_s": round(time.time() - t0, 1),
                                                              "results": summary}, indent = 1))
    bad = {k: v for k, v in summary.items() if v not in ("ok", "skipped", "dry")}
    C.log(f"done in {time.time() - t0:.0f}s: {len(summary) - len(bad)} ok/skipped, {len(bad)} not ok {bad or ''}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run ONE benchmark cell in this process and write <out>/<tag>/record.json plus the rendered media.

The timing protocol is the same for every backend, so numbers are comparable across Studio, Studio over
HTTP (Desktop), plain diffusers, ComfyUI and stable-diffusion.cpp:

  load            -> load_s (cold, includes any download / compile the backend does at load)
  warmup x W      -> cold_s (first render: compile, autotune, CUDA-graph capture, pinning); a DIFFERENT seed
                     from every timed render, because ComfyUI caches node outputs and would return a timed
                     render from cache
  N prompts       -> renders[] (new-prompt s/image, what a user sees on each new prompt), media saved
  short_n pairs   -> long[] / short[] on prompts already seen once: steady s/image, and
                     s/step = (median long - median short) / (steps - short_steps), which cancels the text
                     encoder and VAE out of the per-step figure
  GpuSampler      -> peak_smi_gib (driver view, all processes on the device); torch peak for in-process
  host memory     -> RSS anon / file / peak

Use one process per cell (``matrix.py`` does): compile caches, allocator state and cuDNN autotuning leak
across cells otherwise. A failing cell writes verdict BROKEN with the error, never a partial "ok".

Usage:
  python diffusion_bench/run_cell.py --cell '{"tag":"zimg_bf16","backend":"studio","model":"..."}' --out DIR
  python diffusion_bench/run_cell.py --spec diffusion_bench/specs/smoke.json --tag zimg_bf16 --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
from backends.base import get_backend  # noqa: E402


def run(cell: dict, out_root: Path, quiet: bool = False) -> dict:
    cell = C.expand_env(cell)
    for key, value in (cell.get("env") or {}).items():
        os.environ[str(key)] = str(value)
    if not cell.get("keep_tokens"):
        C.scrub_tokens()
    out = out_root / cell["tag"]
    out.mkdir(parents = True, exist_ok = True)
    rec: dict = {"tag": cell["tag"], "backend": cell["backend"], "cell": cell, "renders": [], "long": [],
                 "short": [], "verdict": "running"}
    sampler = C.GpuSampler()
    sampler.start()
    backend = None
    say = (lambda m: None) if quiet else (lambda m: C.log(f"[{cell['tag']}] {m}"))
    try:
        cls = get_backend(cell["backend"])
        backend = cls(cell, out)
        sampler.extra_pids = backend.extra_pids
        rows = C.load_prompts(cell["prompts"], n = cell.get("n"), ids = cell.get("ids"))
        steps = int(cell["steps"])
        t0 = time.perf_counter()
        rec["status"] = backend.load() or {}
        rec["load_s"] = round(time.perf_counter() - t0, 2)
        rec["host_after_load"] = C.host_memory()
        say(f"loaded in {rec['load_s']}s: {json.dumps(rec['status'], default = str)[:400]}")

        def one(row: dict, n_steps: int) -> tuple:
            backend.reset_peak()
            s = time.perf_counter()
            result = backend.render(row, n_steps)
            backend.sync()
            wall = time.perf_counter() - s
            if result.peak_alloc_gib is None and backend.in_process:
                result.peak_alloc_gib = backend.peak_alloc_gib()
            return result, wall

        tw = time.perf_counter()
        for i in range(int(cell.get("warmup") or 0)):
            _, wall = one({**rows[0], "seed": rows[0]["seed"] + 1_000_003 + i}, steps)
            rec.setdefault("cold_s", round(wall, 3))
        rec["warmup_s"] = round(time.perf_counter() - tw, 2)
        sampler.reset()
        for row in rows:
            result, wall = one(row, steps)
            entry = {**row, "wall_s": round(wall, 3), "step_s": result.step_s, "peak_alloc_gib": result.peak_alloc_gib,
                     **({"extra": result.extra} if result.extra else {})}
            if cell["kind"] == "video":
                entry.update(C.save_video(result.frames, out / row["id"], fps = result.fps or cell.get("fps") or 16,
                                          every = int(cell.get("save_frames_every") or 1)))
            else:
                C.save_image(result.image, out / f"{row['id']}.png")
                entry["file"] = f"{row['id']}.png"
            rec["renders"].append(entry)
            say(f"{row['id']} {wall:.2f}s" + (f" step {result.step_s:.4f}s" if result.step_s else ""))
        short_steps = int(cell.get("short_steps") or 0)
        if short_steps and short_steps < steps:
            for row in rows[: int(cell.get("short_n") or 0)]:
                rec["long"].append(round(one(row, steps)[1], 3))
                rec["short"].append(round(one(row, short_steps)[1], 3))
        rec["status_after"] = backend.status() or {}
        rec["verdict"] = "ok"
    except Exception as exc:  # noqa: BLE001 - a broken cell is a result, recorded as such
        rec["verdict"] = "BROKEN"
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:2000]}"
        rec["traceback"] = traceback.format_exc()[-4000:]
        say(f"BROKEN {rec['error'][:300]}")
    finally:
        sampler.stop()
        rec["peak_smi_gib"] = round(sampler.peak_mib / 1024, 3)
        rec["smi_baseline_gib"] = round(sampler.baseline_mib / 1024, 3)
        # The driver view counts every process on the device; the delta over the pre-load reading is this cell's.
        rec["peak_smi_delta_gib"] = round(max(0, sampler.peak_mib - sampler.baseline_mib) / 1024, 3)
        # This process and the servers it launched only (NVIDIA): the figure to compare on a shared device.
        rec["peak_own_gib"] = round(sampler.own_peak_mib / 1024, 3)
        rec["peak_tree_rss_gib"] = round(sampler.tree_rss_peak_mib / 1024, 3)
        rec["host_end"] = C.host_memory()
        try:
            rec["env"] = C.env_fingerprint(backend.trees() if backend else None)
        except Exception as exc:  # noqa: BLE001
            rec["env"] = {"error": str(exc)}
        C.finalize_record(rec)
        C.write_record(out, rec)
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001
                say(f"close failed: {exc}")
    say(f"{rec['verdict']}: median {rec.get('wall_s_median')}s/{'clip' if cell['kind'] == 'video' else 'image'}, "
        f"s/step {rec.get('step_s_derived')}, peak {rec['peak_smi_gib']} GiB -> {out / C.RECORD}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", help = "cell JSON (merged over the built-in defaults)")
    ap.add_argument("--spec", help = "spec file; with --tag picks one of its cells")
    ap.add_argument("--tag")
    ap.add_argument("--set", action = "append", default = [], metavar = "KEY=JSON",
                    help = "override a cell field, e.g. --set n=2 --set options.speed_mode='\"off\"'")
    ap.add_argument("--out", required = True)
    args = ap.parse_args()
    if args.spec:
        spec = C.load_spec(args.spec)
        cells = {c["tag"]: c for c in spec["cells"]}
        if args.tag not in cells:
            ap.error(f"--tag must be one of {sorted(cells)}")
        cell = C.merge_cell(spec["defaults"], cells[args.tag])
    elif args.cell:
        cell = C.merge_cell({}, json.loads(args.cell))
    else:
        ap.error("--cell or --spec/--tag is required")
    for item in args.set:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        target = cell
        *path, leaf = key.split(".")
        for part in path:
            target = target.setdefault(part, {})
        target[leaf] = value
    rec = run(cell, Path(args.out))
    return 0 if rec["verdict"] == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())

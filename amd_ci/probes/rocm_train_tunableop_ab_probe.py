#!/usr/bin/env python3
"""Probe: Studio's own ROCm TunableOp training path (install_for_training), base vs head, on gfx1151.

Observes only; JSON to --out. base = origin/main, head = base + the patch (states.json). FLUX.2-klein-base-4B LoRA
through run_diffusion_training_process, Studio defaults (nf4 auto, 1024 px, rank 16), 20 steps, NO TunableOp env vars
(every PYTORCH_TUNABLEOP_* and UNSLOTH_DIFFUSION_ROCM_TUNABLEOP stripped), fresh process per arm, fp16 4096^3 matmul
sentinel before / after each arm. Per arm: the trainer's rocm_tunableop log lines, the live TunableOp row count and
enabled flag at step 5, install_for_training's return value. A ``tables`` stage per state checks which shipped table
matches this build, that a non-matching build loads nothing, and the solutions an inference pipe load reads.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

DATASET = "multimodalart/1920-raider-waite-tarot-public-domain"
BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
ORDER = ["base1", "head1", "base2", "head2", "base3", "headws", "headsnap"]
# Short extra head cells: (a) a capped hipBLASLt workspace must decline; (b) non-default pre-run settings come back.
EXTRA = {"headws": {"HIPBLASLT_WORKSPACE_SIZE": "0"}, "headsnap": {}}
EXTRA_STEPS = 6
TUNE_ENV = ("PYTORCH_TUNABLEOP_", "UNSLOTH_DIFFUSION_ROCM_TUNABLEOP", "HIPBLASLT_WORKSPACE_SIZE", "CUBLASLT_WORKSPACE_SIZE")


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent = 2, default = str), encoding = "utf-8")
    os.replace(tmp, path)


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding = "utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"_read_error": f"{type(exc).__name__}: {exc}"}


def _err(exc) -> str:
    return f"{type(exc).__name__}: {str(exc)[:800]}"


def _studio_env(backend: Path) -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    sys.path.insert(0, str(backend))
    os.chdir(backend)
    try:
        from utils.rocm_flash_attn import default_triton_amd_env

        default_triton_amd_env()
    except Exception:  # noqa: BLE001
        pass


class _Q:
    def __init__(self, sink = None):
        self.items: list = []
        self.sink = sink

    def put(self, item):
        self.items.append(item)
        if self.sink:
            self.sink(item)

    def empty(self):
        return True

    def get_nowait(self):
        raise RuntimeError("empty")


def _table_lines(tunable) -> tuple:
    results = list(tunable.get_results() or ())
    validators = list(tunable.get_validators() or ())
    lines = [f"Validator,{k},{v}" for k, v in validators] + [",".join(str(x) for x in r) for r in results]
    return lines, len(results)


def _dump_table(tunable, path: Path) -> int:
    lines, n = _table_lines(tunable)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding = "utf-8")
    os.replace(tmp, path)
    return n


# ----------------------------------------------------------------------------------------------------------- stages


def stage_sentinel(args) -> dict:
    import torch

    n, inner = 4096, 10
    a = torch.randn(n, n, dtype = torch.float16, device = "cuda")
    b = torch.randn(n, n, dtype = torch.float16, device = "cuda")
    for _ in range(10):
        a @ b
    torch.cuda.synchronize()
    tf = []
    for _ in range(9):
        s, e = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        s.record()
        for _ in range(inner):
            a @ b
        e.record()
        e.synchronize()
        tf.append(2 * n ** 3 * inner / (s.elapsed_time(e) / 1e3) / 1e12)
    res = {"tflops_median": round(statistics.median(tf), 3), "tflops_min": round(min(tf), 3),
           "tflops_max": round(max(tf), 3), "tunableop_enabled": torch.cuda.tunable.is_enabled()}
    try:
        sclk = Path("/sys/class/drm")
        vals = {}
        for f in sclk.glob("card*/device/pp_dpm_sclk"):
            vals[str(f)] = f.read_text(encoding = "utf-8").strip().replace("\n", " | ")
        res["pp_dpm_sclk"] = vals
    except Exception:  # noqa: BLE001
        pass
    return res


def stage_train(args) -> dict:
    import logging

    import torch

    from core.training import diffusion_lora_trainer as dlt

    arm = args.arm
    lines: list = []

    class _Grab(logging.Handler):
        def emit(self, record):
            try:
                msg = record.getMessage()
            except Exception:  # noqa: BLE001
                return
            if "tunableop" in msg.lower() or "tunableop" in record.name.lower():
                lines.append(f"{record.name}: {msg}"[:600])

    logging.getLogger().addHandler(_Grab())
    logging.getLogger().setLevel(logging.INFO)
    res: dict = {"arm": arm, "checkout": args.checkout,
                 "env_tunable": {k: v for k, v in os.environ.items() if k.startswith(TUNE_ENV)}}
    hook: dict = {"calls": []}
    try:
        from core.inference import diffusion_rocm_tunableop as rt

        res["has_install_for_training"] = hasattr(rt, "install_for_training")
        if hasattr(rt, "install_for_training"):
            real = rt.install_for_training

            def spy(*a, **k):
                r = real(*a, **k)
                hook["calls"].append({"returned": r, "written": rt._STATE.get("written"), "file": rt._STATE.get("file"),
                                      "active": rt._STATE.get("active")})
                return r

            rt.install_for_training = spy
    except Exception as exc:  # noqa: BLE001
        res["spy_error"] = _err(exc)
    stamps: dict = {}
    live: dict = {}
    events_path = Path(args.stage_out).with_suffix(".events.jsonl")
    ev_fh = open(events_path, "w", encoding = "utf-8")

    def sink(ev):
        t = time.time()
        ev_fh.write(json.dumps({**ev, "_t": t}, default = str) + "\n")
        ev_fh.flush()
        if ev.get("type") == "progress" and ev.get("step") is not None:
            step = int(ev["step"])
            stamps.setdefault(step, t)
            if step == 5 and not live:
                try:
                    live.update(rows = len(torch.cuda.tunable.get_results() or ()),
                                enabled = torch.cuda.tunable.is_enabled(),
                                tuning = torch.cuda.tunable.tuning_is_enabled())
                except Exception as exc:  # noqa: BLE001
                    live["error"] = _err(exc)
        if ev.get("type") not in ("preparing", "download_progress", "progress"):
            print("event", json.dumps(ev, default = str)[:600], flush = True)

    out_dir = Path(args.work) / "runs" / arm
    if out_dir.exists():
        shutil.rmtree(out_dir)
    config = {
        "base_model": BASE_MODEL, "data_dir": args.data_dir, "output_dir": str(out_dir),
        "train_steps": EXTRA_STEPS if arm in EXTRA else args.steps, "lora_rank": 16, "seed": 42,
        "adapter_name": f"amdci_ab_{arm}", "log_every": 1, "hf_token": None,
        "cond_cache_dir": str(Path(args.work) / "cond_cache"),
    }
    res["config"] = config
    tn = torch.cuda.tunable

    def settings():
        out = {}
        for k, g in (("enabled", "is_enabled"), ("tuning", "tuning_is_enabled"), ("filename", "get_filename"),
                     ("iterations", "get_max_tuning_iterations"), ("duration", "get_max_tuning_duration")):
            try:
                out[k] = getattr(tn, g)()
            except Exception as exc:  # noqa: BLE001
                out[k] = _err(exc)
        return out

    if arm == "headsnap":
        # Non-default values a caller could hold before training; release must hand exactly these back.
        tn.set_filename(str(Path(args.work) / "caller_owned_%d.csv"), insert_device_ordinal = False)
        tn.set_max_tuning_iterations(7)
        tn.set_max_tuning_duration(11)
        tn.tuning_enable(False)
        tn.enable(False)
    res["settings_before"] = settings()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    q_ev = _Q(sink)
    try:
        dlt.run_diffusion_training_process(event_queue = q_ev, stop_queue = _Q(), config = config)
    except Exception as exc:  # noqa: BLE001
        res["raised"] = _err(exc)
        res["traceback"] = traceback.format_exc()[-4000:]
    ev_fh.close()
    res["wall_s"] = round(time.time() - t0, 1)
    res["max_allocated_gib"] = round(torch.cuda.max_memory_allocated() / (1 << 30), 3)
    res["max_reserved_gib"] = round(torch.cuda.max_memory_reserved() / (1 << 30), 3)
    events = q_ev.items
    prog = [e for e in events if e.get("type") == "progress"]
    res["losses"] = [e.get("loss") for e in prog]
    res["steps_seen"] = len(prog)
    if prog:
        res["trainer_peak_memory_gb"] = max((e.get("peak_memory_gb") or 0) for e in prog)
    steps = sorted(stamps)
    dts = {s: stamps[s] - stamps[s - 1] for s in steps if s - 1 in stamps}
    res["step_s"] = {s: round(v, 4) for s, v in dts.items()}
    tail = [v for s, v in dts.items() if 5 <= s <= 20]
    res["s_per_step_median_5_20"] = round(statistics.median(tail), 4) if tail else None
    res["errors"] = [str(e.get("message") or e.get("error"))[:1500] for e in events if e.get("type") == "error"]
    loaded = [e for e in events if e.get("type") == "model_load_completed"]
    if loaded:
        res["model_load_completed"] = {k: v for k, v in loaded[-1].items() if k != "type"}
    res["complete"] = any(e.get("type") == "complete" for e in events)
    res["settings_after"] = settings()
    res["settings_restored"] = res["settings_after"] == res["settings_before"]
    res["install_for_training_calls"] = hook["calls"]
    res["tunable_at_step5"] = live
    res["tunableop_log_lines"] = lines
    try:
        res["tunable_after"] = {"rows": len(torch.cuda.tunable.get_results() or ()),
                                "enabled": torch.cuda.tunable.is_enabled()}
    except Exception as exc:  # noqa: BLE001
        res["tunable_after"] = _err(exc)
    return res


def stage_tables(args) -> dict:
    """Which shipped table this build matches; a non-matching build loads nothing; what an inference load reads."""
    import logging
    import types

    import torch
    from torch import nn

    from core.inference import diffusion_rocm_tunableop as rt

    res: dict = {"checkout": args.checkout}
    # Validators register on the first tunable GEMM: enable, prime as Studio's _activate does, read, switch back off.
    torch.cuda.tunable.enable(True)
    torch.cuda.tunable.tuning_enable(False)
    rt._prime(torch)
    real_v = dict(torch.cuda.tunable.get_validators() or ())
    torch.cuda.tunable.enable(False)
    res["validators"] = real_v
    arch = rt._arch()
    res["arch"] = arch
    res["eligible"] = rt.eligible()
    res["matching_tables"] = [p.name for p in rt.shipped_tables(arch, real_v)]
    bad = {**real_v, "HIPBLASLT_VERSION": "0-nonmatching"}
    res["matching_tables_nonmatching_build"] = [p.name for p in rt.shipped_tables(arch, bad)]
    res["all_tables"] = sorted(p.name for p in rt._TABLE_DIR.glob("*.csv"))
    lines: list = []

    class _Grab(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage()[:600])

    log = logging.getLogger("amdci.tables")
    log.addHandler(_Grab())
    log.setLevel(logging.INFO)
    # Inference: the pipe load path, read-only (env unset).
    pipe = types.SimpleNamespace(transformer = nn.Linear(4, 4), vae = None)
    res["install_for_pipe"] = rt.install_for_pipe(pipe, None, log)
    rows = [tuple(map(str, r)) for r in (torch.cuda.tunable.get_results() or ())]
    res["inference_rows"] = len(rows)
    res["inference_solutions"] = sorted("|".join(r[:3]) for r in rows)
    rt.release(pipe)
    res["inactive_after_release"] = not rt.is_active() and not torch.cuda.tunable.is_enabled()
    if hasattr(rt, "install_for_training"):
        res["train_install"] = rt.install_for_training(None, log)
        res["train_rows"] = len(torch.cuda.tunable.get_results() or ())
        rt.release_training()
        res["train_inactive_after_release"] = not rt.is_active() and not torch.cuda.tunable.is_enabled()
        # Non-matching build: the same call with torch's validators reporting a different hipBLASLt.
        orig = torch.cuda.tunable.get_validators
        torch.cuda.tunable.get_validators = lambda: tuple(bad.items())
        try:
            res["train_install_nonmatching_build"] = rt.install_for_training(None, log)
            res["train_enabled_nonmatching_build"] = torch.cuda.tunable.is_enabled()
        finally:
            torch.cuda.tunable.get_validators = orig
            if rt.is_active():
                rt.release_training()
    res["log_lines"] = lines
    return res


STAGES = {"sentinel": stage_sentinel, "train": stage_train, "tables": stage_tables}


def child_main(args) -> int:
    _studio_env(Path(args.checkout) / "studio" / "backend")
    out = Path(args.stage_out)
    try:
        res = STAGES[args.stage](args)
    except Exception as exc:  # noqa: BLE001
        res = {"stage_error": _err(exc), "traceback": traceback.format_exc()[-5000:]}
    _write(out, res)
    return 0


# ----------------------------------------------------------------------------------------------------------- parent


def fetch_dataset(dest: Path, n: int) -> dict:
    if (dest / "done").is_file():
        return _read(dest / "done")
    from huggingface_hub import hf_hub_download

    meta = hf_hub_download(DATASET, "output_file.jsonl", repo_type = "dataset", token = False)
    rows = [json.loads(line) for line in open(meta, encoding = "utf-8") if line.strip()]
    rows = sorted(rows, key = lambda r: r["file_name"])[:n]
    dest.mkdir(parents = True, exist_ok = True)
    for r in rows:
        p = hf_hub_download(DATASET, "train/" + r["file_name"], repo_type = "dataset", token = False)
        stem = Path(r["file_name"]).stem
        shutil.copy(p, dest / (stem + Path(r["file_name"]).suffix.lower()))
        (dest / (stem + ".txt")).write_text(r["text"], encoding = "utf-8")
    info = {"dataset": DATASET, "images": len(rows)}
    _write(dest / "done", info)
    return info


def run_child(args, stage: str, stage_out: Path, timeout_s: int, extra: list, env_add = None) -> dict:
    cmd = [sys.executable, "-u", __file__, "--child", "--stage", stage, "--checkout", args.checkout, "--out",
           args.out, "--stage-out", str(stage_out), "--work", args.work, "--data-dir", args.data_dir, "--steps",
           str(args.steps), "--table", args.table, *extra]
    env = {k: v for k, v in os.environ.items() if not k.startswith(TUNE_ENV)}
    env.update(env_add or {})
    log = stage_out.with_suffix(".log")
    t0 = time.time()
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, timeout = timeout_s, env = env).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    res = _read(stage_out) if stage_out.is_file() else {"missing_output": True}
    res["_rc"] = rc
    res["_wall_s"] = round(time.time() - t0, 1)
    if rc != 0 or "stage_error" in res or "missing_output" in res or res.get("errors") or res.get("raised"):
        try:
            res["_log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-3000:]
        except Exception:  # noqa: BLE001
            pass
    print(f"{stage} {extra} rc={rc} {res['_wall_s']}s", flush = True)
    return res


def _count_lines(path: Path) -> int:
    try:
        return sum(1 for ln in path.read_text(encoding = "utf-8").splitlines()
                   if ln.strip() and not ln.startswith("Validator"))
    except Exception:  # noqa: BLE001
        return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default = "")
    ap.add_argument("--checkout", default = "")
    ap.add_argument("--out", required = True)
    ap.add_argument("--steps", type = int, default = 20)
    ap.add_argument("--images", type = int, default = 8)
    ap.add_argument("--work", default = "")
    ap.add_argument("--data-dir", default = "")
    ap.add_argument("--table", default = "")
    ap.add_argument("--child", action = "store_true")
    ap.add_argument("--stage", default = "")
    ap.add_argument("--stage-out", default = "")
    ap.add_argument("--arm", default = "")
    args = ap.parse_args()
    if args.child:
        return child_main(args)

    out = Path(args.out).resolve()
    args.out = str(out)
    paths = json.loads(Path(args.states).read_text(encoding = "utf-8"))["paths"]
    work = Path(args.work or (Path(os.environ.get("AMD_CI_WORK", out.parent)) / "tune_ab")).resolve()
    args.work = str(work)
    args.table = str(work / "unused.csv")
    work.mkdir(parents = True, exist_ok = True)
    os.environ.setdefault("HF_HOME", str(work / "hf"))
    os.environ.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")
    os.environ.pop("HF_TOKEN", None)
    for k in [k for k in os.environ if k.startswith(TUNE_ENV)]:
        os.environ.pop(k)
    obs: dict = {"paths": paths, "steps": args.steps, "base_model": BASE_MODEL, "order": ORDER, "arms": {},
                 "sentinels": [], "tables": {}}
    stage_dir = out.parent / "stages"
    stage_dir.mkdir(parents = True, exist_ok = True)
    try:
        obs["dataset"] = fetch_dataset(work / "data", args.images)
    except Exception as exc:  # noqa: BLE001
        obs["dataset_error"] = _err(exc)
    args.data_dir = str(work / "data")
    for state in ("base", "head"):
        args.checkout = str(Path(paths[state]).resolve())
        obs["tables"][state] = run_child(args, "tables", stage_dir / f"tables_{state}.json", 600, [])
    _write(out, obs)
    k = 0
    for arm in ORDER:
        args.checkout = str(Path(paths["head" if arm.startswith("head") else "base"]).resolve())
        obs["sentinels"].append({"before": arm, **run_child(args, "sentinel", stage_dir / f"sent_{k}.json", 300, [])})
        k += 1
        obs["arms"][arm] = run_child(args, "train", stage_dir / f"train_{arm}.json", 40 * 60, ["--arm", arm],
                                     env_add = EXTRA.get(arm))
        _write(out, obs)
        obs["sentinels"].append({"after": arm, **run_child(args, "sentinel", stage_dir / f"sent_{k}.json", 300, [])})
        k += 1
        _write(out, obs)
    obs["finished"] = True
    _write(out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

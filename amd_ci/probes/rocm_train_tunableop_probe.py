#!/usr/bin/env python3
"""Probe: does a ROCm TunableOp GEMM table speed up Studio diffusion LoRA training on gfx1151?

Observes only. Writes JSON to --out, never stdout. One Studio checkout (unsloth main), FLUX.2-klein-base-4B, rank 16,
Studio defaults otherwise (resolution 1024, compile auto), through run_diffusion_training_process. Fresh process per
arm; an fp16 4096^3 matmul TFLOP/s sentinel (its own process, TunableOp env stripped) before and after every arm.

  A*     main as-is, 20 steps (A1, A2, A3 = the A/A spread)
  Btune  PYTORCH_TUNABLEOP_ENABLED=1, tuning switched on after step 1's progress event and off after step 3's
         (steps 2-3 tune every GEMM the training step runs; the one-off text-encode / latent-cache GEMMs are not
         tuned), Studio's inference limits (30 ms / 20 iters). The table is written by a thread every 60 s and at
         the end, so a timeout keeps a partial table.
  B*     replay: ENABLED=1, TUNING=0, the Btune table (copied fresh per arm), RECORD_UNTUNED=1 to count misses.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

DATASET = "multimodalart/1920-raider-waite-tarot-public-domain"
BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
ORDER = ["A1", "A2", "Btune", "B1", "A3", "B2"]
TUNE_ENV = ("PYTORCH_TUNABLEOP_",)


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
    import torch

    from core.training import diffusion_lora_trainer as dlt

    arm = args.arm
    tune = arm == "Btune"
    table = Path(args.table)
    res: dict = {"arm": arm, "env": {k: v for k, v in os.environ.items() if k.startswith(TUNE_ENV)}}
    stamps: dict = {}
    tl: list = []
    tune_t: dict = {}
    stop_writer = threading.Event()
    lock = threading.Lock()

    def writer():
        while not stop_writer.wait(60):
            try:
                with lock:
                    n = _dump_table(torch.cuda.tunable, table)
                tl.append({"t": round(time.time() - t0, 1), "rows": n})
            except Exception as exc:  # noqa: BLE001
                tl.append({"t": round(time.time() - t0, 1), "error": _err(exc)})

    events_path = Path(args.stage_out).with_suffix(".events.jsonl")
    ev_fh = open(events_path, "w", encoding = "utf-8")

    def sink(ev):
        t = time.time()
        ev_fh.write(json.dumps({**ev, "_t": t}, default = str) + "\n")
        ev_fh.flush()
        if ev.get("type") == "progress" and ev.get("step") is not None:
            step = int(ev["step"])
            stamps.setdefault(step, t)
            if tune and step == 1 and "on" not in tune_t:
                torch.cuda.tunable.tuning_enable(True)
                tune_t["on"] = t
            elif tune and step == 3 and "off" not in tune_t:
                torch.cuda.tunable.tuning_enable(False)
                tune_t["off"] = t
        if ev.get("type") not in ("preparing", "download_progress", "progress"):
            print("event", json.dumps(ev, default = str)[:600], flush = True)

    out_dir = Path(args.work) / "runs" / arm
    if out_dir.exists():
        shutil.rmtree(out_dir)
    config = {
        "base_model": BASE_MODEL, "data_dir": args.data_dir, "output_dir": str(out_dir),
        "train_steps": 4 if tune else args.steps, "lora_rank": 16, "seed": 42,
        "adapter_name": f"amdci_tune_{arm}", "log_every": 1, "hf_token": None,
        "cond_cache_dir": str(Path(args.work) / "cond_cache"),
    }
    res["config"] = config
    res["tunable_enabled_at_start"] = torch.cuda.tunable.is_enabled()
    fn = os.environ.get("PYTORCH_TUNABLEOP_FILENAME")
    if arm.startswith("B") and not tune and fn:
        try:
            x = torch.ones(8, 8, dtype = torch.bfloat16, device = "cuda")
            (x @ x).sum().item()
            res["read_file_ok"] = bool(torch.cuda.tunable.read_file(fn))
        except Exception as exc:  # noqa: BLE001
            res["read_file_ok"] = _err(exc)
        res["tunable_rows_at_start"] = len(torch.cuda.tunable.get_results() or ())
    if tune:
        torch.cuda.tunable.set_max_tuning_duration(30)
        torch.cuda.tunable.set_max_tuning_iterations(20)
        torch.cuda.tunable.tuning_enable(False)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    th = None
    if tune:
        th = threading.Thread(target = writer, daemon = True)
        th.start()
    q_ev = _Q(sink)
    try:
        dlt.run_diffusion_training_process(event_queue = q_ev, stop_queue = _Q(), config = config)
    except Exception as exc:  # noqa: BLE001
        res["raised"] = _err(exc)
        res["traceback"] = traceback.format_exc()[-4000:]
    stop_writer.set()
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
    res["warnings"] = [str(e.get("message"))[:600] for e in events if e.get("type") == "warning"]
    loaded = [e for e in events if e.get("type") == "model_load_completed"]
    if loaded:
        res["model_load_completed"] = {k: v for k, v in loaded[-1].items() if k != "type"}
    res["complete"] = any(e.get("type") == "complete" for e in events)
    try:
        res["tunable_results_rows"] = len(torch.cuda.tunable.get_results() or ())
        res["tunable_validators"] = dict(torch.cuda.tunable.get_validators() or ())
        res["tunable_enabled_at_end"] = torch.cuda.tunable.is_enabled()
    except Exception as exc:  # noqa: BLE001
        res["tunable_err"] = _err(exc)
    if tune:
        res["tune_window_s"] = round(tune_t["off"] - tune_t["on"], 1) if "on" in tune_t and "off" in tune_t else None
        res["tune_on"] = "on" in tune_t
        res["table_timeline"] = tl
        try:
            with lock:
                res["table_rows_written"] = _dump_table(torch.cuda.tunable, table)
        except Exception as exc:  # noqa: BLE001
            res["table_write_error"] = _err(exc)
    try:
        from torch._dynamo.utils import counters

        res["dynamo"] = {"graph_breaks": sum(counters["graph_break"].values()),
                         "unique_graphs": counters["stats"].get("unique_graphs")}
    except Exception as exc:  # noqa: BLE001
        res["dynamo"] = _err(exc)
    return res


STAGES = {"sentinel": stage_sentinel, "train": stage_train}


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
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--steps", type = int, default = 20)
    ap.add_argument("--images", type = int, default = 8)
    ap.add_argument("--tune-timeout-min", type = float, default = 100.0)
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
    args.checkout = str(Path(args.checkout).resolve())
    work = Path(args.work or (Path(os.environ.get("AMD_CI_WORK", out.parent)) / "tune_probe")).resolve()
    args.work = str(work)
    work.mkdir(parents = True, exist_ok = True)
    master = work / "tunableop_train_table.csv"
    args.table = str(master)
    os.environ.setdefault("HF_HOME", str(work / "hf"))
    os.environ.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")
    os.environ.pop("HF_TOKEN", None)
    obs: dict = {"checkout": args.checkout, "steps": args.steps, "base_model": BASE_MODEL, "order": ORDER,
                 "arms": {}, "sentinels": []}
    stage_dir = out.parent / "stages"
    stage_dir.mkdir(parents = True, exist_ok = True)
    try:
        obs["dataset"] = fetch_dataset(work / "data", args.images)
    except Exception as exc:  # noqa: BLE001
        obs["dataset_error"] = _err(exc)
    args.data_dir = str(work / "data")
    _write(out, obs)
    k = 0
    for arm in ORDER:
        obs["sentinels"].append({"before": arm, **run_child(args, "sentinel", stage_dir / f"sent_{k}.json", 300, [])})
        k += 1
        env_add: dict = {}
        timeout = 40 * 60
        if arm == "Btune":
            env_add = {"PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "0",
                       "PYTORCH_TUNABLEOP_FILENAME": str(work / "torch_exit_write.csv")}
            timeout = int(args.tune_timeout_min * 60)
        elif arm.startswith("B"):
            if not master.is_file():
                obs["arms"][arm] = {"skipped": "no tuned table"}
                continue
            copy = work / f"table_{arm}.csv"
            shutil.copy(master, copy)
            env_add = {"PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "0",
                       "PYTORCH_TUNABLEOP_FILENAME": str(copy), "PYTORCH_TUNABLEOP_RECORD_UNTUNED": "1",
                       "PYTORCH_TUNABLEOP_UNTUNED_FILENAME": str(work / f"untuned_{arm}.csv")}
        r = run_child(args, "train", stage_dir / f"train_{arm}.json", timeout, ["--arm", arm], env_add = env_add)
        if arm.startswith("B") and arm != "Btune":
            r["table_rows_in"] = _count_lines(master)
            unt = list(work.glob(f"untuned_{arm}*"))
            r["untuned_files"] = {p.name: _count_lines(p) for p in unt}
        if arm == "Btune":
            r["master_rows"] = _count_lines(master)
        obs["arms"][arm] = r
        _write(out, obs)
        obs["sentinels"].append({"after": arm, **run_child(args, "sentinel", stage_dir / f"sent_{k}.json", 300, [])})
        k += 1
        _write(out, obs)
    if master.is_file():
        shutil.copy(master, out.parent / "tunableop_train_table.csv")
    obs["finished"] = True
    _write(out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

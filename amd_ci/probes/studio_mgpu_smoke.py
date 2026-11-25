"""Drive Studio's real training worker (run_training_process) like the parent does, headless.

python scripts/studio_mgpu_smoke.py --backend-dir <wt>/studio/backend --mode ddp --world 2 --out out.json
mode single = today's path (no parallel field); ddp / fsdp2 / ... = multi-process plan with that spec.
Stop test: --stop-at-step N sends the UI stop at step N.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend-dir", required = True)
    ap.add_argument("--model", default = "trl-internal-testing/tiny-Qwen3ForCausalLM")
    ap.add_argument("--mode", default = "single")
    ap.add_argument("--spec", default = None)
    ap.add_argument("--world", type = int, default = 1)
    ap.add_argument("--gpu-ids", default = None)
    ap.add_argument("--steps", type = int, default = 6)
    ap.add_argument("--batch", type = int, default = 2)
    ap.add_argument("--ga", type = int, default = 1)
    ap.add_argument("--seq", type = int, default = 256)
    ap.add_argument("--lora-r", type = int, default = 8)
    ap.add_argument("--load-in-4bit", action = "store_true")
    ap.add_argument("--full", action = "store_true")
    ap.add_argument("--dataset", default = None, help = "jsonl; default: generated alpaca rows")
    ap.add_argument("--hf-dataset", default = None)
    ap.add_argument("--rows", type = int, default = 64)
    ap.add_argument("--stop-at-step", type = int, default = 0)
    ap.add_argument("--cancel-at-step", type = int, default = 0)
    ap.add_argument("--kill-rank1-at-step", type = int, default = 0)
    ap.add_argument("--output-dir", default = None)
    ap.add_argument("--out", default = None)
    ap.add_argument("--timeout", type = int, default = 1800)
    args = ap.parse_args()

    if args.out:
        args.out = os.path.abspath(args.out)
    sys.path.insert(0, args.backend_dir)
    os.chdir(args.backend_dir)
    from core.training.training import _build_training_worker_config
    from core.training.worker import run_training_process

    work = os.path.abspath(args.output_dir or f"/tmp/studio_mgpu_{os.getpid()}")
    os.makedirs(work, exist_ok = True)
    data = args.dataset
    if not data and not args.hf_dataset:
        data = os.path.join(work, "train.jsonl")
        with open(data, "w") as f:
            for i in range(args.rows):
                f.write(json.dumps({
                    "instruction": f"Add {i} and {i + 1}.",
                    "input": "",
                    "output": f"The sum of {i} and {i + 1} is {2 * i + 1}. " * 4,
                }) + "\n")
    values = {
        "model_name": args.model,
        "training_type": "Full Finetuning" if args.full else "LoRA/QLoRA",
        "load_in_4bit": bool(args.load_in_4bit),
        "max_seq_length": args.seq,
        "local_datasets": [data] if data else None,
        "hf_dataset": args.hf_dataset or "",
        "format_type": "alpaca" if data else "",
        "max_steps": args.steps,
        "batch_size": args.batch,
        "gradient_accumulation_steps": args.ga,
        "warmup_steps": 0,
        "learning_rate": "2e-4",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_r * 2,
        "random_seed": 3407,
        "packing": False,
        "output_dir": os.path.join(os.environ.get("UNSLOTH_STUDIO_HOME", os.path.expanduser("~/.unsloth/studio")), "outputs", os.path.basename(work.rstrip("/"))),
        "save_steps": 0,
        "optim": "adamw_torch",
        "allow_external_output_dir": True,
    }
    config = _build_training_worker_config(values)
    gpu_ids = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids else None
    config["resolved_gpu_ids"] = gpu_ids
    if args.mode != "single":
        spec = args.spec or {"ddp": f"dp={args.world}", "fsdp2": f"dp={args.world},fsdp=1"}.get(args.mode, f"{args.mode}={args.world}")
        config["parallel"] = {
            "requested_mode": args.mode,
            "mode": args.mode,
            "world_size": args.world,
            "gpu_ids": gpu_ids,
            "spec": spec,
            "launcher": "multi_process",
            "distributed_backend": "nccl",
            "reason": "smoke",
        }
    ctx = mp.get_context("spawn")
    events, stops = ctx.Queue(), ctx.Queue()
    proc = ctx.Process(target = run_training_process, kwargs = {"event_queue": events, "stop_queue": stops, "config": config}, daemon = False)
    t0 = time.time()
    proc.start()
    losses, step_times, last = {}, [], None
    result = {"mode": args.mode, "world": args.world, "status": "timeout", "events": []}
    sent = False
    while time.time() - t0 < args.timeout:
        try:
            ev = events.get(timeout = 2)
        except Exception:
            if not proc.is_alive():
                result["status"] = result.get("status") if result["status"] != "timeout" else f"worker_exit_{proc.exitcode}"
                break
            continue
        kind = ev.get("type")
        if kind not in ("progress",):
            result["events"].append({k: v for k, v in ev.items() if k in ("type", "error", "message", "output_dir")})
        if kind == "progress":
            step = ev.get("step")
            loss = ev.get("loss")
            if step and loss is not None and step not in losses:
                losses[step] = loss
                now = time.time()
                if last is not None:
                    step_times.append(now - last)
                last = now
                print(f"step {step} loss {loss}", flush = True)
                if args.stop_at_step and step >= args.stop_at_step and not sent:
                    stops.put({"type": "stop", "save": True})
                    sent = True
                if args.kill_rank1_at_step and step >= args.kill_rank1_at_step and not sent:
                    import psutil

                    sent = True
                    kids = [k for k in psutil.Process(proc.pid).children() if "rank_main" in " ".join(k.cmdline())]
                    result["killed_rank1"] = [k.pid for k in kids]
                    for k in kids:
                        k.kill()
                if args.cancel_at_step and step >= args.cancel_at_step and not sent:
                    sent = True
                    import psutil

                    kids = psutil.Process(proc.pid).children(recursive = True)
                    result["ranks_before_cancel"] = len(kids)
                    proc.kill()
                    time.sleep(5)
                    result["alive_after_cancel"] = [k.pid for k in kids if k.is_running() and k.status() != psutil.STATUS_ZOMBIE]
                    result["status"] = "cancelled"
                    break
        elif kind in ("complete", "error"):
            result["status"] = kind
            if kind == "error":
                print("ERROR:", ev.get("error"), flush = True)
                print(ev.get("stack", ""), flush = True)
            break
    proc.join(timeout = 60)
    result["losses"] = losses
    result["median_step_s"] = sorted(step_times)[len(step_times) // 2] if step_times else None
    result["wall_s"] = round(time.time() - t0, 1)
    result["output_dir"] = values["output_dir"]
    result["saved"] = sorted(os.listdir(values["output_dir"])) if os.path.isdir(values["output_dir"]) else []
    print(json.dumps(result, indent = 1, default = str), flush = True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent = 1, default = str)
    sys.exit(0 if result["status"] in ("complete", "cancelled") else 1)


if __name__ == "__main__":
    main()

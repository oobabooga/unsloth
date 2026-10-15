#!/usr/bin/env python3
"""Probe: does this checkout's Studio planner put a load on a ROCm ggml-rpc-server worker?

Observes only. A worker rpc-server runs with the GPU visible; the coordinator (Studio's planner and
the llama-server it launches) runs with the GPU hidden, so every accelerator it uses is a remote
one. Records the planned args, where each layer's buffer went, and a greedy completion.

Pairs with criteria/cluster_rpc_worker.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

PORT_RPC, PORT_SERVER = 50652, 50653
HIDE = {"HIP_VISIBLE_DEVICES": "-1", "ROCR_VISIBLE_DEVICES": "-1", "GGML_VK_VISIBLE_DEVICES": ""}


def model_path() -> str:
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    root = Path(os.environ.get("RUNNER_TEMP") or "/tmp") / "cluster_models"
    return hf_hub_download("unsloth/gemma-3-270m-it-GGUF", "gemma-3-270m-it-Q8_0.gguf", local_dir = str(root))


def wait_port(port: int, seconds: float) -> bool:
    import socket  # noqa: PLC0415

    end = time.time() + seconds
    while time.time() < end:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout = 1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    work = args.out.parent / f"cluster_{args.state}"
    work.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state}
    procs = []
    try:
        sys.path.insert(0, str(args.checkout / "studio" / "backend"))
        from core.inference import spark_serving as ss  # noqa: PLC0415

        try:
            from core.inference import cluster_serving  # noqa: F401, PLC0415

            obs["has_cluster_module"] = True
        except ImportError:
            obs["has_cluster_module"] = False
        server = ss.llama_server_binary()
        rpc = ss.rpc_server_binary()
        obs["llama_server"], obs["rpc_server"] = server, rpc
        if not server or not rpc:
            raise RuntimeError("Studio's llama.cpp install has no llama-server / ggml-rpc-server")
        model = model_path()
        obs["model"] = os.path.basename(model)

        procs.append(
            subprocess.Popen(
                [rpc, "-H", "127.0.0.1", "-p", str(PORT_RPC)],
                stdout = open(work / "rpc-server.log", "w", encoding = "utf-8"),
                stderr = subprocess.STDOUT,
                preexec_fn = os.setsid,
            )
        )
        obs["rpc_up"] = wait_port(PORT_RPC, 60)
        # The coordinator sees only what it reaches over RPC.
        os.environ.update(HIDE)
        os.environ.pop("UNSLOTH_SPARK_SERVING", None)
        os.environ["UNSLOTH_CLUSTER_NODES"] = f"127.0.0.1:{PORT_RPC}"
        os.environ["UNSLOTH_CLUSTER_TOPOLOGY"] = "layer_split"
        listing = subprocess.run(
            [server, "--rpc", f"127.0.0.1:{PORT_RPC}", "--list-devices"],
            capture_output = True, text = True, timeout = 120,
        )
        obs["coordinator_devices"] = [l.strip() for l in listing.stdout.splitlines() if ":" in l and "MiB" in l]
        obs["worker_log_head"] = (work / "rpc-server.log").read_text(encoding = "utf-8", errors = "replace")[:1500]

        request = SimpleNamespace(
            model_path = model, gguf_variant = None, llama_extra_args = [], max_seq_length = 2048,
            cache_type_kv = None, disable_vision = False,
        )
        planned = asyncio.run(ss.before_load(request, 1))
        obs["planned_args"] = list(getattr(planned, "llama_extra_args", None) or [])
        status = ss.status()
        obs["topology"], obs["reason"] = status.get("topology"), status.get("reason")

        argv = [server, "-m", model, "--port", str(PORT_SERVER), "-c", "2048", "-lv", "4"] + obs["planned_args"]
        log = work / "llama-server.log"
        t0 = time.time()
        procs.append(subprocess.Popen(argv, stdout = open(log, "w", encoding = "utf-8"), stderr = subprocess.STDOUT, preexec_fn = os.setsid))
        healthy = False
        while time.time() - t0 < 600 and procs[-1].poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT_SERVER}/health", timeout = 2) as r:
                    healthy = r.status == 200
                    if healthy:
                        break
            except Exception:
                time.sleep(1)
        obs["healthy"], obs["load_s"] = healthy, round(time.time() - t0, 1)
        if healthy:
            body = json.dumps({"prompt": "The capital of France is", "n_predict": 16, "temperature": 0}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT_SERVER}/completion", data = body, headers = {"Content-Type": "application/json"}
            )
            res = json.load(urllib.request.urlopen(req, timeout = 300))
            obs["text"] = res.get("content")
            obs["tok_s"] = (res.get("timings") or {}).get("predicted_per_second")
        text = log.read_text(encoding = "utf-8", errors = "replace")
        obs["buffers"] = re.findall(r"load_tensors:\s+(\S+) model buffer size =\s+([\d.]+) MiB", text)
        m = re.search(r"offloaded (\d+)/(\d+) layers", text)
        obs["offloaded"] = [int(m.group(1)), int(m.group(2))] if m else None
        if not healthy:
            obs["server_log_tail"] = text[-2000:]
    except Exception as e:  # noqa: BLE001
        obs["error"] = f"{type(e).__name__}: {e}"
    finally:
        for p in reversed(procs):
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                p.wait(timeout = 30)
            except Exception:
                pass
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

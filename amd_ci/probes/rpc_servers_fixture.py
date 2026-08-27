#!/usr/bin/env python3
"""Fixture: N ggml-rpc-server processes, all backed by the SAME physical GPU.

Why this exists. Every reporter of the Qwen3.8-27B crash who gave a full
configuration is on two GPUs, and this runner has one. `ggml-rpc-server -d
<device>` lets several servers share one device, and llama-server then sees
`RPC0`, `RPC1`, ... as distinct backends and splits layers across them through
exactly the multi-device code path a real dual-GPU box uses:
`ggml_backend_sched` with several backends, `-sm layer` / `-sm row`, `-ts`, and
cross-device KV placement. That is the path llama.cpp#24492 blames for the
"pre-allocated tensor ... cannot run the operation (NONE)" abort.

It is a spoof and the report must say so: one physical device means no PCIe
transfer between GPUs, no second driver context, no per-card VRAM ceiling, and
no NUMA effects. What it does reproduce is the multi-backend SCHEDULING that the
reports point at.

A second reason to want RPC specifically: llama.cpp#24177 is an RPC bug.
`ggml_backend_rpc_device_supports_op()` returns true unconditionally, so the
TOP_K guard that makes single-GPU ROCm fall back to the CPU is bypassed and the
oversized argsort aborts instead. If that fires here it fires as itself.

A fixture must hold for EVERY state's probe or the states are not comparable,
which is why this is a fixture rather than something a probe starts. It prints
one JSON line containing "status": "READY" when every port accepts, and the
runner polls for that rather than sleeping.

SAFETY. These processes allocate GPU memory on a shared, persistent host, and a
server that outlives its job holds VRAM against everyone else. So: children are
started in this process's own group, every signal path stops them by PID, and
`--max-seconds` is a hard self-destruct so an abandoned fixture cannot outlive
the job even if the runner never signals it. Never `pkill -f`: the pattern
matches the shell that runs it.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

PROCS: list[subprocess.Popen] = []


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(2)
        return s.connect_ex((host, port)) == 0


def stop_all() -> None:
    """By PID, always, and escalate rather than trusting terminate()."""
    for p in PROCS:
        if p.poll() is None:
            p.terminate()
    deadline = time.time() + 20
    for p in PROCS:
        while p.poll() is None and time.time() < deadline:
            time.sleep(0.2)
    for p in PROCS:
        if p.poll() is None:
            p.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required = True, type = Path)
    ap.add_argument("--device", default = "Vulkan0",
                    help = "device each server binds; the same one for all of them, "
                           "which is the point")
    ap.add_argument("--port", action = "append", type = int, default = [],
                    help = "repeatable; fixed ports, because the probe's --rpc "
                           "argument is fixed when the differential is invoked")
    ap.add_argument("--log-dir", required = True, type = Path)
    ap.add_argument("--ready-timeout", type = int, default = 300)
    ap.add_argument("--max-seconds", type = int, default = 5400,
                    help = "hard self-destruct; a fixture that outlives its job would "
                           "hold VRAM on a shared host")
    args = ap.parse_args()

    ports = args.port or [50152, 50153]
    args.log_dir.mkdir(parents = True, exist_ok = True)

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(args.bin_dir), env.get("LD_LIBRARY_PATH", "")]).strip(os.pathsep)

    atexit.register(stop_all)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: (stop_all(), sys.exit(0)))

    binary = args.bin_dir / "ggml-rpc-server"
    if not binary.is_file():
        print(json.dumps({"status": "FAILED", "error": f"no ggml-rpc-server at {binary}"}),
              flush = True)
        return 1

    for port in ports:
        log = args.log_dir / f"rpc_{port}.log"
        fh = open(log, "w")
        PROCS.append(subprocess.Popen(
            [str(binary), "-H", "127.0.0.1", "-p", str(port), "-d", args.device],
            stdout = fh, stderr = subprocess.STDOUT, env = env))

    deadline = time.time() + args.ready_timeout
    ready: list[int] = []
    while time.time() < deadline and len(ready) < len(ports):
        dead = [p.pid for p in PROCS if p.poll() is not None]
        if dead:
            print(json.dumps({"status": "FAILED", "error": "a server exited early",
                              "pids": dead}), flush = True)
            stop_all()
            return 1
        ready = [p for p in ports if port_open(p)]
        if len(ready) < len(ports):
            time.sleep(2)

    if len(ready) < len(ports):
        print(json.dumps({"status": "FAILED", "error": "not every port opened",
                          "ready": ready, "wanted": ports}), flush = True)
        stop_all()
        return 1

    print(json.dumps({
        "status": "READY",
        "ports": ports,
        "pids": [p.pid for p in PROCS],
        "device": args.device,
        "rpc_arg": ",".join(f"127.0.0.1:{p}" for p in ports),
        # Recorded so the report cannot quietly claim two GPUs.
        "spoofed": True,
        "note": ("several RPC servers share ONE physical device; this reproduces "
                 "multi-backend scheduling, not two cards"),
    }), flush = True)

    end = time.time() + args.max_seconds
    while time.time() < end:
        if any(p.poll() is not None for p in PROCS):
            print(json.dumps({"status": "DEGRADED",
                              "error": "a server died while the fixture was up"}), flush = True)
            break
        time.sleep(5)
    stop_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())

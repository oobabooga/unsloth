#!/usr/bin/env python3
"""Probe (Windows gfx1151): llama-server working set with every layer on the GPU and
--load-mode mmap, for Unsloth's b12347 ROCm bundle with three llama.dll builds:
  orig  the bundle's own llama.dll
  base  llama.dll rebuilt from the bundle's source (toolchain control for the swap)
  head  the same plus unslothai/llama.cpp#376
Arms interleave inside one process (orig, base, head, base, head, then --load-mode none for base
and head); each state's criteria read its own arm. Writes JSON to --out."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from win_igpu_fit_probe import LLAMA_URL, MODEL_URL, fetch, find_server, gpu_counters, procmem, sysmem  # noqa: E402

DLLS = Path(__file__).resolve().parent.parent / "llama_dlls"
BASE = ["-c", "8192", "--parallel", "1", "-fa", "on", "--fit", "on"]


def run(server: Path, model: Path, arm: str, mode: str, log: Path, port: int) -> dict:
    cmd = [str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(port), *BASE,
           "--load-mode", mode]
    res: dict = {"arm": arm, "mode": mode, "before": sysmem()}
    with open(log, "wb") as fh:
        p = subprocess.Popen(cmd, stdout = fh, stderr = subprocess.STDOUT, cwd = str(server.parent))
    status, t0 = "timeout", time.time()
    while time.time() - t0 < 600:
        if p.poll() is not None:
            status = f"exited {p.returncode}"
            break
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout = 5) as r:
                if r.status == 200:
                    status = "ready"
                    break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    res["status"] = status
    if status == "ready":
        body = json.dumps({"prompt": "The capital of France is", "n_predict": 24, "temperature": 0,
                           "seed": 1}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data = body,
                                     headers = {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout = 600) as r:
            res["text"] = json.loads(r.read())["content"]
        time.sleep(10)
        res["proc"] = procmem(p.pid)
        res["sys"] = sysmem()
        res["gpu"] = [ln for ln in gpu_counters(p.pid).splitlines() if "process" in ln]
    subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output = True)
    try:
        p.wait(timeout = 60)
    except Exception:  # noqa: BLE001
        pass
    text = log.read_text(encoding = "utf-8", errors = "replace")
    res["log"] = [ln for ln in text.splitlines()
                  if any(k in ln for k in ("offloaded", "ROCm0 model buffer", "VirtualUnlock", "Device 0:"))][:12]
    res["warnings"] = [ln for ln in text.splitlines() if " W " in ln or "warning" in ln.lower()][:12]
    time.sleep(10)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    a = ap.parse_args()
    import os

    work = Path(os.environ.get("AMD_CI_WORK") or os.environ.get("RUNNER_TEMP") or ".") / "winmmap_cache"
    work.mkdir(parents = True, exist_ok = True)
    out: dict = {"state": a.state}
    try:
        zpath = work / "llama.zip"
        fetch(LLAMA_URL, zpath)
        model = work / "Qwen3-8B-Q4_K_M.gguf"
        fetch(MODEL_URL, model)
        servers = {}
        for arm in ("orig", "base", "head"):
            d = work / f"bundle_{arm}"
            if not d.is_dir():
                zipfile.ZipFile(zpath).extractall(d)
            srv = find_server(d)
            if arm != "orig":
                shutil.copyfile(DLLS / arm / "llama.dll", srv.parent / "llama.dll")
            servers[arm] = srv
        out["dll_sizes"] = {arm: (s.parent / "llama.dll").stat().st_size for arm, s in servers.items()}
        out["model_mib"] = model.stat().st_size // (1024 * 1024)
        order = [("orig", "mmap"), ("base", "mmap"), ("head", "mmap"), ("base", "mmap"),
                 ("head", "mmap"), ("base", "none"), ("head", "none")]
        logdir = a.out.parent / f"llama_logs_{a.state}"
        logdir.mkdir(parents = True, exist_ok = True)
        out["runs"] = [run(servers[arm], model, arm, mode, logdir / f"{i}_{arm}_{mode}.log", 18400 + i)
                       for i, (arm, mode) in enumerate(order)]
    except BaseException as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    a.out.write_text(json.dumps(out, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

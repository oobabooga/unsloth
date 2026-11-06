#!/usr/bin/env python3
"""PR 16857 argv cells on a real gfx1151 iGPU. Observes only; the report judges.

Per backend (a published unslothai/llama.cpp bundle dir), per cell, one fresh
llama-server: -m MODEL -c 4096 --fit on + the cell's flags, wait for /health,
one /completion (n_predict 16), read peak RSS / peak working set, stop, keep the
whole log. Cells:
  A  base argv  (no load / lazy flags)
  B  head argv  (--load-mode mmap --lazy-mode on, + --fit-target 16384 on Windows)
  C  control    (--load-mode mmap alone)
Optional Linux ballast cells (A_ballast, B_ballast): a separate process holds
touched host RAM so MemAvailable sits below the model file size, then A and B rerun.
Writes <out>/cells.jsonl and <out>/logs/*.log. Stdlib only (Linux python3, Windows Python312).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WIN = os.name == "nt"
EXE = "llama-server.exe" if WIN else "llama-server"

PATTERNS = [
    r"load_mode =", r"lazy", r"model buffer size", r"CPU_Mapped", r"system_info",
    r"ggml_vulkan: \d", r"ggml_cuda_init", r"Device \d:", r"using device", r"llama_params_fit",
    r"fit", r"mmap", r"error", r"failed", r"out of memory", r"KV buffer size",
    r"compute buffer size", r"offloaded \d+/\d+ layers", r"warning", r"main: model loaded",
    r"memory breakdown", r"- (ROCm|Vulkan|Host|CPU)",
]
RX = re.compile("|".join(PATTERNS), re.IGNORECASE)


def find_exe(root: Path) -> Path | None:
    for p in sorted(root.rglob(EXE)):
        return p.resolve()
    return None


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def child_env(bindir: Path) -> dict:
    env = dict(os.environ)
    if WIN:
        env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    else:
        env["LD_LIBRARY_PATH"] = str(bindir) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def peak_mem_mib(proc: subprocess.Popen) -> float | None:
    try:
        if WIN:
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            psapi = ctypes.WinDLL("psapi")
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(int(proc._handle), ctypes.byref(c), c.cb):
                return round(c.PeakWorkingSetSize / 2**20, 1)
            return None
        txt = Path(f"/proc/{proc.pid}/status").read_text(encoding = "utf-8")
        m = re.search(r"VmHWM:\s+(\d+) kB", txt)
        return round(int(m.group(1)) / 1024, 1) if m else None
    except Exception as e:  # noqa: BLE001
        return None


def meminfo() -> dict:
    if WIN:
        return {}
    out = {}
    for line in Path("/proc/meminfo").read_text(encoding = "utf-8").splitlines():
        k, v = line.split(":", 1)
        if k in ("MemTotal", "MemAvailable", "MemFree", "SwapTotal", "SwapFree", "Cached"):
            out[k + "_MiB"] = int(v.split()[0]) // 1024
    return out


def http(url: str, body: dict | None = None, timeout: float = 5.0):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data = data, headers = {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout = timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def run_cell(exe: Path, model: Path, extra: list[str], logf: Path, load_timeout: float, oom_first: bool) -> dict:
    port = free_port()
    argv = [str(exe), "-m", str(model), "-c", "4096", "--fit", "on",
            "--host", "127.0.0.1", "--port", str(port), "-lv", "4"] + extra
    row: dict = {"argv": argv[1:], "mem_before": meminfo()}
    t0 = time.time()
    with open(logf, "w", encoding = "utf-8", errors = "replace") as lf:
        proc = subprocess.Popen(argv, stdout = lf, stderr = subprocess.STDOUT, cwd = str(exe.parent),
                                env = child_env(exe.parent))
        if oom_first and not WIN:
            try:
                Path(f"/proc/{proc.pid}/oom_score_adj").write_text("1000", encoding = "utf-8")
            except Exception:  # noqa: BLE001
                pass
        loaded = False
        while time.time() - t0 < load_timeout:
            if proc.poll() is not None:
                break
            try:
                st, _ = http(f"http://127.0.0.1:{port}/health", timeout = 2)
                if st == 200:
                    loaded = True
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
        row["load_s"] = round(time.time() - t0, 1) if loaded else None
        row["loaded"] = loaded
        row["completion"] = None
        if loaded:
            try:
                st, txt = http(f"http://127.0.0.1:{port}/completion",
                               {"prompt": "The capital of France is", "n_predict": 16, "temperature": 0},
                               timeout = 300)
                j = json.loads(txt)
                row["completion"] = {"status": st, "content": j.get("content"),
                                     "tokens_predicted": j.get("tokens_predicted"),
                                     "timings": j.get("timings")}
            except Exception as e:  # noqa: BLE001
                row["completion"] = {"error": repr(e)}
        row["peak_mem_mib"] = peak_mem_mib(proc) if proc.poll() is None else None
        row["mem_loaded"] = meminfo()
        exited_early = proc.poll() is not None
        row["exited_before_stop"] = exited_early
        if not exited_early:
            proc.terminate()
            try:
                proc.wait(timeout = 60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout = 60)
        row["exit_code"] = proc.returncode
    text = logf.read_text(encoding = "utf-8", errors = "replace")
    row["log_lines"] = [l for l in text.splitlines() if RX.search(l)][:200]
    row["load_mode_lines"] = [l for l in text.splitlines() if "load_mode =" in l]
    row["lazy_lines"] = [l for l in text.splitlines() if "lazy" in l.lower()]
    row["buffer_lines"] = [l for l in text.splitlines() if "model buffer size" in l]
    row["log_tail"] = text.splitlines()[-25:]
    return row


def start_ballast(model_bytes: int, py: str, out: Path) -> tuple[subprocess.Popen | None, dict]:
    mi = meminfo()
    avail = mi.get("MemAvailable_MiB", 0) * 2**20
    target_avail = model_bytes - 2 * 2**30          # below the file size
    hold = avail - target_avail
    info = {"mem_before_ballast": mi, "target_avail_MiB": target_avail // 2**20, "hold_MiB": hold // 2**20}
    if hold <= 0:
        info["skipped"] = "MemAvailable already below target"
        return None, info
    code = ("import sys,time\n"
            f"n={hold}\n"
            "chunks=[]\n"
            "step=1<<30\n"
            "while n>0:\n"
            "  k=min(step,n); b=bytearray(k)\n"
            "  for i in range(0,k,4096): b[i]=1\n"
            "  chunks.append(b); n-=k\n"
            "print('READY',flush=True)\n"
            "time.sleep(3600)\n")
    # touching every page with a python loop is slow; memset via slice assignment is faster
    # one written byte per 4 KiB page faults every page in (resident, anonymous) without a full memset
    code = code.replace("  for i in range(0,k,4096): b[i]=1\n", "  b[::4096]=b'\\x01'*len(range(0,k,4096))\n")
    p = subprocess.Popen([py, "-c", code], stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True)
    t0 = time.time()
    line = p.stdout.readline() if p.stdout else ""
    info["ballast_ready_s"] = round(time.time() - t0, 1)
    info["ballast_ready"] = line.strip() == "READY"
    info["mem_after_ballast"] = meminfo()
    if p.poll() is not None:
        info["ballast_rc"] = p.returncode
    return p, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", required = True, help = "dir with one subdir per backend bundle")
    ap.add_argument("--backends", default = "rocm,vulkan")
    ap.add_argument("--model", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--fit-target", default = None, help = "Windows head: --fit-target value")
    ap.add_argument("--ballast-backend", default = None, help = "Linux only: run A/B under a RAM ballast")
    ap.add_argument("--load-timeout", type = float, default = 600)
    a = ap.parse_args()

    out = Path(a.out)
    (out / "logs").mkdir(parents = True, exist_ok = True)
    rows = out / "cells.jsonl"
    model = Path(a.model).resolve()

    def emit(row: dict) -> None:
        with open(rows, "a", encoding = "utf-8") as f:
            f.write(json.dumps(row) + "\n")
        c = row.get("completion") or {}
        print(f"[{row.get('backend')}/{row.get('cell')}] loaded={row.get('loaded')} load_s={row.get('load_s')} "
              f"rc={row.get('exit_code')} peak={row.get('peak_mem_mib')} "
              f"content={(c.get('content') or c.get('error') or '')[:60]!r}", flush = True)
        for l in row.get("load_mode_lines", []) + row.get("lazy_lines", []) + row.get("buffer_lines", []):
            print("    " + l, flush = True)

    head = ["--load-mode", "mmap", "--lazy-mode", "on"] + (["--fit-target", a.fit_target] if a.fit_target else [])
    cells = [("A_base", []), ("B_head", head), ("C_mmap_only", ["--load-mode", "mmap"])]

    for be in a.backends.split(","):
        exe = find_exe(Path(a.bins) / be)
        if exe is None:
            emit({"backend": be, "cell": "locate", "error": f"no {EXE} under {Path(a.bins) / be}"})
            continue
        env = child_env(exe.parent)
        info: dict = {"backend": be, "cell": "info", "exe": str(exe)}
        for flag in ("--version", "--help", "--list-devices"):
            try:
                r = subprocess.run([str(exe), flag], capture_output = True, text = True, errors = "replace",
                                   env = env, cwd = str(exe.parent), timeout = 120)
                txt = (r.stdout or "") + (r.stderr or "")
                (out / "logs" / f"{be}_{flag.strip('-')}.log").write_text(txt, encoding = "utf-8")
                if flag == "--help":
                    lines = txt.splitlines()
                    keep = []
                    for i, l in enumerate(lines):
                        if re.search(r"--load-mode|--lazy-mode|--fit-target|--mmap|--no-mmap", l):
                            keep.extend(lines[i:i + 4])
                    info["help_lines"] = keep
                else:
                    info[flag.strip("-") + "_out"] = txt.strip().splitlines()[-12:]
            except Exception as e:  # noqa: BLE001
                info[flag] = repr(e)
        emit(info)
        for name, extra in cells:
            row = run_cell(exe, model, extra, out / "logs" / f"{be}_{name}.log", a.load_timeout, False)
            row.update({"backend": be, "cell": name})
            emit(row)

    if a.ballast_backend and not WIN:
        exe = find_exe(Path(a.bins) / a.ballast_backend)
        if exe is not None:
            for name, extra in cells[:2]:
                p, binfo = start_ballast(model.stat().st_size, sys.executable, out)
                row = {"backend": a.ballast_backend, "cell": name + "_ballast", "ballast": binfo}
                if p is not None and binfo.get("ballast_ready"):
                    row.update(run_cell(exe, model, extra, out / "logs" / f"{a.ballast_backend}_{name}_ballast.log",
                                        300, True))
                    row["ballast_alive_after"] = p.poll() is None
                if p is not None:
                    p.kill()
                    p.wait(timeout = 60)
                    row.setdefault("ballast", {})["rc_at_kill"] = p.returncode
                time.sleep(5)
                emit(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())

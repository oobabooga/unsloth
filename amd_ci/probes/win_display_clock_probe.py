#!/usr/bin/env python3
"""Probe: does the gfx1151 GPU clock / GEMM throughput drop on a Windows runner once the display idle timeout passes?

Observes only. Writes JSON to --out (rewritten after every sample), never stdout.

  host      power plan, VIDEOIDLE / STANDBYIDLE (AC and DC), session id, powercfg /requests, torch device props
  clocks    every clock source we can try on Windows: torch.cuda.clock_rate, python amdsmi, amd-smi.exe forms
  timeline  an fp16 4096^3 matmul TFLOP/s sentinel every --interval seconds through three arms on this one runner:
              A   as-is (no execution-state hold), long enough to pass the display timeout
              B   SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED) held
              A2  hold released again, long enough to pass the timeout a second time
            each sample also records the idle time since last input, the monitor Availability and a clock reading.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


def _write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent = 2, default = str), encoding = "utf-8")
    os.replace(tmp, path)


def _run(cmd: list, timeout: int = 30) -> dict:
    t = time.time()
    try:
        p = subprocess.run(cmd, capture_output = True, timeout = timeout)
        out = p.stdout.decode("utf-8", "replace")
        err = p.stderr.decode("utf-8", "replace")
        return {"rc": p.returncode, "out": out[-6000:], "err": err[-1500:], "s": round(time.time() - t, 2)}
    except Exception as exc:  # noqa: BLE001
        return {"rc": None, "error": f"{type(exc).__name__}: {exc}", "s": round(time.time() - t, 2)}


def _ps(command: str, timeout: int = 30) -> dict:
    return _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command], timeout)


def _powercfg_seconds(text: str) -> dict:
    ac = re.search(r"Current AC Power Setting Index:\s*0x([0-9a-fA-F]+)", text)
    dc = re.search(r"Current DC Power Setting Index:\s*0x([0-9a-fA-F]+)", text)
    return {"ac_s": int(ac.group(1), 16) if ac else None, "dc_s": int(dc.group(1), 16) if dc else None}


def host_facts() -> dict:
    h: dict = {"python": sys.version, "computername": os.environ.get("COMPUTERNAME"),
               "runner_name": os.environ.get("RUNNER_NAME")}
    h["activescheme"] = _run(["powercfg", "/getactivescheme"])
    vid = _run(["powercfg", "/q", "SCHEME_CURRENT", "SUB_VIDEO", "VIDEOIDLE"])
    slp = _run(["powercfg", "/q", "SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE"])
    hib = _run(["powercfg", "/q", "SCHEME_CURRENT", "SUB_SLEEP", "HIBERNATEIDLE"])
    h["videoidle"] = {**vid, **_powercfg_seconds(vid.get("out", ""))}
    h["standbyidle"] = {**slp, **_powercfg_seconds(slp.get("out", ""))}
    h["hibernateidle"] = {**hib, **_powercfg_seconds(hib.get("out", ""))}
    h["requests"] = _run(["powercfg", "/requests"])
    try:
        sid = ctypes.c_ulong()
        ctypes.windll.kernel32.ProcessIdToSessionId(os.getpid(), ctypes.byref(sid))
        h["session_id"] = sid.value
    except Exception as exc:  # noqa: BLE001
        h["session_id"] = f"{type(exc).__name__}: {exc}"
    h["query_user"] = _run(["query", "user"])
    h["monitors"] = _ps("Get-CimInstance Win32_DesktopMonitor | Select-Object Name,Availability,Status | "
                        "ConvertTo-Json -Compress")
    h["video_controllers"] = _ps("Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,"
                                 "CurrentRefreshRate,CurrentHorizontalResolution,Availability | ConvertTo-Json -Compress")
    h["runner_service"] = _ps("Get-CimInstance Win32_Service | Where-Object { $_.Name -like 'actions.runner*' } | "
                              "Select-Object Name,StartName,State | ConvertTo-Json -Compress")
    return h


def _idle_s():
    try:
        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        li = LASTINPUTINFO()
        li.cbSize = ctypes.sizeof(li)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(li)):
            return None
        return round((ctypes.windll.kernel32.GetTickCount() - li.dwTime) / 1000.0, 1)
    except Exception:  # noqa: BLE001
        return None


def _set_state(flags: int) -> int:
    fn = ctypes.windll.kernel32.SetThreadExecutionState
    fn.argtypes = [ctypes.c_uint]
    fn.restype = ctypes.c_uint
    return int(fn(flags))


AMDSMI_FORMS = (
    ["metric", "-g", "0", "-c", "--json"],
    ["metric", "--clock", "--json"],
    ["metric", "-c"],
    ["metric"],
    ["static", "-g", "0", "--json"],
    ["list"],
)


def clock_sources(torch) -> dict:
    """Try every clock reader once; return raw answers plus which readers are cheap enough to sample."""
    res: dict = {}
    try:
        props = torch.cuda.get_device_properties(0)
        res["device_props"] = {k: str(getattr(props, k)) for k in dir(props) if not k.startswith("_")}
    except Exception as exc:  # noqa: BLE001
        res["device_props"] = f"{type(exc).__name__}: {exc}"
    try:
        res["torch_clock_rate"] = torch.cuda.clock_rate(0)
    except Exception as exc:  # noqa: BLE001
        res["torch_clock_rate"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    try:
        import amdsmi  # noqa: F401

        res["python_amdsmi"] = "importable"
    except Exception as exc:  # noqa: BLE001
        res["python_amdsmi"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    exe = None
    for c in ("amd-smi.exe", "amd-smi"):
        try:
            import shutil

            exe = shutil.which(c)
        except Exception:  # noqa: BLE001
            exe = None
        if exe:
            break
    if not exe:
        for c in (r"C:\Program Files\AMD\amd-smi\amd-smi.exe", r"C:\Windows\System32\amd-smi.exe"):
            if Path(c).is_file():
                exe = c
                break
    res["amd_smi_exe"] = exe
    res["amd_smi"] = {}
    if exe:
        for form in AMDSMI_FORMS:
            res["amd_smi"][" ".join(form)] = _run([exe, *form], 60)
    return res


def _amdsmi_clock(exe: str, form: list) -> dict:
    r = _run([exe, *form], 30)
    out = r.get("out", "")
    # Keep only clock-looking lines so the timeline stays small.
    lines = [ln.strip() for ln in out.splitlines() if re.search(r"(?i)clk|clock|freq|mhz", ln)]
    return {"rc": r.get("rc"), "s": r.get("s"), "lines": lines[:40]}


def sentinel(torch, n_inner: int = 10, reps: int = 7) -> dict:
    n = 4096
    a = torch.randn(n, n, dtype = torch.float16, device = "cuda")
    b = torch.randn(n, n, dtype = torch.float16, device = "cuda")
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    tf = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        s.record()
        for _ in range(n_inner):
            a @ b
        e.record()
        e.synchronize()
        ms = s.elapsed_time(e)
        tf.append(2 * n ** 3 * n_inner / (ms / 1e3) / 1e12)
    del a, b
    return {"tflops_median": round(statistics.median(tf), 3), "tflops_max": round(max(tf), 3),
            "tflops_min": round(min(tf), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--interval", type = float, default = 30.0)
    ap.add_argument("--min-a-min", type = float, default = 30.0, help = "arm A at least this long")
    ap.add_argument("--b-min", type = float, default = 15.0)
    ap.add_argument("--max-arm-min", type = float, default = 60.0)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents = True, exist_ok = True)
    obs: dict = {"started": time.time()}
    try:
        obs["host"] = host_facts()
    except Exception as exc:  # noqa: BLE001
        obs["host_error"] = traceback.format_exc()[-3000:]
    _write(out, obs)
    import torch

    obs["torch"] = {"version": torch.__version__, "hip": getattr(torch.version, "hip", None),
                    "available": torch.cuda.is_available(),
                    "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if not torch.cuda.is_available():
        obs["fatal"] = "no GPU visible to torch"
        _write(out, obs)
        return 1
    obs["clocks"] = clock_sources(torch)
    _write(out, obs)
    exe = obs["clocks"].get("amd_smi_exe")
    sample_form = None
    for form, r in (obs["clocks"].get("amd_smi") or {}).items():
        if r.get("rc") == 0 and re.search(r"(?i)clk|clock|mhz", r.get("out", "")) and (r.get("s") or 99) < 8:
            sample_form = form.split(" ")
            break
    obs["amd_smi_sample_form"] = sample_form
    torch_clock_ok = isinstance(obs["clocks"].get("torch_clock_rate"), (int, float))

    vid = ((obs.get("host") or {}).get("videoidle") or {}).get("ac_s")
    t_disp = vid if isinstance(vid, int) and vid > 0 else None
    a_min = args.min_a_min
    if t_disp:
        a_min = max(a_min, t_disp / 60 + 15)
    a_min = min(a_min, args.max_arm_min)
    a2_min = min(max((t_disp or 0) / 60 + 8, 12), args.max_arm_min)
    plan = [("A", a_min, 0), ("B", args.b_min, ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED),
            ("A2", a2_min, 0)]
    obs["plan"] = {"display_timeout_ac_s": vid, "arms_min": {k: round(v, 1) for k, v, _ in plan},
                   "interval_s": args.interval}
    obs["samples"] = []
    obs["hold_calls"] = []
    t_job = time.time()
    for arm, minutes, flags in plan:
        if flags:
            prev = _set_state(flags)
            obs["hold_calls"].append({"arm": arm, "set": hex(flags), "returned_prev": hex(prev), "t": time.time(),
                                      "requests_while_held": _run(["powercfg", "/requests"])})
        arm_t0 = time.time()
        while time.time() - arm_t0 < minutes * 60:
            t0 = time.time()
            s = {"arm": arm, "t_job_s": round(t0 - t_job, 1), "t_arm_s": round(t0 - arm_t0, 1), "idle_s": _idle_s()}
            try:
                s.update(sentinel(torch))
            except Exception as exc:  # noqa: BLE001
                s["sentinel_error"] = f"{type(exc).__name__}: {exc}"
            if torch_clock_ok:
                try:
                    s["torch_clock_rate"] = torch.cuda.clock_rate(0)
                except Exception as exc:  # noqa: BLE001
                    s["torch_clock_rate"] = f"{type(exc).__name__}"
            if exe and sample_form:
                s["amd_smi"] = _amdsmi_clock(exe, sample_form)
            mon = _ps("(Get-CimInstance Win32_DesktopMonitor | ForEach-Object { $_.Availability }) -join ','", 20)
            s["monitor_availability"] = (mon.get("out") or "").strip()
            obs["samples"].append(s)
            _write(out, obs)
            left = args.interval - (time.time() - t0)
            if left > 0:
                time.sleep(left)
        if flags:
            prev = _set_state(ES_CONTINUOUS)
            obs["hold_calls"].append({"arm": arm, "set": "0x80000000 (clear)", "returned_prev": hex(prev),
                                      "t": time.time()})
    obs["host_end"] = {"requests": _run(["powercfg", "/requests"]), "idle_s": _idle_s()}
    obs["finished"] = True
    _write(out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

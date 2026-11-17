#!/usr/bin/env python3
"""Probe (Windows): this checkout's GPU monitor readings as a real standard local user, under a HIP load.

Observes only. Creates a throwaway local account (Users only, random name and password), runs the
readings in a child logged on as it (CreateProcessWithLogonW, its own profile: the UAC-filtered,
Medium-integrity token a non-admin desktop user holds), and deletes the account and its profile in
a finally (the workflow repeats that in an always() step). While the child runs, a Toolhelp watcher
records every process that appears, the child logs every subprocess it starts and any elevation
error (WinError 740), and the probe's own Get-Counter is the independent load ground truth. The
elevation control launches diskpart.exe, the manifest-elevating helper amd-smi pulled in (#14041),
as the same user, so the watcher is shown to see an elevation attempt rather than assumed to.
Pairs with criteria/win_uac_user_18005.py.
"""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from win_gpu_monitor_probe import LOAD, raw_engine_types  # noqa: E402
from win_uac_probe import Watcher  # noqa: E402

CHILD = r"""
import json, os, sys, time, traceback, ctypes, subprocess
from ctypes import wintypes
out = {"pid": os.getpid(), "user": os.environ.get("USERNAME"), "spawns": [], "spawn_errors": []}
def save():
    open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(out, default=str))
_real = subprocess.Popen.__init__
def _logged(self, args, *a, **k):
    out["spawns"].append(args if isinstance(args, str) else " ".join(map(str, args))[:200])
    try:
        _real(self, args, *a, **k)
    except OSError as e:
        out["spawn_errors"].append({"args": str(args)[:200], "winerror": getattr(e, "winerror", None), "error": repr(e)})
        raise
subprocess.Popen.__init__ = _logged
save()
try:
    out["is_admin"] = bool(ctypes.windll.shell32.IsUserAnAdmin())
    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    adv.GetSidSubAuthorityCount.restype = ctypes.c_void_p
    adv.GetSidSubAuthority.restype = ctypes.c_void_p
    adv.GetSidSubAuthority.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    tok = wintypes.HANDLE()
    adv.OpenProcessToken(wintypes.HANDLE(-1), 0x0008, ctypes.byref(tok))
    buf = ctypes.create_string_buffer(256); n = wintypes.DWORD()
    adv.GetTokenInformation(tok, 25, buf, 256, ctypes.byref(n))
    sid = ctypes.cast(buf, ctypes.POINTER(ctypes.c_void_p))[0]
    cnt = ctypes.cast(adv.GetSidSubAuthorityCount(ctypes.c_void_p(sid)), ctypes.POINTER(ctypes.c_ubyte))[0]
    out["integrity_rid"] = ctypes.cast(adv.GetSidSubAuthority(ctypes.c_void_p(sid), cnt - 1), ctypes.POINTER(wintypes.DWORD))[0]
    elev = wintypes.DWORD()
    adv.GetTokenInformation(tok, 20, ctypes.byref(elev), 4, ctypes.byref(n))
    out["token_elevated"] = bool(elev.value)
    etype = wintypes.DWORD()
    adv.GetTokenInformation(tok, 18, ctypes.byref(etype), 4, ctypes.byref(n))
    out["elevation_type"] = etype.value
    save()
    if sys.argv[1] == "control":
        try:
            subprocess.run(["diskpart.exe", "/?"], capture_output=True, timeout=30)
            out["createprocess"] = "started"
        except OSError as e:
            out["createprocess"] = {"winerror": getattr(e, "winerror", None)}
        sh = ctypes.WinDLL("shell32", use_last_error=True)
        sh.ShellExecuteW.restype = ctypes.c_void_p
        out["shellexecute_rc"] = sh.ShellExecuteW(None, "open", "diskpart.exe", None, None, 0)
        out["shellexecute_lasterror"] = ctypes.get_last_error()
        time.sleep(15)
    else:
        sys.path.insert(0, sys.argv[3])
        import utils.hardware.hardware as hw
        from utils.hardware import gpu_query
        devices, _ = hw._rocm_windows_per_device_vram([0])
        luid = (devices or [{}])[0].get("luid")
        out["luid"] = luid
        out["util"], out["temp"], out["payload"] = [], [], []
        f = getattr(hw, "_rocm_windows_d3dkmt_temperature_c", None)
        out["temp_fn_present"] = f is not None
        for _ in range(3):
            out["util"].append(hw._rocm_windows_perf_counter_gpu_util_pct(luid))
            out["temp"].append(f(luid) if f else None)
            with gpu_query.display_reads(max_stale=0):
                dev = (hw.get_gpu_utilization().get("devices") or [{}])[0]
            out["payload"].append({k: dev.get(k) for k in ("gpu_utilization_pct", "temperature_c")})
            save()
            time.sleep(3)
except Exception:
    out["error"] = traceback.format_exc()[-1500:]
save()
"""


def ps(cmd: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                          capture_output = True, text = True, timeout = timeout)


def create_user() -> tuple[str, str]:
    name = "amdciuac" + secrets.token_hex(3)
    pw = "Aa1!" + secrets.token_urlsafe(18)
    # Password over stdin, never argv; Users by SID so a localized group name cannot miss.
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                        "$p = ConvertTo-SecureString ([Console]::In.ReadLine()) -AsPlainText -Force;"
                        f"New-LocalUser -Name {name} -Password $p -PasswordNeverExpires -UserMayNotChangePassword"
                        " -AccountNeverExpires | Out-Null;"
                        f"Add-LocalGroupMember -SID S-1-5-32-545 -Member {name}"],
                       input = pw + "\n", capture_output = True, text = True, timeout = 120)
    if r.returncode != 0:
        subprocess.run(["net", "user", name, "/delete"], capture_output = True)
        raise RuntimeError(f"New-LocalUser failed: {r.stdout} {r.stderr}")
    return name, pw


def delete_user(name: str) -> dict:
    out = {"user_deleted": subprocess.run(["net", "user", name, "/delete"], capture_output = True).returncode == 0}
    r = ps(f"Get-CimInstance Win32_UserProfile | Where-Object {{ $_.LocalPath -like '*\\{name}*' }} | Remove-CimInstance;"
           f"if (Test-Path C:\\Users\\{name}) {{ Remove-Item -Recurse -Force C:\\Users\\{name} -ErrorAction SilentlyContinue }};"
           f"[bool](Get-LocalUser -Name {name} -ErrorAction SilentlyContinue)")
    out["still_exists"] = r.stdout.strip().lower() == "true"
    return out


def run_as(user: str, pw: str, argv: list[str], timeout_s: float, watcher: Watcher) -> dict:
    import ctypes
    from ctypes import wintypes

    class SI(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR), ("lpDesktop", wintypes.LPWSTR),
                    ("lpTitle", wintypes.LPWSTR), ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
                    ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD), ("dwXCountChars", wintypes.DWORD),
                    ("dwYCountChars", wintypes.DWORD), ("dwFillAttribute", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
                    ("lpReserved2", ctypes.c_void_p), ("hStdInput", wintypes.HANDLE),
                    ("hStdOutput", wintypes.HANDLE), ("hStdError", wintypes.HANDLE)]

    class PI(ctypes.Structure):
        _fields_ = [("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
                    ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD)]
    adv = ctypes.WinDLL("advapi32", use_last_error = True)
    k = ctypes.WinDLL("kernel32", use_last_error = True)
    k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.WaitForSingleObject.restype = wintypes.DWORD
    k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    si = SI(); si.cb = ctypes.sizeof(SI)
    pi = PI()
    cmd = ctypes.create_unicode_buffer(subprocess.list2cmdline(argv))
    t0 = time.monotonic()
    # LOGON_WITH_PROFILE; CREATE_NO_WINDOW. seclogon grants the new logon SID this window station and desktop.
    if not adv.CreateProcessWithLogonW(user, ".", pw, 1, None, cmd, 0x08000000, None,
                                       str(Path(argv[1]).parent), ctypes.byref(si), ctypes.byref(pi)):
        return {"spawn_error": ctypes.get_last_error()}
    rc = k.WaitForSingleObject(pi.hProcess, int(timeout_s * 1000))
    code = wintypes.DWORD()
    k.GetExitCodeProcess(pi.hProcess, ctypes.byref(code))
    out = {"child_pid": pi.dwProcessId, "timed_out": rc != 0, "exit_code": hex(code.value),
           "seconds": round(time.monotonic() - t0, 2)}
    if rc != 0:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pi.dwProcessId)], capture_output = True)
    time.sleep(0.5)
    out["spawned"] = watcher.spawned_under(pi.dwProcessId)
    out["prompt_processes_anywhere"] = watcher.anywhere(("consent.exe", "diskpart.exe"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    obs: dict = {"state": args.state}
    work = args.out.parent / f"uacuser_{args.state}"
    work.mkdir(parents = True, exist_ok = True)
    child_py = work / "child.py"
    child_py.write_text(CHILD, encoding = "utf-8")
    for d in {Path(sys.prefix), args.checkout, work}:
        subprocess.run(["icacls", str(d), "/grant", "*S-1-5-32-545:(OI)(CI)M", "/T", "/C", "/Q"], capture_output = True)
    user = pw = None
    load = None
    try:
        user, pw = create_user()
        obs["user"] = user
    except Exception:
        import traceback
        obs["error"] = traceback.format_exc()[-1500:]
    try:
        if user and not obs.get("error"):
            sys.path.insert(0, str(args.checkout / "studio" / "backend"))
            import utils.hardware.hardware as hw
            luid = (hw._rocm_windows_per_device_vram([0])[0] or [{}])[0].get("luid")
            obs["luid"] = luid
            load = subprocess.Popen([sys.executable, "-c", LOAD, "120"], stdout = subprocess.PIPE,
                                    stderr = subprocess.DEVNULL, text = True)
            obs["load_ready"] = load.stdout.readline().strip() == "READY"
            time.sleep(6)
            obs["raw_before"] = raw_engine_types(luid)
            res = work / "readings.json"
            w = Watcher(); w.start()
            obs["readings"] = run_as(user, pw, [sys.executable, str(child_py), "readings", str(res),
                                                str(args.checkout / "studio" / "backend")], 240, w)
            w.stop.set(); w.join(2)
            obs["raw_after"] = raw_engine_types(luid)
            obs["load_alive_after"] = load.poll() is None
            obs["readings"]["child"] = json.loads(res.read_text(encoding = "utf-8")) if res.exists() else None
            res = work / "control.json"
            w = Watcher(); w.start()
            obs["control"] = run_as(user, pw, [sys.executable, str(child_py), "control", str(res), "-"], 90, w)
            w.stop.set(); w.join(2)
            obs["control"]["child"] = json.loads(res.read_text(encoding = "utf-8")) if res.exists() else None
            subprocess.run(["taskkill", "/F", "/IM", "consent.exe"], capture_output = True)
            subprocess.run(["taskkill", "/F", "/IM", "diskpart.exe"], capture_output = True)
    except Exception:
        import traceback
        obs["error"] = traceback.format_exc()[-1500:]
    finally:
        if load is not None:
            load.kill()
        if user:
            obs["cleanup"] = delete_user(user)
    args.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

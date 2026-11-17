#!/usr/bin/env python3
"""Probe (Windows): do this checkout's GPU monitor readings run as a standard user with no UAC prompt?

Observes only. The probe records the runner's own elevation, then reruns the readings in a child
holding a restricted copy of the runner's token (Administrators disabled, all privileges but
SeChangeNotify dropped, Medium integrity: what an unelevated desktop user holds). While the child
runs, a Toolhelp process watcher records every process that appears, so an elevation prompt
(consent.exe) or a manifest-elevating helper (diskpart.exe) is seen, not inferred. A positive
control runs amd-smi.exe, which #14041 found pops a UAC/DiskPart prompt, under the same token.
Pairs with criteria/win_uac_18005.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

CHILD = r"""
import json, sys, time, traceback, ctypes
from ctypes import wintypes
out = {"pid": __import__("os").getpid()}
def save():
    open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(out, default=str))
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
    save()
    mode = sys.argv[1]
    if mode == "amd_smi":
        import shutil
        exe = shutil.which("amd-smi") or r"C:\Windows\System32\amd-smi.exe"
        t0 = time.monotonic()
        try:
            r = __import__("subprocess").run([exe, "static", "--json"], capture_output=True, text=True, timeout=60)
            out["amd_smi_rc"] = r.returncode
            out["amd_smi_stderr"] = (r.stderr or "")[-400:]
            out["amd_smi_stdout_len"] = len(r.stdout or "")
        except Exception as e:
            out["amd_smi_error"] = repr(e)
        out["amd_smi_s"] = round(time.monotonic() - t0, 2)
    else:
        sys.path.insert(0, sys.argv[3])
        import utils.hardware.hardware as hw
        from utils.hardware import gpu_query
        devices, _ = hw._rocm_windows_per_device_vram([0])
        luid = (devices or [{}])[0].get("luid")
        out["luid"] = luid
        save()
        out["util"] = [hw._rocm_windows_perf_counter_gpu_util_pct(luid) for _ in range(2)]
        f = getattr(hw, "_rocm_windows_d3dkmt_temperature_c", None)
        out["temp_fn_present"] = f is not None
        out["temp"] = [f(luid) for _ in range(2)] if f else None
        with gpu_query.display_reads(max_stale=0):
            dev = (hw.get_gpu_utilization().get("devices") or [{}])[0]
        out["payload"] = {k: dev.get(k) for k in ("backend", "gpu_utilization_pct", "temperature_c", "vram_used_gb")}
except Exception:
    out["error"] = traceback.format_exc()[-1500:]
save()
"""


def _k32():
    import ctypes
    return ctypes.WinDLL("kernel32", use_last_error = True)


def snapshot() -> dict[int, tuple[int, str]]:
    """pid -> (parent pid, exe name) via Toolhelp: in process, so the watcher spawns nothing itself."""
    import ctypes
    from ctypes import wintypes

    class PE(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_void_p),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]
    k = _k32()
    k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k.Process32FirstW.argtypes = k.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k.CreateToolhelp32Snapshot(0x2, 0)
    pe = PE(); pe.dwSize = ctypes.sizeof(PE)
    out = {}
    ok = k.Process32FirstW(h, ctypes.byref(pe))
    while ok:
        out[pe.th32ProcessID] = (pe.th32ParentProcessID, pe.szExeFile.lower())
        ok = k.Process32NextW(h, ctypes.byref(pe))
    k.CloseHandle(h)
    return out


class Watcher(threading.Thread):
    def __init__(self):
        super().__init__(daemon = True)
        self.base = snapshot()
        self.seen: dict[int, tuple[int, str]] = {}
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            for pid, v in snapshot().items():
                if pid not in self.base and pid not in self.seen:
                    self.seen[pid] = v
            time.sleep(0.03)

    def spawned_under(self, root: int) -> list[str]:
        """Names of processes the watcher saw descend from ``root``."""
        tree, names = {root}, []
        changed = True
        while changed:
            changed = False
            for pid, (ppid, name) in self.seen.items():
                if ppid in tree and pid not in tree:
                    tree.add(pid); names.append(name); changed = True
        return names

    def anywhere(self, names: tuple[str, ...]) -> list[str]:
        return sorted({n for _, n in self.seen.values() if n in names})


def runner_identity() -> dict:
    import ctypes
    from ctypes import wintypes
    out = {"user": os.environ.get("USERNAME"), "is_admin": bool(ctypes.windll.shell32.IsUserAnAdmin())}
    sess = wintypes.DWORD()
    _k32().ProcessIdToSessionId(os.getpid(), ctypes.byref(sess))
    out["session_id"] = sess.value
    return out


def safer_token():
    import ctypes
    from ctypes import wintypes
    adv = ctypes.WinDLL("advapi32", use_last_error = True)
    lvl, tok = wintypes.HANDLE(), wintypes.HANDLE()
    if not adv.SaferCreateLevel(2, 0x20000, 1, ctypes.byref(lvl), None):
        raise OSError(ctypes.get_last_error(), "SaferCreateLevel")
    if not adv.SaferComputeTokenFromLevel(lvl, None, ctypes.byref(tok), 0, None):
        raise OSError(ctypes.get_last_error(), "SaferComputeTokenFromLevel")
    return tok


def restricted_token(medium: bool = True):
    """A restricted copy of this process's token: Administrators deny-only, max privileges dropped, Medium IL."""
    import ctypes
    from ctypes import wintypes
    adv = ctypes.WinDLL("advapi32", use_last_error = True)
    tok = wintypes.HANDLE()
    if not adv.OpenProcessToken(wintypes.HANDLE(-1), 0x0001 | 0x0002 | 0x0008 | 0x0080 | 0x0100, ctypes.byref(tok)):
        raise OSError(ctypes.get_last_error(), "OpenProcessToken")

    class SA(ctypes.Structure):
        _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]
    admins = ctypes.create_string_buffer(68); size = wintypes.DWORD(68)
    if not adv.CreateWellKnownSid(26, None, admins, ctypes.byref(size)):
        raise OSError(ctypes.get_last_error(), "CreateWellKnownSid")
    disable = SA(ctypes.cast(admins, ctypes.c_void_p), 0)
    new = wintypes.HANDLE()
    if not adv.CreateRestrictedToken(tok, 0x1, 1, ctypes.byref(disable), 0, None, 0, None, ctypes.byref(new)):
        raise OSError(ctypes.get_last_error(), "CreateRestrictedToken")
    if not medium:
        return new
    medium = ctypes.c_void_p()
    adv.ConvertStringSidToSidW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
    adv.GetLengthSid.argtypes = [ctypes.c_void_p]
    adv.ConvertStringSidToSidW("S-1-16-8192", ctypes.byref(medium))
    label = SA(medium, 0x20)
    if not adv.SetTokenInformation(new, 25, ctypes.byref(label), ctypes.sizeof(SA) + adv.GetLengthSid(medium)):
        raise OSError(ctypes.get_last_error(), "SetTokenInformation(IntegrityLevel)")
    return new


_DESKTOP: list[str] = []


def probe_desktop() -> str:
    """A window station + desktop with an open DACL: a Medium-IL child cannot open the session-0 service's (0xC0000142)."""
    if _DESKTOP:
        return _DESKTOP[0]
    import ctypes
    from ctypes import wintypes
    u32 = ctypes.WinDLL("user32", use_last_error = True)
    adv = ctypes.WinDLL("advapi32", use_last_error = True)

    class SA(ctypes.Structure):
        _fields_ = [("nLength", wintypes.DWORD), ("lpSecurityDescriptor", ctypes.c_void_p),
                    ("bInheritHandle", wintypes.BOOL)]
    sd = ctypes.create_string_buffer(64)
    adv.InitializeSecurityDescriptor(sd, 1)
    adv.SetSecurityDescriptorDacl(sd, True, None, False)
    sa = SA(ctypes.sizeof(SA), ctypes.cast(sd, ctypes.c_void_p), False)
    u32.CreateWindowStationW.restype = u32.GetProcessWindowStation.restype = wintypes.HANDLE
    u32.CreateDesktopW.restype = wintypes.HANDLE
    u32.SetProcessWindowStation.argtypes = [wintypes.HANDLE]
    ws = u32.CreateWindowStationW("amdciuac", 0, 0x37F, ctypes.byref(sa))
    if not ws:
        raise OSError(ctypes.get_last_error(), "CreateWindowStationW")
    old = u32.GetProcessWindowStation()
    u32.SetProcessWindowStation(ws)
    try:
        if not u32.CreateDesktopW("default", None, None, 0, 0x10000000, ctypes.byref(sa)):
            raise OSError(ctypes.get_last_error(), "CreateDesktopW")
    finally:
        u32.SetProcessWindowStation(old)
    _DESKTOP.append("amdciuac\\default")
    return _DESKTOP[0]


def run_restricted(argv: list[str], timeout_s: float, watcher: Watcher, variant: str = "medium_desk") -> dict:
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
    k = _k32()
    si = SI(); si.cb = ctypes.sizeof(SI)
    if variant.endswith("_desk"):
        desk = ctypes.create_unicode_buffer(probe_desktop())
        si.lpDesktop = ctypes.cast(desk, wintypes.LPWSTR)
    pi = PI()
    cmd = ctypes.create_unicode_buffer(subprocess.list2cmdline(argv))
    t0 = time.monotonic()
    token = (safer_token() if variant.startswith("safer") else restricted_token(medium = not variant.startswith("high")))
    if not adv.CreateProcessAsUserW(token, None, cmd, None, None, False, 0x08000000,
                                    None, None, ctypes.byref(si), ctypes.byref(pi)):
        return {"spawn_error": ctypes.get_last_error()}
    k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.WaitForSingleObject.restype = wintypes.DWORD
    rc = k.WaitForSingleObject(pi.hProcess, int(timeout_s * 1000))
    out = {"child_pid": pi.dwProcessId, "timed_out": rc != 0, "seconds": round(time.monotonic() - t0, 2)}
    code = wintypes.DWORD()
    k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k.GetExitCodeProcess(pi.hProcess, ctypes.byref(code))
    out["exit_code"] = hex(code.value)
    if rc != 0:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pi.dwProcessId)], capture_output = True)
    time.sleep(0.5)
    out["spawned"] = watcher.spawned_under(pi.dwProcessId)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    obs: dict = {"state": args.state}
    work = args.out.parent
    work.mkdir(parents = True, exist_ok = True)
    child_py = work / f"uac_child_{args.state}.py"
    child_py.write_text(CHILD, encoding = "utf-8")
    try:
        obs["runner"] = runner_identity()
        # A desktop user can read their own install; the runner's temp tree may grant only Administrators.
        for d in {Path(sys.prefix), args.checkout, work}:
            r = subprocess.run(["icacls", str(d), "/grant", "*S-1-1-0:(OI)(CI)M", "/T", "/C", "/Q"],
                               capture_output = True, text = True)
            obs.setdefault("icacls_rc", {})[str(d)] = r.returncode
        diag = {}
        for variant in ("high", "medium", "medium_desk", "safer", "safer_desk"):
            for name, argv in (("cmd", ["cmd.exe", "/c", f"whoami /groups /fo csv > {work / ('diag_' + variant + '.txt')}"]),
                               ("python", [sys.executable, "-c", "print(1)"])):
                w = Watcher(); w.start()
                try:
                    diag[f"{variant}/{name}"] = run_restricted(argv, 60.0, w, variant)
                except Exception as e:
                    diag[f"{variant}/{name}"] = {"error": repr(e)}
                w.stop.set(); w.join(2)
            f = work / ("diag_" + variant + ".txt")
            if f.exists():
                txt = f.read_text(encoding = "utf-8", errors = "replace")
                diag[f"{variant}/groups"] = [l for l in txt.splitlines() if "Administrators" in l or "Mandatory Label" in l]
        obs["diag"] = diag
        good = next((v for v in ("medium_desk", "medium", "safer_desk", "safer") if diag.get(f"{v}/python", {}).get("exit_code") == "0x0"), "medium_desk")
        obs["variant"] = good
        for mode, timeout_s in (("readings", 240.0), ("amd_smi", 90.0)):
            res = work / f"uac_{args.state}_{mode}.json"
            res.unlink(missing_ok = True)
            w = Watcher(); w.start()
            run = run_restricted([sys.executable, str(child_py), mode, str(res),
                                  str(args.checkout / "studio" / "backend")], timeout_s, w, good)
            w.stop.set(); w.join(2)
            run["prompt_processes_anywhere"] = w.anywhere(("consent.exe", "diskpart.exe"))
            run["child"] = json.loads(res.read_text(encoding = "utf-8")) if res.exists() else None
            obs[mode] = run
    except Exception:
        import traceback
        obs["error"] = traceback.format_exc()[-2000:]
    args.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

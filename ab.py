"""A/B: does Studio's Vulkan probe raise a Windows loader dialog when vulkan-1.dll lacks an import?

A = studio/backend/core/inference/_vulkan_probe.py at unslothai/unsloth BASE (main).
B = the same file with the SetErrorMode fix (_vulkan_probe_fixed.py next to this script).

Both run exactly the way LlamaCppBackend._run_vulkan_probe starts them: [sys.executable, probe, bindir],
capture_output, timeout 15 s, utf8_child_env, and Studio's own windows_hidden_subprocess_kwargs.

Two loaders:
  good: the bundle as installed, so ggml-vulkan.dll binds the system vulkan-1.dll.
  old:  the same bundle plus a copy of the system vulkan-1.dll whose vkGetPhysicalDeviceFeatures2 export name is
        renamed, standing in for a pre-1.1 Vulkan loader. It sits next to ggml-vulkan.dll, so the loader binds it
        first, the same way an old System32 copy is bound on the reporter's machine.

A dialog is detected three ways: a top-level window titled "... Entry Point Not Found" (dismissed after 1 s, as a
user clicking OK), System log event 26 ("Application Popup"), and the probe's wall time.
"""

import ctypes
import difflib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from ctypes import wintypes

import pefile

BASE = "6f4d9974dceafd70d29192e43805677d3992d63f"
RAW = f"https://raw.githubusercontent.com/unslothai/unsloth/{BASE}/studio/backend/"
BUNDLE = (
    "https://github.com/unslothai/llama.cpp/releases/download/b11160-mix-a6922cc/"
    "app-b11160-mix-a6922cc-windows-x64-vulkan.zip"
)
MISSING = b"vkGetPhysicalDeviceFeatures2"

work = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def fetch(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok = True)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    return dest


# Host context.
kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
user32 = ctypes.WinDLL("user32", use_last_error = True)
session = wintypes.DWORD()
kernel32.ProcessIdToSessionId(os.getpid(), ctypes.byref(session))
winsta = ctypes.create_unicode_buffer(256)
user32.GetUserObjectInformationW(user32.GetProcessWindowStation(), 2, winsta, 512, None)
print(f"python {sys.version.split()[0]}  session {session.value}  window station {winsta.value}")
print(f"runner error mode {kernel32.GetErrorMode():#x}")
# The runner service hands its children SEM_FAILCRITICALERRORS, which would hide the dialog from main's probe too.
# A Studio started from a desktop session has mode 0 (the reporter's dialog proves it), and children inherit it.
kernel32.SetErrorMode(0)
child_mode = subprocess.run(
    [sys.executable, "-c", "import ctypes; print(hex(ctypes.WinDLL('kernel32').GetErrorMode()))"],
    capture_output = True, text = True,
).stdout.strip()
print(f"harness error mode {kernel32.GetErrorMode():#x}, inherited by a child python as {child_mode}")
sys32 = os.path.join(os.environ["SystemRoot"], "System32")
for dll in ("vulkan-1.dll", "msvcp140.dll", "vcruntime140.dll"):
    print(f"System32\\{dll}: {os.path.isfile(os.path.join(sys32, dll))}")

# Studio code under test, all at BASE except the fixed probe.
studio = os.path.join(work, "studio_main")
for rel in ("core/inference/_vulkan_probe.py", "utils/subprocess_compat.py", "utils/child_stdio.py"):
    fetch(RAW + rel, os.path.join(studio, *rel.split("/")))
sys.path.insert(0, studio)
from utils.child_stdio import utf8_child_env  # noqa: E402
from utils.subprocess_compat import windows_hidden_subprocess_kwargs  # noqa: E402

probes = {
    "A main": os.path.join(studio, "core", "inference", "_vulkan_probe.py"),
    "B fix ": os.path.join(here, "_vulkan_probe_fixed.py"),
}
with open(probes["A main"], encoding = "utf-8") as a, open(probes["B fix "], encoding = "utf-8") as b:
    print("".join(difflib.unified_diff(a.readlines(), b.readlines(), "A/_vulkan_probe.py", "B/_vulkan_probe.py")))

# The Windows Vulkan bundle a fresh Studio install takes.
zpath = fetch(BUNDLE, os.path.join(work, "bundle.zip"))
good = os.path.join(work, "bin_good")
with zipfile.ZipFile(zpath) as z:
    z.extractall(good)
pe = pefile.PE(os.path.join(good, "ggml-vulkan.dll"))
imports = {e.dll.decode().lower(): [i.name.decode() for i in e.imports if i.name] for e in pe.DIRECTORY_ENTRY_IMPORT}
print("ggml-vulkan.dll imports from vulkan-1.dll:", imports.get("vulkan-1.dll"))
check(MISSING.decode() in imports.get("vulkan-1.dll", []), "bundle's ggml-vulkan.dll statically imports " + MISSING.decode())
pe.close()

# The stand-in pre-1.1 loader: rename the export, keeping the name table sorted ("...Features1" < "...Features2KHR").
old = os.path.join(work, "bin_old")
shutil.copytree(good, old)
src = pefile.PE(os.path.join(sys32, "vulkan-1.dll"))
data = bytearray(src.__data__)
exp = src.DIRECTORY_ENTRY_EXPORT.struct
patched = 0
for k in range(exp.NumberOfNames):
    rva = src.get_dword_at_rva(exp.AddressOfNames + 4 * k)
    if src.get_string_at_rva(rva) == MISSING:
        off = src.get_offset_from_rva(rva)
        data[off + len(MISSING) - 1] = ord("1")
        patched += 1
src.close()
with open(os.path.join(old, "vulkan-1.dll"), "wb") as f:
    f.write(data)
check_pe = pefile.PE(os.path.join(old, "vulkan-1.dll"))
names = {s.name for s in check_pe.DIRECTORY_ENTRY_EXPORT.symbols if s.name}
check_pe.close()
check(patched == 1 and MISSING not in names and b"vkCreateInstance" in names,
      "stand-in loader exports vkCreateInstance but not " + MISSING.decode())


# Dialog watcher.
WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def window_text(hwnd):
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def popups():
    found = []

    def top(hwnd, _):
        title = window_text(hwnd)
        if "Entry Point" in title:
            texts = []

            def child(h, _):
                t = window_text(h)
                if t:
                    texts.append(t)
                return True

            user32.EnumChildWindows(hwnd, WNDENUMPROC(child), 0)
            found.append((hwnd, title, texts))
        return True

    user32.EnumWindows(WNDENUMPROC(top), 0)
    return found


class Watcher(threading.Thread):
    def __init__(self):
        super().__init__(daemon = True)
        self.stop = threading.Event()
        self.seen = {}

    def run(self):
        while not self.stop.is_set():
            for hwnd, title, texts in popups():
                if hwnd not in self.seen:
                    self.seen[hwnd] = (title, texts, time.monotonic())
                elif time.monotonic() - self.seen[hwnd][2] > 1.0:
                    user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE, as a user clicking OK
            time.sleep(0.1)


def popup_events_after(record_id):
    ps = (
        "Get-WinEvent -FilterHashtable @{LogName='System'; Id=26} -MaxEvents 200 -ErrorAction SilentlyContinue | "
        "Select-Object RecordId, @{n='Message';e={$_.Message -replace '\\s+',' '}} | ConvertTo-Json -Compress"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output = True, text = True).stdout.strip()
    if not out:
        return []
    rows = json.loads(out)
    rows = rows if isinstance(rows, list) else [rows]
    return [r for r in rows if r["RecordId"] > record_id]


def last_record_id():
    events = popup_events_after(-1)
    return max((r["RecordId"] for r in events), default = 0)


def observe(argv, **kwargs):
    rec = last_record_id()
    watcher = Watcher()
    watcher.start()
    t0 = time.monotonic()
    try:
        r = subprocess.run(argv, capture_output = True, encoding = "utf-8", errors = "replace", **kwargs)
        rc, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = "timeout", e.stdout or "", e.stderr or ""
        out = out.decode(errors = "replace") if isinstance(out, bytes) else out
        err = err.decode(errors = "replace") if isinstance(err, bytes) else err
    elapsed = time.monotonic() - t0
    time.sleep(1.5)
    watcher.stop.set()
    watcher.join()
    events = popup_events_after(rec)
    return {"rc": rc, "out": out.strip(), "err": err.strip(), "s": elapsed,
            "windows": [(t, x) for t, x, _ in watcher.seen.values()], "events": [e["Message"] for e in events]}


def run_probe(probe, bindir):
    # Mirrors LlamaCppBackend._run_vulkan_probe on Windows (child_env_without_native_path_secret only drops a secret).
    return observe(
        [sys.executable, probe, bindir],
        timeout = 15,
        env = utf8_child_env(dict(os.environ)),
        **windows_hidden_subprocess_kwargs(),
    )


def show(label, r):
    print(f"--- {label}: rc={r['rc']} {r['s']:.1f}s dialogs={len(r['windows'])} popup_events={len(r['events'])}")
    for title, texts in r["windows"]:
        print(f"    window: {title!r} {texts}")
    for e in r["events"]:
        print(f"    event 26: {e}")
    if r["out"]:
        print("    stdout: " + r["out"].replace("\n", "\n            "))
    if r["err"]:
        print("    stderr: " + r["err"][-400:].replace("\n", "\n            "))


results = {}
for loader, bindir in (("good", good), ("old", old)):
    for name, probe in probes.items():
        for i in (1, 2):
            r = results[(loader, name.strip(), i)] = run_probe(probe, bindir)
            show(f"{name} loader={loader} run {i}", r)


def popped(r):
    # A visible window, a logged Application Popup, or a child left blocked until the 15 s timeout.
    return bool(r["windows"] or r["events"] or r["rc"] == "timeout")


A_old = [results[("old", "A main", i)] for i in (1, 2)]
B_old = [results[("old", "B fix", i)] for i in (1, 2)]
check(all(popped(r) for r in A_old), "A + old loader: every probe raises the Entry Point Not Found dialog")
check(not any(popped(r) for r in B_old), "B + old loader: no probe raises a dialog")
check(all(r["rc"] == 1 and "ggml-vulkan load failed" in r["err"] for r in B_old),
      "B + old loader: probe exits 1 with 'ggml-vulkan load failed' on stderr")
check(all(r["s"] < 10 for r in B_old), "B + old loader: probe returns promptly")
A_good = [results[("good", "A main", i)] for i in (1, 2)]
B_good = [results[("good", "B fix", i)] for i in (1, 2)]
check(not any(popped(r) for r in A_good + B_good), "good loader: no dialog from either probe")
check(all(r["rc"] == 0 and r["out"] for r in A_good + B_good), "good loader: both probes exit 0 with device rows")
strip_free = lambda r: [l.split("\t")[:1] + l.split("\t")[2:] for l in r["out"].splitlines()]  # free bytes drift
check(strip_free(A_good[0]) == strip_free(B_good[0]), "good loader: A and B report the same devices")

# llama-server's own backend loader (GGML_BACKEND_DL) against the same stand-in loader.
for loader, bindir in (("good", good), ("old", old)):
    r = observe([os.path.join(bindir, "llama-server.exe"), "--list-devices"], timeout = 60, cwd = bindir)
    show(f"llama-server --list-devices loader={loader}", r)
    results[("server", loader)] = r
check(not popped(results[("server", "old")]), "llama-server + old loader: no dialog (ggml's loader sets the error mode)")

print("\nSUMMARY")
print(f"{'case':34} {'rc':>8} {'secs':>6} {'dialogs':>8} {'event26':>8}")
for key, r in results.items():
    print(f"{' '.join(map(str, key)):34} {str(r['rc']):>8} {r['s']:6.1f} {len(r['windows']):8} {len(r['events']):8}")
print(f"\n{len(failures)} check(s) failed" if failures else "\nall checks passed")
sys.exit(1 if failures else 0)

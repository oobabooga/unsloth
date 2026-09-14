import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

T = os.environ["RUNNER_TEMP"]
AB = os.path.join(T, "ab")
SUMMARY = os.path.join(AB, "summary.jsonl")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
NOT_TTY = re.compile(r"(stdin|stdout) is not a terminal")

PROBE_PY = r'''
import ctypes, json, sys
from ctypes import wintypes
k = ctypes.WinDLL("kernel32", use_last_error=True)
k.GetStdHandle.restype = ctypes.c_void_p
k.GetStdHandle.argtypes = [wintypes.DWORD]
k.GetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
k.GetFileType.argtypes = [ctypes.c_void_p]
k.GetFileInformationByHandleEx.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
k.GetConsoleWindow.restype = ctypes.c_void_p
res = {}
for name, n in (("stdin", -10), ("stdout", -11), ("stderr", -12)):
    h = k.GetStdHandle(n & 0xFFFFFFFF)
    mode = wintypes.DWORD()
    ctypes.set_last_error(0)
    ok = k.GetConsoleMode(h, ctypes.byref(mode))
    err = ctypes.get_last_error()
    ft = k.GetFileType(h)
    buf = ctypes.create_string_buffer(4 + 2 * 2048)
    fname = None
    if k.GetFileInformationByHandleEx(h, 2, buf, ctypes.sizeof(buf)):
        ln = int.from_bytes(buf.raw[:4], "little")
        fname = buf.raw[4:4 + ln].decode("utf-16-le", "replace")
    res[name] = {"handle": h, "file_type": {0: "UNKNOWN", 1: "DISK", 2: "CHAR", 3: "PIPE"}.get(ft, ft),
                 "is_console": bool(ok), "mode": hex(mode.value), "err": 0 if ok else err, "name": fname}
res["console_window"] = bool(k.GetConsoleWindow())
with open(sys.argv[1], "w") as f:
    json.dump(res, f)
'''

# Same spawn shape as codex.js: async spawn with stdio "inherit".
PROBE_JS = r'''
const { spawn } = require("node:child_process");
const child = spawn(process.argv[2], process.argv.slice(3), { stdio: "inherit" });
child.on("error", (e) => { console.error(e); process.exit(1); });
child.on("exit", (code) => process.exit(code ?? 1));
'''


def write_probes():
    os.makedirs(AB, exist_ok=True)
    with open(os.path.join(AB, "probe.py"), "w") as f:
        f.write(PROBE_PY)
    with open(os.path.join(AB, "probe.js"), "w") as f:
        f.write(PROBE_JS)


def kill_tree(pid):
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)


def record(name, verdict, detail):
    with open(SUMMARY, "a", encoding="utf-8") as f:
        f.write(json.dumps({"case": name, "verdict": verdict, **{k: v for k, v in detail.items() if k != "tail"}}) + "\n")
    print(f"::group::{name}: {verdict}")
    for key, value in detail.items():
        if key != "tail":
            print(f"{key}: {value}")
    if "tail" in detail:
        print("--- screen text (ANSI stripped, last 3000 chars) ---")
        print(detail["tail"])
    print("::endgroup::")
    print(f"RESULT {name}: {verdict}", flush=True)


def read_probe(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def pty_case(name, argv, env=None, ready_marker=None, alive_secs=15, cap=240, probe_out=None, backend=None):
    from winpty import PtyProcess

    argv = [str(a) for a in argv]
    env = dict(os.environ if env is None else env)
    if probe_out and os.path.exists(probe_out):
        os.remove(probe_out)
    t0 = time.time()
    try:
        proc = PtyProcess.spawn(argv, env=env, dimensions=(50, 200), backend=backend)
    except Exception as exc:
        record(name, "SPAWN_ERROR", {"argv": argv, "error": repr(exc)})
        return
    chunks = []

    def reader():
        while True:
            try:
                data = proc.read(4096)
            except EOFError:
                return
            except Exception as exc:
                chunks.append(f"\n<read error {exc!r}>")
                return
            if data:
                chunks.append(data)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    marker_at = t0 if ready_marker is None else None
    while time.time() - t0 < cap and proc.isalive():
        if marker_at is None and ready_marker in ANSI.sub("", "".join(chunks)):
            marker_at = time.time()
        if marker_at is not None and time.time() - marker_at >= alive_secs:
            break
        time.sleep(0.25)
    alive = proc.isalive()
    if alive:
        kill_tree(proc.pid)
    thread.join(3)
    code = None if alive else proc.exitstatus
    raw = "".join(chunks)
    clean = ANSI.sub("", raw)
    with open(os.path.join(AB, f"{name}.raw.txt"), "w", encoding="utf-8") as f:
        f.write(raw)
    probe = read_probe(probe_out)
    if NOT_TTY.search(clean):
        verdict = "NOT_A_TERMINAL"
    elif probe is not None:
        verdict = "PROBE stdin=%s stdout=%s" % (probe["stdin"]["is_console"], probe["stdout"]["is_console"])
    elif alive:
        verdict = "ALIVE_AFTER_WAIT"
    else:
        verdict = f"EXITED({code})"
    record(name, verdict, {
        "argv": argv,
        "backend": backend,
        "seconds": round(time.time() - t0, 1),
        "killed_alive": alive,
        "exit_code": code,
        "ready_marker_seen": None if ready_marker is None else marker_at is not None,
        "alt_screen": "\x1b[?1049h" in raw,
        "probe": probe,
        "tail": clean[-3000:],
    })


def newconsole_case(name, argv, env=None, wait=60, probe_out=None):
    argv = [str(a) for a in argv]
    if probe_out and os.path.exists(probe_out):
        os.remove(probe_out)
    t0 = time.time()
    proc = subprocess.Popen(argv, env=dict(os.environ if env is None else env),
                            creationflags=subprocess.CREATE_NEW_CONSOLE)
    while time.time() - t0 < wait and proc.poll() is None:
        time.sleep(0.25)
    alive = proc.poll() is None
    if alive:
        kill_tree(proc.pid)
        proc.wait(10)
    probe = read_probe(probe_out)
    if probe is not None:
        verdict = "PROBE stdin=%s stdout=%s" % (probe["stdin"]["is_console"], probe["stdout"]["is_console"])
    else:
        verdict = "ALIVE_AFTER_WAIT" if alive else f"EXITED({proc.returncode})"
    record(name, verdict, {"argv": argv, "seconds": round(time.time() - t0, 1),
                           "killed_alive": alive, "exit_code": None if alive else proc.returncode,
                           "probe": probe})


def run_show(argv, env=None, timeout=600):
    argv = [str(a) for a in argv]
    print(f"$ {subprocess.list2cmdline(argv)}", flush=True)
    try:
        r = subprocess.run(argv, env=env, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, stdin=subprocess.DEVNULL)
        print(r.stdout[-6000:])
        print(r.stderr[-6000:])
        print(f"exit={r.returncode}", flush=True)
        return r
    except Exception as exc:
        print(f"error: {exc!r}", flush=True)
        return None

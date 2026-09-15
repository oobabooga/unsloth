"""Which per-process counters move when a child pages a model in, and which stay flat when it stalls.

usage:
  fault_probe.py mmap WORKDIR                 child pages a 256 MiB file in via mmap at ~16 MiB/s, keep then release
  fault_probe.py idle-http                    child is an HTTP server answering 503, probed every 0.5s
  fault_probe.py llama BINARY GGUF WORKDIR    real llama-server: counters during load, then idle while probed
"""

import json
import mmap
import os
import subprocess
import sys
import threading
import time
import urllib.request

import psutil

MiB = 1 << 20
SAMPLE_S = 5.0


def proc_stat_faults(pid):
    # /proc/<pid>/stat: minflt is field 10, majflt field 12 (1-based), after the ")" of comm.
    try:
        with open(f"/proc/{pid}/stat") as f:
            rest = f.read().rsplit(")", 1)[1].split()
        return int(rest[7]), int(rest[9])
    except OSError:
        return None


def counters(ps):
    with ps.oneshot():
        c = ps.cpu_times()
        mi = ps.memory_info()
        try:
            io = ps.io_counters()
        except (AttributeError, psutil.AccessDenied):
            io = None
    out = {"cpu": c.user + c.system, "rss": mi.rss}
    if io is not None:
        out["read_bytes"] = io.read_bytes
        if hasattr(io, "read_chars"):
            out["read_chars"] = io.read_chars
        for extra in ("read_count", "other_bytes"):
            if hasattr(io, extra):
                out[extra] = getattr(io, extra)
    for field in ("pfaults", "pageins", "num_page_faults"):
        if hasattr(mi, field):
            out[field] = getattr(mi, field)
    faults = proc_stat_faults(ps.pid)
    if faults:
        out["minflt"], out["majflt"] = faults
    return out


def monitor(label, ps, seconds, stop=None):
    prev = counters(ps)
    peak = prev["rss"]
    end = time.monotonic() + seconds
    while time.monotonic() < end and not (stop and stop.is_set()):
        time.sleep(SAMPLE_S)
        try:
            cur = counters(ps)
        except psutil.Error:
            return
        d = {}
        for k, v in cur.items():
            if k == "cpu":
                d["cpu_s"] = round(v - prev[k], 3)
            elif k == "rss":
                d["rss_MiB"] = round((v - prev[k]) / MiB, 1)
                d["peak_gain_MiB"] = round((max(peak, v) - peak) / MiB, 1)
                peak = max(peak, v)
            elif k.endswith("bytes") or k == "read_chars":
                d[k + "_MiB"] = round((v - prev[k]) / MiB, 1)
            else:
                d[k] = v - prev[k]
        print(label, json.dumps(d), flush=True)
        prev = cur


def make_cold_file(path, size):
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o644)
    if sys.platform == "darwin":
        import fcntl

        fcntl.fcntl(fd, 48, 1)  # F_NOCACHE: keep the written pages out of the buffer cache
    chunk = os.urandom(4 * MiB)
    for _ in range(size // len(chunk)):
        os.write(fd, chunk)
    os.fsync(fd)
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    if sys.platform == "win32":
        print("purge standby list:", purge_windows_standby(), flush=True)


def purge_windows_standby():
    """Drop the standby (file cache) list so mapped reads hit the disk. Needs an elevated token."""
    import ctypes
    from ctypes import wintypes

    advapi, ntdll, kernel = ctypes.WinDLL("advapi32"), ctypes.WinDLL("ntdll"), ctypes.WinDLL("kernel32")

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    class TOKEN_PRIVILEGES(ctypes.Structure):
        _fields_ = [("PrivilegeCount", wintypes.DWORD), ("Luid", LUID), ("Attributes", wintypes.DWORD)]

    token = wintypes.HANDLE()
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), 0x20 | 0x8, ctypes.byref(token)):
        return "OpenProcessToken failed"
    luid = LUID()
    if not advapi.LookupPrivilegeValueW(None, "SeProfileSingleProcessPrivilege", ctypes.byref(luid)):
        return "LookupPrivilegeValue failed"
    tp = TOKEN_PRIVILEGES(1, luid, 2)
    advapi.AdjustTokenPrivileges(token, False, ctypes.byref(tp), 0, None, None)
    command = ctypes.c_int(4)  # MemoryPurgeStandbyList
    status = ntdll.NtSetSystemInformation(80, ctypes.byref(command), ctypes.sizeof(command))
    return f"NTSTATUS 0x{status & 0xFFFFFFFF:08x}"


CHILD_MMAP = r"""
import mmap, os, sys, time
path, release = sys.argv[1], sys.argv[2] == "release"
fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
size = os.fstat(fd).st_size
chunk = 2 << 20
held = []
for off in range(0, size, chunk):
    m = mmap.mmap(fd, chunk, access=mmap.ACCESS_READ, offset=off)
    s = 0
    for p in range(0, chunk, mmap.PAGESIZE):
        s += m[p]
    if release:
        m.close()
    else:
        held.append(m)
    time.sleep(0.125)
time.sleep(600)
"""

CHILD_HTTP = r"""
import http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"error":{"message":"Loading model","type":"unavailable_error","code":503}}'
        self.send_response(503); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""


def prober(url, stop):
    while not stop.is_set():
        try:
            urllib.request.urlopen(url, timeout=2).read()
        except Exception:
            pass
        time.sleep(0.5)


def main():
    mode = sys.argv[1]
    print("platform", sys.platform, "pagesize", mmap.PAGESIZE, "psutil", psutil.__version__, flush=True)
    if mode == "mmap":
        work = sys.argv[2]
        os.makedirs(work, exist_ok=True)
        path = os.path.join(work, "probe.bin")
        for release in ("keep", "release"):
            make_cold_file(path, 256 * MiB)
            child = subprocess.Popen([sys.executable, "-c", CHILD_MMAP, path, release])
            time.sleep(1)
            monitor(f"mmap-{release}", psutil.Process(child.pid), 15)
            child.kill()
            child.wait()
        os.remove(path)
    elif mode == "mmap-existing":
        path = sys.argv[2]
        for release in ("keep", "release"):
            child = subprocess.Popen([sys.executable, "-c", CHILD_MMAP, path, release])
            time.sleep(1)
            monitor(f"mmap-existing-{release}", psutil.Process(child.pid), 15)
            child.kill()
            child.wait()
    elif mode == "idle-http":
        port = 18655
        child = subprocess.Popen([sys.executable, "-c", CHILD_HTTP, str(port)])
        time.sleep(2)
        stop = threading.Event()
        threading.Thread(target=prober, args=(f"http://127.0.0.1:{port}/health", stop), daemon=True).start()
        monitor("idle-http-probed", psutil.Process(child.pid), 30)
        stop.set()
        child.kill()
        child.wait()
    elif mode == "llama-stall":
        binary, gguf, pause, work = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
        port = 18657
        log = open(os.path.join(work, "llama-stall.log"), "w")
        child = subprocess.Popen(
            [binary, "-m", gguf, "--port", str(port), "-c", "2048", "-ngl", "0"],
            stdout=log, stderr=subprocess.STDOUT,
        )
        ps = psutil.Process(child.pid)
        stop = threading.Event()
        threading.Thread(target=prober, args=(f"http://127.0.0.1:{port}/health", stop), daemon=True).start()
        monitor("llama-loading-probed", ps, 15)
        open(pause, "w").close()
        time.sleep(3)
        monitor("llama-stalled-probed", ps, 40)
        stop.set()
        os.remove(pause)
        child.kill()
        child.wait()
    elif mode == "llama":
        binary, gguf, work = sys.argv[2], sys.argv[3], sys.argv[4]
        port = 18656
        log = open(os.path.join(work, "llama.log"), "w")
        child = subprocess.Popen(
            [binary, "-m", gguf, "--port", str(port), "-c", "2048", "-ngl", "0"],
            stdout=log, stderr=subprocess.STDOUT,
        )
        ps = psutil.Process(child.pid)
        stop = threading.Event()
        threading.Thread(target=prober, args=(f"http://127.0.0.1:{port}/health", stop), daemon=True).start()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 300:
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200:
                    break
            except Exception:
                pass
            time.sleep(0.2)
        print("llama healthy after", round(time.monotonic() - t0, 1), "s", flush=True)
        monitor("llama-healthy-probed", ps, 30)
        stop.set()
        child.kill()
        child.wait()


if __name__ == "__main__":
    main()

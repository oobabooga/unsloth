# A/B: does an SHChangeNotify from a child that exits immediately reach a registered listener,
# with and without SHCNF_FLUSH? Every child gets its own directory, so coalescing into a single
# UPDATEDIR cannot hide a delivery.
import os, subprocess, sys, time, statistics

py, work, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(work, exist_ok=True)
log = os.path.join(work, "ab.log"); stop = os.path.join(work, "ab.stop")
for f in (log, stop):
    if os.path.exists(f): os.remove(f)
listener = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), "listener.py"),
                             log, stop, "600", work])
for _ in range(200):
    if os.path.exists(log) and "READY" in open(log, encoding="utf-8").read(): break
    time.sleep(0.1)
print(open(log, encoding="utf-8").read().strip())

def probe(flags_item, flags_global):
    return ("import ctypes,sys\nfrom ctypes import wintypes\n"
            "s32=ctypes.WinDLL('shell32',use_last_error=True)\n"
            "s32.SHChangeNotify.restype=None\n"
            "s32.SHChangeNotify.argtypes=[wintypes.LONG,wintypes.UINT,wintypes.LPCWSTR,wintypes.LPCWSTR]\n"
            "for p in sys.argv[1:]:\n"
            f"    s32.SHChangeNotify(0x00002000,{flags_item},p,None)\n"
            f"s32.SHChangeNotify(0x08000000,{flags_global},None,None)\n"
            "sys.stdout.write('ok')")

variants = {"unflushed": probe("0x0005", "0"), "flushed": probe("0x1005", "0x1000")}
targets, times = {k: [] for k in variants}, {k: [] for k in variants}
for i in range(rounds):
    for name, script in variants.items():
        d = os.path.join(work, f"{name}-{i}"); os.makedirs(d, exist_ok=True)
        t = os.path.join(d, f"Unsloth Studio {name} {i} é中.lnk")
        open(t, "wb").close()
        start = time.time()
        r = subprocess.run([py, "-I", "-S", "-c", script, t], capture_output=True, text=True, timeout=30)
        times[name].append(time.time() - start)
        assert r.stdout == "ok", (r.returncode, r.stdout, r.stderr)
        targets[name].append(t)
        time.sleep(0.4)
time.sleep(5)
open(stop, "w").close(); listener.wait(timeout=30)
events = [l.split("\t") for l in open(log, encoding="utf-8").read().splitlines() if "\tEVENT\t" in l]
seen = {e[3].lower() for e in events if e[2] == "0x00002000"}
seen_dirs = {e[3].lower() for e in events if e[2] == "0x00001000"}
assoc = sum(1 for e in events if e[2] == "0x08000000")
print(f"total events {len(events)}, ASSOCCHANGED {assoc}")
for name in variants:
    got = sum(1 for t in targets[name] if t.lower() in seen or os.path.dirname(t).lower() in seen_dirs)
    print(f"{name}: delivered {got}/{len(targets[name])}  child time median {statistics.median(times[name]):.3f}s max {max(times[name]):.3f}s")

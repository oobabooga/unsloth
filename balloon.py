"""Hold host RAM so MemAvailable drops to about --leave GiB. Writes the held size to --out, then sleeps."""
import argparse, json, mmap, os, time

ap = argparse.ArgumentParser()
ap.add_argument("--leave", type=float, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    return 0.0


before = avail_gib()
want = max(0.0, before - a.leave)
held = []
chunk = 1 << 30
for _ in range(int(want)):
    m = mmap.mmap(-1, chunk, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    m[::4096] = b"\x01" * (chunk // 4096)  # touch every page
    held.append(m)
json.dump({"before_gib": before, "held_gib": len(held), "after_gib": avail_gib(), "pid": os.getpid()}, open(a.out, "w"))
while True:
    time.sleep(60)

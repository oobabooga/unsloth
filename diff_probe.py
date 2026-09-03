"""Diff two probe dumps, ignoring only fields that legitimately move between
processes (live memory readings on the AFFECTED Apple path)."""
import json, sys

base = json.load(open(sys.argv[1]))
head = json.load(open(sys.argv[2]))
apple = bool(base.get("is_apple_silicon")) and base.get("get_device") == "mlx"

# On Apple the free figure is the intended change; everything else must match.
INTENDED = {
    ("gpu_memory_info", "free_gb"),
    ("gpu_summary", "vram_free_gb"),
}
# Live counters that drift between two processes even on one machine.
VOLATILE = {
    ("gpu_memory_info", "free_gb"), ("gpu_memory_info", "allocated_gb"),
    ("gpu_memory_info", "reserved_gb"), ("gpu_memory_info", "utilization_pct"),
    ("gpu_memory_info", "total_gb"),
    ("gpu_summary", "vram_free_gb"),
}

diffs = []
for key in sorted(set(base) | set(head)):
    b, h = base.get(key), head.get(key)
    if b == h:
        continue
    if isinstance(b, dict) and isinstance(h, dict):
        for sub in sorted(set(b) | set(h)):
            if b.get(sub) == h.get(sub):
                continue
            tag = (key, sub)
            if apple and tag in INTENDED:
                diffs.append(("INTENDED", key, sub, b.get(sub), h.get(sub)))
            elif tag in VOLATILE:
                diffs.append(("VOLATILE", key, sub, b.get(sub), h.get(sub)))
            else:
                diffs.append(("UNEXPECTED", key, sub, b.get(sub), h.get(sub)))
    else:
        diffs.append(("UNEXPECTED", key, "", b, h))

for d in diffs:
    print(f"  {d[0]:10s} {d[1]}.{d[2]}: base={d[3]!r} head={d[4]!r}")
bad = [d for d in diffs if d[0] == "UNEXPECTED"]
print(f"\napple_path={apple}  total_diffs={len(diffs)}  unexpected={len(bad)}")
if bad:
    print("ROUTING PROBE FAIL: the PR changed something outside the Apple free figure")
    sys.exit(1)
print("ROUTING PROBE OK")

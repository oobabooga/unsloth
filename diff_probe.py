"""Structural + value diff of two probe dumps.

Two separate assertions:
  1. STRUCTURE: the set of keys each routing answer exposes must be identical,
     except the one key the PR intentionally adds on the Apple path.
  2. VALUES: every non-telemetry value must be identical. Live telemetry (power,
     temperature, utilisation, used/free memory) drifts between two processes on
     a real machine and is reported but not gated.
"""
import json, re, sys

base = json.load(open(sys.argv[1]))
head = json.load(open(sys.argv[2]))
apple = bool(base.get("is_apple_silicon")) and base.get("get_device") == "mlx"

IGNORE_TOP = {"label"}
# The PR's intended change, Apple path only.
INTENDED = {"gpu_memory_info.free_gb", "gpu_summary.vram_free_gb"}
INTENDED_RE = re.compile(r"^visible_utilization\.devices\[\d+\]\.vram_free_gb$")
# Live counters, inherently different between two processes on real hardware.
TELEMETRY = {"free_gb", "allocated_gb", "reserved_gb", "utilization_pct",
             "vram_free_gb", "vram_used_gb", "vram_utilization_pct",
             "gpu_utilization_pct", "power_draw_w", "power_utilization_pct",
             "temperature_c", "total_gb", "vram_total_gb"}


def paths(o, p=""):
    if isinstance(o, dict):
        for k in sorted(o):
            if not p and k in IGNORE_TOP:
                continue
            yield from paths(o[k], f"{p}.{k}" if p else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from paths(v, f"{p}[{i}]")
    else:
        yield p, o


b, h = dict(paths(base)), dict(paths(head))


def intended(p):
    return apple and (p in INTENDED or INTENDED_RE.match(p))


struct_only_head = [p for p in h if p not in b and not intended(p)]
struct_only_base = [p for p in b if p not in h]
val_diffs = [(p, b[p], h[p]) for p in sorted(set(b) & set(h)) if b[p] != h[p]]
telemetry = [d for d in val_diffs if d[0].rsplit(".", 1)[-1] in TELEMETRY]
material = [d for d in val_diffs if d not in telemetry and not intended(d[0])]

print(f"apple_path={apple}   probed paths: base={len(b)} head={len(h)}")
if apple:
    for p in sorted(set(h) - set(b)):
        print(f"  INTENDED new key   {p} = {h[p]!r}")
for p, x, y in telemetry:
    print(f"  telemetry (ignored) {p}: base={x!r} head={y!r}")
for p in struct_only_head:
    print(f"  STRUCTURE only in head: {p} = {h[p]!r}")
for p in struct_only_base:
    print(f"  STRUCTURE only in base: {p} = {b[p]!r}")
for p, x, y in material:
    print(f"  MATERIAL {p}: base={x!r} head={y!r}")

bad = len(struct_only_head) + len(struct_only_base) + len(material)
print(f"\nstructure_added={len(struct_only_head)} structure_removed={len(struct_only_base)} "
      f"material_value_diffs={len(material)} telemetry_drift={len(telemetry)}")
if bad:
    print("ROUTING PROBE FAIL: the PR changed a routing answer outside the Apple free figure")
    sys.exit(1)
print("ROUTING PROBE OK: identical structure and identical non-telemetry values")

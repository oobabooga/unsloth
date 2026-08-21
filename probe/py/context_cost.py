"""Measure what each torch/HIP telemetry call costs in GPU-visible memory on gfx1151."""
import os
import subprocess
import sys


def sysfs():
    out = {}
    for key in ("mem_info_vram_used", "mem_info_vram_total", "mem_info_gtt_used", "mem_info_gtt_total"):
        for card in ("card0", "card1"):
            p = "/sys/class/drm/%s/device/%s" % (card, key)
            if os.path.exists(p):
                try:
                    out[key] = int(open(p).read().strip())
                except Exception:
                    pass
                break
    return out


def rss():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return line.split()[1] + " kB"
    except Exception:
        pass
    return "?"


def report(tag):
    s = sysfs()
    print("%-34s vram_used=%-12s gtt_used=%-12s rss=%s" % (
        tag, s.get("mem_info_vram_used"), s.get("mem_info_gtt_used"), rss()), flush=True)


report("baseline (no torch)")
import torch  # noqa: E402
report("after import torch")
print("   is_available:", torch.cuda.is_available(), flush=True)
report("after is_available")
print("   device_count:", torch.cuda.device_count(), flush=True)
report("after device_count")
p = torch.cuda.get_device_properties(0)
print("   props.total_memory:", p.total_memory, "name:", p.name, "arch:", getattr(p, "gcnArchName", "?"), flush=True)
report("after get_device_properties")
print("   mem_get_info(0):", torch.cuda.mem_get_info(0), flush=True)
report("after mem_get_info")
x = torch.ones((8, 8), dtype=torch.float16, device="cuda")
print("   matmul:", (x @ x).sum().item(), flush=True)
report("after a real allocation")
print("context_cost.py finished cleanly", flush=True)

"""Allocate and touch GPU memory in 8 GiB steps under ROCm-on-WSL, printing the WSL VM's
MemAvailable after each step, to see which pool backs DXG allocations and where they stop."""

import sys
import time

import torch


def meminfo():
    fields = dict(line.split(":", 1) for line in open("/proc/meminfo"))
    return {k: int(fields[k].split()[0]) >> 20 for k in ("MemTotal", "MemAvailable", "SwapFree")}


free, total = torch.cuda.mem_get_info()
print(f"start mem_get_info free={free >> 30} GiB total={total >> 30} GiB vm={meminfo()}", flush = True)
held = []
step = 8 << 30
for index in range(16):
    started = time.monotonic()
    try:
        block = torch.empty(step, dtype = torch.uint8, device = "cuda")
        block.fill_(1)
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"step {index}: {type(exc).__name__}: {str(exc)[:200]}", flush = True)
        break
    held.append(block)
    free, total = torch.cuda.mem_get_info()
    print(
        f"step {index}: held {(index + 1) * 8} GiB in {time.monotonic() - started:.1f}s, "
        f"mem_get_info free={free >> 30} GiB, vm={meminfo()}",
        flush = True,
    )
sys.exit(0)

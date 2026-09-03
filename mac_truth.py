"""Does the reported free number predict what MLX can ACTUALLY allocate on a Mac?

Reports every candidate source for the ceiling, then allocates for real and compares
what succeeded against what each formula predicted.
"""
import json, os, subprocess, sys

sys.path.insert(0, os.path.abspath(sys.argv[1]))
GB = 1024 ** 3


def sysctl(key):
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except Exception as e:
        return f"<{e}>"


import psutil
import mlx.core as mx
from utils.hardware import hardware as hw

print("=" * 70)
print("SOURCES")
print("=" * 70)
info = mx.device_info()
print("mx.device_info():", json.dumps(info, indent=2, default=str))
rec = int(info.get("max_recommended_working_set_size") or 0)
memsize = int(sysctl("hw.memsize") or 0)
vm = psutil.virtual_memory()
print(f"hw.model                        = {sysctl('hw.model')}")
print(f"hw.memsize                      = {memsize} ({memsize / GB:.3f} GB)")
print(f"mlx memory_size                 = {info.get('memory_size')} "
      f"({int(info.get('memory_size') or 0) / GB:.3f} GB)")
print(f"mlx max_recommended_working_set = {rec} ({rec / GB:.3f} GB) "
      f"= {100 * rec / memsize:.1f}% of hw.memsize" if memsize else "")
print(f"mlx max_buffer_length           = {int(info.get('max_buffer_length') or 0) / GB:.3f} GB")
print(f"iogpu.wired_limit_mb            = {sysctl('iogpu.wired_limit_mb')}")
print(f"psutil total / available        = {vm.total / GB:.3f} / {vm.available / GB:.3f} GB")
print(f"mlx get_memory_limit()          = "
      f"{getattr(mx, 'get_memory_limit', lambda: None)() or 0} bytes")

# Is MLX necessary? torch exposes the same Metal property.
try:
    import torch
    trec = torch.mps.recommended_max_memory()
    print(f"torch.mps.recommended_max_memory= {trec} ({trec / GB:.3f} GB)  "
          f"MATCHES MLX: {trec == rec}")
except Exception as e:
    print(f"torch.mps.recommended_max_memory= unavailable ({type(e).__name__}: {e})")

print()
print("=" * 70)
print("WHAT EACH FORMULA PREDICTS")
print("=" * 70)
agx = hw._read_apple_gpu_stats()
allocated = (agx or {}).get("vram_used_bytes", 0) or 0
old = (vm.total - allocated) / GB
new = min(vm.available, rec) / GB if rec > 0 else vm.available / GB
print(f"agx ioreg stats                 = {agx}")
print(f"OLD (main): total - gpu_used    = {old:.3f} GB")
print(f"NEW (this PR): min(avail, rec)  = {new:.3f} GB")

print()
print("=" * 70)
print("REAL ALLOCATION: how much can MLX actually get?")
print("=" * 70)
held, total_held = [], 0.0
step = 0.25
peak = 0.0
try:
    while total_held < min(old, 64) + 1.0:
        a = mx.zeros((int(step * GB // 4),), dtype=mx.float32)
        mx.eval(a)
        held.append(a)
        total_held += step
        peak = total_held
        if total_held % 1.0 < step:
            av = psutil.virtual_memory().available / GB
            print(f"  held {total_held:6.2f} GB   psutil available now {av:6.2f} GB")
except Exception as e:
    print(f"  allocation stopped at {total_held:.2f} GB: {type(e).__name__}: {e}")
finally:
    actual = peak
    del held

print()
print("=" * 70)
print("VERDICT")
print("=" * 70)
print(f"actually allocated              = {actual:.2f} GB")
print(f"OLD predicted                   = {old:.2f} GB   "
      f"{'OVERSTATED by %.2f GB' % (old - actual) if old > actual else 'ok'}")
print(f"NEW predicted                   = {new:.2f} GB   "
      f"{'OVERSTATED by %.2f GB' % (new - actual) if new > actual else 'conservative by %.2f GB' % (actual - new)}")
print()
print("A prediction that OVERSTATES is the bug: Studio offers LoRA where it does not fit.")
print("A prediction that UNDERSTATES is the deliberate safety margin.")

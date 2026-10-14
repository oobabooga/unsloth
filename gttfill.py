"""Hold GTT so the free pool matches a target size: pins memory with hipHostMalloc, then sleeps."""
import ctypes, glob, sys, time

def gtt():
    d = [p.rsplit("/", 1)[0] for p in glob.glob("/sys/class/drm/card*/device/mem_info_gtt_total")]
    tot = sum(int(open(x + "/mem_info_gtt_total").read()) for x in d)
    used = sum(int(open(x + "/mem_info_gtt_used").read()) for x in d)
    return tot, used

target_free = float(sys.argv[1]) * 1024**3
hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
tot, used = gtt()
need = int(tot - used - target_free)
print(f"gtt total={tot/2**30:.2f} used={used/2**30:.2f} GiB; pinning {need/2**30:.2f} GiB", flush=True)
chunk = 1 << 30
held = []
while need > 0:
    n = min(chunk, need)
    p = ctypes.c_void_p()
    rc = hip.hipHostMalloc(ctypes.byref(p), ctypes.c_size_t(n), 0)
    if rc != 0:
        print("hipHostMalloc rc", rc, flush=True); break
    ctypes.memset(p, 1, n)
    held.append(p); need -= n
tot, used = gtt()
print(f"FILLED gtt used={used/2**30:.2f} free={(tot-used)/2**30:.2f} GiB", flush=True)
while True:
    time.sleep(60)

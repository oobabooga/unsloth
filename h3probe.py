"""Run sd-cli with Studio's H3 flags and record peak amdgpu GTT/VRAM use and min MemAvailable."""
import glob, json, os, subprocess, sys, threading, time

def sysfs(name):
    if os.name == "nt":
        shared, dedicated = win_sample()
        return shared if "gtt" in name else dedicated
    tot = 0
    for p in glob.glob(f"/sys/class/drm/card*/device/{name}"):
        try:
            tot += int(open(p).read().strip())
        except Exception:
            pass
    return tot

WIN = os.name == "nt"

def win_sample():
    """(shared, dedicated) GPU adapter memory in bytes, summed over adapters, via perf counters."""
    ps = ("$s=(Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage','\\GPU Adapter Memory(*)\\Dedicated Usage')"
          ".CounterSamples; $a=0;$b=0; foreach($x in $s){ if($x.Path -like '*shared usage'){$a+=$x.CookedValue}else{$b+=$x.CookedValue} }; [string]$a + ' ' + [string]$b")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=20).stdout.split()
        return int(float(out[0])), int(float(out[1]))
    except Exception:
        return 0, 0

def meminfo(key):
    if WIN:
        import ctypes
        class M(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tot", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                        ("pf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("v", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
        m = M(); m.l = ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.avail
    for line in open("/proc/meminfo"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) * 1024
    return 0

def main():
    label, binary, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    w = os.environ.get("VW", "640"); h = os.environ.get("VH", "384"); fr = os.environ.get("FRAMES", "25")
    M = os.environ["MODELS"]
    out = os.path.join(os.environ["OUT"], f"{label}.webm")
    cmd = [binary, "--mode", "vid_gen",
           "--diffusion-model", f"{M}/minimax_h3_fl2va_pruned-{os.environ.get('Q','Q4_K')}.gguf",
           "--vae", f"{M}/vae/minimax_h3_video_vae_fp16.safetensors",
           "--audio-vae", f"{M}/vae/minimax_h3_audio_vae_fp32.safetensors",
           "--llm", f"{M}/{os.environ.get('TE','qwen3vl_32b_minimax_h3-Q4_K_M.gguf')}",
           "--prompt", "a red fox running through snow, cinematic", "--cfg-scale", "1",
           "--width", w, "--height", h, "--rng", "cpu", "--fps", "24", "--video-frames", fr,
           "--steps", os.environ.get("STEPS", "2"), "--seed", "1", "--output", out, "-v", *extra]
    base = {"gtt": sysfs("mem_info_gtt_used"), "vram": sysfs("mem_info_vram_used"), "avail": meminfo("MemAvailable")}
    peak = dict(gtt=0, vram=0, avail_min=base["avail"])
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            peak["gtt"] = max(peak["gtt"], sysfs("mem_info_gtt_used"))
            peak["vram"] = max(peak["vram"], sysfs("mem_info_vram_used"))
            peak["avail_min"] = min(peak["avail_min"], meminfo("MemAvailable"))
            time.sleep(0.25 if not WIN else 0.5)
    t = threading.Thread(target=sample, daemon=True); t.start()
    print("CMD:", " ".join(cmd), flush=True)
    t0 = time.time()
    log = open(os.path.join(os.environ["OUT"], f"{label}.log"), "w")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    tail = []
    for line in p.stdout:
        log.write(line)
        tail.append(line.rstrip()); tail = tail[-60:]
        low = line.lower()
        if any(k in low for k in ("error", "radv", "lost", "abort", "assert", "budget", "segment", "prepared params", "vram", "fit")):
            print(f"[{time.time()-t0:7.1f}s] {line.rstrip()[:400]}", flush=True)
    rc = p.wait(); stop.set(); t.join()
    G = 1024**3
    res = {"label": label, "rc": rc, "secs": round(time.time()-t0, 1),
           "base_gtt_gib": round(base["gtt"]/G, 2), "peak_gtt_gib": round(peak["gtt"]/G, 2),
           "base_vram_gib": round(base["vram"]/G, 2), "peak_vram_gib": round(peak["vram"]/G, 2),
           "base_avail_gib": round(base["avail"]/G, 2), "min_avail_gib": round(peak["avail_min"]/G, 2),
           "output": os.path.exists(out) and os.path.getsize(out)}
    print("RESULT", json.dumps(res), flush=True)
    if rc != 0:
        print("\n".join(tail[-30:]))
    with open(os.path.join(os.environ["OUT"], "results.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")

main()

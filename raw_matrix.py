"""Run llama-server directly per load mode; record whether it loads, how long, peak memory, and speed."""
import argparse, json, os, re, signal, subprocess, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--server", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--logdir", required=True)
ap.add_argument("--port", type=int, default=8950)
ap.add_argument("--cases", default=None, help="JSON object: name -> extra argv")
a = ap.parse_args()
CASES = {"auto": [], "mmap": ["--load-mode", "mmap"], "lazy_on": ["--lazy-mode", "on"],
         "mmap_lazy_on": ["--load-mode", "mmap", "--lazy-mode", "on"]}
if a.cases:
    CASES = json.load(open(a.cases)) if a.cases.endswith(".json") else json.loads(a.cases)


def avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2


def swap_used_gib():
    vals = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith(("SwapTotal", "SwapFree"))}
    return (vals.get("SwapTotal", 0) - vals.get("SwapFree", 0)) / 1024**2


results = {}
for name, extra in CASES.items():
    log = os.path.join(a.logdir, f"raw-{name}.log")
    cmd = [a.server, "-m", a.model, "--port", str(a.port), "-c", "4096", "-np", "1", "--flash-attn", "on",
           "--fit", "on", "--no-mmproj", "-lv", "3", *extra]
    t0 = time.time(); r = {"cmd": " ".join(cmd[1:])}
    p = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    min_avail = avail_gib(); hwm = 0; ok = False; swap0 = swap_used_gib(); swap_peak = swap0
    while time.time() - t0 < 1500:
        if p.poll() is not None:
            break
        min_avail = min(min_avail, avail_gib()); swap_peak = max(swap_peak, swap_used_gib())
        try:
            hwm = max(hwm, int(re.search(r"VmHWM:\s+(\d+)", open(f"/proc/{p.pid}/status").read()).group(1)) / 1024**2)
        except Exception:
            pass
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=5) as h:
                if h.status == 200:
                    ok = True
                    break
        except Exception:
            pass
        time.sleep(2)
    r.update(loaded=ok, load_seconds=round(time.time() - t0, 1), exit=p.poll())
    if ok:
        for tag, prompt, n in (("short", "Say hello in five words.", 32), ("long", "Summarize the history of the printing press. " * 60, 128)):
            body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0, "cache_prompt": False}).encode()
            t1 = time.time()
            try:
                with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{a.port}/completion", data=body, headers={"Content-Type": "application/json"}), timeout=1800) as resp:
                    out = json.loads(resp.read())
                r[tag] = {"seconds": round(time.time() - t1, 1), "timings": out.get("timings"), "text": out.get("content", "")[:120]}
            except Exception as e:
                r[tag] = {"error": repr(e)[:300]}
            min_avail = min(min_avail, avail_gib())
            try:
                hwm = max(hwm, int(re.search(r"VmHWM:\s+(\d+)", open(f"/proc/{p.pid}/status").read()).group(1)) / 1024**2)
            except Exception:
                pass
    r.update(min_avail_gib=round(min_avail, 2), peak_rss_gib=round(hwm, 2), swap_used_start_gib=round(swap0, 2), swap_used_peak_gib=round(swap_peak, 2))
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        pass
    p.wait()
    txt = open(log, errors="replace").read()
    r["log"] = [l for l in txt.splitlines() if re.search(r"load_mode =|lazy read|failed to allocate|unable to allocate|error loading|out of memory|model buffer size|file size|projected to use|Killed", l)][:40]
    results[name] = r
    print(name, json.dumps({k: r.get(k) for k in ("loaded", "load_seconds", "exit", "peak_rss_gib", "min_avail_gib")}), flush=True)
    time.sleep(10)
json.dump(results, open(a.out, "w"), indent=1)

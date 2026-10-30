#!/usr/bin/env python3
"""k15_bench.py: llama.cpp k15 (cea00b81800) vs k10 (a162dce854) on gfx1151, one platform (Linux / Windows), stdlib only.

Derived from the k10s k6p_bench.py (ci_k6_amd 737e0eb). Every measurement is a fresh process; arm order rotates per rep.
env: K6P_DIR work root (bins/<arm dir>, models/, out/), ARMS "name:dir[:VAR=VAL;...]", K6P_DEV device (ROCm0).
arm keys (third ARMS field, not exported): K6P_PAD=N sets K15_PAD to N bytes ("" for 0: the environment-layout arm),
K6P_MOE=N|auto the --moe-cache-mib value, K6P_ARGS="..." extra server arguments, K6P_NO_MOE=1 no --moe-cache-mib.
cells: model:mode, mode full = --fit on, spillN = --fit on --fit-target (free - N MiB); MoE models add --moe-cache-mib
(K6P_MOE, default auto). Per launch (llama-server -c 8192 -fa on -np 1): warm 8 tokens, dec = 40-token prompt + 128 greedy,
p2k = 2000 token ids + 128 greedy; md5 of the text + token ids; per-launch counters of the k15 log lines (memstats).
usage: k15_bench.py dl --models q35,q3m | devinfo | time --cell M:mode --reps N | tbo --bin ARM --ops "OP[@REGEX];..." | ALL
       | exe --bin ARM --exe NAME
rows -> K6P_DIR/out/rows.jsonl (also printed as ROW lines), filtered logs -> K6P_DIR/out/logs/
"""
import argparse, glob, hashlib, json, os, platform, re, shutil, signal, subprocess, sys, threading, time, urllib.request

K = os.environ.get("K6P_DIR") or os.getcwd()
OUT = os.environ.get("K6P_OUT") or os.path.join(K, "out")   # k15: K6P_OUT inside the uploaded dir, so rows survive a step timeout
LOGS = os.path.join(OUT, "logs")
os.makedirs(LOGS, exist_ok=True)
WIN = os.name == "nt"
EXE = ".exe" if WIN else ""
DEV = os.environ.get("K6P_DEV", "ROCm0")
T0 = time.time()

ap = argparse.ArgumentParser()
ap.add_argument("what")
ap.add_argument("--models", default="")
ap.add_argument("--model", default="")
ap.add_argument("--cell", default="")
ap.add_argument("--modes", default="full")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--arms", default="")
ap.add_argument("--bin", default="k6")
ap.add_argument("--ops", default="")
ap.add_argument("--tbmode", default="test")
ap.add_argument("--exe", default="")
ap.add_argument("--chunks", type=int, default=int(os.environ.get("KLD_CHUNKS", "40")))
ap.add_argument("--timeout", type=int, default=2400)
a = ap.parse_args()

MODELS = {
    "q35": ("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf", "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0", 532517120, False),
    # k15 r2: the smallest arch with a lazy-marked PLE table (gemma4 per_layer_tok_embd, TENSOR_READ_LAZY); --lazy-mode on makes it lazy
    "g4e2b": ("unsloth/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q4_K_M.gguf", "0314792d7f1f7e229411f620751375812bb9faf2", 3106738272, False),
    "q3m": ("unsloth/Qwen3.6-35B-A3B-MTP-GGUF", "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf", "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d", 17104402720, True),
}
SHARDS = {}
MD = os.environ.get("MODEL_DIR", os.path.join(K, "models"))


def log(*x):
    print("[%6.0fs]" % (time.time() - T0), *x, flush=True)


def rd(p):
    with open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def find_bindir(d):
    for root, _, files in os.walk(d):
        if "llama-server" + EXE in files:
            return root
    return None


# ARMS "name:dir[:VAR=VAL;VAR2=VAL2]": the optional third field is extra environment for that arm (kill-switch arms)
ARMS_SPEC = os.environ.get("ARMS", "rel:rel,rel2:rel,k6:k6,k6b:k6")
BINS, ARM_ENV = {}, {}
for x in ARMS_SPEC.split(","):
    n, d = x.split(":")[:2]
    BINS[n] = find_bindir(os.path.join(K, "bins", d)) or os.path.join(K, "bins", d)
    ARM_ENV[n] = dict(kv.split("=", 1) for kv in x.split(":", 2)[2].split(";") if kv) if x.count(":") >= 2 else {}
ARMS = [x.split(":")[0] for x in ARMS_SPEC.split(",")]
if a.arms:
    ARMS = a.arms.split(",")
TAG = {}
for n, d in BINS.items():
    for c in (os.path.join(d, "TAG"), os.path.join(d, "..", "TAG")):
        if os.path.exists(c):
            TAG[n] = rd(c).strip()
            break


INTERNAL = ("K6P_PAD", "K6P_MOE", "K6P_ARGS", "K6P_NO_MOE")


def benv(arm):
    e = dict(os.environ)
    ae = ARM_ENV.get(arm, {})
    e.update({k: v for k, v in ae.items() if k not in INTERNAL})
    if "K6P_PAD" in ae:
        # the environment-layout arm: N bytes of padding move the heap / stack layout, so a graph replayed only because addresses
        # happened to coincide shows up as a different result (k15 standard gate TBOPAD)
        e["K15_PAD"] = "x" * int(ae["K6P_PAD"])
    d = BINS[arm]
    if WIN:
        e["PATH"] = d + os.pathsep + e.get("PATH", "")
    elif sys.platform == "darwin":
        e["DYLD_LIBRARY_PATH"] = d + os.pathsep + e.get("DYLD_LIBRARY_PATH", "")
    else:
        e["LD_LIBRARY_PATH"] = d + os.pathsep + e.get("LD_LIBRARY_PATH", "")
    return e


def binp(arm, name):
    return os.path.join(BINS[arm], name + EXE)


def sh(cmd, env=None, timeout=600):
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, env=env, timeout=timeout,
                           encoding="utf-8", errors="replace")
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return "ERR %r" % e


SYSFS = sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_used"))


def amd_used():
    """Strix Halo allocates from both the VRAM carve-out and GTT: report the sum (MiB), -1 if no sysfs"""
    if not SYSFS:
        return -1
    tot = 0
    for f in SYSFS:
        for k in ("mem_info_vram_used", "mem_info_gtt_used"):
            try:
                tot += int(rd(f.replace("mem_info_vram_used", k))) >> 20
            except Exception:
                pass
    return tot


def meminfo():
    if os.path.exists("/proc/meminfo"):
        s = rd("/proc/meminfo")
        g = lambda k: int((re.search(k + r":\s+(\d+) kB", s) or [0, 0])[1]) // 1024
        return {"total": g("MemTotal"), "avail": g("MemAvailable"), "cached": g("Cached")}
    return None


HOST = {"os": platform.platform(), "machine": platform.machine(), "nproc": os.cpu_count(), "runner": os.environ.get("RUNNER_NAME", ""),
        "mem": meminfo(), "sysfs": {os.path.basename(f): rd(f).strip() for f in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_*_total"))}}


def proc_mem(pid):
    """current host memory of pid (MiB): Linux /proc VmRSS / RssAnon / RssFile, Windows working set / private bytes (psapi)"""
    if WIN:
        try:
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(k, ctypes.c_size_t) for k in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.OpenProcess.restype = wintypes.HANDLE
            h = k32.OpenProcess(0x1000 | 0x0010, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ
            if not h:
                return None
            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            ok = k32.K32GetProcessMemoryInfo(h, ctypes.byref(c), c.cb)
            k32.CloseHandle(h)
            if not ok:
                return None
            # anon = private bytes (commit charge of the process), commit = PagefileUsage (the same counter on NT), peak_commit = PeakPagefileUsage
            return {"rss": c.WorkingSetSize >> 20, "anon": c.PrivateUsage >> 20, "file": None, "peak_ws": c.PeakWorkingSetSize >> 20,
                    "commit": c.PagefileUsage >> 20, "peak_commit": c.PeakPagefileUsage >> 20}
        except Exception:
            return None
    try:
        s = rd("/proc/%d/status" % pid)
    except Exception:
        return None
    g = lambda k: int((re.search(k + r":\s+(\d+) kB", s) or [0, 0])[1]) // 1024
    return {"rss": g("VmRSS"), "anon": g("RssAnon"), "file": g("RssFile"), "peak_ws": g("VmHWM"), "vmdata": g("VmData"), "vmsize": g("VmSize"),
            "shmem": g("RssShmem")}


class Sampler:
    """peak device memory used (Linux amdgpu sysfs, minus the baseline before the process) and peak host RSS (Linux /proc; Windows
    working set + private bytes)"""

    def __init__(self, pid):
        self.pid, self.base = pid, amd_used()
        self.gpu_peak, self.rss, self.anon, self.file = -1, 0, 0, 0
        self.stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self.stop:
            if SYSFS:
                self.gpu_peak = max(self.gpu_peak, amd_used())
            m = proc_mem(self.pid)
            if m:
                self.rss, self.anon, self.file = max(self.rss, m["rss"] or 0), max(self.anon, m["anon"] or 0), max(self.file, m["file"] or 0)
            time.sleep(0.2)

    def done(self):
        self.stop = True
        time.sleep(0.3)
        return {"dev_peak_mib": self.gpu_peak - self.base if self.gpu_peak >= 0 else None, "dev_base_mib": self.base,
                "rss_peak_mib": self.rss or None, "rss_anon_peak_mib": self.anon or None, "rss_file_peak_mib": self.file or None}


PAT = re.compile(r"buffer size|memory breakdown|\| +- |MoE cache|moe_cache|out of memory|error|failed|GGML_ASSERT|offload|load_tensors|"
                 r"n_ubatch|n_batch|mmap|load.mode|ROCm0|MTL0|ggml_cuda_init|Device 0|system_info|build|fit|pinned|host|prefetch|lend|"
                 r"weight.stream|integrated|unified|warning|abort|exception|non-consecutive|PLE|lazy|draft length|without the graph|gather kept|draft|speculative", re.I)


def memstats(s):
    g = lambda p: re.findall(p, s)
    return {"model_buf": [l.strip()[:200] for l in g(r"[^\n]*model buffer size[^\n]*")][:6],
            "compute_buf": [l.strip()[:200] for l in g(r"[^\n]*compute buffer size[^\n]*")][-4:],
            "kv_buf": [l.strip()[:200] for l in g(r"[^\n]*KV buffer size[^\n]*")][:2],
            "breakdown": [l.strip()[:240] for l in g(r"[^\n]*memory_breakdown[^\n]*")][:8],
            "moe_cache": [l.strip()[:260] for l in g(r"[^\n]*(?:MoE cache|llama_moe_cache)[^\n]*")][:12],
            "fit": [l.strip()[:260] for l in g(r"[^\n]*(?:common_fit|llama_params_fit|fit_params)[^\n]*")][:16],
            "ubatch": [l.strip()[:200] for l in g(r"[^\n]*n_ubatch[^\n]*")][:3],
            "load_mode": [l.strip()[:200] for l in g(r"[^\n]*(?:load mode|load_mode|--load-mode|with mmap|use_mmap)[^\n]*")][:4],
            "oom": bool(re.search(r"out of memory|failed to allocate|hipMalloc failed|cudaMalloc failed", s)),
            "abort": bool(re.search(r"GGML_ASSERT|Segmentation|Aborted|core dumped|CUDA error|HIP error|ROCm error|exception|ggml_abort", s)),
            "err_lines": [l.strip()[:240] for l in g(r"[^\n]*(?:GGML_ASSERT|CUDA error|HIP error|error:|ERROR|terminate called)[^\n]*")][:4],
            # k9: layer-major groups (merged graphs), refusals / splits, the auto-on line, the device-seed fallback, prefetch disabled
            "n_batch": [int(x) for x in g(r"llama_context: n_batch\s+=\s+(\d+)")][:3],
            "n_ubatch_ctx": [int(x) for x in g(r"llama_context: n_ubatch\s+=\s+(\d+)")][:3],
            "lm_n_groups": len(g(r"decode_layer_major: \d+ ubatches")),
            "lm_lines": [l.strip()[-260:] for l in g(r"[^\n]*decode_layer_major:[^\n]*")][:12],
            "lm_cfg": [l.strip()[-240:] for l in g(r"[^\n]*layer-major[^\n]*") if "CMD" not in l][:6],
            "devseed": [l.strip()[-260:] for l in g(r"[^\n]*(?:device seed from prompt|host seed of a prompt does not fit|seed the cache on the device)[^\n]*")][:4],
            "prefetch_disabled": len(g(r"host weight prefetch disabled")),
            "spec": [l.strip()[-200:] for l in g(r"[^\n]*(?:draft acceptance|n_draft|accept)[^\n]*")][-2:],
            # kpp: the unified-memory rule line (every offload device reported GGML_BACKEND_DEVICE_TYPE_IGPU, i.e. hipDeviceProp.integrated)
            "uma": [l.strip()[-200:] for l in g(r"[^\n]*MoE cache on unified memory[^\n]*")][:2],
            # k10: the explicit-ub n_batch raise (absent on Windows unified memory: KEEP_BATCH), the CPU-miss calibration / trials / totals
            "lm_explicit_ub": [l.strip()[-220:] for l in g(r"[^\n]*layer-major prompts at the given n_ubatch[^\n]*")][:2],
            "cpu_miss": [l.strip()[-300:] for l in g(r"[^\n]*(?:CPU misses|host computes missing experts|host computes the missing)[^\n]*")][:16],
            "cpu_miss_n": len(g(r"CPU misses trial \d+")),
            "resident_ub": [l.strip()[-200:] for l in g(r"[^\n]*MoE experts on the device[^\n]*")][:2],
            "dev_lines": [l.strip()[-200:] for l in g(r"[^\n]*(?:Device 0:|using device|integrated)[^\n]*")][:4],
            # k10s: recurrent find_slot warning spam (be905cf), PLE reader / prefetch lines, the zero-draft-length line (31b3863)
            "nonconsec_n": len(g(r"non-consecutive token position")),
            "ple": [l.strip()[-240:] for l in g(r"[^\n]*(?:PLE|ple_prefetch|ple-prefetch|direct reads|lazy)[^\n]*")][:10],
            "spec_off": len(g(r"draft length is 0")),
            # k15: eager fallback after a failed graph instantiation (be6d2579151), GDN in-place gather kept (f4345bce9db, with
            # GGML_CUDA_GDN_STATE_IN_PLACE_DEBUG=1), the post-draft free-memory floor (828b560b09c), the MTP draft context
            "eager_fallback_n": len(g(r"this evaluation runs without the graph")),
            "eager_fallback": [l.strip()[-260:] for l in g(r"[^\n]*this evaluation runs without the graph[^\n]*")][:3],
            "gather_kept_n": len(g(r"gather kept")),
            "gather_kept": [l.strip()[-220:] for l in g(r"[^\n]*gather kept[^\n]*")][:3],
            "draft_floor_n": len(g(r"MiB free after the draft was created")),
            "draft_floor": [l.strip()[-220:] for l in g(r"[^\n]*MiB free after the draft was created[^\n]*")][:2],
            "draft_off_n": len(g(r"the draft could not be created")),
            "mtp_ctx_fail_n": len(g(r"failed to create MTP context")),
            "spec_lines": [l.strip()[-220:] for l in g(r"[^\n]*(?:draft acceptance rate|n_draft|speculative|draft-mtp|MTP)[^\n]*") if "CMD" not in l][:8],
            "graph_lines": [l.strip()[-220:] for l in g(r"[^\n]*(?:CUDA graph|cuda_graph|cudaGraph|hipGraph)[^\n]*")][:6],
            # k15 r2 (c5429be4abd): single-graph host misses / mailbox (c98b2f662f2, 5f9e82fd578: CUDA only, HIP has no mapped mailbox),
            # PLE read-in (ab9df2b3658 / d9aaecc4448 / 1595d7a813d: Linux only), replay-commit fallback (1811f0c7468)
            "single_on_n": len(g(r"single-graph host misses on")),
            "single_nomb_n": len(g(r"no mapped mailbox|has no mapped mailboxes")),
            "ple_resident_n": len(g(r"PLE table '[^']*' \([^)]*\) made resident")),
            "ple_lazy_n": len(g(r"PLE table '[^']*' \([^)]*\) left lazy")),
            "ple_res_lines": [l.strip()[-260:] for l in g(r"[^\n]*PLE table '[^\n]*")][:3],
            "rs_fallback_commits": max([int(x) for x in g(r"(\d+) commits in the reserved fallback buffer")] or [0]),
            "rs_fallback_warn_n": len(g(r"runs in the reserved fallback buffer")),
            "rs_commits": max([int(x) for x in g(r"recurrent replay commits: (\d+)")] or [0]),
            "rs_lines": [l.strip()[-220:] for l in g(r"[^\n]*(?:replay commit|reserved fallback)[^\n]*")][:4],
            # r4 (c5429be4abd, release-split deb1079e81c + pool trim a4b47119d1a + stream creation c5429be4abd). The log lines are capped
            # in the code (decision 4, bank release 2 per cache, trim 4, free-after 4), so these are counts up to the caps
            "relsplit_n": len(g(r"the cache banks are released for it")),
            "bank_release_n": len(g(r"MoE cache banks released \(")),
            "pool_trim_n": len(g(r"MiB of CUDA pool memory the released group took given back")),
            "free_after_release_n": len(g(r"MiB free after the released banks returned")),
            # r5: the lend free check (f3a105b68e3) lends the graph in one part and frees the own buffer (the scheduler trim 0363724a80a logs nothing)
            "lend_whole_n": len(g(r"is lent in one part and the own compute buffer")),
            # the prefetch reserve check (0a4c75c3eb6): runs without the prefetch instead of a failed allocation
            "pf_notused_n": len(g(r"host weight prefetch not used")),
            "oom_n": len(g(r"out of memory")),
            "release_lines": [l.strip()[-240:] for l in g(r"[^\n]*(?:banks are released for it|MoE cache banks released|pool memory the released group|free after the released banks)[^\n]*")][:6],
            "load_mode_lines": [l.strip()[-200:] for l in g(r"[^\n]*(?:load mode|lazy mode|lazy_mode)[^\n]*")][:4]}


def filt(lp, s, head=120):
    lines = s.splitlines()
    keep = lines[:head] + ["..."] + [l[:320] for l in lines[head:] if PAT.search(l)][:400] + ["--- tail ---"] + lines[-60:]
    with open(lp.replace(".log", ".filtered.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(keep))


def row(r):
    r.update({"host": HOST, "tags": TAG, "t": round(time.time() - T0)})
    line = json.dumps(r)
    with open(os.path.join(OUT, "rows.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print("ROW " + line[:1500], flush=True)


# ---------------------------------------------------------------- downloads
def download(name):
    if name in SHARDS:
        ok = True
        for fname, size in SHARDS[name]:
            ok = download_file(name, MODELS[name][0], fname, MODELS[name][2], size) is not None and ok
        return mpath(name) if ok else None
    repo, fname, rev, size, _ = MODELS[name]
    return download_file(name, repo, fname, rev, size)


def download_file(name, repo, fname, rev, size):
    os.makedirs(MD, exist_ok=True)
    p = os.path.join(MD, fname)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    du = shutil.disk_usage(MD)
    if not (os.path.exists(p) and os.path.getsize(p) == size) and du.free < size + (20 << 30):
        log("NO DISK for", fname, "free GiB", du.free >> 30)
        row({"kind": "dl", "model": name, "file": fname, "ok": False, "err": "disk", "free": du.free})
        return None
    if os.path.exists(p) and os.path.getsize(p) == size:
        return p
    url = "https://huggingface.co/%s/resolve/%s/%s" % (repo, rev, fname)
    t = time.time()
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(p + ".part", "wb") as fh:
                shutil.copyfileobj(r, fh, 16 << 20)
            os.replace(p + ".part", p)
            if os.path.getsize(p) == size:
                break
            log("size mismatch", fname, os.path.getsize(p), size)
        except Exception as e:
            log("download retry", fname, attempt, repr(e)[:200])
            time.sleep(10)
    ok = os.path.exists(p) and os.path.getsize(p) == size
    log("download", fname, ok, "%.2f GiB in %.0f s" % (size / 2**30, time.time() - t))
    row({"kind": "dl", "model": name, "file": fname, "ok": ok, "s": round(time.time() - t)})
    return p if ok else None


def mpath(name):
    return os.path.join(MD, MODELS[name][1])


# ---------------------------------------------------------------- devices
def list_devices(arm):
    s = sh([binp(arm, "llama-server"), "--list-devices"], env=benv(arm), timeout=120)
    devs = re.findall(r"^\s*(\S+): (.*?) \((\d+) MiB, (\d+) MiB free\)", s, re.M)
    return s, [{"name": d[0], "desc": d[1], "total": int(d[2]), "free": int(d[3])} for d in devs]


def dev_free(arm):
    _, devs = list_devices(arm)
    for d in devs:
        if d["name"] == DEV:
            return d["free"]
    return None


def do_devinfo():
    for arm in dict.fromkeys(BINS[x] for x in ARMS):
        n = [k for k in ARMS if BINS[k] == arm][0]
        s, devs = list_devices(n)
        v = sh([binp(n, "llama-server"), "--version"], env=benv(n), timeout=60)
        log("DEVINFO", n, devs, v.splitlines()[-2:] if v else v)
        row({"kind": "devinfo", "arm": n, "devs": devs, "list_devices": s[-1500:], "version": v[-600:]})


# ---------------------------------------------------------------- server cells
def post(port, path, body, timeout=3600):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def complete(port, prompt, n):
    body = {"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 42, "ignore_eos": True, "cache_prompt": False,
            "return_tokens": True, "stream": False}
    t0 = time.time()
    r = post(port, "/completion", body)
    t = r.get("timings", {})
    text = r.get("content") or ""
    toks = r.get("tokens")
    return {"wall_s": round(time.time() - t0, 3), "n_prompt": t.get("prompt_n"), "pp_tps": round(t.get("prompt_per_second", 0), 2),
            "n_gen": t.get("predicted_n"), "tg_tps": round(t.get("predicted_per_second", 0), 3),
            "md5": hashlib.md5(text.encode("utf-8")).hexdigest(), "tok_md5": hashlib.md5(json.dumps(toks).encode()).hexdigest() if toks else None,
            "head": text[:80]}


DEC_PROMPT = ("Write a detailed, step by step explanation of how a hash table handles collisions, comparing separate chaining "
              "with open addressing, and give a short example in C.")
PORT = [18800]


# K6P_P8K=N: a third cell per launch, N wiki token ids (7900 fits -c 8192 with 128 generated) + 128 greedy, then the host memory
P8K = int(os.environ.get("K6P_P8K", "0") or 0)


def mem_snap(pid, smp):
    """host memory of the server now (idle after load / after a cell) + the amdgpu sysfs device use above the pre-launch baseline"""
    time.sleep(1.0)
    m = proc_mem(pid) or {}
    d = amd_used()
    m["dev_mib"] = d - smp.base if d >= 0 and smp.base >= 0 else None
    return m


def prompt_tokens(port, model, n=2000):
    f = os.path.join(K, "prompt_%s%s.json" % (model, "" if n == 2000 else "_%d" % n))
    if not os.path.exists(f):
        src = rd(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wiki.test.raw"))[:200000]
        toks = post(port, "/tokenize", {"content": src, "add_special": True})["tokens"][:n]
        with open(f, "w", encoding="utf-8") as fh:
            json.dump(toks, fh)
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)


def mode_flags(model, mode, target, arm=None):
    """mode[_ubN]: full / spillN / cpu, _ubN appends -ub N. Arm keys (ARMS third field): K6P_NO_MOE=1 drops --moe-cache-mib (plain
    upstream does not know it), K6P_ARGS="..." appends arguments (e.g. -ub 4096)."""
    moe = MODELS[model][4] and not ARM_ENV.get(arm, {}).get("K6P_NO_MOE")
    ub = None
    if "_ub" in mode:
        mode, ub = mode.split("_ub")
    f = []
    if mode == "cpu":
        f += ["-ngl", "0"]
    else:
        f += ["--fit", "on"]
        if mode.startswith("spill") and target is not None:
            f += ["--fit-target", str(target)]
    if moe and mode != "cpu":
        f += ["--moe-cache-mib", ARM_ENV.get(arm, {}).get("K6P_MOE") or os.environ.get("K6P_MOE") or "auto"]
    if ub:
        f += ["-ub", ub]
    return f + os.environ.get("K6P_EXTRA", "").split() + ARM_ENV.get(arm, {}).get("K6P_ARGS", "").split()


def kill(p):
    try:
        if WIN:
            p.terminate()
        else:
            os.killpg(p.pid, signal.SIGINT)
        p.wait(120)
    except Exception:
        try:
            if WIN:
                p.kill()
            else:
                os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
        try:
            p.wait(30)
        except Exception:
            pass


def popen(argv, fh, arm):
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WIN else {"start_new_session": True}
    p = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, env=benv(arm), cwd=BINS[arm], **kw)
    # k15: the live child's pid, so k15_plan.py can stop it (and its group) when a step overruns its budget
    with open(os.path.join(K, "child.pid"), "w", encoding="utf-8") as pf:
        pf.write(str(p.pid))
    return p


def run_server(arm, model, mode, target, tag):
    PORT[0] += 1
    port = PORT[0]
    lp = os.path.join(LOGS, tag + ".log")
    argv = [binp(arm, "llama-server"), "-m", mpath(model), "-c", "8192", "-fa", "on", "-np", "1", "--host", "127.0.0.1",
            "--port", str(port), "-lv", "4"] + mode_flags(model, mode, target, arm)
    fh = open(lp, "w", encoding="utf-8")
    fh.write("CMD %s\n" % " ".join(argv))
    fh.flush()
    t0 = time.time()
    p = popen(argv, fh, arm)
    smp = Sampler(p.pid)
    ok = False
    while time.time() - t0 < 1800:
        if p.poll() is not None:
            break
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    ok = True
                    break
        except Exception:
            pass
        time.sleep(0.25)
    res = {"load_s": round(time.time() - t0, 2), "ok": ok, "argv": argv[1:]}
    if ok:
        try:
            res["mem_idle"] = mem_snap(p.pid, smp)
            res["warm"] = complete(port, "The capital of France is", 8)
            res["dec"] = complete(port, DEC_PROMPT, 128)
            res["mem_dec"] = mem_snap(p.pid, smp)
            res["p2k"] = complete(port, prompt_tokens(port, model), 128)
            res["mem_p2k"] = mem_snap(p.pid, smp)
            if P8K:
                res["p8k"] = complete(port, prompt_tokens(port, model, P8K), 128)
                res["mem_p8k"] = mem_snap(p.pid, smp)
        except Exception as e:
            body = ""
            try:
                body = e.read().decode()[:200]
            except Exception:
                pass
            res["err"] = (repr(e) + " " + body)[:300]
    else:
        res["err"] = "server not ready rc=%s" % p.poll()
    alive = p.poll() is None
    kill(p)
    res["rc"] = p.returncode
    res["died_before_kill"] = not alive
    res.update(smp.done())
    fh.close()
    s = rd(lp)
    res["stats"] = memstats(s)
    filt(lp, s)
    os.remove(lp)
    return res


def warm_cache(model):
    t = time.time()
    for f in ([os.path.join(MD, x[0]) for x in SHARDS[model]] if model in SHARDS else [mpath(model)]):
        with open(f, "rb") as fh:
            while fh.read(64 << 20):
                pass
    log("page cache warm %s %.0f s" % (model, time.time() - t), meminfo())


_TGT = {}


def cell_target(model, mode, arm):
    """spillN: --fit-target = this arm's own free device memory - N MiB, so every arm gets the same N MiB budget even when the
    builds' HIP runtimes report different free memory (run 2 Windows: release 100414 MiB vs TheRock 7.14 build 78471 MiB)"""
    if not mode.startswith("spill"):
        return None, None
    budget = int(mode[5:].split("_")[0])
    key = (BINS[arm], tuple(sorted(ARM_ENV.get(arm, {}).items())))
    if key not in _TGT:
        _TGT[key] = dev_free(arm)
    free = _TGT[key]
    if free is None:
        return None, budget
    return max(free - budget, 0), budget


def do_time():
    model, mode = a.cell.split(":")
    if not os.path.exists(mpath(model)):
        log("no model", model)
        row({"kind": "server", "model": model, "mode": mode, "err": "model missing"})
        return
    warm_cache(model)
    for arm in ARMS:
        log("cell", a.cell, arm, "fit-target / budget", cell_target(model, mode, arm))
    for rep in range(a.reps):
        order = ARMS[rep % len(ARMS):] + ARMS[:rep % len(ARMS)]
        for arm in order:
            target, budget = cell_target(model, mode, arm)
            tag = "%s_%s_%s_r%d" % (model, mode, arm, rep)
            s = run_server(arm, model, mode, target, tag)
            row(dict(s, kind="server", model=model, mode=mode, arm=arm, rep=rep, fit_target=target, budget=budget, arm_env=ARM_ENV.get(arm)))
            d, p, q = s.get("dec") or {}, s.get("p2k") or {}, s.get("p8k") or {}
            log(tag, "ok", s["ok"], "load", s["load_s"], "dec", d.get("tg_tps"), (d.get("md5") or "")[:8], "pp2k", p.get("pp_tps"), "tg2k",
                p.get("tg_tps"), (p.get("md5") or "")[:8], "pp8k", q.get("pp_tps"), "tg8k", q.get("tg_tps"), (q.get("md5") or "")[:8],
                "ub", s["stats"]["n_ubatch_ctx"], "nb", s["stats"]["n_batch"], "lm", s["stats"]["lm_n_groups"], "uma", s["stats"]["uma"],
                "dev", s["dev_peak_mib"], "rss", s["rss_peak_mib"], "idle", s.get("mem_idle"), "end", s.get("mem_p8k") or s.get("mem_p2k"),
                s.get("err", ""), s["stats"]["err_lines"], s["stats"]["moe_cache"][:2])


# ---------------------------------------------------------------- KLD
def do_kld():
    """llama-perplexity -c 512 --chunks N on wiki.test.raw: ARMS[0] in mode full writes the reference logits; every (arm, mode) pair scores KLD vs it"""
    model = a.model
    if not os.path.exists(mpath(model)):
        row({"kind": "kld", "model": model, "err": "model missing"})
        return
    warm_cache(model)
    ref = ARMS[0]
    f = os.path.join(K, "kld_%s.bin" % model)
    pairs = [(ref, "full", True)]
    for mode in a.modes.split(","):
        for arm in ARMS:
            if (arm, mode) != (ref, "full"):
                pairs.append((arm, mode, False))
    for arm, mode, is_ref in pairs:
        target, budget = cell_target(model, mode, arm)
        lp = os.path.join(LOGS, "kld_%s_%s_%s.log" % (model, mode, arm))
        argv = [binp(arm, "llama-perplexity"), "-m", mpath(model), "-f", os.path.join(os.path.dirname(os.path.abspath(__file__)), "wiki.test.raw"),
                "-c", "512", "--chunks", str(a.chunks), "-fa", "on"] + mode_flags(model, mode, target, arm) + ["--kl-divergence-base", f] + \
               ([] if is_ref else ["--kl-divergence"])
        t = time.time()
        with open(lp, "w", encoding="utf-8") as fh:
            p = popen(argv, fh, arm)
            smp = Sampler(p.pid)
            try:
                rc = p.wait(a.timeout)
            except subprocess.TimeoutExpired:
                kill(p)
                rc = "timeout"
            mem = smp.done()
        s = rd(lp)
        g = lambda pat: (re.findall(pat, s) or [None])[-1]
        d = {"kind": "kld", "model": model, "mode": mode, "arm": arm, "arm_env": ARM_ENV.get(arm), "ref": ref + ":full", "rc": rc, "wall_s": round(time.time() - t),
             "fit_target": target, "budget": budget, "ppl": g(r"Final estimate: PPL = ([\d.]+ \+/- [\d.]+)"),
             "mean_kld": g(r"Mean +KLD: +(-?[\d.e+-]+)"), "max_kld": g(r"Maximum KLD: +(-?[\d.e+-]+)"),
             "p99_kld": g(r"99\.0% +KLD: +(-?[\d.e+-]+)"), "same_top": g(r"Same top p: +([\d.]+ ± [\d.]+)"),
             "stats": memstats(s), "argv": argv[1:]}
        d.update(mem)
        log("KLD", model, mode, arm, d["rc"], d["ppl"], d["mean_kld"], d["max_kld"], d["same_top"], d["stats"]["err_lines"])
        row(d)
        filt(lp, s)
        os.remove(lp)
        if is_ref and rc != 0:
            log("reference failed; stop KLD for", model)
            break
    try:
        os.remove(f)
    except OSError:
        pass


# ---------------------------------------------------------------- test-backend-ops
def do_tbo():
    arm = a.bin
    tb = binp(arm, "test-backend-ops")
    if not os.path.exists(tb):
        row({"kind": "tbo", "bin": arm, "err": "missing " + tb})
        return
    # k10s: ops "OP" or "OP@REGEX" (-p params filter), --tbmode test | perf (perf: the us/run lines are kept in the row)
    for spec in [x for x in a.ops.split(";" if ";" in a.ops else ",") if x]:
        op, _, pf = spec.partition("@")
        tagn = re.sub(r"[^A-Za-z0-9_]+", "_", spec)[:60]
        lp = os.path.join(LOGS, "tbo%s_%s_%s.log" % ("p" if a.tbmode == "perf" else "", arm, tagn))
        t = time.time()
        with open(lp, "w", encoding="utf-8") as fh:
            # k15: op ALL = the full suite (no -o)
            p = popen([tb, a.tbmode, "-b", DEV] + ([] if op == "ALL" else ["-o", op]) + (["-p", pf] if pf else []), fh, arm)
            try:
                rc = p.wait(a.timeout)
            except subprocess.TimeoutExpired:
                kill(p)
                rc = "timeout"
        s = rd(lp)
        # per backend: "Backend i/n: NAME" ... "x/y tests passed" (the device under test, not the skipped ones)
        per = {}
        for blk in re.split(r"\nBackend \d+/\d+: ", "\n" + s)[1:]:
            name = blk.split("\n", 1)[0].strip()
            mm = re.findall(r"(\d+)/(\d+) tests passed", blk)
            if mm:
                per[name] = mm[-1]
        m = [per[DEV]] if DEV in per else re.findall(r"(\d+)/(\d+) tests passed", s)
        fl = [re.sub(r"\x1b\[[0-9;]*m", "", l.strip())[:240] for l in s.splitlines() if "FAIL" in l and not l.lstrip().startswith("Backend")]
        # k15: failing test IDs (the case text before the first ": "), compared head vs base by ID, never by count
        fids = sorted({(re.match(r"\s*(\S.*?\)):", re.sub(r"\x1b\[[0-9;]*m", "", l)) or [None, l.strip()[:200]])[1] for l in fl})
        perf = [re.sub(r"\x1b\[[0-9;]*m", "", l.strip())[:300] for l in s.splitlines() if "us/run" in l]
        r = {"kind": "tbo" if a.tbmode == "test" else "tbop", "bin": arm, "op": spec, "rc": rc, "perf": perf, "arm_env": ARM_ENV.get(arm), "passed": m[-1] if m else None, "per_backend": per, "n_fail": len(fl), "fails": fl[:40], "fail_ids": fids[:5000],
             "pad": ARM_ENV.get(arm, {}).get("K6P_PAD"), "aborted": bool(re.search(r"GGML_ASSERT|Segmentation|Aborted|core dumped|HIP error|CUDA error", s)),
             "n_ok": len(re.findall(r"\bOK\b", s)), "not_supported": len(re.findall(r"not supported", s)), "wall_s": round(time.time() - t),
             "backend_line": re.findall(r"Backend \d+/\d+: [^\n]*", s)[:4], "status": re.findall(r"\d+/\d+ backends passed[^\n]*", s)[-1:],
             "tail": s[-400:]}
        log("TBO", arm, op, rc, r["passed"], len(fl), fl[:2])
        row(r)
        with open(lp.replace(".log", ".filtered.log"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(fl[:400] + ["--- tail ---", s[-3000:]]))
        os.remove(lp)


def do_exe():
    """k10s: run one test binary of an arm's bundle (e.g. the GDN tiled vs AR memcmp test), rc + tail in the row"""
    arm = a.bin
    exe = binp(arm, a.exe)
    if not os.path.exists(exe):
        row({"kind": "exe", "bin": arm, "exe": a.exe, "err": "missing " + exe})
        return
    lp = os.path.join(LOGS, "exe_%s_%s.log" % (arm, a.exe))
    t = time.time()
    with open(lp, "w", encoding="utf-8") as fh:
        p = popen([exe], fh, arm)
        try:
            rc = p.wait(a.timeout)
        except subprocess.TimeoutExpired:
            kill(p)
            rc = "timeout"
    s = rd(lp)
    r = {"kind": "exe", "bin": arm, "exe": a.exe, "rc": rc, "wall_s": round(time.time() - t), "arm_env": ARM_ENV.get(arm),
         "lines": [l[:300] for l in s.splitlines() if re.search(r"PASS|FAIL|OK|memcmp|identical|differ|error|bitwise|evaluations match|skipped", l, re.I)][:200],
         "match": (re.findall(r"(\d+) / (\d+) evaluations match the CPU", s) or [None])[-1], "skipped": "skipped" in s,
         "ok_line": (re.findall(r"test-mmid-multi: (OK|FAIL)", s) or [None])[-1], "n_bitwise": len(re.findall(r"bitwise", s)),
         "src_type_recaptures": len(re.findall(r"src type", s)), "pad": ARM_ENV.get(arm, {}).get("K6P_PAD"), "tail": s[-1500:]}
    log("EXE", arm, a.exe, rc, r["lines"][-3:])
    row(r)
    with open(lp.replace(".log", ".filtered.log"), "w", encoding="utf-8") as fh:
        fh.write(s[-20000:])
    os.remove(lp)


def do_dl():
    for m in a.models.split(","):
        download(m)
    with open(os.path.join(K, "DL_%s_DONE" % "_".join(a.models.split(","))), "w", encoding="utf-8") as fh:
        fh.write("done\n")


log("HOST", json.dumps(HOST), "BINS", BINS, "ARM_ENV", ARM_ENV, "TAGS", TAG, "DEV", DEV)
{"time": do_time, "kld": do_kld, "tbo": do_tbo, "dl": do_dl, "devinfo": do_devinfo, "exe": do_exe}[a.what]()
log("=== K6P %s %s DONE ===" % (a.what, a.cell or a.model or a.ops[:40]))

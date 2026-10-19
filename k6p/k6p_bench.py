#!/usr/bin/env python3
"""k6p_bench.py: release b11696-mix prebuilt vs llama.cpp moe-combined-k6, one platform (Linux / Windows / macOS), stdlib only.

Derived from r273_bench.py. Every measurement is a fresh process; arm order rotates per rep.
env: K6P_DIR work root (bins/<arm dir>, models/, out/), ARMS "name:dir,..." (dir under K6P_DIR/bins; default
     rel:rel,rel2:rel,k6:k6,k6b:k6), K6P_DEV device name for --list-devices / test-backend-ops (ROCm0, MTL0; "" = CPU-only).
cells: model:mode, mode full = --fit on (default margin), spillN = --fit on --fit-target (free - N MiB), i.e. an N MiB device
       budget; the target is computed once per cell from the first arm's --list-devices and used for every arm. cpu = -ngl 0.
       MoE models add --moe-cache-mib auto (both builds accept it).
per launch (llama-server -c 8192 -fa on -np 1): warm 8 tokens, dec = 40-token prompt + 128 greedy (decode tok/s),
       p2k = 2000 token ids (identical for every arm) + 128 greedy (prompt tok/s, decode after 2k); md5 of the text + token ids.
usage: k6p_bench.py dl --models q35,lfm2 | devinfo | time --cell M:mode --reps 3 | kld --model M --modes full,spill1536
       | tbo --bin k6 --ops A,B
rows -> K6P_DIR/out/rows.jsonl (also printed as ROW lines), filtered logs -> K6P_DIR/out/logs/
"""
import argparse, glob, hashlib, json, os, platform, re, shutil, signal, subprocess, sys, threading, time, urllib.request

K = os.environ.get("K6P_DIR") or os.getcwd()
OUT = os.path.join(K, "out")
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
ap.add_argument("--chunks", type=int, default=int(os.environ.get("KLD_CHUNKS", "20")))
ap.add_argument("--timeout", type=int, default=2400)
a = ap.parse_args()

MODELS = {
    "q35": ("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf", "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0", 532517120, False),
    "lfm2": ("unsloth/LFM2-8B-A1B-GGUF", "LFM2-8B-A1B-Q2_K.gguf", "01c1c9ba807324289806d69f5895e11aff6de784", 3079650368, True),
    "gptoss": ("unsloth/gpt-oss-20b-GGUF", "gpt-oss-20b-Q2_K.gguf", "d449b42d93e1c2c7bda5312f5c25c8fb91dfa9b4", 11468317888, True),
}
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


ARMS_SPEC = os.environ.get("ARMS", "rel:rel,rel2:rel,k6:k6,k6b:k6")
BINS = {}
for x in ARMS_SPEC.split(","):
    n, d = x.split(":")
    BINS[n] = find_bindir(os.path.join(K, "bins", d)) or os.path.join(K, "bins", d)
ARMS = [x.split(":")[0] for x in ARMS_SPEC.split(",")]
if a.arms:
    ARMS = a.arms.split(",")
TAG = {}
for n, d in BINS.items():
    for c in (os.path.join(d, "TAG"), os.path.join(d, "..", "TAG")):
        if os.path.exists(c):
            TAG[n] = rd(c).strip()
            break


def benv(arm):
    e = dict(os.environ)
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


class Sampler:
    """peak device memory used (Linux amdgpu sysfs, minus the baseline before the process) and peak host RSS (Linux /proc)"""

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
            try:
                s = rd("/proc/%d/status" % self.pid)
                g = lambda k: int((re.search(k + r":\s+(\d+) kB", s) or [0, 0])[1]) // 1024
                self.rss, self.anon, self.file = max(self.rss, g("VmRSS")), max(self.anon, g("RssAnon")), max(self.file, g("RssFile"))
            except Exception:
                pass
            time.sleep(0.2)

    def done(self):
        self.stop = True
        time.sleep(0.3)
        return {"dev_peak_mib": self.gpu_peak - self.base if self.gpu_peak >= 0 else None, "dev_base_mib": self.base,
                "rss_peak_mib": self.rss or None, "rss_anon_peak_mib": self.anon or None, "rss_file_peak_mib": self.file or None}


PAT = re.compile(r"buffer size|memory breakdown|\| +- |MoE cache|moe_cache|out of memory|error|failed|GGML_ASSERT|offload|load_tensors|"
                 r"n_ubatch|n_batch|mmap|load.mode|ROCm0|MTL0|ggml_cuda_init|Device 0|system_info|build|fit|pinned|host|prefetch|lend|"
                 r"weight.stream|integrated|unified|warning|abort|exception", re.I)


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
            "abort": bool(re.search(r"GGML_ASSERT|Segmentation|Aborted|core dumped|CUDA error|HIP error|ROCm error|exception", s)),
            "err_lines": [l.strip()[:240] for l in g(r"[^\n]*(?:GGML_ASSERT|CUDA error|HIP error|error:|ERROR|terminate called)[^\n]*")][:4]}


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
    repo, fname, rev, size, _ = MODELS[name]
    os.makedirs(MD, exist_ok=True)
    p = os.path.join(MD, fname)
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
    row({"kind": "dl", "model": name, "ok": ok, "s": round(time.time() - t)})
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


def prompt_tokens(port, model):
    f = os.path.join(K, "prompt_%s.json" % model)
    if not os.path.exists(f):
        src = rd(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wiki.test.raw"))[:200000]
        toks = post(port, "/tokenize", {"content": src, "add_special": True})["tokens"][:2000]
        with open(f, "w", encoding="utf-8") as fh:
            json.dump(toks, fh)
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)


def mode_flags(model, mode, target):
    moe = MODELS[model][4]
    f = []
    if mode == "cpu":
        f += ["-ngl", "0"]
    else:
        f += ["--fit", "on"]
        if mode.startswith("spill"):
            f += ["--fit-target", str(target)]
    if moe and mode != "cpu":
        f += ["--moe-cache-mib", "auto"]
    return f + os.environ.get("K6P_EXTRA", "").split()


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
    return subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, env=benv(arm), cwd=BINS[arm], **kw)


def run_server(arm, model, mode, target, tag):
    PORT[0] += 1
    port = PORT[0]
    lp = os.path.join(LOGS, tag + ".log")
    argv = [binp(arm, "llama-server"), "-m", mpath(model), "-c", "8192", "-fa", "on", "-np", "1", "--host", "127.0.0.1",
            "--port", str(port), "-lv", "4"] + mode_flags(model, mode, target)
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
            res["warm"] = complete(port, "The capital of France is", 8)
            res["dec"] = complete(port, DEC_PROMPT, 128)
            res["p2k"] = complete(port, prompt_tokens(port, model), 128)
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
    with open(mpath(model), "rb") as fh:
        while fh.read(64 << 20):
            pass
    log("page cache warm %s %.0f s" % (model, time.time() - t), meminfo())


def cell_target(model, mode):
    if not mode.startswith("spill"):
        return None, None
    budget = int(mode[5:])
    free = dev_free(ARMS[0])
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
    target, budget = cell_target(model, mode)
    log("cell", a.cell, "fit-target", target, "budget", budget)
    for rep in range(a.reps):
        order = ARMS[rep % len(ARMS):] + ARMS[:rep % len(ARMS)]
        for arm in order:
            tag = "%s_%s_%s_r%d" % (model, mode, arm, rep)
            s = run_server(arm, model, mode, target, tag)
            row(dict(s, kind="server", model=model, mode=mode, arm=arm, rep=rep, fit_target=target, budget=budget))
            d, p = s.get("dec") or {}, s.get("p2k") or {}
            log(tag, "ok", s["ok"], "load", s["load_s"], "dec", d.get("tg_tps"), (d.get("md5") or "")[:8], "pp2k", p.get("pp_tps"), "tg2k",
                p.get("tg_tps"), (p.get("md5") or "")[:8], "dev", s["dev_peak_mib"], "rss", s["rss_peak_mib"], s.get("err", ""),
                s["stats"]["err_lines"], s["stats"]["moe_cache"][:2])


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
    targets = {}
    for arm, mode, is_ref in pairs:
        if mode not in targets:
            targets[mode] = cell_target(model, mode)
        target, budget = targets[mode]
        lp = os.path.join(LOGS, "kld_%s_%s_%s.log" % (model, mode, arm))
        argv = [binp(arm, "llama-perplexity"), "-m", mpath(model), "-f", os.path.join(os.path.dirname(os.path.abspath(__file__)), "wiki.test.raw"),
                "-c", "512", "--chunks", str(a.chunks), "-fa", "on"] + mode_flags(model, mode, target) + ["--kl-divergence-base", f] + \
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
        d = {"kind": "kld", "model": model, "mode": mode, "arm": arm, "ref": ref + ":full", "rc": rc, "wall_s": round(time.time() - t),
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
    for op in a.ops.split(","):
        lp = os.path.join(LOGS, "tbo_%s_%s.log" % (arm, op))
        t = time.time()
        with open(lp, "w", encoding="utf-8") as fh:
            p = popen([tb, "test", "-b", DEV, "-o", op], fh, arm)
            try:
                rc = p.wait(a.timeout)
            except subprocess.TimeoutExpired:
                kill(p)
                rc = "timeout"
        s = rd(lp)
        m = re.findall(r"(\d+)/(\d+) tests passed", s)
        fl = [re.sub(r"\x1b\[[0-9;]*m", "", l.strip())[:240] for l in s.splitlines() if "FAIL" in l]
        r = {"kind": "tbo", "bin": arm, "op": op, "rc": rc, "passed": m[-1] if m else None, "n_fail": len(fl), "fails": fl[:40],
             "n_ok": len(re.findall(r"\bOK\b", s)), "not_supported": len(re.findall(r"not supported", s)), "wall_s": round(time.time() - t),
             "backend_line": re.findall(r"Backend \d+/\d+: [^\n]*", s)[:4], "status": re.findall(r"\d+/\d+ backends passed[^\n]*", s)[-1:],
             "tail": s[-400:]}
        log("TBO", arm, op, rc, r["passed"], len(fl), fl[:2])
        row(r)
        with open(lp.replace(".log", ".filtered.log"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(fl[:400] + ["--- tail ---", s[-3000:]]))
        os.remove(lp)


def do_dl():
    for m in a.models.split(","):
        download(m)


log("HOST", json.dumps(HOST), "BINS", BINS, "TAGS", TAG, "DEV", DEV)
{"time": do_time, "kld": do_kld, "tbo": do_tbo, "dl": do_dl, "devinfo": do_devinfo}[a.what]()
log("=== K6P %s %s DONE ===" % (a.what, a.cell or a.model or a.ops[:40]))

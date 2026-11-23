#!/usr/bin/env python3
"""Fair, rerunnable llama-bench decode (tg) benchmark across llama.cpp builds ("arms").

One fresh llama-bench process per measurement, arms interleaved per round (ABCD, DCBA, ABCD, ...),
one discarded warm-up per cell, the same -ngl / -ncmoe for every arm on a given (GPU, file), host
load sampled during every run, GPU / driver / runtime recorded. Stdlib only; Linux and Windows.

  python3 tg_bench.py --out OUT \
      --arm PREV=gh:unslothai/llama.cpp@b12588-mix-c0c3de8/app-b12588-mix-c0c3de8-linux-x64-cuda12-newer.tar.gz \
      --arm CUR=gh:unslothai/llama.cpp@b12631-mix-acc0843/app-b12631-mix-acc0843-linux-x64-cuda12-newer.tar.gz \
      --arm HEAD=/path/to/build/bin \
      --arm MAIN=gh:ggml-org/llama.cpp@b12649/llama-b12649-bin-ubuntu-cuda-12.8-x64.tar.gz \
      --model q4=/m/Inkling-2-Mini-UD-Q4_K_XL.gguf --model mq4=/m/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf \
      --arm-models MAIN=mq4 --kv f16,q8_0 --rounds 3 --baseline CUR \
      --ratios CUR/PREV,HEAD/CUR,HEAD/MAIN,CUR/MAIN

Arm SPEC: a llama-bench binary, a directory holding one, or gh:OWNER/REPO@TAG/ASSET (a public release asset,
.tar.gz or .zip, downloaded once into --cache). Re-running with the same --out resumes (finished runs are kept).
See README.md next to this file.
"""
import argparse, datetime, json, os, platform, re, shutil, statistics, subprocess, sys, tarfile, threading, time, zipfile
import urllib.request
from pathlib import Path

EXE = ".exe" if os.name == "nt" else ""
T0 = time.time()


def log(*a):
    print(f"[tg_bench {time.time() - T0:7.0f}s]", *a, flush=True)


# ------------------------------------------------------------------ host / GPU facts
def loadavg():
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return None  # Windows: no 1-min load average


class LoadSampler:
    def __init__(self, every=2.0):
        self.every, self.samples, self._stop = every, [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            v = loadavg()
            if v is not None:
                self.samples.append(v)
            self._stop.wait(self.every)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(5)


def sh(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace")
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # noqa: BLE001
        return -1, repr(e)


def gpu_facts():
    f = {"os": platform.platform(), "host": platform.node(), "cpus": os.cpu_count(), "python": platform.python_version()}
    smi = shutil.which("nvidia-smi")
    if smi:
        q = [smi, "--query-gpu=index,name,driver_version,memory.total,pci.bus_id,uuid", "--format=csv,noheader"]
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cvd and "," not in cvd:
            q.insert(1, f"--id={cvd}")
        rc, out = sh(q)
        if rc == 0 and out.strip():
            f["nvidia_gpu"] = out.strip().splitlines()
        rc, out = sh([smi])
        m = re.search(r"CUDA Version:\s*([0-9.]+)", out)
        if m:
            f["cuda_driver_api"] = m.group(1)
    for tool, args in (("amd-smi", ["static", "--asic", "--driver"]), ("amd-smi", ["version"]), ("rocm-smi", ["--showproductname", "--showdriverversion"]),
                       ("hipconfig", ["--version"])):
        p = shutil.which(tool)
        if p:
            rc, out = sh([p, *args])
            if rc == 0 and out.strip():
                f.setdefault("amd", {})[f"{tool} {' '.join(args)}"] = out.strip()[-3000:]
    for p in ("/opt/rocm/.info/version",):
        try:
            f["rocm_version_file"] = Path(p).read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return f


# ------------------------------------------------------------------ arms
def fetch_release(spec, cache):
    m = re.fullmatch(r"gh:([^/@]+/[^/@]+)@([^/]+)/(.+)", spec)
    if not m:
        raise SystemExit(f"bad gh: spec {spec!r} (want gh:OWNER/REPO@TAG/ASSET)")
    repo, tag, asset = m.groups()
    d = Path(cache) / re.sub(r"[^A-Za-z0-9._-]", "_", f"{repo}@{tag}@{asset}")
    done = d / ".extracted"
    if not done.exists():
        d.mkdir(parents=True, exist_ok=True)
        arc = d / asset
        url = f"https://github.com/{repo}/releases/download/{tag}/{asset}"
        for attempt in range(5):
            try:
                log(f"download {url}")
                with urllib.request.urlopen(url, timeout=600) as r, open(arc, "wb") as w:
                    shutil.copyfileobj(r, w, 1 << 22)
                break
            except Exception as e:  # noqa: BLE001
                log(f"download failed ({e!r}), retry {attempt}")
                time.sleep(10 * (attempt + 1))
        else:
            raise SystemExit(f"could not download {url}")
        x = d / "x"
        if asset.endswith(".zip"):
            with zipfile.ZipFile(arc) as z:
                z.extractall(x)
        else:
            with tarfile.open(arc) as t:
                try:
                    t.extractall(x, filter="data")
                except TypeError:  # Python < 3.10.12 has no filter
                    t.extractall(x)  # noqa: S202 (public release asset, own empty dir)
        arc.unlink()
        done.write_text(url, encoding="utf-8")
    return d / "x", {"repo": repo, "tag": tag, "asset": asset}


def find_bin(root, name="llama-bench"):
    root = Path(root)
    if root.is_file():
        return root
    hits = sorted(root.rglob(name + EXE), key=lambda p: len(p.parts))
    if not hits:
        raise SystemExit(f"no {name}{EXE} under {root}")
    return hits[0]


def resolve_arms(specs, cache):
    arms = {}
    for s in specs:
        name, _, spec = s.partition("=")
        info = {"spec": spec}
        if spec.startswith("gh:"):
            root, rel = fetch_release(spec, cache)
            info.update(rel)
        else:
            root = Path(spec)
        b = find_bin(root)
        if os.name != "nt":
            for p in b.parent.iterdir():  # zip extraction drops the exec bit
                if p.is_file() and p.name.startswith("llama-") and not p.suffix:
                    p.chmod(p.stat().st_mode | 0o111)
        info["bench"] = str(b)
        for extra in ("BUILD_INFO.txt",):
            q = b.parent / extra
            if q.exists():
                info["build_info"] = q.read_text(encoding="utf-8", errors="replace")[:4000]
        arms[name] = info
    return arms


def arm_env(arms, name, lib_dirs, extra_env, arm_libs=None):
    env = dict(os.environ)
    bindir = str(Path(arms[name]["bench"]).parent)
    key = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    env[key] = os.pathsep.join([bindir, *(arm_libs or {}).get(name, []), *lib_dirs, env.get(key, "")])
    env.setdefault("CUDA_CACHE_MAXSIZE", str(4 << 30))  # PTX-JIT arms keep their cache across runs
    for kv in extra_env.get(name, []):
        k, _, v = kv.partition("=")
        env[k] = v
    return env


# ------------------------------------------------------------------ one measurement
def bench_cmd(bin_, model, n, kv, ngl, ncmoe, extra):
    c = [bin_, "-m", model, "-p", "0", "-n", str(n), "-fa", "1", "-ngl", str(ngl), "-r", "1", "-o", "json"]
    if kv != "f16":
        c += ["-ctk", kv, "-ctv", kv]
    if ncmoe:
        c += ["-ncmoe", str(ncmoe)]
    return c + list(extra)


def gpu_busy():
    """GPU busy % right before a run (amdgpu sysfs on Linux, nvidia-smi elsewhere); None when unreadable."""
    vals = []
    for p in Path("/sys/class/drm").glob("card*/device/gpu_busy_percent") if os.name != "nt" else []:
        try:
            vals.append(int(p.read_text().strip()))
        except (OSError, ValueError):
            pass
    if not vals and shutil.which("nvidia-smi"):
        q = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cvd and "," not in cvd:
            q.insert(1, f"--id={cvd}")
        rc, out = sh(q, timeout=20)
        vals = [int(x) for x in re.findall(r"\d+", out)] if rc == 0 else []
    return max(vals) if vals else None


def run_bench(cmd, env, timeout):
    busy0 = gpu_busy()
    t = time.time()
    with LoadSampler() as ls:
        l0 = loadavg()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout, errors="replace")
            rc, out, err = r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired as e:
            rc, out, err = "timeout", e.stdout or "", e.stderr or ""
    rec = {"rc": rc, "wall_s": round(time.time() - t, 2), "gpu_busy_start": busy0, "load_start": l0, "load_max": max(ls.samples, default=None),
           "load_end": loadavg(), "cmd": [str(x) for x in cmd]}
    rows = None
    if rc == 0:
        try:
            s = out[out.index("["):]
            rows = json.loads(s)
        except (ValueError, json.JSONDecodeError):
            rec["rc"] = "badjson"
    if rows:
        r0 = rows[-1]
        rec["ts"] = r0.get("avg_ts")
        rec["llama"] = {k: r0.get(k) for k in ("build_commit", "build_number", "cpu_info", "gpu_info", "backends", "model_type",
                                               "model_size", "model_n_params", "n_gpu_layers", "n_cpu_moe", "n_threads",
                                               "type_k", "type_v", "flash_attn", "n_ubatch", "n_batch")}
    else:
        rec["err_tail"] = ((err or "") + (out or ""))[-2500:]
    return rec


# ------------------------------------------------------------------ -ncmoe selection
def fits(arms, cells, model, ncmoe, a, envs):
    # same -n as the measurement: the context (and with it the compute buffers) grows with -n, so a short probe
    # can fit where the measured run does not (seen on a 16 GB T4: -n 32 fits at -ncmoe 4, -n 1024 does not)
    for arm, kv in cells:
        r = run_bench(bench_cmd(arms[arm]["bench"], model, a.probe_n or a.n, kv, a.ngl, ncmoe, a.extra), envs[arm], a.timeout)
        if r["rc"] != 0 or not r.get("ts"):
            return False, f"{arm}/{kv}: rc {r['rc']} {r.get('err_tail', '')[-300:]}"
    return True, ""


def pick_ncmoe(arms, mk, model, arm_list, a, envs):
    """Smallest -ncmoe at which every arm and KV type loads and decodes; the search uses the first arm / first KV,
    the result is then checked on every (arm, kv) and raised until all pass."""
    first = [(arm_list[0], a.kv[0])]
    every = [(arm, kv) for arm in arm_list for kv in a.kv]
    ok, why = fits(arms, first, model, 0, a, envs)
    trail = [(0, ok, why)]
    if ok:
        n = 0
    else:
        lo, hi = 1, a.ncmoe_max
        while lo < hi:
            mid = (lo + hi) // 2
            ok, why = fits(arms, first, model, mid, a, envs)
            trail.append((mid, ok, why))
            log(f"ncmoe probe {mk} {mid}: {'fits' if ok else 'no'}")
            lo, hi = (lo, mid) if ok else (mid + 1, hi)
        n = lo
    while n <= a.ncmoe_max:
        ok, why = fits(arms, every, model, n, a, envs)
        trail.append((n, ok, why))
        log(f"ncmoe check {mk} {n} all arms x kv: {'fits' if ok else why}")
        if ok:
            return n, trail
        n += 1
    raise SystemExit(f"{mk}: nothing fits up to -ncmoe {a.ncmoe_max}")


# ------------------------------------------------------------------ summary
def med_rng(xs):
    if not xs:
        return None
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs),
            "spread_pct": 100.0 * (max(xs) - min(xs)) / statistics.median(xs)}


def summarize(runs, arm_order, models, kvs, ratios, baseline):
    cells = {}
    for r in runs:
        if r.get("kind") != "measure" or not r.get("ts") or r.get("voided"):
            continue
        cells.setdefault((r["model"], r["kv"], r["arm"]), []).append(r["ts"])
    rows = []
    for mk in models:
        for kv in kvs:
            row = {"model": mk, "kv": kv, "arms": {}, "ratios": {}}
            for arm in arm_order:
                s = med_rng(cells.get((mk, kv, arm), []))
                if s:
                    row["arms"][arm] = s
            pairs = list(ratios) or [f"{x}/{baseline}" for x in arm_order if baseline and x != baseline]
            for p in pairs:
                n_, _, d_ = p.partition("/")
                if n_ in row["arms"] and d_ in row["arms"]:
                    row["ratios"][p] = row["arms"][n_]["median"] / row["arms"][d_]["median"]
            if row["arms"]:
                rows.append(row)
    return rows


def render(rows, arm_order, ratio_names, meta):
    L = []
    hdr = ["file", "KV"] + [f"{a} tok/s" for a in arm_order] + list(ratio_names) + ["max A/A spread"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for r in rows:
        c = [r["model"], r["kv"]]
        for a in arm_order:
            s = r["arms"].get(a)
            c.append(f"{s['median']:.2f} [{s['min']:.2f}-{s['max']:.2f}]" if s else "-")
        for p in ratio_names:
            v = r["ratios"].get(p)
            c.append(f"{v:.3f}" if v else "-")
        sp = [s["spread_pct"] for s in r["arms"].values()]
        c.append(f"{max(sp):.1f}%" if sp else "-")
        L.append("| " + " | ".join(c) + " |")
    L.append("")
    for k, v in meta.items():
        L.append(f"- {k}: {v}")
    return "\n".join(L)


# ------------------------------------------------------------------ optional PPL sanity
def ppl_check(arms, arm_names, model, ncmoe, text, a, envs, out):
    res = {}
    for arm in arm_names:
        pbin = find_bin(Path(arms[arm]["bench"]).parent, "llama-perplexity")
        cmd = [str(pbin), "-m", model, "-f", text, "-c", "512", "--chunks", "4", "-ngl", str(a.ngl), "-fa", "on"]
        if ncmoe:
            cmd += ["-ncmoe", str(ncmoe)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, env=envs[arm], timeout=a.timeout, errors="replace")
            txt = (r.stdout or "") + (r.stderr or "")
            rc = r.returncode
        except subprocess.TimeoutExpired:
            txt, rc = "", "timeout"
        (out / f"ppl_{arm}.log").write_text(txt, encoding="utf-8")
        m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", txt)
        chunks = re.findall(r"\[\d+\]([0-9.]+),?", txt)
        res[arm] = {"rc": rc, "ppl": m.group(1) if m else None, "pm": m.group(2) if m else None, "chunks": chunks}
        log(f"ppl {arm}: {res[arm]}")
    return res


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output dir (runs.jsonl, summary.json, summary.md); reused = resume")
    ap.add_argument("--arm", action="append", required=True, help="NAME=SPEC (binary, dir, or gh:OWNER/REPO@TAG/ASSET)")
    ap.add_argument("--model", action="append", required=True, help="NAME=PATH.gguf")
    ap.add_argument("--arm-models", action="append", default=[], help="ARM=m1,m2: run that arm only on these models")
    ap.add_argument("--arm-env", action="append", default=[], help="ARM=KEY=VALUE extra env for one arm")
    ap.add_argument("--lib-dir", action="append", default=[], help="prepended to LD_LIBRARY_PATH (PATH on Windows) for every arm")
    ap.add_argument("--arm-lib", action="append", default=[], help="ARM=DIR prepended to LD_LIBRARY_PATH (PATH on Windows) for one arm")
    ap.add_argument("--kv", default="f16,q8_0", help="comma list of KV cache types (f16 = default, else -ctk X -ctv X)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n", type=int, default=1024, help="tokens generated (llama-bench -n)")
    ap.add_argument("--warmup-n", type=int, default=128, help="tokens of the discarded warm-up per cell (0 = no warm-up)")
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--ncmoe", action="append", default=[], help="MODEL=N fixes -ncmoe; unset models are auto-fitted (smallest that fits)")
    ap.add_argument("--ncmoe-max", type=int, default=99)
    ap.add_argument("--probe-n", type=int, default=0, help="-n of the -ncmoe fit probes (0 = the measured --n)")
    ap.add_argument("--extra", default="", help="extra llama-bench args, same for every arm (e.g. '-t 8')")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--ratios", default="", help="comma list NUM/DEN of arm medians")
    ap.add_argument("--load-start-max", type=float, default=None, help="wait until 1-min load is below this before each run")
    ap.add_argument("--load-run-max", type=float, default=None, help="redo a run whose sampled load exceeded this")
    ap.add_argument("--load-redo", type=int, default=3, help="max redos per run for load")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--cache", default=None, help="release download cache (default OUT/../tg_bench_cache)")
    ap.add_argument("--ppl-arms", default="", help="comma list: llama-perplexity -c 512 --chunks 4 sanity on --ppl-model")
    ap.add_argument("--ppl-model", default=None)
    ap.add_argument("--ppl-text", default=None, help="wiki.test.raw")
    ap.add_argument("--note", action="append", default=[], help="free text recorded in the summary")
    a = ap.parse_args()
    a.kv = [x for x in a.kv.split(",") if x]
    a.extra = a.extra.split() if a.extra else []
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache) if a.cache else out.parent / "tg_bench_cache"
    models = dict(m.split("=", 1) for m in a.model)
    arms = resolve_arms(a.arm, cache)
    arm_order = list(arms)
    only = {k: v.split(",") for k, v in (x.split("=", 1) for x in a.arm_models)}
    extra_env = {}
    for x in a.arm_env:
        k, _, v = x.partition("=")
        extra_env.setdefault(k, []).append(v)
    arm_libs = {}
    for x in a.arm_lib:
        k, _, v = x.partition("=")
        arm_libs.setdefault(k, []).append(v)
    envs = {n: arm_env(arms, n, a.lib_dir, extra_env, arm_libs) for n in arm_order}
    runs_f = out / "runs.jsonl"
    runs = [json.loads(l) for l in runs_f.read_text(encoding="utf-8").splitlines() if l.strip()] if runs_f.exists() else []
    done = {(r["kind"], r["model"], r["kv"], r["arm"], r["round"]) for r in runs if r.get("ts") and not r.get("voided")}
    facts = gpu_facts()
    meta_f = out / "meta.json"
    meta = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else {}
    meta.update({"facts_now": facts, "arms": arms, "models": models, "kv": a.kv, "rounds": a.rounds, "n": a.n,
                 "warmup_n": a.warmup_n, "ngl": a.ngl, "extra": a.extra, "notes": a.note, "argv": sys.argv,
                 "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    meta.setdefault("ncmoe", {})
    meta.setdefault("started", datetime.datetime.now(datetime.timezone.utc).isoformat())
    log("gpu:", facts.get("nvidia_gpu") or facts.get("amd", "?"), "cuda", facts.get("cuda_driver_api"))
    for n, i in arms.items():
        log(f"arm {n}: {i['bench']}")
    arm_models = {arm: [m for m in models if arm not in only or m in only[arm]] for arm in arm_order}

    # an arm that cannot load a file at all (e.g. an older build and a newer GGUF layout) is dropped for that file:
    # tried fully on the GPU first, then with every MoE expert on the CPU (so VRAM is not the reason); a failure of
    # only one of the two is kept as a finding (meta "load_check") and does not drop the arm
    meta.setdefault("incompatible", {})
    meta.setdefault("load_check", {})
    for mk, mp in models.items():
        for arm in arm_order:
            key = f"{arm}:{mk}"
            if mk not in arm_models[arm]:
                continue
            if key not in meta["incompatible"]:
                chk = {}
                for nc in (0, a.ncmoe_max):
                    r = run_bench(bench_cmd(arms[arm]["bench"], mp, 8, a.kv[0], a.ngl, nc, a.extra), envs[arm], a.timeout)
                    chk[f"ncmoe{nc}"] = "ok" if (r["rc"] == 0 and r.get("ts")) else f"rc {r['rc']}: " + r.get("err_tail", "")[-4000:]
                meta["load_check"][key] = chk
                bad = [v for v in chk.values() if v != "ok"]
                meta["incompatible"][key] = bad[-1] if len(bad) == len(chk) else None
                if bad and len(bad) < len(chk):
                    log(f"{arm} on {mk}: fails only in one placement: {[k for k, v in chk.items() if v != 'ok']}")
                meta_f.write_text(json.dumps(meta, indent=1), encoding="utf-8")
            if meta["incompatible"][key]:
                log(f"{arm} cannot run {mk}: {meta['incompatible'][key][-200:]}")
                arm_models[arm] = [m for m in arm_models[arm] if m != mk]

    # -ncmoe per model, once, applied to every arm
    fixed = dict(x.split("=", 1) for x in a.ncmoe)
    for mk, mp in models.items():
        if mk in meta["ncmoe"]:
            continue
        al = [arm for arm in arm_order if mk in arm_models[arm]]
        if not al:
            meta["ncmoe"][mk] = {"value": None, "how": "no arm can load this file"}
            continue
        if mk in fixed:
            meta["ncmoe"][mk] = {"value": int(fixed[mk]), "how": "fixed by --ncmoe"}
        else:
            v, trail = pick_ncmoe(arms, mk, mp, al, a, envs)
            meta["ncmoe"][mk] = {"value": v, "how": "auto: smallest that loads + decodes for every arm x kv", "trail": trail}
        log(f"ncmoe {mk} = {meta['ncmoe'][mk]['value']} ({meta['ncmoe'][mk]['how']})")
        meta_f.write_text(json.dumps(meta, indent=1), encoding="utf-8")

    def one(kind, rnd, mk, kv, arm, n):
        key = (kind, mk, kv, arm, rnd)
        if key in done:
            return
        for attempt in range(a.load_redo + 1):
            if a.load_start_max is not None:
                w0 = time.time()
                while (loadavg() or 0) >= a.load_start_max:
                    if time.time() - w0 > 60 and int(time.time() - w0) % 300 < 10:
                        log(f"waiting for load < {a.load_start_max} (now {loadavg():.0f})")
                    time.sleep(10)
            cmd = bench_cmd(arms[arm]["bench"], models[mk], n, kv, a.ngl, meta["ncmoe"][mk]["value"], a.extra)
            rec = run_bench(cmd, envs[arm], a.timeout)
            rec.update({"kind": kind, "round": rnd, "model": mk, "kv": kv, "arm": arm, "attempt": attempt,
                        "t": datetime.datetime.now(datetime.timezone.utc).isoformat(), "ncmoe": meta["ncmoe"][mk]["value"]})
            lm = rec.get("load_max")
            if a.load_run_max is not None and lm is not None and lm > a.load_run_max:
                rec["voided"] = f"load_max {lm:.0f} > {a.load_run_max}"
            if rec["rc"] != 0:
                rec["voided"] = rec.get("voided") or f"rc {rec['rc']}"
            with open(runs_f, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            runs.append(rec)
            ts = rec.get("ts")
            log(f"{kind} r{rnd} {mk} {kv} {arm}: {ts and round(ts, 2)} tok/s  load {rec['load_start']}->{lm}  "
                f"{'VOID ' + rec['voided'] if rec.get('voided') else ''}")
            if not rec.get("voided"):
                done.add(key)
                return
            if rec["rc"] != 0 and attempt >= 1:
                return  # a failing binary fails twice: record and move on

    for rnd in range(0 if a.warmup_n else 1, a.rounds + 1):
        order = arm_order if rnd % 2 == 1 or rnd == 0 else arm_order[::-1]
        for mk in models:
            for kv in a.kv:
                for arm in order:
                    if mk not in arm_models[arm]:
                        continue
                    if rnd == 0:
                        one("warmup", 0, mk, kv, arm, a.warmup_n)
                    else:
                        one("measure", rnd, mk, kv, arm, a.n)

    if a.ppl_arms and a.ppl_model and a.ppl_text and "ppl" not in meta:
        pm = a.ppl_model
        meta["ppl"] = {"model": pm, "res": ppl_check(arms, a.ppl_arms.split(","), models[pm], meta["ncmoe"][pm]["value"],
                                                     a.ppl_text, a, envs, out)}
    meta["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta["facts_end"] = gpu_facts()
    meas = [r for r in runs if r.get("kind") == "measure" and r.get("ts") and not r.get("voided")]
    loads = [r["load_max"] for r in meas if r.get("load_max") is not None]
    meta["load_max_over_runs"] = {"median": statistics.median(loads), "max": max(loads)} if loads else None
    meta["build_commits"] = {arm: sorted({str((r.get("llama") or {}).get("build_commit")) + " #" + str((r.get("llama") or {}).get("build_number"))
                                          for r in meas if r["arm"] == arm}) for arm in arm_order}
    meta["gpu_info_llama"] = sorted({str((r.get("llama") or {}).get("gpu_info")) for r in meas})
    meta["n_voided"] = sum(1 for r in runs if r.get("voided"))
    ratios = [x for x in a.ratios.split(",") if x]
    rows = summarize(runs, arm_order, list(models), a.kv, ratios, a.baseline)
    ratio_names = ratios or [f"{x}/{a.baseline}" for x in arm_order if a.baseline and x != a.baseline]
    meta_f.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"rows": rows, "meta": meta}, indent=1), encoding="utf-8")
    md = render(rows, arm_order, ratio_names, {
        "GPU": facts.get("nvidia_gpu") or meta["gpu_info_llama"],
        "driver CUDA API": facts.get("cuda_driver_api"),
        "ncmoe": {k: v["value"] for k, v in meta["ncmoe"].items()},
        "builds": meta["build_commits"],
        "host load (1-min, max sampled per run)": meta["load_max_over_runs"],
        "voided runs (redone)": meta["n_voided"],
        "arm cannot load file (not run)": sorted(k for k, v in meta.get("incompatible", {}).items() if v) or None,
        "ppl": {k: v.get("ppl") for k, v in (meta.get("ppl") or {}).get("res", {}).items()} or None,
    })
    (out / "summary.md").write_text(md + "\n", encoding="utf-8")
    print("TG_BENCH_SUMMARY_BEGIN")
    print(md)
    print("TG_BENCH_SUMMARY_END", flush=True)


if __name__ == "__main__":
    main()

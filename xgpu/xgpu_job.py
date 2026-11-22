#!/usr/bin/env python3
"""Inkling-2-Mini llama.cpp pin-set A/B on one GPU: BASE vs HEAD1 vs HEAD2, built from source in this job.

Stdlib only (huggingface_hub used for downloads when importable). Works on Linux CUDA (Colab / Kaggle),
Linux ROCm and Windows HIP (AMD CI).

  python xgpu_job.py --kit DIR --work DIR --backend cuda|hip [--models IQ1_S,Q4_K_XL] [--quick]

Arms:
  rel   = ggml-org b12588 + rel.diff    (release b12588-mix-c0c3de8 pin set, tree b055ac5e)
  head  = ggml-org b12628 + head.diff   (unslothai/llama.cpp#399 pin set = master cfc9ea370, tree f1646263)
  headp = head + headp.diff             (probe: GGML_CUDA_BANDED_FORCE_DENSE=1 sends banded attention to the dense FA path)
All arms are built sequentially in one build dir (incremental), binaries copied per arm.
Results: WORK/out/results.json, printed between XGPU_RESULTS_BEGIN / XGPU_RESULTS_END.
"""
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

UPSTREAM = "https://github.com/ggml-org/llama.cpp"
TAGS = {"b12588": "8853ab54e142e2c3826e59847e64c0ec99b0983f", "b12628": "72fb0b3c8d79a53defda086ea07a2383eab719a9"}
# (name, diff, expected tree, parent arm or None, upstream tag or None)
ARMS = [
    ("rel", "rel.diff", "b055ac5e8eb60e933b2f24f5c97b5a0d0f466c2d", None, "b12588"),     # release b12588-mix-c0c3de8 = b12588 + master pr-set fcea32f26 (30 pins)
    ("head", "head.diff", "f164626390620c0f0045adeeb93cddd9d34d9be4", None, "b12628"),   # b12628 + #399 pr-set (34 pins) = master cfc9ea370
    ("headp", "headp.diff", "8c7c3f70237c70569b46267a363a50abd10e4d30", "head", None),   # head + GGML_CUDA_BANDED_FORCE_DENSE probe switch
]
SEL = ["rel", "head", "headp"]
REF = "rel"
KILLS = {"rel": {"headp_dense": ("headp", {"GGML_CUDA_BANDED_FORCE_DENSE": "1"})}}
TBO_KILLS = {}
REPO = "unsloth/Inkling-2-Mini-GGUF"
MODEL_FILES = {"IQ1_S": "Inkling-2-Mini-UD-IQ1_S.gguf", "Q4_K_XL": "Inkling-2-Mini-UD-Q4_K_XL.gguf",
               "Q8_K_XL": "Inkling-2-Mini-UD-Q8_K_XL.gguf"}
OPS = ["MUL_MAT", "MUL_MAT_ID", "CONCAT", "CLIP_ROWS_FUSE", "SITU_FUSE", "FLASH_ATTN_EXT", "FLASH_ATTN_EXT_BANDED"]
TARGETS = ["test-backend-ops", "llama-perplexity", "llama-bench", "llama-batched-bench", "test-llama-archs"]
EXE = ".exe" if os.name == "nt" else ""
SMOKE = False
BACKEND = "cuda"
UB_SWEEP = []
KLD_KILLS = False
KLD_CFGS = []
ARCHS_RE = ""
NO_BENCH = False
FIRST_MODEL = "IQ1_S"
NPL = "1,4"
T0 = time.time()
STATUS = {"step": "start"}


def log(*a):
    print(f"[xgpu {time.time() - T0:7.0f}s]", *a, flush=True)


def heartbeat():
    while True:
        time.sleep(240)
        log("HB", STATUS.get("step"))


def run(cmd, cwd=None, logf=None, env=None, timeout=None, check=False, stdout_file=None):
    """Run cmd, output to logf (or captured). Returns (rc, text, seconds). stdout_file: stdout alone there."""
    e = dict(os.environ)
    if env:
        e.update(env)
    t = time.time()
    try:
        if stdout_file:
            with open(logf, "w", encoding="utf-8", errors="replace") as f, open(stdout_file, "w", encoding="utf-8") as so:
                p = subprocess.run(cmd, cwd=cwd, stdout=so, stderr=f, env=e, timeout=timeout)
            rc = p.returncode
            text = Path(logf).read_text(encoding="utf-8", errors="replace")
        elif logf:
            with open(logf, "w", encoding="utf-8", errors="replace") as f:
                p = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=e, timeout=timeout)
            rc = p.returncode
            text = Path(logf).read_text(encoding="utf-8", errors="replace")
        else:
            p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace", env=e, timeout=timeout)
            rc, text = p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        rc, text = -9, "TIMEOUT"
        if logf:
            text = Path(logf).read_text(encoding="utf-8", errors="replace") + "\nTIMEOUT"
    if check and rc != 0:
        raise SystemExit(f"FATAL: {' '.join(map(str, cmd))} -> {rc}\n{text[-3000:]}")
    return rc, text, time.time() - t


def git(*a, cwd):
    cmd = ["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "-c", "user.name=xgpu",
           "-c", "user.email=xgpu@localhost", *a]
    return run(cmd, cwd=cwd, check=True)[1].strip()


# ---------------------------------------------------------------- trees + build
def prepare(kit, src):
    src.mkdir(parents=True, exist_ok=True)
    git("init", "-q", cwd=src)
    for tag, tree in TAGS.items():
        git("fetch", "-q", "--depth", "1", UPSTREAM, f"refs/tags/{tag}:refs/tags/{tag}", cwd=src)
        got = git("rev-parse", f"{tag}^{{tree}}", cwd=src)
        if got != tree:
            raise SystemExit(f"FATAL: {tag} tree {got} != {tree}")
    commits = {}
    for name, diff, want, parent, tag in ARMS:
        if name not in SEL and not any(a[3] == name for a in ARMS if a[0] in SEL):
            continue
        git("checkout", "-q", "-f", "--detach", tag if tag else commits[parent], cwd=src)
        git("clean", "-qfdx", cwd=src)
        git("apply", "--binary", "--whitespace=nowarn", str(kit / diff), cwd=src)
        git("add", "-A", cwd=src)
        got = git("write-tree", cwd=src)
        if got != want:
            raise SystemExit(f"FATAL: {name} tree {got} != expected {want}")
        git("commit", "-q", "-m", name, cwd=src)
        commits[name] = git("rev-parse", "HEAD", cwd=src)
        log(f"tree {name} = {got} ok")
    return commits


def cmake_args(backend, arch):
    a = ["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_BUILD_TESTS=ON", "-DLLAMA_CURL=OFF",
         "-DLLAMA_BUILD_SERVER=OFF", "-DGGML_NATIVE=ON"]
    if os.name != "nt":
        a.append("-DBUILD_SHARED_LIBS=OFF")  # per-arm binaries must not load another arm's libggml
    if backend == "cuda":
        a += ["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={arch}"]
    else:
        a += ["-DGGML_HIP=ON", f"-DGPU_TARGETS={arch}", f"-DAMDGPU_TARGETS={arch}"]
        if os.name == "nt" and os.environ.get("CLANG_EXE"):
            c = os.environ["CLANG_EXE"]
            a += [f"-DCMAKE_C_COMPILER={c}", f"-DCMAKE_CXX_COMPILER={re.sub(r'clang[.]exe$', 'clang++.exe', c)}"]
    return a


def build_all(src, work, backend, arch, commits, out):
    bdir = work / "build"
    bins, info = {}, {}
    jobs = str(os.cpu_count() or 4)
    for arm in ARMS:
        name = arm[0]
        if name not in SEL:
            continue
        STATUS["step"] = f"build {name}"
        git("checkout", "-q", "-f", "--detach", commits[name], cwd=src)
        if not (bdir / "build.ninja").exists():
            rc, text, dt = run(["cmake", "-S", str(src), "-B", str(bdir), *cmake_args(backend, arch)],
                               logf=out / f"cmake_{name}.log")
            if rc != 0:
                info[name] = {"rc": f"cmake{rc}", "s": round(dt)}
                log(f"cmake {name} FAILED rc {rc}\n{text[-2000:]}")
                continue
        rc, text, dt = run(["cmake", "--build", str(bdir), "-j", jobs, "--target", *TARGETS],
                           logf=out / f"build_{name}.log", timeout=3 * 3600)
        info[name] = {"rc": rc, "s": round(dt)}
        log(f"build {name} rc={rc} in {dt:.0f} s")
        if rc != 0:
            errs = [ln for ln in text.splitlines() if "error" in ln.lower()][:40]
            log("\n".join(errs))
            continue
        dst = work / "bins" / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(bdir / "bin", dst)
        bins[name] = dst
    return bins, info


# ---------------------------------------------------------------- downloads
def download(url, dst, token=None):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 24)
    tmp.rename(dst)


def get_model(key, mdir):
    fn = MODEL_FILES[key]
    dst = mdir / fn
    if dst.exists():
        return dst
    mdir.mkdir(parents=True, exist_ok=True)
    t = time.time()
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None
    ok = False
    for attempt in range(12):  # hf_hub_download resumes its .incomplete file on retry
        if hf_hub_download is None:
            break
        try:
            dst = Path(hf_hub_download(REPO, fn, local_dir=str(mdir)))
            ok = True
            break
        except Exception as e:  # noqa: BLE001
            log(f"hf_hub_download {key} attempt {attempt} failed ({e!r}); retrying (resumes)")
            time.sleep(10)
    if not ok:
        log("urllib fallback")
        download(f"https://huggingface.co/{REPO}/resolve/main/{fn}", dst, os.environ.get("HF_TOKEN"))
    log(f"model {key} {dst.stat().st_size / 2**30:.2f} GiB in {time.time() - t:.0f} s")
    return dst


def get_wiki(work):
    d = work / "wikitext"
    f = d / "wikitext-2-raw" / "wiki.test.raw"
    if f.exists():
        return f
    d.mkdir(parents=True, exist_ok=True)
    z = d / "w.zip"
    download("https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip", z)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(d)
    return f


# ---------------------------------------------------------------- tests
TBO_RE = re.compile(r"^\s+([A-Z_0-9]+\(.*\)):\s+.*?(OK|FAIL|not supported)", re.M)


def tbo(bins, dev, out, timeout):
    res = {}
    variants = [(n, b, {}) for n, b in bins.items()] + [(k, bins[a], e) for k, (a, e) in TBO_KILLS.items() if a in bins]
    for name, b, env in variants:
        res[name] = {}
        for op in OPS:
            STATUS["step"] = f"tbo {name} {op}"
            lf = out / f"tbo_{name}_{op}.log"
            rc, text, dt = run([str(b / f"test-backend-ops{EXE}"), "-b", dev, "-o", op], logf=lf, env=env, timeout=timeout)
            cases = {}
            for m in TBO_RE.finditer(text):
                cases[m.group(1)] = m.group(2)
            ok = sum(v == "OK" for v in cases.values())
            fail = sorted(k for k, v in cases.items() if v == "FAIL")
            uns = sum(v == "not supported" for v in cases.values())
            res[name][op] = {"rc": rc, "ok": ok, "fail": len(fail), "unsup": uns, "fail_ids": fail[:50], "s": round(dt)}
            log(f"tbo {name} {op}: ok {ok} fail {len(fail)} unsup {uns} rc {rc} ({dt:.0f} s)")
    return res


def parse_ppl(text):
    def g(p):
        m = re.search(p, text)
        return float(m.group(1)) if m else None
    return {
        "ppl_final": g(r"Final estimate: PPL = ([\d.]+)"),
        "ppl_q": g(r"Mean PPL\(Q\)\s*:\s*([\d.]+)"),
        "ppl_base": g(r"Mean PPL\(base\)\s*:\s*([\d.]+)"),
        "mean_kld": g(r"Mean\s+KLD:\s+([-\d.e+]+)"),
        "max_kld": g(r"Maximum KLD:\s+([-\d.e+]+)"),
        "top1": g(r"Same top p:\s+([\d.]+)"),
        "buf": re.findall(r"(\S+) compute buffer size =\s+([\d.]+) MiB", text)[:4],
    }


def kld(bins, models, wiki, work, out, ngl_args, timeout):
    """For each model and config: BASE saves logits; BASE (A/A), HEAD1, HEAD2 scored against them.
    Extra configs score HEAD2 against HEAD1 logits at small ubatches (round-2 cc 10.x rules)."""
    res = {}
    sref = "head1" if REF == "base" else REF
    cfgs = [("c512", ["-c", "512", "--chunks", "20"], REF),
            ("c2048_ub1024", ["-c", "2048", "-b", "2048", "-ub", "1024", "--chunks", "6"], REF),
            ("c512_ub4", ["-c", "512", "-b", "512", "-ub", "4", "--chunks", "4"], sref),
            ("c512_ub1", ["-c", "512", "-b", "512", "-ub", "1", "--chunks", "2"], sref)]
    if SMOKE:
        cfgs = [("c512", ["-c", "512", "--chunks", "2"], REF), ("c512_ub4", ["-c", "512", "-b", "512", "-ub", "4", "--chunks", "1"], sref)]
    if KLD_CFGS:
        allc = {"c512": ["-c", "512", "--chunks", "20"],
                "c4096_b4096": ["-c", "4096", "-b", "4096", "--chunks", "4"],
                "c2048_ub1024": ["-c", "2048", "-b", "2048", "-ub", "1024", "--chunks", "6"]}
        cfgs = [(k, allc[k], REF) for k in KLD_CFGS]
    if KLD_KILLS and not KLD_CFGS:
        cfgs = [("c512k", ["-c", "512", "--chunks", "6"], REF), ("c512_ub1k", ["-c", "512", "-b", "512", "-ub", "1", "--chunks", "1"], REF)]
    for mk, mpath in models.items():
        for cname, args, ref in cfgs:
            if cname in ("c512_ub4", "c512_ub1") and mk != FIRST_MODEL:
                continue  # small-ubatch checks on the first model only
            if ref not in bins:
                continue
            key = f"{mk}/{cname}"
            STATUS["step"] = f"kld {key}"
            logits = work / f"logits_{mk}_{cname}.bin"
            common = ["-m", str(mpath), "-f", str(wiki), *ngl_args, "-fa", "on", *args]
            rc, text, dt = run([str(bins[ref] / f"llama-perplexity{EXE}"), *common, "--kl-divergence-base", str(logits)],
                               logf=out / f"ppl_{mk}_{cname}_{ref}_save.log", timeout=timeout)
            r = {"ref": ref, "save": {"rc": rc, **parse_ppl(text), "s": round(dt)}}
            log(f"kld {key} save on {ref}: rc {rc} PPL {r['save']['ppl_final']} ({dt:.0f} s)")
            if rc != 0:
                res[key] = r
                continue
            arms = [a for a in SEL if a in bins and SEL.index(a) >= SEL.index(ref)]
            variants = [(a, a, {}) for a in arms]
            if ref == REF:
                variants += [(k, a, e) for k, (a, e) in KILLS.get(REF, {}).items() if a in bins]
            if KLD_KILLS:
                # every switch at once
                for a in {a for _, (a, _) in KILLS.get(REF, {}).items() if a in bins}:
                    allenv = {}
                    for _, (aa, e) in KILLS.get(REF, {}).items():
                        if aa == a:
                            allenv.update(e)
                    variants.append((f"{a}_allkills", a, allenv))
            for a, abin, env in variants:
                rc, text, dt = run([str(bins[abin] / f"llama-perplexity{EXE}"), *common, "--kl-divergence-base", str(logits),
                                    "--kl-divergence"], logf=out / f"ppl_{mk}_{cname}_{a}_vs_{ref}.log", env=env, timeout=timeout)
                r[a] = {"rc": rc, **parse_ppl(text), "s": round(dt)}
                log(f"kld {key} {a} vs {ref}: rc {rc} PPL(Q) {r[a]['ppl_q']} KLD {r[a]['mean_kld']} max {r[a]['max_kld']} top1 {r[a]['top1']}")
            try:
                logits.unlink()
            except OSError:
                pass
            res[key] = r
    return res


def gpu_probe(backend, n=3):
    """Co-tenancy probe before a timed run: GPU busy % samples and foreign GPU processes (AMD: KFD procs)."""
    r = {"busy": [], "foreign": None}
    for _ in range(n):
        v = None
        try:
            if backend == "cuda":
                rc, text, _ = run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
                v = float(text.split()[0])
            elif os.name != "nt":
                vals = [float(Path(f).read_text().strip()) for f in Path("/sys/class/drm").glob("card*/device/gpu_busy_percent")]
                v = max(vals) if vals else None
            else:
                rc, text, _ = run(["powershell", "-NoProfile", "-Command",
                                   "((Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples"
                                   " | Measure-Object CookedValue -Sum).Sum"], timeout=60)
                v = float(text.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            v = None
        r["busy"].append(v)
        time.sleep(1)
    try:
        if backend == "cuda":
            rc, text, _ = run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
            r["foreign"] = len([x for x in text.split() if x.strip().isdigit()])
        elif os.name != "nt":
            r["foreign"] = len([d for d in Path("/sys/class/kfd/kfd/proc").iterdir() if d.name != str(os.getpid())])
    except Exception:  # noqa: BLE001
        pass
    return r


def bench(bins, models, out, ngl_args, timeout, quick):
    """llama-bench -fa 1 -p 512,4096 -n 128 -r 3, interleaved rounds (fresh process each), plus kill switches on head1."""
    present = [a for a in SEL if a in bins]
    order = present + present[::-1] + ([] if quick else present)  # B H1 H2 H2 H1 B B H1 H2
    if SMOKE:
        order = present
    kills = {k: v for k, v in KILLS.get(REF, {}).items() if v[0] in bins}
    res = {}
    for mk, mpath in models.items():
        runs = []
        seq = [(a, a, {}) for a in order]
        seq += [(k, a, e) for k, (a, e) in kills.items()] * (1 if SMOKE else 2)
        for ub in UB_SWEEP:
            for a in present:
                seq.append((f"{a}_ub{ub}", a, {"_args": ["-p", "4096", "-n", "0", "-b", "4096", "-ub", str(ub)]}))
        for i, (label, a, env) in enumerate(seq):
            STATUS["step"] = f"bench {mk} {i} {label}"
            pr = gpu_probe(BACKEND)
            jf = out / f"bench_{mk}_{i:02d}_{label}.json"
            env = dict(env)
            pargs = env.pop("_args", None) or ["-p", "512" if SMOKE else "512,4096", "-n", "16" if SMOKE else "128"]
            rc, text, dt = run([str(bins[a] / f"llama-bench{EXE}"), "-m", str(mpath), *ngl_args, "-fa", "1",
                                *pargs, "-r", "1" if SMOKE else "3", "-o", "json", "-oe", "md"],
                               logf=out / f"bench_{mk}_{i:02d}_{label}.log", env=env, timeout=timeout, stdout_file=jf)
            rows = []
            if jf.exists():
                try:
                    for j in json.loads(jf.read_text(encoding="utf-8")):
                        test = f"pp{j['n_prompt']}" if j.get("n_gen", 0) == 0 else f"tg{j['n_gen']}"
                        rows.append({"test": test, "avg": j["avg_ts"], "sd": j["stddev_ts"], "samples": j.get("samples_ts")})
                except Exception as e:  # noqa: BLE001
                    log(f"bench json parse failed: {e!r}")
            runs.append({"i": i, "arm": label, "rc": rc, "rows": rows, "s": round(dt), "probe": pr})
            log(f"bench {mk} #{i} {label} rc {rc}: " + ", ".join(f"{r['test']} {r['avg']:.2f}" for r in rows) + f"  probe {pr}")
        res[mk] = runs
    return res


BB_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", re.M)


def batched(bins, models, out, ngl_args, timeout):
    present = [a for a in SEL if a in bins]
    order = present if SMOKE else present + present[::-1]
    res = {}
    for mk, mpath in models.items():
        runs = []
        for i, a in enumerate(order):
            STATUS["step"] = f"batched {mk} {i} {a}"
            pr = gpu_probe(BACKEND)
            rc, text, dt = run([str(bins[a] / f"llama-batched-bench{EXE}"), "-m", str(mpath), *ngl_args, "-fa", "on",
                                "-c", "4096", "-npp", "128", "-ntg", "64", "-npl", NPL, "-kvu"],
                               logf=out / f"batched_{mk}_{i:02d}_{a}.log", timeout=timeout)
            rows = [{"npl": int(m.group(3)), "s_pp": float(m.group(6)), "s_tg": float(m.group(8)), "s": float(m.group(10))}
                    for m in BB_RE.finditer(text)]
            runs.append({"i": i, "arm": a, "rc": rc, "rows": rows, "probe": pr})
            log(f"batched {mk} #{i} {a} rc {rc}: " + ", ".join(f"npl{r['npl']} tg {r['s_tg']:.2f}" for r in rows))
        res[mk] = runs
    return res


def gpu_info(backend):
    if backend == "cuda":
        rc, text, _ = run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version", "--format=csv,noheader"])
        return text.strip()
    for c in (["rocminfo"], ["amd-smi", "static", "--asic"]):
        try:
            rc, text, _ = run(c)
            if rc == 0:
                return "\n".join(ln.strip() for ln in text.splitlines() if "gfx" in ln or "Marketing" in ln)[:600]
        except FileNotFoundError:
            pass
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--backend", choices=["cuda", "hip"], required=True)
    ap.add_argument("--arch", default="")
    ap.add_argument("--dev", default="")
    ap.add_argument("--models", default="IQ1_S")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-tbo", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--arms", default="rel,head,headp")
    ap.add_argument("--ref", default="")
    ap.add_argument("--ops", default="")
    ap.add_argument("--ub-sweep", default="")
    ap.add_argument("--kld-kills", action="store_true")
    ap.add_argument("--kld-configs", default="")
    ap.add_argument("--archs", default="", help="run test-llama-archs -a REGEX on every arm")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--bench-models", default="", help="comma list: bench only these models (default all)")
    ap.add_argument("--npl", default="1,4")
    ap.add_argument("--model-wait-min", type=float, default=60.0)
    ap.add_argument("--prebuilt", default="", help="arm=bindir,... : skip prepare/build (harness smoke)")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes, harness check only")
    a = ap.parse_args()
    global BACKEND, SEL, REF, OPS, SMOKE, UB_SWEEP, NPL, KLD_KILLS, NO_BENCH, FIRST_MODEL, KLD_CFGS, ARCHS_RE
    KLD_CFGS = [x for x in a.kld_configs.split(",") if x]
    ARCHS_RE = a.archs
    FIRST_MODEL = a.models.split(",")[0]
    KLD_KILLS, NO_BENCH = a.kld_kills, a.no_bench
    UB_SWEEP = [int(x) for x in a.ub_sweep.split(",") if x]
    NPL = a.npl
    BACKEND = a.backend
    SEL = a.arms.split(",")
    REF = a.ref or SEL[0]
    if a.ops:
        OPS = a.ops.split(",")
    if a.smoke:
        if not a.ops:
            OPS = ["CONCAT", "CLIP_ROWS_FUSE"]
        SMOKE = True
    threading.Thread(target=heartbeat, daemon=True).start()
    kit, work = Path(a.kit).resolve(), Path(a.work).resolve()
    out = work / "out"
    out.mkdir(parents=True, exist_ok=True)
    R = {"host": platform.node(), "os": platform.platform(), "cpus": os.cpu_count(), "backend": a.backend,
         "gpu": gpu_info(a.backend), "t_start": time.strftime("%Y-%m-%dT%H:%M:%S")}
    log("gpu:", R["gpu"])
    arch = a.arch
    if not arch:
        if a.backend == "cuda":
            cc = R["gpu"].split(",")[1].strip()
            arch = cc.replace(".", "")
        else:
            arch = os.environ.get("ROCM_GFX", "gfx1151")
    R["arch"] = arch
    dev = a.dev or ("CUDA0" if a.backend == "cuda" else "ROCm0")

    # model + corpus downloads overlap the builds (network only)
    models, dl_err = {}, []

    ready = {}

    def dl():
        try:
            R["wiki"] = str(get_wiki(work))
            ready["wiki"] = True
            for k in a.models.split(","):
                models[k] = get_model(k, Path(a.model_dir) if a.model_dir else work / "models")
                ready[k] = True
        except Exception as e:  # noqa: BLE001
            dl_err.append(repr(e))
        ready["done"] = True
    th = threading.Thread(target=dl, daemon=True)
    th.start()

    STATUS["step"] = "prepare"
    if a.prebuilt:
        bins = {k: Path(v) for k, v in (x.split("=", 1) for x in a.prebuilt.split(","))}
        R["build"] = "prebuilt"
    else:
        commits = prepare(kit, work / "src")
        R["commits"] = commits
        bins, R["build"] = build_all(work / "src", work, a.backend, arch, commits, out)
    if dl_err:
        R["download_error"] = dl_err
        log("download error", dl_err)
    R["bins"] = sorted(bins)
    R["sel"], R["ref"] = SEL, REF
    if REF not in bins:
        R["verdict_void"] = f"reference arm {REF} did not build"
    ngl = ["-ngl", "99"]
    if not a.skip_tbo:
        R["tbo"] = tbo(bins, dev, out, a.timeout)
    if ARCHS_RE:
        R["archs"] = {}
        for name, b in bins.items():
            STATUS["step"] = f"test-llama-archs {name}"
            rc, text, dt = run([str(b / f"test-llama-archs{EXE}"), "-a", ARCHS_RE], logf=out / f"archs_{name}.log", timeout=a.timeout)
            rows = [ln.strip() for ln in text.splitlines() if ("OK" in ln or "FAIL" in ln) and "|" in ln]
            R["archs"][name] = {"rc": rc, "ok": sum("OK" in r for r in rows), "fail": [r for r in rows if "FAIL" in r][:20], "s": round(dt)}
            log(f"test-llama-archs {name}: rc {rc} ok {R['archs'][name]['ok']} fail {len(R['archs'][name]['fail'])}")
    R["kld"], R["bench"], R["batched"], R["model_skipped"] = {}, {}, {}, []
    first = True
    for mk in a.models.split(","):
        t_wait = time.time()
        cap = 1e9 if first else a.model_wait_min * 60
        while mk not in ready and "done" not in ready and time.time() - t_wait < cap:
            STATUS["step"] = f"waiting for download {mk}"
            time.sleep(15)
        first = False
        if mk not in ready or REF not in bins:
            log(f"model {mk} not available after {time.time() - t_wait:.0f} s: skipped")
            R["model_skipped"].append(mk)
            continue
        one = {mk: models[mk]}
        R["kld"].update(kld(bins, one, Path(R["wiki"]), work, out, ngl, a.timeout))
        if not NO_BENCH and (not a.bench_models or mk in a.bench_models.split(",")):
            R["bench"].update(bench(bins, one, out, ngl, a.timeout, a.quick))
            R["batched"].update(batched(bins, one, out, ngl, a.timeout))
    R["t_total_s"] = round(time.time() - T0)
    (out / "results.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    print("XGPU_RESULTS_BEGIN", flush=True)
    print(json.dumps(R), flush=True)
    print("XGPU_RESULTS_END", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Probe: upstream stable-diffusion.cpp `--linear-type dit=i8` (runtime Q8_0 -> INT8 requantization of DiT linears,
leejet/stable-diffusion.cpp #2445) vs `--linear-type raw` on ONE binary, on this card.

Observes only; criteria/sdcpp_dit_i8.py judges. Both states are flags on the same binary, so the whole measurement
is done once into --shared (shared.json) and every state's observation carries it.

  * binary: upstream release --tag, tried in order: Vulkan zip -> ROCm zip -> build from source at --commit
    (Vulkan when glslc exists, else HIP gfx1151). A candidate is accepted when `sd-cli --list-devices` exits 0 and
    lists a GPU device. Which one ran and its device list are recorded.
  * public models (no token is sent: every *TOKEN* env var is dropped for curl).
  * per model, per prompt, arms in rotating order raw / i8 / raw2 (raw2 = raw repeat, the A/A), one fresh sd-cli per
    render, -v. Per run: rc, sampling s, s/step, TE s, total s (if printed), wall, peak RSS, PNG sha256, log lines
    mentioning fallback / unsupported, and lines mentioning int8 / i8 / requant.
  * scoring after all renders (GPU idle): LPIPS-alex on CPU torch in a venv when it installs, else PSNR / max-abs.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "leejet/stable-diffusion.cpp"
HF = "https://huggingface.co"
PROMPTS = [
    ("c00", "a golden retriever puppy playing in autumn leaves, shallow depth of field, warm afternoon light", 20260920),
    ("c01", "a neon-lit ramen shop in a rainy Tokyo alley at night, reflections on wet pavement, cinematic", 20260921),
    ("c02", "an elderly woman knitting a colorful scarf by a fireplace, cozy interior, soft lamp light", 20260922),
    ("c03", "a red vintage pickup truck parked beside a wheat field, clear blue sky, film photography look", 20260923),
]
FILES = {
    "qwen": {
        "dit": ("unsloth/Qwen-Image-2.1-GGUF", "qwen-image-2.1-Q8_0.gguf"),
        "llm": ("unsloth/Qwen3-VL-8B-Instruct-GGUF", "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"),
        "vae": ("unsloth/Qwen-Image-2.1-FP8", "vae/qwen_image_2.1_vae_bf16.safetensors"),
    },
    "krea": {
        "src": ("unsloth/Krea-2-Turbo", "turbo.safetensors"),
        "llm": ("unsloth/Qwen3-VL-4B-Instruct-GGUF", "Qwen3-VL-4B-Instruct-Q8_0.gguf"),
        "vae": ("unsloth/Krea-2-Turbo", "vae/diffusion_pytorch_model.safetensors"),
    },
}
STEPS = {"qwen": 25, "krea": 8}
ARMS = {"raw": ["--linear-type", "raw"], "i8": ["--linear-type", "dit=i8"], "raw2": ["--linear-type", "raw"]}
ORDER = ["raw", "i8", "raw2"]
RX = {
    "sampling_s": re.compile(r"sampling completed, taking ([0-9.]+)(m?s)"),
    "generate_s": re.compile(r"generate_image completed in ([0-9.]+)(m?s)"),
    "load_s": re.compile(r"loading tensors completed, taking ([0-9.]+)(m?s)"),
}
TE_RX = re.compile(r"computing condition graph completed, taking ([0-9.]+) ms")
FALLBACK_RX = re.compile(r"fall ?back|unsupported|not supported|falling back", re.I)
I8_RX = re.compile(r"\bi8\b|int8|requant|linear[_ ]type|convrot", re.I)
T0 = time.time()
KEEP: list = []  # [out dir]: runs + partial results land in the uploaded artifact as they happen


def say(msg: str) -> None:
    print(f"[sdcpp_dit_i8 {time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:.1f}m] {msg}", flush = True)


def wjson(path: Path, obj) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent = 1, default = str), encoding = "utf-8")
    os.replace(tmp, path)
    if KEEP and path.name == "partial.json":
        shutil.copy2(path, KEEP[0] / "partial_shared.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def notoken_env() -> dict:
    return {k: v for k, v in os.environ.items() if "TOKEN" not in k.upper()}


def fetch(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents = True, exist_ok = True)
    if dest.is_file() and dest.stat().st_size > 0:
        return {"url": url, "cached": True, "bytes": dest.stat().st_size}
    t0 = time.time()
    p = subprocess.run(["curl", "-sS", "--fail", "-L", "--retry", "5", "--retry-delay", "5", "-o", str(dest) + ".part",
                        url], env = notoken_env(), capture_output = True, text = True, errors = "replace")
    if p.returncode != 0:
        return {"url": url, "error": f"curl rc={p.returncode} {p.stderr[-300:]}"}
    os.replace(str(dest) + ".part", dest)
    return {"url": url, "bytes": dest.stat().st_size, "seconds": round(time.time() - t0, 1)}


def sh(cmd, timeout = 600, cwd = None, env = None) -> dict:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output = True, text = True, errors = "replace", timeout = timeout, cwd = cwd,
                           env = env)
        return {"cmd": " ".join(map(str, cmd))[:300], "rc": p.returncode, "s": round(time.time() - t0, 1),
                "out": (p.stdout + p.stderr)[-3000:]}
    except subprocess.TimeoutExpired as e:
        return {"cmd": " ".join(map(str, cmd))[:300], "rc": "timeout", "s": round(time.time() - t0, 1),
                "out": str(e.stdout or "")[-1500:]}
    except FileNotFoundError as e:
        return {"cmd": " ".join(map(str, cmd))[:300], "rc": "missing", "out": str(e)}


_LIBDIRS: dict = {}
_ROCM_DIRS: list = []
ROCM_LIBS = ("libhipblas.so", "libamdhip64.so", "librocblas.so", "libhipblaslt.so")


def rocm_lib_dirs() -> list:
    """Dirs under /opt holding the HIP runtime / BLAS libs (the runner's ROCm is not on the default loader path)."""
    if _ROCM_DIRS:
        return _ROCM_DIRS
    found = set()
    for top in sorted(Path("/opt").glob("rocm*")):
        for dirpath, dirnames, filenames in os.walk(top):
            if dirpath.count(os.sep) - str(top).count(os.sep) >= 5:
                dirnames[:] = []
            if any(f.startswith(ROCM_LIBS) for f in filenames):
                found.add(dirpath)
    _ROCM_DIRS.extend(sorted(found) or ["/opt/rocm/lib"])
    return _ROCM_DIRS


def extract(zpath: Path, dest: Path) -> dict:
    """unzip keeps the bundle's symlinks (zipfile.extractall writes them as tiny text files: 'file too short')."""
    if shutil.which("unzip"):
        r = sh(["unzip", "-q", "-o", str(zpath), "-d", str(dest)], timeout = 600)
        if r["rc"] == 0:
            return {"via": "unzip"}
    links = 0
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            target = dest / info.filename
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                target.parent.mkdir(parents = True, exist_ok = True)
                if target.is_symlink() or target.exists():
                    target.unlink()
                os.symlink(z.read(info).decode("utf-8"), target)
                links += 1
            else:
                z.extract(info, dest)
    return {"via": "zipfile", "symlinks": links}


def bin_env(binary: Path) -> dict:
    env = dict(os.environ)
    if binary not in _LIBDIRS:
        root = binary.parent.parent if binary.parent.name == "bin" else binary.parent
        _LIBDIRS[binary] = sorted({str(p.parent) for p in root.rglob("*.so*")} | {str(binary.parent)})[:40] + \
            rocm_lib_dirs()
    libdirs = _LIBDIRS[binary]
    env["LD_LIBRARY_PATH"] = ":".join(libdirs + [env.get("LD_LIBRARY_PATH", "")])
    return env


def try_binary(binary: Path) -> dict:
    binary.chmod(0o755)
    r = sh([str(binary), "--list-devices"], timeout = 120, env = bin_env(binary))
    out = r.get("out", "")
    r["ldd"] = sh(["ldd", str(binary)], timeout = 30, env = bin_env(binary)).get("out", "")[-1500:]
    r["ld_library_path"] = bin_env(binary)["LD_LIBRARY_PATH"][:1500]
    r["gpu_listed"] = r["rc"] == 0 and bool(re.search(r"Vulkan\d|ROCm\d|HIP\d|CUDA\d|Radeon|gfx11", out))
    return r


def get_binary(shared: Path, tag: str, commit: str) -> dict:
    out: dict = {"attempts": []}
    base = f"https://github.com/{REPO}/releases/download/{tag}"
    short = commit[:7]
    for backend, asset in (("vulkan", f"sd-master-{short}-bin-Linux-Ubuntu-26.04-x86_64-vulkan.zip"),
                           ("rocm", f"sd-master-{short}-bin-Linux-Ubuntu-24.04-x86_64-rocm-10.1.0.zip")):
        att: dict = {"source": f"release zip {asset}", "backend": backend}
        att["fetch"] = fetch(f"{base}/{asset}", shared / asset)
        if "error" in att["fetch"]:
            out["attempts"].append(att)
            continue
        att["sha256"] = sha256(shared / asset)
        bdir = shared / f"bundle_{backend}"
        try:
            att["extract"] = extract(shared / asset, bdir)
        except Exception as e:  # noqa: BLE001
            att["error"] = f"unzip: {e}"
            out["attempts"].append(att)
            continue
        clis = sorted(bdir.rglob("sd-cli"))
        if not clis:
            att["error"] = "no sd-cli in zip: " + ", ".join(str(p.relative_to(bdir)) for p in list(bdir.rglob("*"))[:30])
            out["attempts"].append(att)
            continue
        for f in bdir.rglob("*"):
            if f.is_file() and not f.is_symlink():
                f.chmod(0o755)
        att["binary"] = str(clis[0])
        att["list_devices"] = try_binary(clis[0])
        out["attempts"].append(att)
        if att["list_devices"]["gpu_listed"]:
            out.update(binary = str(clis[0]), backend = backend, source = att["source"])
            return out
        say(f"{backend} zip unusable: {att['list_devices'].get('out', '')[-400:]}")
    # both release zips failed: build on the runner at the commit
    att = {"source": f"source build at {commit}"}
    src = shared / "src"
    steps = []
    if not (src / ".git").is_dir():
        src.mkdir(parents = True, exist_ok = True)
        steps.append(sh(["git", "init", "-q"], cwd = src))
        steps.append(sh(["git", "remote", "add", "origin", f"https://github.com/{REPO}.git"], cwd = src))
        steps.append(sh(["git", "fetch", "-q", "--depth", "1", "origin", commit], timeout = 900, cwd = src))
        steps.append(sh(["git", "checkout", "-q", "FETCH_HEAD"], cwd = src))
        steps.append(sh(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], timeout = 1800,
                        cwd = src))
    glslc = shutil.which("glslc")
    nproc = str(os.cpu_count() or 8)
    if glslc:
        att["backend"] = "vulkan"
        cfg = ["cmake", "-S", str(src), "-B", str(src / "build"), "-DCMAKE_BUILD_TYPE=Release", "-DSD_VULKAN=ON"]
    else:
        att["backend"] = "rocm"
        cfg = ["cmake", "-S", str(src), "-B", str(src / "build"), "-DCMAKE_BUILD_TYPE=Release", "-DSD_HIPBLAS=ON",
               "-DGPU_TARGETS=gfx1151"]
        for cc, cxx in (("/opt/rocm/llvm/bin/clang", "/opt/rocm/llvm/bin/clang++"),):
            if Path(cc).is_file():
                cfg += [f"-DCMAKE_C_COMPILER={cc}", f"-DCMAKE_CXX_COMPILER={cxx}"]
    steps.append(sh(cfg, timeout = 900))
    steps.append(sh(["cmake", "--build", str(src / "build"), "--config", "Release", "-j", nproc, "--target", "sd-cli"],
                    timeout = 3600))
    att["steps"] = steps
    clis = sorted((src / "build").rglob("sd-cli")) if (src / "build").is_dir() else []
    clis = [c for c in clis if c.is_file() and os.access(c, os.X_OK)]
    if clis:
        att["binary"] = str(clis[0])
        att["list_devices"] = try_binary(clis[0])
        if att["list_devices"]["gpu_listed"]:
            out.update(binary = str(clis[0]), backend = att["backend"], source = att["source"])
    out["attempts"].append(att)
    return out


def run_one(binary: Path, argv: list, out_png: Path, log_path: Path, timeout: int) -> dict:
    cmd = [str(binary), *argv, "--output", str(out_png), "-v"]
    t0 = time.time()
    timed_out = False
    with open(log_path, "w", encoding = "utf-8", errors = "replace") as fh:
        p = subprocess.Popen(cmd, stdout = fh, stderr = subprocess.STDOUT, env = bin_env(binary))
        try:
            p.wait(timeout = timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            p.kill()
            p.wait()
    li = argv.index("--linear-type")
    rec = {"rc": "timeout" if timed_out else p.returncode, "wall_s": round(time.time() - t0, 3),
           "linear_type": argv[li + 1]}
    text = log_path.read_text(encoding = "utf-8", errors = "replace")
    for key, rx in RX.items():
        vals = [float(v) / (1000.0 if u == "ms" else 1.0) for v, u in rx.findall(text)]
        rec[key] = vals[-1] if vals else None
    te = [float(v) / 1000.0 for v in TE_RX.findall(text)]
    rec["te_s"] = round(sum(te), 3) if te else None
    lines = text.splitlines()
    rec["fallback_lines"] = list(dict.fromkeys(ln.strip()[:300] for ln in lines if FALLBACK_RX.search(ln)))[:30]
    rec["i8_lines"] = list(dict.fromkeys(ln.strip()[:300] for ln in lines if I8_RX.search(ln)))[:30]
    rec["sha256"] = sha256(out_png) if out_png.is_file() and rec["rc"] == 0 else None
    rec["tail"] = text[-1500:] if rec["rc"] != 0 else None
    return rec


def peak_rss_note() -> dict:
    try:
        mem = Path("/proc/meminfo").read_text(encoding = "utf-8").splitlines()[:3]
    except OSError:
        mem = []
    return {"meminfo": mem}


def model_argv(model: str, m: dict, prompt: str, seed: int, flags: list) -> list:
    return ["--diffusion-model", m["dit"], "--llm", m["llm"], "--vae", m["vae"], "-p", prompt, "--seed", str(seed),
            "-W", "1024", "-H", "1024", "--steps", str(STEPS[model]), "--cfg-scale", "1.0", "--sampling-method",
            "euler", "--diffusion-fa", *flags]


def render_model(out: dict, model: str, binary: Path, m: dict, runs_dir: Path, shared: Path, deadline: float,
                 run_timeout: int) -> None:
    res = out["models"][model]
    res["runs"] = {}
    dead_arms: set = set()
    for i, (pid, prompt, seed) in enumerate(PROMPTS):
        order = ORDER[i % 3:] + ORDER[:i % 3]
        for arm in order:
            key = f"{model}__{pid}__{arm}"
            if arm in dead_arms:
                res["runs"].setdefault(pid, {})[arm] = {"rc": "skipped", "why": "this arm timed out earlier"}
                continue
            if time.time() > deadline:
                res["runs"].setdefault(pid, {})[arm] = {"rc": "skipped", "why": "probe time budget exhausted"}
                continue
            say(f"run {key}")
            rec = run_one(binary, model_argv(model, m, prompt, seed, ARMS[arm]), runs_dir / f"{key}.png",
                          runs_dir / f"{key}.log", run_timeout)
            rec["order_pos"] = order.index(arm)
            if isinstance(rec.get("sampling_s"), (int, float)):
                rec["s_per_step"] = round(rec["sampling_s"] / STEPS[model], 4)
            if rec["rc"] == "timeout":
                dead_arms.add(arm)
            res["runs"].setdefault(pid, {})[arm] = rec
            say(f"  rc={rec['rc']} sampling={rec['sampling_s']} s/step={rec.get('s_per_step')} wall={rec['wall_s']} "
                f"sha={str(rec['sha256'])[:12]} fallback={len(rec['fallback_lines'])}")
            wjson(shared / "partial.json", out)


def fetch_models(shared: Path, models: list) -> dict:
    jobs = {}
    with cf.ThreadPoolExecutor(max_workers = 6) as ex:
        for model in models:
            for key, (repo, name) in FILES[model].items():
                dest = shared / "models" / repo.replace("/", "__") / name
                jobs[(model, key)] = (dest, ex.submit(fetch, f"{HF}/{repo}/resolve/main/{name}", dest))
        return {f"{mo}.{k}": {"path": str(d), **f.result()} for (mo, k), (d, f) in jobs.items()}


def score(shared: Path, runs_dir: Path, out: dict) -> dict:
    """LPIPS-alex on CPU torch in a venv; the scorer itself falls back to PSNR / max-abs without torch."""
    sc: dict = {}
    venv = shared / "scorevenv"
    py = venv / "bin" / "python"
    if not py.is_file():
        sc["venv"] = sh([sys.executable, "-m", "venv", str(venv)], timeout = 300)
    if py.is_file():
        env = notoken_env()
        sc["pip_torch"] = sh([str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check", "torch",
                              "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"], timeout = 1200,
                             env = env)
        sc["pip_rest"] = sh([str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check", "numpy", "pillow",
                             "scipy"], timeout = 600, env = env)
        sc["pip_lpips"] = sh([str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-deps",
                              "lpips"], timeout = 300, env = env)
        for k in ("pip_torch", "pip_rest", "pip_lpips"):
            if sc[k].get("rc") == 0:
                sc[k]["out"] = sc[k]["out"][-300:]
    scorer_py = str(py) if py.is_file() else sys.executable
    pairs = []
    for model, res in out["models"].items():
        for pid, arms in (res.get("runs") or {}).items():
            for arm in ("i8", "raw2"):
                a, b = runs_dir / f"{model}__{pid}__raw.png", runs_dir / f"{model}__{pid}__{arm}.png"
                if a.is_file() and b.is_file() and (arms.get(arm) or {}).get("rc") == 0 and \
                        (arms.get("raw") or {}).get("rc") == 0:
                    pairs.append({"model": model, "prompt": pid, "pair": f"{arm}_vs_raw", "a": str(a), "b": str(b)})
    pj = shared / "pairs.json"
    wjson(pj, pairs)
    res_path = shared / "scores.json"
    sc["scorer"] = sh([scorer_py, "-I", str(HERE / "sdcpp_dit_i8_score.py"), str(pj), str(res_path)], timeout = 1800)
    sc["scorer_python"] = scorer_py
    if res_path.is_file():
        sc.update(json.loads(res_path.read_text(encoding = "utf-8")))
    return sc


def shared_half(a) -> dict:
    shared: Path = a.shared
    done = shared / "shared.json"
    if done.is_file():
        return json.loads(done.read_text(encoding = "utf-8"))
    shared.mkdir(parents = True, exist_ok = True)
    deadline = T0 + a.budget_min * 60
    models = [m for m in a.models.split(",") if m]
    out: dict = {"tag": a.tag, "commit": a.commit, "arms": ARMS, "order": ORDER, "steps": STEPS,
                 "prompts": PROMPTS, "models": {m: {} for m in models}, "host": peak_rss_note()}
    du = shutil.disk_usage(shared)
    out["disk_free_gb_start"] = round(du.free / 1e9, 1)
    out["nproc"] = os.cpu_count()
    out["vulkaninfo"] = sh(["vulkaninfo", "--summary"], timeout = 60).get("out", "")[-2500:]
    out["rocminfo_gfx"] = sorted(set(re.findall(r"gfx\d+", sh(["rocminfo"], timeout = 60).get("out", ""))))
    say("acquiring binary")
    out["bin"] = get_binary(shared, a.tag, a.commit)
    wjson(shared / "partial.json", out)
    if not out["bin"].get("binary"):
        wjson(done, out)
        return out
    binary = Path(out["bin"]["binary"])
    out["bin"]["list_devices_final"] = try_binary(binary)
    # Krea needs ~31 GB of downloads plus a ~14 GB GGUF on top of Qwen's ~13.5 GB.
    if "krea" in models and out["disk_free_gb_start"] < 70:
        out["models"]["krea"]["skipped"] = f"disk free {out['disk_free_gb_start']} GB < 70 GB needed"
        models = [m for m in models if m != "krea"]
    say(f"downloading models {models}")
    out["model_fetch"] = fetch_models(shared, models)
    wjson(shared / "partial.json", out)
    runs_dir = (KEEP[0] if KEEP else shared) / "i8_runs"
    runs_dir.mkdir(parents = True, exist_ok = True)
    for model in models:
        mf = {k.split(".", 1)[1]: v for k, v in out["model_fetch"].items() if k.startswith(model + ".")}
        if any("error" in v for v in mf.values()):
            out["models"][model]["skipped"] = "download failed: " + json.dumps(
                {k: v.get("error") for k, v in mf.items() if "error" in v})
            continue
        paths = {k: v["path"] for k, v in mf.items()}
        if model == "krea":
            gguf = shared / "models" / "krea-2-turbo-Q8_0.gguf"
            if time.time() > deadline - 20 * 60:
                out["models"][model]["skipped"] = "time budget too short for conversion + renders"
                continue
            if not gguf.is_file():
                say("converting Krea-2 Turbo to Q8_0")
                t0 = time.time()
                with open(shared / "krea_convert.log", "w", encoding = "utf-8", errors = "replace") as fh:
                    p = subprocess.run([str(binary), "-M", "convert", "--diffusion-model", paths["src"], "-o",
                                        str(gguf) + ".part", "--type", "q8_0", "-v"], stdout = fh,
                                       stderr = subprocess.STDOUT, env = bin_env(binary))
                ct = (shared / "krea_convert.log").read_text(encoding = "utf-8", errors = "replace")
                out["models"][model]["convert"] = {"rc": p.returncode, "s": round(time.time() - t0, 1),
                                                   "tail": ct[-2000:], "meminfo_after": peak_rss_note()}
                if p.returncode == 0 and Path(str(gguf) + ".part").is_file():
                    os.replace(str(gguf) + ".part", gguf)
                    out["models"][model]["convert"]["gguf_gb"] = round(gguf.stat().st_size / 1e9, 2)
                    Path(paths["src"]).unlink(missing_ok = True)  # free 26 GB
                else:
                    out["models"][model]["skipped"] = f"conversion failed rc={p.returncode}"
                    say(out["models"][model]["skipped"])
                    continue
            paths["dit"] = str(gguf)
        out["models"][model]["files"] = paths
        render_model(out, model, binary, paths, runs_dir, shared, deadline, a.run_timeout)
    say("scoring")
    try:
        out["scores"] = score(shared, runs_dir, out)
    except Exception as e:  # noqa: BLE001
        out["scores"] = {"error": f"{type(e).__name__}: {e}"}
    out["probe_minutes"] = round((time.time() - T0) / 60, 1)
    wjson(done, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--shared", required = True, type = Path)
    ap.add_argument("--tag", default = "master-1147-3b0426f")
    ap.add_argument("--commit", default = "3b0426f7faf6256d1dbfa2046330cb2dc0c25861")
    ap.add_argument("--models", default = "qwen,krea")
    ap.add_argument("--budget-min", type = float, default = 140.0)
    ap.add_argument("--run-timeout", type = int, default = 1500)
    a = ap.parse_args()
    a.out.parent.mkdir(parents = True, exist_ok = True)
    KEEP.append(a.out.parent)
    obs: dict = {"state": a.state}
    try:
        obs["shared"] = shared_half(a)
    except Exception as e:  # noqa: BLE001
        import traceback
        obs["shared"] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2000:]}
        p = a.shared / "partial.json"
        if p.is_file():
            obs["shared"]["partial"] = json.loads(p.read_text(encoding = "utf-8"))
    keep = a.out.parent / "i8_runs"
    keep.mkdir(parents = True, exist_ok = True)
    for extra in ("krea_convert.log", "scores.json", "pairs.json"):
        if (a.shared / extra).is_file():
            shutil.copy2(a.shared / extra, keep / extra)
    wjson(a.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Probe: sd.cpp offload VAE decode on THIS card, tiled (what the base state picks) vs one whole-image tile (what
PR 18999's head picks when the budget holds it), FLUX.2 klein 4B Q4_K_M at --sizes px, published sd.cpp bundles.

Observes only; criteria/sdcpp_whole_vae.py judges.

Per state:
  * the state's OWN offload flags for family flux.2-klein, from its checkout: sd_cpp_backend._offload_vae_flags
    (the real chooser, real card reading) when the Studio backend imports on this python, else the same chain
    rebuilt from the dependency-free modules it calls (sd_cpp_args, sd_cpp_multigpu, video_minimax_h3,
    diffusion_memory, diffusion_small_host). Asked for the Vulkan class, and (Linux, --rocm) the ROCm class;
  * a run of that argv per size, plus its untiled reference: the same argv minus --vae-tiling / --vae-tile-size /
    --vae-on-cpu (same sampling and TE placement, so the latent is the same and only the decode differs);
Shared arms (once, cached by canonical argv so a state whose argv matches reuses that process): Vulkan untiled
(twice at the first size: A/A), forced 1024 px tiles, forced 2048 px whole tile, forced CPU whole tile.
Per run: rc, decode / sampling / total seconds, every `<desc> compute buffer size: X MB(VRAM|RAM) on <dev>` line,
VAE tile-reduction / OOM-retry lines, peak RSS, sha256 and PSNR vs its untiled reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO = "unslothai/stable-diffusion.cpp"
HF = "https://huggingface.co"
WIN = sys.platform == "win32"
FILES = {
    "dit": ("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_K_M.gguf"),
    "llm": ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-BF16.gguf"),
    # The ungated unsloth copy of the FLUX.2 klein VAE (the black-forest-labs URL answered HTTP errors to the runners).
    "vae": ("unsloth/FLUX.2-klein-4B", "vae/diffusion_pytorch_model.safetensors"),
}
FAMILY = "flux.2-klein"
PROMPT = "a photo of a red fox sitting in fresh snow, golden hour, detailed fur, a wooden cabin behind"
TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN")
GPU_COMMON = ["--offload-to-cpu", "--clip-on-cpu", "--diffusion-fa"]
RX = {
    "sampling_s": re.compile(r"sampling completed, taking ([0-9.]+)(m?s)"),
    "decode_s": re.compile(r"decode_first_stage completed, taking ([0-9.]+)(m?s)"),
    "vae_graph_s": re.compile(r"computing vae decode graph completed, taking ([0-9.]+)(m?s)"),
    "generate_s": re.compile(r"generate_image completed in ([0-9.]+)(m?s)"),
}
BUF = re.compile(r"(\S+) compute buffer size: ([0-9.]+) MB\((\w+)\) on (\S+)")
REDUCE = re.compile(r"(Reducing VAE [^\n]*|VAE \w+ ran out of memory[^\n]*|vae decode compute failed[^\n]*|"
                    r"graph allocation failed[^\n]*|ErrorOutOfDeviceMemory[^\n]*|out of memory[^\n]*)", re.I)
TILES = re.compile(r"num tiles : (\d+), (\d+)")


def say(msg: str) -> None:
    print(f"[sdcpp_whole_vae {time.strftime('%H:%M:%S')}] {msg}", flush = True)


def wjson(path: Path, obj) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent = 1, default = str), encoding = "utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_env() -> dict:
    return {k: v for k, v in os.environ.items() if k not in TOKEN_VARS and "TOKEN" not in k}


def fetch(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents = True, exist_ok = True)
    if dest.is_file() and dest.stat().st_size > 0:
        return {"url": url, "cached": True, "bytes": dest.stat().st_size}
    t0 = time.time()
    curl = "curl.exe" if WIN else "curl"
    rcs = []
    rc = 1
    for _ in range(8):  # HF answered 502 / HTTP2 CANCEL to the runners on the first run
        rc = subprocess.run([curl, "-sS", "--fail", "-L", "--http1.1", "--retry", "5", "--retry-delay", "5", "--retry-all-errors",
                             "-o", str(dest) + ".part", url], env = clean_env()).returncode
        rcs.append(rc)
        if rc == 0:
            break
        time.sleep(30)
    if rc != 0:
        return {"url": url, "error": f"curl rc={rcs}"}
    os.replace(str(dest) + ".part", dest)
    return {"url": url, "bytes": dest.stat().st_size, "seconds": round(time.time() - t0, 1)}


def ensure_imaging() -> str:
    """numpy + pillow for PSNR; httpx + structlog so the state's sd_cpp_backend imports on a bare venv (Windows)."""
    missing = []
    for mod, pkg in (("numpy", "numpy"), ("PIL", "pillow"), ("httpx", "httpx"), ("structlog", "structlog")):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            missing.append(pkg)
    if not missing:
        return "present"
    p = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], capture_output = True,
                       text = True, errors = "replace", env = clean_env())
    return f"installed {missing}: pip rc={p.returncode} {(p.stdout + p.stderr)[-300:]}"


def psnr(a: Path, b: Path):
    try:
        import numpy as np
        from PIL import Image

        x = np.asarray(Image.open(a).convert("RGB")).astype("float64")
        y = np.asarray(Image.open(b).convert("RGB")).astype("float64")
        if x.shape != y.shape:
            return None
        mse = float(((x - y) ** 2).mean())
        return 99.0 if mse == 0 else round(10 * np.log10(255.0**2 / mse), 3)
    except Exception:  # noqa: BLE001
        return None


def _peak_rss_windows(handle):
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    fn = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    if not fn(wintypes.HANDLE(int(handle)), ctypes.byref(pmc), pmc.cb):
        return None
    return round(pmc.PeakWorkingSetSize / 2**30, 3)


def pairs(flags: list) -> list:
    """Flag list -> sorted (flag, value-or-None) pairs: argv identity regardless of order."""
    out, i = [], 0
    while i < len(flags):
        f = flags[i]
        if i + 1 < len(flags) and not str(flags[i + 1]).startswith("--"):
            out.append((f, str(flags[i + 1])))
            i += 2
        else:
            out.append((f, None))
            i += 1
    return sorted(out, key = lambda p: (p[0], p[1] or ""))


def unpairs(ps: list) -> list:
    out = []
    for f, v in ps:
        out.append(f)
        if v is not None:
            out.append(v)
    return out


def untiled_of(flags: list) -> list:
    return unpairs([p for p in pairs(flags) if p[0] not in ("--vae-tiling", "--vae-tile-size", "--vae-on-cpu")])


def tile_of(flags: list):
    ps = dict(pairs(flags))
    if "--vae-tiling" not in ps:
        return None
    return int(ps["--vae-tile-size"]) if ps.get("--vae-tile-size") else 256  # sd.cpp's own default


def run_key(backend: str, size: int, flags: list, rep: int = 0) -> str:
    return hashlib.sha1(json.dumps([backend, size, pairs(flags), rep]).encode()).hexdigest()[:16]


class Runner:
    def __init__(self, shared: Path, models: dict, binaries: dict, steps: int):
        self.dir = shared / "runs"
        self.dir.mkdir(parents = True, exist_ok = True)
        self.models, self.binaries, self.steps = models, binaries, steps

    def run(self, backend: str, size: int, flags: list, rep: int = 0) -> dict:
        key = run_key(backend, size, flags, rep)
        cache = self.dir / f"{key}.json"
        if cache.is_file():
            rec = json.loads(cache.read_text(encoding = "utf-8"))
            rec["cached"] = True
            return rec
        m = self.models
        argv = ["--diffusion-model", m["dit"], "--vae", m["vae"], "--vae-format", "flux2", "--llm", m["llm"], "--steps", str(self.steps),
                "--cfg-scale", "1.0", "-W", str(size), "-H", str(size), "--prompt", PROMPT, "--seed", "42",
                "-v", *flags]
        png, log = self.dir / f"{key}.png", self.dir / f"{key}.log"
        say(f"run {backend} {size}px {' '.join(flags)} rep {rep} -> {key}")
        t0 = time.time()
        peak = None
        with open(log, "w", encoding = "utf-8", errors = "replace") as fh:
            env = clean_env()
            if not WIN:
                env["LD_LIBRARY_PATH"] = f"{Path(self.binaries[backend]).parent}:{env.get('LD_LIBRARY_PATH', '')}"
            p = subprocess.Popen([self.binaries[backend], *argv, "--output", str(png)], stdout = fh,
                                 stderr = subprocess.STDOUT, env = env)
            if WIN:
                rc = p.wait()
                try:
                    peak = _peak_rss_windows(p._handle)
                except Exception:  # noqa: BLE001
                    peak = None
            else:
                _, status, ru = os.wait4(p.pid, 0)
                p.returncode = rc = os.waitstatus_to_exitcode(status)
                peak = round(ru.ru_maxrss / 1048576, 3)
        text = log.read_text(encoding = "utf-8", errors = "replace")
        rec = {"key": key, "backend": backend, "size": size, "flags": flags, "rep": rep, "tile_passed": tile_of(flags),
               "rc": rc, "wall_s": round(time.time() - t0, 2), "peak_rss_gb": peak}
        for k, rx in RX.items():
            vals = [float(v) / (1000.0 if u == "ms" else 1.0) for v, u in rx.findall(text)]
            rec[k] = vals[-1] if vals else None
        bufs = [{"desc": d, "mib": float(mb), "where": w, "dev": dev} for d, mb, w, dev in BUF.findall(text)]
        rec["buffers"] = bufs
        vae = [b for b in bufs if "vae" in b["desc"].lower()]
        rec["vae_buffer_mib"] = max((b["mib"] for b in vae), default = None)
        rec["vae_buffer_where"] = sorted({f"{b['where']}:{b['dev']}" for b in vae})
        rec["vae_tiles"] = [list(map(int, t)) for t in TILES.findall(text)][-1:] or None
        rec["reduce_or_oom"] = sorted(set(x.strip()[:200] for x in REDUCE.findall(text)))[:8]
        rec["png"] = png.name if png.is_file() else None
        rec["sha256"] = sha256(png) if png.is_file() else None
        rec["tail"] = text[-1200:] if rc != 0 or not png.is_file() else None
        say(f"  rc={rc} decode={rec['decode_s']} sampling={rec['sampling_s']} vae_buf={rec['vae_buffer_mib']} "
            f"{rec['vae_buffer_where']} tiles={rec['vae_tiles']} reduce={rec['reduce_or_oom']}")
        wjson(cache, rec)
        return rec


def prepare(shared: Path, tag: str, rocm: bool, checkout: Path) -> dict:
    done = shared / "prepared.json"
    if done.is_file():
        return json.loads(done.read_text(encoding = "utf-8"))
    out: dict = {"tag": tag, "platform": sys.platform, "imaging": ensure_imaging(), "bundles": {}}
    base = f"https://github.com/{REPO}/releases/download/{tag}"
    out["sums_fetch"] = fetch(f"{base}/sd-prebuilt-sha256.json", shared / "sd-prebuilt-sha256.json")
    try:
        sums = json.loads((shared / "sd-prebuilt-sha256.json").read_text(encoding = "utf-8"))
    except Exception:  # noqa: BLE001
        sums = {}
    wanted = {"vulkan": (base, f"sd-{tag}-bin-win-vulkan-x64.zip" if WIN
                         else f"sd-{tag}-bin-Linux-Ubuntu-22.04-x86_64-vulkan.zip")}
    if rocm and not WIN:
        # What Studio's installer resolves for --accelerator rocm on Linux x86_64 at this pin (`install_sd_cpp_prebuilt.py
        # --print-asset --accelerator rocm`, checked when this probe was written): the mirror ships no ROCm asset, so
        # the upstream build of the same commit. Unpatched upstream: no mirror sha256 to check it against.
        utag = re.sub(r"-u[0-9a-f]+$", "", tag)
        short = re.sub(r"^master-\d+-", "master-", utag)
        wanted["rocm"] = (f"https://github.com/leejet/stable-diffusion.cpp/releases/download/{utag}",
                          f"sd-{short}-bin-Linux-Ubuntu-24.04-x86_64-rocm-10.1.0.zip")
    binaries = {}
    for backend, (burl, asset) in wanted.items():
        rec = {"asset": asset, "fetch": fetch(f"{burl}/{asset}", shared / asset)}
        got = sha256(shared / asset) if (shared / asset).is_file() else None
        want = sums.get(asset) if backend == "vulkan" else None
        rec["sha256"] = {"want": want, "got": got, "match": bool(want and want == got)}
        bdir = shared / f"bundle_{backend}"
        if got:
            with zipfile.ZipFile(shared / asset) as z:
                z.extractall(bdir)
        clis = sorted(bdir.rglob("sd-cli.exe" if WIN else "sd-cli"))
        if clis:
            b = clis[0]
            if not WIN:
                b.chmod(0o755)
                for so in b.parent.glob("*.so*"):
                    so.chmod(0o755)
            env = clean_env()
            if not WIN:
                env["LD_LIBRARY_PATH"] = f"{b.parent}:{env.get('LD_LIBRARY_PATH', '')}"
            p = subprocess.run([str(b), "--list-devices"], capture_output = True, text = True, errors = "replace",
                               env = env, timeout = 300)
            text = p.stdout + p.stderr
            rec["list_devices"] = {"rc": p.returncode, "out": text[-1500:]}
            prefix = "Vulkan0" if backend == "vulkan" else "ROCm0"
            rec["gpu_listed"] = p.returncode == 0 and prefix in text
            rec["description"] = next((ln.split("\t", 1)[1].strip() for ln in text.splitlines()
                                       if ln.startswith(prefix + "\t")), None)
            h = subprocess.run([str(b), "--help"], capture_output = True, text = True, errors = "replace", env = env)
            rec["help_vae_tile"] = [x.strip() for x in (h.stdout + h.stderr).splitlines() if "vae-tile" in x][:4]
            rec["binary"] = str(b)
            if rec["gpu_listed"]:
                binaries[backend] = str(b)
        out["bundles"][backend] = rec
    out["binaries"] = binaries
    models, out["model_fetch"] = {}, {}
    for key, (repo, name) in FILES.items():
        dest = shared / "models" / repo.replace("/", "__") / name
        out["model_fetch"][key] = fetch(f"{HF}/{repo}/resolve/main/{name}", dest)
        models[key] = str(dest)
    out["models"] = models
    out["models_ok"] = all("error" not in v for v in out["model_fetch"].values())
    wjson(done, out)
    return out


CHOOSER = r'''
import inspect, json, os, sys, types
sys.path.insert(0, sys.argv[1])
klass, binary, desc, dit, vae, llm, fam = sys.argv[2:9]
import core.inference.sd_cpp_args as a
files = types.SimpleNamespace(diffusion_model = dit, vae = vae, llm = llm)
res = {"klass": klass, "family": fam, "has_whole_table": hasattr(a, "whole_vae_tiles")}
if res["has_whole_table"]:
    res["table"] = {str(k): v for k, v in a.whole_vae_tiles(fam)}
    res["gpu_margin"] = getattr(a, "GPU_WHOLE_VAE_MARGIN", None)
    res["cpu_margin"] = getattr(a, "CPU_WHOLE_VAE_MARGIN", None)
# Readings, recorded whichever path decides.
try:
    from core.inference.video_minimax_h3 import amdgpu_vram_reading, vulkan_device_is_amd
    from core.inference.sd_cpp_multigpu import CARD_RESERVE_MIB, file_mib
    from core.inference.diffusion_memory import _user_budget_reserve_mib
    free = total = limit = None
    if klass == "vulkan" and vulkan_device_is_amd(desc):
        fb, tb = amdgpu_vram_reading()
        if fb is not None:
            free, total = fb >> 20, (None if tb is None else tb >> 20)
            limit = int(free) - max(CARD_RESERVE_MIB, int(_user_budget_reserve_mib(total or free) or 0))
    res["card_free_mib"], res["card_total_mib"], res["card_limit_mib"] = free, total, limit
    res["vae_file_mib"] = file_mib(vae)
    res["tile_in_pixels"] = bool(a.sd_cpp_vae_tile_size_in_pixels(binary))
except Exception as e:
    res["reading_error"] = f"{type(e).__name__}: {e}"
try:
    from core.inference import sd_cpp_backend as b
    kw = {"family": fam} if "family" in inspect.signature(b._offload_vae_flags).parameters else {}
    res["flags"] = b._offload_vae_flags(a.OFFLOAD_MODEL, files, binary, klass, None, desc, **kw)
    res["chooser"] = "sd_cpp_backend._offload_vae_flags"
except Exception as e:
    res["backend_import_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    from core.inference.sd_cpp_multigpu import TEXT_ENCODER_FIELDS, file_mib
    def wkw(fn):
        return {"whole_tiles": a.whole_vae_tiles(fam)} if "whole_tiles" in inspect.signature(fn).parameters else {}
    limit = res.get("card_limit_mib")
    gpu_tile = None
    if klass in ("cuda", "vulkan") and a.sd_cpp_vae_tile_size_in_pixels(binary) and limit is not None:
        gpu_tile = a.gpu_vae_tile_size(limit, file_mib(vae), **wkw(a.gpu_vae_tile_size))
    if gpu_tile:
        te_gpu = a.gpu_te_fits(limit)
        res["flags"] = a.offload_flags(a.OFFLOAD_MODEL, vae_on_cpu = False, vae_tile_size = gpu_tile, te_on_cpu = not te_gpu)
    else:
        tile = None
        if a.sd_cpp_vae_tile_size_in_pixels(binary):
            from core.inference.diffusion_small_host import host_commit_mib, host_ram_mib, host_reserve_mib
            total, available = host_ram_mib()
            commit = host_commit_mib()
            if available is not None and commit is not None:
                available = min(available, commit)
            spare = None if available is None else available - sum(
                file_mib(getattr(files, f, None)) for f in ("diffusion_model", "vae", *TEXT_ENCODER_FIELDS)) - host_reserve_mib(total)
            res["host_spare_mib"] = spare
            tile = a.cpu_vae_tile_size(spare, **wkw(a.cpu_vae_tile_size))
        res["flags"] = a.offload_flags(a.OFFLOAD_MODEL, vae_tile_size = tile)
    res["chooser"] = "replica of _offload_vae_flags over the state's sd_cpp_args"
print("CHOOSER " + json.dumps(res))
'''


def choose(checkout: Path, klass: str, binary: str, desc, models: dict) -> dict:
    backend = checkout / "studio" / "backend"
    p = subprocess.run([sys.executable, "-c", CHOOSER, str(backend), klass, binary, desc or "", models["dit"],
                        models["vae"], models["llm"], FAMILY], capture_output = True, text = True, encoding = "utf-8",
                       errors = "replace", cwd = str(backend), env = clean_env(), timeout = 600)
    line = [x for x in p.stdout.splitlines() if x.startswith("CHOOSER ")]
    if not line:
        return {"error": (p.stdout + p.stderr)[-2000:]}
    res = json.loads(line[-1][8:])
    res["sd_cpp_args_sha256"] = sha256(backend / "core" / "inference" / "sd_cpp_args.py")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--shared", default = "")
    ap.add_argument("--tag", default = "master-982-0518b34-u27b9ddc")
    ap.add_argument("--sizes", default = "1536,2048")
    ap.add_argument("--steps", type = int, default = 4)
    ap.add_argument("--rocm", action = "store_true", help = "Linux: also the ROCm build Studio installs")
    ap.add_argument("--rocm-sizes", default = "2048")
    a = ap.parse_args()
    for k in TOKEN_VARS:
        os.environ.pop(k, None)
    shared = Path(a.shared or (Path(os.environ.get("AMD_CI_WORK") or a.out.parent) / "sdcpp_whole_shared")).resolve()
    sizes = [int(s) for s in a.sizes.split(",") if s]
    obs: dict = {"state": a.state, "platform": sys.platform, "sizes": sizes, "steps": a.steps, "arms": []}
    try:
        prep = prepare(shared, a.tag, a.rocm, a.checkout)
    except Exception as e:  # noqa: BLE001
        prep = {"error": f"{type(e).__name__}: {e}"}
    obs["prepared"] = prep
    bins = prep.get("binaries") or {}
    if not prep.get("models_ok") or "vulkan" not in bins:
        wjson(a.out, obs)
        return 0
    runner = Runner(shared, prep["models"], bins, a.steps)
    arms = obs["arms"]

    def arm(name: str, backend: str, size: int, flags: list, rep: int = 0, ref_flags = None) -> None:
        rec = runner.run(backend, size, flags, rep)
        entry = {"name": name, "backend": backend, "size": size, "flags": flags, "tile_passed": tile_of(flags),
                 "key": rec["key"]}
        ref = runner.run(backend, size, untiled_of(flags) if ref_flags is None else ref_flags)
        entry["ref_key"] = ref["key"]
        entry["psnr_vs_untiled"] = (psnr(runner.dir / rec["png"], runner.dir / ref["png"])
                                    if rec.get("png") and ref.get("png") else None)
        arms.append(entry)
        obs.setdefault("runs", {})[rec["key"]] = rec
        obs["runs"][ref["key"]] = ref

    # Each state's own choice, per backend class.
    obs["choice"] = {}
    for backend in [b for b in ("vulkan", "rocm") if b in bins]:
        desc = (prep["bundles"][backend] or {}).get("description")
        obs["choice"][backend] = choose(a.checkout, backend, bins[backend], desc, prep["models"])
    for size in sizes:
        ref = untiled_of(GPU_COMMON)
        rec = runner.run("vulkan", size, ref, 0)
        obs.setdefault("runs", {})[rec["key"]] = rec
        if size == sizes[0]:
            arm("vulkan_untiled_rep1 (A/A)", "vulkan", size, ref, rep = 1, ref_flags = ref)
        arm("forced_gpu_tile1024", "vulkan", size, GPU_COMMON + ["--vae-tiling", "--vae-tile-size", "1024"])
        arm("forced_gpu_whole2048", "vulkan", size, GPU_COMMON + ["--vae-tiling", "--vae-tile-size", "2048"])
        flags = (obs["choice"].get("vulkan") or {}).get("flags")
        if flags:
            arm(f"state_{a.state}_vulkan", "vulkan", size, list(flags))
        wjson(a.out, obs)
    if "rocm" in bins:
        for size in [int(s) for s in a.rocm_sizes.split(",") if s]:
            flags = (obs["choice"].get("rocm") or {}).get("flags")
            if flags:
                arm(f"state_{a.state}_rocm", "rocm", size, list(flags))
            wjson(a.out, obs)
    # Forced CPU whole decode once (the path a card without a VRAM reading takes on the head).
    arm("forced_cpu_whole2048", "vulkan", sizes[-1], GPU_COMMON + ["--vae-on-cpu", "--vae-tiling", "--vae-tile-size",
                                                                     "2048"])
    keep = a.out.parent / "whole_vae_runs"
    keep.mkdir(parents = True, exist_ok = True)
    for k in set(obs.get("runs", {})):
        for ext in ("log", "png"):
            src = runner.dir / f"{k}.{ext}"
            if src.is_file() and not (keep / src.name).is_file():
                (keep / src.name).write_bytes(src.read_bytes())
    wjson(a.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

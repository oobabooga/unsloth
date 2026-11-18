#!/usr/bin/env python3
"""Probe: sd.cpp Vulkan VAE direct conv on THIS card, global ``--conv-direct off`` vs ``off,vae=auto`` vs forced
``off,vae=on``, with the published Vulkan bundle Studio installs (Linux RADV or Windows AMD driver).

Observes only; criteria/sdcpp_vaecd.py judges.

Shared half (once into --shared, recorded in every observation):
  * the pinned unslothai/stable-diffusion.cpp Vulkan bundle for this OS (--tag), sha256 checked against the
    release's sd-prebuilt-sha256.json; `sd-cli --list-devices`;
  * public models (no token): Z-Image-Turbo Q4_K_M + Qwen3-4B BF16 + FLUX.1 AE;
  * the matrix: every arm per case, interleaved per repeat, one fresh sd-cli per run. Per run: rc, TE / sampling /
    VAE decode / total seconds from sd-cli's own log, wall, peak RSS (wait4 on Linux, PeakWorkingSetSize on
    Windows), sha256 of the PNG, PSNR against the first cd_off render of the case, and the `conv2d direct auto` /
    `conv3d direct auto` decisions the run printed (engagement). Repeats of one arm are the A/A noise.

Per state: the state's own Studio flags for this bundle (core.inference.sd_cpp_args.native_speed_flags with the
bundle's listed accelerator and the binary, imported from the checkout on the Studio python), then that argv run
--state-reps times on the first case.
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
    "z_dit": ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf"),
    "qwen3": ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-BF16.gguf"),
    "ae": ("unsloth/FLUX.1-schnell", "ae.safetensors"),
}
PROMPT = "a photo of a red fox sitting in fresh snow, golden hour, detailed fur"
TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN")


def case_argv(m: dict) -> dict:
    z = ["--diffusion-model", m["z_dit"], "--vae", m["ae"], "--llm", m["qwen3"], "--steps", "8", "--cfg-scale", "1.0"]
    return {"zimage1024": z + ["-W", "1024", "-H", "1024"], "zimage512": z + ["-W", "512", "-H", "512"]}


# cd_off = what Studio main sends on Vulkan with the pin; vae_auto = this change; vae_on = forced VAE direct conv.
ARMS = {
    "cd_off": ["--diffusion-fa", "--conv-direct", "off"],
    "vae_auto": ["--diffusion-fa", "--conv-direct", "off,vae=auto"],
    "vae_on": ["--diffusion-fa", "--conv-direct", "off,vae=on"],
}
RX = {
    "te_s": re.compile(r"get_learned_condition completed, taking ([0-9.]+)(m?s)"),
    "sampling_s": re.compile(r"sampling completed, taking ([0-9.]+)(m?s)"),
    "decode_s": re.compile(r"decode_first_stage completed, taking ([0-9.]+)(m?s)"),
    "generate_s": re.compile(r"generate_image completed in ([0-9.]+)(m?s)"),
}
DECISION = re.compile(r"(conv[23]d direct auto: \S+ -> \w+[^\n]*|flash attention auto: \S+ -> \w+[^\n]*)")


def say(msg: str) -> None:
    print(f"[sdcpp_vaecd {time.strftime('%H:%M:%S')}] {msg}", flush = True)


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
    rc = subprocess.run([curl, "-sS", "--fail", "-L", "--retry", "5", "--retry-delay", "5", "-o", str(dest) + ".part",
                         url], env = clean_env()).returncode
    if rc != 0:
        return {"url": url, "error": f"curl rc={rc}"}
    os.replace(str(dest) + ".part", dest)
    return {"url": url, "bytes": dest.stat().st_size, "seconds": round(time.time() - t0, 1)}


def _peak_rss_windows(handle) -> float | None:
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


def run_one(binary: Path, argv: list, flags: list, out_png: Path, log_path: Path) -> dict:
    cmd = [str(binary), *argv, "--prompt", PROMPT, "--seed", "42", "--output", str(out_png), "-v", *flags]
    t0 = time.time()
    peak = None
    with open(log_path, "w", encoding = "utf-8", errors = "replace") as fh:
        p = subprocess.Popen(cmd, stdout = fh, stderr = subprocess.STDOUT, env = clean_env())
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
    rec = {"rc": rc, "wall_s": round(time.time() - t0, 3), "peak_rss_gb": peak, "flags": flags}
    text = log_path.read_text(encoding = "utf-8", errors = "replace")
    for key, rx in RX.items():
        vals = [float(v) / (1000.0 if u == "ms" else 1.0) for v, u in rx.findall(text)]
        rec[key] = vals[-1] if vals else None
    rec["decisions"] = sorted(set(m.strip()[:200] for m in DECISION.findall(text)))
    rec["sha256"] = sha256(out_png) if out_png.is_file() else None
    rec["tail"] = text[-800:] if rc != 0 else None
    return rec


def psnr(a: Path, b: Path) -> float | None:
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


def shared_half(shared: Path, tag: str, reps: int, cases: list, arms: list) -> dict:
    done = shared / "shared.json"
    if done.is_file():
        return json.loads(done.read_text(encoding = "utf-8"))
    out: dict = {"tag": tag, "platform": sys.platform, "arms": {k: ARMS[k] for k in arms}}
    base = f"https://github.com/{REPO}/releases/download/{tag}"
    asset = f"sd-{tag}-bin-win-vulkan-x64.zip" if WIN else f"sd-{tag}-bin-Linux-Ubuntu-22.04-x86_64-vulkan.zip"
    out["asset"] = asset
    out["bundle_fetch"] = fetch(f"{base}/{asset}", shared / asset)
    out["sums_fetch"] = fetch(f"{base}/sd-prebuilt-sha256.json", shared / "sd-prebuilt-sha256.json")
    try:
        want = json.loads((shared / "sd-prebuilt-sha256.json").read_text(encoding = "utf-8")).get(asset)
    except Exception as e:  # noqa: BLE001
        want = None
        out["sums_error"] = str(e)
    got = sha256(shared / asset) if (shared / asset).is_file() else None
    out["bundle_sha256"] = {"want": want, "got": got, "match": bool(want and want == got)}
    bdir = shared / "bundle"
    if got:
        with zipfile.ZipFile(shared / asset) as z:
            z.extractall(bdir)
    clis = sorted(bdir.rglob("sd-cli.exe" if WIN else "sd-cli"))
    binary = clis[0] if clis else None
    out["binary"] = str(binary) if binary else None
    if binary:
        if not WIN:
            binary.chmod(0o755)
        p = subprocess.run([str(binary), "--list-devices"], capture_output = True, text = True, errors = "replace")
        out["list_devices"] = {"rc": p.returncode, "out": (p.stdout + p.stderr)[-2000:]}
        out["gpu_listed"] = p.returncode == 0 and "Vulkan0" in (p.stdout + p.stderr)
        h = subprocess.run([str(binary), "--help"], capture_output = True, text = True, errors = "replace")
        out["help_conv_direct"] = [x.strip() for x in (h.stdout + h.stderr).splitlines() if "conv-direct" in x][:6]
    models: dict = {}
    out["model_fetch"] = {}
    for key, (repo, name) in FILES.items():
        dest = shared / "models" / repo.replace("/", "__") / name
        out["model_fetch"][key] = fetch(f"{HF}/{repo}/resolve/main/{name}", dest)
        models[key] = str(dest)
    out["models"] = models
    out["matrix"] = {}
    if binary and out.get("gpu_listed") and all("error" not in v for v in out["model_fetch"].values()):
        argvs = case_argv(models)
        runs = shared / "runs"
        runs.mkdir(parents = True, exist_ok = True)
        for rep in range(reps):
            for case in cases:
                for arm in arms:
                    tagname = f"{case}__{arm}__r{rep}"
                    say(f"run {tagname}")
                    rec = run_one(binary, argvs[case], ARMS[arm], runs / f"{tagname}.png", runs / f"{tagname}.log")
                    rec["rep"] = rep
                    rec["psnr_vs_cd_off_r0"] = psnr(runs / f"{tagname}.png", runs / f"{case}__cd_off__r0.png")
                    out["matrix"].setdefault(case, {}).setdefault(arm, []).append(rec)
                    say(f"  rc={rec['rc']} te={rec['te_s']} sampling={rec['sampling_s']} decode={rec['decode_s']} "
                        f"generate={rec['generate_s']} rss={rec['peak_rss_gb']} psnr={rec['psnr_vs_cd_off_r0']} "
                        f"{rec['decisions']}")
                    wjson(shared / "partial.json", out)
    wjson(done, out)
    return out


def state_flags(checkout: Path, binary: str) -> dict:
    code = (
        "import json, sys; sys.path.insert(0, sys.argv[1])\n"
        "import core.inference.sd_cpp_args as a\n"
        "from core.inference import sd_cpp_backend as b\n"
        "acc = b.sd_cpp_listed_device(sys.argv[2], probe = True)[0]\n"
        "print('FLAGS ' + json.dumps({'accelerator': acc, 'flags': a.native_speed_flags('default', [], acc, sys.argv[2])}))\n"
    )
    backend = checkout / "studio" / "backend"
    p = subprocess.run([sys.executable, "-c", code, str(backend), binary], capture_output = True, text = True,
                       encoding = "utf-8", errors = "replace", cwd = str(backend), env = clean_env())
    line = [x for x in p.stdout.splitlines() if x.startswith("FLAGS ")]
    if p.returncode != 0 or not line:
        return {"error": (p.stdout + p.stderr)[-1500:]}
    res = json.loads(line[-1][6:])
    res["sd_cpp_args_sha256"] = sha256(backend / "core" / "inference" / "sd_cpp_args.py")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--shared", default = "", help = "default: $AMD_CI_WORK/sdcpp_shared")
    ap.add_argument("--tag", default = "master-982-0518b34-u27b9ddc")
    ap.add_argument("--reps", type = int, default = 3)
    ap.add_argument("--state-reps", type = int, default = 2)
    ap.add_argument("--cases", default = "zimage1024,zimage512")
    ap.add_argument("--arms", default = ",".join(ARMS))
    a = ap.parse_args()
    for k in TOKEN_VARS:
        os.environ.pop(k, None)
    shared = Path(a.shared or (Path(os.environ.get("AMD_CI_WORK") or a.out.parent) / "sdcpp_shared")).resolve()
    cases = a.cases.split(",")
    obs: dict = {"state": a.state, "platform": sys.platform}
    try:
        sh = shared_half(shared, a.tag, a.reps, cases, a.arms.split(","))
    except Exception as e:  # noqa: BLE001
        sh = {"error": f"{type(e).__name__}: {e}"}
    obs["shared"] = sh
    keep = a.out.parent / "vk_runs"
    if (shared / "runs").is_dir() and not keep.is_dir():
        import shutil
        shutil.copytree(shared / "runs", keep)
    st = state_flags(a.checkout, sh["binary"]) if sh.get("binary") else {"error": "no binary"}
    obs["studio_vulkan"] = st
    if sh.get("binary") and sh.get("gpu_listed") and "flags" in st:
        argv = case_argv(sh["models"])[cases[0]]
        sdir = shared / f"state_{a.state}"
        sdir.mkdir(parents = True, exist_ok = True)
        st["case"] = cases[0]
        st["runs"] = []
        for rep in range(a.state_reps):
            rec = run_one(Path(sh["binary"]), argv, st["flags"], sdir / f"r{rep}.png", sdir / f"r{rep}.log")
            rec["psnr_vs_cd_off_r0"] = psnr(sdir / f"r{rep}.png", shared / "runs" / f"{cases[0]}__cd_off__r0.png")
            st["runs"].append(rec)
            keep.mkdir(parents = True, exist_ok = True)
            for ext in ("png", "log"):
                src = sdir / f"r{rep}.{ext}"
                if src.is_file():
                    (keep / f"state_{a.state}_r{rep}.{ext}").write_bytes(src.read_bytes())
            say(f"state {a.state} r{rep}: rc={rec['rc']} generate={rec['generate_s']} decode={rec['decode_s']} "
                f"{rec['decisions']}")
    wjson(a.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

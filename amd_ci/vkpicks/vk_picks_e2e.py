#!/usr/bin/env python3
"""Probe (observes only): ggml-patches/0070 Vulkan AMD picks, BASE vs HEAD sd.cpp builds on THIS card.

  python vk_picks_e2e.py --base DIR --head DIR --models DIR --out DIR [--reps 2]

DIR = a build's bin directory holding sd-cli(.exe) and test-backend-ops(.exe).
Records, into OUT/obs.json:
  * the Vulkan device: driverID name, driverVersion (raw + AMD major.build decode), via the loader (ctypes), and
    ggml's own device line from sd-cli -v (coopmat / int dot / warp size);
  * test-backend-ops test -o MUL_MAT -b Vulkan0 per arm: pass / fail counts and the FAIL lines;
  * test-backend-ops perf -o MUL_MAT -b Vulkan0 per arm, arms interleaved B H H B per round: us/run per case;
  * sd-cli Z-Image Turbo Q4_K_M (Studio's Vulkan argv: --diffusion-fa), seed 42, 1024^2 and 512^2, 8 steps, arms
    interleaved B H H B per round: sampling s, s/step, TE / decode / generate s, wall, png sha256; then per case the
    pixel diff HEAD vs BASE (max abs, mean abs, PSNR) and BASE vs BASE.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

HF = "https://huggingface.co"
FILES = {
    "z_dit": ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf"),
    "qwen3": ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-BF16.gguf"),
    "ae": ("unsloth/FLUX.1-schnell", "ae.safetensors"),
}
PROMPT = "a photo of a red fox sitting in fresh snow, golden hour, detailed fur"
STEPS = 8
RX = {
    "te_s": re.compile(r"get_learned_condition completed, taking ([0-9.]+)(m?s)"),
    "sampling_s": re.compile(r"sampling completed, taking ([0-9.]+)(m?s)"),
    "decode_s": re.compile(r"decode_first_stage completed, taking ([0-9.]+)(m?s)"),
    "generate_s": re.compile(r"generate_image completed in ([0-9.]+)(m?s)"),
}
EXE = ".exe" if os.name == "nt" else ""


def wjson(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent = 1, default = str), encoding = "utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> dict:
    if dest.is_file() and dest.stat().st_size > 0:
        return {"url": url, "cached": True, "bytes": dest.stat().st_size}
    env = {k: v for k, v in os.environ.items() if "TOKEN" not in k}
    t0 = time.time()
    rc = subprocess.run(["curl" + EXE, "-sS", "--fail", "-L", "--retry", "5", "--retry-delay", "5", "-o",
                         str(dest) + ".part", url], env = env).returncode
    if rc != 0:
        return {"url": url, "error": f"curl rc={rc}"}
    os.replace(str(dest) + ".part", dest)
    return {"url": url, "bytes": dest.stat().st_size, "seconds": round(time.time() - t0, 1)}


class VkAppInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("pApplicationName", ctypes.c_char_p),
                ("applicationVersion", ctypes.c_uint32), ("pEngineName", ctypes.c_char_p),
                ("engineVersion", ctypes.c_uint32), ("apiVersion", ctypes.c_uint32)]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("pApplicationInfo", ctypes.POINTER(VkAppInfo)), ("enabledLayerCount", ctypes.c_uint32),
                ("ppEnabledLayerNames", ctypes.c_void_p), ("enabledExtensionCount", ctypes.c_uint32),
                ("ppEnabledExtensionNames", ctypes.c_void_p)]


class VkDriverProps(ctypes.Structure):  # VkPhysicalDeviceDriverProperties (Vulkan 1.2 core)
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("driverID", ctypes.c_int),
                ("driverName", ctypes.c_char * 256), ("driverInfo", ctypes.c_char * 256),
                ("conformanceVersion", ctypes.c_uint8 * 4)]


class VkProps2(ctypes.Structure):  # VkPhysicalDeviceProperties2, properties as an opaque blob read by offset
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("properties", ctypes.c_uint8 * 1024)]


def vulkan_devices() -> dict:
    """driverVersion / driverID straight from the loader; no vulkaninfo needed (absent on the Windows boxes)."""
    try:
        lib = ctypes.WinDLL("vulkan-1.dll") if os.name == "nt" else ctypes.CDLL("libvulkan.so.1")
        app = VkAppInfo(0, None, b"vkpicks", 1, b"vkpicks", 1, (1 << 22) | (2 << 12))
        ci = VkInstanceCreateInfo(1, None, 0, ctypes.pointer(app), 0, None, 0, None)
        inst = ctypes.c_void_p()
        rc = lib.vkCreateInstance(ctypes.byref(ci), None, ctypes.byref(inst))
        if rc != 0:
            return {"error": f"vkCreateInstance {rc}"}
        n = ctypes.c_uint32(0)
        lib.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), None)
        devs = (ctypes.c_void_p * n.value)()
        lib.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), devs)
        out = []
        for d in devs:
            drv = VkDriverProps(1000196000, None)
            p2 = VkProps2(1000059001, ctypes.cast(ctypes.pointer(drv), ctypes.c_void_p))
            lib.vkGetPhysicalDeviceProperties2(ctypes.c_void_p(d), ctypes.byref(p2))
            raw = bytes(p2.properties)
            api, ver, vendor, device, dtype = (int.from_bytes(raw[i:i + 4], "little") for i in range(0, 20, 4))
            name = raw[20:276].split(b"\0", 1)[0].decode(errors = "replace")
            out.append({"name": name, "vendor_id": hex(vendor), "device_id": hex(device), "device_type": dtype,
                        "api_version": f"{api >> 22}.{(api >> 12) & 0x3ff}.{api & 0xfff}",
                        "driver_version_raw": ver,
                        "amd_driver_major_build": f"{ver >> 22}.{ver & 0x3fffff}",
                        "driver_id": drv.driverID,
                        "driver_name": drv.driverName.decode(errors = "replace"),
                        "driver_info": drv.driverInfo.decode(errors = "replace"),
                        # #31587 gate: proprietary (driverID 1) and (major > 2 or major == 2 and build >= 416)
                        "large_tile_gate_open": (drv.driverID != 1) or (ver >> 22) > 2
                        or ((ver >> 22) == 2 and (ver & 0x3fffff) >= 416)})
        return {"devices": out}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def windows_driver() -> dict:
    if os.name != "nt":
        return {}
    p = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate |"
                        " ConvertTo-Json"], capture_output = True, text = True, errors = "replace")
    try:
        return {"win32_video": json.loads(p.stdout)}
    except Exception:  # noqa: BLE001
        return {"win32_video_raw": p.stdout[-800:]}


def run_tbo(binary: Path, mode: str, log: Path) -> dict:
    t0 = time.time()
    with open(log, "w", encoding = "utf-8", errors = "replace") as fh:
        rc = subprocess.run([str(binary), mode, "-o", "MUL_MAT", "-b", "Vulkan0"], stdout = fh,
                            stderr = subprocess.STDOUT, timeout = 5400).returncode
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(encoding = "utf-8", errors = "replace"))
    rec = {"rc": rc, "wall_s": round(time.time() - t0, 1)}
    if mode == "test":
        rec["ok"] = len(re.findall(r"^\s*MUL_MAT\(.*\):\s+OK", text, re.M))
        fails = [ln.strip() for ln in text.splitlines() if "MUL_MAT(" in ln and "FAIL" in ln]
        rec["fail"] = len(fails)
        rec["fail_lines"] = fails[:60]
        m = re.search(r"(\d+)/(\d+) tests passed", text)
        rec["passed_line"] = m.group(0) if m else None
    else:
        perf = {}
        for ln in text.splitlines():
            m = re.match(r"\s*(MUL_MAT\(.*\)):\s+\d+\s+runs\s+-\s+([0-9.]+)\s+us/run", ln)
            if m:
                perf[m.group(1)] = float(m.group(2))
        rec["us_per_run"] = perf
    return rec


def run_sd(binary: Path, models: dict, size: int, out_png: Path, log: Path) -> dict:
    cmd = [str(binary), "--diffusion-model", models["z_dit"], "--vae", models["ae"], "--llm", models["qwen3"],
           "--steps", str(STEPS), "--cfg-scale", "1.0", "-W", str(size), "-H", str(size), "--diffusion-fa",
           "--prompt", PROMPT, "--seed", "42", "--output", str(out_png), "-v"]
    t0 = time.time()
    with open(log, "w", encoding = "utf-8", errors = "replace") as fh:
        rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, timeout = 1800).returncode
    text = log.read_text(encoding = "utf-8", errors = "replace")
    rec = {"rc": rc, "wall_s": round(time.time() - t0, 2)}
    for key, rx in RX.items():
        vals = [float(v) / (1000.0 if u == "ms" else 1.0) for v, u in rx.findall(text)]
        rec[key] = vals[-1] if vals else None
    rec["s_per_step"] = round(rec["sampling_s"] / STEPS, 4) if rec.get("sampling_s") else None
    rec["ggml_vk_lines"] = sorted(set(ln.strip()[:300] for ln in text.splitlines() if "ggml_vulkan:" in ln))[:6]
    rec["sha256"] = sha256(out_png) if out_png.is_file() else None
    rec["tail"] = text[-1200:] if rc != 0 else None
    return rec


def pixel_diff(a: Path, b: Path) -> dict:
    try:
        import numpy as np
        from PIL import Image
        x = np.asarray(Image.open(a).convert("RGB"), dtype = np.float64)
        y = np.asarray(Image.open(b).convert("RGB"), dtype = np.float64)
        d = np.abs(x - y)
        mse = float((d ** 2).mean())
        return {"max_abs": float(d.max()), "mean_abs": round(float(d.mean()), 5),
                "frac_px_diff": round(float((d.max(axis = 2) > 0).mean()), 6),
                "psnr_db": 999.0 if mse == 0 else round(10 * np.log10(255.0 ** 2 / mse), 2)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required = True)
    ap.add_argument("--head", required = True)
    ap.add_argument("--models", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--reps", type = int, default = 2)
    ap.add_argument("--skip-tbo", action = "store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents = True, exist_ok = True)
    bins = {"base": Path(a.base), "head": Path(a.head)}
    obs: dict = {"host": {"platform": platform.platform(), "python": sys.version.split()[0]},
                 "vulkan": vulkan_devices(), **windows_driver(), "errors": {}}
    jp = out / "obs.json"
    wjson(jp, obs)
    md = Path(a.models)
    md.mkdir(parents = True, exist_ok = True)
    models, fetched = {}, {}
    for k, (repo, f) in FILES.items():
        dest = md / Path(f).name
        fetched[k] = fetch(f"{HF}/{repo}/resolve/main/{f}", dest)
        models[k] = str(dest)
    obs["fetch"] = fetched
    wjson(jp, obs)
    order = ["base", "head", "head", "base"]
    if not a.skip_tbo:
        obs["tbo_test"] = {arm: run_tbo(bins[arm] / f"test-backend-ops{EXE}", "test", out / f"tbo_test_{arm}.txt")
                           for arm in ("base", "head")}
        wjson(jp, obs)
        perf = {"base": [], "head": []}
        for r in range(a.reps):
            for i, arm in enumerate(order):
                perf[arm].append(run_tbo(bins[arm] / f"test-backend-ops{EXE}", "perf",
                                         out / f"tbo_perf_{arm}_r{r}_{i}.txt"))
                wjson(jp, {**obs, "tbo_perf_partial": perf})
        rows = {}
        for case in sorted(set().union(*[p["us_per_run"] for p in perf["base"] + perf["head"]])):
            b = [p["us_per_run"][case] for p in perf["base"] if case in p["us_per_run"]]
            h = [p["us_per_run"][case] for p in perf["head"] if case in p["us_per_run"]]
            if b and h:
                rows[case] = {"base_med": statistics.median(b), "head_med": statistics.median(h),
                              "base_spread": (max(b) - min(b)) / statistics.median(b),
                              "ratio": round(statistics.median(h) / statistics.median(b), 4)}
        obs["tbo_perf"] = rows
        wjson(jp, obs)
    sd = {}
    for size in (512, 1024):
        runs = {"base": [], "head": []}
        for r in range(a.reps):
            for i, arm in enumerate(order):
                png = out / f"sd_{size}_{arm}_r{r}_{i}.png"
                runs[arm].append({**run_sd(bins[arm] / f"sd-cli{EXE}", models, size, png, out / f"sd_{size}_{arm}_r{r}_{i}.log"),
                                  "png": png.name})
                sd[str(size)] = {"runs": runs}
                obs["sd"] = sd
                wjson(jp, obs)
        summ = {}
        for arm in ("base", "head"):
            ok = [x for x in runs[arm] if x["rc"] == 0 and x.get("s_per_step")]
            if ok:
                summ[arm] = {"s_per_step_med": statistics.median(x["s_per_step"] for x in ok),
                             "s_per_step_all": [x["s_per_step"] for x in ok],
                             "generate_med": statistics.median(x["generate_s"] or 0 for x in ok),
                             "shas": sorted(set(x["sha256"] for x in ok))}
        b0 = next((out / x["png"] for x in runs["base"] if x["rc"] == 0), None)
        b1 = next((out / x["png"] for x in runs["base"][1:] if x["rc"] == 0), None)
        h0 = next((out / x["png"] for x in runs["head"] if x["rc"] == 0), None)
        if b0 and h0:
            summ["head_vs_base"] = pixel_diff(b0, h0)
        if b0 and b1:
            summ["base_vs_base"] = pixel_diff(b0, b1)
        sd[str(size)]["summary"] = summ
        obs["sd"] = sd
        wjson(jp, obs)
    obs["status"] = "DONE"
    wjson(jp, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

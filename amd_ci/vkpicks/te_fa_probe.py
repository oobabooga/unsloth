#!/usr/bin/env python3
"""Probe (observes only): Studio PR #18787's sd.cpp flag change on THIS card, pinned published Vulkan sd-cli.

  python te_fa_probe.py --work DIR --out DIR [--rounds 2]

Same binary both arms (Studio's pin master-982-0518b34-u27b9ddc, sha256 checked), only the flag differs:
  main = --diffusion-fa                                  (text-encoder flash attention OFF)
  pr   = --flash-attn off,diffusion=on,te=on             (#18787)
Z-Image Turbo Q4_K_M + Qwen3-4B Q8_0 + FLUX.1 AE, a >4k-token prompt, 1 step, 1024^2, seed 42, arms main pr pr main
per round. Per run: TE encode s, sampling s, decode s, generate s, wall, peak process memory (RSS / peak working
set), peak amdgpu VRAM+GTT used (Linux sysfs), the log's token count / buffer-size / fallback lines, png sha256.
Then pixel diff pr vs main beside main vs main.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vk_picks_e2e import RX, fetch, pixel_diff, sha256, vulkan_devices, windows_driver, wjson  # noqa: E402

TAG = "master-982-0518b34-u27b9ddc"
ASSETS = {
    "nt": (f"sd-{TAG}-bin-win-vulkan-x64.zip", "c9cf9e08c832d20417a4a5189d5294731ed4f1919e7e0f4aee3ec233cff035d5"),
    "posix": (f"sd-{TAG}-bin-Linux-Ubuntu-22.04-x86_64-vulkan.zip",
              "8d5342d4cb32f849dc1a377ebf690fcaec10432681469db9632b76d09f2066b3"),
}
HF = "https://huggingface.co"
FILES = {
    "z_dit": ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf"),
    "qwen3": ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf"),
    "ae": ("unsloth/FLUX.1-schnell", "ae.safetensors"),
}
ARMS = {"main": ["--diffusion-fa"], "pr": ["--flash-attn", "off,diffusion=on,te=on"]}
EXE = ".exe" if os.name == "nt" else ""

SUBJECTS = ["a lighthouse", "a red fox", "an old library", "a mountain village", "a koi pond", "a steam train",
            "a desert canyon", "a night market", "a glass greenhouse", "a harbour at dawn", "a snowy forest",
            "a jazz club", "a windmill", "a coral reef", "a cathedral nave", "a rice terrace"]
DETAILS = ["soft golden light falling across weathered surfaces", "fine film grain and gentle chromatic aberration",
           "a shallow depth of field with creamy bokeh", "intricate textures in wood, stone and fabric",
           "a muted teal and amber colour palette", "long shadows and a hazy atmosphere",
           "reflections shimmering on wet ground", "a wide angle composition with strong leading lines",
           "tiny figures that give a sense of scale", "subtle mist drifting between layers of depth"]


def long_prompt(target_words: int = 4900) -> str:
    """Deterministic, varied English (Qwen3 BPE runs ~1.3 tokens per word here, so ~6.1k tokens: 32 heads x 6.1k^2 x 4 B = 4.4 GiB of scores, past the 4 GiB line of leejet #1986)."""
    out, i = [], 0
    while sum(len(s.split()) for s in out) < target_words:
        s = SUBJECTS[i % len(SUBJECTS)]
        d1, d2 = DETAILS[i % len(DETAILS)], DETAILS[(i * 7 + 3) % len(DETAILS)]
        out.append(f"Panel {i + 1} shows {s} rendered with {d1}, and in the background {d2}; the scene number {i + 1} "
                   f"is described carefully so that every object, colour and material is unambiguous.")
        i += 1
    return " ".join(out)


class MemPeak:
    """Peak process memory (psutil rss / Windows peak_wset) and Linux amdgpu VRAM+GTT used, sampled every 0.2 s."""

    def __init__(self, pid: int):
        self.pid, self.stop, self.rss, self.gpu, self.gpu0 = pid, False, 0, 0, gpu_used()
        self.t = threading.Thread(target = self.loop, daemon = True)
        self.t.start()

    def loop(self) -> None:
        try:
            import psutil
            p = psutil.Process(self.pid)
        except Exception:  # noqa: BLE001
            p = None
        while not self.stop:
            try:
                if p is not None:
                    mi = p.memory_info()
                    self.rss = max(self.rss, getattr(mi, "peak_wset", 0) or mi.rss)
            except Exception:  # noqa: BLE001
                pass
            g = gpu_used()
            if g is not None:
                self.gpu = max(self.gpu, g)
            time.sleep(0.2)

    def done(self) -> dict:
        self.stop = True
        self.t.join(timeout = 2)
        return {"peak_proc_gib": round(self.rss / 2 ** 30, 3) if self.rss else None,
                "peak_gpu_used_gib": round(self.gpu / 2 ** 30, 3) if self.gpu else None,
                "gpu_used_before_gib": round(self.gpu0 / 2 ** 30, 3) if self.gpu0 else None}


def gpu_used():
    if os.name == "nt":
        return None
    tot = None
    for d in Path("/sys/class/drm").glob("card*/device"):
        try:
            v = int((d / "mem_info_vram_used").read_text()) + int((d / "mem_info_gtt_used").read_text())
            tot = (tot or 0) + v
        except Exception:  # noqa: BLE001
            continue
    return tot


def run_one(cli: Path, models: dict, prompt_file: Path, flags: list, png: Path, log: Path) -> dict:
    cmd = [str(cli), "--diffusion-model", models["z_dit"], "--vae", models["ae"], "--llm", models["qwen3"],
           "--steps", "1", "--cfg-scale", "1.0", "-W", "1024", "-H", "1024", *flags, "--prompt-file", str(prompt_file),
           "--seed", "42", "--output", str(png), "-v"]
    t0 = time.time()
    with open(log, "w", encoding = "utf-8", errors = "replace") as fh:
        p = subprocess.Popen(cmd, stdout = fh, stderr = subprocess.STDOUT, cwd = str(cli.parent))
        mp = MemPeak(p.pid)
        try:
            rc = p.wait(timeout = 3600)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = "timeout"
        mem = mp.done()
    text = log.read_text(encoding = "utf-8", errors = "replace")
    rec = {"rc": rc, "wall_s": round(time.time() - t0, 2), **mem, "flags": flags}
    for key, rx in RX.items():
        vals = [float(v) / (1000.0 if u == "ms" else 1.0) for v, u in rx.findall(text)]
        rec[key] = vals[-1] if vals else None
    keep = re.compile(r"token|buffer size|fallback|falling back|flash|out of memory|ErrorOutOfDeviceMemory|CPU",
                      re.I)
    rec["log_lines"] = [ln.strip()[:240] for ln in text.splitlines() if keep.search(ln)][:40]
    rec["sha256"] = sha256(png) if png.is_file() else None
    rec["tail"] = text[-1500:] if rc != 0 else None
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--rounds", type = int, default = 2)
    a = ap.parse_args()
    work, out = Path(a.work), Path(a.out)
    work.mkdir(parents = True, exist_ok = True)
    out.mkdir(parents = True, exist_ok = True)
    jp = out / "obs.json"
    obs: dict = {"host": {"platform": platform.platform()}, "tag": TAG, "vulkan": vulkan_devices(),
                 **windows_driver()}
    name, want = ASSETS["nt" if os.name == "nt" else "posix"]
    z = work / name
    obs["bundle_fetch"] = fetch(f"https://github.com/unslothai/stable-diffusion.cpp/releases/download/{TAG}/{name}", z)
    got = sha256(z) if z.is_file() else None
    obs["bundle_sha256_ok"] = got == want
    wjson(jp, obs)
    if got != want:
        obs["error"] = f"bundle sha256 {got} != {want}"
        wjson(jp, obs)
        return 0
    bd = work / "bundle"
    with zipfile.ZipFile(z) as zf:
        zf.extractall(bd)
    cli = next(bd.rglob(f"sd-cli{EXE}"))
    if os.name != "nt":
        for f in cli.parent.iterdir():
            f.chmod(0o755)
    models = {}
    for k, (repo, f) in FILES.items():
        dest = work / Path(f).name
        obs.setdefault("fetch", {})[k] = fetch(f"{HF}/{repo}/resolve/main/{f}", dest)
        models[k] = str(dest)
    pf = out / "prompt.txt"
    pf.write_text(long_prompt(), encoding = "utf-8")
    obs["prompt_words"] = len(pf.read_text(encoding = "utf-8").split())
    wjson(jp, obs)
    runs = {"main": [], "pr": []}
    for r in range(a.rounds):
        for i, arm in enumerate(["main", "pr", "pr", "main"]):
            png = out / f"{arm}_r{r}_{i}.png"
            runs[arm].append({**run_one(cli, models, pf, ARMS[arm], png, out / f"{arm}_r{r}_{i}.log"),
                              "png": png.name})
            obs["runs"] = runs
            wjson(jp, obs)
    summ = {}
    for arm in runs:
        ok = [x for x in runs[arm] if x["rc"] == 0]
        if ok:
            med = lambda k: statistics.median(x[k] for x in ok if x.get(k) is not None) \
                if any(x.get(k) is not None for x in ok) else None  # noqa: E731
            summ[arm] = {k: med(k) for k in ("te_s", "sampling_s", "decode_s", "generate_s", "wall_s",
                                             "peak_proc_gib", "peak_gpu_used_gib")}
            summ[arm]["te_all"] = [x.get("te_s") for x in ok]
            summ[arm]["shas"] = sorted(set(x["sha256"] for x in ok))
        summ[arm + "_failed"] = len(runs[arm]) - len(ok)
    m = [out / x["png"] for x in runs["main"] if x["rc"] == 0]
    p = [out / x["png"] for x in runs["pr"] if x["rc"] == 0]
    if m and p:
        summ["pr_vs_main"] = pixel_diff(m[0], p[0])
    if len(m) > 1:
        summ["main_vs_main"] = pixel_diff(m[0], m[1])
    obs["summary"] = summ
    obs["status"] = "DONE"
    wjson(jp, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

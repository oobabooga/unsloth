#!/usr/bin/env python3
"""Probe (observes only) for Studio PR #18787 on THIS card: sd-cli text-encode time per flash-attention flag.

  python te_fa_matrix_probe.py --work DIR --out DIR [--rounds 3]

Pinned published sd-cli master-982-0518b34-u27b9ddc (sha256 checked), the same binary for every arm:
  a = --diffusion-fa                              (main: TE flash attention off)
  b = --flash-attn off,diffusion=on,te=on         (#18787)
  c = --flash-attn off,diffusion=on,te=auto       (the binary decides per backend)
FLUX.2 klein 4B Q4_0 + Qwen3-4B Q8_0 + FLUX.2 VAE, 512^2, 1 step, seed 42. Prompts: short (~100 tokens), long (~4.1k)
and xlong (~6.2k, 32 heads x 6.2k^2 x 4 B = 4.4 GiB of f32 scores: past leejet #1986's 4 GiB point). Rounds rotate
the arm order (a b c / b c a / c a b) per prompt. Per run: TE s, generate s, wall, peak process / amdgpu memory, the
log's token count, compute-buffer, fallback and flash-attention lines (the CPU-fallback evidence), png sha256.
Summary per prompt x arm: median + spread of TE s, and whether any run logged a CPU fallback.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import statistics
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from te_fa_probe import ASSETS, DETAILS, SUBJECTS, TAG, MemPeak  # noqa: E402,F401
from vk_picks_e2e import RX, fetch, pixel_diff, sha256, vulkan_devices, windows_driver, wjson  # noqa: E402

import subprocess  # noqa: E402
import time  # noqa: E402

HF = "https://huggingface.co"
FILES = {
    "k_dit": ("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_0.gguf"),
    "qwen3": ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf"),
    "f2vae": ("black-forest-labs/FLUX.2-klein-4B", "vae/diffusion_pytorch_model.safetensors"),
}
ARMS = {"a": ["--diffusion-fa"],
        "b": ["--flash-attn", "off,diffusion=on,te=on"],
        "c": ["--flash-attn", "off,diffusion=on,te=auto"]}
ROT = [["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"]]
EXE = ".exe" if os.name == "nt" else ""
FALLBACK = re.compile(r"fall(ing|s)? ?back|fallback|running on cpu|use cpu|offload.*cpu|out of (device )?memory|"
                      r"ErrorOutOfDeviceMemory|failed to allocate", re.I)


def prompt_of(words: int) -> str:
    out, i = [], 0
    while sum(len(s.split()) for s in out) < words:
        s = SUBJECTS[i % len(SUBJECTS)]
        d1, d2 = DETAILS[i % len(DETAILS)], DETAILS[(i * 7 + 3) % len(DETAILS)]
        out.append(f"Panel {i + 1} shows {s} rendered with {d1}, and in the background {d2}; the scene number {i + 1} "
                   f"is described carefully so that every object, colour and material is unambiguous.")
        i += 1
    return " ".join(out)


PROMPTS = {  # word targets -> Qwen3 tokens (counted with the Qwen3-4B tokenizer): 50 -> 103, 3260 -> 4.1k, 4900 -> 6.2k
    "short": 50, "long": 3260, "xlong": 4900}


def run_one(cli: Path, m: dict, pf: Path, flags: list, png: Path, log: Path) -> dict:
    cmd = [str(cli), "--diffusion-model", m["k_dit"], "--vae", m["f2vae"], "--vae-format", "flux2", "--llm", m["qwen3"],
           "--steps", "1", "--cfg-scale", "1.0", "-W", "512", "-H", "512", *flags, "--prompt-file", str(pf),
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
    rec = {"rc": rc, "wall_s": round(time.time() - t0, 2), **mem}
    for key, rx in RX.items():
        vals = [float(v) / (1000.0 if u == "ms" else 1.0) for v, u in rx.findall(text)]
        rec[key] = vals[-1] if vals else None
    keep = re.compile(r"token|compute buffer|flash|attention", re.I)
    rec["log_lines"] = [ln.strip()[:240] for ln in text.splitlines() if keep.search(ln)][:30]
    rec["fallback_lines"] = [ln.strip()[:240] for ln in text.splitlines() if FALLBACK.search(ln)][:12]
    rec["sha256"] = sha256(png) if png.is_file() else None
    rec["tail"] = text[-1500:] if rc != 0 else None
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--rounds", type = int, default = 3)
    ap.add_argument("--prompts", default = "short,long,xlong")
    a = ap.parse_args()
    work, out = Path(a.work), Path(a.out)
    work.mkdir(parents = True, exist_ok = True)
    out.mkdir(parents = True, exist_ok = True)
    jp = out / "obs.json"
    obs: dict = {"host": {"platform": platform.platform()}, "tag": TAG, "vulkan": vulkan_devices(),
                 **windows_driver(), "arms": ARMS}
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
    p = subprocess.run([str(cli), "--help"], capture_output = True, text = True, errors = "replace", cwd = str(cli.parent))
    obs["help_flash_attn"] = [ln.strip() for ln in (p.stdout + p.stderr).splitlines() if "flash" in ln.lower()][:6]
    m = {}
    for k, (repo, f) in FILES.items():
        dest = work / Path(f).name
        obs.setdefault("fetch", {})[k] = fetch(f"{HF}/{repo}/resolve/main/{f}", dest)
        m[k] = str(dest)
    wjson(jp, obs)
    runs: dict = {}
    for pn in a.prompts.split(","):
        pf = out / f"prompt_{pn}.txt"
        pf.write_text(prompt_of(PROMPTS[pn]), encoding = "utf-8")
        runs[pn] = {arm: [] for arm in ARMS}
        for r in range(a.rounds):
            for arm in ROT[r % 3]:
                png = out / f"{pn}_{arm}_r{r}.png"
                runs[pn][arm].append({**run_one(cli, m, pf, ARMS[arm], png, out / f"{pn}_{arm}_r{r}.log"),
                                      "round": r, "png": png.name})
                obs["runs"] = runs
                wjson(jp, obs)
    summ: dict = {}
    for pn, by in runs.items():
        summ[pn] = {}
        for arm, rs in by.items():
            ok = [x for x in rs if x["rc"] == 0 and x.get("te_s") is not None]
            te = [x["te_s"] for x in ok]
            summ[pn][arm] = {
                "n_ok": len(ok), "n_failed": len(rs) - len(ok),
                "te_med": statistics.median(te) if te else None,
                "te_spread": (max(te) - min(te)) if te else None, "te_all": te,
                "generate_med": statistics.median(x["generate_s"] for x in ok if x.get("generate_s")) if ok else None,
                "peak_proc_gib": max((x["peak_proc_gib"] or 0) for x in ok) if ok else None,
                "peak_gpu_used_gib": max((x["peak_gpu_used_gib"] or 0) for x in ok) if ok else None,
                "fallback_logged": any(x["fallback_lines"] for x in rs),
                "shas": sorted(set(x["sha256"] for x in ok))}
        pa = [out / x["png"] for x in by["a"] if x["rc"] == 0]
        for other in ("b", "c"):
            po = [out / x["png"] for x in by[other] if x["rc"] == 0]
            if pa and po:
                summ[pn][f"{other}_vs_a"] = pixel_diff(pa[0], po[0])
        if len(pa) > 1:
            summ[pn]["a_vs_a"] = pixel_diff(pa[0], pa[1])
    obs["summary"] = summ
    obs["status"] = "DONE"
    wjson(jp, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

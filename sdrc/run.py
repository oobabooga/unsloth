"""auto-fit iGPU A/B on the gfx1151 boxes. Observes only; the verdict is drawn from results.json afterwards.
base and head of one backend run in this one job on this one machine, interleaved per rep.

  run.py --bin base=DIR --bin head=DIR --models DIR --out DIR [--reps N] [--env K=V ...]
"""
import argparse, hashlib, json, os, platform, re, subprocess, sys, threading, time, urllib.request
from pathlib import Path

HF = "https://huggingface.co/{}/resolve/main/{}"
MODELS = {
    "sd15": ("Comfy-Org/stable-diffusion-v1-5-archive", "v1-5-pruned-emaonly-fp16.safetensors"),
    "zdit": ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf"),
    "qwen3": ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
    "ae": ("unsloth/FLUX.1-schnell", "ae.safetensors"),
}
PROMPT = "a photo of a red fox sitting in fresh snow, golden hour, detailed fur"
PAT = {
    "load": r"loading tensors completed, taking ([\d.]+)s",
    "te": r"get_learned_condition completed, taking ([\d.]+)s",
    "sampling": r"sampling completed, taking ([\d.]+)s",
    "vae": r"decode_first_stage completed, taking ([\d.]+)s",
    "generate": r"generate_image completed in ([\d.]+)s",
}
KEY = re.compile(r"auto-fit|devices:|RAM +free|params \d|ggml_vulkan: \d|ROCm\d|Vulkan\d|using .*device|"
                 r"compute buffer|backend|graph.cut|segment|tiling|out of memory", re.I)
EXE = ".exe" if os.name == "nt" else ""


def fetch(models_dir, names):
    out = {}
    for n in names:
        repo, f = MODELS[n]
        dest = Path(models_dir) / f
        if not dest.exists():
            for _ in range(5):
                try:
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    with urllib.request.urlopen(HF.format(repo, f), timeout=120) as r, open(tmp, "wb") as fh:
                        while True:
                            b = r.read(1 << 22)
                            if not b:
                                break
                            fh.write(b)
                    tmp.replace(dest)
                    break
                except Exception as e:  # noqa: BLE001
                    print("download retry", f, e, flush=True)
                    time.sleep(10)
        print("model", n, dest, dest.stat().st_size if dest.exists() else "MISSING", flush=True)
        out[n] = str(dest)
    return out


def pix(png):
    try:
        from PIL import Image
        return hashlib.md5(Image.open(png).convert("RGB").tobytes()).hexdigest()
    except Exception as e:  # noqa: BLE001
        return f"ERR:{e}"


def diff(a, b):
    try:
        import numpy as np
        from PIL import Image
        x = np.asarray(Image.open(a).convert("RGB"), dtype=np.float64)
        y = np.asarray(Image.open(b).convert("RGB"), dtype=np.float64)
        d = np.abs(x - y)
        mse = float((d ** 2).mean())
        psnr = float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))
        return {"max": int(d.max()), "mean": float(d.mean()), "frac_diff": float((d > 0).mean()),
                "psnr": "inf" if mse == 0 else round(psnr, 2)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def drm_mem_files():
    """amdgpu sysfs counters (Linux): VRAM carve-out + GTT used, bytes."""
    files = []
    for dev in Path("/sys/class/drm").glob("card*/device"):
        for k in ("mem_info_vram_used", "mem_info_gtt_used"):
            f = dev / k
            if f.exists():
                files.append(f)
    return files


def read_sum(files):
    s = 0
    for f in files:
        try:
            s += int(f.read_text().strip())
        except Exception:  # noqa: BLE001
            pass
    return s


def sd(label, bindir, argv, env_extra, out, tag, timeout=3600):
    exe = str(Path(bindir) / f"sd-cli{EXE}")
    png = out / f"{tag}.png"
    log = out / f"{tag}.log"
    env = dict(os.environ)
    if os.name != "nt":
        env["LD_LIBRARY_PATH"] = str(bindir) + ":" + env.get("LD_LIBRARY_PATH", "")
    env.update(env_extra)
    cmd = [exe, *argv, "-o", str(png), "-v"]
    files = drm_mem_files()
    base_mem = read_sum(files)
    peak = [base_mem]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], read_sum(files))
            time.sleep(0.1)

    th = threading.Thread(target=sampler, daemon=True)
    if files:
        th.start()
    t = time.time()
    rss_kib = None
    rc = None
    logf = open(log, "w", encoding="utf-8", errors="replace")
    logf.write(" ".join(cmd) + "\n")
    logf.flush()
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    if hasattr(os, "wait4"):
        deadline = t + timeout
        while True:
            pid, status, ru = os.wait4(p.pid, os.WNOHANG)
            if pid:
                rc = os.waitstatus_to_exitcode(status)
                rss_kib = ru.ru_maxrss
                p.returncode = rc
                break
            if time.time() > deadline:
                p.kill()
                rc = "timeout"
                break
            time.sleep(0.05)
    else:
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = "timeout"
    wall = time.time() - t
    stop.set()
    logf.close()
    text = log.read_text(encoding="utf-8", errors="replace")
    r = {"label": label, "tag": tag, "rc": rc, "wall": round(wall, 2), "argv": argv, "env": env_extra}
    if rss_kib is not None:
        r["peak_rss_mib"] = round(rss_kib / 1024, 1)
    if files:
        r["gpu_mem_idle_mib"] = round(base_mem / 2**20, 1)
        r["gpu_mem_peak_delta_mib"] = round((peak[0] - base_mem) / 2**20, 1)
    for k, pat in PAT.items():
        m = re.findall(pat, text)
        if m:
            r[k] = sum(float(x) for x in m) if k == "load" else float(m[-1])
    r["planned"] = "auto-fit plan:" in text
    r["no_gpu_devices"] = "auto-fit: no GPU devices" in text
    r["autofit"] = [ln.strip()[:240] for ln in text.splitlines() if "auto-fit" in ln or re.search(r"^\s*\[\w+\s*\]\s.*(free .* budget|params \d+ MiB)", ln)][:30]
    r["key"] = [ln.strip()[:240] for ln in text.splitlines() if KEY.search(ln)][:80]
    if png.exists():
        r["png"] = str(png)
        r["pix_md5"] = pix(png)
    print(json.dumps({k: v for k, v in r.items() if k not in ("key", "autofit")}), flush=True)
    for ln in r["autofit"]:
        print("   ", ln, flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", action="append", default=[])
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--skip-zimg", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    bins = dict(b.split("=", 1) for b in a.bin)
    env_extra = dict(e.split("=", 1) for e in a.env)
    Path(a.models).mkdir(parents=True, exist_ok=True)
    res = {"host": {"platform": platform.platform(), "node": platform.node()}, "bins": bins, "env": env_extra, "runs": []}
    for lab, d in bins.items():
        si = Path(d) / "srcinfo.txt"
        res.setdefault("srcinfo", {})[lab] = si.read_text(encoding="utf-8") if si.exists() else None
    first = next(iter(bins.values()))
    try:
        p = subprocess.run([str(Path(first) / f"sd-cli{EXE}"), "--list-devices"], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, errors="replace", timeout=300,
                           env={**os.environ, **env_extra,
                                **({"LD_LIBRARY_PATH": first + ":" + os.environ.get("LD_LIBRARY_PATH", "")} if os.name != "nt" else {})})
        res["list_devices"] = p.stdout[-3000:]
    except Exception as e:  # noqa: BLE001
        res["list_devices"] = str(e)
    print(res["list_devices"], flush=True)
    if Path("/proc/meminfo").exists():
        res["meminfo"] = Path("/proc/meminfo").read_text().splitlines()[:5]
    res["drm_mem"] = {str(f): f.read_text().strip() for f in drm_mem_files()}
    for dev in Path("/sys/class/drm").glob("card*/device"):
        for k in ("mem_info_vram_total", "mem_info_gtt_total"):
            if (dev / k).exists():
                res["drm_mem"][str(dev / k)] = (dev / k).read_text().strip()

    def save():
        (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    names = ["sd15"] if a.skip_zimg else ["sd15", "zdit", "qwen3", "ae"]
    M = fetch(a.models, names)
    sd15 = ["-m", M["sd15"], "-p", PROMPT, "--steps", "20", "-W", "512", "-H", "512", "--seed", "42"]
    # warm-up (file cache, shader/kernel caches) per arm, not counted
    for lab, d in bins.items():
        res.setdefault("warmup", []).append(sd(lab, d, sd15[:4] + ["--steps", "2", "-W", "512", "-H", "512", "--seed", "42"],
                                               env_extra, out, f"warm_{lab}"))
    save()
    for rep in range(a.reps):
        for lab, d in bins.items():
            res["runs"].append(sd(lab, d, sd15, env_extra, out, f"sd15_{lab}_r{rep}"))
            save()
    if not a.skip_zimg:
        zimg = ["--diffusion-model", M["zdit"], "--llm", M["qwen3"], "--vae", M["ae"], "-p", PROMPT, "--cfg-scale", "1",
                "--steps", "8", "-W", "1024", "-H", "1024", "--seed", "42"]
        for rep in range(2):
            for lab, d in bins.items():
                res["runs"].append(sd(lab, d, zimg, env_extra, out, f"zimg_{lab}_r{rep}"))
                save()
    firsts = {}
    for r in res["runs"]:
        pre = r["tag"].split("_")[0]
        if "png" in r and r["label"] == "base" and pre not in firsts:
            firsts[pre] = r["png"]
    for r in res["runs"]:
        pre = r["tag"].split("_")[0]
        if "png" in r and pre in firsts:
            r["vs_base"] = diff(firsts[pre], r["png"])
    save()
    print("done", flush=True)


if __name__ == "__main__":
    main()

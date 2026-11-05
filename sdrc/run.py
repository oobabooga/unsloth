"""sd.cpp PR recheck runner for the gfx1151 boxes (Linux + Windows). Observes only; the verdict is drawn from
results.json afterwards. Every build of one plan runs in this one job on this one machine, interleaved per rep.

  run.py --plan vk|cpu|rocm --bin LABEL=DIR [--bin ...] --models DIR --out DIR [--reps N]
"""
import argparse, hashlib, json, os, platform, re, statistics, subprocess, sys, time, urllib.request
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
KEY = re.compile(r"direct auto|flash attention auto|loaded \w+ backend|ggml_vulkan: \d|ROCm\d|Vulkan\d|using .*device|"
                 r"n_threads|threads|compute buffer|backend", re.I)
EXE = ".exe" if os.name == "nt" else ""


def fetch(models_dir, names):
    out = {}
    for n in names:
        repo, f = MODELS[n]
        dest = Path(models_dir) / f
        if not dest.exists():
            for attempt in range(5):
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
        im = Image.open(png).convert("RGB")
        return hashlib.md5(im.tobytes()).hexdigest()
    except Exception as e:  # noqa: BLE001
        return f"ERR:{e}"


def diff(a, b):
    try:
        import numpy as np
        from PIL import Image
        x = np.asarray(Image.open(a).convert("RGB"), dtype=np.int16)
        y = np.asarray(Image.open(b).convert("RGB"), dtype=np.int16)
        d = np.abs(x - y)
        return {"max": int(d.max()), "mean": float(d.mean()), "frac_diff": float((d > 0).mean())}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def sd(label, bindir, argv, env_extra, out, tag, timeout=3600):
    exe = str(Path(bindir) / f"sd-cli{EXE}")
    png = out / f"{tag}.png"
    log = out / f"{tag}.log"
    env = dict(os.environ)
    if os.name != "nt":
        env["LD_LIBRARY_PATH"] = str(bindir)
    for k, v in env_extra.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    cmd = [exe, *argv, "-o", str(png), "-v"]
    t = time.time()
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=timeout,
                           text=True, errors="replace")
        rc, text = p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        rc, text = "timeout", (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
    wall = time.time() - t
    log.write_text(" ".join(cmd) + "\n" + text, encoding="utf-8")
    r = {"label": label, "tag": tag, "rc": rc, "wall": round(wall, 2), "env": env_extra, "argv": argv}
    for k, pat in PAT.items():
        m = re.findall(pat, text)
        if m:
            r[k] = sum(float(x) for x in m) if k == "load" else float(m[-1])
    m = re.search(r"loaded CPU backend from (\S+)", text)
    if m:
        r["cpu_variant"] = os.path.basename(m.group(1))
    r["key"] = [ln.strip()[:240] for ln in text.splitlines() if KEY.search(ln)][:60]
    rms = {}
    for ln in text.splitlines():
        mm = re.match(r"(\S*RMS_NORM\S*): (\d+) x ([\d.e+]+) us = ([\d.e+]+) us", ln)
        if mm:
            rms[mm.group(1)] = rms.get(mm.group(1), 0.0) + float(mm.group(4))
    if rms:
        r["vk_rms_us"] = rms
    if png.exists():
        r["png"] = str(png)
        r["pix_md5"] = pix(png)
    print(json.dumps({k: v for k, v in r.items() if k != "key"}), flush=True)
    return r


def tbo(label, bindir, mode, op, backend, out, extra_env=None, timeout=1800):
    exe = str(Path(bindir) / f"test-backend-ops{EXE}")
    if not Path(exe).exists():
        return {"label": label, "mode": mode, "op": op, "error": "no test-backend-ops"}
    env = dict(os.environ)
    if os.name != "nt":
        env["LD_LIBRARY_PATH"] = str(bindir)
    env.update(extra_env or {})
    try:
        p = subprocess.run([exe, mode, "-b", backend, "-o", op], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           env=env, timeout=timeout, text=True, errors="replace")
        rc, text = p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        rc, text = "timeout", ""
    (out / f"tbo_{label}_{mode}_{op}.log").write_text(text, encoding="utf-8")
    r = {"label": label, "mode": mode, "op": op, "rc": rc}
    m = re.findall(r"(\d+)/(\d+) tests passed", text)
    if m:
        r["passed"], r["total"] = int(m[-1][0]), int(m[-1][1])
    r["fail_lines"] = [ln.strip()[:200] for ln in text.splitlines() if "FAIL" in ln][:20]
    if mode == "perf":
        r["perf"] = {}
        for ln in text.splitlines():
            mm = re.match(r"\s*(RMS_NORM\(.*?\))\s.*?([\d.]+) us/run", ln)
            if mm:
                r["perf"][mm.group(1)] = float(mm.group(2))
    print(json.dumps({k: v for k, v in r.items() if k != "fail_lines"}), flush=True)
    return r


def host_info():
    info = {"platform": platform.platform(), "machine": platform.machine(), "node": platform.node(),
            "cpu_count": os.cpu_count()}
    if sys.platform.startswith("linux"):
        try:
            cpu = Path("/proc/cpuinfo").read_text()
            info["model"] = re.search(r"model name\s*:\s*(.*)", cpu).group(1)
            cores = set(re.findall(r"physical id\s*:\s*(\d+)\n(?:.*\n)*?core id\s*:\s*(\d+)", cpu))
            info["physical_cores"] = len(cores) or None
            flags = re.search(r"flags\s*:\s*(.*)", cpu).group(1).split()
            info["isa"] = sorted(f for f in flags if f.startswith(("avx512", "avx_vnni", "amx")) or f in ("avx2", "f16c", "fma"))
        except Exception as e:  # noqa: BLE001
            info["cpuinfo_error"] = str(e)
    else:
        info["model"] = platform.processor()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--bin", action="append", default=[])
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    bins = dict(b.split("=", 1) for b in a.bin)
    Path(a.models).mkdir(parents=True, exist_ok=True)
    res = {"plan": a.plan, "host": host_info(), "bins": bins, "tbo": [], "runs": []}
    for lab, d in bins.items():
        si = Path(d) / "srcinfo.txt"
        res.setdefault("srcinfo", {})[lab] = si.read_text(encoding="utf-8").splitlines()[0] if si.exists() else None
    first = next(iter(bins.values()))
    try:
        p = subprocess.run([str(Path(first) / f"sd-cli{EXE}"), "--list-devices"], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, errors="replace", timeout=300,
                           env={**os.environ, **({"LD_LIBRARY_PATH": first} if os.name != "nt" else {})})
        res["list_devices"] = p.stdout[-3000:]
    except Exception as e:  # noqa: BLE001
        res["list_devices"] = str(e)
    print(res["list_devices"], flush=True)

    def save():
        (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    if a.plan == "vk":
        for lab, d in bins.items():
            for op in ("RMS_NORM", "RMS_NORM_MUL_ADD", "RMS_NORM_MUL_ROPE", "CONV_2D", "IM2COL"):
                res["tbo"].append(tbo(lab, d, "test", op, "Vulkan0", out))
        save()
        for rep in range(a.reps):
            for lab, d in bins.items():
                res["tbo"].append(tbo(f"{lab}_r{rep}", d, "perf", "RMS_NORM", "Vulkan0", out))
        save()
        M = fetch(a.models, ["sd15", "zdit", "qwen3", "ae"])
        sd15 = ["-m", M["sd15"], "-p", PROMPT, "--steps", "4", "-W", "512", "-H", "512", "--seed", "42"]
        zimg = ["--diffusion-model", M["zdit"], "--llm", M["qwen3"], "--vae", M["ae"], "-p", PROMPT, "--cfg-scale", "1",
                "--steps", "4", "-W", "256", "-H", "256", "--seed", "42"]
        arms = [(lab, d, []) for lab, d in bins.items()]
        if "base" in bins:
            arms.append(("base_cdon", bins["base"], ["--conv-direct", "on"]))
            arms.append(("base_cdoff", bins["base"], ["--conv-direct", "off"]))
        for rep in range(a.reps):
            for lab, d, x in arms:
                res["runs"].append(sd(lab, d, sd15 + x, {}, out, f"sd15_{lab}_r{rep}"))
                save()
        for rep in range(2):
            for lab, d in bins.items():
                res["runs"].append(sd(lab, d, zimg, {}, out, f"zimg_{lab}_r{rep}"))
                save()
        for lab, d in bins.items():
            res["runs"].append(sd(lab, d, zimg, {"GGML_VK_PERF_LOGGER": "1"}, out, f"zimgperf_{lab}"))
            save()
    elif a.plan == "cpu":
        M = fetch(a.models, ["sd15"])
        t = str(res["host"].get("physical_cores") or max(1, (os.cpu_count() or 2) // 2))
        base = ["-m", M["sd15"], "-p", PROMPT, "--steps", "4", "-W", "256", "-H", "256", "--seed", "42",
                "--backend", "cpu", "-t", t]
        b, h = bins["base"], bins["h48"]
        arms = [("base", b, [], {}), ("h48", h, [], {}),
                ("h48_sgemm_none", h, [], {"GGML_CPU_SGEMM_TYPES": "none"}),
                ("h48_sgemm_q8", h, [], {"GGML_CPU_SGEMM_TYPES": "q8_0"}),
                ("h48_iqk0", h, [], {"GGML_CPU_IQK": "0"}),
                ("h48_cdoff", h, ["--conv-direct", "off"], {}),
                ("base_cdoff", b, ["--conv-direct", "off"], {})]
        for typ in ("fp16", "q8_0"):
            ta = [] if typ == "fp16" else ["--type", "q8_0"]
            for rep in range(a.reps):
                for lab, d, x, e in arms:
                    res["runs"].append(sd(lab, d, base + ta + x, e, out, f"{typ}_{lab}_r{rep}"))
                    save()
        # default thread count each build picks (no -t)
        for lab, d in (("base", b), ("h48", h)):
            res["runs"].append(sd(lab, d, ["-m", M["sd15"], "-p", PROMPT, "--steps", "1", "-W", "128", "-H", "128",
                                           "--seed", "42", "--backend", "cpu"], {}, out, f"defthreads_{lab}"))
            save()
    elif a.plan == "rocm":
        M = fetch(a.models, ["sd15"])
        sd15 = ["-m", M["sd15"], "-p", PROMPT, "--steps", "4", "-W", "512", "-H", "512", "--seed", "42"]
        for rep in range(2):
            for lab, d in bins.items():
                # the bundle must stand alone: no ROCm on the loader path besides its own lib/
                res["runs"].append(sd(lab, d, sd15, {"LD_LIBRARY_PATH": None, "ROCM_PATH": None, "HIP_PATH": None},
                                      out, f"rocm_{lab}_r{rep}"))
                save()
    # pixel diffs of each run against the first run of label "base" with the same prefix
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

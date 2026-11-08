"""sd.cpp AMD runtime A/B on one gfx1151 box. Observes only (results.json); the verdict is drawn afterwards.
Every arm runs in this one process on this one machine, interleaved per rep (rep 0 of every arm, then rep 1, ...).

  run.py --bin LABEL=DIR [--bin ...] --models DIR --out DIR [--reps 3] [--rocm-lib DIR] [--zimage]
Labels used: vk_master vk_h37 vk_h38 vk_h51 cpu_master rocm14 hv_master hv_h51 (missing ones are skipped).
"""
import argparse, hashlib, json, math, os, platform, re, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HF = "https://huggingface.co/{}/resolve/main/{}"
MODELS = {
    "sd15": ("stable-diffusion-v1-5/stable-diffusion-v1-5", "v1-5-pruned-emaonly.safetensors"),
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
KEY = re.compile(r"direct auto|flash attention auto|auto-fit|devices:|budget|-> compute|params backend|"
                 r"ggml_vulkan: \d|ggml_cuda_init|Device \d:|ROCm\d|Vulkan\d|using .*backend|loaded \w+ backend|"
                 r"compute buffer|graph.cut|segment|n_threads|available .* MB|out of memory|error|abort", re.I)


def fetch_one(models_dir, n):
    repo, f = MODELS[n]
    dest = Path(models_dir) / f
    for attempt in range(5):
        if dest.exists():
            break
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(HF.format(repo, f), timeout=120) as r, open(tmp, "wb") as fh:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    fh.write(b)
            tmp.replace(dest)
        except Exception as e:  # noqa: BLE001
            print("download retry", f, e, flush=True)
            time.sleep(10)
    print("model", n, dest, dest.stat().st_size if dest.exists() else "MISSING", flush=True)
    return n, str(dest)


def fetch(models_dir, names):
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        return dict(ex.map(lambda n: fetch_one(models_dir, n), names))


def load_rgb(png):
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(png).convert("RGB"), dtype=np.float64)


def pix(png):
    try:
        from PIL import Image
        return hashlib.md5(Image.open(png).convert("RGB").tobytes()).hexdigest()
    except Exception as e:  # noqa: BLE001
        return f"ERR:{e}"


def diff(a, b):
    try:
        import numpy as np
        x, y = load_rgb(a), load_rgb(b)
        if x.shape != y.shape:
            return {"error": f"shape {x.shape} vs {y.shape}"}
        d = np.abs(x - y)
        mse = float((d ** 2).mean())
        return {"identical": bool(mse == 0.0), "max": int(d.max()), "mean": round(float(d.mean()), 4),
                "frac_diff": round(float((d > 0).mean()), 5),
                "psnr": None if mse == 0 else round(10 * math.log10(255.0 ** 2 / mse), 2)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def sd(arm, argv, out, tag, timeout=3600):
    exe = str(Path(arm["bin"]) / "sd-cli")
    png = out / f"{tag}.png"
    log = out / f"{tag}.log"
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join([arm["bin"]] + arm.get("ldpath", []))
    for k, v in arm.get("env", {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    cmd = [exe, *argv, *arm.get("argv", []), "-o", str(png), "-v"]
    t = time.time()
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=timeout,
                           text=True, errors="replace")
        rc, text = p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        rc = "timeout"
        text = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
    wall = time.time() - t
    log.write_text(" ".join(cmd) + "\n" + text, encoding="utf-8")
    r = {"arm": arm["label"], "tag": tag, "rc": rc, "wall": round(wall, 2), "argv_extra": arm.get("argv", []),
         "env": arm.get("env", {})}
    for k, pat in PAT.items():
        m = re.findall(pat, text)
        if m:
            r[k] = float(m[-1])
    r["key"] = list(dict.fromkeys(ln.strip()[:260] for ln in text.splitlines() if KEY.search(ln)))[:80]
    if rc == 0 and png.exists():
        r["png"] = str(png)
        r["pix_md5"] = pix(png)
    print(json.dumps({k: v for k, v in r.items() if k not in ("key", "env")}), flush=True)
    return r


def tbo(arm, mode, op, out, timeout=1800):
    exe = Path(arm["bin"]) / "test-backend-ops"
    if not exe.exists():
        return {"arm": arm["label"], "mode": mode, "op": op, "error": "no test-backend-ops"}
    env = {**os.environ, "LD_LIBRARY_PATH": arm["bin"], **{k: v for k, v in arm.get("env", {}).items() if v}}
    try:
        p = subprocess.run([str(exe), mode, "-b", "Vulkan0", "-o", op], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, env=env, timeout=timeout, text=True, errors="replace")
        rc, text = p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        rc, text = "timeout", ""
    (out / f"tbo_{arm['label']}_{mode}_{op}_{int(time.time())}.log").write_text(text, encoding="utf-8")
    r = {"arm": arm["label"], "mode": mode, "op": op, "rc": rc}
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
    info = {"platform": platform.platform(), "node": platform.node(), "cpu_count": os.cpu_count()}
    try:
        cpu = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        info["model"] = re.search(r"model name\s*:\s*(.*)", cpu).group(1)
        info["physical_cores"] = len(set(re.findall(r"physical id\s*:\s*(\d+)\n(?:.*\n)*?core id\s*:\s*(\d+)", cpu))) or None
    except Exception as e:  # noqa: BLE001
        info["cpuinfo_error"] = str(e)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", action="append", default=[])
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rocm-lib", default="")
    ap.add_argument("--zimage", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bins = {k: v for k, v in (b.split("=", 1) for b in a.bin) if Path(v, "sd-cli").exists()}
    res = {"host": host_info(), "bins": bins, "srcinfo": {}, "list_devices": {}, "tbo": [], "runs": []}

    def save():
        (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    VK = {"GGML_VK_VISIBLE_DEVICES": "0"}
    arms = {}
    for lab, d in bins.items():
        arm = {"label": lab, "bin": d, "env": {}}
        if lab.startswith("vk_"):
            arm["env"] = dict(VK)
        elif lab == "cpu_master":
            arm["env"] = {"GGML_VK_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
        elif lab == "rocm14":
            # the bundle must stand alone: rpath $ORIGIN/lib only, no ROCm variables, no loader path
            arm["env"] = {"LD_LIBRARY_PATH": None, "ROCM_PATH": None, "HIP_PATH": None}
        elif lab.startswith("hv_"):
            arm["env"] = dict(VK)
            arm["ldpath"] = [a.rocm_lib] if a.rocm_lib else []
        arms[lab] = arm
        si = Path(d) / "srcinfo.txt"
        res["srcinfo"][lab] = si.read_text(encoding="utf-8").splitlines()[0] if si.exists() else None
    for lab, arm in arms.items():
        env = {**os.environ, "LD_LIBRARY_PATH": ":".join([arm["bin"]] + arm.get("ldpath", []))}
        for k, v in arm["env"].items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
        try:
            p = subprocess.run([str(Path(arm["bin"]) / "sd-cli"), "--list-devices"], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, errors="replace", timeout=300, env=env)
            res["list_devices"][lab] = p.stdout[-3000:]
        except Exception as e:  # noqa: BLE001
            res["list_devices"][lab] = str(e)
        print("==", lab, res["list_devices"][lab], flush=True)
    save()

    # (b) #37 kernel-level: correctness + RMS_NORM perf, master vs head, interleaved
    for lab in ("vk_master", "vk_h37"):
        if lab in arms:
            for op in ("RMS_NORM", "RMS_NORM_MUL_ADD", "RMS_NORM_MUL_ROPE"):
                res["tbo"].append(tbo(arms[lab], "test", op, out))
    save()
    for rep in range(a.reps):
        for lab in ("vk_master", "vk_h37"):
            if lab in arms:
                res["tbo"].append(tbo(arms[lab], "perf", "RMS_NORM", out))
    save()

    M = fetch(a.models, ["sd15"] + (["zdit", "qwen3", "ae"] if a.zimage else []))
    sd15 = ["-m", M["sd15"], "-p", PROMPT, "--steps", "20", "-W", "512", "-H", "512", "--seed", "42",
            "--sampling-method", "euler"]
    order = ["vk_master", "vk_h37", "vk_h38", "vk_h51", "rocm14", "hv_master", "hv_h51", "cpu_master"]
    plan = []  # (case, arm, argv)
    for typ in ("f16", "q8_0"):
        for lab in order:
            if lab in arms:
                plan.append((f"sd15{typ}", arms[lab], sd15 + ["--type", typ]))
        if typ == "f16" and "vk_master" in arms:
            for mode in ("on", "off"):
                arm = {**arms["vk_master"], "label": f"vk_master_cd{mode}", "argv": ["--conv-direct", mode]}
                plan.append((f"sd15{typ}", arm, sd15 + ["--type", typ]))
    for rep in range(a.reps):
        for case, arm, argv in plan:
            res["runs"].append(sd(arm, argv, out, f"{case}_{arm['label']}_r{rep}"))
            save()
    if a.zimage and all(Path(M[k]).exists() for k in ("zdit", "qwen3", "ae")):
        zimg = ["--diffusion-model", M["zdit"], "--llm", M["qwen3"], "--vae", M["ae"], "-p", PROMPT,
                "--cfg-scale", "1", "--steps", "8", "-W", "512", "-H", "512", "--seed", "42",
                "--sampling-method", "euler"]
        for rep in range(a.reps):
            for lab in ("vk_master", "vk_h37", "vk_h38", "vk_h51", "rocm14", "hv_master", "hv_h51"):
                if lab in arms:
                    res["runs"].append(sd(arms[lab], zimg, out, f"zimg_{lab}_r{rep}"))
                    save()
    # pixel diffs: every run vs the first vk_master run of the same case, and vs its own arm's rep 0
    ref, own = {}, {}
    for r in res["runs"]:
        case = r["tag"].split("_")[0]
        if "png" in r:
            if r["arm"] == "vk_master":
                ref.setdefault(case, r["png"])
            own.setdefault((case, r["arm"]), r["png"])
    for r in res["runs"]:
        case = r["tag"].split("_")[0]
        if "png" in r:
            if case in ref:
                r["vs_vk_master"] = diff(ref[case], r["png"])
            r["vs_own_r0"] = diff(own[(case, r["arm"])], r["png"])
    save()
    print("done", flush=True)


if __name__ == "__main__":
    main()

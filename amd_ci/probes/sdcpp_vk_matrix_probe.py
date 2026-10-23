#!/usr/bin/env python3
"""Probe: the sd.cpp Vulkan flag matrix on THIS card, with the published Vulkan bundle Studio installs.

Observes only; criteria/sdcpp_vk_matrix.py judges.

Shared half (done once into --shared, recorded in every observation):
  * the published unslothai/stable-diffusion.cpp Linux Vulkan bundle (--tag), sha256 checked against the
    release's sd-prebuilt-sha256.json; `sd-cli --list-devices`;
  * public models (no token): Z-Image-Turbo Q4_K_M + Qwen3-4B BF16 + FLUX.1 AE, FLUX.2-klein-4B Q4_K_M + FLUX.2 VAE;
  * the matrix: every arm of ARMS per case, interleaved per repeat, one fresh sd-cli per run. Per run: rc, TE /
    sampling / VAE decode / total seconds from sd-cli's own log, wall, peak RSS, sha256 of the PNG, and the
    `conv2d direct auto` / `flash attention auto` decisions the run printed. Repeats of one arm are the A/A noise;
    differing sha256 across repeats of one arm means the arm is not reproducible.

Per state: the state's own Studio native-speed flags for Vulkan (core.inference.sd_cpp_args.native_speed_flags,
imported from the checkout on the system python: the module needs no torch), then that argv run --state-reps times
on the first case.
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
FILES = {
    "z_dit": ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf"),
    "qwen3": ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-BF16.gguf"),
    "ae": ("unsloth/FLUX.1-schnell", "ae.safetensors"),
    "k_dit": ("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_K_M.gguf"),
    "f2vae": ("black-forest-labs/FLUX.2-klein-4B", "vae/diffusion_pytorch_model.safetensors"),
}
PROMPT = "a photo of a red fox sitting in fresh snow, golden hour, detailed fur"


def case_argv(m: dict) -> dict:
    z = ["--diffusion-model", m["z_dit"], "--vae", m["ae"], "--llm", m["qwen3"], "--steps", "8", "--cfg-scale", "1.0"]
    k = ["--diffusion-model", m["k_dit"], "--vae", m["f2vae"], "--vae-format", "flux2", "--llm", m["qwen3"],
         "--steps", "4", "--cfg-scale", "1.0"]
    return {
        "zimage1024": z + ["-W", "1024", "-H", "1024"],
        "zimage512": z + ["-W", "512", "-H", "512"],
        "klein1024": k + ["-W", "1024", "-H", "1024"],
    }


# name -> (sd-cli flags, extra env). studio_pin = what Studio main sends on Vulkan with the master-982 pin.
ARMS = {
    "studio_pin": (["--diffusion-fa"], {}),
    "cd_off": (["--diffusion-fa", "--conv-direct", "off"], {}),
    "cd_on": (["--diffusion-fa", "--conv-direct", "on"], {}),
    "fa_all_cd_off": (["--flash-attn", "on", "--conv-direct", "off"], {}),
    "auto": ([], {}),
    "cd_off_nocoopmat": (["--diffusion-fa", "--conv-direct", "off"], {"GGML_VK_DISABLE_COOPMAT": "1"}),
}
RX = {
    "te_s": re.compile(r"get_learned_condition completed, taking ([0-9.]+)(m?s)"),
    "sampling_s": re.compile(r"sampling completed, taking ([0-9.]+)(m?s)"),
    "decode_s": re.compile(r"decode_first_stage completed, taking ([0-9.]+)(m?s)"),
    "generate_s": re.compile(r"generate_image completed in ([0-9.]+)(m?s)"),
}
DECISION = re.compile(r"(conv[23]d direct auto: \S+ -> \w+[^-\n]*|flash attention auto: \S+ -> \w+[^-\n]*)")


def say(msg: str) -> None:
    print(f"[sdcpp_vk_matrix {time.strftime('%H:%M:%S')}] {msg}", flush = True)


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


def fetch(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents = True, exist_ok = True)
    if dest.is_file() and dest.stat().st_size > 0:
        return {"url": url, "cached": True, "bytes": dest.stat().st_size}
    env = {k: v for k, v in os.environ.items() if "TOKEN" not in k}
    t0 = time.time()
    rc = subprocess.run(["curl", "-sS", "--fail", "-L", "--retry", "5", "--retry-delay", "5", "-o", str(dest) + ".part",
                         url], env = env).returncode
    if rc != 0:
        return {"url": url, "error": f"curl rc={rc}"}
    os.replace(str(dest) + ".part", dest)
    return {"url": url, "bytes": dest.stat().st_size, "seconds": round(time.time() - t0, 1)}


def run_one(binary: Path, argv: list, flags: list, env_extra: dict, out_png: Path, log_path: Path) -> dict:
    cmd = [str(binary), *argv, "--prompt", PROMPT, "--seed", "42", "--output", str(out_png), "-v", *flags]
    env = dict(os.environ)
    env.update(env_extra)
    t0 = time.time()
    with open(log_path, "w", encoding = "utf-8", errors = "replace") as fh:
        p = subprocess.Popen(cmd, stdout = fh, stderr = subprocess.STDOUT, env = env)
        _, status, ru = os.wait4(p.pid, 0)
    rec = {"rc": os.waitstatus_to_exitcode(status), "wall_s": round(time.time() - t0, 3),
           "peak_rss_gb": round(ru.ru_maxrss / 1048576, 3), "flags": flags, "env": env_extra}
    text = log_path.read_text(encoding = "utf-8", errors = "replace")
    for key, rx in RX.items():
        vals = [float(v) / (1000.0 if u == "ms" else 1.0) for v, u in rx.findall(text)]
        rec[key] = vals[-1] if vals else None
    rec["decisions"] = sorted(set(m.strip() for m in DECISION.findall(text)))
    rec["sha256"] = sha256(out_png) if out_png.is_file() else None
    rec["tail"] = text[-600:] if rec["rc"] != 0 else None
    return rec


def shared_half(shared: Path, tag: str, reps: int, cases: list) -> dict:
    done = shared / "shared.json"
    if done.is_file():
        return json.loads(done.read_text(encoding = "utf-8"))
    out: dict = {"tag": tag, "arms": {k: {"flags": v[0], "env": v[1]} for k, v in ARMS.items()}}
    base = f"https://github.com/{REPO}/releases/download/{tag}"
    asset = f"sd-{tag}-bin-Linux-Ubuntu-22.04-x86_64-vulkan.zip"
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
    clis = sorted(bdir.rglob("sd-cli"))
    binary = clis[0] if clis else None
    out["binary"] = str(binary) if binary else None
    if binary:
        binary.chmod(0o755)
        p = subprocess.run([str(binary), "--list-devices"], capture_output = True, text = True, errors = "replace")
        out["list_devices"] = {"rc": p.returncode, "out": (p.stdout + p.stderr)[-2000:]}
        out["gpu_listed"] = p.returncode == 0 and "Vulkan0" in (p.stdout + p.stderr)
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
                for arm, (flags, env) in ARMS.items():
                    tagname = f"{case}__{arm}__r{rep}"
                    say(f"run {tagname}")
                    rec = run_one(binary, argvs[case], flags, env, runs / f"{tagname}.png", runs / f"{tagname}.log")
                    rec["rep"] = rep
                    out["matrix"].setdefault(case, {}).setdefault(arm, []).append(rec)
                    say(f"  rc={rec['rc']} te={rec['te_s']} sampling={rec['sampling_s']} decode={rec['decode_s']} "
                        f"generate={rec['generate_s']} sha={str(rec['sha256'])[:12]} {rec['decisions']}")
                    wjson(shared / "partial.json", out)
    wjson(done, out)
    return out


def state_flags(checkout: Path) -> dict:
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); import core.inference.sd_cpp_args as a; "
            "print(json.dumps(a.native_speed_flags('default', [], 'vulkan')))")
    p = subprocess.run(["/usr/bin/python3", "-I", "-c", code, str(checkout / "studio" / "backend")],
                       capture_output = True, text = True, errors = "replace")
    if p.returncode != 0:
        return {"error": p.stderr[-1500:]}
    return {"flags": json.loads(p.stdout.strip().splitlines()[-1]),
            "sd_cpp_args_sha256": sha256(checkout / "studio/backend/core/inference/sd_cpp_args.py")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--shared", required = True, type = Path)
    ap.add_argument("--tag", default = "master-982-0518b34-ud331c66")
    ap.add_argument("--reps", type = int, default = 3)
    ap.add_argument("--state-reps", type = int, default = 2)
    ap.add_argument("--cases", default = "zimage1024,klein1024")
    a = ap.parse_args()
    cases = a.cases.split(",")
    obs: dict = {"state": a.state}
    try:
        sh = shared_half(a.shared, a.tag, a.reps, cases)
    except Exception as e:  # noqa: BLE001
        sh = {"error": f"{type(e).__name__}: {e}"}
    obs["shared"] = sh
    # The renders and sd-cli logs go next to the observation (the uploaded artifact); the models stay in --shared.
    keep = a.out.parent / "vk_runs"
    if (a.shared / "runs").is_dir() and not keep.is_dir():
        import shutil
        shutil.copytree(a.shared / "runs", keep)
    st = state_flags(a.checkout)
    obs["studio_vulkan"] = st
    if sh.get("binary") and sh.get("gpu_listed") and "flags" in st:
        argv = case_argv(sh["models"])[cases[0]]
        sdir = a.shared / f"state_{a.state}"
        sdir.mkdir(parents = True, exist_ok = True)
        st["case"] = cases[0]
        st["runs"] = []
        for rep in range(a.state_reps):
            rec = run_one(Path(sh["binary"]), argv, st["flags"], {}, sdir / f"r{rep}.png", sdir / f"r{rep}.log")
            st["runs"].append(rec)
            keep.mkdir(parents = True, exist_ok = True)
            for ext in ("png", "log"):
                src = sdir / f"r{rep}.{ext}"
                if src.is_file():
                    (keep / f"state_{a.state}_r{rep}.{ext}").write_bytes(src.read_bytes())
            say(f"state {a.state} r{rep}: rc={rec['rc']} generate={rec['generate_s']} decode={rec['decode_s']}")
    wjson(a.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

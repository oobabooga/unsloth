#!/usr/bin/env python3
"""Probe: Segmind Vega (distilled SDXL, 1024x1024, 20 steps, CFG 7.5) on the XDNA2 NPU
(Ryzen AI 1.9.0 Linux runtime + amd/segmind-vega-amdnpu via the community
mcolsen/amd-npu-stable-diffusion-linux runner) versus the gfx1151 iGPU (diffusers, ROCm
torch from the Studio venv), same job, same host. Observes only: timings, socket power
samples, return codes, log tails. Heavy work only in the `base` state; other states skip.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

RAI = "https://download.amd.com/opendownload/RyzenAI/1.9.0"
DEB = "ryzen-ai-runtime_1.9.0-1_amd64.deb"
MC_REPO = "https://github.com/mcolsen/amd-npu-stable-diffusion-linux"
MC_SHA = "46837978a3d6c0c98357272b0cfe1294c07d2d6b"
PROMPT = "a corgi sitting in a field of wildflowers, photograph"
STEPS = 20
GUIDANCE = 7.5
SEED = 42


def sh(cmd, log: Path, timeout=3600, env=None, cwd=None):
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n$ {cmd}\n")
        f.flush()
        try:
            p = subprocess.run(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT,
                               timeout=timeout, env=env, cwd=cwd)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    return {"rc": rc, "s": round(time.time() - t0, 1)}


def tail(p: Path, n=60):
    try:
        return "\n".join(p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except Exception:
        return ""


class Power:
    """Samples the amdgpu hwmon power reading (on this APU: socket / PPT power)."""

    def __init__(self):
        self.path = None
        for h in Path("/sys/class/hwmon").glob("hwmon*"):
            try:
                if (h / "name").read_text().strip() == "amdgpu":
                    for fn in ("power1_input", "power1_average"):
                        if (h / fn).exists():
                            self.path = h / fn
                            break
            except Exception:
                pass
        self.samples = []
        self._stop = threading.Event()

    def read(self):
        try:
            return int(self.path.read_text()) / 1e6
        except Exception:
            return None

    def __enter__(self):
        self.samples = []
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                v = self.read()
                if v is not None:
                    self.samples.append((time.time(), v))
                time.sleep(0.2)
        self.t = threading.Thread(target=loop, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.t.join(timeout=2)

    def stats(self):
        v = [s[1] for s in self.samples]
        if not v:
            return {"source": str(self.path), "n": 0}
        return {"source": str(self.path), "n": len(v), "mean_w": round(sum(v) / len(v), 2),
                "max_w": round(max(v), 2), "dur_s": round(self.samples[-1][0] - self.samples[0][0], 2)}


def idle_power(pw, secs=10):
    with pw:
        time.sleep(secs)
    return pw.stats()


def npu_arm(W: Path, obs: dict, pw: Power):
    log = W / "npu.log"
    o = obs.setdefault("npu", {})
    o["log"] = str(log)
    env = dict(os.environ)
    env.pop("HF_TOKEN", None)
    env["HF_HOME"] = str(W / "hf")
    env["XILINX_XRT"] = "/opt/xilinx/xrt"
    env["LD_LIBRARY_PATH"] = "/opt/xilinx/xrt/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["PATH"] = "/opt/xilinx/xrt/bin:" + env["PATH"]
    steps = o.setdefault("steps", {})
    steps["fetch"] = sh(f"curl -sSfL -o {W}/{DEB} {RAI}/{DEB} && curl -sSfL -o {W}/setup_ryzen_ai.sh "
                        f"{RAI}/setup_ryzen_ai.sh && ls -la {W}/{DEB}", log, 1800)
    if steps["fetch"]["rc"] != 0:
        return
    # rootless install: -c extracts the .deb with dpkg-deb -x instead of sudo dpkg -i
    sc = W / "setup_ryzen_ai.sh"
    has_ensurepip = subprocess.run(["python3.12", "-c", "import ensurepip"], capture_output=True).returncode == 0
    o["ensurepip"] = has_ensurepip
    if not has_ensurepip:
        uv = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
        s = sc.read_text(encoding="utf-8")
        s = s.replace('python3.12 -c "import ensurepip"', "true")
        s = s.replace('python3.12 -m venv "${VENV_PATH}" --copies --prompt "${PROMPT_NAME}"',
                      f'{uv} venv --seed --python python3.12 "${{VENV_PATH}}" --prompt "${{PROMPT_NAME}}"')
        sc.write_text(s, encoding="utf-8")
        o["venv_via"] = uv
    steps["rai_install"] = sh(f"bash {sc} -a yes -d {W}/{DEB} -c {W}/rai_cpp -p {W}/rai_venv", log, 3600, env=env)
    if steps["rai_install"]["rc"] != 0:
        return
    custom = subprocess.run(["bash", "-c", f"find {W}/rai_cpp -name libonnx_custom_ops.so | head -1"],
                            capture_output=True, text=True).stdout.strip()
    o["custom_ops"] = custom
    if not custom:
        return
    libdir = str(Path(custom).parent)
    o["deploy_libs"] = sorted(os.listdir(libdir))[:80]
    py = f"{W}/rai_venv/bin/python"
    steps["deps"] = sh(f"{W}/rai_venv/bin/uv pip install --python {py} diffusers==0.31.0 'transformers==4.57.6' "
                       f"'huggingface_hub<1' safetensors Pillow 2>&1 | tail -20; "
                       f"{py} -c 'import onnxruntime as o; print(o.__version__, o.__file__)'", log, 1800, env=env)
    steps["clone"] = sh(f"git clone -q {MC_REPO} {W}/mc && git -C {W}/mc checkout -q {MC_SHA}", log, 600)
    if steps["clone"]["rc"] != 0:
        return
    steps["model"] = sh(f"cd {W}/mc && {py} download_model.py --model vega", log, 3600, env=env)
    if steps["model"]["rc"] != 0:
        return
    env["DD_ROOT"] = libdir
    env["LD_LIBRARY_PATH"] = libdir + ":" + env["LD_LIBRARY_PATH"]
    runs = o.setdefault("runs", [])
    for i in range(3):
        rlog = W / f"npu_run{i}.log"
        with pw:
            r = sh(f"cd {W}/mc && {py} run_npu.py --model vega --steps {STEPS} --guidance-scale {GUIDANCE} "
                   f"--seed {SEED} --custom-op-path {custom} --output {W}/npu_{i}.png \"{PROMPT}\"",
                   rlog, 1800, env=env)
        txt = rlog.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"Total: ([\d.]+)s \(load: ([\d.]+)s, denoise: ([\d.]+)s, vae: ([\d.]+)s\)", txt)
        e = re.search(r"Encod\w+ prompt.*?([\d.]+)s", txt)
        r.update({"power": pw.stats(), "tail": tail(rlog, 25)})
        if m:
            r.update({"total_s": float(m[1]), "load_s": float(m[2]), "denoise_s": float(m[3]),
                      "vae_s": float(m[4])})
        runs.append(r)
        if r["rc"] != 0:
            break
    steps["dmesg_hint"] = sh("xrt-smi examine 2>&1 | tail -8", log, 60, env=env)


IGPU_SCRIPT = r'''
import json, sys, time, torch
from diffusers import StableDiffusionXLPipeline
out = sys.argv[1]
r = {"torch": torch.__version__, "hip": torch.version.hip, "dev": torch.cuda.get_device_name(0)}
t = time.time()
pipe = StableDiffusionXLPipeline.from_pretrained("segmind/Segmind-Vega", torch_dtype=torch.float16,
                                                 variant="fp16", use_safetensors=True).to("cuda")
pipe.set_progress_bar_config(disable=True)
r["load_s"] = time.time() - t
runs = []
for i in range(4):
    g = torch.Generator("cuda").manual_seed(%(seed)d)
    torch.cuda.synchronize(); t0 = time.time()
    pe, npe, ppe, nppe = pipe.encode_prompt(prompt="%(prompt)s", device="cuda", do_classifier_free_guidance=True)
    torch.cuda.synchronize(); t1 = time.time()
    lat = pipe(prompt_embeds=pe, negative_prompt_embeds=npe, pooled_prompt_embeds=ppe,
               negative_pooled_prompt_embeds=nppe, num_inference_steps=%(steps)d, guidance_scale=%(cfg)s,
               height=1024, width=1024, generator=g, output_type="latent").images
    torch.cuda.synchronize(); t2 = time.time()
    vae = pipe.vae
    if vae.config.force_upcast:
        vae.to(torch.float32)
    img = vae.decode(lat.to(next(vae.parameters()).dtype) / vae.config.scaling_factor).sample
    torch.cuda.synchronize(); t3 = time.time()
    runs.append({"te_s": t1 - t0, "denoise_s": t2 - t1, "vae_s": t3 - t2, "total_s": t3 - t0,
                 "nan": bool(torch.isnan(img).any())})
    print(json.dumps(runs[-1]), flush=True)
r["runs"] = runs
r["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
open(out, "w").write(json.dumps(r))
'''


def igpu_arm(W: Path, obs: dict, pw: Power):
    log = W / "igpu.log"
    o = obs.setdefault("igpu", {})
    env = dict(os.environ)
    env.pop("HF_TOKEN", None)
    env["HF_HOME"] = str(W / "hf")
    py = sys.executable
    o["python"] = py
    o["deps"] = sh(f"{py} -c 'import diffusers' || (uv pip install --python {py} diffusers || {py} -m pip install diffusers)",
                   log, 1200, env=env)
    o["prefetch"] = sh(f"{py} -c \"from huggingface_hub import snapshot_download as s; "
                       f"s('segmind/Segmind-Vega', allow_patterns=['*.json','*.txt','*fp16.safetensors'])\"",
                       log, 3600, env=env)
    scr = W / "igpu_bench.py"
    scr.write_text(IGPU_SCRIPT % {"seed": SEED, "prompt": PROMPT, "steps": STEPS, "cfg": GUIDANCE},
                   encoding="utf-8")
    res = W / "igpu.json"
    with pw:
        o["run"] = sh(f"{py} {scr} {res}", log, 3600, env=env)
    # per-run power windows are approximated by the whole-script window (warmup run included)
    o["power_whole_script"] = pw.stats()
    o["power_samples"] = pw.samples[-2000:]
    try:
        o["result"] = json.loads(res.read_text())
    except Exception as e:  # noqa: BLE001
        o["result_error"] = repr(e)
    o["tail"] = tail(log, 40)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--checkout", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    obs = {"state": a.state}
    if a.state != "base":
        obs["_skipped"] = "heavy NPU vs iGPU timing runs only in the base state"
        a.out.write_text(json.dumps(obs), encoding="utf-8")
        return 0
    W = Path(os.environ.get("AMD_CI_WORK", os.environ.get("RUNNER_TEMP", "/tmp"))) / "npu_sd"
    W.mkdir(parents=True, exist_ok=True)
    pw = Power()
    obs["power_source"] = str(pw.path)
    obs["idle_power"] = idle_power(pw)
    for name, fn in (("npu", npu_arm), ("igpu", igpu_arm)):
        try:
            fn(W, obs, pw)
        except Exception as e:  # noqa: BLE001
            obs.setdefault(name, {})["exception"] = repr(e)
        a.out.write_text(json.dumps(obs, indent=2, default=str), encoding="utf-8")
    obs["npu"]["log_tail"] = tail(W / "npu.log", 80)
    obs["idle_power_after"] = idle_power(pw)
    # keep the NPU sample image + one iGPU-free artefact next to the observations
    for f in W.glob("npu_*.png"):
        shutil.copy(f, a.out.parent / f.name)
    a.out.write_text(json.dumps(obs, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

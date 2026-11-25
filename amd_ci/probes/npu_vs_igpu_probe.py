#!/usr/bin/env python3
"""Probe (2026-11-24 re-check): same-host XDNA2 NPU (Ryzen AI 1.9 + amd/sd-sandbox) vs
gfx1151 iGPU (diffusers, ROCm torch from the Studio venv) on SD-Turbo / SDXL-Turbo 512px 1 step.
Also lists the Ryzen AI Linux tgz for a GenAI-SD / lib/transaction/stx tree BEFORE deleting it
(the 2026-11-21 probe only searched the installed venv).

Earlier docstring:

Observes only; criteria/npu_diffusion_bench.py judges. Writes JSON via --out.

Survey (every state, cheap): PCI id, /dev/accel, amdxdna module + params
(watchdog timeout_in_sec), firmware dirs, XRT, FastFlowLM (flm), Lemonade,
Ryzen AI installs, python3.12, disk, RAM, hwmon power sensors.

Heavy (only at --heavy-state, default head; the checkout is irrelevant to it):
  1. XRT userspace: system /opt/xilinx/xrt, else the public RAI 1.9 XRT debs
     extracted under the work dir (no root), then `xrt-smi examine`.
  2. Ryzen AI 1.9 Linux SDK (public download.amd.com/opendownload tgz, 12.9 GB),
     installed with AMD's own install_ryzen_ai.sh into a venv (EULA flag -a yes).
  3. quicktest.py (AMD's NPU smoke test).
  4. amd/sd-sandbox (MIT, pinned commit) test/run.py on public amd/*-amdnpu
     repos, no HF token: sd-turbo 1 step, FLUX.2-klein-4B 4 steps, sdxl-base
     30 steps, sdxl-turbo 1 step (each at its model default resolution); one cold process run and
     the pipeline's own --enable_profile rounds; SoC power sampled from hwmon.
No secrets are used or needed: every repo and download here is public.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

RAI_TGZ = "https://download.amd.com/opendownload/RyzenAI/1.9.0/ryzen_ai-1.9.0.tgz"
RAI_XRT = "https://download.amd.com/opendownload/RyzenAI/1.9.0/RAI_1.9_Linux_NPU_XRT.zip"
SANDBOX = "https://github.com/amd/sd-sandbox"
SANDBOX_SHA = os.environ.get("SD_SANDBOX_SHA", "58631e6011d95eb8fd5328e9745c6f52f710bb5d")  # pinned; override via env

# (tag, model_id, extra args, est GB download, timeout s)
MODELS = [
    ("sd_turbo_512_1", "amd/sd-turbo-amdnpu", ["-n", "1"], 4, 1500),
    ("sdxl_turbo_512_1", "amd/sdxl-turbo-amdnpu", ["-n", "1"], 7, 1500),
    ("klein4b_1024_4", "amd/FLUX.2-klein-4B-amdnpu", ["-n", "4"], 9, 2400),
]
# iGPU twins (public, ungated): (tag, repo, pipeline class, res, steps)
IGPU_MODELS = [
    ("sd_turbo_512_1", "stabilityai/sd-turbo", "AutoPipelineForText2Image", 512, 1),
    ("sdxl_turbo_512_1", "stabilityai/sdxl-turbo", "AutoPipelineForText2Image", 512, 1),
]
PROMPT = "a photo of a red fox sitting in a snowy forest, detailed, sharp focus"


def sh(cmd: str, timeout: int = 60, env: dict | None = None, cwd: str | None = None,
       log: Path | None = None) -> dict:
    t0 = time.time()
    try:
        p = subprocess.run(["bash", "-c", "set -o pipefail; " + cmd], capture_output = True, text = True,
                           timeout = timeout, env = env, cwd = cwd)
        out, rc = (p.stdout or "") + (p.stderr or ""), p.returncode
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""))
        out += "\n<TIMEOUT>"
        rc = -9
    except Exception as e:  # noqa: BLE001
        out, rc = f"{type(e).__name__}: {e}", -1
    if log is not None:
        log.parent.mkdir(parents = True, exist_ok = True)
        with log.open("a", encoding = "utf-8") as fh:
            fh.write(f"\n$ {cmd}\n[rc={rc} {time.time() - t0:.1f}s]\n{out}\n")
    return {"rc": rc, "s": round(time.time() - t0, 2), "out": out[-6000:]}


def read(p: str) -> str | None:
    try:
        return Path(p).read_text(encoding = "utf-8", errors = "replace").strip()
    except Exception:  # noqa: BLE001
        return None


def power_sensors() -> list[dict]:
    out = []
    for d in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name = read(f"{d}/name")
        for f in ("power1_average", "power1_input"):
            v = read(f"{d}/{f}")
            if v is not None:
                out.append({"path": f"{d}/{f}", "name": name, "uW": v})
    return out


class PowerSampler:
    """Samples every hwmon power*_average/_input once a second while running."""

    def __init__(self):
        self.paths = [s["path"] for s in power_sensors()]
        self.samples: dict[str, list[float]] = {p: [] for p in self.paths}
        self._stop = threading.Event()
        self._t = threading.Thread(target = self._run, daemon = True)

    def _run(self):
        while not self._stop.is_set():
            for p in self.paths:
                v = read(p)
                try:
                    self.samples[p].append(int(v) / 1e6)
                except Exception:  # noqa: BLE001
                    pass
            self._stop.wait(1.0)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout = 5)

    def summary(self) -> dict:
        res = {}
        for p, v in self.samples.items():
            if v:
                s = sorted(v)
                res[p] = {"n": len(v), "median_W": round(s[len(s) // 2], 2), "max_W": round(s[-1], 2)}
        return res


def survey(log: Path) -> dict:
    o: dict = {}
    o["uname"] = sh("uname -a", log = log)["out"].strip()
    o["os_release"] = sh("grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release", log = log)["out"].strip()
    o["id"] = sh("id", log = log)["out"].strip()
    o["nproc"] = sh("nproc", log = log)["out"].strip()
    o["mem"] = sh("free -g | head -2", log = log)["out"].strip()
    work = os.environ.get("AMD_CI_WORK", "/tmp")
    o["disk_work"] = sh(f"df -BG --output=avail,size,target '{work}' | tail -1", log = log)["out"].strip()
    o["lspci_npu"] = sh("lspci -nn 2>/dev/null | grep -i -E '1022:(17f0|1502)|Signal processing|Neural' ", log = log)["out"].strip()
    o["dev_accel"] = sh("ls -l /dev/accel/ 2>&1", log = log)["out"].strip()
    o["accel_rw"] = os.access("/dev/accel/accel0", os.R_OK | os.W_OK)
    o["lsmod_amdxdna"] = sh("lsmod 2>/dev/null | grep -i xdna", log = log)["out"].strip()
    o["amdxdna_params"] = {Path(p).name: read(p) for p in glob.glob("/sys/module/amdxdna/parameters/*")}
    o["amdxdna_version"] = read("/sys/module/amdxdna/version")
    o["modinfo_amdxdna"] = sh("modinfo amdxdna 2>&1 | grep -E '^(filename|version|srcversion|vermagic|firmware):' | head -12", log = log)["out"].strip()
    acc = {}
    for d in glob.glob("/sys/class/accel/accel*"):
        acc[d] = {k: read(f"{d}/device/{k}") for k in ("vendor", "device", "revision", "fw_version", "vbnv")}
        acc[d]["driver"] = os.path.realpath(f"{d}/device/driver") if os.path.exists(f"{d}/device/driver") else None
    o["sys_accel"] = acc
    o["fw_dirs"] = sh("ls -la /lib/firmware/amdnpu/*/ 2>&1 | head -30", log = log)["out"].strip()
    o["xrt_system"] = os.path.isdir("/opt/xilinx/xrt")
    o["xrt_smi_path"] = shutil.which("xrt-smi")
    o["flm_path"] = shutil.which("flm") or next((p for p in ("/usr/bin/flm", "/usr/local/bin/flm", "/opt/fastflowlm/bin/flm") if os.path.exists(p)), None)
    if o["flm_path"]:
        o["flm_version"] = sh(f"'{o['flm_path']}' version 2>&1 | head -20", timeout = 60, log = log)
        o["flm_validate"] = sh(f"'{o['flm_path']}' validate 2>&1 | tail -40", timeout = 120, log = log)
        o["flm_list"] = sh(f"'{o['flm_path']}' list 2>&1 | tail -40", timeout = 60, log = log)
    o["fastflowlm_dirs"] = sh("ls -d /opt/fastflowlm* /usr/share/flm* /usr/local/share/flm* ~/.config/flm ~/.flm 2>/dev/null; dpkg -l 2>/dev/null | grep -i -E 'fastflow|flm|xrt|xdna|ryzen|lemonade' | head -20", log = log)["out"].strip()
    o["lemonade"] = shutil.which("lemonade-server") or shutil.which("lemonade")
    o["ryzen_ai_dirs"] = sh("ls -d /opt/amd/ryzenai* /opt/ryzen_ai* ~/ryzen_ai* 2>/dev/null", log = log)["out"].strip()
    o["python312"] = shutil.which("python3.12")
    o["power_sensors"] = power_sensors()
    o["xdna_procs"] = sh("ps -eo user,pid,etime,args 2>/dev/null | grep -i -E 'flm|lemonade|xrt' | grep -v grep | head", log = log)["out"].strip()
    return o


def setup_xrt(work: Path, log: Path) -> tuple[dict, dict]:
    """Return (obs, env) with XRT userspace usable without root."""
    env = dict(os.environ)
    obs: dict = {}
    if os.path.isdir("/opt/xilinx/xrt/lib"):
        root = Path("/opt/xilinx/xrt")
        obs["xrt_source"] = "system"
    else:
        d = work / "xrt"
        d.mkdir(parents = True, exist_ok = True)
        r = sh(f"cd '{d}' && curl -fsSL -A 'Mozilla/5.0' -o xrt.zip '{RAI_XRT}' && unzip -oq xrt.zip && "
               f"for f in *.deb; do dpkg-deb -x \"$f\" root; done && ls root/opt/xilinx/xrt/lib | head -50",
               timeout = 600, log = log)
        obs["xrt_extract"] = {"rc": r["rc"], "s": r["s"]}
        root = d / "root/opt/xilinx/xrt"
        obs["xrt_source"] = "extracted RAI_1.9_Linux_NPU_XRT.zip"
    env["XILINX_XRT"] = str(root)
    env["LD_LIBRARY_PATH"] = f"/lib/x86_64-linux-gnu:{root}/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["PATH"] = f"{root}/bin:" + env.get("PATH", "")
    smi = root / "bin/unwrapped/xrt-smi"
    if not smi.exists():
        smi = root / "bin/xrt-smi"
    obs["xrt_root"] = str(root)
    obs["xrt_smi_examine"] = sh(f"'{smi}' examine 2>&1 | head -80", timeout = 120, env = env, log = log)
    obs["xrt_smi_platform"] = sh(f"'{smi}' examine -r platform 2>&1 | head -80", timeout = 120, env = env, log = log)
    obs["xrt_smi_cmd"] = str(smi)
    return obs, env


def install_rai(work: Path, env: dict, log: Path) -> tuple[dict, dict]:
    obs: dict = {}
    pkg = work / "rai_pkg"
    venv = work / "rai_venv"
    pkg.mkdir(parents = True, exist_ok = True)
    r = sh(f"cd '{pkg}' && curl -fsSL -A 'Mozilla/5.0' '{RAI_TGZ}' | tar -xz && ls | wc -l",
           timeout = 3600, log = log)
    obs["download_extract"] = {"rc": r["rc"], "s": r["s"], "tail": r["out"][-500:]}
    if r["rc"] != 0:
        return obs, env
    r = sh(f"cd '{pkg}' && ./install_ryzen_ai.sh -a yes -p '{venv}' 2>&1 | tail -60",
           timeout = 3600, env = env, log = log)
    obs["install"] = {"rc": r["rc"], "s": r["s"], "tail": r["out"][-3000:]}
    # The *.whl/*.tar.gz are no longer needed once installed; free ~13 GB for models.
    # 2026-11-24: look for a GenAI-SD / transaction tree in the tgz itself before deleting it.
    obs["tgz_top"] = sh(f"cd '{pkg}' && find . -maxdepth 2 | sort | head -80", log = log)["out"]
    obs["tgz_genai_dirs"] = sh(f"cd '{pkg}' && find . \\( -iname '*genai*' -o -iname 'transaction' -o -path '*transaction/stx' \\) | head -40", timeout = 300, log = log)["out"]
    obs["tgz_inner_hits"] = sh(f"cd '{pkg}' && set +o pipefail; for t in $(find . -name '*.tar.gz' -o -name '*.tgz' -o -name '*.zip' -o -name '*.whl'); do "
                               f"( case $t in *.zip|*.whl) python3 -c 'import zipfile,sys;print(chr(10).join(zipfile.ZipFile(sys.argv[1]).namelist()))' $t ;; *) tar tzf $t ;; esac ) 2>/dev/null | grep -i -E 'genai-sd|transaction/(stx|mds)' | head -5 | sed \"s|^|$t: |\"; done | head -60",
                               timeout = 1200, log = log)["out"]
    obs["install_script_genai"] = sh(f"grep -n -i -E 'genai|transaction|RYZEN_AI_INSTALLATION_PATH' '{pkg}'/install_ryzen_ai.sh | head -30", log = log)["out"]
    gsd = work / "genai_lib"
    r = sh(f"cd '{pkg}' && d=$(find . -type d -path '*transaction/stx' | head -1); if [ -n \"$d\" ]; then L=$(dirname $(dirname $d)); cp -a \"$L\" '{gsd}' && echo COPIED $L; fi; "
           f"if [ ! -d '{gsd}' ]; then for t in $(find . -name '*.tar.gz' -o -name '*.tgz'); do if [ \"$(set +o pipefail; tar tzf $t 2>/dev/null | grep -c -E 'transaction/stx/')\" != 0 ]; then mkdir -p '{work}/gx' && tar xzf $t -C '{work}/gx' && d=$(find '{work}/gx' -type d -path '*transaction/stx' | head -1) && cp -a $(dirname $(dirname $d)) '{gsd}' && echo EXTRACTED $t; break; fi; done; fi; ls '{gsd}' 2>&1 | head", timeout = 1800, log = log)
    obs["genai_lib_copy"] = r["out"][-1500:]
    sh(f"rm -rf '{pkg}' '{work}/gx'", timeout = 600, log = log)
    venv_env = dict(env)
    venv_env["VIRTUAL_ENV"] = str(venv)
    venv_env["PATH"] = f"{venv}/bin:" + env["PATH"]
    venv_env["RYZEN_AI_INSTALLATION_PATH"] = str(venv)
    venv_env["LD_LIBRARY_PATH"] = f"/lib/x86_64-linux-gnu:{venv}/deployment/lib:{venv}/onnxruntime/lib:" + env["LD_LIBRARY_PATH"]
    obs["venv_python"] = sh(f"'{venv}/bin/python' -c 'import sys,onnxruntime as o;print(sys.version);print(o.__version__, o.__file__);print(o.get_available_providers())'",
                            timeout = 300, env = venv_env, log = log)
    obs["venv_genai"] = sh(f"find '{venv}' \\( -iname '*genai*' -o -iname transaction \\) 2>/dev/null | head -20", timeout = 300, log = log)["out"]
    obs["deployment_ls"] = sh(f"ls '{venv}/deployment' '{venv}/deployment/lib' 2>&1 | head -60", log = log)["out"]
    obs["pip_freeze"] = sh(f"'{venv}/bin/python' -m pip list 2>/dev/null | grep -i -E 'torch|onnx|diffusers|transformers|numpy|ryzen|voe|flexml' ",
                           timeout = 120, env = venv_env, log = log)["out"]
    q = sh(f"cd '{venv}/quicktest' && '{venv}/bin/python' quicktest.py 2>&1 | tail -40",
           timeout = 900, env = venv_env, log = log)
    obs["quicktest"] = q
    return obs, venv_env


def run_sandbox(work: Path, venv_env: dict, out_dir: Path, log: Path, deadline: float) -> dict:
    obs: dict = {}
    sb = work / "sd-sandbox"
    r = sh(f"git clone -q {SANDBOX} '{sb}' && cd '{sb}' && "
           + (f"git checkout -q {SANDBOX_SHA} && " if SANDBOX_SHA else "")
           + "git log -1 --format='%H %cd'", timeout = 300, log = log)
    obs["clone"] = r
    if r["rc"] != 0:
        return obs
    py = f"{venv_env['VIRTUAL_ENV']}/bin/python"
    # sd-sandbox imports diffusers / transformers / onnx; install what the SDK venv lacks.
    req = sh(f"cd '{sb}' && uv pip install --python '{py}' 'diffusers==0.31.0' 'transformers==4.56.2' 'tokenizers==0.22.1' 'numpy==2.5.3' openpyxl tabulate sentencepiece psutil pandas scipy scikit-image pillow pyyaml 2>&1 | tail -15 && '{py}' -c 'import diffusers, transformers; print(diffusers.__version__, transformers.__version__)'",
             timeout = 900, env = venv_env, log = log)
    obs["deps"] = req
    venv = venv_env["VIRTUAL_ENV"]
    env = dict(venv_env)
    env["RYZENAI_GENAI_SD_MODELS_ROOT"] = str(work / "models")
    env["HF_HOME"] = str(work / "hf")
    env.pop("HF_TOKEN", None)
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    # sd-sandbox looks for <project>/lib (DD_ROOT) and lib/transaction/stx (DD_PLUGINS_ROOT).
    # The Linux SDK ships no GenAI-SD tree; point lib/ at the SDK deployment libs and record
    # whether a transaction dir exists anywhere in the install.
    lib = sb / "lib"
    gsd = work / "genai_lib"
    if not (gsd / "transaction" / "stx").is_dir():
        hit = next(iter(sorted(Path(venv).rglob("transaction/stx"))), None)
        if hit is not None and hit.is_dir():
            gsd = hit.parent.parent
            obs["genai_lib_from_venv"] = str(gsd)
    if (gsd / "transaction" / "stx").is_dir():
        shutil.copytree(gsd, lib, dirs_exist_ok = True)
        obs["genai_lib_used"] = True
    else:
        obs["genai_lib_used"] = False
    lib.mkdir(exist_ok = True)
    sh(f"cp -P '{venv}'/deployment/lib/*.so* '{lib}/' 2>/dev/null; cp -P '{venv}'/deployment/*.so* '{lib}/' 2>/dev/null; ls '{lib}' | head -60", log = log)
    obs["transaction_dirs"] = sh(f"find '{venv}' -type d -name transaction 2>/dev/null | head; find '{venv}' -type d -path '*transaction/stx' 2>/dev/null | head", timeout = 300, log = log)["out"]
    custom = lib / "libonnxruntime_providers_ryzenai.so"
    obs["custom_op_path"] = str(custom) if custom.exists() else None
    obs["idle_power"] = None
    with PowerSampler() as ps:
        time.sleep(10)
    obs["idle_power"] = ps.summary()
    runs = {}
    for tag, mid, extra, gb, tmo in MODELS:
        if time.time() > deadline:
            runs[tag] = {"skipped": "time budget"}
            continue
        if tag.startswith("klein") and not obs["genai_lib_used"]:
            runs[tag] = {"skipped": "no lib/transaction/stx found (needed for FLUX)"}
            continue
        free = shutil.disk_usage(str(work)).free / 1e9
        if free < gb + 8:
            runs[tag] = {"skipped": f"disk {free:.0f} GB free < {gb + 8}"}
            continue
        mo = {"model_id": mid}
        outp = out_dir / "images" / tag
        outp.mkdir(parents = True, exist_ok = True)
        base = (f"cd '{sb}' && '{py}' test/run.py --model_id {mid} --prompt \"{PROMPT}\" --seed 42 "
                + " ".join(extra) + (f" --custom_op_path '{custom}'" if custom.exists() else ""))
        # Run 1: cold process (includes HF download on first use: timed separately below).
        dl = sh(f"HF_HUB_DISABLE_PROGRESS_BARS=1 '{py}' -c \"from huggingface_hub import snapshot_download as s; print('MPATH=' + s('{mid}'))\"",
                timeout = 1800, env = env, log = log)
        m = re.search(r"^MPATH=(.+)$", dl["out"], re.M)
        mo["download"] = {"rc": dl["rc"], "s": dl["s"], "path": m.group(1).strip() if m else None}
        mpath = mo["download"]["path"] if dl["rc"] == 0 else None
        if not mpath:
            runs[tag] = mo | {"skipped": "download failed or path not parsed"}
            continue
        mp = f" --model_path '{mpath}'" if mpath else ""
        with PowerSampler() as ps:
            r1 = sh(base + mp + f" --output_path '{outp}' 2>&1 | tail -120", timeout = tmo, env = env,
                    log = out_dir / "npu_logs" / f"{tag}_cold.log")
        mo["cold"] = {"rc": r1["rc"], "wall_s": r1["s"], "tail": r1["out"][-4000:], "power": ps.summary()}
        ok_img = sorted(str(p.name) for p in outp.glob("*.png"))
        mo["images"] = ok_img
        if r1["rc"] == 0 and time.time() < deadline:
            with PowerSampler() as ps:
                r2 = sh(base + mp + f" -p -pr 3 --no_excel --no_images --output_path '{outp}' 2>&1 | tail -150",
                        timeout = tmo, env = env, log = out_dir / "npu_logs" / f"{tag}_profile.log")
            mo["profile"] = {"rc": r2["rc"], "wall_s": r2["s"], "tail": r2["out"][-5000:], "power": ps.summary()}
        # Free the model before the next one.
        cache_dir = work / "hf" / "hub" / ("models--" + mid.replace("/", "--"))
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir, ignore_errors = True)
        runs[tag] = mo
        # The amdxdna watchdog latches at module init: record the kernel log line if one fired.
        mo["timeout_hint"] = bool(re.search(r"ERT_CMD_STATE_TIMEOUT|timeout", (r1["out"] or ""), re.I))
    obs["runs"] = runs
    return obs


IGPU_SCRIPT = r"""
import json, sys, time, torch
import diffusers
from diffusers import AutoPipelineForText2Image
repo, res, steps, out, prompt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
# Same ROCm settings Studio uses (diffusion_speed.py): MIOpen immediate mode, no cudnn benchmark.
try:
    torch.backends.miopen.immediate = True
except Exception:
    pass
torch.backends.cudnn.benchmark = False
r = {"torch": torch.__version__, "hip": torch.version.hip, "dev": torch.cuda.get_device_name(0),
     "diffusers": diffusers.__version__}
t = time.time()
pipe = AutoPipelineForText2Image.from_pretrained(repo, torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.set_progress_bar_config(disable=True)
r["load_s"] = time.time() - t
xl = "XL" in type(pipe).__name__
runs = []
for i in range(6):
    g = torch.Generator("cuda").manual_seed(42)
    torch.cuda.synchronize(); t0 = time.time()
    if xl:
        pe, _, ppe, _ = pipe.encode_prompt(prompt=prompt, device="cuda", do_classifier_free_guidance=False)
        kw = dict(prompt_embeds=pe, pooled_prompt_embeds=ppe)
    else:
        pe, _ = pipe.encode_prompt(prompt, "cuda", 1, False)
        kw = dict(prompt_embeds=pe)
    torch.cuda.synchronize(); t1 = time.time()
    lat = pipe(**kw, num_inference_steps=steps, guidance_scale=0.0, height=res, width=res,
               generator=g, output_type="latent").images
    torch.cuda.synchronize(); t2 = time.time()
    vae = pipe.vae
    if getattr(vae.config, "force_upcast", False) and vae.dtype == torch.float16:
        vae.to(torch.float32)
    img = vae.decode(lat.to(vae.dtype) / vae.config.scaling_factor).sample
    torch.cuda.synchronize(); t3 = time.time()
    runs.append({"te_s": t1 - t0, "denoise_s": t2 - t1, "vae_s": t3 - t2, "total_s": t3 - t0,
                 "vae_dtype": str(vae.dtype), "nan": bool(torch.isnan(img).any()),
                 "img_mean": float(img.float().mean()), "img_std": float(img.float().std())})
    print(json.dumps(runs[-1]), flush=True)
r["runs"] = runs
r["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
open(out, "w").write(json.dumps(r))
"""


def run_igpu(work: Path, out_dir: Path, log: Path) -> dict:
    obs: dict = {"python": sys.executable}
    env = dict(os.environ)
    env.pop("HF_TOKEN", None)
    env["HF_HOME"] = str(work / "hf_igpu")
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    py = sys.executable
    obs["torch"] = sh(f"'{py}' -c 'import torch;print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))'",
                      timeout = 300, env = env, log = log)
    obs["deps"] = sh(f"'{py}' -c 'import diffusers, transformers; print(diffusers.__version__, transformers.__version__)' || "
                     f"(uv pip install --python '{py}' diffusers || '{py}' -m pip install diffusers)", timeout = 1200, env = env, log = log)
    scr = work / "igpu_bench.py"
    scr.write_text(IGPU_SCRIPT, encoding = "utf-8")
    runs = {}
    for tag, repo, _cls, res, steps in IGPU_MODELS:
        o: dict = {"repo": repo}
        o["prefetch"] = sh(f"'{py}' -c \"from huggingface_hub import snapshot_download as s; s('{repo}', allow_patterns=['*.json','*.txt','*fp16.safetensors','tokenizer*/*'])\"",
                           timeout = 1800, env = env, log = log)
        res_p = work / f"igpu_{tag}.json"
        with PowerSampler() as ps:
            o["run"] = sh(f"'{py}' '{scr}' {repo} {res} {steps} '{res_p}' \"{PROMPT}\"", timeout = 1800, env = env,
                          log = out_dir / "npu_logs" / f"igpu_{tag}.log")
        o["power"] = ps.summary()
        try:
            o["result"] = json.loads(res_p.read_text(encoding = "utf-8"))
        except Exception as e:  # noqa: BLE001
            o["result_error"] = repr(e)
        runs[tag] = o
    obs["runs"] = runs
    return obs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--heavy-state", default = "head")
    ap.add_argument("--budget-min", type = float, default = 150)
    args = ap.parse_args()
    t0 = time.time()
    out_dir = args.out.parent / f"npu_{args.state}"
    out_dir.mkdir(parents = True, exist_ok = True)
    log = out_dir / "npu_logs" / "probe.log"
    obs: dict = {"state": args.state}
    try:
        obs["survey"] = survey(log)
    except Exception as e:  # noqa: BLE001
        obs["survey_error"] = f"{type(e).__name__}: {e}"
    if args.state != args.heavy_state:
        obs["heavy"] = "not run at this state (survey only; the heavy part ignores the checkout)"
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
        return 0
    work = Path(os.environ.get("AMD_CI_WORK", "/tmp")) / "npu"
    work.mkdir(parents = True, exist_ok = True)
    deadline = t0 + args.budget_min * 60
    heavy: dict = {}
    obs["heavy"] = heavy
    try:
        s = obs.get("survey", {})
        if not os.path.exists("/dev/accel/accel0"):
            heavy["stop"] = "no /dev/accel/accel0 on this runner: no NPU is exposed to the job"
        elif not s.get("accel_rw"):
            heavy["stop"] = "/dev/accel/accel0 exists but the runner user cannot open it read-write"
        else:
            xo, env = setup_xrt(work, log)
            heavy["xrt"] = xo
            ro, venv_env = install_rai(work, env, log)
            heavy["rai"] = ro
            if "VIRTUAL_ENV" in venv_env and ro.get("install", {}).get("rc") == 0:
                heavy["sandbox"] = run_sandbox(work, venv_env, out_dir, log, deadline)
            else:
                heavy["stop"] = "Ryzen AI install failed"
    except Exception as e:  # noqa: BLE001
        heavy["error"] = f"{type(e).__name__}: {e}"
    # iGPU arm runs whatever happened on the NPU side (same host, same job, after the NPU arm).
    try:
        heavy["igpu"] = run_igpu(work, out_dir, log)
    except Exception as e:  # noqa: BLE001
        heavy["igpu"] = {"error": f"{type(e).__name__}: {e}"}
    obs["wall_s"] = round(time.time() - t0, 1)
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    # Free disk (models / venv) so nothing outlives the job.
    shutil.rmtree(work, ignore_errors = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

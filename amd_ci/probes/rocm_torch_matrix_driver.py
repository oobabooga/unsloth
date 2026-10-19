"""For each torch build (arm): fresh venv, install the torch trio from its index, Unsloth on top
with torch overridden away, run rocm_torch_matrix_probe.py, record, delete the venv.

The first arm of each OS is the CONTROL: the build `install.sh` / `install.ps1` pick on main
today for gfx1151. Observes only; rocm_torch_matrix_judge.py decides.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

UNSLOTH = "git+https://github.com/unslothai/unsloth@355985f3e99f24f18f4c966ee2270a325c4774aa"
ZOO = "git+https://github.com/unslothai/unsloth-zoo@e782f78c6f47ec77d06bf764f5c9fd920cdd0861"
PYPI = "https://pypi.org/simple"
AMD_ARCH = "https://repo.amd.com/rocm/whl/gfx1151/"
AMD_MULTI = "https://repo.amd.com/rocm/whl-multi-arch/"
AMD_NEXT = "https://stable.repo.amd.com/rocm/whl-next/"
PT = "https://download.pytorch.org/whl/"

ARMS = {
    "linux": [
        ("control_amd_gfx1151_2.11_rocm7.13", AMD_ARCH,
         ["torch>=2.11.0,<2.12.0", "torchvision>=0.26.0,<0.27.0", "torchaudio>=2.11.0,<2.12.0"]),
        ("pytorch_rocm7.2_2.14.1", PT + "rocm7.2", ["torch==2.14.1+rocm7.2", "torchvision==0.29.1+rocm7.2"]),
        ("pytorch_rocm7.14_2.14.1", PT + "rocm7.14", ["torch==2.14.1+rocm7.14", "torchvision==0.29.1+rocm7.14"]),
        # whl-next pairs torchvision 0.29 only with rocm10.0, so 10.1's newest matched pair is torch 2.13.
        ("amd_whlnext_2.13.0_rocm10.1", AMD_NEXT,
         ["torch[device-gfx1151]==2.13.0+rocm10.1.0", "torchvision[device-gfx1151]==0.28.0+rocm10.1.0"]),
        ("amd_whlnext_2.14.0_rocm10.0", AMD_NEXT,
         ["torch[device-gfx1151]==2.14.0+rocm10.0.0", "torchvision[device-gfx1151]==0.29.0+rocm10.0.0"]),
        ("amd_multiarch_2.11.0_rocm7.14.0", AMD_MULTI,
         ["torch[device-gfx1151]==2.11.0+rocm7.14.0", "torchvision[device-gfx1151]==0.26.0+rocm7.14.0"]),
    ],
    "windows": [
        ("control_amd_multiarch_2.11.0_rocm7.14.0", AMD_MULTI,
         ["torch[device-gfx1151]==2.11.0+rocm7.14.0", "torchvision[device-gfx1151]==0.26.0+rocm7.14.0"]),
        # whl-next pairs torchvision 0.29 only with rocm10.0, so 10.1's newest matched pair is torch 2.13.
        ("amd_whlnext_2.13.0_rocm10.1", AMD_NEXT,
         ["torch[device-gfx1151]==2.13.0+rocm10.1.0", "torchvision[device-gfx1151]==0.28.0+rocm10.1.0"]),
        # No Windows torch 2.14.0+rocm10.0.0 wheel on whl-next.
    ],
}
OVERRIDES = ["torch", "torchvision", "torchaudio", "triton"]

ap = argparse.ArgumentParser()
ap.add_argument("--os", choices = sorted(ARMS), required = True)
ap.add_argument("--work", required = True)
ap.add_argument("--out-dir", required = True)
ap.add_argument("--only", default = "")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok = True)
here = os.path.dirname(os.path.abspath(__file__))
win = a.os == "windows"


def sh(cmd, log, timeout):
    print("+", " ".join(cmd), flush = True)
    t = time.time()
    with open(log, "a", encoding = "utf-8") as lf:
        lf.write("+ " + " ".join(cmd) + "\n")
        lf.flush()
        try:
            p = subprocess.run(cmd, stdout = lf, stderr = subprocess.STDOUT, timeout = timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = -999
    print(f"  rc={rc} in {time.time() - t:.0f}s", flush = True)
    return rc, round(time.time() - t, 1)


def tail(path, n = 4000):
    with open(path, encoding = "utf-8", errors = "replace") as f:
        return f.read()[-n:]


uv = shutil.which("uv")
if uv is None:
    # A system python without ensurepip (Debian python3 lacking python3-venv) cannot make the
    # boot venv, so try a user-site pip install of uv as the second route.
    boot = os.path.join(a.work, "boot")
    if subprocess.run([sys.executable, "-m", "venv", boot]).returncode == 0:
        bpy = os.path.join(boot, "Scripts" if win else "bin", "python.exe" if win else "python")
        subprocess.run([bpy, "-m", "pip", "install", "-q", "uv"], check = True)
        uv = os.path.join(boot, "Scripts" if win else "bin", "uv.exe" if win else "uv")
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--user", "uv"], check = True)
        uv = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
print("uv:", uv, subprocess.run([uv, "--version"], capture_output = True, text = True).stdout.strip(), flush = True)
ov = os.path.join(a.work, "overrides.txt")
with open(ov, "w", encoding = "utf-8") as f:
    f.write("".join(f"{p} ; sys_platform == 'never'\n" for p in OVERRIDES))
env_hf = os.path.join(a.work, "hf_home")
os.environ["HF_HOME"] = env_hf
os.environ["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"
os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)

summary = []
for name, index, specs in ARMS[a.os]:
    if a.only and name not in a.only.split(","):
        continue
    print(f"\n===== ARM {name} =====", flush = True)
    rec = {"arm": name, "index": index, "specs": specs, "control": name.startswith("control")}
    log = os.path.join(a.out_dir, f"{name}.install.log")
    venv = os.path.join(a.work, "venvs", name)
    py = os.path.join(venv, "Scripts" if win else "bin", "python.exe" if win else "python")
    rc, rec["venv_s"] = sh([uv, "venv", "--no-config", "--python", "3.12", "--seed", venv], log, 600)
    if rc != 0:
        rc, rec["venv_s"] = sh([uv, "venv", "--no-config", "--python", sys.executable, "--seed", venv], log, 600)
    rec["venv_rc"] = rc
    if rc == 0:
        rc, rec["torch_install_s"] = sh(
            [uv, "pip", "install", "--no-config", "--python", py, *specs, "--default-index", index], log, 2400)
        rec["torch_install_rc"] = rc
    if rc == 0:
        rc, rec["unsloth_install_s"] = sh(
            [uv, "pip", "install", "--no-config", "--python", py, f"unsloth[amd] @ {UNSLOTH}", f"unsloth_zoo @ {ZOO}",
             "--override", ov, "--default-index", PYPI], log, 2400)
        rec["unsloth_install_rc"] = rc
    if rc == 0 and win:
        # As studio/setup.ps1 Get-TritonWindowsSpec: Unsloth imports triton, and Studio installs
        # triton-windows paired with the torch minor.
        minor = int(subprocess.run([py, "-c", "import torch; print(torch.__version__.split('.')[1])"],
                                   capture_output = True, text = True).stdout.strip() or 0)
        spec = {13: "triton-windows>=3.7,<3.8", 14: "triton-windows>=3.8,<3.9"}.get(minor, "triton-windows<3.7")
        rc, rec["triton_install_s"] = sh(
            [uv, "pip", "install", "--no-config", "--python", py, spec, "--default-index", PYPI], log, 900)
        rec["triton_install_rc"] = rc
    if rc == 0:
        fr = subprocess.run([uv, "pip", "freeze", "--python", py], capture_output = True, text = True)
        rec["freeze"] = fr.stdout
        obs_path = os.path.join(a.out_dir, f"{name}.obs.json")
        work = os.path.join(a.work, "runs", name)
        os.makedirs(work, exist_ok = True)
        rc, rec["probe_s"] = sh(
            [py, "-u", os.path.join(here, "rocm_torch_matrix_probe.py"), "--out", obs_path,
             "--train-script", os.path.join(here, "rocm_torch_matrix_train.py"), "--work", work],
            os.path.join(a.out_dir, f"{name}.probe.log"), 4800)
        rec["probe_rc"] = rc
        if os.path.exists(obs_path):
            with open(obs_path, encoding = "utf-8") as f:
                rec["obs"] = json.load(f)
        shutil.rmtree(work, ignore_errors = True)
    else:
        rec["install_tail"] = tail(log)
    shutil.rmtree(venv, ignore_errors = True)
    summary.append(rec)
    with open(os.path.join(a.out_dir, "observations.json"), "w", encoding = "utf-8") as f:
        json.dump({"os": a.os, "arms": summary}, f, indent = 1)
print("driver done", flush = True)

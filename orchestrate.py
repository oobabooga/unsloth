"""Windows AMD: the installer's real _ensure_rocm_torch() on base vs head, with and without an only-binary policy."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

BASE = "5c04ec19305ecb5f0153c5b4d806a9d7e6e4422d"
HEAD = "1a0417e4d9d85f69d1596bf80588667303765b57"
WORK = Path(os.environ["RUNNER_TEMP"]) / "pr10902"
shutil.rmtree(WORK, ignore_errors=True)
WORK.mkdir(parents=True)
UV = shutil.which("uv")
print("uv:", UV, subprocess.run([UV, "--version"], capture_output=True, text=True).stdout.strip(), flush=True)


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
        raise SystemExit(f"failed: {cmd}")
    return r.stdout


for name, sha in (("base", BASE), ("head", HEAD)):
    d = WORK / name
    d.mkdir()
    sh(["git", "init", "-q", str(d)])
    sh(["git", "-C", str(d), "fetch", "-q", "--depth", "1", "https://github.com/unslothai/unsloth", sha])
    sh(["git", "-C", str(d), "checkout", "-q", "FETCH_HEAD"])

DRIVE = r'''
import os, sys, io, contextlib
sys.path.insert(0, os.path.join(sys.argv[1], "studio"))
import install_python_stack as ips
ips.USE_UV = True
print("IS_WINDOWS", ips.IS_WINDOWS, "gfx", ips._detect_windows_gfx_arch(), "index", ips._windows_rocm_index_url(ips._detect_windows_gfx_arch()), flush=True)
ips._ensure_rocm_torch()
print("ROCM_TORCH_INSTALLED_FLAG", ips._rocm_windows_torch_installed, flush=True)
'''
(WORK / "drive.py").write_text(DRIVE)
PROBE = r'''
import importlib.metadata as md
try:
    print("torch dist", md.version("torch"), "| rocm dist", md.version("rocm"))
except md.PackageNotFoundError as e:
    print("not installed:", e); raise SystemExit
import torch
print("torch", torch.__version__, "hip", getattr(torch.version, "hip", None), "gpu", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    a = torch.randn(2048, 2048, device="cuda"); torch.cuda.synchronize()
    print("matmul ok", bool((a @ a).abs().mean() > 0))
'''
(WORK / "probe.py").write_text(PROBE)

cache = WORK / "uvcache"
results = {}
for label, tree, policy in (("A", "base", ":all:"), ("B", "head", ":all:"), ("C", "head", None), ("D", "head", ":all:,rocm")):
    v = WORK / f"v_{label}"
    sh([UV, "venv", "-q", "-p", "3.12", str(v)])
    py = v / "Scripts" / "python.exe"
    sh([UV, "pip", "install", "-q", "--python", str(py), "pip==26.2.1"])
    if policy:
        (v / "pip.ini").write_text(f"[global]\nonly-binary = {policy}\n")
    appdata = WORK / f"appdata_{label}"
    appdata.mkdir()
    env = {k: val for k, val in os.environ.items() if not k.startswith(("PIP_", "UV_"))}
    env.update(APPDATA=str(appdata), UV_CACHE_DIR=str(cache), UNSLOTH_ROCM_GFX_ARCH="gfx1151", PYTHONIOENCODING="utf-8")
    print(f"\n===== {label} ({tree}, pip.ini only-binary={policy})", flush=True)
    print(sh([str(py), "-m", "pip", "config", "list"], env=env).strip() or "(no pip config)", flush=True)
    t = time.time()
    r = subprocess.run([str(py), str(WORK / "drive.py"), str(WORK / tree)], capture_output=True, text=True, env=env, timeout=3600)
    out = (r.stdout + r.stderr).splitlines()
    keep = [l for l in out if any(k in l for k in ("IS_WINDOWS", "installing torch", "Warning", "ROCM_TORCH_INSTALLED_FLAG", "no usable wheels", "Because", "rocm==", "error", "ERROR"))]
    print("\n".join(keep[-25:]), flush=True)
    print(f"driver exit {r.returncode} in {time.time() - t:.0f}s", flush=True)
    p = subprocess.run([str(py), str(WORK / "probe.py")], capture_output=True, text=True, env=env, timeout=600)
    print((p.stdout + p.stderr).strip()[-1500:], flush=True)
    results[label] = {"driver_exit": r.returncode, "flag": any("ROCM_TORCH_INSTALLED_FLAG True" in l for l in out), "probe": p.stdout.strip().splitlines()}

print("\nSUMMARY " + json.dumps(results, indent=1))

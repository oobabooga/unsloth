#!/usr/bin/env python3
"""Make an AMD runner look like an NVIDIA one to the code under test.

Much of Unsloth branches on the vendor before it does anything: `torch.version.hip`,
`torch.version.cuda`, the device name, the compute capability, and `nvidia-smi`
output. On this pool every one of those says AMD, so the NVIDIA branches (and on
Windows, the only branches most users ever take) are unreachable. This builds a
directory that, once on PYTHONPATH and PATH, flips them:

- `sitecustomize.py`: a post-import hook on `torch` setting `torch.version.hip = None`,
  `torch.version.cuda`, and wrapping `torch.cuda.get_device_name`,
  `get_device_capability` and `get_device_properties` (name / major / minor
  overridden, every other attribute read from the real device). `is_available` and
  `device_count` stay real, and tensors still run on the real HIP device.
- `nvidia-smi` (POSIX sh) and `nvidia-smi.cmd` / `nvidia-smi.bat` (Windows), both
  calling `nvidia_smi_fake.py`: `-L`, `--query-gpu=... --format=csv[,noheader][,nounits]`
  and a bare table. Anything else exits 2 naming the flag, so an unhandled call is
  visible instead of parsed as an empty answer.

**For wiring only.** Vendor detection, nvidia-smi parsing, capability gates, install
branch selection. NVIDIA kernels, CUDA-only wheels (bitsandbytes CUDA build,
flash-attn, xformers, vLLM CUDA), NVML, and every performance number are NOT reached:
code that goes past the vendor check runs HIP underneath or fails.

<critical>Activating this sets AMD_CI_SPOOFED_VENDOR=nvidia, and capability.py then
keeps `nvidia` UNMET and names the spoof in "Not tested here", reading the real
vendor from the originals this hook stashes. Do not set the marker by hand.</critical>

Limits:
- Windows `CreateProcess` appends only `.exe`, so `subprocess.run(["nvidia-smi"])`
  without `shell=True` does not find the `.cmd` shim; `shutil.which("nvidia-smi")`
  (PATHEXT) and `shell=True` do. Code that resolves the path first is reached.
- The hook fires on `import torch`. If torch is already imported when sitecustomize
  runs (it cannot be under a normal interpreter start) the patch is applied at once.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ENV_MARKER = "AMD_CI_SPOOFED_VENDOR"
CONFIG_NAME = "nvidia_identity.json"

DEFAULTS = {
    "name": "NVIDIA GeForce RTX 4090",
    "capability": "8.9",
    "vram_mib": 24564,
    "driver": "580.65.06",
    "cuda": "12.8",
}

SITECUSTOMIZE = r'''# Written by amd_ci/lib/nvidia_identity.py. Presents torch as a CUDA build on an
# NVIDIA card; compute still runs on the real device. Wiring tests only.
import importlib.abc
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "nvidia_identity.json"), encoding = "utf-8") as _fh:
    _CFG = json.load(_fh)
_MAJOR, _MINOR = (int(x) for x in str(_CFG["capability"]).split(".")[:2])


class _Props:
    """The real device properties with name / major / minor overridden."""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def __getattr__(self, attr):
        if attr == "name":
            return _CFG["name"]
        if attr == "major":
            return _MAJOR
        if attr == "minor":
            return _MINOR
        if attr == "gcnArchName":
            return ""
        if attr == "total_memory" and self._real is None:
            return int(_CFG["vram_mib"]) * 1024 * 1024
        if self._real is None:
            raise AttributeError(attr)
        return getattr(self._real, attr)

    def __repr__(self):
        return (f"_CudaDeviceProperties(name='{_CFG['name']}', major={_MAJOR}, "
                f"minor={_MINOR}, spoofed=True)")


def _patch(torch):
    if getattr(torch, "_amd_ci_nvidia_identity", False):
        return
    v = torch.version
    v._amd_ci_real_hip = getattr(v, "hip", None)
    v._amd_ci_real_cuda = getattr(v, "cuda", None)
    v.hip = None
    v.cuda = str(_CFG["cuda"])
    cuda = torch.cuda
    real_props = cuda.get_device_properties
    real_name = cuda.get_device_name
    cuda._amd_ci_real_get_device_properties = real_props
    cuda._amd_ci_real_get_device_name = real_name

    def get_device_properties(device = None):
        try:
            real = real_props(device) if device is not None else real_props(0)
        except Exception:  # no device (CPU torch): identity only
            real = None
        return _Props(real)

    def get_device_name(device = None):
        return _CFG["name"]

    def get_device_capability(device = None):
        return (_MAJOR, _MINOR)

    cuda.get_device_properties = get_device_properties
    cuda.get_device_name = get_device_name
    cuda.get_device_capability = get_device_capability
    torch._amd_ci_nvidia_identity = True


class _Loader(importlib.abc.Loader):
    def __init__(self, real):
        self._real = real

    def create_module(self, spec):
        return self._real.create_module(spec)

    def exec_module(self, module):
        self._real.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target = None):
        if name != "torch":
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader)
        return spec


if "torch" in sys.modules:
    _patch(sys.modules["torch"])
else:
    sys.meta_path.insert(0, _Finder())


def _chain():
    """Run the sitecustomize this one shadows, if any, as if it had been found first."""
    for entry in sys.path:
        try:
            if os.path.abspath(entry or os.getcwd()) == _HERE:
                continue
        except Exception:
            continue
        cand = os.path.join(entry or os.getcwd(), "sitecustomize.py")
        if os.path.isfile(cand):
            spec = importlib.util.spec_from_file_location("_amd_ci_chained_sitecustomize", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return


_chain()
'''

FAKE_SMI = r'''# Written by amd_ci/lib/nvidia_identity.py: a fake nvidia-smi for wiring tests.
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "nvidia_identity.json"), encoding = "utf-8") as fh:
    CFG = json.load(fh)
MIB = int(CFG["vram_mib"])
USED = 1
UUID = "GPU-00000000-0000-0000-0000-00000000a0d0"
FIELDS = {
    "index": ("0", "0"),
    "name": (CFG["name"], CFG["name"]),
    "gpu_name": (CFG["name"], CFG["name"]),
    "uuid": (UUID, UUID),
    "gpu_uuid": (UUID, UUID),
    "memory.total": (f"{MIB} MiB", str(MIB)),
    "memory.used": (f"{USED} MiB", str(USED)),
    "memory.free": (f"{MIB - USED} MiB", str(MIB - USED)),
    "utilization.gpu": ("0 %", "0"),
    "driver_version": (CFG["driver"], CFG["driver"]),
    "compute_cap": (str(CFG["capability"]), str(CFG["capability"])),
    "pci.bus_id": ("00000000:01:00.0", "00000000:01:00.0"),
    "count": ("1", "1"),
}
HEADERS = {"memory.total": "memory.total [MiB]", "memory.used": "memory.used [MiB]",
           "memory.free": "memory.free [MiB]", "utilization.gpu": "utilization.gpu [%]"}


def die(msg):
    sys.stderr.write(f"nvidia-smi (amd_ci spoof): {msg}\n")
    sys.exit(2)


def main(argv):
    if not argv:
        print(f"| NVIDIA-SMI {CFG['driver']}    Driver Version: {CFG['driver']}    CUDA Version: {CFG['cuda']} |")
        print(f"|   0  {CFG['name']}    {USED}MiB / {MIB}MiB |")
        return 0
    if argv in (["-L"], ["--list-gpus"]):
        print(f"GPU 0: {CFG['name']} (UUID: {UUID})")
        return 0
    query, fmt, skip = None, None, False
    for a in argv:
        if skip:  # the value of -i / --id: one GPU, so any index is GPU 0
            skip = False
        elif a.startswith("--query-gpu="):
            query = a.split("=", 1)[1]
        elif a.startswith("--format="):
            fmt = a.split("=", 1)[1]
        elif a in ("-i", "--id"):
            skip = True
        elif a.startswith("--id="):
            continue
        else:
            die(f"unsupported argument {a!r}")
    if query is None:
        die("only -L, --query-gpu and a bare call are spoofed")
    opts = set((fmt or "").split(","))
    if "csv" not in opts:
        die(f"unsupported --format {fmt!r}; only csv is spoofed")
    names = [q.strip() for q in query.split(",") if q.strip()]
    unknown = [n for n in names if n not in FIELDS]
    if unknown:
        die(f"unsupported --query-gpu field(s) {unknown}")
    nounits = "nounits" in opts
    if "noheader" not in opts:
        print(", ".join(HEADERS.get(n, n) if not nounits else n for n in names))
    print(", ".join(FIELDS[n][1 if nounits else 0] for n in names))
    return 0


sys.exit(main(sys.argv[1:]))
'''


def build(dest_dir: Path, name: str = DEFAULTS["name"], capability: str = DEFAULTS["capability"],
          vram_mib: int = DEFAULTS["vram_mib"], driver: str = DEFAULTS["driver"],
          cuda: str = DEFAULTS["cuda"], python: str | None = None) -> Path:
    """Write the spoof into dest_dir and return it."""
    dest = Path(dest_dir)
    dest.mkdir(parents = True, exist_ok = True)
    major, _, minor = str(capability).partition(".")
    if not (major.isdigit() and minor.isdigit()):
        raise ValueError(f"--capability must look like 8.9, got {capability!r}")
    cfg = {"name": name, "capability": str(capability), "vram_mib": int(vram_mib),
           "driver": driver, "cuda": str(cuda)}
    (dest / CONFIG_NAME).write_text(json.dumps(cfg, indent = 2), encoding = "utf-8")
    (dest / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding = "utf-8")
    (dest / "nvidia_smi_fake.py").write_text(FAKE_SMI, encoding = "utf-8")
    py = python or sys.executable
    sh = dest / "nvidia-smi"
    sh.write_text(f'#!/bin/sh\nexec "{py}" "$(dirname "$0")/nvidia_smi_fake.py" "$@"\n',
                  encoding = "utf-8", newline = "\n")
    sh.chmod(0o755)
    cmd = f'@echo off\r\n"{py}" "%~dp0nvidia_smi_fake.py" %*\r\nexit /b %ERRORLEVEL%\r\n'
    for ext in ("cmd", "bat"):
        (dest / f"nvidia-smi.{ext}").write_text(cmd, encoding = "utf-8", newline = "")
    return dest


def env_for(spoof_dir: Path, base: dict | None = None) -> dict:
    """An environment with the spoof active."""
    env = dict(os.environ if base is None else base)
    d = str(spoof_dir)
    for key in ("PYTHONPATH", "PATH"):
        cur = env.get(key)
        env[key] = f"{d}{os.pathsep}{cur}" if cur else d
    env[ENV_MARKER] = "nvidia"
    return env


def spoofed_vendor() -> str:
    """'nvidia' when the spoof is active in THIS process, else ''."""
    return os.environ.get(ENV_MARKER, "").strip().lower()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description = __doc__.split("\n")[0])
    ap.add_argument("--build-into", type = Path, required = True)
    ap.add_argument("--name", default = DEFAULTS["name"])
    ap.add_argument("--capability", default = DEFAULTS["capability"])
    ap.add_argument("--vram-mib", type = int, default = DEFAULTS["vram_mib"])
    ap.add_argument("--driver", default = DEFAULTS["driver"])
    ap.add_argument("--cuda", default = DEFAULTS["cuda"])
    ap.add_argument("--python", default = None,
                    help = "interpreter the nvidia-smi shims call (default: this one)")
    ap.add_argument("--github-env", action = "store_true",
                    help = "activate for later steps via $GITHUB_ENV and $GITHUB_PATH")
    args = ap.parse_args()

    dest = build(args.build_into, args.name, args.capability, args.vram_mib, args.driver,
                 args.cuda, args.python)
    print(f"built {dest}: {args.name}, sm_{args.capability.replace('.', '')}, CUDA {args.cuda}")

    if args.github_env:
        gh_env, gh_path = os.environ.get("GITHUB_ENV"), os.environ.get("GITHUB_PATH")
        if not gh_env or not gh_path:
            raise SystemExit("--github-env outside GitHub Actions: $GITHUB_ENV / $GITHUB_PATH unset")
        pre = os.environ.get("PYTHONPATH")
        lines = [
            f"PYTHONPATH={dest}{os.pathsep}{pre}" if pre else f"PYTHONPATH={dest}",
            # Read by capability.py, which then keeps `nvidia` UNMET and names the spoof.
            f"{ENV_MARKER}=nvidia",
        ]
        with open(gh_env, "a", encoding = "utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        # GITHUB_PATH prepends for later steps on both OSes; writing PATH= into
        # GITHUB_ENV would freeze this step's PATH, runner-added entries included.
        with open(gh_path, "a", encoding = "utf-8") as fh:
            fh.write(f"{dest}\n")
        print(f"activated: torch and nvidia-smi report {args.name}; {ENV_MARKER} set so the "
              f"verdict declares it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

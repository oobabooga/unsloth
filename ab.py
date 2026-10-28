"""RD-825 A/B: run Studio's own sd.cpp accelerator choice, install and spawn path from one checkout.

usage: python ab.py <label> <checkout> <out.json>
Environment decides the scenario (HIP_PATH, UNSLOTH_SD_CPP_WINDOWS_ROCM, UNSLOTH_STUDIO_HOME, RD825_MODEL).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

label, checkout, out_json = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])

if os.name == "nt":
    import ctypes

    # Inherited by every child: a missing DLL becomes exit 0xC0000135 instead of a modal box on a headless runner.
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)

sys.path.insert(0, str(checkout / "studio" / "backend"))
from core.inference import sd_cpp_backend as bk  # noqa: E402
from core.inference.sd_cpp_engine import managed_install_root, runtime_env  # noqa: E402

res: dict = {
    "label": label,
    "checkout_head": subprocess.run(
        ["git", "-C", str(checkout), "log", "-1", "--format=%h %s"], capture_output = True, text = True
    ).stdout.strip(),
    "checkout_dirty": subprocess.run(
        ["git", "-C", str(checkout), "diff", "--stat"], capture_output = True, text = True
    ).stdout.strip(),
    "HIP_PATH": os.environ.get("HIP_PATH"),
    "UNSLOTH_SD_CPP_WINDOWS_ROCM": os.environ.get("UNSLOTH_SD_CPP_WINDOWS_ROCM"),
    "hipblas_on_PATH": shutil.which("hipblas.dll"),
}
res["hip_dlls_missing"] = bk.windows_hip_runtime_missing_dlls()
acc = bk.preferred_accelerator("rocm")
res["preferred_accelerator(rocm)"] = acc
t0 = time.time()
cli = bk.ensure_sd_cpp_binary(accelerator = acc)
res["ensure_seconds"] = round(time.time() - t0, 1)
res["sd_cli"] = cli
rec = bk._installer_module().read_install_record(managed_install_root())
res["install_record"] = {k: rec.get(k) for k in ("accelerator", "asset", "tag", "repo")}
res["rocm_failure_recorded"] = bk.accelerator_runtime_failed("rocm")

model = os.environ.get("RD825_MODEL")
runs = []
if cli and model:
    env = runtime_env(cli)
    res["child_PATH_tail"] = env.get("PATH", "").split(os.pathsep)[-2:]
    for i in range(int(os.environ.get("RD825_RUNS", "2"))):
        png = out_json.with_name(f"{label}-{i}.png")
        cmd = [
            cli, "-m", model, "-p", "a red apple on a wooden table", "--steps", "4", "--cfg-scale", "1",
            "-W", "512", "-H", "512", "--seed", "42", "-o", str(png),
        ]
        t = time.time()
        try:
            p = subprocess.run(
                cmd, env = env, capture_output = True, text = True, errors = "replace", timeout = 1200
            )
            rc, text = p.returncode, (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired as exc:
            rc, text = "timeout", str(exc.stdout or "") + str(exc.stderr or "")
        code = rc + (1 << 32) if isinstance(rc, int) and rc < 0 else rc
        keep = [
            ln.strip()
            for ln in text.splitlines()
            if re.search(r"rocm|hip|vulkan|gfx|radeon|device|backend|sampling completed|generate_image completed|error|fail", ln, re.I)
        ]
        runs.append(
            {
                "exit": f"0x{code:08X}" if isinstance(code, int) else code,
                "seconds": round(time.time() - t, 1),
                "png_bytes": png.stat().st_size if png.is_file() else None,
                "log": keep[:40],
                "tail": text.splitlines()[-6:],
            }
        )
res["generations"] = runs
out_json.write_text(json.dumps(res, indent = 1), encoding = "utf-8")
print(json.dumps(res, indent = 1))

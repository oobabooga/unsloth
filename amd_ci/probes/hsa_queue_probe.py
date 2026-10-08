#!/usr/bin/env python3
"""Probe: can this llama.cpp ROCm bundle create its first HIP queue, with the real
KFD topology and with max_waves_per_simd spoofed (unslothai/unsloth#12205)?

The "checkout" is an extracted llama.cpp bundle directory whose
libhsa-runtime64.so.1 is the variable under test. Each state is run twice through
the same LD_PRELOAD shim (amd_ci/i12205/topo_spoof.c): once inert (real topology)
and once with AMD_CI_SPOOF_WAVES set, which makes libhsakmt read a different
max_waves_per_simd than amdkfd uses. That is the userspace/KFD disagreement a GPU
whose firmware reports 20 waves/SIMD has natively.

Observes only: return codes, the HIP error signature, GPU offload lines, the
generated text, which libhsa-runtime64 the dynamic loader actually initialised,
and whether the shim rewrote anything. Judging is criteria/hsa_queue_create.py.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

PROMPT = "Once upon a time"


def sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def kfd_gpu_nodes() -> list[dict]:
    """KFD topology as the kernel reports it, read directly (no shim in this process)."""
    nodes = []
    for props in sorted(glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties")):
        vals: dict[str, int] = {}
        try:
            for line in Path(props).read_text(encoding = "utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1].isdigit():
                    vals[parts[0]] = int(parts[1])
        except OSError:
            continue
        if vals.get("simd_count", 0) > 0:
            keep = ("simd_count", "simd_per_cu", "max_waves_per_simd", "array_count",
                    "simd_arrays_per_engine", "cu_per_simd_array", "gfx_target_version",
                    "num_xcc", "cwsr_size", "ctl_stack_size")
            nodes.append({"node": props.split("/")[-2], **{k: vals[k] for k in keep if k in vals}})
    return nodes


# tag -> (spoofed max_waves_per_simd or None, HSA_USE_SVM value or None).
# libhsakmt puts the CWSR area in an SVM range when the kernel supports it; amdkfd then
# only checks that the range COVERS the expected size (kfd_queue_buffer_svm_get), so an
# oversized area passes there. HSA_USE_SVM=0 forces the BO path, whose check is exact.
MODES = {
    "real": (None, None),
    "real_nosvm": (None, "0"),
    "spoof20": ("20", None),
    "spoof20_nosvm": ("20", "0"),
    "spoof8": ("8", None),
}


def run_once(bundle: Path, model: Path, shim: Path, work: Path, tag: str,
             spoof: str | None, use_svm: str | None, timeout: int) -> dict:
    spoof_log = work / f"spoof_{tag}.log"
    ld_debug = work / f"lddebug_{tag}"
    for stale in [spoof_log, *work.glob(f"lddebug_{tag}.*")]:
        stale.unlink(missing_ok = True)
    env = {k: v for k, v in os.environ.items()
           if k not in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                        "HSA_OVERRIDE_GFX_VERSION", "AMD_CI_SPOOF_WAVES", "HSA_USE_SVM")}
    env.update({
        "LD_LIBRARY_PATH": str(bundle),
        "LD_PRELOAD": str(shim),  # loaded in BOTH modes; only the env var differs
        "AMD_CI_SPOOF_LOG": str(spoof_log),
        "LD_DEBUG": "libs",
        "LD_DEBUG_OUTPUT": str(ld_debug),
    })
    if spoof is not None:
        env["AMD_CI_SPOOF_WAVES"] = spoof
    if use_svm is not None:
        env["HSA_USE_SVM"] = use_svm
    # -v: this build logs "offloaded N/N layers to GPU" only above the default verbosity
    cmd = [str(bundle / "llama-completion"), "-m", str(model), "-p", PROMPT, "-n", "24",
           "-ngl", "99", "--temp", "0", "--seed", "1", "-c", "256", "-v"]
    res: dict = {"cmd": " ".join(cmd), "spoof_waves": spoof, "hsa_use_svm": use_svm}
    try:
        p = subprocess.run(cmd, env = env, stdin = subprocess.DEVNULL, capture_output = True,
                           text = True, errors = "replace", timeout = timeout)
        res["rc"], out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        res["rc"], out, err = "timeout", (e.stdout or ""), (e.stderr or "")
        out = out.decode(errors = "replace") if isinstance(out, bytes) else out
        err = err.decode(errors = "replace") if isinstance(err, bytes) else err
    both = f"{out}\n{err}"
    res["stdout"] = out[-2000:]
    res["stderr_tail"] = err[-4000:]
    res["queue_create_oom"] = bool(re.search(r"ROCm error: out of memory", both)) and \
        bool(re.search(r"hipStreamCreate", both))
    res["any_rocm_error"] = re.findall(r"ROCm error: [^\n]+", both)[:3]
    # "offloaded N/N layers to GPU" is printed even when ROCm found no device and every
    # layer went to the CPU, so placement is read from the per-layer lines and buffers.
    m = re.search(r"offloaded (\d+)/(\d+) layers to GPU", both)
    res["offloaded_line"] = [int(m.group(1)), int(m.group(2))] if m else None
    res["layers_rocm"] = len(re.findall(r"layer\s+\d+ assigned to device ROCm\d+", both))
    res["layers_cpu"] = len(re.findall(r"layer\s+\d+ assigned to device CPU", both))
    res["rocm_model_buffer_mib"] = sum(
        float(x) for x in re.findall(r"ROCm\d+ model buffer size\s*=\s*([\d.]+) MiB", both))
    res["rocm_init_failed"] = "failed to initialize ROCm" in both
    # generated continuation (stdout echoes the prompt first)
    res["generated"] = out.split(PROMPT, 1)[1].strip()[:400] if PROMPT in out else ""
    # which libhsa-runtime64 the loader initialised (LD_DEBUG writes <prefix>.<pid>)
    loaded = set()
    for f in work.glob(f"lddebug_{tag}.*"):
        for line in f.read_text(encoding = "utf-8", errors = "replace").splitlines():
            if "calling init:" in line and "libhsa-runtime64" in line:
                loaded.add(line.split("calling init:", 1)[1].strip())
    res["hsa_loaded"] = sorted(loaded)
    res["spoof_rewrites"] = spoof_log.read_text(encoding = "utf-8").splitlines()[:8] \
        if spoof_log.is_file() else []
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--model", required = True, type = Path)
    ap.add_argument("--shim", required = True, type = Path)
    ap.add_argument("--timeout", type = int, default = 240)
    args = ap.parse_args()

    bundle = args.checkout.resolve()
    work = args.out.parent / f"work_{args.state}"
    work.mkdir(parents = True, exist_ok = True)
    obs: dict = {
        "state": args.state,
        "bundle": str(bundle),
        "hsa_sha256": sha256(bundle / "libhsa-runtime64.so.1"),
        "model_sha256": sha256(args.model),
        "kfd_gpu_nodes": kfd_gpu_nodes(),
    }
    for tag, (spoof, use_svm) in MODES.items():
        obs[tag] = run_once(bundle, args.model, args.shim, work, tag, spoof, use_svm, args.timeout)
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

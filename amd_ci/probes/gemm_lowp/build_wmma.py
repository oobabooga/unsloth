"""Find a HIP compiler for gfx1151 and build wmma_kernels.hip once per KIND into --lib-dir.

Observes only: records every toolchain candidate (host hipcc, /opt/rocm, the venv's rocm-sdk-core / rocm-sdk-devel
packages) and the outcome of each build in --out JSON. Never prints JSON to stdout.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import time

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = {0: "f16", 1: "bf16", 2: "iu8", 3: "iu4", 4: "w4a16"}


def run(cmd, timeout=600, env=None):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return {"cmd": " ".join(cmd), "rc": p.returncode, "s": round(time.time() - t0, 2),
                "stdout": p.stdout[-3000:], "stderr": p.stderr[-6000:]}
    except Exception as exc:  # noqa: BLE001
        return {"cmd": " ".join(cmd), "rc": None, "error": repr(exc)}


def candidates(python):
    """(label, root) pairs that look like a ROCm toolchain root."""
    out = []
    purelib = subprocess.run([python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                             capture_output=True, text=True).stdout.strip()
    for name in ("_rocm_sdk_devel", "_rocm_sdk_core"):
        p = os.path.join(purelib, name)
        if os.path.isdir(p):
            out.append((f"venv:{name}", p))
    rocm_sdk = os.path.join(os.path.dirname(python), "rocm-sdk")
    if os.path.exists(rocm_sdk):
        r = subprocess.run([rocm_sdk, "path", "--root"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            out.append(("rocm-sdk path --root", r.stdout.strip()))
    for p in sorted(glob.glob("/opt/rocm*")):
        out.append((f"host:{p}", p))
    hipcc = shutil.which("hipcc")
    if hipcc:
        out.append(("host:PATH hipcc", os.path.dirname(os.path.dirname(os.path.realpath(hipcc)))))
    seen, uniq = set(), []
    for label, root in out:
        if root not in seen:
            seen.add(root)
            uniq.append((label, root))
    return uniq, purelib


def describe(root):
    d = {"root": root}
    for rel in ("bin/hipcc", "lib/llvm/bin/clang++", "llvm/bin/clang++", "bin/amdclang++",
                "lib/llvm/amdgcn/bitcode/ockl.bc", "amdgcn/bitcode/ockl.bc", "include/hip/hip_runtime.h",
                "lib/libamdhip64.so", "lib/libamdhip64.so.7"):
        d[rel] = os.path.exists(os.path.join(root, rel))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--lib-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.lib_dir, exist_ok=True)
    src = os.path.join(HERE, "wmma_kernels.hip")
    cands, purelib = candidates(a.python)
    res = {"purelib": purelib, "host_hipcc_on_path": shutil.which("hipcc"),
           "candidates": [dict(label=l, **describe(r)) for l, r in cands], "builds": {}, "asm": {}}
    if shutil.which("hipcc"):
        res["host_hipcc_version"] = run(["hipcc", "--version"], timeout=60)
    chosen = None
    for label, root in cands:
        clang = None
        for rel in ("lib/llvm/bin/clang++", "llvm/bin/clang++"):
            if os.path.exists(os.path.join(root, rel)):
                clang = os.path.join(root, rel)
                break
        bc = None
        for rel in ("lib/llvm/amdgcn/bitcode", "amdgcn/bitcode"):
            if os.path.exists(os.path.join(root, rel, "ockl.bc")):
                bc = os.path.join(root, rel)
                break
        hiplib = None
        for rel in ("lib/libamdhip64.so.7", "lib/libamdhip64.so"):
            if os.path.exists(os.path.join(root, rel)):
                hiplib = os.path.join(root, rel)
                break
        if clang and bc and hiplib and os.path.exists(os.path.join(root, "include/hip/hip_runtime.h")):
            chosen = dict(label=label, root=root, clang=clang, bitcode=bc, hiplib=hiplib)
            break
    res["chosen"] = chosen
    if chosen is None:
        res["error"] = "no usable clang++ + device bitcode + HIP headers + libamdhip64 found"
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
        return
    res["clang_version"] = run([chosen["clang"], "--version"], timeout=60)
    common = [chosen["clang"], "--offload-arch=gfx1151", "-O3", "-std=c++17", f"--rocm-path={chosen['root']}",
              f"--rocm-device-lib-path={chosen['bitcode']}", "-Wno-pass-failed"]
    for k, name in KINDS.items():
        so = os.path.join(a.lib_dir, f"libwm_{name}.so")
        r = run(common + [f"-DKIND={k}", "-fPIC", "-shared", "-no-hip-rt", "-o", so, "-x", "hip", src,
                          "-x", "none", chosen["hiplib"]])
        r["so"] = so if r.get("rc") == 0 and os.path.exists(so) else None
        res["builds"][name] = r
        asm = os.path.join(a.lib_dir, f"wm_{name}.s")
        ra = run(common + [f"-DKIND={k}", "--offload-device-only", "-S", "-o", asm, "-x", "hip", src])
        info = {"rc": ra.get("rc")}
        if ra.get("rc") == 0 and os.path.exists(asm):
            text = open(asm, encoding="utf-8", errors="replace").read()
            import re
            info["wmma_ops"] = {m: text.count(m) for m in sorted(set(re.findall(r"v_wmma_[a-z0-9_]+", text)))}
            info["vgpr"] = re.findall(r"\.vgpr_count:\s+(\d+)", text)
            info["spill"] = re.findall(r"\.vgpr_spill_count:\s+(\d+)", text)
            info["names"] = re.findall(r"\.name:\s+(_Z\S+)", text)
        else:
            info["stderr"] = ra.get("stderr", "")[-2000:]
        res["asm"][name] = info
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

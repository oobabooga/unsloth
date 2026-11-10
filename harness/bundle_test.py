#!/usr/bin/env python3
"""Run a prebuilt llama.cpp bundle on this machine and report what it does.

  bundle_test.py --kind rocm|cpu --new <archive> [--old <archive-or-url>] --model <gguf-or-url> --work <dir>

Checks the new bundle and, for comparison, the published one. Only the new bundle can fail the run.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

WIN = os.name == "nt"
EXE = ".exe" if WIN else ""
PROMPT = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,"


def fetch(src: str, dest: Path) -> Path:
    if not re.match(r"^https?://", src):
        return Path(src)
    if not dest.exists():
        print(f"downloading {src}", flush=True)
        req = urllib.request.Request(src, headers={"User-Agent": "bundle-test"})
        with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    return dest


def extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="fully_trusted")


def tree_stats(root: Path) -> dict:
    files = links = size = 0
    for p in root.rglob("*"):
        if p.is_symlink():
            links += 1
        elif p.is_file():
            files += 1
            size += p.stat().st_size
    return {"files": files, "links": links, "bytes": size}


def run(cmd: list[str], cwd: Path, timeout: int = 900) -> tuple[int, str]:
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env["LC_ALL"] = "C"
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        return r.returncode, r.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as e:
        return -999, (e.stdout or b"").decode("utf-8", "replace") + "\n[timed out]"
    except OSError as e:
        return -998, f"[could not start: {e}]"


def elf_rpaths(root: Path) -> dict[str, str]:
    """RUNPATH/RPATH of every regular ELF file at the bundle root (Linux only)."""
    out = {}
    if WIN or not shutil.which("readelf"):
        return out
    for p in sorted(root.iterdir()):
        if p.is_symlink() or not p.is_file():
            continue
        with open(p, "rb") as f:
            if f.read(4) != b"\x7fELF":
                continue
        d = subprocess.run(["readelf", "-d", str(p)], capture_output=True, text=True).stdout
        m = re.search(r"\((?:RUNPATH|RPATH)\).*\[(.*)\]", d)
        out[p.name] = m.group(1) if m else ""
    return out


def check(label: str, kind: str, archive: Path, model: Path, work: Path, elsewhere: Path, model2: Path | None = None) -> list[str]:
    problems: list[str] = []
    root = work / label
    if root.exists():
        shutil.rmtree(root)
    extract(archive, root)
    st = tree_stats(root)
    print(f"\n===== {label}: {archive.name}")
    print(f"archive {archive.stat().st_size / 1e6:.1f} MB; unpacked {st['bytes'] / 1e6:.1f} MB in {st['files']} files and {st['links']} links")

    names = {p.name for p in root.iterdir()}
    for want in ["LICENSE"] + (["libomp.dll", "LICENSE-LLVM-OpenMP"] if (WIN and kind == "cpu") else []):
        present = want in names
        print(f"contains {want}: {present}")
        if not present:
            problems.append(f"{want} is missing")
    omp = sorted(n for n in names if n.lower().startswith("libomp"))
    if omp:
        print(f"OpenMP runtime files: {omp}")
    if WIN and kind == "cpu" and any(n.lower().startswith("libomp140") for n in names):
        problems.append("the Visual Studio libomp140 runtime is still in the bundle")

    rp = elf_rpaths(root)
    if rp:
        bad = {k: v for k, v in rp.items() if v != "$ORIGIN"}
        print(f"ELF files: {len(rp)}; with a search path other than $ORIGIN: {len(bad)}")
        for k, v in bad.items():
            print(f"  {k}: {v!r}")
            problems.append(f"{k} search path is {v!r}")
        dangling = [p.name for p in root.iterdir() if p.is_symlink() and not p.exists()]
        if dangling:
            problems.append(f"dangling links: {dangling}")

    # Started from another directory: a binary that only loads from the bundle directory fails here.
    for tool, args in (("llama", ["--version"]), ("ggml-rpc-server", ["--help"]), ("llama-server", ["--version"])):
        exe = root / f"{tool}{EXE}"
        if not exe.exists():
            print(f"{tool}: not in the bundle")
            continue
        rc, out = run([str(exe), *args], elsewhere, timeout=120)
        ok = rc == 0 or (tool == "ggml-rpc-server" and "Usage" in out)
        first = next((ln for ln in out.splitlines() if ln.strip() and "load_backend" not in ln and "ggml_" not in ln), "")
        print(f"{tool} {' '.join(args)}: rc={rc} {'ok' if ok else 'FAILED'} | {first[:160]}")
        if not ok:
            problems.append(f"{tool} does not start from another directory (rc={rc}): {out.strip().splitlines()[-1][:200] if out.strip() else ''}")

    if kind == "static":
        return problems
    rc, out = run([str(root / f"llama-server{EXE}"), "--list-devices"], elsewhere, timeout=180)
    devs = [ln.strip() for ln in out.splitlines() if re.search(r"^\s*(ROCm|CUDA|Vulkan|HIP)\d*:", ln)]
    print(f"--list-devices rc={rc}: {devs if devs else out.strip().splitlines()[-3:]}")

    if kind == "rocm" and not WIN:
        env = dict(os.environ); env.pop("LD_LIBRARY_PATH", None); env["LD_DEBUG"] = "libs"; env["LC_ALL"] = "C"
        r = subprocess.run([str(root / "llama-server"), "--list-devices"], cwd=elsewhere, env=env, stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        loaded = sorted(set(re.findall(r"calling init: (\S+)", r.stdout.decode("utf-8", "replace"))))
        rocm_like = [x for x in loaded if re.search(r"amdhip|rocblas|hipblas|comgr|hsa-runtime|rocm|LLVM|rocsolver|tensile|origami|roctx|rocprofiler", os.path.basename(x))]
        outside = [x for x in rocm_like if os.path.dirname(os.path.realpath(x)) != str(root.resolve())]
        print(f"libraries loaded: {len(loaded)}; ROCm libraries: {len(rocm_like)}; ROCm libraries from outside the bundle: {outside}")
        if not rocm_like:
            problems.append("no ROCm library was loaded")
        if outside:
            problems.append(f"ROCm libraries loaded from outside the bundle: {outside}")

    ngl = "99" if kind == "rocm" else "0"
    cmd = [str(root / f"llama-completion{EXE}"), "-m", str(model), "-p", PROMPT, "-n", "48", "-no-cnv",
           "--temp", "0", "--seed", "1", "-ngl", ngl, "--simple-io", "--no-display-prompt", "-lv", "4"]
    rc, out = run(cmd, elsewhere)
    (work / f"{label}-completion.log").write_text(out, encoding="utf-8")
    off = re.search(r"offloaded (\d+)/(\d+) layers to GPU", out)
    tps = re.findall(r"eval time\s*=.*?([\d.]+) tokens per second", out)
    dev = sorted(set(re.findall(r"\b(ROCm\d+|Vulkan\d+|CUDA\d+)\b", out)))
    buf = [(d, float(m)) for d, m in re.findall(r"(ROCm\d+) model buffer size\s*=\s*([0-9.]+) MiB", out)]
    err = [ln for ln in out.splitlines() if re.search(r"ROCm error|hipError|abort|Segmentation|Assertion|out of memory", ln)]
    lines = out.splitlines()
    start = max([i for i, ln in enumerate(lines) if " generate: " in ln] or [0])
    body = [ln for ln in lines[start + 1:] if ln.strip() and not re.match(r"^\d+\.\d+\.\d+\.\d+ [IWED] ", ln)]
    text = " ".join(body)[:300]
    print(f"generation rc={rc}; offloaded={off.group(0) if off else None}; devices={dev}; model buffers={buf}; tokens/s (prompt, gen)={tps}")
    gen = re.search(r"13, ?14, ?15", text)
    print(f"generated: {text!r}")
    for ln in err[:5]:
        print(f"  error line: {ln[:200]}")
    if rc != 0:
        problems.append(f"generation exited {rc}")
        print("\n".join(out.splitlines()[-25:]))
    if kind == "rocm":
        if not off or int(off.group(1)) == 0:
            problems.append("no layers were offloaded to the GPU")
        if not any(d.startswith("ROCm") for d in dev):
            problems.append("no ROCm device in the log")
        # "offloaded N/N layers" is printed from what -ngl asked for; a model buffer on the device is the proof.
        if not any(mib > 0 for _, mib in buf):
            problems.append("no model memory was placed on a ROCm device")
    if not gen:
        problems.append("the generated text does not continue the count (13, 14, 15)")

    # Unquantized weights and a long prompt: the batched matmul goes through hipBLAS / rocBLAS and its kernel packs.
    if kind == "rocm" and model2:
        pf = work / "long-prompt.txt"
        pf.write_text(("The quick brown fox jumps over the lazy dog. " * 40) + "\n1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,", encoding="utf-8")
        cmd = [str(root / f"llama-completion{EXE}"), "-m", str(model2), "-f", str(pf), "-n", "24", "-no-cnv",
               "--temp", "0", "--seed", "1", "-ngl", "99", "--simple-io", "--no-display-prompt", "-lv", "4"]
        rc, out = run(cmd, elsewhere)
        (work / f"{label}-completion-bf16.log").write_text(out, encoding="utf-8")
        tps = re.findall(r"eval time\s*=.*?([\d.]+) tokens per second", out)
        buf = [(d, float(m)) for d, m in re.findall(r"(ROCm\d+) model buffer size\s*=\s*([0-9.]+) MiB", out)]
        lines = out.splitlines()
        start = max([i for i, ln in enumerate(lines) if " generate: " in ln] or [0])
        text = " ".join(ln for ln in lines[start + 1:] if ln.strip() and not re.match(r"^\d+\.\d+\.\d+\.\d+ [IWED] ", ln))[:200]
        err = [ln for ln in lines if re.search(r"ROCm error|hipError|rocBLAS|abort|Segmentation|Assertion", ln)]
        print(f"BF16 long-prompt generation rc={rc}; model buffers={buf}; tokens/s (prompt, gen)={tps}; generated: {text!r}")
        for ln in err[:5]:
            print(f"  error line: {ln[:200]}")
        if rc != 0:
            problems.append(f"BF16 generation exited {rc}")
            print("\n".join(lines[-25:]))
        elif not re.search(r"13, ?14", text):
            problems.append("the BF16 generation does not continue the count")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["rocm", "cpu", "static"], required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--model2", default="")
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    model = fetch(a.model, work / "model.gguf").resolve() if a.model else Path()
    model2 = fetch(a.model2, work / "model2.gguf").resolve() if a.model2 else None
    elsewhere = Path(tempfile.mkdtemp(prefix="elsewhere-", dir=work))
    results = {}
    if a.old:
        old = fetch(a.old, work / ("old" + (".zip" if a.old.endswith(".zip") else ".tar.gz"))).resolve()
        results["old"] = check("old", a.kind, old, model, work, elsewhere, model2)
    results["new"] = check("new", a.kind, Path(a.new).resolve(), model, work, elsewhere, model2)
    print("\n===== summary")
    for label, problems in results.items():
        print(f"{label}: {'OK' if not problems else 'PROBLEMS'}")
        for p in problems:
            print(f"  - {p}")
    return 1 if results["new"] else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""CPU-only speed of several published Windows bundles on this machine.

  bench_cpu.py --work <dir> --model <url> <label>=<zip url> ...
"""
import argparse, os, re, shutil, subprocess, sys, urllib.request, zipfile, statistics
from pathlib import Path

def fetch(url, dest):
    if not dest.exists():
        print("downloading", url, flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "bench"})
        with urllib.request.urlopen(req, timeout=900) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    return dest

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--threads", default="")
ap.add_argument("bundles", nargs="+")
a = ap.parse_args()
work = Path(a.work).resolve(); work.mkdir(parents=True, exist_ok=True)
model = fetch(a.model, work / "model.gguf")
prompt = work / "prompt.txt"
prompt.write_text(("The quick brown fox jumps over the lazy dog. " * 160) + "\n1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,", encoding="utf-8")
rows = []
for spec in a.bundles:
    label, url = spec.split("=", 1)
    root = work / label
    if not root.exists():
        z = fetch(url, work / f"{label}.zip")
        with zipfile.ZipFile(z) as zf: zf.extractall(root)
    names = sorted(p.name for p in root.iterdir())
    variants = [n for n in names if n.startswith("ggml-cpu")]
    omp = [n for n in names if n.lower().startswith(("libomp", "vcomp"))]
    exe = root / "llama-completion.exe"
    cmd = [str(exe), "-m", str(model), "-f", str(prompt), "-n", "96", "-no-cnv", "--temp", "0", "--seed", "1",
           "-ngl", "0", "-dev", "none", "--simple-io", "--no-display-prompt", "-c", "4096"]
    if a.threads: cmd += ["-t", a.threads]
    pe, ge, loaded, rc = [], [], "", 0
    for i in range(4):
        env = dict(os.environ); env["LC_ALL"] = "C"
        r = subprocess.run(cmd, cwd=work, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1200)
        out = r.stdout.decode("utf-8", "replace"); rc = r.returncode
        (work / f"{label}-{i}.log").write_text(out, encoding="utf-8")
        m = re.findall(r"loaded CPU backend from .*?(ggml-cpu[\w-]*)\.dll", out)
        loaded = m[-1] if m else loaded
        p = re.search(r"prompt eval time\s*=.*?([\d.]+) tokens per second", out)
        g = re.search(r"\n[^\n]*?(?<!prompt )eval time\s*=.*?([\d.]+) tokens per second", out)
        si = re.search(r"system_info: (.*)", out)
        if i == 0:
            print(f"[{label}] rc={rc} loaded={loaded} omp_files={omp} variants={len(variants)}")
            print(f"[{label}] {si.group(1)[:230] if si else ''}")
            if rc != 0: print("\n".join(out.splitlines()[-15:]))
            continue  # warm-up
        if p: pe.append(float(p.group(1)))
        if g: ge.append(float(g.group(1)))
    rows.append((label, loaded, len(variants), ",".join(omp) or "-", statistics.median(pe) if pe else 0, statistics.median(ge) if ge else 0, pe, ge))
print("\n| bundle | CPU variant loaded | variants shipped | OpenMP runtime | prompt tok/s (median of 3) | gen tok/s (median of 3) |")
print("|---|---|---|---|---|---|")
for label, loaded, nv, omp, p, g, pe, ge in rows:
    print(f"| {label} | {loaded} | {nv} | {omp} | {p:.1f} | {g:.1f} |")
for label, *_rest, pe, ge in rows:
    print(f"raw {label}: prompt={pe} gen={ge}")

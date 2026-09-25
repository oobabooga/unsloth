#!/usr/bin/env python3
"""Generate a self-contained .ipynb that runs the portability harness on a rented Colab / Kaggle GPU.

    python make_notebook.py --out notebooks/portability_quick.ipynb [--cases all] [--full] [--iters 10]
    python $WORKSPACE/notebook_cloud_run.py --backend colab --gpu T4 --per-cell-timeout 3300 --wall-timeout 3600 \\
        --outdir $WORKSPACE/outputs/portability/_runs/t4 notebooks/portability_quick.ipynb
    python collect.py $WORKSPACE/outputs/portability/_runs/t4        # -> outputs/portability/<device>/

The bundle (harness + PR kernel sources + verifier, see snapshot.py) is embedded as base64, so the notebook needs no
clone and no token. It installs only a torchao wheel matched to the preinstalled torch (--no-deps); everything else is
the image's own torch / triton. The run cell raises only if the harness itself dies; a failing case is a result.
Results come back as a base64 tarball printed between markers, which collect.py extracts from the executed notebook.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import snapshot  # noqa: E402

BEGIN, END = "PORTABILITY_RESULTS_BEGIN", "PORTABILITY_RESULTS_END"

CELL_ENV = r'''
import os, subprocess, sys, json
import shutil
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"], check=False)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "hip", getattr(torch.version, "hip", None))
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
      torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "")
if not torch.cuda.is_available():
    raise RuntimeError("no GPU attached to this kernel")
# torchao wheel matched to this torch (pytorch/ao compatibility table); --no-deps so torch is never replaced
v = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:3])
AO = {(2, 6): "0.12.0", (2, 7): "0.12.0", (2, 8): "0.13.0", (2, 10): "0.16.0", (2, 11): "0.17.0", (2, 12): "0.17.0"}
ao = "0.14.1" if v[:3] == (2, 9, 0) else "0.15.0" if v[:2] == (2, 9) else AO.get(v[:2], "0.17.0")
try:
    import torchao
    print("torchao preinstalled", torchao.__version__)
except Exception:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", f"torchao=={ao}"],
                       capture_output=True, text=True)
    print("pip torchao", ao, "rc", r.returncode, r.stderr[-500:])
r = subprocess.run([sys.executable, "-c", "import torchao; print(torchao.__version__)"], capture_output=True, text=True)
print("torchao import:", r.returncode, (r.stdout + r.stderr).strip()[-300:])
try:
    import triton
    print("triton", triton.__version__)
except Exception as exc:
    print("triton import failed:", exc)
'''

CELL_RUN = r'''
import base64, io, os, subprocess, sys, tarfile, time
root = os.path.abspath("port_run")
os.makedirs(root, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE_B64)), mode="r:gz") as t:
    t.extractall(root)
bundle = os.path.join(root, "bundle")
out = os.path.join(root, "out")
cmd = [sys.executable, "-u", os.path.join(bundle, "portability", "harness.py"), "--cases", CASES, "--out", out,
       "--bundle", bundle, "--iters", str(ITERS), "--timeout", str(CASE_TIMEOUT)] + (["--quick"] if QUICK else [])
print(" ".join(cmd), flush=True)
t0 = time.time()
env = dict(os.environ, PYTHONUNBUFFERED="1", WORKSPACE=root)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
keep = ("=====", "[", "wrote", "device ", "Traceback", "Error")
for line in p.stdout:
    if line.startswith(keep) or "rror" in line:
        print(line, end="", flush=True)
rc = p.wait()
print(f"harness rc={rc} in {time.time() - t0:.0f}s", flush=True)
if rc != 0 or not os.path.exists(os.path.join(out, "results.json")):
    raise RuntimeError(f"harness died rc={rc}")
print(open(os.path.join(out, "report.md")).read())
'''

CELL_EXPORT = r'''
import base64, io, os, tarfile
out = os.path.abspath(os.path.join("port_run", "out"))
buf = io.BytesIO()
budget = 6 * 2**20
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    for dp, dn, fn in os.walk(out):
        for f in sorted(fn):
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, out)
            big = os.path.getsize(p) > 256 * 1024
            if big and not rel.endswith((".json", ".md")):
                continue
            if budget <= 0 and not rel.endswith((".json", ".md")):
                continue
            budget -= os.path.getsize(p)
            t.add(p, arcname=rel)
b = base64.b64encode(buf.getvalue()).decode()
print("PORTABILITY_RESULTS_BEGIN")
for i in range(0, len(b), 16000):
    print(b[i:i + 16000])
print("PORTABILITY_RESULTS_END")
'''


def cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.strip("\n").splitlines(keepends = True)}


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default = os.path.join(HERE, "notebooks", "portability_quick.ipynb"))
    ap.add_argument("--cases", default = "all")
    ap.add_argument("--full", action = "store_true", help = "full shape lists (default: --quick sizes)")
    ap.add_argument("--iters", type = int, default = 10)
    ap.add_argument("--case-timeout", type = int, default = 1500)
    ap.add_argument("--bundle", default = None, help = "existing bundle dir (default: rebuild from the worktrees)")
    a = ap.parse_args()
    bundle = a.bundle
    if bundle is None:
        bundle = os.path.join(snapshot.WS, "temp", "portability_bundle")
        snapshot.build(bundle, os.path.join(snapshot.WS, "wt_port_h3"), "origin/studio-h3-vae-fast",
                       "origin/studio-nvfp4-kernels", "origin/main", fetch = True)
    b64 = base64.b64encode(snapshot.tar_bytes(bundle)).decode()
    params = (f"BUNDLE_B64 = {b64!r}\nCASES = {a.cases!r}\nQUICK = {not a.full}\nITERS = {a.iters}\n"
              f"CASE_TIMEOUT = {a.case_timeout}\n")
    with open(os.path.join(bundle, "MANIFEST.json"), encoding = "utf-8") as f:
        man = json.load(f)
    md = {"cell_type": "markdown", "metadata": {},
          "source": [f"Portability harness, cases={a.cases}, quick={not a.full}. Bundle: " + json.dumps(man)]}
    nb = {"cells": [md, cell(CELL_ENV), cell(params + CELL_RUN), cell(CELL_EXPORT)],
          "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python",
                                                            "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok = True)
    with open(a.out, "w", encoding = "utf-8") as f:
        json.dump(nb, f, indent = 1)
    print("wrote", a.out, f"({os.path.getsize(a.out) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()

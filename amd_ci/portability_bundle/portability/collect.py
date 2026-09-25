#!/usr/bin/env python3
"""Extract harness results from executed notebooks (notebook_cloud_run.py --outdir) or AMD CI artifacts.

    python collect.py <dir or .ipynb> [...]      -> outputs/portability/<device>/ (device read from results.json)
    python collect.py --tag g4 <dir>             force the device folder name
Walks the given paths for *.ipynb (decodes the base64 tarball printed between the markers) and for results.json
(copies that run's folder as is, e.g. a downloaded CI artifact). Then rebuilds outputs/portability/MATRIX.md.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
WS = os.environ.get("WORKSPACE") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(WS, "outputs", "portability")
BEGIN, END = "PORTABILITY_RESULTS_BEGIN", "PORTABILITY_RESULTS_END"


def _texts(nb):
    for c in nb.get("cells", []):
        for o in c.get("outputs", []):
            t = o.get("text")
            if t is None and "data" in o:
                t = o["data"].get("text/plain")
            if t is not None:
                yield "".join(t) if isinstance(t, list) else t


def from_notebook(path: str, tag: str | None) -> str | None:
    with open(path, encoding = "utf-8") as f:
        nb = json.load(f)
    text = "".join(_texts(nb))
    if BEGIN not in text:
        print(f"{path}: no results block")
        return None
    blob = text.split(BEGIN, 1)[1].split(END, 1)[0]
    data = base64.b64decode("".join(blob.split()))
    tmp = os.path.join(OUT, "_incoming")
    shutil.rmtree(tmp, ignore_errors = True)
    os.makedirs(tmp)
    with tarfile.open(fileobj = io.BytesIO(data), mode = "r:gz") as t:
        t.extractall(tmp)
    return place(tmp, tag, path)


def place(src: str, tag: str | None, origin: str) -> str:
    import portlib

    with open(os.path.join(src, "results.json"), encoding = "utf-8") as f:
        res = json.load(f)
    tag = tag or portlib.device_tag(res["device"])
    dst = os.path.join(OUT, tag)
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.rmtree(dst, ignore_errors = True)
        shutil.copytree(src, dst)
    with open(os.path.join(dst, "ORIGIN.txt"), "w", encoding = "utf-8") as f:
        f.write(origin + "\n")
    print(f"{origin} -> {dst} ({len(res['rows'])} rows)")
    return dst


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs = "+")
    ap.add_argument("--tag", default = None)
    a = ap.parse_args()
    for p in a.paths:
        if p.endswith(".ipynb"):
            from_notebook(p, a.tag)
            continue
        for dp, dn, fn in os.walk(p):
            for f in fn:
                if f.endswith(".ipynb"):
                    from_notebook(os.path.join(dp, f), a.tag)
            if "results.json" in fn and "parts" in dn:
                place(dp, a.tag, dp)
    import matrix

    matrix.main([])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Probe: run the qwen4exp backend cells against one llama.cpp binary dir.

`--checkout` is the directory holding llama-server for this state (a prebuilt
release, not a source tree). Observes only: the cells record corruption
signatures, the criteria module decides what they mean. JSON to --out.
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, tempfile
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True); ap.add_argument("--checkout", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", required=True); ap.add_argument("--sentinel-model", default="")
    ap.add_argument("--env", action="append", default=[]); ap.add_argument("--unset-env", action="append", default=[])
    ap.add_argument("--gpu-var", action="append", default=["NONE"]); ap.add_argument("--gpu", default="0")
    ap.add_argument("--cells", default="c1,c2,c3"); ap.add_argument("--port", type=int, default=8650)
    ap.add_argument("--load-timeout", type=int, default=1800); ap.add_argument("--wikitext", default="")
    a = ap.parse_args()
    out_dir = a.out.parent / f"cells_{a.state}"; out_dir.mkdir(parents=True, exist_ok=True)
    cells = Path(__file__).resolve().parent.parent / "lib" / "qwen4exp_cells.py"
    cmd = [sys.executable, str(cells), "--bin", str(a.checkout), "--label", a.state, "--out", str(out_dir),
           "--model", a.model, "--cells", a.cells, "--port", str(a.port), "--gpu", a.gpu, "--load-timeout", str(a.load_timeout)]
    for v in a.gpu_var: cmd += ["--gpu-var", v]
    for e in a.env: cmd += ["--env", e]
    for e in a.unset_env: cmd += ["--unset-env", e]
    if a.sentinel_model: cmd += ["--sentinel-model", a.sentinel_model]
    if a.wikitext: cmd += ["--wikitext", a.wikitext]
    rc = subprocess.run(cmd).returncode
    obs: dict = {"state": a.state, "checkout": str(a.checkout), "cells_rc": rc}
    j = out_dir / f"cells_{a.state}.json"
    if j.is_file():
        obs.update(json.loads(j.read_text()))
    else:
        obs["setup_error"] = f"cells produced no JSON (rc={rc})"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(obs, indent=1))
    return 0

if __name__ == "__main__":
    sys.exit(main())

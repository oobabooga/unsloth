#!/usr/bin/env python3
"""Probe: inventory of the XDNA2 NPU software stack on this runner. Observes only."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

CMDS = {
    "runner": "echo ${RUNNER_NAME:-?}; hostname",
    "recipes": "curl -s -m 10 http://127.0.0.1:13305/api/v1/system-info | python3 -c 'import json,sys\nd=json.load(sys.stdin)\nfor r,v in d.get(\"recipes\",{}).items():\n  print(r, \"|\", v.get(\"modality\"), \"| default\", v.get(\"default_backend\"))\n  for b,bv in v.get(\"backends\",{}).items(): print(\"   \", b, bv.get(\"state\"), bv.get(\"devices\"), bv.get(\"version\"), (bv.get(\"message\") or \"\")[:120])'",
    "image_models": "curl -s -m 10 'http://127.0.0.1:13305/api/v1/models?show_all=true' | python3 -c 'import json,sys\nd=json.load(sys.stdin)\nfor m in d.get(\"data\",[]):\n  L=m.get(\"labels\") or []\n  r=str(m.get(\"recipe\"))\n  if \"image\" in L or \"npu\" in r.lower() or \"flm\" in r.lower() or \"sd\" in r.lower() or \"NPU\" in m.get(\"id\",\"\"): print(m.get(\"id\"), r, L, m.get(\"checkpoint\"), m.get(\"downloaded\"))'",
    "backends_all": "lemonade backends --all 2>&1 | head -120",
    "downloaded": "lemonade list --downloaded 2>&1 | head -60",
}

def run(c):
    try:
        p = subprocess.run(["bash", "-lc", c], capture_output=True, text=True, timeout=150)
        return (p.stdout + p.stderr).strip()[-20000:]
    except Exception as e:
        return f"ERR {type(e).__name__}: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--checkout", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    obs = {"state": a.state, "cmds": {k: run(v) for k, v in CMDS.items()}}
    obs["has_accel_node"] = os.path.exists("/dev/accel/accel0")
    obs["has_flm"] = shutil.which("flm") is not None
    a.out.write_text(json.dumps(obs, indent=2), encoding="utf-8")
    return 0

if __name__ == "__main__":
    sys.exit(main())

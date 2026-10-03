#!/usr/bin/env python3
"""Observe: run the sft_stack_probe LoRA SFT inside an installed Studio venv, as Studio's worker would.

Windows ROCm: the worker sets TORCHDYNAMO_DISABLE=1 and puts the venv's Scripts dir on PATH (hipInfo,
rocm-sdk). Records the same fields as sft_stack_probe plus the Studio venv's package versions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sft_stack_probe import MODELS, _TRAIN  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required = True, help = "the Studio venv's python")
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--work", required = True, type = Path)
    args = ap.parse_args()
    obs: dict = {"python": args.python}
    scripts = str(Path(args.python).parent)
    env = dict(os.environ, UNSLOTH_DISABLE_AUTO_UPDATES = "1", TORCHDYNAMO_DISABLE = "1",
               UNSLOTH_COMPILE_LOCATION = str(args.work / "cc"),
               PATH = scripts + os.pathsep + os.environ.get("PATH", ""))
    p = subprocess.run([args.python, "-m", "pip", "list", "--format=json"], capture_output = True, text = True)
    try:
        obs["installed"] = {d["name"].lower(): d["version"] for d in json.loads(p.stdout)
                            if any(k in d["name"].lower() for k in ("torch", "triton", "rocm", "bitsandbytes",
                                                                    "unsloth", "transformers", "trl", "peft"))}
    except Exception as e:  # noqa: BLE001
        obs["installed"] = f"{type(e).__name__}: {p.stderr[-500:]}"
    obs["runs"] = {}
    for name, model in MODELS.items():
        work = args.work / f"run_{name}"
        work.mkdir(parents = True, exist_ok = True)
        script = work / "train.py"
        script.write_text(_TRAIN, encoding = "utf-8")
        p = subprocess.run([args.python, str(script), model], capture_output = True, text = True,
                           encoding = "utf-8", errors = "replace", env = env, cwd = str(work), timeout = 2400)
        so, err = p.stdout, p.stderr
        line = [l for l in so.splitlines() if l.startswith("SFT_STACK_RESULT ")]
        if line:
            obs["runs"][name] = json.loads(line[-1].split(" ", 1)[1])
        else:
            obs["runs"][name] = {
                "error": f"rc={p.returncode}: {(so + err)[-2500:]}",
                "diag": next((json.loads(l.split(" ", 1)[1]) for l in so.splitlines()
                              if l.startswith("SFT_STACK_DIAG ")), None),
                "error_lines": [l for l in (so + err).splitlines()
                                if any(k in l.lower() for k in ("error", "fatal", "cannot open"))][:60]}
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    print(json.dumps({k: (v if k != "runs" else {n: {kk: r.get(kk) for kk in ("losses", "finite", "error_lines", "diag")}
                                                  for n, r in v.items()}) for k, v in obs.items()}, indent = 1)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

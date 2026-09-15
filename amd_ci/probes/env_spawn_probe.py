#!/usr/bin/env python3
"""Probe: what value does TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL have after each
Unsloth entry point imports, from each starting state, and do spawned children
see it?

Observes only; no torch needed. Every cell is a FRESH interpreter, because the
question is about import-time side effects. Cells:

  entry in {unsloth, studio_main} x start in {unset, "0", "1"}
      -> value after the import (read in a `finally`, so an import that fails on
         a missing dependency still reports whether the setdefault ran first)
  spawn shapes, from a parent that imported `unsloth` from the unset start:
      mp_spawn        multiprocessing.get_context("spawn").Process, what Studio's
                      training worker uses
      subprocess      subprocess.run with no env=
      subprocess_copy subprocess.run with env=os.environ.copy()
      scrubbed        control: env with the key removed; must read unset
  and the same three shapes from a parent that started with "0" (opt-out must
  propagate as "0"), and a reader check from a parent that started with "1".

Pairs with criteria/env_spawn_inherits.py. Writes JSON via --out, never stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

KEY = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"

IMPORT_BODY = r'''
import json, os, sys
entry, checkout, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
KEY = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"
obs = {"before": os.environ.get(KEY)}
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
try:
    if entry == "unsloth":
        sys.path.insert(0, checkout)
        import unsloth  # noqa: F401
    else:
        backend = os.path.join(checkout, "studio", "backend")
        os.chdir(backend)
        sys.path.insert(0, backend)
        import main  # noqa: F401
    obs["import_ok"] = True
except BaseException as e:  # noqa: BLE001
    obs["import_ok"] = False
    obs["import_error"] = f"{type(e).__name__}: {e}"[:300]
finally:
    obs["after"] = os.environ.get(KEY)
with open(out_path, "w", encoding = "utf-8") as f:
    json.dump(obs, f)
'''

SPAWN_BODY = r'''
import json, os, sys, subprocess
checkout, out_path = sys.argv[1], sys.argv[2]
KEY = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"
READER = "import os; print('CHILD=' + repr(os.environ.get(%r)))" % KEY
obs = {"parent_before": os.environ.get(KEY)}
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
sys.path.insert(0, checkout)
try:
    import unsloth  # noqa: F401
except BaseException as e:  # noqa: BLE001
    obs["import_error"] = f"{type(e).__name__}: {e}"[:200]
obs["parent_after"] = os.environ.get(KEY)

def _read(env):
    p = subprocess.run([sys.executable, "-c", READER], env = env, capture_output = True, text = True, timeout = 120)
    for line in (p.stdout or "").splitlines():
        if line.startswith("CHILD="):
            return eval(line[6:])
    return "NO-READING: " + (p.stderr or "")[-200:]

def _mp_child(q):
    import os as _os
    q.put(_os.environ.get(KEY))

if __name__ == "__main__":
    obs["subprocess"] = _read(None)
    obs["subprocess_copy"] = _read(os.environ.copy())
    scrubbed = os.environ.copy(); scrubbed.pop(KEY, None)
    obs["scrubbed"] = _read(scrubbed)
    try:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target = _mp_child, args = (q,))
        p.start()
        obs["mp_spawn"] = q.get(timeout = 120)
        p.join(30)
    except BaseException as e:  # noqa: BLE001
        obs["mp_spawn"] = "ERROR: " + f"{type(e).__name__}: {e}"[:200]
    with open(out_path, "w", encoding = "utf-8") as f:
        json.dump(obs, f)
'''


def _run(python: str, body: Path, args: list[str], env: dict, timeout: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "cell.json"
        try:
            p = subprocess.run([python, str(body), *args, str(out)], env = env,
                               capture_output = True, text = True, timeout = timeout)
            res = {"rc": p.returncode, "stderr_tail": (p.stderr or "")[-500:]}
        except subprocess.TimeoutExpired:
            res = {"rc": -1, "error": "TimeoutExpired"}
        if out.exists():
            res.update(json.loads(out.read_text(encoding = "utf-8")))
        else:
            res["error"] = res.get("error") or "cell wrote nothing"
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 600)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "checkout": args.checkout, "python": args.python,
                 "platform": sys.platform, "cells": {}, "spawn": {}}
    with tempfile.TemporaryDirectory() as td:
        ib = Path(td) / "import_body.py"
        ib.write_text(IMPORT_BODY, encoding = "utf-8")
        sb = Path(td) / "spawn_body.py"
        sb.write_text(SPAWN_BODY, encoding = "utf-8")
        starts = {"unset": None, "0": "0", "1": "1"}
        for entry in ("unsloth", "studio_main"):
            for label, val in starts.items():
                env = dict(os.environ)
                env.pop(KEY, None)
                if val is not None:
                    env[KEY] = val
                obs["cells"][f"{entry}/{label}"] = _run(args.python, ib, [entry, args.checkout], env, args.timeout)
        for label, val in starts.items():
            env = dict(os.environ)
            env.pop(KEY, None)
            if val is not None:
                env[KEY] = val
            obs["spawn"][label] = _run(args.python, sb, [args.checkout], env, args.timeout)
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

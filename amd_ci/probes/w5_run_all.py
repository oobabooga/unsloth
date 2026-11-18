#!/usr/bin/env python3
"""Run w5_cells.py once per model (fresh process each) for one interpreter / PYTHONPATH and write summary.json.

  python w5_run_all.py --python PY --out-dir DIR [--pythonpath A:B] [--models llama,qwen3,...] [--timeout 1200]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["llama", "qwen3", "gemma2", "qwen3moe", "gptoss"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--out-dir", required = True, type = Path)
    ap.add_argument("--pythonpath", default = "")
    ap.add_argument("--models", default = ",".join(ORDER))
    ap.add_argument("--timeout", type = int, default = 1200)
    a = ap.parse_args()
    a.out_dir.mkdir(parents = True, exist_ok = True)
    env = dict(os.environ)
    if a.pythonpath:
        env["PYTHONPATH"] = a.pythonpath + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("UNSLOTH_COMPILE_LOCATION", str(a.out_dir / "unsloth_compiled_cache"))
    summary = {}
    for m in a.models.split(","):
        out, log = a.out_dir / f"{m}.json", a.out_dir / f"{m}.log"
        with open(log, "wb") as fh:
            try:
                rc = subprocess.run([a.python, str(HERE / "w5_cells.py"), "--model", m, "--out", str(out)],
                                    env = env, stdout = fh, stderr = subprocess.STDOUT, timeout = a.timeout).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        rec = {"rc": rc}
        if out.is_file():
            try:
                rec.update(json.loads(out.read_text(encoding = "utf-8")))
            except Exception as e:  # noqa: BLE001
                rec["parse_error"] = repr(e)
        else:
            rec["missing_output"] = True
            rec["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-2000:]
        summary[m] = rec
        (a.out_dir / "summary.json").write_text(json.dumps(summary, indent = 1, default = str), encoding = "utf-8")
    for m, rec in summary.items():
        cells = {k: v.get("ok") for k, v in (rec.get("cells") or {}).items()}
        print(f"W5 {m}: rc={rec.get('rc')} {cells} {rec.get('import_error', '')[:200]}", flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

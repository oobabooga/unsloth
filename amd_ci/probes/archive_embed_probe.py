#!/usr/bin/env python3
"""Probe: CPU archive embedding before inference, and how long it outlives Chat Stop (#14519).

Observes only: runs probes/repro_14519.py (real SentenceTransformer bge-small on CPU behind this
checkout's conversation_archive.archive_turns) and records what it measured.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEPS = [
    "torch", "sentence-transformers", "sqlite-vec", "fastapi", "pydantic", "pydantic-extra-types",
    "email-validator", "orjson", "python-multipart", "rich", "structlog", "numpy",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    obs: dict = {"state": args.state}
    pip = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *DEPS,
         "--extra-index-url", "https://download.pytorch.org/whl/cpu"],
        capture_output = True, text = True, encoding = "utf-8", errors = "replace",
    )
    obs["pip_rc"] = pip.returncode
    raw = args.out.with_suffix(".repro.json")
    run = subprocess.run(
        [sys.executable, "-u", str(HERE / "repro_14519.py"),
         "--backend-dir", str(args.checkout / "studio" / "backend"),
         "--label", args.state, "--out", str(raw),
         "--rounds", "300", "--tool-words", "2400", "--stop-after", "8"],
        capture_output = True, text = True, encoding = "utf-8", errors = "replace",
        env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""},
    )
    obs["rc"] = run.returncode
    obs["stderr_tail"] = run.stderr[-1500:]
    if raw.exists():
        obs["repro"] = json.loads(raw.read_text(encoding = "utf-8"))
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

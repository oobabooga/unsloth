# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required = True, type = Path)
    parser.add_argument("--out", required = True, type = Path)
    args, _ = parser.parse_known_args()
    requirements = args.checkout / "studio/backend/requirements/studio.txt"
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = Path(sys.prefix) / "issue16395_deps.sha256"
    if not marker.exists() or marker.read_text(encoding = "utf-8") != digest:
        with args.out.with_suffix(".deps.log").open("w", encoding = "utf-8") as log:
            installed = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--only-binary=:all:",
                 "-r", str(requirements), "python-multipart", "psutil"],
                stdout = log, stderr = subprocess.STDOUT, timeout = 1200,
            )
        if installed.returncode:
            args.out.write_text(json.dumps({"error": "Studio dependency install failed"}), encoding = "utf-8")
            return 0
        marker.write_text(digest, encoding = "utf-8")
    os.environ.update(CUDA_VISIBLE_DEVICES = "", HIP_VISIBLE_DEVICES = "", ROCR_VISIBLE_DEVICES = "")
    from pytest_probe import main as observe
    return observe()


if __name__ == "__main__":
    raise SystemExit(main())

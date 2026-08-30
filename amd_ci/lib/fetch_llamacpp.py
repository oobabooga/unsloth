#!/usr/bin/env python3
"""Download a llama.cpp prebuilt and report where its binaries ended up.

Kept out of the workflow shell on purpose. The archives disagree about layout -
the upstream tarballs contain a single `llama-bNNNNN/` directory, the Lemonade
zips do not necessarily - and resolving that with a glob inside a `run:` block
is exactly the shape the toolkit's E005 lint rule exists to stop, because a
failed glob under `pipefail` reports the wrong status and `2>/dev/null` hides
the message rather than the failure.

Writes a JSON object describing what it got, so a later step reads a recorded
fact instead of re-deriving it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

MARKER = "llama-server"


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents = True, exist_ok = True)
    with urllib.request.urlopen(url, timeout = 600) as r, open(dest, "wb") as fh:
        shutil.copyfileobj(r, fh)
    return dest


def extract(archive: Path, into: Path) -> None:
    into.mkdir(parents = True, exist_ok = True)
    if archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive) as tf:
            # `filter` became required-in-practice in 3.12 and defaults to
            # rejecting in 3.14; passing it explicitly keeps one code path.
            if sys.version_info >= (3, 12):
                tf.extractall(into, filter = "data")
            else:
                tf.extractall(into)  # noqa: S202 - trusted release artifacts
    elif archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(into)  # noqa: S202
    else:
        raise SystemExit(f"unknown archive type: {archive.name}")


def find_bin_dir(root: Path) -> Path | None:
    """The directory holding llama-server, shallowest first."""
    hits = sorted(root.rglob(MARKER), key = lambda p: len(p.parts))
    for h in hits:
        if h.is_file():
            return h.parent
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required = True)
    ap.add_argument("--dest", required = True, type = Path,
                    help = "directory to extract into")
    ap.add_argument("--out", type = Path, default = None)
    args = ap.parse_args()

    info: dict = {"url": args.url, "dest": str(args.dest)}
    archive = args.dest.parent / args.url.rsplit("/", 1)[-1]
    try:
        download(args.url, archive)
        info["archive_bytes"] = archive.stat().st_size
        extract(archive, args.dest)
        archive.unlink(missing_ok = True)
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"

    bin_dir = find_bin_dir(args.dest) if args.dest.is_dir() else None
    info["bin_dir"] = str(bin_dir) if bin_dir else None
    if bin_dir:
        # The release archives do not always carry the executable bit through,
        # and a non-executable llama-server reads downstream as "no binary".
        for f in bin_dir.iterdir():
            if f.is_file() and (f.name.startswith("llama-") or f.name.startswith("test-")):
                f.chmod(f.stat().st_mode | 0o111)
        info["binaries"] = sorted(p.name for p in bin_dir.iterdir()
                                  if p.is_file() and p.name.startswith(("llama-", "test-")))
        info["has_test_backend_ops"] = (bin_dir / "test-backend-ops").is_file()
        env = {"LD_LIBRARY_PATH": str(bin_dir)}
        try:
            p = subprocess.run([str(bin_dir / MARKER), "--version"], capture_output = True,
                               text = True, timeout = 300, env = env)
            info["version"] = ((p.stdout or "") + (p.stderr or "")).strip()[:400]
        except Exception as e:  # noqa: BLE001
            info["version_error"] = f"{type(e).__name__}: {e}"

    print(json.dumps(info, indent = 2))
    if args.out:
        args.out.parent.mkdir(parents = True, exist_ok = True)
        args.out.write_text(json.dumps(info, indent = 2))
    # A missing binary is a setup failure, not a finding: fail loudly here rather
    # than letting every downstream probe report "no llama-server".
    return 0 if bin_dir else 1


if __name__ == "__main__":
    sys.exit(main())

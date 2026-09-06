#!/usr/bin/env python3
"""Download one file, record what arrived.

For inputs that are not release archives and not Hub files: a single library
supplied by hand, for instance. Records size and sha256 so the report can name
exactly which artifact was measured, and so a swapped file is visible without
trusting the URL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required = True)
    ap.add_argument("--dest", required = True, help = "file path to write")
    ap.add_argument("--expect-sha256", default = "")
    ap.add_argument("--out", default = "")
    a = ap.parse_args()

    dest = Path(a.dest)
    dest.parent.mkdir(parents = True, exist_ok = True)
    with urllib.request.urlopen(a.url, timeout = 600) as r, open(dest, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    info = {"url": a.url, "path": str(dest), "bytes": dest.stat().st_size, "sha256": digest}
    print(json.dumps(info, indent = 2))
    if a.out:
        Path(a.out).write_text(json.dumps(info, indent = 2), encoding = "utf-8")
    if a.expect_sha256 and a.expect_sha256.lower() != digest:
        # A file that is not the one asked for is not a file that will do.
        print(f"FATAL: expected sha256 {a.expect_sha256}, got {digest}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

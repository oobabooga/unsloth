#!/usr/bin/env python3
"""Fetch wikitext-2's test split (ggml-org's CI copy) as prompts for the
llama-server probe. stdlib only; written as bytes, read back as UTF-8.

  python amd_ci/lib/fetch_wikitext.py --out "$AMD_CI_WORK/data/wikitext2_test.txt"
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
MEMBER = "wiki.test.raw"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--url", default = URL)
    ap.add_argument("--member", default = MEMBER, help = "archive member name suffix")
    args = ap.parse_args()

    data = urllib.request.urlopen(args.url, timeout = 300).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist() if n.endswith(args.member)]
        if not names:
            print(f"no member ending in {args.member!r} in {args.url}", file = sys.stderr)
            return 1
        args.out.parent.mkdir(parents = True, exist_ok = True)
        args.out.write_bytes(z.read(names[0]))
    size = args.out.stat().st_size
    print(f"wrote {args.out} ({size} bytes)")
    # A truncated download is a bad prompt set, not a defect; refuse it here.
    return 0 if size > 100_000 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Fetch wikitext-2 test text for the c3 prompts; stdlib only, works on Windows."""
import argparse, io, urllib.request, zipfile
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); a = ap.parse_args()
url = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
data = urllib.request.urlopen(url, timeout=300).read()
with zipfile.ZipFile(io.BytesIO(data)) as z:
    name = [n for n in z.namelist() if n.endswith("wiki.test.raw")][0]
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_bytes(z.read(name))
print("wikitext ok", a.out.stat().st_size)

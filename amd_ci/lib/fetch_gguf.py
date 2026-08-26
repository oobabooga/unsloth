#!/usr/bin/env python3
"""Download one file from a Hugging Face repo, at a pinned revision.

Plain HTTPS against `/{repo}/resolve/{revision}/{file}`, not `huggingface_hub`.
The runner has no hub package and `pip install --user` did not put one on
`sys.path`, which cost a run; more importantly a 18 GB model download wants
resume and retry, and adding a dependency to get them is the wrong trade on a
host where nothing is cached and every run re-downloads.

`--revision` is the point of the whole exercise on the model branch: the two
states are two revisions of the same repository, and a state that silently
resolved to `main` would compare a file with itself. It is required, with no
default, for exactly that reason.

Everything lands under `--local-dir`. `$RUNNER_TEMP` is the only tree the runner
reclaims between jobs, and this host is persistent and shared.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CHUNK = 8 << 20


def _open(url: str, start: int, token: str | None):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if start:
        req.add_header("Range", f"bytes={start}-")
    return urllib.request.urlopen(req, timeout = 120)


def download(url: str, dest: Path, token: str | None, attempts: int) -> dict:
    """Resume on failure rather than restarting.

    A dropped connection 15 GB into a 18 GB file should cost seconds, not the
    whole download, and on this runner a restart also costs the GPU concurrency
    slot the job is holding.
    """
    dest.parent.mkdir(parents = True, exist_ok = True)
    info: dict = {"attempts": []}
    for attempt in range(attempts):
        start = dest.stat().st_size if dest.is_file() else 0
        try:
            with _open(url, start, token) as r:
                total = int(r.headers.get("Content-Length") or 0) + start
                mode = "ab" if start else "wb"
                with open(dest, mode) as fh:
                    while True:
                        buf = r.read(CHUNK)
                        if not buf:
                            break
                        fh.write(buf)
            got = dest.stat().st_size
            info["attempts"].append({"n": attempt, "from": start, "to": got, "expected": total})
            if not total or got >= total:
                info["bytes"] = got
                return info
        except urllib.error.HTTPError as e:
            info["attempts"].append({"n": attempt, "http": e.code})
            # 416 means the file is already complete; anything 4xx other than
            # 429 will not become true by retrying.
            if e.code == 416 and dest.is_file():
                info["bytes"] = dest.stat().st_size
                return info
            if 400 <= e.code < 500 and e.code != 429:
                info["error"] = f"HTTP {e.code} {e.reason}"
                return info
        except Exception as e:  # noqa: BLE001
            info["attempts"].append({"n": attempt, "error": f"{type(e).__name__}: {e}"})
        time.sleep(min(30, 5 * (attempt + 1)))
    info["error"] = "download did not complete"
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required = True)
    ap.add_argument("--file", required = True)
    ap.add_argument("--revision", required = True,
                    help = "no default on purpose: an accidental 'main' would make two "
                           "states the same file")
    ap.add_argument("--local-dir", required = True, type = Path)
    ap.add_argument("--as-name", default = "",
                    help = "save under this filename instead of the repo's")
    ap.add_argument("--attempts", type = int, default = 5)
    ap.add_argument("--out", type = Path, default = None)
    args = ap.parse_args()

    url = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/{args.file}"
    dest = args.local_dir / (args.as_name or args.file.rsplit("/", 1)[-1])
    info: dict = {"repo": args.repo, "file": args.file, "revision": args.revision,
                  "url": url, "path": str(dest)}
    info.update(download(url, dest, os.environ.get("HF_TOKEN"), args.attempts))

    print(json.dumps(info, indent = 2))
    if args.out:
        args.out.parent.mkdir(parents = True, exist_ok = True)
        args.out.write_text(json.dumps(info, indent = 2))
    return 0 if info.get("bytes") and not info.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())

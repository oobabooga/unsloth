#!/usr/bin/env python3
"""Download one file from a Hugging Face repo, at a pinned revision.

`--local-dir` is not optional here. The runner is a persistent, shared host and
`$RUNNER_TEMP` is the only tree it reclaims between jobs; the default hub cache
would leave tens of GB in `$HOME` after every run.

`--revision` is the point of the whole exercise on the model branch: the two
states are two revisions of the same repository, and a state that silently
resolved to `main` would compare a file with itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required = True)
    ap.add_argument("--file", required = True)
    ap.add_argument("--revision", default = "main")
    ap.add_argument("--local-dir", required = True, type = Path)
    ap.add_argument("--out", type = Path, default = None)
    args = ap.parse_args()

    info: dict = {"repo": args.repo, "file": args.file, "revision": args.revision}
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(args.repo, args.file, revision = args.revision,
                               local_dir = str(args.local_dir))
        info["path"] = path
        info["bytes"] = Path(path).stat().st_size
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"

    print(json.dumps(info, indent = 2))
    if args.out:
        args.out.parent.mkdir(parents = True, exist_ok = True)
        args.out.write_text(json.dumps(info, indent = 2))
    return 0 if info.get("path") else 1


if __name__ == "__main__":
    sys.exit(main())

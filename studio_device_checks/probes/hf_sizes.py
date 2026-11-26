"""Bytes a check may download, and how much of it is already in the Hugging Face cache.
usage: hf_sizes.py --out FILE --spec SPEC.json   SPEC: [{"repo": R, "patterns": ["transformer/*", ...]}, ...]"""

from __future__ import annotations

import fnmatch
import json
import os
import sys

import _dc


def main() -> None:
    out = _dc.out_path()
    args = sys.argv[3:]
    with open(args[args.index("--spec") + 1], encoding = "utf-8") as fh:
        spec = json.load(fh)
    rec = {"repos": [], "total_bytes": 0, "cached_bytes": 0}
    try:
        from huggingface_hub import HfApi, try_to_load_from_cache

        api = HfApi()
        token = os.environ.get("HF_TOKEN") or None
        for item in spec:
            repo, pats = item["repo"], item.get("patterns") or ["*"]
            entry = {"repo": repo, "bytes": 0, "cached": 0, "files": 0}
            try:
                info = api.model_info(repo, files_metadata = True, token = token)
                for s in info.siblings or []:
                    if not any(fnmatch.fnmatch(s.rfilename, p) for p in pats):
                        continue
                    size = int(s.size or 0)
                    entry["bytes"] += size
                    entry["files"] += 1
                    alts = [repo] + list(item.get("cache_aliases") or [])
                    if any(isinstance(try_to_load_from_cache(r, s.rfilename), str) for r in alts):
                        entry["cached"] += size
            except Exception as exc:  # noqa: BLE001
                entry["error"] = _dc.err(exc, 300)
            rec["repos"].append(entry)
            rec["total_bytes"] += entry["bytes"]
            rec["cached_bytes"] += entry["cached"]
    except Exception as exc:  # noqa: BLE001
        rec["fatal"] = _dc.err(exc)
    _dc.write(out, rec)


if __name__ == "__main__":
    main()

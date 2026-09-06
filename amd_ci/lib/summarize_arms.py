#!/usr/bin/env python3
"""One table over the differential's obs_<state>.json files and every control
arm's, through the criteria module's `table()`, plus each arm's fingerprint.

  python amd_ci/lib/summarize_arms.py "$AMD_CI_WORK/out" \\
      amd_ci/criteria/llamacpp_server_clean.py --dirs D1 controls
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def gather(root: Path, dirs: list[str]) -> dict[str, dict]:
    obs: dict[str, dict] = {}
    for d in dirs:
        for f in sorted((root / d).glob("obs_*.json")):
            try:
                doc = json.loads(f.read_text(encoding = "utf-8"))
                obs[doc.get("state") or f.stem[len("obs_"):]] = doc
            except Exception as e:  # noqa: BLE001
                obs[f.stem[len("obs_"):]] = {"setup_error": f"unreadable: {type(e).__name__}: {e}"}
    return obs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type = Path)
    ap.add_argument("criteria", type = Path)
    ap.add_argument("--dirs", nargs = "*", default = ["D1", "controls"])
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("amd_ci_criteria", args.criteria)
    crit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(crit)

    obs = gather(args.root, args.dirs)
    print("## Every arm, one table\n")
    print(crit.table(obs) if obs else "(no observations)")
    print("\nfingerprints:\n")
    for name, doc in obs.items():
        fp = doc.get("fingerprint") or {}
        print(f"- {name}: sha256 {str(fp.get('llama-server_sha256', '?'))[:16]}; "
              f"libs {fp.get('backend_libs')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

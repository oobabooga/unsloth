#!/usr/bin/env python3
"""Turn the fetch reports into `NAME_BIN=<dir>` lines for $GITHUB_ENV.

A three-line job for jq, except jq is not guaranteed on this runner and a
missing jq under `set -e` fails the step with a message about jq rather than
about the fetch. Reading the file in python also means a malformed report says
so instead of yielding an empty variable that only fails three steps later.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    for report in sorted(out_dir.glob("bin_*.json")):
        name = report.stem[len("bin_"):].upper()
        try:
            info = json.loads(report.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"# {report.name}: unreadable ({type(e).__name__}: {e})")
            continue
        if info.get("bin_dir"):
            print(f"{name}_BIN={info['bin_dir']}")
        else:
            print(f"# {report.name}: no bin_dir ({info.get('error', 'unknown')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Print `<NAME>_BIN=<dir>` for every `bin_<name>.json` report under a
directory, for GITHUB_ENV. A report with no binary prints a `#` comment, so
the missing variable fails at the state-resolution step.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    for report in sorted(out_dir.glob("bin_*.json")):
        # bash cannot expand `ROCM-HEAD_BIN`; keep the name an identifier.
        name = re.sub(r"[^A-Za-z0-9_]", "_", report.stem[len("bin_"):]).upper()
        try:
            info = json.loads(report.read_text(encoding = "utf-8"))
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

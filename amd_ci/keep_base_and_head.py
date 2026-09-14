#!/usr/bin/env python3
"""Trim a states.json to base and head only.

Each state in the end-to-end installer probe is a complete Studio dependency install, so
a third state buys a repeat of the head answer at the price of another half hour. merge is
covered against the same hardware by the constraints-resolution run.

A file rather than an inline snippet: a PowerShell here-string inside a `run:` block is
not parseable YAML, so the workflow linter cannot check the step at all.
"""

import json
import sys
from pathlib import Path

KEEP = ("base", "head")


def main() -> int:
    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding = "utf-8"))
    for key in ("commits", "paths"):
        data[key] = {s: v for s, v in data.get(key, {}).items() if s in KEEP}
    path.write_text(json.dumps(data, indent = 2), encoding = "utf-8")
    print("states kept:", list(data["paths"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

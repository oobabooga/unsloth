#!/usr/bin/env python3
"""Surface a verdict in the run list, and fail the job when there is no result.

The Differential step ends in `exit 0` so that a real finding (FIX_INCOMPLETE,
a confirmed regression) is reported rather than thrown away as a broken job.
The side effect was that a run which selected zero tests and returned
INCONCLUSIVE in 90 seconds showed a green tick, and the run list is what anyone
looks at before the artifact.

So: findings keep exit 0, non-results do not. VOID and INCONCLUSIVE mean the run
produced no answer, which is the same shape as a hardware gate missing, and the
toolkit's rule there is already that it fails rather than skips.

  python amd_ci/lib/announce.py <out-dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# A verdict that means "no answer was obtained", as opposed to an unwelcome one.
NO_RESULT = {"VOID", "INCONCLUSIVE"}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: announce.py <out-dir>", file = sys.stderr)
        return 2
    p = Path(sys.argv[1]) / "verdict.json"
    if not p.is_file():
        print(f"::error::no verdict.json under {sys.argv[1]}: the differential did not "
              f"complete, so this job has no result")
        return 1

    d = json.loads(p.read_text())
    verdict, why = d.get("verdict", "?"), d.get("why", "")
    if verdict in NO_RESULT:
        print(f"::error::{verdict} - {why}. This run produced no answer; a green tick here "
              f"would mean only that the harness executed.")
        return 1
    print(f"::notice::{verdict} - {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

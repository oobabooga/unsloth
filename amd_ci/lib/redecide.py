#!/usr/bin/env python3
"""Re-run the gates and the verdict over a finished run's observations.json,
without re-running the probes. For a harness fix where the observations are
sound. Writes VERDICT_redecided.md beside the original, which is left as is.

  python amd_ci/lib/redecide.py --dir out/D1 --criteria amd_ci/criteria/x.py --title "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import differential  # noqa: E402


def redecide(out_dir: Path, criteria: Path, title: str = "") -> tuple[str, str]:
    crit = differential.load_criteria(criteria)
    obs = json.loads((out_dir / "observations.json").read_text(encoding = "utf-8"))
    gates = list(getattr(crit, "gates", lambda o: [])(obs))
    table = getattr(crit, "table", lambda o: "")(obs)
    verdict, why = differential._decide(crit, obs, gates)
    lines = [f"## {title or getattr(crit, 'TITLE', 'AMD CI differential')} (re-decided offline)", ""]
    if gates:
        lines += ["| gate | ok | evidence |", "|---|---|---|"]
        lines += [f"| {n} | {'yes' if ok else 'NO'} | {ev} |" for n, ok, ev in gates]
        lines.append("")
    if table and verdict != "INCONCLUSIVE":
        lines += [table, ""]
    lines.append(f"**{verdict}** - {why}")
    (out_dir / "VERDICT_redecided.md").write_text("\n".join(lines) + "\n", encoding = "utf-8")
    return verdict, why


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required = True, type = Path, help = "a differential's --out-dir")
    ap.add_argument("--criteria", required = True, type = Path)
    ap.add_argument("--title", default = "")
    args = ap.parse_args()
    verdict, why = redecide(args.dir, args.criteria, args.title)
    print(f"{verdict} - {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Re-run the gates and the verdict over a finished cell's observations.json,
without the probes. For a harness fix (the tuple-truthiness bug in _decide, a
gate that could not read the host) where the observations themselves are
sound. Writes VERDICT_redecided.md beside the original and never touches it.

  python amd_ci/lib/redecide.py --dir out/D1 --criteria amd_ci/criteria/x.py --title "..."
"""
from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import differential  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required = True, type = Path)
    ap.add_argument("--criteria", required = True)
    ap.add_argument("--title", default = "")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("crit", a.criteria)
    crit = importlib.util.module_from_spec(spec); spec.loader.exec_module(crit)
    obs = json.loads((a.dir / "observations.json").read_text())
    gates = list(getattr(crit, "gates", lambda o: [])(obs))
    table = getattr(crit, "table", lambda o: "")(obs)
    verdict, why = differential._decide(crit, obs, gates)
    lines = [f"## {a.title or getattr(crit, 'TITLE', 'redecided')} (re-decided offline)", ""]
    if gates:
        lines += ["| gate | ok | evidence |", "|---|---|---|"]
        lines += [f"| {n} | {'yes' if ok else 'NO'} | {ev} |" for n, ok, ev in gates]
        lines.append("")
    if table:
        lines += [table, ""]
    lines.append(f"**{verdict}** - {why}")
    (a.dir / "VERDICT_redecided.md").write_text("\n".join(lines) + "\n")
    print(f"{verdict} - {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

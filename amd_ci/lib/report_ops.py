#!/usr/bin/env python3
"""Render the test-backend-ops observations as a markdown table.

Deliberately verdict-free. This measurement has no base and no head, so calling
it CONFIRMED or NO_REGRESSION would be inventing a comparison that was never
made. It is a fact about the host, printed next to the controls that make the
fact readable, and a human draws the conclusion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# iq4_nl is the suspect; the rest are context. iq4_xs matters most: it appears in
# a GGUF that works in the field, so "iq4_nl fails while iq4_xs passes in the
# same invocation" is a signal, and "everything fails" is a broken invocation.
WATCH = ["iq4_nl", "iq4_xs", "iq3_s", "q3_K", "q4_K", "q5_K", "q6_K", "q8_0"]


def render(out_dir: Path) -> str:
    files = sorted(out_dir.glob("ops_*.json"))
    if not files:
        return "\n## Kernel characterisation\n\nNo observations were produced.\n"

    lines = ["", "## Kernel characterisation (test-backend-ops)", "",
             "No base, no head, so no verdict: this is what the host does, not a comparison.",
             ""]
    for f in files:
        try:
            obs = json.loads(f.read_text())
        except Exception as e:  # noqa: BLE001
            lines.append(f"- `{f.name}`: unreadable ({type(e).__name__}: {e})")
            continue

        backend = obs.get("backend", f.stem)
        if obs.get("setup_error"):
            lines += [f"### {backend}", "",
                      f"Not run: {obs['setup_error']}. `test-backend-ops` is not shipped in "
                      f"every prebuilt, so this is a gap in coverage rather than a result.", ""]
            continue

        lines += [f"### {backend}", "", "| type | " +
                  " | ".join(obs.get("ops", [])) + " |",
                  "|---" * (1 + len(obs.get("ops", []))) + "|"]
        watched = obs.get("watched_types") or {}
        for t in WATCH:
            per_op = watched.get(t)
            if not per_op:
                continue
            cells = []
            for op in obs.get("ops", []):
                counts = per_op.get(op)
                cells.append("-" if not counts else
                             " ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
            lines.append(f"| `{t}` | " + " | ".join(cells) + " |")
        lines.append("")
        failing = obs.get("failing_types") or []
        lines.append(f"Failing types: **{', '.join(failing) if failing else 'none'}**")
        for case in (obs.get("results") or {}).values():
            for fail in (case.get("failures") or [])[:10]:
                lines.append(f"    {fail}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(render(Path(sys.argv[1] if len(sys.argv) > 1 else ".")))

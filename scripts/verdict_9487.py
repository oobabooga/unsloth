#!/usr/bin/env python3
"""Turn the 9487 probe output into a verdict, refusing vacuous passes.

The head leg passing is not the result. The result is the base leg FAILING and
the head leg passing, on the same asset in the same run. If the base leg did not
flatten the links, this harness did not reproduce #9268 and has nothing to say
about whether the fix works.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(d: Path, state: str) -> dict | None:
    f = d / f"9487_{state}.json"
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text())
    except Exception as e:  # noqa: BLE001
        return {"parse_error": f"{type(e).__name__}: {e}"}


def main() -> int:
    out = Path(sys.argv[1])
    states = {s: load(out, s) for s in ("base", "head", "merge")}
    lines: list[str] = ["## PR 9487, sd.cpp symlink extraction", ""]

    b, h = states.get("base"), states.get("head")
    if not b or not h:
        lines.append("**VOID** - a required leg is missing, so no comparison exists.")
        print("\n".join(lines))
        return 0

    n_links = b.get("symlink_member_count") or h.get("symlink_member_count") or 0
    lines += [
        f"Asset `{h.get('asset')}`, sha256 `{(h.get('asset_sha256') or '')[:16]}...`, "
        f"{h.get('member_count')} members, **{n_links} symlink members**.", "",
        "| state | install | links restored | links flattened | file too short | ldd missing |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("base", "head", "merge"):
        s = states.get(name)
        if not s:
            continue
        for i in s.get("installs", []):
            if not i.get("install_ok"):
                lines.append(f"| {name} | {i['n']} | install FAILED: "
                             f"{i.get('install_error','')[:80]} | | | |")
                continue
            lines.append(
                f"| {name} | {i['n']} | {i.get('links_that_are_symlinks')}/{n_links} "
                f"| {i.get('links_flattened_to_files')} "
                f"| {'yes' if i.get('saw_file_too_short') else 'no'} "
                f"| {'yes' if i.get('ldd_missing_libs') else 'no'} |"
            )
    lines.append("")

    def first(s):
        ins = (s or {}).get("installs") or []
        return ins[0] if ins else {}

    base1, head_ins = first(b), (h.get("installs") or [])
    base_reproduced = base1.get("links_flattened_to_files", 0) > 0
    head_ok = bool(head_ins) and all(
        i.get("install_ok") and i.get("links_that_are_symlinks") == n_links
        and not i.get("links_flattened_to_files")
        for i in head_ins
    )
    head_stable = len(head_ins) >= 2 and len({
        tuple(sorted((r.get("member"), r.get("target_sha256")) for r in i.get("links", [])))
        for i in head_ins
    }) == 1

    lines.append(f"- base reproduced the defect: **{base_reproduced}** "
                 f"({base1.get('links_flattened_to_files', 0)} links written as flat files)")
    lines.append(f"- head restored every link on every install: **{head_ok}**")
    lines.append(f"- head stable across reinstalls: **{head_stable}**")
    lines.append("")

    if not base_reproduced:
        lines.append("**VOID** - the base leg did not flatten the links, so this run did not "
                     "reproduce #9268 and cannot speak to the fix.")
    elif head_ok and head_stable:
        lines.append("**CONFIRMED** - the defect reproduces on the merge base and is gone at "
                     "the head, across three installs into the same tree.")
    else:
        lines.append("**FIX INCOMPLETE** - the base reproduced the defect but the head did not "
                     "fully satisfy the link checks.")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

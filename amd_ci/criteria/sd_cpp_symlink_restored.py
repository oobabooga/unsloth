#!/usr/bin/env python3
"""Criteria: does the ROCm sd.cpp install actually start, and did it fail to before?

Judges only. Pairs with sd_cpp_rocm_probe.py.

The defect (#9268) is `sd-cli` unable to load its libraries because the archive's symlink
members were written as regular files holding the target text. So the defect is read off the
loader, not off the tree: `ldd` saying "file too short", or a non-zero `sd-cli --help`.

Non-vacuity matters here more than usual. If the asset ever stops shipping symlink members
the probe would show a clean base and the run would say VOID, which is correct: there would
be nothing to fix. The gate makes that explicit rather than leaving it to be inferred.
"""

from __future__ import annotations

TITLE = "sd.cpp ROCm install on gfx1151: are the archive's symlinks restored?"
MODE = "differential"
NEEDS: list[str] = []


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ran = "error" not in o and o.get("archive_symlinks") is not None
        out.append((f"{name} probe produced a reading", ran,
                    o.get("error") or f"archive_symlinks={o.get('archive_symlinks')}"))
    # Non-vacuity: the asset must actually carry symlink members, or there is no defect to see.
    head = obs.get("head") or {}
    n = head.get("archive_symlinks") or 0
    out.append(("the ROCm asset still ships symlink members", n > 0,
                f"{n} symlink members in {head.get('asset', 'the asset')}"))
    return out


def _passes(o: dict) -> list[dict]:
    return o.get("per_pass") or []


def table(obs: dict) -> str:
    rows = ["| state | pass | links restored | flattened | ldd 'file too short' | sd-cli --help |",
            "|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        want = o.get("archive_symlinks", 0)
        for p in _passes(o):
            rc = p.get("rc")
            rc_s = "raised: " + str(p["raised"])[:40] if p.get("raised") else f"rc={rc}"
            rows.append(
                f"| {name} | {p.get('pass')} | {p.get('restored', 0)}/{want} "
                f"| {p.get('flattened', 0)} | {p.get('file_too_short')} | {rc_s} |")
    stderr0 = ((obs.get("base") or {}).get("stderr0") or "").strip()
    note = f"\n\nBase loader error: `{stderr0}`" if stderr0 else ""
    return "\n".join(rows) + note


def base_shows_defect(base: dict) -> bool:
    """The reported failure: libraries flattened, and sd-cli cannot start."""
    ps = _passes(base)
    if not ps:
        return False
    want = base.get("archive_symlinks") or 0
    # Every pass must show it; a base that works even once is not the reported state.
    for p in ps:
        broken_tree = p.get("restored", 0) < want and p.get("flattened", 0) > 0
        broken_load = bool(p.get("file_too_short")) or (p.get("rc") not in (0,))
        if not (broken_tree and broken_load):
            return False
    return True


def head_is_fixed(head: dict) -> bool:
    """Every link restored, nothing flattened, no dangling link, and sd-cli starts.

    Checked on every pass, not just the first: extraction merges into the managed directory,
    so a fix that works once and eats the tree on reinstall is not a fix.
    """
    ps = _passes(head)
    if not ps:
        return False
    want = head.get("archive_symlinks") or 0
    if want == 0:
        return False
    for p in ps:
        if p.get("raised"):
            return False
        if p.get("restored", 0) != want:
            return False
        if p.get("flattened", 0) != 0 or p.get("dangling", 0) != 0:
            return False
        if p.get("file_too_short"):
            return False
        if p.get("rc") != 0:
            return False
    return True

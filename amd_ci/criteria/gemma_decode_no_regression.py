#!/usr/bin/env python3
"""Criteria: Gemma / Gemma2 generation through Unsloth is no further from transformers eager at the head.

Pairs with gemma_decode_probe.py. Regression mode: the head fails if any case's max
|logit| difference to eager grows past max(TOL, 1.5 x base), if a batched row stops
reproducing its single-row tokens where the base did, or if any PR test fails.
"""

from __future__ import annotations

TITLE = "Gemma / Gemma2 decode vs transformers eager, base versus head"
MODE = "regression"
NEEDS: list[str] = ["rocm", "gpu", "nvidia", "windows"]
TOL = 0.25


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        info = o.get("info") or {}
        cases = o.get("cases") or {}
        out.append((f"{name} generated and scored both archs", set(cases) == {"gemma", "gemma2"},
                    o.get("error") or f"archs {sorted(cases)}"))
        # A case that crashes counts as infinitely far from eager in head_is_worse, so a crash shared
        # with the base is equal and a new one is a regression; no separate gate needed.
        out.append((f"{name} imported unsloth from its own checkout", bool(info.get("unsloth_from_checkout")),
                    str(info.get("unsloth_file"))))
        out.append((f"{name} ran on ROCm", bool(info.get("hip")),
                    f"hip={info.get('hip')} device={info.get('device')} torch={info.get('torch')}"))
    h = obs.get("head") or {}
    if h.get("tests_present"):
        p = h.get("pytest") or {}
        n = len(p.get("passed", [])) + len(p.get("failed", []))
        out.append(("head PR tests executed", n > 0, f"rc={h.get('pytest_rc')} ran={n} skipped={len(p.get('skipped', []))}"))
    return out


def _worst(case: dict) -> tuple[float, float]:
    # A crashed case counts as infinitely far from eager.
    b = case.get("batched_max_diff_per_row")
    s = case.get("single_max_diff") or []
    bw = float("inf") if b is None else max(b)
    sw = float("inf") if any(x is None for x in s) or not s else max(s)
    return bw, sw


def _fmt(xs):
    return None if xs is None else [None if x is None else round(x, 4) for x in xs]


def table(obs: dict) -> str:
    rows = [f"Pass bar per case: head max |logit diff| <= max({TOL}, 1.5 x base); a crashed case counts as inf.", "",
            "| state | arch | batched max diff per row | single max diff | batched == single tokens | flash softcap | flash decode | transformers |",
            "|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        for arch, c in ((obs.get(name) or {}).get("cases") or {}).items():
            for e in c.get("errors", []):
                rows.insert(1, f"{name} {arch} error: `{e[:160]}`")
    for name in ("base", "head"):
        o = obs.get(name) or {}
        info = o.get("info") or {}
        for arch, c in (o.get("cases") or {}).items():
            rows.append(
                f"| {name} | {arch} | {_fmt(c['batched_max_diff_per_row'])} | "
                f"{_fmt(c['single_max_diff'])} | {c['batched_tokens_match_single']} | "
                f"{info.get('has_flash_softcapping')} | {info.get('flash_decode_gate')} | {info.get('transformers')} |"
            )
    h = obs.get("head") or {}
    if h.get("pytest"):
        p = h["pytest"]
        rows.append("")
        rows.append(f"Head PR tests: {len(p['passed'])} passed, {len(p['failed'])} failed, {len(p['skipped'])} skipped"
                    + (": " + ", ".join(f"`{f.split('::')[-1]}`" for f in p["failed"]) if p["failed"] else ""))
        if p["skipped"]:
            rows.append("Skipped: " + ", ".join(f"`{s.split('::')[-1]}`" for s in p["skipped"]))
    return "\n".join(rows)


def _rows(case: dict, key: str, n: int) -> list[float]:
    v = case.get(key)
    if v is None:
        return [float("inf")] * n
    return [float("inf") if x is None else x for x in v]


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    # Row by row; a crashed row counts as inf, so a crash shared with the base is equal.
    worse, notes = [], []
    for arch in ("gemma", "gemma2"):
        b, h = (base.get("cases") or {}).get(arch), (head.get("cases") or {}).get(arch)
        if not b or not h:
            continue
        n = len(b.get("single_max_diff") or h.get("single_max_diff") or [])
        for label, key in (("batched", "batched_max_diff_per_row"), ("single", "single_max_diff")):
            for i, (bv, hv) in enumerate(zip(_rows(b, key, n), _rows(h, key, n))):
                tag = f"{arch} {label} row {i}"
                if hv == float("inf") and bv == float("inf"):
                    notes.append(f"{tag} crashes at both states")
                elif hv > max(TOL, 1.5 * bv):
                    worse.append(f"{tag} {bv:.4f} -> {hv:.4f}")
                elif bv > TOL and hv <= TOL:
                    notes.append(f"{tag} fixed {bv:.4f} -> {hv:.4f}")
    failed = (head.get("pytest") or {}).get("failed") or []
    if failed:
        worse.append("PR tests failing: " + ", ".join(f"`{f.split('::')[-1]}`" for f in failed))
    if worse:
        return True, "; ".join(worse)
    return False, "no row further from eager at the head" + ("; " + "; ".join(notes) if notes else "")

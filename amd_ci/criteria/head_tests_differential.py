#!/usr/bin/env python3
"""Criteria: do the PR's own tests fail on the base's code and pass on the head's?

Pairs with probes/head_tests_probe.py, which runs the HEAD's test files against every
state. Differential: the base must fail at least one of them (else VOID), and the head
(and merge, when present) must fail none and pass at least one. Compared by test id.

Also tabulates the per-state host observations (Studio GPU detection, MiniMax-H3 host
guard) so a reader sees whether detection moved between base and head. Those are
reported, not judged here: "unchanged or widened" is read off the table.
"""

from __future__ import annotations

import json

TITLE = "PR tests (head's files) on base vs head code"
MODE = "differential"
# Every capability the change touches: the PRs target NVIDIA CUDA offload on discrete
# cards first, Windows ROCm for detection, and this host is one integrated gfx1151.
NEEDS = ["gpu", "rocm", "nvidia", "discrete_gpu", "windows", "windows_rocm_wddm",
         "linux", "multi_gpu"]

_RAN = (0, 1)


def _bad(o: dict) -> list[str]:
    return sorted(set(o.get("failed", [])) | set(o.get("errors", [])))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        rc = o.get("rc")
        ran = rc in _RAN
        detail = o.get("error") or f"rc={rc}"
        if not ran and o.get("stderr_tail"):
            detail += f"; stderr: {str(o['stderr_tail'])[-200:]}"
        out.append((f"{name}: pytest ran", ran, detail))
        n = sum(o.get(k, 0) or 0 for k in ("n_passed", "n_failed", "n_errors", "n_skipped"))
        out.append((f"{name}: tests collected", n > 0,
                    f"{n} (passed {o.get('n_passed', 0)}, failed {o.get('n_failed', 0)}, "
                    f"errors {o.get('n_errors', 0)}, skipped {o.get('n_skipped', 0)})"))
    b = obs.get("base") or {}
    ov = b.get("overlay") or {}
    sel, want = set(b.get("selected") or []), set(b.get("tests") or [])
    out.append(("base ran the HEAD's test files (overlay applied)",
                not b.get("is_head_source") and bool(want) and sel == want
                and not ov.get("missing_at_head"),
                f"{len(sel)}/{len(want)} selected present; this run added {len(ov.get('added', []))}, "
                f"replaced {len(ov.get('replaced', []))}"))
    return out


def _ids(items: list[str]) -> str:
    return "<br>".join(f"`{i}`" for i in items) or "none"


def table(obs: dict) -> str:
    rows = ["| state | passed | failed | errors | skipped |", "|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if o:
            rows.append(f"| {name} | {o.get('n_passed', 0)} | {o.get('n_failed', 0)} | "
                        f"{o.get('n_errors', 0)} | {o.get('n_skipped', 0)} |")
    parts = ["\n".join(rows), ""]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        bad = _bad(o)
        parts.append(f"**{name}: failing / erroring test ids ({len(bad)})**\n")
        parts.append("\n".join(f"- `{t}`: {str((o.get('messages') or {}).get(t, ''))[:160]}"
                               for t in bad) or "- none")
        parts.append("")
    head_skips = (obs.get("head") or {}).get("skipped") or []
    if head_skips:
        msgs = (obs.get("head") or {}).get("messages") or {}
        parts.append(f"**head: skipped ({len(head_skips)})**\n")
        parts.append("\n".join(f"- `{t}`: {str(msgs.get(t, ''))[:120]}" for t in head_skips))
        parts.append("")
    det = {n: (obs.get(n) or {}).get("detection") for n in ("base", "head") if obs.get(n)}
    if any(det.values()):
        def strip(d):
            return {k: v for k, v in (d or {}).items() if k != "rc"}
        same = strip(det.get("base")) == strip(det.get("head"))
        parts.append(f"**Studio GPU detection, base vs head: {'IDENTICAL' if same else 'DIFFERS'}**\n")
        for n, d in det.items():
            parts.append(f"- {n}: `{json.dumps(strip(d), sort_keys = True)[:900]}`")
        parts.append("")
    h3 = {n: (obs.get(n) or {}).get("h3_guard") for n in ("base", "head") if obs.get(n)}
    if any(h3.values()):
        parts.append("**MiniMax-H3 Diffusers int8 host-RAM guard on this host (streamed TE + DiT)**\n")
        parts.append("| state | ok | host total / avail GB | VRAM free GB | required host GB | refusal | fit tiers |")
        parts.append("|---|---|---|---|---|---|---|")
        for n, h in h3.items():
            h = h or {}
            parts.append(f"| {n} | {h.get('ok')} | {h.get('host_total_gb')} / {h.get('host_available_gb')} "
                         f"| {h.get('vram_free_gb')} | {h.get('required_host_gb')} "
                         f"| {h.get('refusal') or h.get('error') or 'none (would proceed)'} "
                         f"| {json.dumps(h.get('fit_tiers'))} |")
        parts.append("")
    return "\n".join(parts)


def base_shows_defect(base: dict) -> bool:
    return bool(_bad(base))


def head_is_fixed(head: dict) -> bool:
    return not _bad(head) and (head.get("n_passed", 0) or 0) > 0

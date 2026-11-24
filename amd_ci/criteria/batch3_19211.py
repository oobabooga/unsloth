#!/usr/bin/env python3
"""PR 19211: a MOSS-Image forward captured in a HIP graph, replayed after the RoPE FIFO cache evicted its tables and
the allocator reused them, must equal its eager output. Differential: the base must diverge."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import base_gates, states  # noqa: E402

TITLE = "PR 19211: MOSS-Image RoPE tables under HIP graph replay (tiny random model, gfx1151)"
MODE = "differential"
NEEDS = ["rocm", "gpu"]
REL = 1e-2


def _rel(c, k):
    return c[k] / max(c["ref_absmax"], 1e-12)


def gates(obs):
    out = base_gates(obs)
    for name, o in states(obs).items():
        g = o.get("graph") or {}
        per = g.get("per_length") or {}
        out.append((f"{name}: graphs captured and replayed", bool(per) and "error" not in g,
                    str(g.get("error", ""))[-400:] or f"{len(per)} lengths"))
        out.append((f"{name}: eager reference non-trivial", bool(per) and all(c["ref_absmean"] > 1e-4 for c in per.values()),
                    ", ".join(f"{c['ref_absmean']:.3g}" for c in per.values())))
        out.append((f"{name}: first replay (right after capture) equals eager (harness sane)",
                    bool(per) and all(_rel(c, "first_replay_max_abs") <= REL for c in per.values()),
                    ", ".join(f"{c['first_replay_max_abs']:.3g}" for c in per.values())))
        if name != "base":
            out.append((f"{name}: uncached-under-capture path ran", (g.get("capturing_true_calls") or 0) > 0,
                        f"_capturing True {g.get('capturing_true_calls')} times"))
    return out


def table(obs):
    rows = ["| state | Studio graph gate | cache entries | late replay max_abs per length | diverged |", "|---|---|---|---|---|"]
    for name, o in states(obs).items():
        g = o.get("graph") or {}
        per = g.get("per_length") or {}
        bad = _diverged(o)
        lates = ", ".join("%s:%.3g" % (L, c["late_replay_max_abs"]) for L, c in per.items())
        rows.append(f"| {name} | {o.get('studio_graph_eligible')} | {g.get('cache_entries_after_capture')} "
                    f"| {lates} | {bad} |")
    return "\n".join(rows)


def _diverged(o):
    per = (o.get("graph") or {}).get("per_length") or {}
    return [L for L, c in per.items() if _rel(c, "late_replay_max_abs") > REL or not c["late_finite"]]


def base_shows_defect(base):
    bad = _diverged(base)
    return bool(bad), f"base replay diverged from eager at text lengths {bad}"


def head_is_fixed(head):
    bad = _diverged(head)
    n = len((head.get("graph") or {}).get("per_length") or {})
    return (not bad and n > 0), (f"diverged at {bad}" if bad else f"all {n} replays equal eager")

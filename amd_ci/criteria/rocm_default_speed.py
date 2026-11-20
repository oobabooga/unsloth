#!/usr/bin/env python3
"""Criteria: with UNSLOTH_EXPERIMENTAL_HYBRID_PACKING unset, gfx1151 packs a Qwen3.5-style hybrid by default
(head) where main keeps the padded path; packed must beat the =0 padded run with matching losses."""

from __future__ import annotations

import statistics

TITLE = "gfx1151 hybrid packing default (variable unset), base versus head"
MODE = "differential"
NEEDS = ["rocm", "gpu", "integrated_gpu", "linux", "exclusive_gpu"]
MIN_GAIN = 1.15


def _states(obs):
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _rec(o, mode):
    return (o.get("modes") or {}).get(mode) or {}


def _tps(rec):
    t = [s for s in rec.get("steps") or [] if s["step"] >= 3 and not s["profiled"]]
    return statistics.median(s["tokens"] / s["s"] for s in t) if t else None


def _packed(rec):
    return (rec.get("hybrid_stats") or {}).get("patched", 0) > 0 and (rec.get("hf_packed_stats") or {}).get("fast", 0) > 0


def gates(obs):
    out = []
    for name, o in _states(obs).items():
        for mode in ("default", "off"):
            r = _rec(o, mode)
            out.append((f"{name}/{mode}: trained", r.get("rc") == 0 and not r.get("error") and bool(r.get("steps")),
                        str(r.get("error"))[:200]))
        off = _rec(o, "off")
        out.append((f"{name}/off: padded rows", bool(off.get("steps")) and all(s["shape"][0] > 1 for s in off["steps"]),
                    str({tuple(s["shape"][:1]) for s in off.get("steps") or []})))
        out.append((f"{name}: ROCm", bool(_rec(o, "default").get("hip")), str(_rec(o, "default").get("hip"))))
    return out


def table(obs):
    rows = ["| state | mode | packed | median useful tok/s | median s/step | first losses |", "|---|---|---|---|---|---|"]
    for name, o in _states(obs).items():
        for mode in ("default", "off"):
            r = _rec(o, mode)
            st = sorted(s["s"] for s in r.get("steps") or [] if s["step"] >= 3)
            rows.append(f"| {name} | {mode} | {_packed(r)} | {_tps(r)} | {st[len(st) // 2] if st else '-'} | "
                        f"{[round(s['loss'], 4) for s in (r.get('steps') or [])[:4]]} |")
        rows.append(f"| {name} | selfcheck | {o.get('selfcheck')} | | | |")
    return "\n".join(rows)


def base_shows_defect(base):
    return not _packed(_rec(base, "default"))


def head_is_fixed(head):
    d, off = _rec(head, "default"), _rec(head, "off")
    td, to = _tps(d), _tps(off)
    return _packed(d) and bool(td and to) and td >= MIN_GAIN * to

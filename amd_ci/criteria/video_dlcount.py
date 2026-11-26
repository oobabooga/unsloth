#!/usr/bin/env python3
"""Criteria for probes/video_dlcount_probe.py: base reads 0 downloaded bytes while the hosted pre-cast encoder's
partial grows on disk; head counts that growth. Gated on the download actually moving bytes and each state importing
its own checkout, so a stalled network or a cross-imported tree cannot pass."""
from __future__ import annotations

TITLE = "Windows gfx1151 box: video download counter during the hosted pre-cast encoder fetch, base versus head"
MODE = "differential"
NEEDS = ["windows"]
MIN_GROWTH = 50 * 1024 * 1024


def _disk(o):
    s = o.get("samples") or []
    return max((x.get("disk_st_size") or 0 for x in s), default = 0)


def _counted(o):
    s = o.get("samples") or []
    return max((x.get("downloaded") or 0 for x in s), default = 0)


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name}: probe ran", not o.get("error") and bool(o.get("samples")), str(o.get("error") or "")[-300:]))
        out.append((f"{name}: own checkout imported", name in str(o.get("file") or ""), str(o.get("file"))))
        out.append((f"{name}: partial grew on disk (>= 50 MiB)", _disk(o) >= MIN_GROWTH, f"{_disk(o)} bytes"))
    return out


def base_shows_defect(base):
    return (_counted(base) == 0 and _disk(base) >= MIN_GROWTH,
            f"counted {_counted(base)} while disk reached {_disk(base)}; expected {base.get('expected')}")


def head_is_fixed(head):
    disk, counted = _disk(head), _counted(head)
    s = head.get("samples") or []
    advancing = len({x.get("downloaded") for x in s if x.get("downloaded")}) >= 2
    ok = counted >= MIN_GROWTH and counted >= 0.5 * disk and advancing and (head.get("expected") or 0) > 7 * 10**9
    return ok, f"counted {counted} of disk {disk}; distinct readings advancing {advancing}; expected {head.get('expected')}"


def table(obs):
    rows = ["| state | t (s) | phase | downloaded | expected | disk st_size | disk present |", "|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        for x in (obs.get(name) or {}).get("samples") or []:
            rows.append(f"| {name} | {x.get('t')} | {x.get('phase')} | {x.get('downloaded')} | {x.get('expected')} | "
                        f"{x.get('disk_st_size')} | {x.get('disk_present')} |")
    return "\n".join(rows)

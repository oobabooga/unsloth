#!/usr/bin/env python3
"""Criteria for probes/pr19936_win_probe.py (PR 19936 on the WINDOWS gfx1151 runners, Qwen-Image-2.1 loads).

Windows differs from Linux: main already reads converting loads through pread on win32, and when the PR's pressure
gate picks mmap the Windows commit rule still applies, so with plenty of memory the head must route every shard
exactly as main does. Gates (a miss = non-result): gfx1151, smoke, >= 2 ok loads per arm.
Judged (head_is_worse):
  auto_plenty   per-shard routing identical to main's; median load s <= slowest main load * 1.05; median peak
                working set <= main's max + 1 GiB; if the gate engaged, it decided mmap
  auto_pressure only judged when the ballast really held MemAvailable under the switch point at load start:
                the gate (if engaged) decided pread and peak working set is not above auto_plenty's
  checksums     transformer / text_encoder / vae identical on every load of every arm
Reported, never gating: whether the standby list could be purged (cold vs warm loads), the gate engagement on Windows.
"""

from __future__ import annotations

import statistics as st
from collections import Counter

TITLE = "PR 19936 WINDOWS gfx1151: safetensors reader gate (Qwen-Image-2.1 loads)"
MODE = "regression"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "windows"]
ARMS = ("main", "auto_plenty", "auto_pressure", "pread_forced", "mmap_forced")
CHECKSUMS = ("transformer_checksum", "text_encoder_checksum", "vae_checksum")


def _loads(h):
    return [r for r in h.get("loads") or [] if r.get("arm") in ARMS]


def _ok(r):
    return r.get("verdict") == "ok" and r.get("rc") == 0


def _by(h):
    out = {a: [] for a in ARMS}
    for r in _loads(h):
        out[r["arm"]].append(r)
    return out


def _reader(r):
    d = r.get("reader_decision")
    return d.get("reader") if isinstance(d, dict) else None


def _med(rs, key):
    xs = [r.get(key) for r in rs if isinstance(r.get(key), (int, float))]
    return round(st.median(xs), 3) if xs else None


def _routing(r):
    # effective choices only: None = the caller's default (mmap), whether or not a scope asked
    return Counter((c.get("file"), c.get("backend")) for c in r.get("backend_calls") or [] if c.get("backend") is not None)


def _pressure_held(h):
    sw = (h.get("pressure_plan") or {}).get("switch_point_gib")
    pre = [r.get("mem_available_before_load_gib") for r in _by(h)["auto_pressure"]]
    return bool(sw) and bool(pre) and all(isinstance(x, (int, float)) and 0 < x < sw for x in pre), sw, pre


def gates(obs):
    h = obs.get("head") or {}
    arch = (h.get("env") or {}).get("arch") or ""
    by = _by(h)
    return [("gfx1151", arch.startswith("gfx1151"), arch),
            ("smoke", bool(h.get("smoke_ok")), str(h.get("smoke"))[:500] + " " + str(h.get("smoke_error", ""))[-300:]),
            (">= 2 ok loads per arm", all(len([r for r in by[a] if _ok(r)]) >= 2 for a in ARMS),
             str({a: len([r for r in by[a] if _ok(r)]) for a in ARMS}) + " " + str(h.get("gate_error", ""))[-300:])]


def head_is_worse(base, head):
    by = _by(head)
    why = []
    main, plenty, press = by["main"], by["auto_plenty"], by["auto_pressure"]
    engaged = bool(head.get("gate_engages_here"))
    room = all(_reader(r) != "pread" or "headroom" not in str((r.get("reader_decision") or {}).get("reason", ""))
               or ">=" in str((r.get("reader_decision") or {}).get("reason", "")) for r in plenty)
    if not room:
        why.append(f"auto_plenty: host lacked room (gate saw pressure without ballast: "
                   f"{[(r.get('reader_decision') or {}).get('reason') for r in plenty]}); routing check not applicable")
    if main and plenty and room:
        ref = _routing(main[0])
        for r in main[1:] + plenty:
            if _routing(r) != ref:
                why.append(f"{r.get('arm')} load {r.get('i')}: shard routing differs from main's first load "
                           f"({sorted((_routing(r) - ref).items())[:4]} extra, {sorted((ref - _routing(r)).items())[:4]} missing)")
        if not any(r.get("backend_calls") for r in plenty):
            why.append("auto_plenty: no shard went through the reader routing (path not engaged)")
    if engaged and [_reader(r) for r in plenty] != ["mmap"] * len(plenty):
        why.append(f"auto_plenty gate decided {[_reader(r) for r in plenty]}, expected mmap")
    main_s = [r.get("load_s") for r in main if r.get("load_s")]
    mp = _med(plenty, "load_s")
    if not main_s or mp is None or mp > 1.05 * max(main_s):
        why.append(f"auto_plenty median load {mp} s above main's slowest {main_s} * 1.05")
    wm = [r.get("load_vmhwm_gib") for r in main if isinstance(r.get("load_vmhwm_gib"), (int, float))]
    wp = _med(plenty, "load_vmhwm_gib")
    if not wm or wp is None or wp > max(wm) + 1.0:
        why.append(f"auto_plenty median peak working set {wp} GiB above main's max {wm} + 1 GiB")
    held, sw, pre = _pressure_held(head)
    if held:
        if engaged and [_reader(r) for r in press] != ["pread"] * len(press):
            why.append(f"auto_pressure (MemAvailable {pre} < switch {sw}) decided {[_reader(r) for r in press]}")
        wq = _med(press, "load_vmhwm_gib")
        if wq is not None and wp is not None and wq > wp + 1.0:
            why.append(f"auto_pressure peak working set {wq} GiB above auto_plenty {wp} GiB")
    for key in CHECKSUMS:
        vals = {r.get(key) for r in _loads(head) if _ok(r)}
        if len(vals) != 1 or None in vals:
            why.append(f"{key} differs or missing across loads: {sorted(map(str, vals))}")
    if why:
        return True, "; ".join(why)
    return False, (f"auto_plenty routes as main on every load, median {mp} s vs main {main_s}, peak WS {wp} vs main "
                   f"{wm} GiB; gate engages on Windows: {engaged}; pressure held: {held} (switch {sw}, at load {pre}); "
                   f"checksums identical on all loads")


def table(obs):
    h = obs.get("head") or {}
    held, sw, pre = _pressure_held(h)
    L = [f"env {h.get('env')} platform {h.get('platform')}", f"trees {h.get('tree_sha')}",
         f"gate engages on Windows (rocm_is_integrated): {h.get('gate_engages_here')}",
         f"pressure plan {h.get('pressure_plan')}; pressure held at every auto_pressure load: {held} ({pre})",
         f"smoke pytest rc {(h.get('smoke_pytest') or {}).get('rc')}: {str((h.get('smoke_pytest') or {}).get('tail', ''))[-300:]}", "",
         "| i | arm | reader | reason | load s | peak WS GiB | MemAvail at load GiB | standby purged | sys cache before -> after purge GiB | pread shards | transformer checksum |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in h.get("loads") or []:
        d = r.get("reader_decision")
        wp = r.get("win_purge") or {}
        L.append(f"| {r.get('i')} | {r.get('arm')} | {_reader(r)} | {(d or {}).get('reason') if isinstance(d, dict) else d} | "
                 f"{r.get('load_s')} | {r.get('load_vmhwm_gib')} | {r.get('mem_available_before_load_gib')} | "
                 f"{wp.get('purged')} | {wp.get('cache_before_gib')} -> {wp.get('cache_after_gib')} | "
                 f"{r.get('pread_shards')} | {r.get('transformer_checksum')} |")
    L.append("")
    first = next((r.get("win_purge") for r in h.get("loads") or [] if r.get("win_purge")), None)
    L.append(f"first purge record: {first}")
    for r in h.get("loads") or []:
        if r.get("ballast_ready") or r.get("ballast_stop"):
            L.append(f"ballast {r.get('i')}: ready {r.get('ballast_ready')} stop {r.get('ballast_stop')}")
    L.append("")
    L.append("Per-shard reader routing (component: backends seen) and every gate decision:")
    for r in h.get("loads") or []:
        comp = {}
        for c in r.get("backend_calls") or []:
            comp.setdefault(str(c.get("file")).split("/")[0], set()).add(str(c.get("backend")))
        L.append(f"- {r.get('i')} {r.get('arm')}: {dict((k, sorted(v)) for k, v in sorted(comp.items()))}; gate "
                 f"{[(d.get('shard'), d.get('pread')) for d in r.get('gate_decisions') or []]}"
                 + ("" if _ok(r) else f"; FAILED rc {r.get('rc')} {str(r.get('error'))[:300]} {str(r.get('log_tail'))[-600:]}"))
    L.append("")
    by = _by(h)
    for a in ARMS:
        L.append(f"{a}: load s {[r.get('load_s') for r in by[a]]} median {_med(by[a], 'load_s')}; peak WS median "
                 f"{_med(by[a], 'load_vmhwm_gib')}; logged {[ln[-160:] for r in by[a] for ln in r.get('reader_log_lines') or []]}")
    return "\n".join(L)

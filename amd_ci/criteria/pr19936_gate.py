#!/usr/bin/env python3
"""Criteria for probes/pr19936_probe.py (PR 19936: pread on a ROCm integrated GPU only under host-memory pressure).

Gates (a miss = non-result): gfx1151, smoke, all 15 cold loads ok and cold, the ballast really held MemAvailable under
the gate's switch point for every pressure load, the no-ballast arms really had room.
Judged (head_is_worse = the PR misses its claim):
  auto_plenty   decides mmap on every load and routes no shard (transformer, text encoder, VAE) through pread, median load s inside main's A/A band (<= slowest main load)
  auto_pressure decides pread on every load, median peak RSS >= 10 GiB under auto_plenty's
  checksums     transformer / text_encoder / vae identical on every load of every arm
"""

from __future__ import annotations

import statistics as st

TITLE = "PR 19936 gfx1151: safetensors pread gated on host-memory pressure (Qwen-Image-2.1 cold loads)"
MODE = "regression"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "windows"]
ARMS = ("main", "auto_plenty", "auto_pressure", "pread_forced", "mmap_forced")
CHECKSUMS = ("transformer_checksum", "text_encoder_checksum", "vae_checksum")


def _loads(h: dict) -> list:
    return [r for r in h.get("loads") or [] if r.get("arm") in ARMS]


def _ok(r: dict) -> bool:
    return r.get("verdict") == "ok" and r.get("rc") == 0


def _by(h: dict) -> dict:
    out: dict = {a: [] for a in ARMS}
    for r in _loads(h):
        out[r["arm"]].append(r)
    return out


def _reader(r: dict):
    d = r.get("reader_decision")
    return d.get("reader") if isinstance(d, dict) else None


def gates(obs: dict) -> list:
    h = obs.get("head") or {}
    arch = (h.get("env") or {}).get("arch") or ""
    out = [("gfx1151", arch.startswith("gfx1151"), arch),
           ("smoke", bool(h.get("smoke_ok")), str(h.get("smoke"))[:400] + " " + str(h.get("smoke_error", ""))[-300:])]
    by = _by(h)
    out.append(("15 cold loads ok", all(len([r for r in by[a] if _ok(r)]) == 3 for a in ARMS),
                str({a: len([r for r in by[a] if _ok(r)]) for a in ARMS}) + " " + str(h.get("gate_error", ""))[-300:]))
    cold = [r.get("resident_before") for r in _loads(h)]
    out.append(("page cache dropped before every load", bool(cold) and all(c is not None and c < 0.05 for c in cold),
                str(cold)))
    plan = h.get("pressure_plan") or {}
    sw = plan.get("switch_point_gib")
    press = [(r.get("ballast_ready") or {}).get("mem_available_gib") for r in by["auto_pressure"]]
    pre_load = [r.get("mem_available_before_load_gib") for r in by["auto_pressure"]]
    out.append(("ballast held MemAvailable under the switch point (pressure loads)",
                bool(sw) and len(pre_load) == 3 and all(x is not None and 0 < x < sw for x in pre_load + press),
                f"switch {sw} GiB, ballast READY {press}, at load start {pre_load}"))
    room = [r.get("mem_available_before_load_gib") for a in ARMS if a != "auto_pressure" for r in by[a]]
    out.append(("no-ballast loads had room", bool(sw) and all(x is not None and x > sw + 10 for x in room),
                f"switch {sw} GiB, at load start {room}"))
    return out


def _med(rs, key):
    xs = [r.get(key) for r in rs if isinstance(r.get(key), (int, float))]
    return st.median(xs) if xs else None


def head_is_worse(base: dict, head: dict) -> tuple:
    by = _by(head)
    why = []
    plenty, press, main = by["auto_plenty"], by["auto_pressure"], by["main"]
    if [_reader(r) for r in plenty] != ["mmap"] * 3:
        why.append(f"auto_plenty decided {[_reader(r) for r in plenty]}, expected mmap x3")
    for r in plenty:
        calls, gd = r.get("backend_calls") or [], r.get("gate_decisions") or []
        if not calls:
            why.append(f"auto_plenty load {r.get('i')}: no shard went through the reader routing (path not engaged)")
        bad = [c for c in calls if c.get("backend") == "pread"]
        if bad or r.get("pread_shards") or any(d.get("pread") for d in gd):
            why.append(f"auto_plenty load {r.get('i')}: shards read through pread {[c.get('file') for c in bad]}, "
                       f"gate decisions {[d.get('pread') for d in gd]}")
    for r in press:
        gd = r.get("gate_decisions") or []
        if not gd or not all(d.get("pread") for d in gd):
            why.append(f"auto_pressure load {r.get('i')}: gate decisions {[d.get('pread') for d in gd]}")
    main_s = [r.get("load_s") for r in main if r.get("load_s")]
    if not main_s or _med(plenty, "load_s") is None or _med(plenty, "load_s") > max(main_s):
        why.append(f"auto_plenty median load {_med(plenty, 'load_s')} s outside main's A/A band {main_s}")
    if [_reader(r) for r in press] != ["pread"] * 3:
        why.append(f"auto_pressure decided {[_reader(r) for r in press]}, expected pread x3")
    rp, rq = _med(plenty, "load_vmhwm_gib"), _med(press, "load_vmhwm_gib")
    if rp is None or rq is None or rp - rq < 10.0:
        why.append(f"peak RSS saving {None if rp is None or rq is None else round(rp - rq, 2)} GiB < 10 GiB")
    for key in CHECKSUMS:
        vals = {r.get(key) for r in _loads(head)}
        if len(vals) != 1 or None in vals:
            why.append(f"{key} differs or missing across loads: {sorted(map(str, vals))}")
    if why:
        return True, "; ".join(why)
    return False, (f"auto_plenty mmap x3, median {_med(plenty, 'load_s')} s vs main {main_s}; auto_pressure pread x3, "
                   f"peak RSS {rq} vs {rp} GiB (saves {round(rp - rq, 2)}); checksums identical on all 15 loads")


def table(obs: dict) -> str:
    h = obs.get("head") or {}
    L = [f"env {h.get('env')}", f"trees {h.get('tree_sha')} (PR head at submit {h.get('pr_head_expected_at_submit')})", f"pressure plan {h.get('pressure_plan')}", "",
         "| i | arm | reader | reason | load s | peak RSS GiB | cache growth GiB | MemAvail at load GiB | pread shards | "
         "transformer checksum |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in h.get("loads") or []:
        d = r.get("reader_decision")
        L.append(f"| {r.get('i')} | {r.get('arm')} | {_reader(r)} | {(d or {}).get('reason') if isinstance(d, dict) else d} | "
                 f"{r.get('load_s')} | {r.get('load_vmhwm_gib')} | {r.get('cache_growth_gib')} | "
                 f"{r.get('mem_available_before_load_gib')} | {r.get('pread_shards')} | {r.get('transformer_checksum')} |")
    L.append("")
    L.append("Per-shard reader routing (component: backends seen) and every gate decision:")
    for r in h.get("loads") or []:
        comp: dict = {}
        for c in r.get("backend_calls") or []:
            comp.setdefault(str(c.get("file")).split("/")[0], set()).add(str(c.get("backend")))
        L.append(f"- {r.get('i')} {r.get('arm')}: {dict((k, sorted(v)) for k, v in sorted(comp.items()))}; gate "
                 f"{[(d.get('shard'), d.get('pread')) for d in r.get('gate_decisions') or []]}")
    L.append("")
    by = _by(h)
    for a in ARMS:
        L.append(f"{a}: load s {[r.get('load_s') for r in by[a]]} median {_med(by[a], 'load_s')}; peak RSS median "
                 f"{_med(by[a], 'load_vmhwm_gib')}; cache growth median {_med(by[a], 'cache_growth_gib')}; "
                 f"logged {[ln[-160:] for r in by[a] for ln in r.get('reader_log_lines') or []]}")
    return "\n".join(L)

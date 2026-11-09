#!/usr/bin/env python3
"""Criteria: do Studio's Windows ROCm display polls share one Get-Counter spawn per TTL?

Pairs with probes/rocm_win_counter_cache_probe.py (PR 17306). The defect is that
`_rocm_windows_perf_counter_vram_by_adapter` / `_gpu_util_pct` start a
`powershell Get-Counter` (~2 s on the 8060S boxes) on every call, display polls
included. The head reads all three counters in ONE Get-Counter call in display
mode ("combined" in the probe's counts), which serves every counter, and reuses
it for WINDOW = min(10 s, max(display TTL, 4x the duration of that call)).

  base shows the defect   inside display_reads(), every counter spawns about once
                          per call (>= n - 1 of n calls).
  head is fixed           ALL of:
    display     per counter, its own spawns plus combined ones (the effective
                count) <= 2 per WINDOW of the phase's wall time (WINDOW from the
                fastest combined spawn in the phase, so the bound is loose),
                and fewer than the base made
    fresh       no scope: spawns per counter == the base's and no combined spawn
                (fit / placement reads keep their own call; a cache or the
                combined read leaking out of the display scope fails here)
    fresh-in-display   display_reads() + fresh_reads(): == the base's
    invalidation       warm read, re-read without a bump spawns 0, re-read after
                gpu_memory_events.invalidate_gpu_memory spawns 1
    values      the last display-phase reading against a fresh read taken right
                after it: same adapter instance names; bytes within TOL; the
                utilization reading present on both or neither
    pytest      no test failing / erroring at the head that did not at the base,
                and more passing tests than the base (the PR adds 5, so they ran)
    public      for each of get_gpu_utilization / get_visible_gpu_utilization that
                reached a Get-Counter on this box in BOTH states: the display and
                fresh rules above, plus payload shape equal to a fresh call. Paths
                that never reached a counter are reported as not engaged, not passed.

TOL for VRAM bytes is the larger of 128 MiB and 4x the A/A drift: the largest
change between consecutive FRESH reads of the same adapter counter, measured on
this box in this job across every state's n-read fresh phase. Live counters move
(the compositor, other processes), so equality would fail a correct cache; tying
the slack to the measured drift keeps it from swallowing a real mismatch, and the
128 MiB floor is far below any figure the System tab rounds to (0.1 GB).
"""

from __future__ import annotations

import math

TITLE = "Windows ROCm Get-Counter reuse for display polls (PR 17306)"
MODE = "differential"
# The change lives on the Windows ROCm WDDM path; the cache key is per counter,
# not per adapter, so several AMD adapters and a discrete card are untested here.
NEEDS = ["windows", "windows_rocm_wddm", "multi_gpu_amd", "discrete_gpu"]

COUNTERS = ("vram_dedicated", "vram_shared", "util")
PUBLIC = ("get_gpu_utilization", "get_visible_gpu_utilization")
MIB = 1024 ** 2
FLOOR = 128 * MIB
COMBINED = "combined"
SPACING = 4.0
MAX_WINDOW_S = 10.0
NEW_TEST_FILE = "tests/test_rocm_windows_counter_display_cache.py"

_CTX: dict = {"base": {}, "tol": FLOOR, "drift": 0.0}


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _sp(phase: dict | None, counter: str) -> int:
    return int(((phase or {}).get("spawns") or {}).get(counter, 0))


def _eff(phase: dict | None, counter: str) -> int:
    """Spawns that served ``counter``: its own plus every combined one."""
    return _sp(phase, counter) + _sp(phase, COMBINED)


def _window(phase: dict, ttl: float) -> float:
    durs = [float(d) for d in (phase.get("combined_dur_s") or []) if d is not None]
    if not durs:
        return ttl
    return min(MAX_WINDOW_S, max(ttl, SPACING * min(durs)))


def _total(spawns: dict | None) -> int:
    return sum((spawns or {}).values())


def _allowed(phase: dict, ttl: float) -> int:
    return 2 * max(1, math.ceil(float(phase.get("wall_s") or 0.0) / max(ttl, 1e-6)))


def _drift(obs: dict) -> float:
    worst = 0.0
    for st in _states(obs).values():
        vals = (((st.get("direct") or {}).get("fresh") or {}).get("values")) or []
        for key in ("vram_dedicated", "vram_shared"):
            prev: dict = {}
            for v in vals:
                rows = (v or {}).get(key) if isinstance(v, dict) else None
                cur = {r[0]: float(r[1]) for r in (rows or [])}
                for inst, b in cur.items():
                    if inst in prev:
                        worst = max(worst, abs(b - prev[inst]))
                prev = cur
    return worst


def _pytest_ran(p: dict) -> tuple[bool, str]:
    rc = p.get("rc")
    n = sum(int(p.get(k) or 0) for k in ("n_passed", "n_failed", "n_skipped", "n_error"))
    # Something must have PASSED: a suite that only errors at import ran nothing.
    ok = rc in (0, 1) and n > 0 and int(p.get("n_passed") or 0) > 0
    return ok, (f"rc={rc}, passed {p.get('n_passed', 0)}, failed {p.get('n_failed', 0)}, "
                f"skipped {p.get('n_skipped', 0)}, errors {p.get('n_error', 0)}"
                + (f"; {p.get('error') or p.get('note')}" if (p.get('error') or p.get('note')) else ""))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    base = obs.get("base") or {}
    _CTX["base"] = base
    drift = _drift(obs)
    _CTX["drift"] = drift
    _CTX["tol"] = max(FLOOR, 4 * drift)
    for name, st in _states(obs).items():
        err = st.get("error") or st.get("_parse_error") or (
            "probe wrote no output" if st.get("_missing_output") else None)
        out.append((f"{name}: probe completed", not err and "direct" in st,
                    (err or "ok")[:300]))
        d = st.get("direct") or {}
        n = int(st.get("n") or 0)
        fresh = d.get("fresh") or {}
        counts = {c: _sp(fresh, c) for c in COUNTERS}
        # Engagement: every counter really ran Get-Counter once per fresh read. A
        # box whose platform check or PowerShell never reaches it reads 0 here.
        out.append((f"{name}: counter path engaged (fresh reads spawn Get-Counter)",
                    n > 0 and all(v == n for v in counts.values()),
                    f"n={n}, spawns {counts}"))
        ref = ((d.get("fresh_after_display") or {}).get("values") or [{}])[0] or {}
        ded = ref.get("vram_dedicated") if isinstance(ref, dict) else None
        out.append((f"{name}: Dedicated Usage returned adapters", bool(ded),
                    f"{len(ded or [])} adapter(s): "
                    + ", ".join(f"{r[0]}={float(r[1]) / MIB:.0f} MiB" for r in (ded or []))[:300]))
        out.append((f"{name}: gpu_query display cache enabled",
                    st.get("gpu_query_cache_enabled") is True,
                    f"enabled={st.get('gpu_query_cache_enabled')}, ttl={st.get('display_ttl_s')}"))
        ok, ev = _pytest_ran(st.get("pytest") or {})
        out.append((f"{name}: PR tests ran", ok, ev))
    # The states must differ in the thing being tested, or the comparison is A/A.
    head = obs.get("head") or {}
    out.append(("base lacks the counter cache, head has it",
                base.get("has_counter_cache") is False and head.get("has_counter_cache") is True,
                f"base={base.get('has_counter_cache')}, head={head.get('has_counter_cache')}"))
    # The PR head is resolved on GitHub at run time: an older head (per-counter TTL
    # cache, no combined read) would measure the design this run exists to replace.
    out.append(("head has the combined display read",
                head.get("has_combined_read") is True and base.get("has_combined_read") is not True,
                f"base={base.get('has_combined_read')}, head={head.get('has_combined_read')}"))
    return out


def _shape_mismatch(a, b, key: str, tol_gb: float, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path or '/'} keys differ: {sorted(set(a) ^ set(b))}"]
        out: list[str] = []
        for k in a:
            out += _shape_mismatch(a[k], b[k], k, tol_gb, f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path} length {len(a)} vs {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _shape_mismatch(x, y, key, tol_gb, f"{path}[{i}]")
        return out
    num = (int, float)
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str):
        # Presence must agree; a None on one side is a reading lost or invented.
        if (a is None) != (b is None):
            return [f"{path} {a!r} vs {b!r}"]
        if isinstance(a, str) and a != b:
            return [f"{path} {a!r} vs {b!r}"]
        return []
    if isinstance(a, num) and isinstance(b, num):
        k = key.lower()
        if "total" in k and abs(a - b) > max(0.01 * abs(b), 1e-6):
            return [f"{path} total {a} vs {b}"]
        if "used" in k and k.endswith("gb") and abs(a - b) > tol_gb:
            return [f"{path} used {a} vs {b} (tol {tol_gb:.3f} GB)"]
        return []
    if type(a) is not type(b):
        return [f"{path} type {type(a).__name__} vs {type(b).__name__}"]
    return []


def _direct_values(st: dict, tol: float) -> tuple[bool, str]:
    d = st.get("direct") or {}
    disp = ((d.get("display") or {}).get("values") or [None])[-1]
    ref = ((d.get("fresh_after_display") or {}).get("values") or [None])[0]
    if not isinstance(disp, dict) or not isinstance(ref, dict):
        return False, "no display or fresh reading to compare"
    bad, worst = [], 0.0
    for key in ("vram_dedicated", "vram_shared"):
        a = {r[0]: float(r[1]) for r in (disp.get(key) or [])}
        b = {r[0]: float(r[1]) for r in (ref.get(key) or [])}
        if set(a) != set(b):
            bad.append(f"{key} instances differ: {sorted(set(a) ^ set(b))}")
            continue
        for inst in a:
            delta = abs(a[inst] - b[inst])
            worst = max(worst, delta)
            if delta > tol:
                bad.append(f"{key} {inst} off by {delta / MIB:.0f} MiB")
    if (disp.get("util") is None) != (ref.get("util") is None):
        bad.append(f"util presence differs: {disp.get('util')!r} vs {ref.get('util')!r}")
    if bad:
        return False, "; ".join(bad)[:400]
    return True, (f"instances equal, max delta {worst / MIB:.1f} MiB <= tol {tol / MIB:.0f} MiB; "
                  f"util {disp.get('util')} vs fresh {ref.get('util')}")


def _checks(st: dict) -> list[tuple[str, bool, str]]:
    base = _CTX["base"]
    tol = _CTX["tol"]
    ttl = float(st.get("display_ttl_s") or 3.0)
    d, bd = st.get("direct") or {}, base.get("direct") or {}
    out: list[tuple[str, bool, str]] = []

    disp, bdisp = d.get("display") or {}, bd.get("display") or {}
    window = _window(disp, ttl)
    allowed = _allowed(disp, window)
    per = {c: _eff(disp, c) for c in COUNTERS}
    bper = {c: _eff(bdisp, c) for c in COUNTERS}
    out.append(("display: <= 2 effective spawns per counter per window, fewer than base",
                all(per[c] <= allowed and per[c] < bper[c] for c in COUNTERS),
                f"{per} ({disp.get('spawns')}) in {disp.get('wall_s')} s, window {window:.2f} s "
                f"(allowed {allowed}); base {bper} in {bdisp.get('wall_s')} s"))
    for phase, label in (("fresh", "fresh (no scope) spawns == base"),
                         ("display_fresh", "display_reads + fresh_reads spawns == base")):
        h = {c: _sp(d.get(phase), c) for c in COUNTERS + (COMBINED,)}
        b = {c: _sp(bd.get(phase), c) for c in COUNTERS + (COMBINED,)}
        out.append((label, h == b and h[COMBINED] == 0 and _total(h) > 0, f"{h} vs base {b}"))
    inv = d.get("invalidation") or {}
    w, c0, a = (_total(inv.get("warm_spawns")), _total(inv.get("no_bump_spawns")),
                _total(inv.get("after_bump_spawns")))
    out.append(("memory event invalidates (warm 1, no-bump 0, after-bump 1)",
                (w, c0, a) == (1, 0, 1), f"warm {w}, no bump {c0}, after bump {a}"))
    out.append(("display values match a fresh read",) + _direct_values(st, tol))

    p, bp = st.get("pytest") or {}, base.get("pytest") or {}
    hf = set(p.get("failed") or []) | set(p.get("errors") or [])
    bf = set(bp.get("failed") or []) | set(bp.get("errors") or [])
    new = sorted(hf - bf)
    in_new_file = sorted(f for f in hf if NEW_TEST_FILE in f.replace("\\", "/"))
    more = int(p.get("n_passed") or 0) > int(bp.get("n_passed") or 0)
    out.append(("pytest: no new failures, PR tests pass and ran",
                not new and not in_new_file and more,
                f"passed {p.get('n_passed', 0)} vs base {bp.get('n_passed', 0)}; "
                f"new failures {[n.split('::')[-1] for n in new] or 'none'}"))

    pub, bpub = st.get("public") or {}, base.get("public") or {}
    both = [f for f in PUBLIC if f in (pub.get("engaged") or []) and f in (bpub.get("engaged") or [])]
    for f in PUBLIC:
        if f not in both:
            out.append((f"public {f}", True,
                        f"NOT ENGAGED (reached no Get-Counter: head engaged={pub.get('engaged')}, "
                        f"base engaged={bpub.get('engaged')}, device={pub.get('device')}, "
                        f"is_rocm={pub.get('is_rocm')}); not evidence either way"))
            continue
        h, b = pub.get(f) or {}, bpub.get(f) or {}
        hd, bdp = h.get("display") or {}, b.get("display") or {}
        window = _window(hd, ttl)
        allowed = _allowed(hd, window)
        # The counters the BASE read on this path; a combined spawn serves each of them.
        counters = sorted(c for c in (bdp.get("spawns") or {}) if c != COMBINED)
        hper = {c: _eff(hd, c) for c in counters}
        bper = {c: _eff(bdp, c) for c in counters}
        out.append((f"public {f}: display spawns bounded, fewer than base",
                    all(hper[c] <= allowed and hper[c] < bper[c] for c in counters) and bool(counters),
                    f"{hper} ({hd.get('spawns')}) in {hd.get('wall_s')} s, window {window:.2f} s "
                    f"(allowed {allowed}); base {bper}"))
        hfz, bfz = (h.get("fresh") or {}).get("spawns") or {}, (b.get("fresh") or {}).get("spawns") or {}
        out.append((f"public {f}: fresh spawns == base, none combined",
                    hfz == bfz and _total(hfz) > 0 and COMBINED not in hfz,
                    f"{hfz} vs base {bfz}"))
        last = (hd.get("values") or [None])[-1]
        ref = ((h.get("fresh_after_display") or {}).get("values") or [None])[0]
        mm = _shape_mismatch(last, ref, "", tol / 1024 ** 3)
        out.append((f"public {f}: display payload matches a fresh call",
                    last is not None and not mm, "; ".join(mm)[:400] or "same shape, totals, used within tol"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | check | ok | evidence |", "|---|---|---|---|"]
    for name, st in _states(obs).items():
        if name == "base":
            continue
        for label, ok, ev in _checks(st):
            rows.append(f"| {name} | {label} | {'yes' if ok else 'NO'} | {ev} |")
    base = obs.get("base") or {}
    bd = (base.get("direct") or {}).get("display") or {}
    rows += ["", f"Base display phase: {bd.get('spawns')} Get-Counter spawns for "
                 f"{base.get('n')} calls in {bd.get('wall_s')} s (the defect: one per call)."]
    for name, st in _states(obs).items():
        pub = st.get("public") or {}
        disc = {f: (v.get("second") or {}).get("spawns") for f, v in (pub.get("discovery") or {}).items()}
        env = st.get("env") or {}
        rows.append(f"{name}: public paths engaged={pub.get('engaged')} (discovery spawns {disc}); "
                    f"device={pub.get('device')}, IS_ROCM={pub.get('is_rocm')}; "
                    f"torch={' '.join(str(env.get('torch')).split())[-300:]}")
    rows.append(f"\nVRAM tolerance {_CTX['tol'] / MIB:.0f} MiB = max(128 MiB, 4 x A/A drift "
                f"{_CTX['drift'] / MIB:.1f} MiB between consecutive fresh reads).")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    n = int(base.get("n") or 0)
    disp = (base.get("direct") or {}).get("display") or {}
    per = {c: _eff(disp, c) for c in COUNTERS}
    ok = n > 1 and all(v >= n - 1 for v in per.values())
    return ok, f"base display spawns {per} for {n} calls"


def head_is_fixed(head: dict) -> tuple[bool, str]:
    checks = _checks(head)
    bad = [label for label, ok, _ in checks if not ok]
    return (not bad), ("failed: " + ", ".join(bad)) if bad else "every check held"

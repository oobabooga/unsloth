#!/usr/bin/env python3
"""Criteria: does Unsloth DESKTOP collapse where the Studio web UI did, and on what renderer?

Judges only. Observations come from probes/desktop_ladder_probe.py.

Scored to be DIRECTLY COMPARABLE with the web UI ladder on the same host, which means using the
same frame statistic, and that statistic was settled by experiment rather than argument. The web
UI run's first published table used `GdkFrameClock::after-paint`, on the reasoning that it is one
emission per presented frame while rAF under a headless X server is not vsync locked. The
reasoning is sound and the conclusion was wrong: the driver calls `begin_updating()`, which ticks
the clock at the display rate whether or not anything asked for a frame, and a jammed control
with the main thread blocked 200 ms out of every 250 ms still read 60.0 fps in every phase. It
cannot move, in either direction, so it cannot carry a claim.

The headline here is therefore the EFFECTIVE rAF rate -- callbacks delivered over the window's
wall time, which is what studiobench's own scoring/frames.py calls `effective_fps`. Not
`1000/p50`: a bursty block leaves the median untouched, and the same jammed control demonstrated
exactly that, reporting an identical p50 jammed and unjammed while its effective rate fell by
more than two thirds. Desktop has no after-paint channel at all -- the GTK frame clock belongs
to the app's own process and nothing outside it can connect to that signal -- which is a
limitation only of a column the web UI run should not have quoted either.

Three things decide the verdict, in order, and the order matters.

1. **Did Desktop actually function?** Not "the process started": the app shell has to have
   replaced the startup screen, the seeded thread has to have opened, its messages have to be in
   the DOM keyed on the seeder's own last-turn marker, and a reply has to have streamed. A
   Desktop that sits on its installer screen produces a clean, flat, meaningless table.

2. **What rendered?** Read from amdgpu's per-process fdinfo counters for the app's own web
   process, differenced per `drm-client-id`, against a software negative control run on the same
   binary and the same X server with `LIBGL_ALWAYS_SOFTWARE=1` the only difference. No in-page
   string is consulted, because WebKitGTK hardcodes `Apple GPU` into WEBGL_debug_renderer_info on
   Linux, on AMD, in cross-platform WebCore, and the runtime unmask switch was deleted upstream.
   Beside it, the rendering decision `linux_webkit.rs` made and the inputs it made it from, read
   out of `/proc/<pid>/environ` of the live process -- because main.rs only LOGS that decision
   when a workaround is applied, so silence there is ambiguous between "no workaround needed" and
   "crashed before the log line".

3. **Only then the ladder**, and only through a channel a jammed positive control shows can move.
"""

from __future__ import annotations

TITLE = "Does Unsloth Desktop (Tauri) collapse from 0K to 500K on the AMD CI runner?"
MODE = "capability"

NEEDS = [
    "linux", "drm_render_node", "egl_hardware_gl", "webkitgtk", "headless_display_server",
    "desktop_tauri_app", "desktop_gpu_compositing", "gpu_browser_compositing",
    "studio_production_bundle", "studiobench_ladder", "rust_toolchain", "tauri_build_deps",
    # Declared because the QUESTION touches them. Desktop on Windows renders through WebView2
    # and on macOS through WKWebView; neither is this engine and neither is reachable here.
    "windows", "windows_webview2", "macos", "macos_wkwebview", "wayland_session",
    "discrete_gpu", "nvidia", "mlx",
]

ORDER = ["0K", "1K", "10K", "100K", "500K", "1M"]
MATERIAL_DROP_PCT = 10.0
REPORTED_FLOOR_FPS = 5.0
MIN_DOM_GROWTH = 5.0
MIN_TOP_BUSY_PCT = 15.0
CONTROL_MIN_DROP_PCT = 25.0
# How far below the real leg the SOFTWARE leg's GFX engine time has to fall before the real
# leg's counters are accepted as caused by the browser's rendering. A software rasteriser that
# still books GFX time means something else in the process tree is using the device.
SOFTWARE_MAX_FRACTION = 0.10


def _runs(obs):
    return [r for r in (obs.get("runs") or []) if isinstance(r, dict)]


def _bench(r):
    """The bench script's own record for a run."""
    return r.get("payload") or {}


def _scene(r):
    """The scene payload, i.e. what the page reported. Nested inside the bench record."""
    return (_bench(r).get("payload") or {})


def _is_hog(r):
    return bool(r.get("hog")) or str(r.get("rep")) == "hog"


def _is_software(r):
    return bool(r.get("software")) or str(r.get("rep")) == "sw"


def _is_control(r):
    return _is_hog(r) or _is_software(r)


def _ok_runs(obs):
    out = []
    for r in _runs(obs):
        if _is_control(r):
            continue
        if _bench(r).get("ok") and _scene(r).get("marks"):
            out.append(r)
    return out


def _find(obs, pred):
    for r in _runs(obs):
        if pred(r):
            return r
    return None


# ── frame statistics ────────────────────────────────────────────────────────────────────

def _raf_fps(scene: dict, phase: str):
    """EFFECTIVE frames per second: callbacks delivered over the window's wall time."""
    for ph in scene.get("phases") or []:
        if ph.get("phase") != phase:
            continue
        n = (ph.get("raf") or {}).get("n")
        el = ph.get("elapsed_ms")
        return (1000.0 * n / el) if n and el else None
    return None


def _raf_stat(scene, phase, key):
    for ph in scene.get("phases") or []:
        if ph.get("phase") == phase:
            return (ph.get("raf") or {}).get(key)
    return None


def _busy(scene, phase):
    for ph in scene.get("phases") or []:
        if ph.get("phase") == phase:
            return (ph.get("busy") or {}).get("busy_pct")
    return None


def _blocked(scene, phase):
    for ph in scene.get("phases") or []:
        if ph.get("phase") == phase:
            return (ph.get("busy") or {}).get("blocked_ms")
    return None


def _elements(scene):
    return (scene.get("final") or {}).get("elements")


def _by_rung(obs):
    out = {}
    for r in _ok_runs(obs):
        out.setdefault(r.get("rung", "?"), []).append(r)
    return out


def _rungs_sorted(obs):
    return sorted(_by_rung(obs), key = lambda r: ORDER.index(r) if r in ORDER else 99)


def _agg(rs, phase):
    vals = [v for v in (_raf_fps(_scene(r), phase) for r in rs) if v is not None]
    worst = [v for v in (_raf_stat(_scene(r), phase, "max_ms") for r in rs) if v is not None]
    busy = [v for v in (_busy(_scene(r), phase) for r in rs) if v is not None]
    blocked = [v for v in (_blocked(_scene(r), phase) for r in rs) if v is not None]
    els = [v for v in (_elements(_scene(r)) for r in rs) if v is not None]
    return {"fps": vals, "n": len(vals),
            "fps_min": min(vals) if vals else None, "fps_max": max(vals) if vals else None,
            "spread": (max(vals) - min(vals)) if len(vals) >= 2 else None,
            "worst_ms": max(worst) if worst else None,
            "busy": (sum(busy) / len(busy)) if busy else None,
            "blocked_ms": (sum(blocked) / len(blocked)) if blocked else None,
            "elements": (sum(els) / len(els)) if els else None}


def _control_fps(obs):
    """(jammed fps, unjammed fps at the same rung, the phase they are read from)."""
    c = _find(obs, _is_hog)
    if not c or not _bench(c).get("ok"):
        return None, None, "scroll"
    peers = [r for r in _ok_runs(obs) if r.get("rung") == c.get("rung")]
    best = (None, None, "scroll")
    for phase in ("scroll", "idle", "stream"):
        j = _raf_fps(_scene(c), phase)
        if j is None:
            continue
        u = [x for x in (_raf_fps(_scene(r), phase) for r in peers) if x is not None]
        if not u:
            continue
        if best[0] is None or j < best[0]:
            best = (j, sum(u) / len(u), phase)
    return best


# ── the rendering path ──────────────────────────────────────────────────────────────────

def _gfx_ns(r):
    return ((_bench(r).get("amdgpu") or {}).get("total_gfx_ns_delta"))


def _renderer_env(r):
    return (_bench(r).get("app") or {}).get("renderer_env") or {}


def _workaround_applied(r):
    e = _renderer_env(r)
    return {k: v for k, v in e.items()
            if k.startswith(("WEBKIT_", "UNSLOTH_WEBKIT")) and v is not None}


def gates(obs):
    out = []

    x = obs.get("xserver") or {}
    out.append(("a display server was claimed by THIS job", bool(x.get("display")),
                obs.get("fatal") or f"{x.get('binary')} on {x.get('display')} "
                                    f"(owned={x.get('owned')})"))

    il = obs.get("install_layout") or {}
    inst = obs.get("install") or {}
    out.append(("Studio installed into the layout the Tauri app requires",
                bool(il.get("cli_exists") and il.get("studio_install_id")),
                f"install rc={inst.get('rc')} in {inst.get('seconds')}s; "
                f"cli={il.get('cli_exists')} studio_install_id={il.get('studio_install_id')}"))

    bm = obs.get("build_manifest") or {}
    out.append(("the instrumented bundle and this probe agree on the control port",
                obs.get("control_port_matches_build", True) is not False,
                f"bundle={bm.get('control_port')}"))

    ladder = [r for r in _runs(obs) if not _is_control(r)]
    ok = _ok_runs(obs)
    out.append(("every ladder run completed", bool(ladder) and len(ok) == len(ladder),
                f"{len(ok)}/{len(ladder)} ok" + ("" if len(ok) == len(ladder) else "; " +
                "; ".join(f"{r.get('rung')}#{r.get('rep')}: "
                          f"{str(_bench(r).get('fatal') or _bench(r).get('error') or r.get('rc'))[:140]}"
                          for r in ladder if r not in ok))))

    # THE APP REALLY ATTACHED, rather than sitting on its installer screen. Every one of these
    # conditions fails silently into that screen, and a run measured there is flat and empty at
    # every rung.
    attach = [bool((_bench(r).get("attach_preconditions_after_provision") or {}).get("all_ok"))
              for r in ok]
    out.append(("the app took the AttachedReady path, not the installer",
                bool(attach) and all(attach),
                f"{sum(attach)}/{len(attach)} runs met every preflight precondition"))

    navs = [(_scene(r).get("__desktop") or {}).get("nav") or {} for r in ok]
    routes = sorted({n.get("via") for n in navs if n})
    out.append(("the seeded thread was opened, by the same route every time",
                bool(navs) and all(n.get("ok") for n in navs) and len(routes) == 1,
                f"routes used: {routes}"))

    binaries = {_bench(r).get("binary_sha256_16") for r in ok}
    out.append(("one binary across every rung", len(binaries) == 1, f"{sorted(binaries)}"))

    base, top = (_rungs_sorted(obs) or [None])[0], (_rungs_sorted(obs) or [None])[-1]
    grew, detail = False, "no runs"
    if base and top and base != top:
        b_el = _agg(_by_rung(obs)[base], "idle")["elements"]
        t_el = _agg(_by_rung(obs)[top], "idle")["elements"]
        if b_el and t_el:
            grew = (t_el / b_el) >= MIN_DOM_GROWTH
            detail = f"{base}: {b_el:,.0f} elements -> {top}: {t_el:,.0f} ({t_el / b_el:.1f}x)"
    out.append((f"the seeded thread really mounted (DOM grew >= {MIN_DOM_GROWTH:.0f}x)",
                grew, detail))

    # THE POSITIVE CONTROL. Without it a flat frame rate cannot be told from a channel that
    # cannot read anything else, which is precisely how the web UI's first table came to be an
    # artefact.
    jam, unjam, phase = _control_fps(obs)
    c = _find(obs, _is_hog)
    resolves = (jam is not None and unjam is not None
                and jam < unjam * (1.0 - CONTROL_MIN_DROP_PCT / 100.0))
    out.append((f"the headline frame channel resolves a jammed main thread "
                f"(>= {CONTROL_MIN_DROP_PCT:.0f}% below the same rung unjammed)", resolves,
                "no control run" if c is None else
                (f"control did not complete: "
                 f"{str(_bench(c).get('fatal') or _bench(c).get('error') or c.get('rc'))[:160]}"
                 if jam is None else
                 f"{c.get('rung')} with a {c.get('hog')} ms jam every "
                 f"{_bench(c).get('hog_period_ms', 250)} ms: {jam:.1f} fps against "
                 f"{unjam:.1f} fps unjammed, {phase} phase")))

    # DESKTOP REALLY FUNCTIONS: a reply streamed, and the interaction windows the users'
    # symptom lives in actually ran.
    streamed = sum(1 for r in ok
                   if _scene(r).get("first_token_ms") is not None
                   and not _scene(r).get("still_running_at_deadline"))
    acts = {}
    for r in ok:
        for a in _scene(r).get("actions") or []:
            if a.get("not_applicable"):
                continue
            acts.setdefault(a.get("name", "?"), []).append(bool(a.get("ok")))
    toggles = acts.get("reasoning_toggle") or []
    copies = acts.get("select_all_copy") or []
    works = bool(ok) and streamed == len(ok) and toggles and all(toggles) \
        and copies and all(copies)
    out.append(("Desktop really functions here: thread rendered, reply streamed, reasoning pane "
                "opened and closed, selection copied", works,
                f"streamed {streamed}/{len(ok)}; reasoning_toggle {sum(toggles)}/{len(toggles)}; "
                f"select_all_copy {sum(copies)}/{len(copies)}"))

    have_repeat = any(len(rs) >= 2 for rs in _by_rung(obs).values())
    out.append(("at least one rung was measured twice, so a difference has a floor", have_repeat,
                ", ".join(f"{k}x{len(v)}" for k, v in sorted(_by_rung(obs).items())) or "none"))
    return out


def _fmt(v, suffix="", nd=0):
    return "-" if v is None else f"{v:,.{nd}f}{suffix}"


def table(obs):
    rows = ["| rung | reps | DOM elements | phase | fps (effective rAF) | worst frame | busy | "
            "blocked |", "|---|---|---|---|---|---|---|---|"]
    by = _by_rung(obs)
    for rung in _rungs_sorted(obs):
        rs = by[rung]
        el = _agg(rs, "idle")["elements"]
        for phase in ("idle", "scroll", "stream"):
            a = _agg(rs, phase)
            if not a["fps"]:
                fps = "-"
            elif a["n"] == 1:
                fps = f"{a['fps_min']:.1f}"
            else:
                fps = f"{a['fps_min']:.1f}-{a['fps_max']:.1f}"
            rows.append("| " + " | ".join([
                rung, str(len(rs)), _fmt(el), phase, f"**{fps}**",
                _fmt(a["worst_ms"], " ms"),
                "null" if a["busy"] is None else f"{a['busy']:.0f}%",
                _fmt(a["blocked_ms"], " ms")]) + " |")

    rows += [
        "",
        "**fps is the EFFECTIVE rate: rAF callbacks delivered over the window's wall time, the "
        "same statistic studiobench's `scoring/frames.py` reports as `effective_fps`, and the "
        "same one the web UI ladder on this host is scored on.** It is deliberately not "
        "`1000/p50`, which a bursty block leaves untouched, and deliberately not the "
        "`GdkFrameClock::after-paint` series: that channel is driven by `begin_updating()` at "
        "the display rate whether or not a frame was requested, it read 60.0 fps in every phase "
        "with the main thread blocked 200 ms out of every 250 ms, and it is unavailable to a "
        "Tauri app from outside its process in any case.",
        "",
        "Repeat spread at the same rung, which is the only floor a rung-to-rung difference has:",
        "",
        "| rung | phase | readings (fps) | spread |", "|---|---|---|---|",
    ]
    for rung in _rungs_sorted(obs):
        for phase in ("idle", "scroll", "stream"):
            a = _agg(by[rung], phase)
            if a["spread"] is None:
                continue
            rows.append(f"| {rung} | {phase} | {', '.join(f'{v:.1f}' for v in a['fps'])} | "
                        f"{a['spread']:.1f} fps |")

    # The rendering path, which is the other half of the question.
    rows += ["", "### What actually rendered", "",
             "| leg | GFX engine ns accrued during the run | VRAM | render nodes | "
             "workaround applied by linux_webkit.rs |", "|---|---|---|---|---|"]
    for r in _runs(obs):
        b = _bench(r)
        amd = b.get("amdgpu") or {}
        vram = ", ".join(sorted({c.get("vram") or "" for c in amd.get("clients") or []})) or "-"
        label = f"{r.get('rung')} rep {r.get('rep')}" + (" (SOFTWARE control)" if _is_software(r)
                                                         else " (jammed control)" if _is_hog(r)
                                                         else "")
        rows.append("| " + " | ".join([
            label, _fmt(amd.get("total_gfx_ns_delta")), vram,
            str(len(amd.get("render_nodes_open") or [])),
            str(_workaround_applied(r) or "none (PreserveEnvironment)")]) + " |")

    sw = _find(obs, _is_software)
    real = _ok_runs(obs)
    if sw is not None and real:
        rows += ["",
                 "The software leg is the NEGATIVE CONTROL: the same binary, the same X server "
                 "and the same fixture home, with `LIBGL_ALWAYS_SOFTWARE=1` the only difference. "
                 "It is what makes the GFX nanoseconds in the other rows evidence rather than a "
                 "number, because WebKitGTK hardcodes `Apple GPU` into "
                 "`WEBGL_debug_renderer_info` on Linux on AMD and there is no in-page device "
                 "string to read."]

    ri = [(_bench(r).get("app") or {}) for r in _runs(obs)]
    nv = {bool(a.get("nvidia_module_present")) for a in ri if a}
    rows += ["", "### The decision linux_webkit.rs made, and why", "",
             f"- `/proc/driver/nvidia/version` present: **{nv or 'unknown'}**. This is the "
             f"single most load-bearing input: the probe is module presence and not the GPU "
             f"that will render, as the comment at linux_webkit.rs:170 concedes.",
             "- Wayland socket: absent. This runner drives Xvfb, so the Wayland arm, which is "
             "the one that applies `WEBKIT_DMABUF_RENDERER_FORCE_SHM`, is not reachable here.",
             "- `APPIMAGE`: unset. The binary is built with `cargo build --release` and not "
             "bundled, deliberately, because an AppImage sets that variable and takes a "
             "different branch (linux_webkit.rs:139, :197).",
             "",
             "The applied column is read from `/proc/<pid>/environ` of the LIVE process, not "
             "from the log: main.rs:1841 logs the decision only when a workaround IS applied, "
             "so silence there is ambiguous between `PreserveEnvironment` and a crash before "
             "the line."]

    jam, unjam, cphase = _control_fps(obs)
    if jam is not None:
        rows += ["", f"Positive control, the same rung with the main thread deliberately "
                 f"jammed: **{jam:.1f} fps** against {unjam:.1f} fps unjammed ({cphase} phase). "
                 f"This is what makes a reading here a finding rather than a broken instrument."]

    bad = [r for r in _runs(obs) if r not in _ok_runs(obs) and not _is_control(r)]
    if bad:
        rows += ["", "Runs that did not complete:", ""]
        for r in bad:
            b = _bench(r)
            rows.append(f"- {r.get('rung')} rep {r.get('rep')}: rc={r.get('rc')} "
                        f"{str(b.get('fatal') or b.get('error') or r.get('error'))[:220]}")
    return "\n".join(rows)


def _worst_drop(obs):
    by = _by_rung(obs)
    order = _rungs_sorted(obs)
    if len(order) < 2:
        return None, None, None, None, None
    base = order[0]
    worst, worst_pct = (None, None, None, None, None), 0.0
    for phase in ("idle", "scroll", "stream"):
        b = _agg(by[base], phase)
        if not b["fps"]:
            continue
        for rung in order[1:]:
            t = _agg(by[rung], phase)
            if not t["fps"]:
                continue
            drop_pct = 100.0 * (b["fps_max"] - t["fps_min"]) / b["fps_max"]
            floor = max(x for x in (b["spread"] or 0.0, t["spread"] or 0.0, 0.0))
            if drop_pct > worst_pct:
                worst_pct, worst = drop_pct, (phase, rung, b["fps_max"], t["fps_min"], floor)
    return worst


def _gpu_story(obs) -> str:
    sw = _find(obs, _is_software)
    real = [r for r in _ok_runs(obs)]
    real_ns = max([_gfx_ns(r) or 0 for r in real] or [0])
    sw_ns = _gfx_ns(sw) if sw is not None else None
    applied = {}
    for r in real:
        applied.update(_workaround_applied(r))
    where = ("with NO workaround applied, i.e. RenderingPlan::PreserveEnvironment"
             if not applied else f"with {', '.join(sorted(applied))} applied")
    if sw_ns is None:
        return (f"the app's own web process accrued {real_ns:,} ns of amdgpu GFX engine time "
                f"during the run, {where}, but the software negative control did not complete, "
                f"so that figure is a number and not yet evidence")
    if real_ns and sw_ns <= real_ns * SOFTWARE_MAX_FRACTION:
        return (f"the app composited ON THE GPU: {real_ns:,} ns of amdgpu GFX engine time "
                f"attributed to its own web process during the run, against {sw_ns:,} ns in an "
                f"otherwise identical leg with LIBGL_ALWAYS_SOFTWARE=1, {where}")
    return (f"GPU compositing is NOT established: the real leg booked {real_ns:,} ns of GFX "
            f"engine time and the software control booked {sw_ns:,} ns, which is not the "
            f"collapse a working negative control produces, so something other than the "
            f"browser's rendering may account for both")


def verdict(obs):
    by = _by_rung(obs)
    order = _rungs_sorted(obs)
    base, top = order[0], order[-1]
    phase, rung, base_fps, top_fps, floor = _worst_drop(obs)

    top_busy = max((_agg(by[top], p)["busy"] or 0.0) for p in ("idle", "scroll", "stream"))
    base_busy = max((_agg(by[base], p)["busy"] or 0.0) for p in ("idle", "scroll", "stream"))
    top_el = _agg(by[top], "idle")["elements"] or 0
    base_el = _agg(by[base], "idle")["elements"] or 0
    gpu = _gpu_story(obs)

    if phase is None:
        return "INCONCLUSIVE", ("no phase produced a frame reading at two different rungs, so "
                                "nothing can be compared")

    drop_pct = 100.0 * (base_fps - top_fps) / base_fps
    real = drop_pct > MATERIAL_DROP_PCT and (base_fps - top_fps) > floor

    if real:
        reached = (f"at or below the {REPORTED_FLOOR_FPS:.0f} fps the user's report names"
                   if top_fps <= REPORTED_FLOOR_FPS else
                   f"well above the {REPORTED_FLOOR_FPS:.0f} fps the report names, so this is a "
                   f"smaller effect than the one described")
        return "COLLAPSE_REPRODUCED", (
            f"the effective frame rate fell from {base_fps:.1f} fps at {base} to {top_fps:.1f} "
            f"fps at {rung} during the {phase} phase, a {drop_pct:.0f}% drop against a same-rung "
            f"repeat spread of {floor:.1f} fps, {reached}. The main thread was {top_busy:.0f}% "
            f"busy at {top} against {base_busy:.0f}% at {base}, over {top_el:,.0f} DOM elements "
            f"against {base_el:,.0f}. On the rendering path: {gpu}")

    if top_busy < MIN_TOP_BUSY_PCT or (base_el and top_el / base_el < MIN_DOM_GROWTH):
        return "VOID", (
            f"the frame rate is flat ({base_fps:.1f} fps at {base}, {top_fps:.1f} fps at {top}) "
            f"but the venue was not loaded: the main thread reached only {top_busy:.0f}% busy at "
            f"{top} over {top_el:,.0f} DOM elements. A flat reading on an unloaded page is "
            f"evidence of no load, not evidence of no effect")

    return "NO_COLLAPSE", (
        f"Desktop does NOT collapse: {base_fps:.1f} fps at {base} against {top_fps:.1f} fps at "
        f"{top} in the {phase} phase, a {drop_pct:.0f}% difference against a same-rung repeat "
        f"spread of {floor:.1f} fps, with the venue loaded ({base_busy:.0f}% busy at {base} to "
        f"{top_busy:.0f}% at {top}, {top_el:,.0f} DOM elements against {base_el:,.0f}). On the "
        f"rendering path: {gpu}")


def observed_capabilities(obs):
    ok = _ok_runs(obs)
    sw = _find(obs, _is_software)
    real_ns = max([_gfx_ns(r) or 0 for r in ok] or [0])
    sw_ns = _gfx_ns(sw) if sw is not None else None
    gpu_ok = bool(real_ns and sw_ns is not None and sw_ns <= real_ns * SOFTWARE_MAX_FRACTION)
    return {
        "webkitgtk": bool(ok),
        "headless_display_server": bool((obs.get("xserver") or {}).get("display")),
        "studio_production_bundle": bool((obs.get("install_layout") or {}).get("cli_exists")),
        "studiobench_ladder": len(_by_rung(obs)) >= 2,
        "desktop_tauri_app": bool(ok),
        "desktop_gpu_compositing": gpu_ok,
        "gpu_browser_compositing": gpu_ok,
    }

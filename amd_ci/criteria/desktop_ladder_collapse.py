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

   The 0K rung is EMPTY BY CONSTRUCTION and is still a rung: it is the ~61 fps reference every
   other rung's collapse is measured against, so an empty thread must be able to report a real
   result rather than be dropped. It has no last-turn marker and no `[data-role]` node, so what
   stands in for them there is the router's own query string plus a mounted shell and a live
   composer with an empty message list; see amdv_desktop_boot.js::threadRendered.

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
MIN_TOP_BUSY_PCT = 15.0

# ── DID THE SEEDED THREAD MOUNT: a MEASURED floor, not a growth ratio ──────────────────────
#
# What this replaces, and why. The gate used to be "`final.elements` at the top rung is >= 5x
# `final.elements` at the bottom rung". In run 32808701910 that failed a healthy seed: 62,666
# elements at 100K against 274,973 at 500K is 4.4x, and the gate wanted 5x. The threshold
# assumed DOM elements scale LINEARLY with tokens. They do not, for two compounding reasons:
# longer messages amortise into fewer elements per token, and `final` is taken AFTER this
# harness has streamed its own ~6,000-character tail into the thread, which adds a similar
# ABSOLUTE number of nodes at every rung and so dilutes a large ratio more than a small one.
# Lowering 5.0 until 4.4 passes would have made the gate decorative; the quantity was wrong,
# not the number.
#
# What is asserted instead is three things that actually discriminate a seeded thread from an
# unseeded one, all read from the MOUNT census -- taken the instant the thread is mounted and
# before this harness touches it, which is precisely the quantity the gate's own name claims:
#
#   1. the mounted message count EQUALS the seeder's own count for that thread, exactly. Not a
#      threshold at all: the seeder knows it wrote 18 messages, and 18 `[data-role]` nodes is
#      the thread having mounted. This is what catches the failure the old ratio was reaching
#      for, namely the PREVIOUS thread's DOM still being on screen;
#   2. the mounted element count clears a per-rung FLOOR calibrated below;
#   3. element counts increase STRICTLY with rung, so the ladder is a ladder.
#
# CALIBRATION SOURCE, so this is measured rather than guessed. `mount.census.elements` over
# every completed run against studiobench corpus 23cd2464 on this runner:
#
#   rung   Studio web UI, WebKitGTK 2.52.3                        Unsloth Desktop, Tauri
#   0K     546 x 7    (runs 32799285502, 32803626271, 32805404479)  none yet (all 0K legs failed)
#   100K   19,958; 19,959 x3; 20,415                                19,955 x2   (run 32808701910)
#   500K   110,328 x8; 110,350 x2 (incl. 32807566159 x4)            110,322 x2  (run 32808701910)
#
# The two engines agree to within 0.03% at 100K and 500K, and the spread within an engine is
# under 2.3%, so the quantity is stable enough to floor. Each floor is ~60% of the LOWEST value
# ever observed at that rung: generous against engine and version drift, and still one to two
# orders of magnitude above the failure being guarded against, which is a thread that never
# mounted and leaves the bare app shell of ~546 elements (0K's whole DOM) or less.
#
# A rung with no entry here is not silently waved through: the gate says so and fails, because
# an uncalibrated rung is an unjudged one.
MIN_MOUNT_ELEMENTS = {"0K": 300, "100K": 12000, "500K": 66000}
CONTROL_MIN_DROP_PCT = 25.0
# How much of the commanded scroll distance has to be actually travelled before the scroll
# phase counts as a traversal rather than a jiggle. The unrepaired gesture measured 12% in
# Chromium (6,610 px of 54,000) and read a comfortable frame rate for it.
MIN_SCROLL_TRAVEL_FRACTION = 0.80
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


def _is_pristine(r):
    return bool(r.get("no_scene")) or str(r.get("rep")) == "pristine"


def _is_ablation(r):
    """The fence-deferral OFF arm. Same rung as a ladder run and NOT a ladder run.

    Kept out of `_by_rung` deliberately. Pooling it with the default arm at the same rung would
    average an ablated reading into the ladder and, worse, inflate that rung's repeat spread --
    which is the floor every rung-to-rung difference is judged against. An arm that changes the
    app's behaviour is a second experiment, not a second repetition.
    """
    return (r.get("defer_fence") in ("off", "on")) or str(r.get("rep", "")).startswith("abl")


def _is_control(r):
    return _is_hog(r) or _is_software(r) or _is_pristine(r) or _is_ablation(r)


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


def _mount_census(scene):
    """The DOM the instant the seeded thread mounted, before this harness added anything."""
    return (scene.get("mount") or {}).get("census") or {}


def _mount_elements(scene):
    return _mount_census(scene).get("elements")


def _seeded(r):
    """What the SEEDER says it wrote, straight from the backend it wrote to."""
    return _bench(r).get("seeded") or {}


def _seed_rows(obs):
    """Per completed ladder run: what was seeded, what mounted, and whether they agree."""
    rows = []
    for r in _ok_runs(obs):
        sc, rung = _scene(r), r.get("rung", "?")
        cen = _mount_census(sc)
        want = _seeded(r).get("messages")
        floor = MIN_MOUNT_ELEMENTS.get(rung)
        el, got = cen.get("elements"), cen.get("messages")
        # At 0K there is no message to find and no marker to match, so the only positive
        # evidence that the app is on THIS thread rather than on the runtime's own bootstrap
        # thread is the router's own query string, recorded by the page at scene start.
        on_thread = (_scene(r).get("__desktop") or {}).get("on_requested_thread")
        empty_ok = True if want else bool(on_thread)
        rows.append({
            "rung": rung, "rep": r.get("rep"), "seeded_messages": want,
            "mounted_messages": got, "mount_elements": el, "floor": floor,
            "on_requested_thread": on_thread,
            "ok": bool(floor is not None and el is not None and el >= floor
                       and want is not None and got == want and empty_ok),
        })
    return rows


def _seed_ok(obs):
    """(passed, one-line evidence). Shared by the gate and by the VOID branch of the verdict."""
    rows = _seed_rows(obs)
    if not rows:
        return False, "no ladder run completed, so nothing mounted to check"
    order = _rungs_sorted(obs)
    els = [(rung, _agg(_by_rung(obs)[rung], "idle")["mount_elements"]) for rung in order]
    mono = all(a is not None and b is not None and b > a
               for (_, a), (_, b) in zip(els, els[1:]))
    detail = "; ".join(
        f"{rung}: {(el or 0):,.0f} elements at mount"
        + (f" (floor {MIN_MOUNT_ELEMENTS[rung]:,})" if rung in MIN_MOUNT_ELEMENTS
           else " (NO CALIBRATED FLOOR for this rung)")
        for rung, el in els)
    bad = [f"{x['rung']}#{x['rep']}: "
           + ("no floor calibrated" if x["floor"] is None else
              f"{x['mounted_messages']} of {x['seeded_messages']} seeded messages mounted"
              if x["mounted_messages"] != x["seeded_messages"] else
              "the page was not on the requested thread" if not x["on_requested_thread"] else
              f"{x['mount_elements']} elements is under the {x['floor']:,} floor")
           for x in rows if not x["ok"]]
    passed = bool(rows) and not bad and mono and len(order) >= 2
    if bad:
        detail += " -- FAILED: " + "; ".join(bad)
    elif not mono:
        detail += " -- FAILED: element counts are not strictly increasing with rung"
    elif len(order) < 2:
        detail += " -- FAILED: only one rung produced a reading"
    else:
        msgs = "; ".join(f"{x['rung']}#{x['rep']} {x['mounted_messages']}/"
                         f"{x['seeded_messages']} msgs" for x in rows)
        detail += f" | mounted exactly what was seeded: {msgs}"
    return passed, detail


def _scroll_trace(scene):
    return scene.get("scroll_trace") or {}


def _fence_arm(r):
    """The arm the PAGE reports it ran under, not the one the harness asked for."""
    return ((_scene(r).get("__desktop") or {}).get("fence_arm")
            or _bench(r).get("defer_fence_arm") or "default")


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
    mels = [v for v in (_mount_elements(_scene(r)) for r in rs) if v is not None]
    return {"fps": vals, "n": len(vals),
            "mount_elements": (sum(mels) / len(mels)) if mels else None,
            "fps_min": min(vals) if vals else None, "fps_max": max(vals) if vals else None,
            "spread": (max(vals) - min(vals)) if len(vals) >= 2 else None,
            "worst_ms": max(worst) if worst else None,
            "busy": (sum(busy) / len(busy)) if busy else None,
            "blocked_ms": (sum(blocked) / len(blocked)) if blocked else None,
            "elements": (sum(els) / len(els)) if els else None}


def _control_phases(obs, c):
    """Every phase this jam leg can be read on, against the SAME rung unjammed."""
    out = []
    if not c or not _bench(c).get("ok"):
        return out
    peers = [r for r in _ok_runs(obs) if r.get("rung") == c.get("rung")]
    for phase in ("idle", "scroll", "stream"):
        j = _raf_fps(_scene(c), phase)
        if j is None:
            continue
        u = [x for x in (_raf_fps(_scene(r), phase) for r in peers) if x is not None]
        if not u:
            continue
        unjam = sum(u) / len(u)
        out.append({"phase": phase, "jam": j, "unjam": unjam,
                    "drop_pct": (100.0 * (unjam - j) / unjam) if unjam else None,
                    "busy_unjammed": next((v for v in (_busy(_scene(r), phase) for r in peers)
                                           if v is not None), None)})
    return out


def _control_at(obs, c):
    """One jam leg against the SAME rung unjammed: (jam fps, unjam fps, phase).

    THE PHASE IS CHOSEN BY LARGEST RELATIVE DROP, not by lowest jammed frame rate, and that
    distinction decided a whole run. Picking the lowest jammed reading always lands on the
    phase where the page is ALREADY saturated, which is precisely where a jam has no headroom
    left to take. Measured on this runner in run 32819187840, at 500K: the scroll phase read
    1.8 fps jammed against 2.5 fps unjammed, a 28% drop that only just missed the 25% bar with
    the main thread already 97% busy unjammed -- while the IDLE phase of the same two legs read
    17.8 fps against 61.5 fps, a 71% drop, the same figure the control produces at 100K (17.2
    against 61.0). The channel resolves a jammed main thread perfectly well; the old selection
    rule was reading it in the one window where nothing could show.
    """
    rows = _control_phases(obs, c)
    if not rows:
        return None, None, "scroll"
    best = max(rows, key = lambda x: x["drop_pct"] if x["drop_pct"] is not None else -1e9)
    return best["jam"], best["unjam"], best["phase"]


def _control_rows(obs):
    """EVERY jam leg, one row per rung.

    Per rung and not "the" control, because a control scheduled at a single rung is a single
    point of failure for the whole run: in 32808701910 the only jam leg sat at 0K, 0K could not
    complete, and the gate failed with "control did not complete" -- so six perfectly good
    ladder readings had nothing to certify them. It is also the only comparison that means
    anything: a jam has to be priced against the same rung unjammed, or the difference
    confounds the jam with the thread size.
    """
    rows = []
    for c in _runs(obs):
        if not _is_hog(c):
            continue
        jam, unjam, phase = _control_at(obs, c)
        done = bool(_bench(c).get("ok"))
        why = (str(_bench(c).get("fatal") or _bench(c).get("error") or c.get("rc"))[:140]
               if not done else
               "the jam leg completed but no unjammed run at this rung did, so there is "
               "nothing to price it against")
        rows.append({
            "rung": c.get("rung", "?"), "jam": jam, "unjam": unjam, "phase": phase,
            "hog_ms": c.get("hog"), "period_ms": _bench(c).get("hog_period_ms", 250),
            "completed": done, "why": why, "phases": _control_phases(obs, c),
            "resolves": bool(jam is not None and unjam is not None
                             and jam < unjam * (1.0 - CONTROL_MIN_DROP_PCT / 100.0)),
        })
    rows.sort(key = lambda x: ORDER.index(x["rung"]) if x["rung"] in ORDER else 99)
    return rows


def _control_fps(obs):
    """The single worst jam/unjam pair, for the headline sentence in the table."""
    best = (None, None, "scroll")
    for row in _control_rows(obs):
        if row["jam"] is None:
            continue
        if best[0] is None or row["jam"] < best[0]:
            best = (row["jam"], row["unjam"], row["phase"])
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

    seeded_ok, seed_detail = _seed_ok(obs)
    out.append(("the seeded thread really mounted (mounted message count equals the seeder's, "
                "over a calibrated per-rung element floor, strictly increasing with rung)",
                seeded_ok, seed_detail))

    # THE SCROLL GESTURE ACTUALLY TRAVELLED. This is a gate and not a footnote, because the
    # failure it guards against reads as a comfortable frame rate rather than as an error: the
    # unrepaired gesture assigns `scrollTop` into a `scroll-smooth` viewport, so it commands
    # tens of thousands of pixels and travels a few thousand, and `scroll_detail` shows the
    # bottom of the thread either way. A scroll phase that did not move is not a scroll phase.
    travels = []
    for r in ok:
        st = _scroll_trace(_scene(r))
        c, t = st.get("commanded_px"), st.get("travelled_px")
        if c:
            travels.append((r.get("rung"), c, t or 0, (t or 0) / c))
    moved = bool(travels) and all(frac >= MIN_SCROLL_TRAVEL_FRACTION for *_, frac in travels)
    out.append((f"the scroll gesture travelled (>= {MIN_SCROLL_TRAVEL_FRACTION:.0%} of the "
                f"pixels it commanded)", moved,
                "; ".join(f"{rung}: {t:,.0f}/{c:,.0f} px ({frac:.0%})"
                          for rung, c, t, frac in travels) or "no scroll trace recorded"))

    # THE POSITIVE CONTROL. Without it a flat frame rate cannot be told from a channel that
    # cannot read anything else, which is precisely how the web UI's first table came to be an
    # artefact.
    # Run at EVERY rung, so the gate no longer rides on one leg: it passes when at least one
    # rung resolves and NO rung whose jam leg produced a reading fails to. A jam leg that did
    # not complete is named rather than ignored, but it does not by itself cost the run its
    # verdict, which is the whole point of not pinning the control to a single rung.
    crows = _control_rows(obs)
    read = [x for x in crows if x["jam"] is not None]
    resolves = bool(read) and all(x["resolves"] for x in read)
    out.append((f"the headline frame channel resolves a jammed main thread at every rung it was "
                f"run at (>= {CONTROL_MIN_DROP_PCT:.0f}% below the same rung unjammed)", resolves,
                "no control run" if not crows else "; ".join(
                    (f"{x['rung']}: did not complete ({x['why']})" if x["jam"] is None else
                     f"{x['rung']} with a {x['hog_ms']} ms jam every {x['period_ms']} ms: "
                     f"{x['jam']:.1f} vs {x['unjam']:.1f} fps unjammed ({x['phase']})"
                     f"{'' if x['resolves'] else ' -- DOES NOT RESOLVE'}")
                    for x in crows)))

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

    # THE UNMODIFIED APP. Everything above is measured on a binary with our scene compiled
    # into the bundle, which is unavoidable and is also exactly the kind of thing that turns
    # "Desktop works" into "our build of Desktop works". This leg is the shipped code path:
    # no injected script, no control channel, observed only.
    pr = _find(obs, _is_pristine)
    prb = _bench(pr) if pr else {}
    pinfo = prb.get("pristine") or {}
    out.append(("the UNMODIFIED binary reached the Studio shell over a real backend",
                bool(prb.get("ok")),
                "no pristine leg" if pr is None else
                f"window_mapped={pinfo.get('window_mapped')} "
                f"backend_requests={pinfo.get('backend_requests_seen')} "
                f"seeded_thread_id_in_backend_log={pinfo.get('thread_id_in_log')} "
                f"screenshot={pinfo.get('screenshot_bytes')} B"))

    have_repeat = any(len(rs) >= 2 for rs in _by_rung(obs).values())
    out.append(("at least one rung was measured twice, so a difference has a floor", have_repeat,
                ", ".join(f"{k}x{len(v)}" for k, v in sorted(_by_rung(obs).items())) or "none"))
    return out


def _fmt(v, suffix="", nd=0):
    return "-" if v is None else f"{v:,.{nd}f}{suffix}"


def table(obs):
    rows = ["| rung | reps | DOM at mount | DOM elements | phase | fps (effective rAF) | "
            "worst frame | busy | blocked |", "|---|---|---|---|---|---|---|---|---|"]
    by = _by_rung(obs)
    for rung in _rungs_sorted(obs):
        rs = by[rung]
        el = _agg(rs, "idle")["elements"]
        mel = _agg(rs, "idle")["mount_elements"]
        for phase in ("idle", "scroll", "stream"):
            a = _agg(rs, phase)
            if not a["fps"]:
                fps = "-"
            elif a["n"] == 1:
                fps = f"{a['fps_min']:.1f}"
            else:
                fps = f"{a['fps_min']:.1f}-{a['fps_max']:.1f}"
            rows.append("| " + " | ".join([
                rung, str(len(rs)), _fmt(mel), _fmt(el), phase, f"**{fps}**",
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

    # ── THE ABLATION ──
    #
    # Printed as its own table and never folded into the ladder. The comparison is the SAME
    # rung in two arms, so the ladder's rung-to-rung floor does not apply to it and its own
    # arm-to-arm repeat spread does.
    abl = [r for r in _runs(obs) if _is_ablation(r) and _bench(r).get("ok")]
    default_at = [r for r in _ok_runs(obs) if r.get("rung") == (abl[0].get("rung") if abl else None)]
    if abl and default_at:
        rung = abl[0].get("rung")
        rows += ["", f"### Fence-deferral ablation at {rung}", "",
                 "| arm | n | scroll fps | stream fps | idle fps | busy (scroll) | elements | "
                 "highlight spans |", "|---|---|---|---|---|---|---|---|"]
        for label, rs in (("deferral ON (SHIP_DEFAULT)", default_at),
                          ("deferral OFF (ablated)", abl)):
            sc, stm, idl = _agg(rs, "scroll"), _agg(rs, "stream"), _agg(rs, "idle")
            spans = [((_scene(r).get("final") or {}).get("highlightSpans")) for r in rs]
            spans = [x for x in spans if x is not None]
            def rng(a):
                if not a["fps"]:
                    return "-"
                return (f"{a['fps_min']:.1f}" if a["n"] == 1
                        else f"{a['fps_min']:.1f}-{a['fps_max']:.1f}")
            rows.append("| " + " | ".join([
                label, str(len(rs)), f"**{rng(sc)}**", f"**{rng(stm)}**", rng(idl),
                "null" if sc["busy"] is None else f"{sc['busy']:.0f}%",
                _fmt(sc["elements"]),
                _fmt(sum(spans) / len(spans) if spans else None)]) + " |")
        arms = sorted({_fence_arm(r) for r in abl})
        rows += ["",
                 f"The ablated arm reports `fence_arm={arms}`, read back from the page's own "
                 f"global rather than from the flag the harness asked for: an arm that failed to "
                 f"apply is the single failure that would make this table read as *the flag does "
                 f"nothing*. `resolveFenceMode` maps the BOOLEAN `false` to `off` and an ABSENT "
                 f"value to `SHIP_DEFAULT`, which is `defer`, so the two arms here are "
                 f"`false` and unset, never a string.",
                 "",
                 "**What this is testing.** The mechanism attributed in both engines is the "
                 "one-way upgrade of deferred code fences: roughly 335 fences, two commits per "
                 "latch, each forcing a style recalc over the whole document. It is EXCESS and "
                 "not BURST -- both arms end with an identical DOM, so the deferred delivery buys "
                 "nothing and costs the extra traversals. If Desktop moves the way Chromium and "
                 "WebKitGTK already do, the mechanism is shell-independent and Desktop is the "
                 "same defect in a different wrapper."]

    # Scroll travel, so the gesture is auditable rather than assumed.
    rows += ["", "### Did the scroll phase actually scroll", "",
             "| leg | commanded px | travelled px | steps | reached top | deadline hit |",
             "|---|---|---|---|---|---|"]
    for r in _runs(obs):
        st = _scroll_trace(_scene(r))
        if not st:
            continue
        rows.append("| " + " | ".join([
            f"{r.get('rung')} rep {r.get('rep')}", _fmt(st.get("commanded_px")),
            _fmt(st.get("travelled_px")), str(st.get("steps")),
            str(st.get("reached_top")), str(st.get("deadline_hit"))]) + " |")
    rows += ["",
             "Recorded because the unrepaired gesture in `amdv_scene.js` does not scroll: it "
             "assigns `scrollTop` into a viewport carrying `scroll-smooth`, so each write "
             "animates and the next is computed from a stalled position. Measured in Chromium at "
             "r500K it commanded 54,000 px and travelled 6,610, never further than 1,107 px from "
             "the bottom of a 316,829 px thread, and read a comfortable frame rate for it. "
             "`scroll_detail` cannot show that, because it records the first and last position "
             "only and both are the bottom. This ladder uses the repaired gesture "
             "(`scrollTo` with `behavior: instant`, a real `WheelEvent`, one step per painted "
             "frame), so its idle and stream phases stay directly comparable to the web UI "
             "ladder and its scroll phase is comparable only with the gesture named."]

    # The rendering path, which is the other half of the question.
    rows += ["", "### What actually rendered", "",
             "| leg | GFX engine ns accrued during the run | VRAM | render nodes | "
             "workaround applied by linux_webkit.rs |", "|---|---|---|---|---|"]
    for r in _runs(obs):
        b = _bench(r)
        amd = b.get("amdgpu") or {}
        vram = ", ".join(sorted({c.get("vram") or "" for c in amd.get("clients") or []})) or "-"
        label = f"{r.get('rung')} rep {r.get('rep')}" + (
            " (SOFTWARE control)" if _is_software(r) else
            " (jammed control)" if _is_hog(r) else
            " (UNMODIFIED binary)" if _is_pristine(r) else "")
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

    crows = _control_rows(obs)
    if crows:
        rows += ["", "### The jammed positive control, at every rung", "",
                 "| rung | jam | jammed fps | same rung unjammed | phase | resolves |",
                 "|---|---|---|---|---|---|"]
        for x in crows:
            rows.append("| " + " | ".join([
                x["rung"], f"{x['hog_ms']} ms every {x['period_ms']} ms",
                "did not complete" if x["jam"] is None else f"**{x['jam']:.1f}**",
                "-" if x["unjam"] is None else f"{x['unjam']:.1f}",
                x["phase"], "yes" if x["resolves"] else
                ("-" if x["jam"] is None else "NO")]) + " |")
        rows += ["", "Every phase the control can be read on, so the phase choice above is "
                 "auditable rather than asserted:", "",
                 "| rung | phase | jammed fps | unjammed fps | drop | unjammed busy |",
                 "|---|---|---|---|---|---|"]
        for x in crows:
            for ph in x["phases"]:
                rows.append("| " + " | ".join([
                    x["rung"], ph["phase"], f"{ph['jam']:.1f}", f"{ph['unjam']:.1f}",
                    "-" if ph["drop_pct"] is None else f"{ph['drop_pct']:.0f}%",
                    "-" if ph["busy_unjammed"] is None else f"{ph['busy_unjammed']:.0f}%"])
                    + " |")
        rows += ["",
                 "The headline phase is the one with the LARGEST RELATIVE DROP, not the one "
                 "with the lowest jammed frame rate. The lowest jammed reading always lands on "
                 "the phase where the page is already saturated, which is exactly where a jam "
                 "has no headroom left to take: at 500K the scroll phase is 97% busy before "
                 "the jam is installed, so the jam can only move it from 2.5 fps to 1.8 fps, "
                 "while the idle phase of the same two legs moves 61.5 -> 17.8 fps. Reading "
                 "the control in the saturated window would report that the channel cannot see "
                 "a blocked main thread, which is false.",
                 "",
                 "One per rung, and that is a repair rather than thoroughness. In run "
                 "32808701910 the only jam leg was scheduled at 0K, 0K was the rung that could "
                 "not complete, and the gate failed with *control did not complete* -- so six "
                 "ladder readings that were fine had nothing certifying that the channel they "
                 "were read on can move at all. A control placed at one rung is a single point "
                 "of failure for the whole run. Per rung is also the only comparison that "
                 "means anything: a jam has to be priced against the SAME rung unjammed, or "
                 "the difference confounds the jam with the thread size."]

    swpairs = _software_pairs(obs)
    if swpairs:
        rows += ["", "### The software negative control, at every rung", "",
                 "| rung | real leg GFX ns | LIBGL_ALWAYS_SOFTWARE=1 GFX ns | software share |",
                 "|---|---|---|---|"]
        for rung, rns, sns in swpairs:
            share = "-" if not rns or sns is None else f"{100.0 * sns / rns:.1f}%"
            rows.append("| " + " | ".join([rung, _fmt(rns), _fmt(sns), share]) + " |")
        rows += ["",
                 f"Same rung on both sides, because GFX engine time scales with what is on "
                 f"screen: a 500K real leg priced against a 0K software leg, which is what a "
                 f"single-rung control forced, differs in the rung as well as in the "
                 f"rasteriser. The share has to fall below "
                 f"{SOFTWARE_MAX_FRACTION:.0%} before the real leg's counters are accepted as "
                 f"the browser's own rendering."]

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


def _software_pairs(obs):
    """Per rung: (rung, the real leg's GFX ns, the SOFTWARE leg's GFX ns at the SAME rung).

    Same-rung, because GFX engine time scales with what is on screen: pricing a 500K real leg
    against a 0K software leg, which is what a single-rung control forced, compares two things
    that differ in the rung as well as in the rasteriser.
    """
    out = []
    by = _by_rung(obs)
    for r in _runs(obs):
        if not _is_software(r):
            continue
        rung = r.get("rung", "?")
        peers = by.get(rung) or []
        real = max([_gfx_ns(p) or 0 for p in peers] or [0])
        out.append((rung, int(real), _gfx_ns(r)))
    out.sort(key = lambda x: ORDER.index(x[0]) if x[0] in ORDER else 99)
    return out


def _gpu_story(obs) -> str:
    real = [r for r in _ok_runs(obs)]
    real_ns = int(max([_gfx_ns(r) or 0 for r in real] or [0]))
    # Prefer a SAME-RUNG pair; fall back to the old cross-rung comparison only when no software
    # leg ran beside a real one.
    # The heaviest rung that has both legs: the most demanding place to ask the question.
    sw_ns = None
    pairs = [(rns, sns) for _, rns, sns in _software_pairs(obs) if rns and sns is not None]
    if pairs:
        real_ns, sns = max(pairs, key = lambda p: p[0])
        sw_ns = int(sns)
    if sw_ns is None:
        sw = _find(obs, _is_software)
        sw_ns = _gfx_ns(sw)
        sw_ns = int(sw_ns) if sw_ns is not None else None
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

    seeded_ok, seed_detail = _seed_ok(obs)
    if top_busy < MIN_TOP_BUSY_PCT or not seeded_ok:
        return "VOID", (
            f"the frame rate is flat ({base_fps:.1f} fps at {base}, {top_fps:.1f} fps at {top}) "
            f"but the venue was not loaded: the main thread reached only {top_busy:.0f}% busy at "
            f"{top} over {top_el:,.0f} DOM elements, and the seed check says: {seed_detail}. A "
            f"flat reading on an unloaded page is evidence of no load, not evidence of no "
            f"effect")

    return "NO_COLLAPSE", (
        f"Desktop does NOT collapse: {base_fps:.1f} fps at {base} against {top_fps:.1f} fps at "
        f"{top} in the {phase} phase, a {drop_pct:.0f}% difference against a same-rung repeat "
        f"spread of {floor:.1f} fps, with the venue loaded ({base_busy:.0f}% busy at {base} to "
        f"{top_busy:.0f}% at {top}, {top_el:,.0f} DOM elements against {base_el:,.0f}). On the "
        f"rendering path: {gpu}")


def observed_capabilities(obs):
    ok = _ok_runs(obs)
    # Same-rung pairs, for the reason given at _software_pairs.
    pairs = [(rns, sns) for _, rns, sns in _software_pairs(obs) if rns and sns is not None]
    if pairs:
        real_ns, sns = max(pairs, key = lambda p: p[0])
        sw_ns = int(sns)
    else:
        real_ns = int(max([_gfx_ns(r) or 0 for r in ok] or [0]))
        sw_ns = _gfx_ns(_find(obs, _is_software))
        sw_ns = int(sw_ns) if sw_ns is not None else None
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

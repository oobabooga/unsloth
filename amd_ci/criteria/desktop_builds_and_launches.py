#!/usr/bin/env python3
"""Criteria: does Unsloth Desktop build from source here, and does the built binary LAUNCH?

Judges only. Observations come from probes/desktop_build_probe.py.

Three claims, in increasing order of how easy they are to fake, and each is required for the
next to mean anything:

  1. **It builds.** Evidenced by an executable on disk and by cargo's exit code, and before
     either of those by a C program COMPILED AND LINKED against webkit2gtk-4.1 out of the
     rootlessly fetched -dev tree. A `pkg-config --modversion` string is not accepted: it
     succeeds on a .pc file with no headers behind it, which is exactly the state the first
     rootless fetch left behind.

  2. **It launches and stays up.** A process that exits is not an app, and a process that
     survives because it is blocked on a dialog is not one either, so the window tree has to
     name a mapped toplevel.

  3. **It PAINTED.** This is the one worth being strict about. "The process started" is the
     failure this whole report is written against. The evidence is a screenshot of the X
     server, weighed AGAINST a screenshot of the same X server taken before the app was
     launched. A hard byte threshold would be a guess about this framebuffer; the ratio to the
     empty frame is not.

What this deliberately does NOT claim:

  * It does not claim the app reached the Studio chat shell. No Studio backend is installed in
    this job, so studio/frontend/src/app/provider.tsx puts the webview on the StartupScreen
    (status `not-installed`), and that IS the expected content here. Reaching the chat shell
    with a thread mounted is the measuring job's claim, not this one's.
  * It does not claim anything about GPU compositing. This leg runs with
    LIBGL_ALWAYS_SOFTWARE=1 on purpose, as the negative control for the measuring job.
  * It says nothing about Windows/WebView2 or macOS/WKWebView, which are different engines
    entirely and unreachable from here.
"""

from __future__ import annotations

TITLE = "Does Unsloth Desktop (Tauri) build and launch on the AMD CI runner?"
MODE = "capability"

NEEDS = [
    "linux", "rust_toolchain", "tauri_build_deps", "webkitgtk", "headless_display_server",
    "desktop_tauri_app", "studio_production_bundle",
    # Declared because the QUESTION touches them, not because this host has them. An
    # under-declared NEEDS renders a report that bounds nothing.
    "desktop_gpu_compositing", "gpu_browser_compositing", "drm_render_node",
    "windows", "windows_webview2", "macos", "macos_wkwebview", "wayland_session",
]

# How much bigger than the EMPTY framebuffer a painted one has to compress to before "the
# frontend painted" is claimed. A blank X root screenshot is near-uniform and compresses to
# almost nothing; a rendered GTK window with text does not. Two is a deliberately modest
# multiple: the claim is "something was drawn", not "the right thing was drawn", and the
# screenshot is in the artifact for a human to check the second part.
MIN_PAINT_RATIO = 2.0


def _b(obs, *path, default=None):
    cur = obs
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
    return default if cur is None else cur


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = [("probe produced observations",
            not obs.get("_missing_output") and not obs.get("_parse_error"),
            f"rc={obs.get('_probe_rc')} parse_error={obs.get('_parse_error')}")]
    out.append(("repo cloned", bool(obs.get("commit")), f"commit={obs.get('commit', '')[:12]}"))
    disp = _b(obs, "xserver", "display")
    # A missing display is "we could not tell", not "the app cannot launch". That distinction
    # is the whole reason gates are separate from the verdict.
    out.append(("X server started", bool(disp), f"display={disp}"))
    return out


def table(obs: dict) -> str:
    rows = ["| step | ok | detail |", "|---|---|---|"]

    dv = obs.get("devroot") or {}
    rows.append(f"| rootless -dev closure | {'yes' if dv.get('extracted') else 'NO'} | "
                f"{dv.get('downloaded')} of {dv.get('requested')} packages downloaded, "
                f"{dv.get('extracted')} unpacked, {len(dv.get('failed') or [])} unavailable "
                f"({dv.get('seconds')}s) |")
    hs = dv.get("header_spotcheck") or {}
    rows.append(f"| headers on disk | {'yes' if all(hs.values()) else 'partial' if any(hs.values()) else 'NO'} | "
                + ", ".join(f"{k.rsplit('/', 1)[-1]}={'y' if v else 'n'}" for k, v in hs.items()) + " |")

    cl = obs.get("compile_link") or {}
    rows.append(f"| **compile + link against webkit2gtk-4.1** | "
                f"{'yes' if cl.get('stage') == 'ok' else 'NO'} | "
                f"stage={cl.get('stage')} output={cl.get('output') or '-'} |")

    ru = obs.get("rust") or {}
    rows.append(f"| rust toolchain | {'yes' if ru.get('version') else 'NO'} | "
                f"{ru.get('version') or ru.get('fatal') or '-'} "
                f"({'preexisting' if ru.get('preexisting_cargo') else 'rustup, private home'}, "
                f"{ru.get('seconds')}s) |")

    nb = obs.get("npm_build") or {}
    d = obs.get("dist") or {}
    rows.append(f"| production frontend bundle | {'yes' if d.get('index_html') else 'NO'} | "
                f"npm ci {(obs.get('npm_ci') or {}).get('seconds')}s, build "
                f"{nb.get('seconds')}s, dist {d.get('files')} files / "
                f"{round((d.get('bytes') or 0) / 1e6, 1)} MB |")

    cb = obs.get("cargo_build") or {}
    bi = obs.get("binary") or {}
    rows.append(f"| `cargo build --{obs.get('profile')}` | "
                f"{'yes' if cb.get('rc') == 0 else 'NO'} | rc={cb.get('rc')} in "
                f"{cb.get('seconds')}s |")
    rows.append(f"| binary produced | {'yes' if bi.get('path') else 'NO'} | "
                f"{(bi.get('path') or '-').rsplit('/', 1)[-1]}, "
                f"{round((bi.get('bytes') or 0) / 1e6, 1)} MB |")

    la = obs.get("launch") or {}
    rows.append(f"| launched and stayed up | {'yes' if la.get('alive_at_end') else 'NO'} | "
                f"pid={la.get('pid')} exit={la.get('exit_code')} "
                f"samples={len(la.get('samples') or [])} |")
    wins = la.get("windows") or ""
    n_children = wins.count("+-") if wins else 0
    rows.append(f"| toplevel window mapped | {'yes' if n_children else 'NO'} | "
                f"{n_children} windows in the X tree |")

    empty = (obs.get("screenshot_empty") or {}).get("png_bytes") or 0
    painted = ((la.get("screenshot") or {}).get("png_bytes")) or 0
    ratio = round(painted / empty, 1) if empty else None
    rows.append(f"| **frontend painted** | "
                f"{'yes' if (ratio or 0) >= MIN_PAINT_RATIO else 'NO'} | "
                f"screenshot {painted} B against an empty framebuffer of {empty} B "
                f"(x{ratio}) |")

    rows.append("| | | |")
    ri = obs.get("renderer_inputs") or {}
    rows.append(f"| linux_webkit.rs input: NVIDIA module | "
                f"{'present' if ri.get('nvidia_driver_version_path_exists') else 'absent'} | "
                f"/proc/driver/nvidia/version |")
    rows.append(f"| linux_webkit.rs input: Wayland | "
                f"{'yes' if ri.get('WAYLAND_DISPLAY') or ri.get('WAYLAND_SOCKET') else 'no'} | "
                f"WAYLAND_DISPLAY={ri.get('WAYLAND_DISPLAY')!r} "
                f"XDG_RUNTIME_DIR={ri.get('XDG_RUNTIME_DIR')!r} |")
    rows.append(f"| linux_webkit.rs input: APPIMAGE | "
                f"{'set' if ri.get('APPIMAGE') else 'unset'} | "
                f"not bundled on purpose; an AppImage would take a different branch |")
    env = la.get("app_renderer_env") or {}
    applied = {k: v for k, v in env.items() if not k.startswith("_") and v is not None}
    rows.append(f"| **what the app actually set on itself** | "
                f"{'a workaround' if applied else 'nothing'} | "
                f"{applied or 'no WEBKIT_* renderer variable present in /proc/<pid>/environ, '
                             'i.e. RenderingPlan::PreserveEnvironment'} |")
    return "\n".join(rows)


def _built(obs) -> bool:
    return bool(_b(obs, "binary", "path")) and _b(obs, "cargo_build", "rc") == 0


def _painted(obs) -> tuple[bool, float | None]:
    empty = (obs.get("screenshot_empty") or {}).get("png_bytes") or 0
    painted = _b(obs, "launch", "screenshot", "png_bytes") or 0
    if not empty or not painted:
        return False, None
    ratio = painted / empty
    return ratio >= MIN_PAINT_RATIO, round(ratio, 1)


def verdict(obs: dict) -> tuple[str, str]:
    cl = obs.get("compile_link") or {}
    if not _built(obs):
        return ("NOT_BUILDABLE",
                f"cargo returned rc={_b(obs, 'cargo_build', 'rc')} and produced "
                f"{_b(obs, 'binary', 'path') or 'no executable'}. The -dev closure "
                f"{'did' if cl.get('stage') == 'ok' else 'did NOT'} link. This is a finding "
                f"about the host, not a broken run: Desktop cannot be built here as things "
                f"stand, and any Desktop measurement would have to come from a prebuilt "
                f"artifact instead, which sets APPIMAGE and takes a different branch of "
                f"linux_webkit.rs")

    la = obs.get("launch") or {}
    if not la.get("alive_at_end"):
        return ("BUILDS_BUT_DOES_NOT_RUN",
                f"the binary built and then exited with {la.get('exit_code')} within "
                f"{len(la.get('samples') or [])} samples. The stderr and tauri.log are in the "
                f"artifact and name the reason; a process that exits is not an app")

    ok, ratio = _painted(obs)
    if not ok:
        return ("RUNS_BUT_DID_NOT_PAINT",
                f"the process survived and a window was mapped, but the framebuffer is "
                f"{'indistinguishable from empty' if ratio is not None else 'unreadable'}"
                f"{f' (x{ratio} of the empty frame, floor x{MIN_PAINT_RATIO})' if ratio else ''}. "
                f"'The process started' is exactly the claim this report refuses to accept "
                f"for 'it functions'")

    env = la.get("app_renderer_env") or {}
    applied = {k: v for k, v in env.items() if not k.startswith("_") and v is not None}
    return ("BUILDS_AND_LAUNCHES",
            f"built from source with a rootless -dev closure and a private rustup toolchain, "
            f"launched under Xvfb, stayed up, mapped a toplevel and painted it at x{ratio} the "
            f"byte size of the same framebuffer photographed empty. On this host "
            f"linux_webkit.rs "
            f"{'applied ' + ', '.join(applied) if applied else 'applied NO workaround'}, read "
            f"back from /proc/<pid>/environ of the live process rather than from a log line "
            f"that is only written when a workaround IS applied. This leg ran with "
            f"LIBGL_ALWAYS_SOFTWARE=1 and is the negative control for the measuring job, so it "
            f"establishes nothing about GPU compositing")


def observed_capabilities(obs: dict) -> dict[str, bool]:
    ok, _ = _painted(obs)
    return {
        "rust_toolchain": bool((obs.get("rust") or {}).get("version")),
        "tauri_build_deps": (obs.get("compile_link") or {}).get("stage") == "ok",
        "desktop_tauri_app": bool(_built(obs) and _b(obs, "launch", "alive_at_end") and ok),
        "studio_production_bundle": bool(_b(obs, "dist", "index_html")),
    }

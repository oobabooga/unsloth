#!/usr/bin/env python3
"""Criteria: can Unsloth Desktop (Tauri v2) be built and run on this host at all?

Judges only. Observations come from probes/desktop_toolchain_probe.py.

This is a scouting question and it is deliberately separated from the measurement it enables,
for one reason: the measurement needs the GPU concurrency group and this does not. The group
holds one runner plus one waiter and a third arrival cancels the waiter, so spending it on
"does cargo exist" would be a waste of somebody else's run.

What decides, and what does not:

  * A pkg-config MODVERSION is not a build. `pkg-config --modversion webkit2gtk-4.1` succeeds
    on a tree with a .pc file and no headers, which is precisely the state a rootless
    `dpkg-deb -x` leaves behind when a dependency of the -dev package failed to download. So
    the deciding evidence is a COMPILED AND LINKED binary that calls webkit_get_major_version()
    and prints it. Anything less is corroboration.
  * The RUNTIME libraries being present says nothing about the build, and vice versa. This
    host demonstrably runs WebKitGTK 2.52.3 already, so `libwebkit2gtk-4.1.so.0` is a
    foregone conclusion and is reported, not weighed.
  * Missing headers are a FINDING and not a failure, as long as we could tell. NEEDS_FETCH
    means the rootless apt path is required and worked; NOT_BUILDABLE means it did not.
"""

from __future__ import annotations

TITLE = "Can Unsloth Desktop (Tauri v2) be built and run on the AMD CI runner?"
MODE = "capability"

# Every capability the QUESTION touches, not the ones this host has. Desktop on Windows is
# WebView2 and on macOS is WKWebView; neither is reachable here and a report that stays quiet
# about that overstates its reach by exactly the platforms the user cares about.
NEEDS = [
    "linux", "webkitgtk", "headless_display_server", "rust_toolchain",
    "tauri_build_deps", "gpu_browser_compositing", "drm_render_node",
    "windows", "windows_webview2", "macos", "macos_wkwebview", "wayland_session",
]

BUILD_ESSENTIAL_PC = ["webkit2gtk-4.1", "javascriptcoregtk-4.1", "libsoup-3.0", "gtk+-3.0"]


def _pc(obs, key):
    return obs.get(key) or {}


def _best_pc(obs):
    """The state after the rootless fetch if there was one, else the state as found."""
    after = _pc(obs, "pkg_config_build_after")
    return after if after else _pc(obs, "pkg_config_build_before")


def _best_link(obs):
    after = obs.get("compile_link_after")
    return after if after else obs.get("compile_link_before") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    out.append(("probe produced observations",
                not obs.get("_missing_output") and not obs.get("_parse_error"),
                f"rc={obs.get('_probe_rc')} parse_error={obs.get('_parse_error')}"))
    out.append(("linux host", "Linux" in (obs.get("uname") or ""), obs.get("uname", "")[:120]))
    out.append(("repo checked out",
                bool(obs.get("commit")) and obs.get("src_tauri", {}).get("exists", False),
                f"commit={obs.get('commit', '')[:12]} src_tauri="
                f"{obs.get('src_tauri', {}).get('exists')}"))
    net = obs.get("net") or {}
    reachable = all((net.get(k) or {}).get("stdout", "").strip().startswith(("2", "3"))
                    for k in ("crates_io", "npm", "github"))
    out.append(("crates.io, npm and github reachable", reachable,
                ", ".join(f"{k}={(net.get(k) or {}).get('stdout', '?').strip()}"
                          for k in ("crates_io", "npm", "github"))))
    return out


def table(obs: dict) -> str:
    rows = ["| item | found | detail |", "|---|---|---|"]
    v = obs.get("versions") or {}
    w = obs.get("which") or {}
    for tool in ("rustc", "cargo", "node", "npm", "pkg-config", "cc"):
        val = v.get(tool)
        rows.append(f"| `{tool}` | {'yes' if w.get(tool) else 'NO'} | "
                    f"{(val[0] if val else w.get(tool) or '-')} |")

    before = _pc(obs, "pkg_config_build_before")
    after = _pc(obs, "pkg_config_build_after")
    rows.append("| | | |")
    for name in BUILD_ESSENTIAL_PC:
        b = before.get(name, {})
        a = after.get(name, {}) if after else {}
        detail = f"as found: {b.get('version') or 'missing'}"
        if after:
            detail += f"; after rootless fetch: {a.get('version') or 'missing'}"
        rows.append(f"| pkg-config `{name}` | "
                    f"{'yes' if (a.get('present') if after else b.get('present')) else 'NO'} | "
                    f"{detail} |")

    link = _best_link(obs)
    rows.append(f"| **compile + link against webkit2gtk-4.1** | "
                f"{'yes' if link.get('stage') == 'ok' else 'NO'} | "
                f"stage={link.get('stage')} output={link.get('output') or '-'} |")

    rt = obs.get("runtime_sonames") or {}
    for name in ("libwebkit2gtk-4.1.so.0", "libgtk-3.so.0", "libEGL.so.1"):
        hits = rt.get(name) or []
        rows.append(f"| runtime `{name}` | {'yes' if hits else 'NO'} | {(hits[0] if hits else '-')[:90]} |")
    wr = obs.get("webkit_runtime_version") or {}
    rows.append(f"| WebKitGTK runtime version | "
                f"{'yes' if wr.get('rc') == 0 else 'NO'} | {(wr.get('stdout') or '').strip()} |")

    apt = obs.get("apt_fetch")
    if apt:
        got = [p for p, e in (apt.get("packages") or {}).items()
               if e.get("download", {}).get("rc") == 0]
        missed = [p for p, e in (apt.get("packages") or {}).items()
                  if e.get("download", {}).get("rc") != 0]
        rows.append(f"| rootless -dev fetch | {'partial' if missed else 'yes'} | "
                    f"downloaded {len(got)}, failed {len(missed)}"
                    f"{': ' + ', '.join(missed) if missed else ''} |")
    return "\n".join(rows)


def verdict(obs: dict) -> tuple[str, str]:
    w = obs.get("which") or {}
    link = _best_link(obs)
    pc = _best_pc(obs)

    missing_tools = [t for t in ("cargo", "rustc", "node", "npm", "cc") if not w.get(t)]
    missing_pc = [n for n in BUILD_ESSENTIAL_PC if not pc.get(n, {}).get("present")]
    linked = link.get("stage") == "ok"

    if not missing_tools and linked and not missing_pc:
        fetched = bool(obs.get("apt_fetch"))
        how = ("after a rootless apt-get download + dpkg-deb -x of the -dev closure"
               if fetched else "with the headers already on the host")
        return ("BUILDABLE",
                f"cargo, node and a C toolchain are present and a binary compiled and LINKED "
                f"against webkit2gtk-4.1 {how}, printing {link.get('output')!r}. That is a "
                f"linked artifact, not a pkg-config string")

    why = []
    if missing_tools:
        why.append(f"missing tools: {', '.join(missing_tools)}")
    if missing_pc:
        why.append(f"pkg-config still cannot resolve: {', '.join(missing_pc)}")
    if not linked:
        why.append(f"the compile+link proof stopped at stage={link.get('stage')!r}")
    return ("NOT_BUILDABLE",
            "; ".join(why) + ". This is a finding about the host, not a broken run: Desktop "
            "cannot be built from source here as things stand, and any Desktop measurement "
            "would have to come from a prebuilt artifact instead")


def observed_capabilities(obs: dict) -> dict[str, bool]:
    w = obs.get("which") or {}
    link = _best_link(obs)
    return {
        "rust_toolchain": bool(w.get("cargo") and w.get("rustc")),
        "tauri_build_deps": link.get("stage") == "ok",
        "webkitgtk": (obs.get("webkit_runtime_version") or {}).get("rc") == 0,
    }

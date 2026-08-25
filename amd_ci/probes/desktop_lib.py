#!/usr/bin/env python3
"""Shared, rootless build plumbing for the Unsloth Desktop (Tauri) probes.

No judging here and no measuring: this module only makes a build possible on a host with no
root. It is imported by desktop_build_probe.py and desktop_ladder_probe.py.

Two things the AMD CI runner does not have, established by the toolchain scouting run
(oobabooga/unsloth run 32803564877, job "Desktop build toolchain (no GPU)"):

  * NO rustc, cargo or rustup anywhere on PATH.
  * NO -dev headers for the Tauri v2 Linux closure. `pkg-config --modversion webkit2gtk-4.1`
    fails, as do gtk+-3.0, pango, atk, gdk-pixbuf and libsoup-3.0.

Both are fixable without root, and the second one matters more than it looks: the RUNTIME
libwebkit2gtk-4.1 on this host is 2.52.3, and `apt-get download libwebkit2gtk-4.1-dev` resolves
to 2.52.3-0ubuntu0.24.04.1, the headers for exactly that library. So the app is built against
the engine it will run against, rather than against a different WebKit that happens to be
API compatible.

The first attempt at the -dev fetch got this wrong in a way worth recording, because it looked
like a host limitation and was not:

  * `apt-get download` fetches ONE package and none of its dependencies, so gtk+-3.0.pc landed
    with no atk, pango, cairo or gdk-pixbuf .pc behind it and every Requires: line dangled.
  * Setting PKG_CONFIG_LIBDIR to only the fetched tree then removed the SYSTEM .pc directory
    as well, so even glib-2.0, which the host does have, stopped resolving. The result read as
    "nothing is available" when in fact everything was, in two places at once.

  * `apt-cache depends --recurse` then returned 234 packages for eleven roots on this runner,
    where the same command on a developer box returns 390 for `libgtk-3-dev` alone, and the
    ones it omitted were precisely the header-carrying `libpango1.0-dev`, `libatk1.0-dev`,
    `libcairo2-dev` and `libgdk-pixbuf-2.0-dev`. That extracted cleanly, resolved
    `webkit2gtk-4.1`, and failed the build twenty minutes later inside `pango-sys`.

So the closure is walked HERE, breadth-first over single-level `apt-cache depends`, every
header-carrying -dev package is named explicitly rather than inferred, and PKG_CONFIG_PATH is
the fetched tree FOLLOWED BY the system directories, with PKG_CONFIG_LIBDIR left alone.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

# The roots of the Tauri v2 Linux build closure. Everything else arrives as a dependency.
DEV_ROOTS = [
    "libwebkit2gtk-4.1-dev",
    "libjavascriptcoregtk-4.1-dev",
    "libsoup-3.0-dev",
    "libgtk-3-dev",
    "libayatana-appindicator3-dev",
    "librsvg2-dev",
    "libxdo-dev",
    "libssl-dev",
    # Added after a real run: `libdbus-sys` panics its build script when dbus-1.pc is absent,
    # which is not obvious from the Tauri dependency table because nothing in tauri.conf.json
    # or Cargo.toml names dbus. It arrives through the notification / opener plugins.
    "libdbus-1-dev",
    # Insurance of the same kind. These are preinstalled on GitHub's ubuntu-22.04 image, which
    # is why the repo's own studio-tauri-smoke.yml does not name them, and are absent here.
    "libudev-dev",
    "zlib1g-dev",
    # The GTK -dev closure, named explicitly rather than left to a dependency walk. Each of
    # these is a Depends of libgtk-3-dev and each carries headers a -sys crate compiles
    # against; naming them costs one apt round trip apiece and removes a whole class of
    # "the build failed 20 minutes in at pango-sys" cycles.
    "libpango1.0-dev", "libatk1.0-dev", "libatk-bridge2.0-dev", "libcairo2-dev",
    "libgdk-pixbuf-2.0-dev", "libglib2.0-dev", "libepoxy-dev", "libharfbuzz-dev",
    "libfreetype-dev", "libfontconfig-dev", "libfribidi-dev", "libpng-dev", "libjpeg-dev",
    "libx11-dev", "libxext-dev", "libxrender-dev", "libxi-dev", "libxtst-dev",
    "libxcomposite-dev", "libxcursor-dev", "libxdamage-dev", "libxfixes-dev",
    "libxinerama-dev", "libxrandr-dev", "libxkbcommon-dev", "libwayland-dev",
    "wayland-protocols", "libegl1-mesa-dev", "libgl-dev", "libgles-dev",
]

# Where a Debian package puts .pc files. Both are searched under the fetched root.
PC_SUBDIRS = ("usr/lib/x86_64-linux-gnu/pkgconfig", "usr/share/pkgconfig", "usr/lib/pkgconfig")

SYSTEM_PC = ("/usr/lib/x86_64-linux-gnu/pkgconfig", "/usr/share/pkgconfig", "/usr/lib/pkgconfig")


def sh(cmd, timeout=600, env=None, cwd=None):
    try:
        e = dict(os.environ)
        if env:
            e.update(env)
        r = subprocess.run(cmd, capture_output = True, text = True, timeout = timeout,
                           env = e, cwd = cwd)
        return {"rc": r.returncode, "stdout": r.stdout[-20000:], "stderr": r.stderr[-6000:]}
    except Exception as ex:  # noqa: BLE001
        return {"error": f"{type(ex).__name__}: {ex}"}


_PKG_RE = re.compile(r"^[a-z0-9][a-z0-9+.-]*$")


_DEP_RE = re.compile(r"^\s+\|?(?:Pre)?Depends:\s+(\S+)\s*$")


def _closure_bfs(roots: list[str], max_pkgs: int = 900,
                 max_rounds: int = 3) -> tuple[list[str], dict]:
    """Breadth-first over single-level `apt-cache depends`, done here rather than by apt.

    `apt-cache depends --recurse` was tried first and is what this replaces. On the AMD CI
    runner it returned 234 packages for eleven roots, where the same command on a developer
    box returns 390 for `libgtk-3-dev` ALONE, and the packages it omitted were exactly the ones
    that carry the headers: `libpango1.0-dev`, `libatk1.0-dev`, `libcairo2-dev`,
    `libgdk-pixbuf-2.0-dev`. The result extracted cleanly, resolved `webkit2gtk-4.1` and then
    failed the build at `pango-sys`, which is a much more expensive way to find out.

    Single-level `apt-cache depends` is unambiguous and the traversal is ours, so the outcome
    no longer depends on which apt behaviour a host happens to have. Virtual packages
    (`<name>`) are skipped because `apt-get download` cannot fetch them.
    """
    seen: set[str] = set()
    order: list[str] = []
    queue = list(roots)
    rounds = 0
    # Depth-bounded, because every header this build needs lives in a -dev package that is
    # either a root or one hop from one, while an unbounded walk wanders off into the whole
    # runtime archive and costs an apt round trip per hop for nothing.
    while queue and len(order) < max_pkgs and rounds < max_rounds:
        rounds += 1
        batch, queue = queue, []
        for name in batch:
            if name in seen or not _PKG_RE.match(name):
                continue
            seen.add(name)
            order.append(name)
            r = sh(["apt-cache", "depends", "--no-recommends", "--no-suggests", "--no-conflicts",
                    "--no-breaks", "--no-replaces", "--no-enhances", name], timeout = 120)
            for ln in (r.get("stdout") or "").splitlines():
                m = _DEP_RE.match(ln)
                if not m:
                    continue
                dep = m.group(1)
                if dep.startswith("<") or dep.endswith(">") or dep in seen:
                    continue
                if _PKG_RE.match(dep):
                    queue.append(dep)
    return order, {"count": len(order), "rounds": rounds, "roots": len(roots),
                   "stopped_early": bool(queue),
                   "truncated": len(order) >= max_pkgs}


def _closure(roots: list[str]) -> tuple[list[str], dict]:
    """Every package `apt-cache depends --recurse` reaches from the roots.

    Recommends and Suggests are excluded: they pull in half the archive and none of them
    carries a header. Virtual packages (names in <angle brackets>) are dropped, because
    `apt-get download` cannot fetch them.

    Two things here were learned from a run that looked like it worked. The first version
    filtered lines by prefix and let a name through that `apt-get download` then rejected as
    `ibdeflate0`, one character short: a permissive filter turns any stray byte into a package
    name and the only symptom is a download failure buried in a list of 224 successes. So
    names are now matched against the Debian package-name grammar and anything else is
    reported rather than attempted.

    The second is worse and is the reason that run produced no headers at all: the ROOTS
    themselves are only in the output if the parse worked, so a parse that silently drops them
    yields a large closure of runtime libraries with not one -dev package in it, and
    `pkg-config` then reports the same "not found" a host with nothing installed would. The
    roots are therefore unioned in unconditionally and never depend on parsing.
    """
    cmd = ["apt-cache", "depends", "--recurse", "--no-recommends", "--no-suggests",
           "--no-conflicts", "--no-breaks", "--no-replaces", "--no-enhances", *roots]
    r = sh(cmd, timeout = 900)
    names: list[str] = list(roots)
    rejected: list[str] = []
    for ln in (r.get("stdout") or "").splitlines():
        if not ln or ln[0].isspace() or ln.startswith("|"):
            continue
        name = ln.strip()
        if name.startswith("<") or name.endswith(">"):
            continue  # virtual; apt-get download cannot fetch it
        if not _PKG_RE.match(name):
            rejected.append(name[:60])
            continue
        names.append(name)
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out, {"rc": r.get("rc"), "count": len(out), "roots": roots,
                 "rejected_lines": rejected[:20],
                 "stderr": (r.get("stderr") or "")[-500:]}


def fetch_devroot(work: Path, roots: list[str] | None = None) -> dict:
    """Download and unpack the -dev closure into a private tree. No root, nothing installed.

    Returns a dict carrying `env`, the environment fragment a build must run under.
    """
    roots = roots or DEV_ROOTS
    root = work / "devroot"
    debs = work / "debs"
    root.mkdir(parents = True, exist_ok = True)
    debs.mkdir(parents = True, exist_ok = True)

    pkgs, meta = _closure_bfs(roots)
    out: dict = {"root": str(root), "closure": meta, "requested": len(pkgs)}

    # One apt-get download for the whole list. Individually it is 200+ round trips; in one
    # call apt batches them and a single unavailable package does not abort the rest, it is
    # simply reported. Chunked because the command line is otherwise very long.
    downloaded, failed = [], []
    CHUNK = 40
    for i in range(0, len(pkgs), CHUNK):
        chunk = pkgs[i:i + CHUNK]
        r = sh(["apt-get", "download", *chunk], timeout = 1800, cwd = str(debs))
        if r.get("rc") != 0:
            # Retry one at a time, so a single missing package does not cost the chunk.
            for p in chunk:
                r1 = sh(["apt-get", "download", p], timeout = 300, cwd = str(debs))
                (downloaded if r1.get("rc") == 0 else failed).append(p)
        else:
            downloaded.extend(chunk)
    out["downloaded"] = len(downloaded)
    out["failed"] = failed

    extracted, extract_failed = 0, []
    for f in sorted(debs.glob("*.deb")):
        x = sh(["dpkg-deb", "-x", str(f), str(root)], timeout = 300)
        if x.get("rc") == 0:
            extracted += 1
        else:
            extract_failed.append(f.name)
    out["extracted"] = extracted
    out["extract_failed"] = extract_failed[:20]

    # Did the ROOTS actually land? A closure of 224 runtime libraries with no -dev package in
    # it extracts cleanly and resolves nothing, and that is exactly what happened once. Each
    # root is re-fetched individually if its .deb is not on disk, and the outcome is recorded
    # per root rather than as an aggregate.
    out["roots"] = {}
    for p in roots:
        have = sorted(debs.glob(p + "_*.deb"))
        if not have:
            r = sh(["apt-get", "download", p], timeout = 600, cwd = str(debs))
            have = sorted(debs.glob(p + "_*.deb"))
            for f in have:
                sh(["dpkg-deb", "-x", str(f), str(root)], timeout = 300)
            out["roots"][p] = {"retried": True, "rc": r.get("rc"),
                               "deb": [f.name for f in have],
                               "err": (r.get("stderr") or "")[-200:]}
        else:
            out["roots"][p] = {"retried": False, "deb": [f.name for f in have]}

    pcdirs = [str(root / s) for s in PC_SUBDIRS if (root / s).is_dir()]
    out["pkgconfig_dirs"] = pcdirs
    # The fetched tree FIRST, the system directories after it. Not PKG_CONFIG_LIBDIR: that
    # REPLACES the search path rather than extending it, and doing so is what made the first
    # attempt report that even glib was missing.
    libdir = str(root / "usr/lib/x86_64-linux-gnu")
    out["env"] = {
        "PKG_CONFIG_PATH": ":".join(pcdirs + list(SYSTEM_PC)),
        # THE LINK SEARCH PATH, and it is not optional even though pkg-config resolves.
        # Observed: every crate compiled and the final link then failed with `unable to find
        # library -lgdk-3`, `-lwebkit2gtk-4.1`, `-lpango-1.0` and a dozen more. The .pc files
        # hardcode `libdir=/usr/lib/x86_64-linux-gnu`, so the `-L` they emit points at the
        # SYSTEM directory, which has the runtime `libgdk-3.so.0` and not the unversioned
        # `libgdk-3.so` symlink that only the -dev package ships. Both the gcc driver
        # (LIBRARY_PATH) and rustc (-L native=) are told, because the failing link went
        # through gcc to rust-lld and either one alone leaves a way for it to be missed.
        "LIBRARY_PATH": ":".join(
            [libdir] + ([os.environ["LIBRARY_PATH"]] if os.environ.get("LIBRARY_PATH") else [])),
        "RUSTFLAGS": " ".join(
            [f"-L native={libdir}"] + ([os.environ["RUSTFLAGS"]]
                                       if os.environ.get("RUSTFLAGS") else [])),
        # pkgconf recomputes `prefix` from where it found the .pc file, so the -I flags point
        # into the fetched tree without a sysroot. Recorded rather than assumed: the probe
        # verifies it by compiling and LINKING, not by reading a version string.
        "LD_LIBRARY_PATH": ":".join(
            [str(root / "usr/lib/x86_64-linux-gnu")] +
            ([os.environ["LD_LIBRARY_PATH"]] if os.environ.get("LD_LIBRARY_PATH") else [])),
    }
    out["header_spotcheck"] = {
        p: (root / p).exists() for p in (
            "usr/include/webkitgtk-4.1/webkit2/webkit2.h",
            "usr/include/gtk-3.0/gtk/gtk.h",
            "usr/include/atk-1.0/atk/atk.h",
            "usr/include/pango-1.0/pango/pango.h",
            "usr/include/gdk-pixbuf-2.0/gdk-pixbuf/gdk-pixbuf.h",
            "usr/include/libsoup-3.0/libsoup/soup.h",
            "usr/include/cairo/cairo.h",
        )
    }
    return out


C_PROBE = r"""
#include <webkit2/webkit2.h>
#include <gtk/gtk.h>
#include <stdio.h>
int main(void) {
    printf("webkit %u.%u.%u gtk %u.%u.%u\n",
           webkit_get_major_version(), webkit_get_minor_version(), webkit_get_micro_version(),
           gtk_get_major_version(), gtk_get_minor_version(), gtk_get_micro_version());
    return 0;
}
"""


def compile_link_proof(work: Path, env=None) -> dict:
    """A LINKED binary that prints the engine version. Headers present is only a claim."""
    src = work / "wkprobe.c"
    binp = work / "wkprobe"
    src.write_text(C_PROBE)
    # --define-prefix: the fetched .pc files carry `prefix=/usr`, so without it the -I flags
    # point at the SYSTEM include tree, which has no webkit2 headers, and this probe reports a
    # failure on a host where cargo links the real thing perfectly well. Observed exactly that:
    # `cargo build --release` produced a 71 MB binary in the same job where this said
    # `fatal error: webkit2/webkit2.h: No such file or directory`.
    flags = sh(["pkg-config", "--define-prefix", "--cflags", "--libs", "webkit2gtk-4.1"],
               timeout = 120, env = env)
    if flags.get("rc") != 0:
        flags = sh(["pkg-config", "--cflags", "--libs", "webkit2gtk-4.1"], timeout = 120,
                   env = env)
    if flags.get("rc") != 0:
        return {"stage": "pkg-config", "detail": flags}
    cc = shutil.which("cc") or shutil.which("gcc") or ""
    if not cc:
        return {"stage": "no-cc"}
    build = sh([cc, str(src), "-o", str(binp), *(flags.get("stdout") or "").split()],
               timeout = 900, env = env)
    if build.get("rc") != 0:
        return {"stage": "compile", "detail": build}
    run = sh([str(binp)], timeout = 120, env = env)
    return {"stage": "ok" if run.get("rc") == 0 else "run",
            "output": (run.get("stdout") or "").strip(), "detail": run}


def descendants(root_pid: int) -> set[str]:
    """Every process under `root_pid`, by walking /proc ppid chains."""
    parent: dict[str, str] = {}
    for p in _proc_dirs():
        try:
            stat = Path(p, "stat").read_text()
            parent[os.path.basename(p)] = stat.rsplit(")", 1)[1].split()[1]
        except Exception:  # noqa: BLE001
            continue
    out: set[str] = set()
    for pid in parent:
        cur, hops = pid, 0
        while cur in parent and hops < 40:
            if cur == str(root_pid):
                out.add(pid)
                break
            cur = parent[cur]
            hops += 1
    return out


def _proc_dirs():
    import glob as _glob
    return _glob.glob("/proc/[0-9]*")


def sample_amdgpu(root_pid: int, name_filter=("WebKit", "webkit", "unsloth")) -> dict:
    """amdgpu's own per-process accounting for the app and its children.

    This is the ONLY reading that can say what rendered, and the reason is worth restating
    every time it is used: WebKitGTK hardcodes `Apple GPU` / `Apple Inc.` into
    WEBGL_debug_renderer_info on Linux, on AMD, in cross-platform WebCore, and the runtime
    unmask switch was deleted upstream. So there is no in-page device string, and `gl.RENDERER`
    returns `WebKit WebGL`. What is left is the kernel driver's own books: `drm-engine-gfx`
    nanoseconds and `drm-memory-vram`, per open file description, keyed by `drm-client-id`,
    written by amdgpu and by nothing in userspace.

    Sampled twice and DIFFERENCED per client id, because a cumulative counter read once cannot
    say when the time was accrued. Mapped shared objects are recorded unfiltered beside it: a
    name is corroboration only, never the deciding gate, because since Mesa 24.2 one
    `libgallium-*.so` contains radeonsi and llvmpipe alike and `libdrm_amdgpu.so.1` is a hard
    DT_NEEDED of it that maps in the SOFTWARE leg too.
    """
    import glob as _glob
    mine = descendants(root_pid) | {str(root_pid)}
    procs = []
    for p in _proc_dirs():
        pid = os.path.basename(p)
        if pid not in mine:
            continue
        try:
            cmd = Path(p, "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors = "replace").strip()
        except Exception:  # noqa: BLE001
            continue
        if name_filter and not any(t in cmd for t in name_filter):
            continue
        entry = {"pid": pid, "cmdline": cmd[:200], "dri_fds": [], "fdinfo": [],
                 "mapped_all": []}
        try:
            names = set()
            for ln in Path(p, "maps").read_text(errors = "replace").splitlines():
                path = ln.split(" ", 5)[-1].strip() if " " in ln else ""
                base = os.path.basename(path)
                if ".so" in base:
                    names.add(base)
            entry["mapped_all"] = sorted(names)[:400]
        except Exception:  # noqa: BLE001
            pass
        for fd in _glob.glob(f"{p}/fd/*"):
            try:
                tgt = os.readlink(fd)
            except Exception:  # noqa: BLE001
                continue
            if not tgt.startswith("/dev/dri/"):
                continue
            entry["dri_fds"].append(tgt)
            try:
                txt = Path(p, "fdinfo", os.path.basename(fd)).read_text(errors = "replace")
                entry["fdinfo"].append({
                    k: v.strip() for k, v in
                    (ln.split(":", 1) for ln in txt.splitlines() if ":" in ln)
                    if k.startswith("drm-")})
            except Exception:  # noqa: BLE001
                pass
        procs.append(entry)
    return {"t_monotonic": time.monotonic(), "t_wall": time.time(), "processes": procs}


def amdgpu_delta(early: dict, late: dict) -> dict:
    """GFX engine nanoseconds accrued BETWEEN two samples, differenced per drm-client-id."""
    def by_client(sample):
        out = {}
        for pr in (sample or {}).get("processes", []):
            for fi in pr.get("fdinfo", []):
                cid = fi.get("drm-client-id")
                if not cid:
                    continue
                try:
                    gfx = int(str(fi.get("drm-engine-gfx", "0")).split()[0])
                except Exception:  # noqa: BLE001
                    gfx = 0
                vram = str(fi.get("drm-memory-vram", ""))
                prev = out.get(cid, {"gfx_ns": 0, "vram": vram, "pid": pr["pid"],
                                     "cmdline": pr["cmdline"]})
                prev["gfx_ns"] = max(prev["gfx_ns"], gfx)
                prev["vram"] = vram or prev["vram"]
                out[cid] = prev
        return out

    a, b = by_client(early), by_client(late)
    rows = []
    for cid, late_row in b.items():
        base = a.get(cid, {}).get("gfx_ns", 0)
        rows.append({"client_id": cid, "pid": late_row["pid"],
                     "cmdline": late_row["cmdline"], "vram": late_row["vram"],
                     "gfx_ns_early": base, "gfx_ns_late": late_row["gfx_ns"],
                     "gfx_ns_delta": late_row["gfx_ns"] - base})
    return {
        "clients": rows,
        "total_gfx_ns_delta": sum(r["gfx_ns_delta"] for r in rows),
        "total_gfx_ns_late": sum(r["gfx_ns_late"] for r in rows),
        "render_nodes_open": sorted({fd for s in (late or {}).get("processes", [])
                                     for fd in s.get("dri_fds", [])}),
        "seconds": round((late or {}).get("t_monotonic", 0) - (early or {}).get(
            "t_monotonic", 0), 2) if early and late else None,
    }


def start_xserver_exclusive(work: Path, xvfb_root: Path, width=1440, height=900) -> tuple:
    """An X server this job OWNS, on a display number nothing else is using.

    webkit_paint_probe.start_xserver hard-codes `:99`, which is safe for one job and is not
    safe here. Four ephemeral slots share ONE machine and /tmp/.X11-unix is not namespaced, so
    two jobs both asking for :99 is not a conflict that fails loudly: the second Xvfb exits
    because the display is taken, and the poll loop then finds the socket -- the FIRST job's
    socket -- still there and reports success. The screenshot that follows would be a picture
    of somebody else's window, and it would look like proof.

    So: choose a display whose socket does not exist, start the server, and require that OUR
    process is still alive when the socket appears. A socket that appears while our process is
    dead is somebody else's and is refused rather than adopted.
    """
    info: dict = {"attempts": []}
    candidates = [shutil.which("Xvfb"), str(xvfb_root / "usr/bin/Xvfb")]
    binary = next((c for c in candidates if c and os.path.exists(c)), None)
    if binary is None:
        info["error"] = "no Xvfb binary"
        return None, info
    log = work / "xvfb.log"
    # High numbers, and a per-pid stride, so two jobs starting at the same instant do not both
    # begin their scan at the same place.
    start = 120 + (os.getpid() % 40)
    for n in list(range(start, 200)) + list(range(120, start)):
        display = f":{n}"
        sock = f"/tmp/.X11-unix/X{n}"
        if os.path.exists(sock):
            continue
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = (f"{xvfb_root}/usr/lib/x86_64-linux-gnu:{xvfb_root}/usr/lib:"
                                  f"{os.environ.get('LD_LIBRARY_PATH', '')}")
        xkb = (f"{xvfb_root}/usr/share/X11/xkb"
               if (xvfb_root / "usr/share/X11/xkb").is_dir() else "/usr/share/X11/xkb")
        cmd = [binary, display, "-screen", "0", f"{width}x{height}x24", "-nolisten", "tcp",
               "-xkbdir", xkb]
        with open(log, "a") as fh:
            fh.write(f"\n== {cmd}\n")
            fh.flush()
            proc = subprocess.Popen(cmd, stdout = fh, stderr = subprocess.STDOUT, env = env)
        ok = False
        for _ in range(60):
            if proc.poll() is not None:
                break
            if os.path.exists(sock):
                ok = True
                break
            time.sleep(0.5)
        if ok and proc.poll() is None:
            info.update({"display": display, "binary": binary, "screen": f"{width}x{height}",
                         "pid": proc.pid, "owned": True})
            return proc, info
        info["attempts"].append({"display": display, "exit": proc.poll()})
        if proc.poll() is None:
            proc.kill()
    info["error"] = "no display number could be claimed"
    return None, info


def install_rust(work: Path, toolchain: str = "stable") -> dict:
    """rustup into a private CARGO_HOME/RUSTUP_HOME. No root, nothing on the system.

    The crate pins `rust-version = "1.89"`, so a toolchain older than that fails to build and
    says so clearly; nothing here needs to guess a version.
    """
    cargo_home = work / "cargo"
    rustup_home = work / "rustup"
    env = {"CARGO_HOME": str(cargo_home), "RUSTUP_HOME": str(rustup_home)}
    out: dict = {"cargo_home": str(cargo_home), "rustup_home": str(rustup_home)}

    existing = shutil.which("cargo")
    if existing:
        out["preexisting_cargo"] = existing
        out["version"] = sh(["cargo", "--version"], timeout = 120).get("stdout", "").strip()
        out["env"] = {"PATH": os.environ.get("PATH", "")}
        return out

    sh_path = work / "rustup-init.sh"
    dl = sh(["curl", "-sSfL", "--max-time", "300", "-o", str(sh_path),
             "https://sh.rustup.rs"], timeout = 360)
    out["download"] = {"rc": dl.get("rc"), "err": (dl.get("stderr") or "")[-300:]}
    if dl.get("rc") != 0:
        out["fatal"] = "could not download rustup-init"
        return out
    inst = sh(["sh", str(sh_path), "-y", "--no-modify-path", "--profile", "minimal",
               "--default-toolchain", toolchain], timeout = 1800, env = env)
    out["install"] = {"rc": inst.get("rc"), "tail": (inst.get("stdout") or "")[-1500:],
                      "err": (inst.get("stderr") or "")[-800:]}
    bin_dir = cargo_home / "bin"
    out["bin"] = {n: (bin_dir / n).exists() for n in ("cargo", "rustc", "rustup")}
    env["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
    out["env"] = env
    out["version"] = sh([str(bin_dir / "rustc"), "--version"], timeout = 120,
                        env = env).get("stdout", "").strip()
    return out

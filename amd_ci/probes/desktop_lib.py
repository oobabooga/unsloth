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

So the closure is enumerated with `apt-cache depends --recurse`, and PKG_CONFIG_PATH is the
fetched tree FOLLOWED BY the system directories, with PKG_CONFIG_LIBDIR left alone.
"""

from __future__ import annotations

import os
import shutil
import subprocess
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


def _closure(roots: list[str]) -> tuple[list[str], dict]:
    """Every package `apt-cache depends --recurse` reaches from the roots.

    Recommends and Suggests are excluded: they pull in half the archive and none of them
    carries a header. Virtual packages (names in <angle brackets>) are dropped, because
    `apt-get download` cannot fetch them.
    """
    cmd = ["apt-cache", "depends", "--recurse", "--no-recommends", "--no-suggests",
           "--no-conflicts", "--no-breaks", "--no-replaces", "--no-enhances", *roots]
    r = sh(cmd, timeout = 900)
    names: list[str] = []
    for ln in (r.get("stdout") or "").splitlines():
        if ln.startswith(" ") or ln.startswith("|") or not ln.strip():
            continue
        name = ln.strip()
        if name.startswith("<") or ":" in name.split()[0] and not name.startswith("lib"):
            continue
        if name.startswith("<") or name.endswith(">"):
            continue
        names.append(name)
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out, {"rc": r.get("rc"), "count": len(out), "stderr": (r.get("stderr") or "")[-500:]}


def fetch_devroot(work: Path, roots: list[str] | None = None) -> dict:
    """Download and unpack the -dev closure into a private tree. No root, nothing installed.

    Returns a dict carrying `env`, the environment fragment a build must run under.
    """
    roots = roots or DEV_ROOTS
    root = work / "devroot"
    debs = work / "debs"
    root.mkdir(parents = True, exist_ok = True)
    debs.mkdir(parents = True, exist_ok = True)

    pkgs, meta = _closure(roots)
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

    pcdirs = [str(root / s) for s in PC_SUBDIRS if (root / s).is_dir()]
    out["pkgconfig_dirs"] = pcdirs
    # The fetched tree FIRST, the system directories after it. Not PKG_CONFIG_LIBDIR: that
    # REPLACES the search path rather than extending it, and doing so is what made the first
    # attempt report that even glib was missing.
    out["env"] = {
        "PKG_CONFIG_PATH": ":".join(pcdirs + list(SYSTEM_PC)),
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
    flags = sh(["pkg-config", "--cflags", "--libs", "webkit2gtk-4.1"], timeout = 120, env = env)
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

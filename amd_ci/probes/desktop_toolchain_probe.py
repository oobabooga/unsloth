#!/usr/bin/env python3
"""Probe: what does this host have for BUILDING and RUNNING Unsloth Desktop (Tauri v2)?

Observes only. criteria/desktop_toolchain.py judges.

Why this exists as its own run. The Studio WEB UI ladder has already been measured on this
host in real WebKitGTK and did not collapse. Desktop is the next suspect because
studio/src-tauri/src/linux_webkit.rs may apply WEBKIT_DISABLE_DMABUF_RENDERER, which turns
accelerated compositing OFF, on a trigger its own comment concedes is "module presence, not
the GPU that will render". Before any of that can be measured, one question has to be settled
cheaply and WITHOUT taking the GPU concurrency group: can a Tauri v2 app be built and run
here at all, on a host with no root?

So this takes no GPU, judges nothing, and answers:
  * rust toolchain: rustc/cargo present, version, target;
  * the pkg-config DEV closure Tauri v2 on Linux needs (webkit2gtk-4.1, javascriptcoregtk-4.1,
    libsoup-3.0, gtk+-3.0, ...) -- present, or missing and therefore to be fetched rootlessly;
  * the RUNTIME .so set, which is a different question from the headers and may well be
    satisfied when the headers are not, since the host already runs WebKitGTK 2.52.3;
  * node/npm for the frontend bundle;
  * whether a rootless apt-get download + dpkg-deb -x of the missing -dev packages resolves,
    which is the only mechanism available here (the Xvfb fetch in webkit_paint_probe.py uses
    the same one);
  * network reach to crates.io and the npm registry;
  * an actual end-to-end proof: compile and link a MINIMAL Tauri-shaped C program against
    webkit2gtk-4.1 through pkg-config, so "the headers are there" is demonstrated rather than
    inferred from a file listing.

It reports what it found and decides nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Tauri v2 on Linux. Split deliberately: the first list is what wry/webkit2gtk-rs ask
# pkg-config for at BUILD time, the second is what a built binary dlopens or links at RUN
# time. A host can have the second and not the first, and that is the interesting case here.
BUILD_PC = [
    "webkit2gtk-4.1",
    "javascriptcoregtk-4.1",
    "libsoup-3.0",
    "gtk+-3.0",
    "glib-2.0",
    "gobject-2.0",
    "gio-2.0",
    "cairo",
    "pango",
    "gdk-pixbuf-2.0",
    "atk",
    "gdk-3.0",
    "libjpeg",
    "libpng",
    "xdo",
]
RUNTIME_SONAMES = [
    "libwebkit2gtk-4.1.so.0",
    "libjavascriptcoregtk-4.1.so.0",
    "libsoup-3.0.so.0",
    "libgtk-3.so.0",
    "libgdk-3.so.0",
    "libayatana-appindicator3.so.1",
    "libappindicator3.so.1",
    "libGLESv2.so.2",
    "libEGL.so.1",
]
# The Debian -dev packages that provide the build closure, for the rootless fetch.
DEV_PACKAGES = [
    "libwebkit2gtk-4.1-dev",
    "libjavascriptcoregtk-4.1-dev",
    "libsoup-3.0-dev",
    "libgtk-3-dev",
    "libayatana-appindicator3-dev",
    "librsvg2-dev",
    "libxdo-dev",
    "libssl-dev",
]


def sh(cmd, timeout=300, env=None, cwd=None, shell=False):
    try:
        e = dict(os.environ)
        if env:
            e.update(env)
        r = subprocess.run(cmd, capture_output = True, text = True, timeout = timeout,
                           env = e, cwd = cwd, shell = shell)
        return {"rc": r.returncode, "stdout": r.stdout[-8000:], "stderr": r.stderr[-4000:]}
    except Exception as ex:  # noqa: BLE001
        return {"error": f"{type(ex).__name__}: {ex}"}


def which_all(names):
    return {n: shutil.which(n) for n in names}


def pkg_config(names, env=None):
    out = {}
    for n in names:
        r = sh(["pkg-config", "--modversion", n], timeout = 60, env = env)
        ok = r.get("rc") == 0
        out[n] = {"present": ok,
                  "version": (r.get("stdout") or "").strip() if ok else None,
                  "error": None if ok else (r.get("stderr") or r.get("error") or "")[-200:]}
    return out


def find_soname(name):
    r = sh(["/sbin/ldconfig", "-p"], timeout = 60)
    if r.get("rc") != 0:
        r = sh(["ldconfig", "-p"], timeout = 60)
    text = r.get("stdout") or ""
    hits = [ln.strip() for ln in text.splitlines() if name in ln]
    return hits[:3]


def ldconfig_cache():
    r = sh(["/sbin/ldconfig", "-p"], timeout = 60)
    if r.get("rc") != 0:
        r = sh(["ldconfig", "-p"], timeout = 60)
    return r.get("stdout") or ""


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
    """Headers present is a claim; a linked binary that prints the version is evidence."""
    src = work / "wkprobe.c"
    binp = work / "wkprobe"
    src.write_text(C_PROBE)
    cflags = sh(["pkg-config", "--cflags", "--libs", "webkit2gtk-4.1"], timeout = 60, env = env)
    if cflags.get("rc") != 0:
        return {"stage": "pkg-config", "detail": cflags}
    flags = (cflags.get("stdout") or "").split()
    cc = shutil.which("cc") or shutil.which("gcc") or ""
    if not cc:
        return {"stage": "no-cc"}
    build = sh([cc, str(src), "-o", str(binp), *flags], timeout = 600, env = env)
    if build.get("rc") != 0:
        return {"stage": "compile", "cc": cc, "detail": build}
    run = sh([str(binp)], timeout = 60, env = env)
    return {"stage": "ok" if run.get("rc") == 0 else "run", "cc": cc,
            "flags": flags[:40], "output": (run.get("stdout") or "").strip(), "detail": run}


def apt_fetch_dev(work: Path, packages) -> dict:
    """Rootless: apt-get download unpacks with dpkg-deb -x. No root, nothing installed.

    Reported per package rather than aggregated, because a partial closure is the likely
    outcome and knowing WHICH package is unreachable is the whole point.
    """
    root = work / "aptroot"
    debs = work / "debs"
    root.mkdir(parents = True, exist_ok = True)
    debs.mkdir(parents = True, exist_ok = True)
    out = {"root": str(root), "packages": {}}
    for p in packages:
        r = sh(["apt-get", "download", p], timeout = 600, cwd = str(debs))
        entry = {"download": {"rc": r.get("rc"), "err": (r.get("stderr") or "")[-300:]}}
        if r.get("rc") == 0:
            files = sorted(debs.glob(p.split(":")[0] + "_*.deb"))
            entry["deb"] = [f.name for f in files][:3]
            for f in files:
                x = sh(["dpkg-deb", "-x", str(f), str(root)], timeout = 600)
                entry["extract_rc"] = x.get("rc")
        out["packages"][p] = entry
    pcdirs = [str(d) for d in root.rglob("pkgconfig") if d.is_dir()]
    out["pkgconfig_dirs"] = pcdirs[:20]
    out["pc_files"] = sorted({f.name for d in pcdirs for f in Path(d).glob("*.pc")})[:60]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--state", default = "host")
    ap.add_argument("--checkout", default = "")
    ap.add_argument("--work", default = os.environ.get("AMD_CI_WORK", "/tmp/desktop_probe"))
    ap.add_argument("--repo", default = "https://github.com/unslothai/unsloth")
    ap.add_argument("--ref", default = "main")
    ap.add_argument("--try-apt", action = "store_true", default = True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents = True, exist_ok = True)

    work = Path(args.work) / "desktop_probe"
    work.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "ts": time.time(), "work": str(work)}

    obs["os_release"] = Path("/etc/os-release").read_text() if Path("/etc/os-release").is_file() else ""
    obs["uname"] = sh(["uname", "-a"]).get("stdout", "").strip()
    obs["nproc"] = sh(["nproc"]).get("stdout", "").strip()
    obs["free"] = sh(["free", "-g"]).get("stdout", "")
    obs["df_runner_temp"] = sh(["df", "-h", str(work)]).get("stdout", "")
    obs["is_root"] = os.geteuid() == 0

    obs["which"] = which_all(["rustc", "cargo", "rustup", "cc", "gcc", "g++", "pkg-config",
                              "node", "npm", "python3", "git", "curl", "apt-get", "dpkg-deb",
                              "Xvfb", "xvfb-run", "WebKitWebDriver", "patchelf", "file",
                              "ldd", "objdump", "import", "xwd", "convert", "ffmpeg"])
    for tool, flag in (("rustc", "--version"), ("cargo", "--version"), ("node", "--version"),
                       ("npm", "--version"), ("pkg-config", "--version"), ("cc", "--version"),
                       ("rustup", "--version")):
        if obs["which"].get(tool):
            obs.setdefault("versions", {})[tool] = sh([tool, flag], timeout = 120).get(
                "stdout", "").strip().splitlines()[:2]

    # Build-time headers, as the host stands.
    obs["pkg_config_build_before"] = pkg_config(BUILD_PC)
    # Run-time libraries. Different question, likely a different answer.
    cache = ldconfig_cache()
    obs["runtime_sonames"] = {n: [ln.strip() for ln in cache.splitlines() if n in ln][:2]
                              for n in RUNTIME_SONAMES}
    obs["webkit_runtime_version"] = sh(
        ["/usr/bin/python3", "-c",
         "import gi; gi.require_version('WebKit2','4.1'); from gi.repository import WebKit2; "
         "print('%d.%d.%d' % (WebKit2.get_major_version(), WebKit2.get_minor_version(), "
         "WebKit2.get_micro_version()))"], timeout = 120)

    obs["compile_link_before"] = compile_link_proof(work, env = None)

    # The rootless fetch, only if the headers are not already there.
    need_apt = not obs["pkg_config_build_before"].get("webkit2gtk-4.1", {}).get("present")
    obs["needed_apt"] = need_apt
    if need_apt and args.try_apt:
        obs["apt_fetch"] = apt_fetch_dev(work, DEV_PACKAGES)
        root = Path(obs["apt_fetch"]["root"])
        pcpath = ":".join(obs["apt_fetch"]["pkgconfig_dirs"])
        env = {"PKG_CONFIG_PATH": pcpath,
               "PKG_CONFIG_SYSROOT_DIR": "",
               # The .pc files carry absolute /usr paths; without this the -I flags point at
               # the system tree that does not have them.
               "PKG_CONFIG_LIBDIR": pcpath}
        obs["apt_env"] = env
        obs["pkg_config_build_after"] = pkg_config(BUILD_PC, env = env)
        obs["compile_link_after"] = compile_link_proof(work, env = env)

    # Network reach, separately for the two registries a build needs.
    obs["net"] = {
        "crates_io": sh(["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}",
                         "--max-time", "30", "https://static.crates.io/"], timeout = 60),
        "npm": sh(["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "30",
                   "https://registry.npmjs.org/"], timeout = 60),
        "github": sh(["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "30",
                      "https://github.com/"], timeout = 60),
    }

    # The repo, so the manifests are read from the tree that would actually be built rather
    # than from this probe's assumptions about it.
    repo = work / "repo"
    if not repo.exists():
        obs["clone"] = sh(["git", "clone", "--depth", "20", args.repo, str(repo)], timeout = 1800)
        if args.ref and args.ref != "main":
            obs["checkout"] = sh(["git", "checkout", args.ref], cwd = str(repo), timeout = 300)
    obs["commit"] = sh(["git", "rev-parse", "HEAD"], cwd = str(repo)).get("stdout", "").strip()
    st = repo / "studio" / "src-tauri"
    obs["src_tauri"] = {"exists": st.is_dir()}
    if st.is_dir():
        cargo_toml = (st / "Cargo.toml")
        obs["src_tauri"]["cargo_toml_head"] = cargo_toml.read_text()[:4000] if cargo_toml.is_file() else ""
        obs["src_tauri"]["has_lock"] = (st / "Cargo.lock").is_file()
        rt = repo / "rust-toolchain.toml"
        obs["src_tauri"]["rust_toolchain_toml"] = rt.read_text()[:1000] if rt.is_file() else None
        obs["src_tauri"]["cargo_metadata"] = sh(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            cwd = str(st), timeout = 600).get("rc")

    args.out.write_text(json.dumps(obs, indent = 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

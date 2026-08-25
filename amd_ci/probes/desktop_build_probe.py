#!/usr/bin/env python3
"""Probe: build Unsloth Desktop (Tauri v2) from source on this host and prove it LAUNCHES.

Observes only. criteria/desktop_builds_and_launches.py judges.

Deliberately takes NO GPU, and the launch leg is pinned to `LIBGL_ALWAYS_SOFTWARE=1`
`GALLIUM_DRIVER=llvmpipe` for that reason. That is not a compromise: it is the NEGATIVE CONTROL
the GPU run needs. The same binary, the same X server, the same fixture home, run once on the
software rasteriser here and once on the real device in the measuring job, with the renderer the
only difference, is what turns amdgpu's per-process counters into evidence rather than a number.
WebKitGTK hardcodes `Apple GPU` in WEBGL_debug_renderer_info on Linux/AMD, so there is no
in-page string to read and the control is the whole method.

What it does, in order:
  1. fetch the Tauri v2 -dev closure rootlessly and prove it by COMPILING AND LINKING a C
     program against webkit2gtk-4.1 (desktop_lib.fetch_devroot / compile_link_proof);
  2. install a private rustup toolchain, since the runner has none;
  3. clone the repo and build the PRODUCTION frontend bundle (`npm ci && npm run build`). A vite
     dev server is not a substitute and is not attempted: React's dev build inflates the axis
     under investigation by about 3.2x, and Tauri embeds `frontendDist` at compile time anyway;
  4. `cargo build --release`, which is what `tauri build --no-bundle` runs underneath. No
     bundling: an AppImage would set the APPIMAGE environment variable, and
     studio/src-tauri/src/linux_webkit.rs branches on it (linux_webkit.rs:139, :197), so
     bundling would change the very decision this campaign is trying to observe;
  5. start Xvfb and LAUNCH the binary, with every log the app has: its stderr, GTK's own
     `G_MESSAGES_DEBUG=all` stream (without which a fatal X error exits silently), and
     `$HOME/.unsloth/studio/tauri.log`, which is where main.rs::setup_logging writes;
  6. read `/proc/<pid>/environ` of the running app. This is the decisive reading for the
     rendering path and it is NOT the log: main.rs only logs the renderer decision when a
     workaround is APPLIED, so `PreserveEnvironment` is silent and indistinguishable from a
     crash before that line. The environment of the live process says which of
     WEBKIT_DISABLE_DMABUF_RENDERER, WEBKIT_DMABUF_RENDERER_FORCE_SHM and
     UNSLOTH_WEBKIT_RENDERER_WORKAROUND the app set on itself, and the inputs the decision was
     made from are recorded beside it;
  7. screenshot the X server, so "the frontend painted" is a picture and not an assertion.

It reports what happened and decides nothing. In particular it does not decide whether the
window it captured is the Studio shell or the not-installed setup screen: no Studio backend is
installed in this job, so the setup screen is the EXPECTED content here, and the criteria says
so rather than the probe pretending otherwise.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from desktop_lib import (  # noqa: E402
    compile_link_proof, fetch_devroot, install_rust, sh, start_xserver_exclusive,
)
from webkit_paint_probe import fetch_xvfb, inventory  # noqa: E402

# The variables linux_webkit.rs may set on the process, and the marker it uses to remember that
# it set them. Read back out of /proc/<pid>/environ, which is the state the app actually ran
# under rather than what a log line claims.
RENDERER_VARS = [
    "WEBKIT_DISABLE_DMABUF_RENDERER",
    "WEBKIT_DMABUF_RENDERER_FORCE_SHM",
    "WEBKIT_FORCE_DMABUF_RENDERER",
    "UNSLOTH_WEBKIT_RENDERER_WORKAROUND",
]


LADDER = HERE / "desktop_ladder"
# The REPAIRED scene, not studio_ladder/amdv_scene.js. Its own header says it is a copy of
# that file "with ONE block changed: the scroll phase", and the change is not cosmetic: the
# original assigns `el.scrollTop` while the thread viewport carries `scroll-smooth`, so each
# assignment ANIMATES and the next is computed from a stalled position. Measured in Chromium at
# r500K: 54,000 px commanded, 6,610 px travelled, never further than 1,107 px from the bottom of
# a 316,829 px thread. That window is a jiggle that reads a comfortable frame rate because
# almost nothing happens in it, and `scroll_detail` cannot show it because it records only the
# first and last position and both are the bottom either way. The defect is engine-specific --
# WebKitGTK still collapses on the inert gesture -- which makes it exactly the kind of thing to
# fix rather than rely on.
SCENE = HERE / "desktop_ladder" / "chr_scene.js"

# The control server's port has to be a compile-time constant, because the page is served from
# the binary at `tauri://localhost` under a CSP of `default-src 'self'` with no `unsafe-eval`:
# there is no way to tell an already-built bundle where to call. So the number is fixed here,
# the measuring job refuses to run if it is occupied, and both facts are recorded.
AMDV_CONTROL_PORT = 5473


def instrument_frontend(repo: Path) -> dict:
    """Add two classic <script> tags and two files to the frontend, and nothing else.

    Placed in `studio/frontend/public/` and referenced from `studio/frontend/index.html`
    alongside the existing `crypto-boot.js` / `theme-boot.js`, which is the mechanism the app
    already uses and proves survives a vite build: `public/` is copied verbatim and a
    root-absolute classic script src is left alone, because it is a public-dir URL and not a
    module-graph entry. A classic script also runs DURING PARSE, i.e. before the deferred
    module entry, which reproduces the document-start user-script semantics the web UI ladder
    got from WebKitGTK's UserScriptInjectionTime.START.

    Two things this is careful NOT to do. It does not edit `dist/index.html` after the fact:
    that works, but it makes `tauri build` unusable, because `beforeBuildCommand` re-runs vite
    with `emptyOutDir` and silently discards the edit. And it does not touch a single line of
    application code: the scene is the SAME FILE the web UI ladder used, byte for byte, which
    is what makes the two sets of numbers comparable at all.
    """
    fe = repo / "studio" / "frontend"
    pub = fe / "public"
    idx = fe / "index.html"
    out: dict = {"index_html": str(idx)}
    if not idx.is_file() or not pub.is_dir():
        out["error"] = "frontend index.html or public/ missing"
        return out

    scene = SCENE.read_text()
    boot = (LADDER / "amdv_desktop_boot.js").read_text().replace(
        "__AMDV_CONTROL_URL__", f"http://127.0.0.1:{AMDV_CONTROL_PORT}")
    (pub / "av-scene.js").write_text(scene)
    (pub / "av-boot.js").write_text(boot)
    out["scene_bytes"] = len(scene)
    out["boot_bytes"] = len(boot)
    out["scene_sha_source"] = str(SCENE)

    html = idx.read_text()
    tag = '<script src="/av-scene.js"></script>\n    <script src="/av-boot.js"></script>\n    '
    anchor = '<script src="/theme-boot.js"></script>'
    if anchor in html:
        html = html.replace(anchor, anchor + "\n    " + tag.strip(), 1)
    else:
        html = html.replace("</head>", "    " + tag.strip() + "\n</head>", 1)
    idx.write_text(html)
    out["injected"] = ("av-scene.js" in idx.read_text())
    out["anchor_found"] = anchor in html
    return out


def renderer_decision_inputs() -> dict:
    """Every input linux_webkit.rs::rendering_plan reads, sampled from this host.

    Recorded so the decision can be EXPLAINED and not merely observed. The comment at
    linux_webkit.rs:170 concedes the NVIDIA probe is "module presence, not the GPU that will
    render", so whether /proc/driver/nvidia/version exists is the single most load-bearing bit
    on a host like this one.
    """
    x11_dir = "/tmp/.X11-unix"
    return {
        "nvidia_driver_version_path_exists": Path("/proc/driver/nvidia/version").exists(),
        "APPIMAGE": os.environ.get("APPIMAGE"),
        "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY"),
        "WAYLAND_SOCKET": os.environ.get("WAYLAND_SOCKET"),
        "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR"),
        "GDK_BACKEND": os.environ.get("GDK_BACKEND"),
        "DISPLAY": os.environ.get("DISPLAY"),
        "x11_sockets": sorted(os.path.basename(p) for p in glob.glob(x11_dir + "/X*")),
        "libGLESv2_so_2_loadable": _dlopen_ok("libGLESv2.so.2"),
        "preset_renderer_vars": {v: os.environ.get(v) for v in RENDERER_VARS},
    }


def _dlopen_ok(soname: str) -> bool:
    try:
        import ctypes
        ctypes.CDLL(soname)
        return True
    except Exception:  # noqa: BLE001
        return False


def read_environ(pid: int) -> dict:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes().decode(errors = "replace")
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    env = {}
    for item in raw.split("\0"):
        if "=" in item:
            k, v = item.split("=", 1)
            env[k] = v
    return {v: env.get(v) for v in RENDERER_VARS} | {"_count": len(env)}


def process_tree(root_pid: int) -> list[dict]:
    """Every descendant of the app, with its command line. WebKitWebProcess is a child."""
    parent = {}
    for p in glob.glob("/proc/[0-9]*"):
        try:
            stat = Path(p, "stat").read_text()
            parent[os.path.basename(p)] = stat.rsplit(")", 1)[1].split()[1]
        except Exception:  # noqa: BLE001
            continue
    out = []
    for pid in parent:
        cur, hops = pid, 0
        while cur in parent and hops < 40:
            if cur == str(root_pid):
                try:
                    cmd = Path("/proc", pid, "cmdline").read_bytes().replace(
                        b"\0", b" ").decode(errors = "replace").strip()
                except Exception:  # noqa: BLE001
                    cmd = ""
                out.append({"pid": pid, "cmdline": cmd[:200],
                            "renderer_env": read_environ(int(pid))})
                break
            cur = parent[cur]
            hops += 1
    return out


def screenshot(work: Path, display: str, tag: str) -> dict:
    """xwd, then ffmpeg to PNG. `import` and `convert` are absent on this host; ffmpeg is not."""
    xwd = work / f"{tag}.xwd"
    png = work / f"{tag}.png"
    env = {"DISPLAY": display}
    r = sh(["xwd", "-root", "-silent", "-out", str(xwd)], timeout = 120, env = env)
    out = {"xwd_rc": r.get("rc"), "xwd_bytes": xwd.stat().st_size if xwd.is_file() else 0}
    if xwd.is_file() and shutil.which("ffmpeg"):
        c = sh(["ffmpeg", "-y", "-loglevel", "error", "-i", str(xwd), str(png)], timeout = 180)
        out["png_rc"] = c.get("rc")
        out["png"] = str(png)
        out["png_bytes"] = png.stat().st_size if png.is_file() else 0
        # A uniformly blank framebuffer compresses to almost nothing. Not a judgement, a
        # number the criteria can apply a floor to.
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--state", default = "host")
    ap.add_argument("--checkout", default = "")
    ap.add_argument("--work", default = os.environ.get("AMD_CI_WORK", "/tmp/desktop_build"))
    ap.add_argument("--repo", default = "https://github.com/unslothai/unsloth")
    ap.add_argument("--ref", default = "main")
    ap.add_argument("--profile", default = "release", choices = ["release", "debug"])
    ap.add_argument("--launch-seconds", type = int, default = 45)
    ap.add_argument("--stage-dir", default = "",
                    help = "copy the built binary and a manifest here, for the artifact the "
                           "measuring job downloads")
    args = ap.parse_args()
    args.out.parent.mkdir(parents = True, exist_ok = True)

    work = Path(args.work) / "desktop_build"
    work.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "ts_start": time.time(), "work": str(work),
                 "profile": args.profile}
    xproc = None

    try:
        # ── 1. the -dev closure, rootlessly ──
        t0 = time.time()
        obs["devroot"] = fetch_devroot(work)
        obs["devroot"]["seconds"] = round(time.time() - t0, 1)
        build_env = dict(obs["devroot"]["env"])
        obs["compile_link"] = compile_link_proof(work, env = build_env)

        # ── 2. rust ──
        t0 = time.time()
        obs["rust"] = install_rust(work)
        obs["rust"]["seconds"] = round(time.time() - t0, 1)
        build_env.update(obs["rust"].get("env") or {})

        # ── 3. the repo and the PRODUCTION frontend bundle ──
        repo = work / "repo"
        if repo.exists():
            shutil.rmtree(repo, ignore_errors = True)
        obs["clone"] = sh(["git", "clone", "--depth", "50", args.repo, str(repo)], timeout = 2400)
        if args.ref and args.ref != "main":
            obs["checkout"] = sh(["git", "checkout", args.ref], cwd = str(repo), timeout = 600)
        obs["commit"] = sh(["git", "rev-parse", "HEAD"], cwd = str(repo)).get("stdout", "").strip()

        fe = repo / "studio" / "frontend"
        t0 = time.time()
        obs["npm_ci"] = sh(["npm", "ci"], cwd = str(fe), timeout = 3600)
        obs["npm_ci"]["seconds"] = round(time.time() - t0, 1)
        obs["npm_ci"]["stdout"] = (obs["npm_ci"].get("stdout") or "")[-2500:]
        t0 = time.time()
        obs["npm_build"] = sh(["npm", "run", "build"], cwd = str(fe), timeout = 3600)
        obs["npm_build"]["seconds"] = round(time.time() - t0, 1)
        obs["npm_build"]["stdout"] = (obs["npm_build"].get("stdout") or "")[-4000:]
        dist = fe / "dist"
        obs["dist"] = {
            "exists": dist.is_dir(),
            "index_html": (dist / "index.html").is_file(),
            "bytes": sum(f.stat().st_size for f in dist.rglob("*") if f.is_file())
            if dist.is_dir() else 0,
            "files": sum(1 for f in dist.rglob("*") if f.is_file()) if dist.is_dir() else 0,
        }

        # ── 4. cargo build. No bundling: an AppImage would set APPIMAGE and change the
        # rendering decision this campaign exists to observe. ──
        st = repo / "studio" / "src-tauri"
        cargo = shutil.which("cargo") or str(work / "cargo" / "bin" / "cargo")
        cmd = [cargo, "build", "--locked"] + (["--release"] if args.profile == "release" else [])
        t0 = time.time()
        obs["cargo_build"] = sh(cmd, cwd = str(st), timeout = 7200, env = build_env)
        obs["cargo_build"]["seconds"] = round(time.time() - t0, 1)
        # Keep the TAIL of stderr: cargo writes progress there and the error, if any, is last.
        obs["cargo_build"]["stderr"] = (obs["cargo_build"].get("stderr") or "")[-8000:]
        obs["cargo_build"]["cmd"] = cmd

        target = st / "target" / args.profile
        candidates = [p for p in target.glob("*")
                      if p.is_file() and os.access(p, os.X_OK) and p.suffix == ""]
        binary = None
        for name in ("unsloth-studio", "unsloth", "Unsloth"):
            if (target / name).is_file():
                binary = target / name
                break
        if binary is None and candidates:
            binary = max(candidates, key = lambda p: p.stat().st_size)
        obs["binary"] = {"path": str(binary) if binary else None,
                         "bytes": binary.stat().st_size if binary else 0,
                         "candidates": [p.name for p in candidates][:20]}
        if binary is None:
            obs["fatal"] = "cargo produced no executable"
            return 0
        obs["binary"]["ldd"] = sh(["ldd", str(binary)], timeout = 180,
                                  env = build_env).get("stdout", "")[-4000:]

        # ── 4b. the INSTRUMENTED binary, built second and kept separate ──
        #
        # Two binaries on purpose. The pristine one answers "does Unsloth Desktop function on
        # this runner", and that claim has to be about an unmodified app or it is not the
        # claim anyone cares about. The instrumented one carries the ladder, because the
        # scene has to be inside the bundle: `tauri://localhost` runs under
        # `default-src 'self'` with no `unsafe-eval`, so there is no way to inject anything
        # into an already-built binary. Saying which binary produced which number is the whole
        # reason they are not the same build.
        pristine = None
        if args.stage_dir:
            pristine = Path(args.stage_dir) / "desktop-pristine"
            pristine.parent.mkdir(parents = True, exist_ok = True)
            shutil.copy2(binary, pristine)
        obs["instrument"] = instrument_frontend(repo)
        if obs["instrument"].get("injected"):
            t0 = time.time()
            obs["npm_build_instrumented"] = sh(["npm", "run", "build"], cwd = str(fe),
                                               timeout = 3600)
            obs["npm_build_instrumented"]["seconds"] = round(time.time() - t0, 1)
            obs["npm_build_instrumented"]["stdout"] = \
                (obs["npm_build_instrumented"].get("stdout") or "")[-2000:]
            di = dist / "index.html"
            obs["instrument"]["survived_vite"] = (
                di.is_file() and "av-boot.js" in di.read_text()
                and (dist / "av-boot.js").is_file() and (dist / "av-scene.js").is_file())
            t0 = time.time()
            obs["cargo_build_instrumented"] = sh(cmd, cwd = str(st), timeout = 7200,
                                                 env = build_env)
            obs["cargo_build_instrumented"]["seconds"] = round(time.time() - t0, 1)
            obs["cargo_build_instrumented"]["stderr"] = \
                (obs["cargo_build_instrumented"].get("stderr") or "")[-4000:]
            if args.stage_dir and binary.is_file():
                shutil.copy2(binary, Path(args.stage_dir) / "desktop-instrumented")
        obs["binary"]["pristine_staged"] = str(pristine) if pristine else None

        # ── 5. a display, then LAUNCH ──
        obs["inventory"] = inventory()
        if not obs["inventory"].get("Xvfb"):
            obs["fetch_xvfb"] = fetch_xvfb(work)
        # Not webkit_paint_probe.start_xserver: it hard-codes :99, and four ephemeral slots
        # share one /tmp/.X11-unix. See the docstring on start_xserver_exclusive.
        xproc, xinfo = start_xserver_exclusive(work, work / "xroot")
        obs["xserver"] = xinfo
        if not xinfo.get("display"):
            obs["fatal"] = "no display server could be started"
            return 0

        home = work / "fixture_home"
        shutil.rmtree(home, ignore_errors = True)
        for sub in (".unsloth/studio", ".config", "xdg", ".local/share"):
            (home / sub).mkdir(parents = True, exist_ok = True)

        # The PRISTINE binary, not the instrumented one. "Does Unsloth Desktop function here"
        # has to be a claim about an unmodified app; the instrumented build exists only to
        # carry the ladder and is launched by the measuring job, which says so.
        launch_binary = pristine if (pristine and pristine.is_file()) else binary
        obs["launch_binary"] = {"path": str(launch_binary),
                                "pristine": bool(pristine and pristine.is_file())}

        logp = work / "launch_stderr.log"
        env = dict(os.environ)
        env.update(build_env)
        # BUILD-time only. The fetched tree also contains runtime .so files, and leaving it on
        # LD_LIBRARY_PATH would run the app against those copies rather than against the host's
        # libwebkit2gtk-4.1 2.52.3 -- the very library the web UI ladder was measured on. Same
        # versions either way, but "the same library" should be a fact and not a coincidence.
        for k in ("LD_LIBRARY_PATH", "LIBRARY_PATH", "RUSTFLAGS", "PKG_CONFIG_PATH"):
            env.pop(k, None)
        env.update({
            "DISPLAY": xinfo["display"],
            "HOME": str(home),
            "XDG_RUNTIME_DIR": str(home / "xdg"),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_DATA_HOME": str(home / ".local" / "share"),
            # Without this GTK's gdk_x_io_error() reports a fatal X error through g_debug()
            # and then _exit(1)s, so the app dies printing nothing at all.
            "G_MESSAGES_DEBUG": "all",
            "RUST_BACKTRACE": "full",
            # THE NEGATIVE CONTROL. This leg is the software one on purpose; the measuring job
            # runs the same binary with this absent.
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "GALLIUM_DRIVER": "llvmpipe",
        })
        # UNSLOTH_STUDIO_HOME is scrubbed by the desktop app from every child it spawns
        # (commands.rs:302, desktop_auth.rs:241, install.rs:583, process.rs:3194 and two more),
        # so HOME is the only handle on its state root. Removing it here makes that explicit
        # rather than leaving an inherited value to be silently ignored.
        for k in ("UNSLOTH_STUDIO_HOME", "STUDIO_HOME"):
            env.pop(k, None)

        obs["renderer_inputs"] = renderer_decision_inputs()
        obs["renderer_inputs"]["DISPLAY"] = xinfo["display"]

        # The EMPTY framebuffer, before anything is launched into it. Without this the
        # post-launch screenshot has no floor: a PNG is "big" only relative to what this X
        # server produces with nothing on it, and a hard byte threshold would be a guess.
        obs["screenshot_empty"] = screenshot(work, xinfo["display"], "empty_before_launch")

        with open(logp, "wb") as fh:
            proc = subprocess.Popen([str(launch_binary)], env = env, cwd = str(work),
                                    stdin = subprocess.DEVNULL, stdout = fh,
                                    stderr = subprocess.STDOUT, start_new_session = True)
        obs["launch"] = {"pid": proc.pid, "binary": str(launch_binary), "log": str(logp)}

        samples = []
        deadline = time.time() + args.launch_seconds
        while time.time() < deadline:
            time.sleep(5)
            alive = proc.poll() is None
            samples.append({"t": round(time.time() - obs["ts_start"], 1), "alive": alive,
                            "rc": proc.returncode})
            if not alive:
                break
        obs["launch"]["samples"] = samples
        obs["launch"]["alive_at_end"] = proc.poll() is None
        obs["launch"]["exit_code"] = proc.returncode

        if proc.poll() is None:
            # THE decisive reading for the rendering path, taken from the live process rather
            # than from a log line that is only written when a workaround is applied.
            obs["launch"]["app_renderer_env"] = read_environ(proc.pid)
            obs["launch"]["process_tree"] = process_tree(proc.pid)
            obs["launch"]["windows"] = sh(["xwininfo", "-root", "-tree"], timeout = 60,
                                          env = {"DISPLAY": xinfo["display"]}).get("stdout", "")[-4000:]
            obs["launch"]["screenshot"] = screenshot(work, xinfo["display"], "launch_software")

        tauri_log = home / ".unsloth" / "studio" / "tauri.log"
        obs["launch"]["tauri_log_path"] = str(tauri_log)
        obs["launch"]["tauri_log"] = tauri_log.read_text(errors = "replace")[-20000:] \
            if tauri_log.is_file() else None
        obs["launch"]["stderr"] = logp.read_text(errors = "replace")[-20000:] \
            if logp.is_file() else None

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(3)
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass

        # ── stage the artifact the measuring job will download ──
        if args.stage_dir:
            stage = Path(args.stage_dir)
            stage.mkdir(parents = True, exist_ok = True)
            # Both binaries were staged as they were produced; only the manifest is written
            # here, and it names WHICH is which so the measuring job cannot mix them up.
            (stage / "MANIFEST.json").write_text(json.dumps({
                "commit": obs.get("commit"), "profile": args.profile,
                "cargo_binary_name": binary.name,
                "pristine": "desktop-pristine",
                "instrumented": "desktop-instrumented",
                "control_port": AMDV_CONTROL_PORT,
                "instrument": obs.get("instrument"),
                "bytes": obs["binary"]["bytes"],
                "built_at": time.time(),
                "devroot_note": "built against the rootlessly fetched -dev closure; the "
                                "RUNTIME libraries come from the host and are the same "
                                "2.52.3 WebKitGTK the web UI ladder ran on",
            }, indent = 2))
            # The devroot itself, because the binary links against libraries that live in it.
            obs["stage"] = {"dir": str(stage),
                            "files": sorted(p.name for p in stage.glob("*"))}
        return 0
    finally:
        if xproc is not None and xproc.poll() is None:
            # By PID. `pkill -f Xvfb` matches this probe's own command line, and a leaked X
            # server outlives the job on a machine three other slots share.
            try:
                os.kill(xproc.pid, signal.SIGTERM)
                time.sleep(2)
                if xproc.poll() is None:
                    os.kill(xproc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        obs["ts_end"] = time.time()
        args.out.write_text(json.dumps(obs, indent = 2))


if __name__ == "__main__":
    sys.exit(main())

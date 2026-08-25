#!/usr/bin/env python3
"""Probe: run the studiobench ladder against real Studio, in real WebKitGTK, on this host.

Observes only. criteria/studio_ladder_collapse.py judges.

The question it exists to settle: users report Unsloth Studio going from 60 fps to 5 fps as a
thread grows from 0K to 100K to 500K tokens. Our only browser venue until now was headless
Chromium on a server, where all four rungs, both arms and the null sat at 60.0 fps with a 1-2%
busy main thread, so every metric there was VOID against its own measured floor. This host has
been shown to composite WebKitGTK on the real gfx1151, which is the engine and the render path
Studio actually uses on Linux. So the ladder can be run where the symptom is claimed to live.

What it does, in order:
  1. fetch Xvfb rootless (apt-get download + dpkg-deb -x under $RUNNER_TEMP), because GTK needs
     a display server and this host has none and no root;
  2. install Studio from the upstream repo with install.sh --local, which is also what builds
     the production frontend bundle. A dev bundle would inflate the axis under investigation by
     about 3.2x, so a dev server is not an acceptable fallback and its absence is recorded;
  3. seed a thread to each rung with studiobench's own Seeder and frozen corpus;
  4. drive the seeded thread in libwebkit2gtk-4.1 through PyGObject and record, per phase, the
     presented-frame series from GdkFrameClock::after-paint, the rAF series, and a calibrated
     busy percentage;
  5. repeat every rung, because the control for "the rung caused this" is the same rung again.

It reports what happened and decides nothing. In particular it does not decide whether 60 fps
at every rung means the symptom is absent or the venue is unloaded: the DOM census and the busy
percentage are recorded so the criteria can tell those apart.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LADDER = HERE / "studio_ladder"

sys.path.insert(0, str(HERE))
from webkit_paint_probe import fetch_xvfb, inventory, start_xserver  # noqa: E402


def sh(cmd, cwd=None, timeout=600, env=None):
    try:
        e = dict(os.environ)
        if env:
            e.update(env)
        r = subprocess.run(cmd, cwd = cwd, capture_output = True, text = True,
                           timeout = timeout, env = e)
        return {"rc": r.returncode, "stdout": r.stdout[-20000:], "stderr": r.stderr[-8000:]}
    except Exception as ex:  # noqa: BLE001
        return {"error": f"{type(ex).__name__}: {ex}"}


def gi_python() -> tuple[str | None, dict]:
    """A python that can import gi. The venv Studio installs almost certainly cannot."""
    tried = {}
    for cand in ("/usr/bin/python3", "/usr/bin/python3.12", "/usr/bin/python3.13",
                 shutil.which("python3") or ""):
        if not cand or not os.path.exists(cand):
            continue
        r = sh([cand, "-c",
                "import gi; gi.require_version('Gtk','3.0'); gi.require_version('WebKit2','4.1');"
                "from gi.repository import Gtk, WebKit2;"
                "print('%d.%d.%d' % (WebKit2.get_major_version(), WebKit2.get_minor_version(),"
                "WebKit2.get_micro_version()))"], timeout = 120)
        tried[cand] = {"rc": r.get("rc"), "out": (r.get("stdout") or "").strip()[:60],
                       "err": (r.get("stderr") or "").strip()[-200:]}
        if r.get("rc") == 0:
            return cand, tried
    return None, tried


def clone(repo_url: str, ref: str, dest: Path) -> dict:
    out = {"url": repo_url, "ref": ref, "dest": str(dest)}
    if dest.exists():
        shutil.rmtree(dest, ignore_errors = True)
    out["clone"] = sh(["git", "clone", "--depth", "50", repo_url, str(dest)], timeout = 1800)
    if ref and ref != "main":
        out["checkout"] = sh(["git", "checkout", ref], cwd = str(dest), timeout = 300)
    r = sh(["git", "rev-parse", "HEAD"], cwd = str(dest), timeout = 60)
    out["commit"] = (r.get("stdout") or "").strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--state", default = "host")
    ap.add_argument("--checkout", default = "")
    ap.add_argument("--work", default = os.environ.get("AMD_CI_WORK", "/tmp/studio_ladder"))
    ap.add_argument("--repo", default = "https://github.com/unslothai/unsloth")
    ap.add_argument("--ref", default = "main")
    ap.add_argument("--rungs", default = "0K,100K,500K")
    ap.add_argument("--reps", type = int, default = 2)
    ap.add_argument("--first-port", type = int, default = 5451)
    ap.add_argument("--install-timeout", type = int, default = 3600)
    ap.add_argument("--rung-timeout", type = int, default = 1800)
    args = ap.parse_args()
    args.out.parent.mkdir(parents = True, exist_ok = True)

    work = Path(args.work) / "ladder"
    work.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "rungs_requested": args.rungs, "reps": args.reps}

    # ── 1. a display ──
    obs["inventory"] = inventory()
    if not obs["inventory"]["Xvfb"]:
        obs["fetch_xvfb"] = fetch_xvfb(work)
    xproc, xinfo = start_xserver(work, obs)
    obs["xserver"] = xinfo
    if not xinfo.get("display"):
        obs["fatal"] = "no display server could be started"
        args.out.write_text(json.dumps(obs, indent = 2))
        return 0

    py_gi, tried = gi_python()
    obs["gi_python"] = {"chosen": py_gi, "tried": tried}

    try:
        if py_gi is None:
            obs["fatal"] = "no python on this host can import gi + WebKit2 4.1"
            return 0

        # ── 2. Studio, installed from the upstream repo ──
        repo = work / "repo"
        obs["clone"] = clone(args.repo, args.ref, repo)
        home = work / "studio_home"
        home.mkdir(parents = True, exist_ok = True)
        t0 = time.time()
        obs["install"] = sh(["bash", "install.sh", "--local"], cwd = str(repo),
                            timeout = args.install_timeout,
                            env = {"UNSLOTH_STUDIO_HOME": str(home)})
        obs["install"]["seconds"] = round(time.time() - t0, 1)
        # Trim: install.sh is chatty and the tail is what says whether it worked.
        obs["install"]["stdout"] = (obs["install"].get("stdout") or "")[-6000:]

        unsloth_bin = None
        for c in [home / "bin" / "unsloth", *sorted(home.glob(".venv*/bin/unsloth"))]:
            if c.exists() and os.access(c, os.X_OK):
                unsloth_bin = str(c)
                break
        obs["unsloth_bin"] = unsloth_bin
        dist = repo / "studio" / "frontend" / "dist"
        obs["dist"] = {"path": str(dist), "exists": dist.is_dir(),
                       "index_html": (dist / "index.html").is_file(),
                       "asset_files": len(list((dist / "assets").rglob("*")))
                       if (dist / "assets").is_dir() else 0}
        if unsloth_bin is None or not obs["dist"]["exists"]:
            obs["fatal"] = ("Studio did not install: "
                            f"bin={unsloth_bin} dist_exists={obs['dist']['exists']}")
            return 0

        # ── 3-5. the ladder ──
        obs["runs"] = []
        port = args.first_port
        rungs = [r.strip() for r in args.rungs.split(",") if r.strip()]
        for rep in range(1, args.reps + 1):
            for rung in rungs:
                rhome = work / f"home_{rung}_r{rep}"
                # Its own home per run: two runs sharing one home share a studio.db, and the
                # second would mount the first's threads.
                if rhome.exists():
                    shutil.rmtree(rhome, ignore_errors = True)
                rhome.mkdir(parents = True, exist_ok = True)
                for name in ("assets", "bin", "cache", "compiled_cache", "llama.cpp", "share",
                             "unsloth_studio", "whisper.cpp"):
                    src = home / name
                    if src.exists() and not (rhome / name).exists():
                        os.symlink(src, rhome / name)
                for name in ("exports", "outputs", "logs", "runs", "rag", "auth"):
                    (rhome / name).mkdir(parents = True, exist_ok = True)

                outp = work / "out" / f"{rung}_rep{rep}.json"
                outp.parent.mkdir(parents = True, exist_ok = True)
                cmd = [sys.executable, str(LADDER / "amdv_rung_bench.py"),
                       "--rung", rung, "--rep", str(rep),
                       "--dist", str(dist), "--home", str(rhome),
                       "--port", str(port), "--display", xinfo["display"],
                       "--sb-root", str(repo), "--unsloth-bin", unsloth_bin,
                       "--python-gi", py_gi,
                       "--scene", str(LADDER / "amdv_scene.js"),
                       "--driver", str(LADDER / "wkgtk_drive.py"),
                       "--out", str(outp)]
                t0 = time.time()
                r = sh(cmd, timeout = args.rung_timeout,
                       env = {"UNSLOTH_WORKSPACE": str(work)})
                entry = {"rung": rung, "rep": rep, "port": port,
                         "seconds": round(time.time() - t0, 1),
                         "rc": r.get("rc"), "error": r.get("error"),
                         "stdout_tail": "\n".join((r.get("stdout") or "").splitlines()[-40:]),
                         "stderr_tail": (r.get("stderr") or "")[-3000:]}
                if outp.is_file():
                    try:
                        entry["payload"] = json.loads(outp.read_text())
                    except Exception as e:  # noqa: BLE001
                        entry["payload_error"] = f"{type(e).__name__}: {e}"
                obs["runs"].append(entry)
                port += 1
                time.sleep(10)
        return 0
    finally:
        if xproc is not None and xproc.poll() is None:
            # By PID. `pkill -f Xvfb` would match this probe's own command line, and an X server
            # left running would outlive the job.
            try:
                os.kill(xproc.pid, signal.SIGTERM)
                time.sleep(2)
                if xproc.poll() is None:
                    os.kill(xproc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
            obs["xserver"]["stopped_pid"] = xproc.pid
        args.out.write_text(json.dumps(obs, indent = 2))


if __name__ == "__main__":
    sys.exit(main())

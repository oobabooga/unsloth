#!/usr/bin/env python3
"""Probe: run the studiobench ladder inside UNSLOTH DESKTOP, the Tauri app, on this host.

Observes only. criteria/desktop_ladder_collapse.py judges.

The question. The Studio WEB UI has already been driven across the whole ladder on this runner,
in real WebKitGTK 2.52.3. Desktop is the next suspect and there is a concrete mechanism for it
being different: studio/src-tauri/src/linux_webkit.rs decides at startup whether to apply a
rendering workaround, and one option is WEBKIT_DISABLE_DMABUF_RENDERER, which turns accelerated
compositing OFF. Its own comment concedes the trigger is "module presence, not the GPU that will
render" (linux_webkit.rs:170), so it can fire on a host that does not need it. If Desktop ends
up software-composited where the web UI is GPU-composited, that is a very plausible source of a
user-visible collapse the web UI does not show.

What it does, in order:
  1. claim an X display THIS job owns (not the shared :99, see desktop_lib);
  2. install Studio from the repo with `install.sh --local`, into a studio root that IS
     `$HOME/.unsloth/studio` for the fixture home, because the Tauri app hardcodes that path and
     scrubs UNSLOTH_STUDIO_HOME from every child it spawns;
  3. per run: a private HOME, a real Studio backend at maximum verbosity on a port the app
     scans, the rung seeded with studiobench's own Seeder and frozen corpus, desktop auth
     provisioned and VERIFIED against the running server, then the Tauri app launched into it;
  4. the same scene as the web UI ladder, compiled into the bundle, reporting over a loopback
     control channel the app's own CSP permits;
  5. amdgpu per-process fdinfo sampled early and late and differenced per drm-client-id, so
     what actually rendered is read from the kernel driver rather than from an in-page string
     WebKitGTK is known to hardcode as `Apple GPU` on Linux/AMD;
  6. two controls, both run LAST so a failure in either cannot cost the ladder: a JAMMED arm
     (200 ms of main-thread spin every 250 ms) that a usable frame channel must fall on, and a
     SOFTWARE arm (LIBGL_ALWAYS_SOFTWARE=1) whose amdgpu counters must collapse.

It reports what happened and decides nothing.
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
LADDER = HERE / "desktop_ladder"
sys.path.insert(0, str(HERE))
from desktop_lib import sh, start_xserver_exclusive  # noqa: E402
from webkit_paint_probe import fetch_xvfb, inventory  # noqa: E402

# Fixed, because the page is served from the binary under a CSP with no `unsafe-eval` and there
# is no way to tell an already-built bundle where to call. Kept in step with
# desktop_build_probe.AMDV_CONTROL_PORT, and asserted against the build manifest rather than
# assumed.
DEFAULT_CONTROL_PORT = 5473


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--state", default = "host")
    ap.add_argument("--checkout", default = "")
    ap.add_argument("--work", default = os.environ.get("AMD_CI_WORK", "/tmp/desktop_ladder"))
    ap.add_argument("--repo", default = "https://github.com/unslothai/unsloth")
    ap.add_argument("--ref", default = "main")
    ap.add_argument("--binary", default = "", help = "the INSTRUMENTED binary from the build job")
    ap.add_argument("--pristine", default = "", help = "the unmodified binary, launched once")
    ap.add_argument("--manifest", default = "")
    ap.add_argument("--rungs", default = "0K,100K,500K")
    ap.add_argument("--reps", type = int, default = 2)
    ap.add_argument("--first-port", type = int, default = 8888)
    ap.add_argument("--control-port", type = int, default = DEFAULT_CONTROL_PORT)
    ap.add_argument("--install-timeout", type = int, default = 5400)
    ap.add_argument("--rung-timeout", type = int, default = 2400)
    ap.add_argument("--hog-ms", type = int, default = 200)
    ap.add_argument("--hog-period-ms", type = int, default = 250)
    ap.add_argument("--control-rung", default = "0K")
    ap.add_argument("--ablate-rung", default = "500K",
                    help = "the rung the fence-deferral ablation is run at, in BOTH arms")
    ap.add_argument("--ablate-reps", type = int, default = 2)
    args = ap.parse_args()
    args.out.parent.mkdir(parents = True, exist_ok = True)

    work = Path(args.work) / "desktop_ladder"
    work.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "rungs_requested": args.rungs, "reps": args.reps,
                 "ts_start": time.time()}
    xproc = None

    try:
        if args.manifest and Path(args.manifest).is_file():
            obs["build_manifest"] = json.loads(Path(args.manifest).read_text())
            got = obs["build_manifest"].get("control_port")
            obs["control_port_matches_build"] = (got == args.control_port)
            if got is not None and got != args.control_port:
                # The port is baked into the bundle. A mismatch means the page would call a
                # port nothing is listening on, the config would never arrive, and every run
                # would time out looking exactly like "Desktop cannot mount a thread".
                obs["fatal"] = (f"the built bundle calls port {got}, this probe serves "
                                f"{args.control_port}")
                return 0

        binary = Path(args.binary) if args.binary else None
        if binary is None or not binary.is_file():
            obs["fatal"] = f"no instrumented binary at {args.binary!r}"
            return 0
        os.chmod(binary, 0o755)
        if args.pristine and Path(args.pristine).is_file():
            os.chmod(args.pristine, 0o755)

        # ── 1. a display this job owns ──
        obs["inventory"] = inventory()
        if not obs["inventory"].get("Xvfb"):
            obs["fetch_xvfb"] = fetch_xvfb(work)
        xproc, xinfo = start_xserver_exclusive(work, work / "xroot")
        obs["xserver"] = xinfo
        if not xinfo.get("display"):
            obs["fatal"] = "no display server could be claimed"
            return 0

        # ── 2. the repo, and Studio installed into the fixture home ──
        repo = work / "repo"
        if repo.exists():
            shutil.rmtree(repo, ignore_errors = True)
        obs["clone"] = sh(["git", "clone", "--depth", "50", args.repo, str(repo)], timeout = 2400)
        if args.ref and args.ref != "main":
            obs["checkout"] = sh(["git", "checkout", args.ref], cwd = str(repo), timeout = 600)
        obs["commit"] = sh(["git", "rev-parse", "HEAD"], cwd = str(repo)).get("stdout", "").strip()

        base_home = work / "base_home"
        studio_root = base_home / ".unsloth" / "studio"
        studio_root.mkdir(parents = True, exist_ok = True)
        ienv = dict(os.environ)
        ienv["HOME"] = str(base_home)
        ienv["UNSLOTH_STUDIO_HOME"] = str(studio_root)
        ienv.pop("STUDIO_HOME", None)
        t0 = time.time()
        # NOT `--tauri`: that mode rejects a non-legacy UNSLOTH_STUDIO_HOME outright and skips
        # create_studio_shortcuts, which is what mints share/studio_install_id -- without which
        # the backend reports an empty studio_root_id and the app refuses to attach.
        obs["install"] = sh(["bash", "install.sh", "--local"], cwd = str(repo),
                            timeout = args.install_timeout, env = ienv)
        obs["install"]["seconds"] = round(time.time() - t0, 1)
        obs["install"]["stdout"] = (obs["install"].get("stdout") or "")[-6000:]
        cli = studio_root / "unsloth_studio" / "bin" / "unsloth"
        obs["install_layout"] = {
            "cli": str(cli), "cli_exists": cli.exists(),
            "bin_symlink": (studio_root / "bin" / "unsloth").exists(),
            "studio_install_id": (studio_root / "share" / "studio_install_id").is_file(),
        }
        if not cli.exists() or not obs["install_layout"]["studio_install_id"]:
            obs["fatal"] = ("Studio did not install into the layout the Tauri app requires: "
                            f"{obs['install_layout']}")
            return 0

        # ── 3-6. the runs ──
        rungs = [r.strip() for r in args.rungs.split(",") if r.strip()]
        plan = [{"rung": r, "rep": str(rep), "hog": 0, "software": False}
                for rep in range(1, args.reps + 1) for r in rungs]
        # Both controls LAST and on the smallest rung, so a failure in either cannot cost the
        # ladder and neither can be mistaken for a ladder reading.
        if args.hog_ms:
            plan.append({"rung": args.control_rung, "rep": "hog", "hog": args.hog_ms,
                         "software": False})
        plan.append({"rung": args.control_rung, "rep": "sw", "hog": 0, "software": True})
        # The UNMODIFIED binary, launched into the same real backend with a real seeded thread.
        # It cannot open a specific thread -- that needs the injected script -- so it is
        # OBSERVED rather than driven: a mapped window, a painted framebuffer, and the backend's
        # own access log showing the app asked it for the thread list. This is what makes
        # "Unsloth Desktop functions on this runner" a claim about the shipped app rather than
        # about a build with our scene compiled into it.
        if args.pristine and Path(args.pristine).is_file():
            plan.append({"rung": "100K", "rep": "pristine", "hog": 0, "software": False,
                         "no_scene": True, "binary": args.pristine})

        # THE ABLATION, at the top rung, in the arm the ladder does not already cover.
        #
        # The mechanism has been attributed in both engines: the one-way upgrade of deferred code
        # fences, ~335 fences x two commits per latch, each forcing a style recalc over the whole
        # document. Chromium counters at r500K read RecalcStyleCount 718 vs 37 and
        # RecalcStyleDuration 7.53 s vs 0.47 s with the flag on versus off, and it replicates on
        # WebKitGTK 2.50.4. It is EXCESS and not BURST: both arms end with an identical DOM, so
        # deferred delivery buys nothing and costs the extra traversals.
        #
        # The ladder rungs run in the DEFAULT arm, which is `defer` (SHIP_DEFAULT), so only the
        # `off` arm has to be added. If Desktop shows the same ablation response, the mechanism
        # is shell-independent and Desktop is the same defect in a different wrapper. If it does
        # not, that is a Desktop-specific finding and a much more interesting one.
        for rep in range(1, args.ablate_reps + 1):
            plan.append({"rung": args.ablate_rung, "rep": f"abl{rep}", "hog": 0,
                         "software": False, "defer_fence": "off"})

        obs["plan"] = plan
        obs["runs"] = []
        port = args.first_port
        for item in plan:
            tag = f"{item['rung']}_rep{item['rep']}"
            outp = work / "out" / f"{tag}.json"
            outp.parent.mkdir(parents = True, exist_ok = True)
            cmd = [sys.executable, str(LADDER / "amdv_desktop_bench.py"),
                   "--rung", item["rung"], "--rep", item["rep"],
                   "--binary", str(item.get("binary") or binary),
                   "--base-home", str(base_home),
                   "--run-home", str(work / f"home_{tag}"),
                   "--sb-root", str(repo),
                   "--port", str(port), "--control-port", str(args.control_port),
                   "--display", xinfo["display"],
                   "--hog-ms", str(item["hog"]), "--hog-period-ms", str(args.hog_period_ms),
                   "--out", str(outp)]
            if item["hog"]:
                # The jammed arm does not need a reply, and starving the model-chip restore
                # under the jam is what made this control fail and cost a whole run its verdict
                # in the web UI ladder.
                cmd.append("--skip-send")
            if item["software"]:
                cmd.append("--software")
            if item.get("no_scene"):
                cmd.append("--no-scene")
            if item.get("defer_fence"):
                cmd += ["--defer-fence", item["defer_fence"]]
            t0 = time.time()
            r = sh(cmd, timeout = args.rung_timeout)
            entry = {**item, "port": port, "seconds": round(time.time() - t0, 1),
                     "rc": r.get("rc"), "error": r.get("error"),
                     "stdout_tail": "\n".join((r.get("stdout") or "").splitlines()[-40:]),
                     "stderr_tail": (r.get("stderr") or "")[-4000:]}
            if outp.is_file():
                try:
                    entry["payload"] = json.loads(outp.read_text())
                except Exception as e:  # noqa: BLE001
                    entry["payload_error"] = f"{type(e).__name__}: {e}"
            obs["runs"].append(entry)
            port += 1
            if port > 8908:
                port = args.first_port
            time.sleep(10)

        # Every log, into the artifact. A Desktop that boots and shows a half-rendered page
        # produces clean-looking numbers, so the evidence that it worked has to be visible.
        logs = work / "out" / "logs"
        logs.mkdir(parents = True, exist_ok = True)
        collected = []
        for pat in ("*.log", "*.jsonl", "*.png"):
            for f in list((work / "out").rglob(pat)):
                try:
                    if f.parent == logs:
                        continue
                    dest = logs / (f.parent.name + "__" + f.name)
                    shutil.copy2(f, dest)
                    collected.append({"src": str(f), "bytes": dest.stat().st_size})
                except Exception as e:  # noqa: BLE001
                    collected.append({"src": str(f), "error": f"{type(e).__name__}: {e}"})
        obs["logs_collected"] = collected
        return 0
    finally:
        if xproc is not None and xproc.poll() is None:
            try:
                os.kill(xproc.pid, signal.SIGTERM)
                time.sleep(2)
                if xproc.poll() is None:
                    os.kill(xproc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
            obs.setdefault("xserver", {})["stopped_pid"] = xproc.pid
        obs["ts_end"] = time.time()
        args.out.write_text(json.dumps(obs, indent = 2))


if __name__ == "__main__":
    sys.exit(main())

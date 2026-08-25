#!/usr/bin/env python3
"""Probe: what does `content-visibility: auto` on KaTeX actually buy on a real GPU, and does it
cost anything at short context?

Observes only. criteria/studio_cvkatex.py judges.

Run 32869180652 measured the collapse on this venue at 287.6 ms of blocked main thread per ONE
PIXEL scroll event, which is 3.2 fps, and measured `.katex{visibility:hidden}` at 10.0 ms and
61.0 fps, +97%. That arm is a MECHANISM PROBE, not a fix: it works by making the maths invisible,
so its 97% is an UPPER BOUND on what a real fix could recover, not the fix's own number.

This probe measures the real fix in the same instrument, unchanged: same one-pixel gesture, same
fixed park pixel, same drained deferred-fence reservoir, same discarded warm-up run to
quiescence, same interleaved baselines, same thresholds. What is added is three arms and two more
rungs.

THE SHORT RUNGS ARE NOT OPTIONAL. The standing rule is that performance must be good on short AND
long contexts, and the mechanisms most likely to fail that are exactly the ones that add
bookkeeping which buys nothing on a small document. 0K is an empty thread with no scrollable
range and no maths at all, so the scene reports `no_scroll_range` and scores no gesture window
there; what that rung establishes is that the selector matches nothing, measured rather than
argued.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LADDER = HERE / "studio_ladder"
sys.path.insert(0, str(HERE))
from webkit_paint_probe import fetch_xvfb, inventory, start_xserver  # noqa: E402
from studio_ladder_probe import clone, gi_python, sh  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--state", default = "host")
    ap.add_argument("--checkout", default = "")
    ap.add_argument("--work", default = os.environ.get("AMD_CI_WORK", "/tmp/studio_layers"))
    ap.add_argument("--repo", default = "https://github.com/unslothai/unsloth")
    ap.add_argument("--ref", default = "main")
    ap.add_argument("--rungs", default = "500K",
                    help = "comma separated, measured in the order given. The short rungs are "
                           "not optional: a change that wins at 500K and costs anything at 0K is "
                           "not a win, and 0K has no scrollable range at all, which the scene "
                           "reports as `no_scroll_range` rather than as a failure.")
    ap.add_argument("--reps", type = int, default = 2)
    ap.add_argument("--first-port", type = int, default = 5481)
    ap.add_argument("--install-timeout", type = int, default = 3600)
    ap.add_argument("--rung-timeout", type = int, default = 2400)
    args = ap.parse_args()
    args.out.parent.mkdir(parents = True, exist_ok = True)

    rungs = [r.strip() for r in args.rungs.split(",") if r.strip()]
    work = Path(args.work) / "cvkatex"
    work.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "rungs": rungs, "reps": args.reps}

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

        repo = work / "repo"
        obs["clone"] = clone(args.repo, args.ref, repo)
        home = work / "studio_home"
        home.mkdir(parents = True, exist_ok = True)
        t0 = time.time()
        obs["install"] = sh(["bash", "install.sh", "--local"], cwd = str(repo),
                            timeout = args.install_timeout,
                            env = {"UNSLOTH_STUDIO_HOME": str(home)})
        obs["install"]["seconds"] = round(time.time() - t0, 1)
        obs["install"]["stdout"] = (obs["install"].get("stdout") or "")[-4000:]

        unsloth_bin = None
        for c in [home / "bin" / "unsloth", *sorted(home.glob(".venv*/bin/unsloth"))]:
            if c.exists() and os.access(c, os.X_OK):
                unsloth_bin = str(c)
                break
        dist = repo / "studio" / "frontend" / "dist"
        obs["unsloth_bin"] = unsloth_bin
        obs["dist"] = {"path": str(dist), "exists": dist.is_dir(),
                       "index_html": (dist / "index.html").is_file(),
                       "asset_files": len(list((dist / "assets").rglob("*")))
                       if (dist / "assets").is_dir() else 0}
        if unsloth_bin is None or not obs["dist"]["exists"]:
            obs["fatal"] = f"Studio did not install: bin={unsloth_bin} dist={obs['dist']}"
            return 0

        obs["runs"] = []
        port = args.first_port
        # RUNG OUTER, REPETITION INNER. Both repetitions of a rung run back to back on the same
        # host state, so a per-rep disagreement inside a rung is about the rung and not about
        # something that drifted between two distant points of a long job.
        for rung, rep in [(u, r) for u in rungs for r in range(1, args.reps + 1)]:
            rhome = work / f"home_{rung}_r{rep}"
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

            outp = work / "out" / f"cvkatex_{rung}_rep{rep}.json"
            outp.parent.mkdir(parents = True, exist_ok = True)
            cmd = [sys.executable, str(LADDER / "amdv_rung_bench.py"),
                   "--rung", rung, "--rep", str(rep),
                   "--dist", str(dist), "--home", str(rhome),
                   "--port", str(port), "--display", xinfo["display"],
                   "--sb-root", str(repo), "--unsloth-bin", unsloth_bin,
                   "--python-gi", py_gi,
                   "--scene", str(LADDER / "cvk_layers.js"),
                   "--entry", "window.__cvk.run",
                   "--idle-ms", "8000",
                   "--driver", str(LADDER / "amdv_drive.py"),
                   "--frame-clock", "updating",
                   "--skip-send",
                   "--out", str(outp)]
            t0 = time.time()
            r = sh(cmd, timeout = args.rung_timeout, env = {"UNSLOTH_WORKSPACE": str(work)})
            entry = {"rung": rung, "rep": rep, "port": port,
                     "seconds": round(time.time() - t0, 1), "rc": r.get("rc"),
                     "error": r.get("error"),
                     "stdout_tail": "\n".join((r.get("stdout") or "").splitlines()[-30:]),
                     "stderr_tail": (r.get("stderr") or "")[-2500:]}
            if outp.is_file():
                try:
                    entry["payload"] = json.loads(outp.read_text())
                except Exception as e:  # noqa: BLE001
                    entry["payload_error"] = f"{type(e).__name__}: {e}"
            obs["runs"].append(entry)
            port += 1
            time.sleep(10)

        logs = work / "out" / "logs"
        logs.mkdir(parents = True, exist_ok = True)
        collected = []
        for pat in ("*.log", "*.jsonl"):
            for f in list((work / "out").glob(pat)) + list(work.rglob("logs/" + pat)):
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
            obs["xserver"]["stopped_pid"] = xproc.pid
        args.out.write_text(json.dumps(obs, indent = 2))


if __name__ == "__main__":
    sys.exit(main())

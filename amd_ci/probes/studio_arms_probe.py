#!/usr/bin/env python3
"""Probe: two SHIPPED builds of Unsloth Studio, up the ladder, in real WebKitGTK, on a real GPU.

Observes only. criteria/studio_arms_ladder.py judges.

THE QUESTION, which is the user's original one and has never been answered as asked:

    "60 FPS downgrading to 5 FPS when the context length grows from 0K to 100K to 500K."

Every figure this campaign has published was measured on a BRANCH with a flag forced on. The
flags now ship on by default, so the thing to measure is no longer a flag: it is what a user gets
from `main` today against what a user got before any of this started.

  pre   the last commit before the campaign began. The "60 fps" side of the complaint.
  head  today's main, installed as it ships, WITH NOTHING FORCED.

NOTHING IS OVERRIDDEN ON EITHER ARM. No VITE flag, no runtime global, no injected stylesheet. The
head arm's defaults ARE the treatment, and forcing them would measure a configuration no user
runs. The scene instead reads each arm's state back out of the running page, per rung, and the
criteria module refuses to score a head session that did not actually boot into the fixed state.

WHAT IS HELD CONSTANT, and why each one would otherwise turn this into a two-variable experiment:

  * unsloth-zoo. Both arms take it from ITS main, which is what `install.sh --local` already
    does ("overlaying unsloth-zoo from git main"), and the resolved commit is read back out of
    each venv and compared. Two arms on two zoos is not a measurement of either.
  * the INSTRUMENT. studiobench does not exist in the `pre` tree at all, so the pacer, seeder,
    corpus and scene all come from ONE pinned checkout (the head clone) for both arms. An arm
    measured with the harness that shipped alongside it differs from the other arm by the change
    AND by the measuring device.
  * the corpus, which follows from the pinned checkout, and is recorded per run so a mismatch
    cannot hide.
  * the display, the X server and the machine.

WHAT IS DELIBERATELY NOT SHARED: the port and the Studio home. Assets cache on ORIGIN plus PATH,
and vite content-hashes on SOURCE change, so a rebuilt bundle can keep its URL and be served from
the previous arm's bytes. Every session therefore gets a FRESH PORT, which is a fresh origin, and
its own home, because two runs sharing a home share a studio.db and the second mounts the first's
threads.

SEQUENTIAL, ALWAYS. Two WebKitGTK toplevels on one X server occlude each other and the lower one
stops being asked to paint, which reads as a frame rate.

Order: for each repetition, for each rung, both arms back to back, with the ARM ORDER SWAPPED on
alternate repetitions. Adjacency keeps machine drift out of the arm difference; the swap keeps
"whichever arm went first" out of it.
"""

from __future__ import annotations

import argparse
import hashlib
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

ZOO_URL = "https://github.com/unslothai/unsloth-zoo"


def _ok(run: dict) -> bool:
    p = run.get("payload")
    return bool(isinstance(p, dict) and p.get("ok"))


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
    """A FULL clone, then an explicit checkout of the ref.

    Not `--depth`: the two refs are ten days apart and the pre arm is not reachable from a
    shallow fetch of main's tip. A clone that silently lands on the wrong commit is the one
    failure this whole run cannot survive, so the resolved SHA is read back and reported.
    """
    out = {"url": repo_url, "ref": ref, "dest": str(dest)}
    if dest.exists():
        shutil.rmtree(dest, ignore_errors = True)
    out["clone"] = sh(["git", "clone", repo_url, str(dest)], timeout = 3600)
    out["checkout"] = sh(["git", "checkout", "--detach", ref], cwd = str(dest), timeout = 600)
    r = sh(["git", "rev-parse", "HEAD"], cwd = str(dest), timeout = 60)
    out["commit"] = (r.get("stdout") or "").strip()
    r = sh(["git", "log", "-1", "--format=%H %ci %s"], cwd = str(dest), timeout = 60)
    out["commit_line"] = (r.get("stdout") or "").strip()
    out["has_studiobench"] = (dest / "tests" / "studio" / "studiobench").is_dir()
    return out


def bundle_hash(dist: Path) -> str:
    h = hashlib.sha256()
    n = 0
    if (dist / "assets").is_dir():
        for f in sorted((dist / "assets").rglob("*")):
            if f.is_file() and f.suffix in (".js", ".css"):
                h.update(f.name.encode())
                h.update(f.read_bytes())
                n += 1
    idx = dist / "index.html"
    if idx.exists():
        h.update(idx.read_bytes())
    return f"{h.hexdigest()[:16]}({n} files)"


def zoo_identity(home: Path) -> dict:
    """Which unsloth-zoo this arm ended up with, read out of the INSTALLED metadata.

    Not from the install log and not from the URL that was asked for: `install.sh` overlays zoo
    from git main, and what matters is the commit that landed. `direct_url.json` carries the
    resolved `commit_id` for a git install, and `METADATA` carries the version for a wheel.
    """
    out: dict = {"home": str(home), "dist_info": None, "commit_id": None, "version": None,
                 "url": None}
    infos = sorted(home.glob(".venv*/lib/python*/site-packages/unsloth_zoo-*.dist-info"))
    infos += sorted(home.glob("**/site-packages/unsloth_zoo-*.dist-info"))
    seen = set()
    for info in infos:
        if str(info) in seen:
            continue
        seen.add(str(info))
        out["dist_info"] = str(info)
        out["version"] = info.name.split("-", 1)[1].replace(".dist-info", "")
        du = info / "direct_url.json"
        if du.is_file():
            try:
                j = json.loads(du.read_text())
                out["url"] = j.get("url")
                out["commit_id"] = (j.get("vcs_info") or {}).get("commit_id")
            except Exception as e:  # noqa: BLE001
                out["direct_url_error"] = f"{type(e).__name__}: {e}"
        break
    return out


def pin_zoo(home: Path, sha: str, timeout: int = 1800) -> dict:
    """Force this arm's venv onto ONE unsloth-zoo commit.

    `install.sh --local` overlays unsloth-zoo from git main, which is the right default and the
    wrong thing to rely on here: the two arms are installed one after the other and the second
    can be an hour behind the first, so a single merge to unsloth-zoo in between would make this
    a two-variable experiment. Worse, the criteria module's zoo gate would then fail a run in
    which nothing was actually wrong.

    So both arms are pinned to the commit read once, before either install. If the pin cannot be
    applied the failure is RECORDED rather than swallowed: the gate compares what actually landed,
    and an unpinned pair that happens to match is still a pass, while an unpinned pair that does
    not is still a failure.
    """
    out: dict = {"sha": sha, "attempted": False}
    if not sha:
        out["skipped"] = "no unsloth-zoo commit was resolved, so there was nothing to pin to"
        return out
    pys = sorted(home.glob(".venv*/bin/python")) + sorted(home.glob("**/venv/bin/python"))
    if not pys:
        out["skipped"] = f"no venv python under {home}"
        return out
    out["attempted"] = True
    out["python"] = str(pys[0])
    out["pip"] = sh([str(pys[0]), "-m", "pip", "install", "--no-deps", "--force-reinstall",
                     f"unsloth-zoo @ git+{ZOO_URL}@{sha}"], timeout = timeout)
    if isinstance(out["pip"], dict):
        out["pip"]["stdout"] = (out["pip"].get("stdout") or "")[-2000:]
    return out


def install_arm(repo: Path, home: Path, timeout: int, extra_args: list[str]) -> dict:
    """`install.sh --local`, which is also what builds the production frontend bundle.

    `extra_args` exists for OFF-RUNNER smoke tests only (`--no-torch` keeps a local rehearsal from
    pulling a multi-gigabyte training stack twice). It is EMPTY on the measured run, and the args
    that were used are recorded in the observations, because "both arms were installed the same
    way" has to be a checkable fact rather than an intention.
    """
    home.mkdir(parents = True, exist_ok = True)
    t0 = time.time()
    r = sh(["bash", "install.sh", "--local", *extra_args], cwd = str(repo), timeout = timeout,
           env = {"UNSLOTH_STUDIO_HOME": str(home)})
    r["args"] = ["--local", *extra_args]
    r["seconds"] = round(time.time() - t0, 1)
    # install.sh is chatty and the tail is what says whether it worked.
    r["stdout"] = (r.get("stdout") or "")[-8000:]
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--state", default = "host")
    ap.add_argument("--checkout", default = "")
    ap.add_argument("--work", default = os.environ.get("AMD_CI_WORK", "/tmp/studio_arms"))
    ap.add_argument("--repo", default = "https://github.com/unslothai/unsloth")
    ap.add_argument("--ref-pre", default = "c87fe20e3",
                    help = "the last commit before the campaign started")
    # A SHA, NOT `main`. `main` moves: a rehearsal of this probe cloned "main" and got a commit
    # eight days newer than the one the run was supposed to be about, and nothing in the output
    # would have said so. A run that cannot name the commit it measured cannot be quoted against
    # a set of merged PRs.
    ap.add_argument("--ref-head", default = "40b4702cd39f3e51b3f5404b1525c3e1c4fc5bd8",
                    help = "the shipped state under test, pinned to a commit; nothing is forced "
                           "on it")
    ap.add_argument("--rungs", default = "0K,100K,500K")
    ap.add_argument("--reps", type = int, default = 2)
    ap.add_argument("--first-port", type = int, default = 5541)
    ap.add_argument("--install-timeout", type = int, default = 5400)
    ap.add_argument("--rung-timeout", type = int, default = 2700)
    # The positive control, opened as a window inside EVERY session by the scene. Without it a
    # flat frame rate cannot be told apart from a frame channel that reads 60 no matter what.
    ap.add_argument("--install-arg", action = "append", default = [],
                    help = "extra argument for install.sh, applied to BOTH arms. For off-runner "
                           "rehearsals only; the measured run passes none")
    ap.add_argument("--hog-ms", type = int, default = 200)
    ap.add_argument("--hog-period-ms", type = int, default = 250)
    args = ap.parse_args()
    args.out.parent.mkdir(parents = True, exist_ok = True)

    work = Path(args.work) / "arms"
    work.mkdir(parents = True, exist_ok = True)

    def flush() -> None:
        """Write the observations so far.

        Called after every session rather than once at the end. The end used to be the only
        write, in a `finally`: if the workflow's step timeout fired, the process tree was killed,
        the `finally` never ran, and hours of exclusive GPU produced a zero-byte result with
        nothing to re-score offline. A partial artifact is a first-class result here.
        """
        try:
            args.out.write_text(json.dumps(obs, indent = 2))
        except Exception:  # noqa: BLE001
            pass
    obs: dict = {"state": args.state, "rungs_requested": args.rungs, "reps": args.reps,
                 "refs": {"pre": args.ref_pre, "head": args.ref_head},
                 "repo": args.repo,
                 "scene": str(LADDER / "final_scene.js"),
                 "bench": str(LADDER / "final_rung_bench.py"),
                 "install_args": ["--local", *args.install_arg],
                 "hog": {"ms": args.hog_ms, "period_ms": args.hog_period_ms}}

    # ── 1. a display ──
    obs["inventory"] = inventory()
    if not obs["inventory"]["Xvfb"]:
        obs["fetch_xvfb"] = fetch_xvfb(work)
    # A display DERIVED FROM THIS RUN'S PORT RANGE rather than the shared default. Four ephemeral
    # slots run on one machine, and a second job that lands on the same display shares an X server
    # with this one; two WebKitGTK toplevels there occlude each other and the lower one stops
    # being asked to paint, which reads as a frame rate.
    xproc, xinfo = start_xserver(work, obs, display = f":{(args.first_port % 90) + 100}")
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

        # ── 2. what zoo's main is RIGHT NOW, before either install resolves it ──
        # Recorded first so that an arm which somehow landed on a different commit is visible as
        # a disagreement with this line rather than as two numbers nobody compared.
        r = sh(["git", "ls-remote", ZOO_URL, "HEAD"], timeout = 300)
        obs["zoo_main_at_start"] = {"url": ZOO_URL,
                                    "sha": (r.get("stdout") or "").split()[0]
                                    if (r.get("stdout") or "").strip() else None,
                                    "rc": r.get("rc"), "err": (r.get("stderr") or "")[-400:]}

        # ── 3. both arms, cloned and installed ──
        obs["arms"] = {}
        for arm, ref in (("pre", args.ref_pre), ("head", args.ref_head)):
            repo = work / f"repo_{arm}"
            home = work / f"studio_home_{arm}"
            entry: dict = {"arm": arm, "ref": ref}
            entry["clone"] = clone(args.repo, ref, repo)
            entry["install"] = install_arm(repo, home, args.install_timeout, args.install_arg)
            unsloth_bin = None
            for c in [home / "bin" / "unsloth", *sorted(home.glob(".venv*/bin/unsloth"))]:
                if c.exists() and os.access(c, os.X_OK):
                    unsloth_bin = str(c)
                    break
            dist = repo / "studio" / "frontend" / "dist"
            entry["unsloth_bin"] = unsloth_bin
            entry["repo"] = str(repo)
            entry["home"] = str(home)
            entry["dist"] = {
                "path": str(dist), "exists": dist.is_dir(),
                "index_html": (dist / "index.html").is_file(),
                "asset_files": len(list((dist / "assets").rglob("*")))
                if (dist / "assets").is_dir() else 0,
                # A dev bundle would inflate the axis under investigation by about 3.2x, so which
                # bundle this is has to be a recorded fact. A production vite build emits
                # content-hashed assets under assets/ and no dev client shim.
                "bundle_hash": bundle_hash(dist) if dist.is_dir() else None,
            }
            entry["zoo_before_pin"] = zoo_identity(home)
            entry["zoo_pin"] = pin_zoo(home, (obs.get("zoo_main_at_start") or {}).get("sha") or "")
            entry["zoo"] = zoo_identity(home)
            # Free space, per arm, after its install. Two non-shallow clones of a repo whose
            # `.git` alone is over half a gigabyte, plus two venvs, on a disk four ephemeral
            # slots share: an out-of-space second arm should be diagnosable from the artifact
            # rather than from a build error nobody can attribute.
            entry["disk_after_install"] = sh(["df", "-h", str(work)], timeout = 60).get("stdout")
            obs["arms"][arm] = entry
            flush()

        missing = [a for a, e in obs["arms"].items()
                   if not e["unsloth_bin"] or not e["dist"]["exists"]]
        if missing:
            obs["fatal"] = ("Studio did not install for " + ", ".join(missing) + ": "
                            + json.dumps({a: {"bin": obs["arms"][a]["unsloth_bin"],
                                              "dist": obs["arms"][a]["dist"]["exists"]}
                                          for a in missing}))
            return 0

        # THE PINNED INSTRUMENT. `pre` has no tests/studio/studiobench at all, so this is not a
        # preference: it is the only tree that can drive either arm, and using it for both is
        # what keeps the measuring device out of the difference.
        sb_root = Path(obs["arms"]["head"]["repo"])
        obs["instrument"] = {
            "sb_root": str(sb_root),
            "from_arm": "head",
            "exists": (sb_root / "tests" / "studio" / "studiobench").is_dir(),
            "pre_has_studiobench": obs["arms"]["pre"]["clone"]["has_studiobench"],
            "why": "pre carries no studiobench, so one pinned tree drives both arms",
        }

        # ── 4. the ladder, both arms, interleaved ──
        obs["runs"] = []
        port = args.first_port
        rungs = [r.strip() for r in args.rungs.split(",") if r.strip()]
        plan: list[tuple[str, str, str]] = []
        for rep in range(1, args.reps + 1):
            # Swap which arm goes first on alternate repetitions, so "went first" cannot be read
            # as an arm effect.
            order = ["pre", "head"] if rep % 2 == 1 else ["head", "pre"]
            for rung in rungs:
                for arm in order:
                    plan.append((rung, str(rep), arm))
        obs["plan"] = [{"rung": r, "rep": p, "arm": a} for r, p, a in plan]

        for rung, rep, arm in plan:
            info = obs["arms"][arm]
            rhome = work / f"home_{arm}_{rung}_r{rep}"
            # Its own home per run: two runs sharing one home share a studio.db, and the second
            # would mount the first's threads.
            if rhome.exists():
                shutil.rmtree(rhome, ignore_errors = True)
            rhome.mkdir(parents = True, exist_ok = True)
            src_home = Path(info["home"])
            for name in ("assets", "bin", "cache", "compiled_cache", "llama.cpp", "share",
                         "unsloth_studio", "whisper.cpp"):
                src = src_home / name
                if src.exists() and not (rhome / name).exists():
                    os.symlink(src, rhome / name)
            for name in ("exports", "outputs", "logs", "runs", "rag", "auth"):
                (rhome / name).mkdir(parents = True, exist_ok = True)

            outp = work / "out" / f"{arm}_{rung}_rep{rep}.json"
            outp.parent.mkdir(parents = True, exist_ok = True)
            cmd = [sys.executable, str(LADDER / "final_rung_bench.py"),
                   "--rung", rung, "--rep", str(rep), "--arm", arm,
                   "--hog-ms", str(args.hog_ms), "--hog-period-ms", str(args.hog_period_ms),
                   "--dist", info["dist"]["path"], "--home", str(rhome),
                   "--port", str(port), "--display", xinfo["display"],
                   "--sb-root", str(sb_root), "--unsloth-bin", info["unsloth_bin"],
                   "--python-gi", py_gi,
                   "--scene", str(LADDER / "final_scene.js"),
                   "--driver", str(LADDER / "amdv_drive.py"),
                   "--frame-clock", "updating",
                   "--out", str(outp)]
            t0 = time.time()
            r = sh(cmd, timeout = args.rung_timeout,
                   env = {"UNSLOTH_WORKSPACE": str(work)})
            entry = {"rung": rung, "rep": rep, "arm": arm, "port": port,
                     "expected_bundle_hash": info["dist"]["bundle_hash"],
                     "seconds": round(time.time() - t0, 1),
                     "rc": r.get("rc"), "error": r.get("error"),
                     "stdout_tail": "\n".join((r.get("stdout") or "").splitlines()[-40:]),
                     "stderr_tail": (r.get("stderr") or "")[-3000:]}
            if outp.is_file():
                try:
                    entry["payload"] = json.loads(outp.read_text())
                except Exception as e:  # noqa: BLE001
                    entry["payload_error"] = f"{type(e).__name__}: {e}"
            # REAP THIS SESSION'S SERVER BEFORE STARTING THE NEXT ONE.
            #
            # `subprocess.run(timeout=...)` kills the direct child only. The bench launches
            # Studio under `setsid`, and its own teardown lives in a `finally` that a SIGKILL
            # never reaches, so a session that overruns `--rung-timeout` leaves a Studio and a
            # WebKitGTK toplevel alive. The next session then measures a page on an X server
            # that another toplevel is occluding, and the lower one stops being asked to paint,
            # which reads as a frame rate. Every cell after that would be silently wrong and
            # nothing downstream could see it.
            #
            # BY PORT, never by process name: a pattern-matching killer matches the command line
            # of the thing running it, and the bracket trick does not fix that.
            reap = sh(["bash", "-c",
                       f"pid=$(ss -lptnH 'sport = :{port}' 2>/dev/null | "
                       f"grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2); "
                       f"if [ -n \"$pid\" ]; then echo \"killing $pid on {port}\"; "
                       f"kill -TERM \"$pid\"; sleep 2; fi"], timeout = 60)
            entry["reaped"] = (reap.get("stdout") or "").strip() or None
            obs["runs"].append(entry)
            flush()
            # A fresh port per session, monotonically. This is not tidiness: assets cache on
            # ORIGIN plus PATH, so reusing a port can serve the previous arm's bytes from a
            # rebuilt bundle that kept its URL.
            port += 1
            time.sleep(10)

            # EARLY BAIL. The plan puts both arms at the smallest rung first precisely so that an
            # arm which cannot be driven at all shows up in the first two sessions. Without this
            # the loop would carry on and spend another hour of exclusive GPU proving the same
            # thing ten more times.
            if len(obs["runs"]) == 2 and not any(_ok(r) for r in obs["runs"]):
                obs["fatal"] = (
                    "neither arm completed a session at the smallest rung, so the remaining "
                    "sessions were not attempted: "
                    + "; ".join(f"{r['arm']}/{r['rung']}: "
                                f"{str((r.get('payload') or {}).get('error'))[:120]}"
                                for r in obs["runs"]))
                flush()
                break

        # EVERY log, into the artifact. A Studio that boots and serves a half-rendered page
        # produces clean-looking numbers, so the evidence that it worked has to be visible.
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
            # By PID. `pkill -f Xvfb` would match this probe's own command line, and an X server
            # left running would outlive the job and hold VRAM on a machine three other slots
            # share.
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

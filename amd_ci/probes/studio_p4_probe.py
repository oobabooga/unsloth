#!/usr/bin/env python3
"""Probe: PR 9477 piece 4, REASONING PAGINATION, in real WebKitGTK on the gfx1151.

Observes only. criteria/studio_p4_pagination.py judges.

WHAT PIECE 4 IS. `reasoning-pagination.ts` mounts only the newest ~8,192 characters of a reasoning
trace and puts the rest behind Show more / Show less. It was split out of 9477 and dropped without
being measured on the window it was aimed at, on the stated grounds that it sits behind a disabled
flag. That reasoning is circular: the flag can be flipped, and this run flips it.

THE ARMS, AND WHY THERE ARE FIVE. Two of them are one-line isolations, which is the whole design:
a pair that differs by a single boolean prices pagination and nothing else, and it does so twice,
at two different points in the branch's history.

    baseT   90f85fdbf                 the tip's merge base: main as it stands
    C       baseT + C_tip.patch       9477 as written today. `REASONING_PAGINATION_ENABLED` is
                                      false, so this is the 793-line no-op case: what merging the
                                      branch as written would actually ship
    B       C + p4_flag_on.patch      the SAME tree, flag flipped true. ONE LINE from C, so
                                      B vs C is pagination and can be nothing else
    A       33d65ce99, CHECKED OUT    the commit the user named, exactly as its author wrote it.
                                      No patch step, because a reconstruction of the arm that
                                      matters most is one more thing that can be wrong. At that
                                      commit there is NO pagination flag: `thread-feature-flags.ts`
                                      carries only GRID_COLLAPSE_REASONING_ENABLED, and
                                      `reasoning.tsx` hardcodes `paginateReasoning={true}`, so
                                      pagination is LIVE
    Aoff    A + A_pagination_off      the same checkout with that one literal flipped to false.
                                      A vs Aoff is the second one-line isolation

A and B are NOT the same code: six of the branch's own commits sit between them, one of which
(`0b3592ad1`, open-fence re-parsing) is itself a perf change, and `GRID_COLLAPSE_REASONING_ENABLED`
is false at A and true at B. So they are carried separately and any difference between them is
attributed rather than averaged into a single "piece 4" number.

WHERE THE COMMITS COME FROM. baseT is on main. 33d65ce99 and its parent 4351502ac are NOT: they
exist only on the pull request, so `refs/pull/9477/head` is fetched explicitly and with depth, and
every arm's ref is resolved in a scratch clone BEFORE any install runs. The previous run of this
probe cloned `--depth 50` off main, could not resolve the commit, and `git checkout` left HEAD on
the default branch tip -- two arms silently became main. Only the `HEAD == ref` assertion caught
it, and it is now reported as "asked X, landed Y" rather than as a bare False.

The author's branch is never pushed anywhere: a full code branch carries its own workflows and
pushing it would fire all of them. `git apply --check` runs before `git apply`, because a
half-applied patch yields an arm that is neither state and installs and measures happily.

THE JAMMED POSITIVE CONTROL RUNS FIRST. `--hog-ms 200 --hog-period-ms 250` blocks the page's main
thread on a timer. If the frame channel does not fall a long way under it, the channel cannot
report a blocked main thread and no arm below it means anything (INSTRUMENT-DEFECTS #33). Measured
locally on this exact scene at r100K: effective rate 57.9 -> 12.9 fps and busy 25% -> 87%, while
`1000/p50` read 62.5 BOTH jammed and unjammed. That is why the criteria scores `1000*raf.n/elapsed`
and never `raf.fps_p50`.

`--frame-clock passive`, not `updating`. `begin_updating()` drives the clock itself and read 60.0
fps with the main thread 80% blocked, so under it the presented-frame series cannot carry a claim.
The two existing studio probes on this branch pass `updating`; this one deliberately does not.
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

#: The PR's head ref. 33d65ce99 and its parent live ONLY here, not on main, so this ref has to be
#: fetched explicitly and with real depth. The previous run of this probe cloned `--depth 50` from
#: main, could not resolve the commit, and `git checkout` left HEAD on the default branch tip: two
#: arms silently became main, and only the `HEAD == ref` assertion below stopped them being
#: measured and reported as the author's commit.
PR_REF = "refs/pull/9477/head"
FETCH_DEPTH = 400

#: arm -> (ref it is built from, patches applied in order)
#:
#: Arm A is a DIRECT CHECKOUT of the commit the user named, with no patch step. It is the arm that
#: matters most and the one where a reconstruction has the most to go wrong; the faithful thing is
#: to check out exactly what the author wrote. At 33d65ce99 there is no pagination flag at all --
#: `thread-feature-flags.ts` carries only GRID_COLLAPSE_REASONING_ENABLED -- and `reasoning.tsx`
#: hardcodes `paginateReasoning={true}`, so pagination is LIVE. Aoff is that same checkout with
#: only that literal flipped, which is the isolation.
ARMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "baseT": ("90f85fdbf8fd6f9df1b99aff44f1e45da4c808d0", ()),
    "C": ("90f85fdbf8fd6f9df1b99aff44f1e45da4c808d0", ("C_tip.patch",)),
    "B": ("90f85fdbf8fd6f9df1b99aff44f1e45da4c808d0", ("C_tip.patch", "p4_flag_on.patch")),
    "A": ("33d65ce99b40de93361366447a3e34949480008a", ()),
    "Aoff": ("33d65ce99b40de93361366447a3e34949480008a", ("A_pagination_off.patch",)),
}

#: THE INSTRUMENT. One checkout of today's main, used to drive EVERY arm, and deliberately not any
#: arm's own tree. `amdv_rung_bench.py` imports the pacer, seeder and frozen corpus from
#: `--sb-root`; handing it the arm's own repo would measure an August arm with August instruments
#: and a today arm with today's, then report the difference as the effect of pagination. The
#: measuring device would co-vary with the subject. It is a separate clone so that it is also not
#: silently the same directory as an arm that a later edit might patch.
INSTRUMENT_REF = "90f85fdbf8fd6f9df1b99aff44f1e45da4c808d0"

#: Every key a state dict must carry, whatever went wrong. The previous run aborted the whole probe
#: on `KeyError: 'dist'` because a failed checkout returned early with a short dict, so three arms
#: that HAD built were never measured. A build failure must fail its own gate, not the run.
EMPTY_STATE = {
    "patch_ok": False, "dirty_files": 0, "pagination_literal": None,
    "install": {"rc": None}, "unsloth_bin": None, "home": None, "repo_root": None,
    "dist": {"path": None, "exists": False, "index_html": False, "asset_files": 0},
}

#: the pairs that differ by exactly one line. Recorded so the criteria can assert it rather than
#: trust this docstring.
ISOLATIONS = (("C", "B"), ("Aoff", "A"))

ARM_ORDER = ("baseT", "C", "B", "Aoff", "A")

sys.path.insert(0, str(HERE))
from webkit_paint_probe import fetch_xvfb, inventory, start_xserver  # noqa: E402
from studio_ladder_probe import gi_python, sh  # noqa: E402


def clone_at(repo_url: str, ref: str, dest: Path) -> dict:
    """Clone so that BOTH main commits and PR-only commits resolve, then land on `ref` exactly.

    `studio_ladder_probe.clone` is `--depth 50` off the default branch and cannot see a commit
    that only exists on a pull request. It also does not check that the checkout landed, so a
    failure leaves HEAD on the default branch tip and every later step succeeds against the wrong
    tree. Both of those happened, on the two arms that mattered most.
    """
    out: dict = {"url": repo_url, "ref": ref, "dest": str(dest)}
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    # A blobless partial clone: the whole commit graph, without paying for every blob in history.
    out["clone"] = sh(["git", "clone", "--filter=blob:none", repo_url, str(dest)], timeout=2400)
    if out["clone"].get("rc") != 0:
        out["clone_fallback"] = sh(["git", "clone", repo_url, str(dest)], timeout=3600)
    # The PR head, explicitly. 33d65ce99 and its parent 4351502ac exist nowhere else.
    out["fetch_pr"] = sh(["git", "fetch", "--depth", str(FETCH_DEPTH), "origin",
                          f"{PR_REF}:pr9477"], cwd=str(dest), timeout=1800)
    out["resolved"] = sh(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
                         cwd=str(dest), timeout=60)
    out["checkout"] = sh(["git", "checkout", "--detach", ref], cwd=str(dest), timeout=600)
    out["commit"] = (sh(["git", "rev-parse", "HEAD"], cwd=str(dest),
                        timeout=60).get("stdout") or "").strip()
    # SAID versus LANDED, both recorded, so a mismatch reads as itself in the gate evidence
    # instead of as a bare False.
    out["ref_requested"] = ref
    out["ref_landed"] = out["commit"]
    out["checkout_ok"] = out["commit"] == ref
    return out


def resolvable(repo_url: str, refs: list[str], work: Path) -> dict:
    """PRE-FLIGHT. Resolve every arm's ref in one scratch clone BEFORE anything is installed.

    Five installs are the expensive part of this job. Finding out after four of them that the
    fifth arm's commit was never fetched costs the whole GPU hold, which is exactly what happened.
    """
    scratch = work / "refcheck"
    out = clone_at(repo_url, "HEAD", scratch)
    got = {}
    for ref in refs:
        r = sh(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
               cwd=str(scratch), timeout=60)
        got[ref] = (r.get("stdout") or "").strip() or None
    shutil.rmtree(scratch, ignore_errors=True)
    return {"clone": {k: v for k, v in out.items() if k != "clone"}, "resolved": got,
            "all_resolved": all(got.values())}


def build_state(work: Path, name: str, repo_url: str, ref: str, patches: tuple[str, ...],
                install_timeout: int) -> dict:
    """One checkout, patched in order, installed. Returns what it built.

    EVERY return path carries the full key set, so a failure here fails this arm's gate and the
    other arms are still read.
    """
    root = work / f"state_{name}"
    out: dict = {"name": name, "ref": ref, "patches": list(patches), **EMPTY_STATE}
    out["patch_steps"] = []
    out.update(clone_at(repo_url, ref, root))

    ok = out.get("checkout_ok", False)
    if not ok:
        out["patch_ok"] = False
        out["why"] = (f"asked for {ref[:9]} and landed on {str(out.get('commit'))[:9]}; "
                      f"the commit was not fetched, so this arm was NOT built")
        return out
    for p in patches:
        path = LADDER / p
        if not path.is_file():
            out["patch_steps"].append({"patch": p, "missing": str(path)})
            ok = False
            break
        chk = sh(["git", "apply", "--check", "-v", str(path)], cwd=str(root), timeout=300)
        app = sh(["git", "apply", "--stat", "--apply", str(path)], cwd=str(root), timeout=300)
        out["patch_steps"].append({
            "patch": p, "bytes": path.stat().st_size,
            "check_rc": chk.get("rc"), "apply_rc": app.get("rc"),
            "stderr": (chk.get("stderr") or app.get("stderr") or "")[:400]})
        if app.get("rc") != 0:
            ok = False
            break
    out["patch_ok"] = ok
    out["dirty_files"] = len([l for l in (sh(["git", "status", "--porcelain"], cwd=str(root),
                                             timeout=60).get("stdout") or "").splitlines()
                              if l.strip()])
    # READ THE FLAG BACK OUT OF THE SOURCE rather than trusting the patch name. This is the one
    # fact the whole run turns on, and a patch that applied to the wrong hunk would still say rc=0.
    flag = root / "studio" / "frontend" / "src" / "components" / "assistant-ui"
    out["pagination_literal"] = None
    for candidate, needle in ((flag / "thread-feature-flags.ts", "REASONING_PAGINATION_ENABLED"),
                              (flag / "reasoning.tsx", "paginateReasoning=")):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
            if needle in line and ("true" in line or "false" in line):
                out.setdefault("pagination_source_lines", []).append(
                    f"{candidate.name}: {line.strip()[:120]}")
                if out["pagination_literal"] is None:
                    out["pagination_literal"] = "true" if "true" in line else "false"

    home = work / f"home_{name}"
    home.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    inst = sh(["bash", "install.sh", "--local"], cwd=str(root), timeout=install_timeout,
              env={"UNSLOTH_STUDIO_HOME": str(home)})
    inst["seconds"] = round(time.time() - t0, 1)
    inst["stdout"] = (inst.get("stdout") or "")[-3000:]
    out["install"] = inst

    dist = root / "studio" / "frontend" / "dist"
    out["dist"] = {"path": str(dist), "exists": dist.is_dir(),
                   "index_html": (dist / "index.html").is_file(),
                   "asset_files": len(list((dist / "assets").rglob("*")))
                   if (dist / "assets").is_dir() else 0}
    binp = None
    for c in [home / "bin" / "unsloth", *sorted(home.glob(".venv*/bin/unsloth"))]:
        if c.exists() and os.access(c, os.X_OK):
            binp = str(c)
            break
    out["unsloth_bin"] = binp
    out["home"] = str(home)
    out["repo_root"] = str(root)
    return out


def measure(work: Path, st: dict, arm: str, rung: str, rep: str, port: int, display: str,
            py_gi: str, timeout: int, instrument: Path, hog_ms: int = 0) -> dict:
    rhome = work / f"run_{arm}_{rung}_{rep}"
    shutil.rmtree(rhome, ignore_errors=True)
    rhome.mkdir(parents=True, exist_ok=True)
    src_home = Path(st["home"])
    for d in ("assets", "bin", "cache", "compiled_cache", "llama.cpp", "share",
              "unsloth_studio", "whisper.cpp"):
        s = src_home / d
        if s.exists() and not (rhome / d).exists():
            os.symlink(s, rhome / d)
    for d in ("exports", "outputs", "logs", "runs", "rag", "auth"):
        (rhome / d).mkdir(parents=True, exist_ok=True)

    outp = work / "out" / f"{arm}_{rung}_{rep}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(LADDER / "amdv_rung_bench.py"),
           "--rung", rung, "--rep", f"{arm}_{rung}_{rep}",
           "--dist", st["dist"]["path"], "--home", str(rhome),
           "--port", str(port), "--display", display,
           # ONE instrument for every arm, never the arm's own checkout.
           "--sb-root", str(instrument), "--unsloth-bin", st["unsloth_bin"],
           "--python-gi", py_gi,
           # A scene of this probe's own, so no other probe on this branch changes behaviour. It
           # is `amdv_scene.js` plus one thing: a census taken while the reasoning pane is OPEN.
           # Without it, "pagination changed nothing" cannot be told apart from "the flag never
           # reached the DOM", because the gesture closes the pane before census_after runs.
           "--scene", str(LADDER / "amdv_scene_p4.js"),
           "--driver", str(LADDER / "amdv_drive.py"),
           # PASSIVE. `updating` cannot report a blocked main thread; see the module docstring.
           "--frame-clock", "passive",
           "--skip-send",
           "--out", str(outp)]
    if hog_ms:
        cmd += ["--hog-ms", str(hog_ms), "--hog-period-ms", "250"]
    t0 = time.time()
    r = sh(cmd, timeout=timeout, env={"UNSLOTH_WORKSPACE": str(work)})
    entry = {"arm": arm, "rung": rung, "rep": rep, "port": port, "hog_ms": hog_ms,
             "seconds": round(time.time() - t0, 1), "rc": r.get("rc"), "error": r.get("error"),
             "stdout_tail": "\n".join((r.get("stdout") or "").splitlines()[-25:]),
             "stderr_tail": (r.get("stderr") or "")[-2000:]}
    if outp.is_file():
        try:
            entry["payload"] = json.loads(outp.read_text())
        except Exception as e:                                           # noqa: BLE001
            entry["payload_error"] = f"{type(e).__name__}: {e}"
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--state", default="host")
    ap.add_argument("--checkout", default="")
    ap.add_argument("--work", default=os.environ.get("AMD_CI_WORK", "/tmp/studio_p4"))
    ap.add_argument("--repo", default="https://github.com/unslothai/unsloth")
    ap.add_argument("--rungs", default="100K,500K")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--first-port", type=int, default=5461)
    ap.add_argument("--install-timeout", type=int, default=3600)
    ap.add_argument("--rung-timeout", type=int, default=2400)
    ap.add_argument("--hog-ms", type=int, default=200)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    work = Path(args.work) / "p4"
    work.mkdir(parents=True, exist_ok=True)
    rungs = [r.strip() for r in args.rungs.split(",") if r.strip()]
    obs: dict = {"rungs": rungs, "reps": args.reps, "arms": {k: list(v[1]) for k, v in ARMS.items()},
                 "arm_refs": {k: v[0] for k, v in ARMS.items()},
                 "isolations": [list(x) for x in ISOLATIONS],
                 "patch_bytes": {p.name: p.stat().st_size
                                 for p in sorted(LADDER.glob("*.patch"))}}

    obs["inventory"] = inventory()
    if not obs["inventory"]["Xvfb"]:
        obs["fetch_xvfb"] = fetch_xvfb(work)
    xproc, xinfo = start_xserver(work, obs)
    obs["xserver"] = xinfo
    if not xinfo.get("display"):
        obs["fatal"] = "no display server could be started"
        args.out.write_text(json.dumps(obs, indent=2))
        return 0

    py_gi, tried = gi_python()
    obs["gi_python"] = {"chosen": py_gi, "tried": tried}
    try:
        if py_gi is None:
            obs["fatal"] = "no python on this host can import gi + WebKit2 4.1"
            return 0

        # PRE-FLIGHT: can every arm's commit be resolved at all? Five installs is the expensive
        # part of this job, and discovering after four of them that the fifth commit was never
        # fetched costs the whole GPU hold. Cheap to ask, and it names the missing ref.
        obs["preflight"] = resolvable(args.repo, sorted({ARMS[n][0] for n in ARM_ORDER}), work)
        print(f"== preflight: {obs['preflight']['resolved']}", flush=True)
        if not obs["preflight"]["all_resolved"]:
            missing = [k for k, v in obs["preflight"]["resolved"].items() if not v]
            obs["fatal"] = (f"these commits could not be resolved even after fetching {PR_REF}: "
                            f"{missing}. Nothing was installed.")
            print(f"== {obs['fatal']}", flush=True)
            return 0

        # THE INSTRUMENT, cloned once, before any arm.
        instrument = work / "instrument"
        obs["instrument"] = clone_at(args.repo, INSTRUMENT_REF, instrument)
        obs["instrument"]["path"] = str(instrument)
        if not obs["instrument"].get("checkout_ok"):
            obs["fatal"] = (f"the instrument checkout asked for {INSTRUMENT_REF[:9]} and landed "
                            f"on {str(obs['instrument'].get('commit'))[:9]}; refusing to measure "
                            f"any arm with an instrument that is not the one declared")
            return 0
        sb = instrument / "tests" / "studio" / "studiobench"
        obs["instrument"]["studiobench_present"] = sb.is_dir()
        if not sb.is_dir():
            obs["fatal"] = f"the pinned instrument has no studiobench package at {sb}"
            return 0
        print(f"== instrument: {instrument} at {obs['instrument']['commit'][:9]}", flush=True)

        obs["states"] = {}
        for name in ARM_ORDER:
            ref, patches = ARMS[name]
            obs["states"][name] = build_state(work, name, args.repo, ref, patches,
                                              args.install_timeout)
            st = obs["states"][name]
            print(f"== built {name}: asked {ref[:9]} landed {str(st.get('commit'))[:9]} "
                  f"patch_ok={st.get('patch_ok')} flag={st.get('pagination_literal')} "
                  f"install rc={(st.get('install') or {}).get('rc')}", flush=True)

        # An arm that did not build fails its GATE. It does not abort the probe: three arms that
        # DID build were thrown away that way once, and the artifact is worth more than the tick.
        usable = [n for n in ARM_ORDER
                  if (obs["states"][n].get("dist") or {}).get("exists")
                  and obs["states"][n].get("unsloth_bin") and obs["states"][n].get("patch_ok")]
        obs["arms_built"] = usable
        obs["arms_not_built"] = [n for n in ARM_ORDER if n not in usable]
        if not usable:
            obs["fatal"] = "no arm built"
            return 0

        obs["runs"] = []
        port = args.first_port

        # ── the jammed positive control, FIRST, and it gates everything after it ──────────────
        control_arm = "baseT" if "baseT" in usable else usable[0]
        for rung in rungs:
            obs["runs"].append(measure(work, obs["states"][control_arm], "JAM", rung, "jam", port,
                                       xinfo["display"], py_gi, args.rung_timeout,
                                       instrument, hog_ms=args.hog_ms))
            port += 1

        # ── the arms, interleaved within each rep, never blocked ─────────────────────────────
        for rep in range(1, args.reps + 1):
            for rung in rungs:
                for arm in usable:
                    obs["runs"].append(measure(work, obs["states"][arm], arm, rung, str(rep),
                                               port, xinfo["display"], py_gi, args.rung_timeout,
                                               instrument))
                    port += 1
                    time.sleep(5)

        logs = work / "out" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        got = []
        for pat in ("*.log", "*.jsonl"):
            for f in list((work / "out").glob(pat)) + list(work.rglob("logs/" + pat)):
                try:
                    if f.parent == logs:
                        continue
                    shutil.copy2(f, logs / (f.parent.name + "__" + f.name))
                    got.append(f.name)
                except Exception:                                        # noqa: BLE001
                    pass
        obs["logs_collected"] = got
        return 0
    finally:
        if xproc is not None and xproc.poll() is None:
            try:
                os.kill(xproc.pid, signal.SIGTERM)
                time.sleep(2)
                if xproc.poll() is None:
                    os.kill(xproc.pid, signal.SIGKILL)
            except Exception:                                            # noqa: BLE001
                pass
            obs["xserver"]["stopped_pid"] = xproc.pid
        args.out.write_text(json.dumps(obs, indent=2))


if __name__ == "__main__":
    sys.exit(main())

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
    A       5a85104f0 + A_authored    commit 33d65ce99 as the author wrote it. At that commit there
                                      is no flag at all: `reasoning.tsx` says
                                      `paginateReasoning={true}`, hardcoded, so pagination is LIVE.
                                      This is the state that was named, and 5a85104f0 is ITS OWN
                                      merge base, not today's main
    Aoff    A + A_pagination_off      the same commit with that one literal flipped to false.
                                      A vs Aoff is the second one-line isolation

A and B are NOT the same code: six of the branch's own commits sit between them, one of which
(`0b3592ad1`, open-fence re-parsing) is itself a perf change. So they are carried separately and
any difference between them is attributed rather than averaged into a single "piece 4" number.

EVERY ARM IS A PATCH ON AN UPSTREAM COMMIT, never a branch checkout. Pushing the author's branch to
a fork would fire every workflow it carries, and a branch can drift under you; a patch against a
commit that is in unslothai/unsloth history is reproducible by construction. `git apply --check`
runs before `git apply`, because a half-applied patch yields an arm that is neither state and
installs and measures perfectly happily.

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

#: arm -> (upstream ref it is built from, patches applied in order)
ARMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "baseT": ("90f85fdbf8fd6f9df1b99aff44f1e45da4c808d0", ()),
    "C": ("90f85fdbf8fd6f9df1b99aff44f1e45da4c808d0", ("C_tip.patch",)),
    "B": ("90f85fdbf8fd6f9df1b99aff44f1e45da4c808d0", ("C_tip.patch", "p4_flag_on.patch")),
    "A": ("5a85104f09fb4d753e25a7cec625347554ae1c0e", ("A_authored.patch",)),
    "Aoff": ("5a85104f09fb4d753e25a7cec625347554ae1c0e",
             ("A_authored.patch", "A_pagination_off.patch")),
}

#: the pairs that differ by exactly one line. Recorded so the criteria can assert it rather than
#: trust this docstring.
ISOLATIONS = (("C", "B"), ("Aoff", "A"))

ARM_ORDER = ("baseT", "C", "B", "Aoff", "A")

sys.path.insert(0, str(HERE))
from webkit_paint_probe import fetch_xvfb, inventory, start_xserver  # noqa: E402
from studio_ladder_probe import clone, gi_python, sh  # noqa: E402


def build_state(work: Path, name: str, repo_url: str, ref: str, patches: tuple[str, ...],
                install_timeout: int) -> dict:
    """One checkout, patched in order, installed. Returns what it built."""
    root = work / f"state_{name}"
    out: dict = {"name": name, "ref": ref, "patches": list(patches)}
    out.update(clone(repo_url, ref, root))
    out["checkout"] = sh(["git", "checkout", "--detach", ref], cwd=str(root), timeout=300)
    out["commit"] = (sh(["git", "rev-parse", "HEAD"], cwd=str(root),
                        timeout=60).get("stdout") or "").strip()

    out["patch_steps"] = []
    ok = out["commit"] == ref
    if not ok:
        out["patch_ok"] = False
        out["dirty_files"] = 0
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
            py_gi: str, timeout: int, hog_ms: int = 0) -> dict:
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
           "--sb-root", st["repo_root"], "--unsloth-bin", st["unsloth_bin"],
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

        obs["states"] = {}
        for name in ARM_ORDER:
            ref, patches = ARMS[name]
            obs["states"][name] = build_state(work, name, args.repo, ref, patches,
                                              args.install_timeout)
            st = obs["states"][name]
            print(f"== built {name}: commit {st.get('commit', '')[:9]} "
                  f"patch_ok={st.get('patch_ok')} flag={st.get('pagination_literal')} "
                  f"install rc={(st.get('install') or {}).get('rc')}", flush=True)

        bad = [n for n, st in obs["states"].items()
               if not st["dist"]["exists"] or not st["unsloth_bin"] or not st["patch_ok"]]
        if bad:
            obs["fatal"] = f"states did not build: {bad}"
            return 0

        obs["runs"] = []
        port = args.first_port

        # ── the jammed positive control, FIRST, and it gates everything after it ──────────────
        for rung in rungs:
            obs["runs"].append(measure(work, obs["states"]["baseT"], "JAM", rung, "jam", port,
                                       xinfo["display"], py_gi, args.rung_timeout,
                                       hog_ms=args.hog_ms))
            port += 1

        # ── the arms, interleaved within each rep, never blocked ─────────────────────────────
        for rep in range(1, args.reps + 1):
            for rung in rungs:
                for arm in ARM_ORDER:
                    obs["runs"].append(measure(work, obs["states"][arm], arm, rung, str(rep),
                                               port, xinfo["display"], py_gi, args.rung_timeout))
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

#!/usr/bin/env python3
"""Probe: does PR 9695's 5.02x survive a rebase onto today's main?

Observes only. criteria/studio_r9695_rebase.py judges.

THE QUESTION, and why it is being asked again.

PR 9695 renders fenced code inside reasoning panes PLAIN. On this venue it was measured at
**5.021x** on `action:reasoning_toggle_all` at r100K (AMD runs `32833058576` and `32841817590`),
with the span census falling from 74,250 to 10,917. That number was taken against a main that no
longer exists. Since then main merged a great deal of work on EXACTLY this path, including

  * PR 9799, the idle grammar pre-warm fix: `warmGrammars` was warming on an EMPTY STRING;
  * PR 9731, the KaTeX containment work;
  * PR 9787, shiki's `tokenizeTimeLimit`.

and main's own per-fence deferral now ships on by default (`code-fence-mode.ts`
`SHIP_DEFAULT = "defer"`), so a fence nobody has scrolled to ALREADY renders the plain shell on
`main`. If that has absorbed the win, then the win is gone, and there is a live precedent for
exactly this: PR 9731's +92% turned out to be stale for the same reason. A result that says "the
benefit has evaporated" is the correct answer if that is what the box says, and this probe is
built so that answer is as easy to reach as the other one.

TWO ARMS, ONE CHECKOUT, ONE BACKEND.

    arm     built as
    main    0be140dbd, unpatched. The BASE.
    head    0be140dbd + r9695.patch, which is `git diff 0be140dbd..<PR 9695 head, rebased>`.

The two arms differ by three files, two of them under `studio/frontend/src` and one a test that
`tsc -b` never sees (`npm run build` is `tsc -b && vite build`, and the tests live in
`tsconfig.test.json`, which only `npm run typecheck` builds). So an arm's entire identity is its
built `dist`, and the backend can be common.

That is not a saving, it is a CONFOUND REMOVED. Two installs mean two chances for an install to
differ between the arms, and the difference is then reported as the mechanism. Here there is ONE
clone, built twice from the same `node_modules` and the same isolated Node, and ONE backend
install in the measuring job that serves both dists with `unsloth studio -f <dist>`.

WHAT MAKES AN ARM AN ARM, CHECKED RATHER THAN ASSUMED.

`git apply` reports rc=0 for a patch that landed on the wrong hunk, so the patch list is not
evidence. Each arm's SOURCE is grepped for `MarkdownCodeHighlightingContext`, which PR 9695
introduces and which does not exist anywhere in `0be140dbd`: present means head, absent means
main. On top of that the two exported bundles must hash DIFFERENTLY, and the scene reads the
mechanism back out of the running page (fences inside reasoning panes carrying
`data-unsloth-fence-deferred="true"` and no spans).

THE GESTURE IS `action:reasoning_toggle_all`. `reasoning_toggle` names TWO different gestures in
this campaign -- the first pane and every pane -- and the same mechanism read 1.9% on one and
1.913x on the other, both correctly. The scored window opens every pane.

BOTH RUNGS. r100K is where the 5.02x was measured; r500K is the rung the user's complaint is
actually about, and the trade could in principle flip there because the off-arm's standing DOM
grows most. Neither rung is allowed to license the other: the criteria module gates each rung on
its OWN controls, and the r500K rung of a previous campaign was VOID for exactly that reason.

THE JAM IS A WINDOW INSIDE EVERY SESSION, not a session of its own, so the positive control is per
arm, per rung and per repetition. The IDLE window is the other control, and on this question it is
the one most likely to matter: a `plain` arm at r500K once read +50% against the arm it was being
compared with while its idle window was already stalling in 3 repetitions of 5. The probe records
both; the criteria module discards repetitions by name and counts them.
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

#: THE BASE. A SHA, never `main`: a rehearsal of a sibling probe cloned "main" and landed on a
#: commit eight days from the one the run was about, and nothing in its output said so.
BASE_REF = "0be140dbd458535f7f93dc1eaffe703611ff9acf"

#: The instrument (studiobench: pacer, seeder, frozen corpus) comes from ONE pinned checkout and
#: never from an arm's own tree. Both arms are the same commit here, so this is not load-bearing
#: the way it is for a two-commit comparison -- it is kept because the assertion is cheap and its
#: absence is what let a sibling harness measure each arm with the harness that shipped with it.
INSTRUMENT_REF = BASE_REF

REFERENCE = "main"
TREATMENT = "head"
PATCH = "r9695.patch"

#: PR 9695 introduces this symbol. `git grep MarkdownCodeHighlightingContext 0be140dbd` is empty,
#: so its presence in an arm's SOURCE is a fact about the tree that was built rather than a
#: restatement of the patch list. A patch that applies to the wrong hunk still returns rc=0.
HEAD_MARKER = "MarkdownCodeHighlightingContext"

ARMS: "dict[str, tuple[str, ...]]" = {REFERENCE: (), TREATMENT: (PATCH,)}
#: ORDER MATTERS. `main` is built first so that a build job cut short still carries the arm every
#: gate is defined against.
ARM_ORDER = (REFERENCE, TREATMENT)

EMPTY_STATE = {
    "patch_ok": False, "dirty_files": 0, "head_marker_present": None,
    "install": {"rc": None}, "unsloth_bin": None, "home": None, "repo_root": None,
    "dist": {"path": None, "exists": False, "index_html": False, "asset_files": 0},
}

sys.path.insert(0, str(HERE))
from webkit_paint_probe import fetch_xvfb, inventory, start_xserver  # noqa: E402
from studio_ladder_probe import gi_python, sh  # noqa: E402

ZOO_URL = "https://github.com/unslothai/unsloth-zoo"


def clone_at(repo_url: str, ref: str, dest: Path) -> dict:
    """Clone, land on `ref` exactly, and record ASKED versus LANDED either way.

    A shallow clone that cannot resolve the ref leaves HEAD on the default branch tip and every
    later step succeeds against the wrong tree. That has happened on this harness, to the two arms
    that mattered most, and only this assertion caught it.
    """
    out: dict = {"url": repo_url, "ref": ref, "dest": str(dest)}
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    out["clone"] = sh(["git", "clone", "--filter=blob:none", repo_url, str(dest)], timeout=2400)
    if out["clone"].get("rc") != 0:
        out["clone_fallback"] = sh(["git", "clone", repo_url, str(dest)], timeout=3600)
    out["resolved"] = sh(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
                         cwd=str(dest), timeout=60)
    out["checkout"] = sh(["git", "checkout", "--detach", ref], cwd=str(dest), timeout=600)
    out["commit"] = (sh(["git", "rev-parse", "HEAD"], cwd=str(dest),
                        timeout=60).get("stdout") or "").strip()
    out["commit_line"] = (sh(["git", "log", "-1", "--format=%H %ci %s"], cwd=str(dest),
                             timeout=60).get("stdout") or "").strip()
    out["ref_requested"] = ref
    out["ref_landed"] = out["commit"]
    out["checkout_ok"] = out["commit"] == ref
    return out


def bundle_hash(dist: Path) -> str:
    """Content hash of a built frontend bundle, the SAME definition `final_rung_bench.py` uses, so
    the hash recorded at build time and the one recorded at measure time are comparable and a dist
    that changed in between shows up rather than passing silently."""
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


def tree_hash(root: Path) -> str:
    """Content hash of a directory tree. Used on the instrument, so "one instrument for every
    session" is checkable by CONTENT and not only by resolved path: a path assertion cannot see a
    directory that changed under it between the first session and the twentieth."""
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def head_marker_present(root: Path) -> bool:
    """Is PR 9695 in this tree? Read out of the SOURCE, not out of the patch list."""
    src = root / "studio" / "frontend" / "src"
    if not src.is_dir():
        return False
    for f in src.rglob("*.ts*"):
        try:
            if HEAD_MARKER in f.read_text(encoding="utf-8", errors="replace"):
                return True
        except Exception:                                                # noqa: BLE001
            continue
    return False


def zoo_identity(home: Path) -> dict:
    """Which unsloth-zoo this install ended up with, read out of the INSTALLED metadata.

    One backend install serves both arms here, so the two arms cannot differ in it. Recorded
    anyway: a reader who wants to reproduce this needs the commit, and a future variant of this
    probe that installs twice would need the gate.
    """
    out: dict = {"home": str(home), "dist_info": None, "commit_id": None, "version": None,
                 "url": None}
    infos = sorted(home.glob(".venv*/lib/python*/site-packages/unsloth_zoo-*.dist-info"))
    infos += sorted(home.glob("**/site-packages/unsloth_zoo-*.dist-info"))
    for info in infos:
        out["dist_info"] = str(info)
        out["version"] = info.name.split("-", 1)[1].replace(".dist-info", "")
        du = info / "direct_url.json"
        if du.is_file():
            try:
                j = json.loads(du.read_text())
                out["url"] = j.get("url")
                out["commit_id"] = (j.get("vcs_info") or {}).get("commit_id")
            except Exception as e:                                       # noqa: BLE001
                out["direct_url_error"] = f"{type(e).__name__}: {e}"
        break
    return out


def measure(work: Path, st: dict, arm: str, rung: str, rep: str, port: int, display: str,
            py_gi: str, timeout: int, instrument: Path, instrument_hash: str,
            hog_ms: int, hog_period_ms: int) -> dict:
    """One session: one arm, one rung, one repetition, its own port and its own Studio home."""
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
    cmd = [sys.executable, str(LADDER / "final_rung_bench.py"),
           "--rung", rung, "--rep", str(rep), "--arm", arm,
           "--dist", st["dist"]["path"], "--home", str(rhome),
           "--port", str(port), "--display", display,
           "--sb-root", str(instrument), "--unsloth-bin", st["unsloth_bin"],
           "--python-gi", py_gi,
           "--scene", str(LADDER / "r9695_scene.js"),
           "--driver", str(LADDER / "amdv_drive.py"),
           # The scene owns the jam and confines it to one window; nothing is injected page-wide.
           "--hog-ms", str(hog_ms), "--hog-period-ms", str(hog_period_ms),
           # `passive`, never `updating`: `begin_updating()` drives the clock itself and read
           # 60.0 fps with the main thread 80% blocked. The headline is computed in-page from
           # frames over wall time either way, but a driver that spins the clock also changes
           # what the page is asked to do.
           "--frame-clock", "passive",
           # No send, no stream. This question is about a SETTLED thread whose panes are opened.
           "--skip-send",
           "--instrument-hash", instrument_hash,
           "--out", str(outp)]
    t0 = time.time()
    r = sh(cmd, timeout=timeout, env={"UNSLOTH_WORKSPACE": str(work)})
    entry = {"arm": arm, "rung": rung, "rep": rep, "port": port,
             "expected_bundle_hash": st.get("exported_bundle_hash"),
             "seconds": round(time.time() - t0, 1), "rc": r.get("rc"), "error": r.get("error"),
             "stdout_tail": "\n".join((r.get("stdout") or "").splitlines()[-30:]),
             "stderr_tail": (r.get("stderr") or "")[-2500:]}
    if outp.is_file():
        try:
            entry["payload"] = json.loads(outp.read_text())
        except Exception as e:                                           # noqa: BLE001
            entry["payload_error"] = f"{type(e).__name__}: {e}"
    # REAP THIS SESSION'S SERVER BEFORE STARTING THE NEXT ONE, BY PORT.
    #
    # `subprocess.run(timeout=)` kills the direct child only, and the bench launches Studio under
    # `setsid` with its teardown in a `finally` that a SIGKILL never reaches. A session that
    # overran would leave a Studio and a WebKitGTK toplevel alive, and two toplevels on one X
    # server occlude each other: the lower one stops being asked to paint, which reads as a frame
    # rate. A pattern-matching killer is not an option, because the pattern lands on the command
    # line of the shell running it and the `[e]` bracket trick does not fix that.
    reap = sh(["bash", "-c",
               f"pid=$(ss -lptnH 'sport = :{port}' 2>/dev/null | "
               f"grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2); "
               f"if [ -n \"$pid\" ]; then echo \"killing $pid on {port}\"; "
               f"kill -TERM \"$pid\"; sleep 2; fi"], timeout=60)
    entry["reaped"] = (reap.get("stdout") or "").strip() or None
    return entry


def parse_reps(spec: str, rungs: list[str]) -> dict:
    """`--reps 100K:5,500K:5` or a bare `--reps 5`, because the rungs cost different amounts."""
    out: dict = {}
    if ":" not in spec:
        return {r: int(spec) for r in rungs}
    for part in spec.split(","):
        k, _, v = part.partition(":")
        out[k.strip()] = int(v)
    return {r: out.get(r, 2) for r in rungs}


def do_build(args, obs: dict, work: Path) -> int:
    """BUILD MODE. No GPU, no display, no measurement. Runs in its own job.

    ONE CHECKOUT, BUILT TWICE. The unpatched tree is built first and its dist exported; the patch
    is then applied to the SAME tree, `dist` is removed so `setup.sh`'s mtime-based staleness
    check cannot decide the bundle is current, and it is built again. Both builds therefore share
    one `node_modules`, one isolated Node and one lockfile, and the only thing that differs
    between the two exported bundles is the patch.
    """
    root = work / "src"
    obs["clone"] = clone_at(args.repo, BASE_REF, root)
    if not obs["clone"].get("checkout_ok"):
        obs["fatal"] = (f"the checkout asked for {BASE_REF[:9]} and landed on "
                        f"{str(obs['clone'].get('commit'))[:9]}; nothing was built")
        return 0

    # PRE-FLIGHT, before an install is spent: does the patch apply at all, and does it flip the
    # marker? Cheap to ask, and it names the file and the hunk when it does not.
    patch_path = LADDER / PATCH
    pre: dict = {"patch": str(patch_path), "exists": patch_path.is_file(),
                 "bytes": patch_path.stat().st_size if patch_path.is_file() else None,
                 "marker_absent_at_base": not head_marker_present(root)}
    if patch_path.is_file():
        pre["check"] = sh(["git", "apply", "--check", "-v", str(patch_path)],
                          cwd=str(root), timeout=300)
        pre["apply"] = sh(["git", "apply", "--stat", "--apply", str(patch_path)],
                          cwd=str(root), timeout=300)
        pre["marker_present_after_apply"] = head_marker_present(root)
        pre["changed"] = [l[3:] for l in (sh(["git", "status", "--porcelain"], cwd=str(root),
                                             timeout=60).get("stdout") or "").splitlines()
                          if l.strip()]
        # THE PREMISE, CHECKED. Nothing outside the frontend may differ, or one shared backend
        # install cannot stand for both arms and a real difference would be silently erased.
        pre["changed_outside_frontend"] = [c for c in pre["changed"]
                                           if not c.startswith("studio/frontend/")]
        sh(["git", "checkout", "--force", "--", "studio"], cwd=str(root), timeout=300)
        pre["marker_after_revert"] = head_marker_present(root)
    obs["preflight"] = pre
    print(f"== preflight: exists={pre['exists']} check_rc={(pre.get('check') or {}).get('rc')} "
          f"apply_rc={(pre.get('apply') or {}).get('rc')} "
          f"base_clean={pre['marker_absent_at_base']} "
          f"after_apply={pre.get('marker_present_after_apply')} "
          f"after_revert={pre.get('marker_after_revert')} "
          f"outside={pre.get('changed_outside_frontend')}", flush=True)
    if not (pre["exists"] and (pre.get("apply") or {}).get("rc") == 0
            and pre["marker_absent_at_base"] and pre.get("marker_present_after_apply")
            and pre.get("marker_after_revert") is False
            and not pre.get("changed_outside_frontend")):
        obs["fatal"] = ("the preflight on r9695.patch did not hold, so nothing was built; see "
                        "observations.preflight")
        return 0

    dists = Path(args.dist_out)
    dists.mkdir(parents=True, exist_ok=True)
    home = work / "home_build"
    home.mkdir(parents=True, exist_ok=True)
    dist = root / "studio" / "frontend" / "dist"
    obs["states"] = {}

    for name in ARM_ORDER:
        st: dict = {"name": name, "ref": BASE_REF, "patches": list(ARMS[name]),
                    **EMPTY_STATE, "patch_steps": []}
        st.update({k: obs["clone"][k] for k in ("commit", "ref_landed", "checkout_ok")})
        ok = True
        for p in ARMS[name]:
            path = LADDER / p
            # --check BEFORE --apply: a half-applied patch yields a tree that is neither state
            # and installs and measures perfectly happily.
            chk = sh(["git", "apply", "--check", "-v", str(path)], cwd=str(root), timeout=300)
            app = sh(["git", "apply", "--stat", "--apply", str(path)], cwd=str(root), timeout=300)
            st["patch_steps"].append({"patch": p, "bytes": path.stat().st_size,
                                      "check_rc": chk.get("rc"), "apply_rc": app.get("rc"),
                                      "stderr": (chk.get("stderr") or app.get("stderr") or "")[:400]})
            if app.get("rc") != 0:
                ok = False
                break
        st["patch_ok"] = ok
        st["dirty_files"] = len([l for l in (sh(["git", "status", "--porcelain"], cwd=str(root),
                                                timeout=60).get("stdout") or "").splitlines()
                                 if l.strip()])
        st["head_marker_present"] = head_marker_present(root)

        # `setup.sh` decides whether to rebuild by comparing source mtimes against the `dist`
        # DIRECTORY's own mtime. Removing it makes the decision unambiguous rather than relying
        # on `git apply` having touched the files late enough.
        shutil.rmtree(dist, ignore_errors=True)
        t0 = time.time()
        inst = sh(["bash", "install.sh", "--local", *args.install_arg], cwd=str(root),
                  timeout=args.install_timeout, env={"UNSLOTH_STUDIO_HOME": str(home)})
        inst["seconds"] = round(time.time() - t0, 1)
        inst["args"] = ["--local", *args.install_arg]
        inst["stdout"] = (inst.get("stdout") or "")[-6000:]
        st["install"] = inst
        st["bundle_hash"] = bundle_hash(dist)
        st["dist"] = {"path": str(dist), "exists": dist.is_dir(),
                      "index_html": (dist / "index.html").is_file(),
                      "asset_files": len(list((dist / "assets").rglob("*")))
                      if (dist / "assets").is_dir() else 0}
        if st["dist"]["index_html"]:
            dest = dists / name
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(dist, dest)
            st["exported_dist"] = str(dest)
            st["exported_bundle_hash"] = bundle_hash(dest)
        st["disk"] = sh(["df", "-h", str(work)], timeout=60).get("stdout")
        obs["states"][name] = st
        print(f"== built {name}: landed {str(st.get('commit'))[:9]} patch_ok={st['patch_ok']} "
              f"marker={st['head_marker_present']} install rc={inst.get('rc')} "
              f"in {inst.get('seconds')}s bundle={st.get('exported_bundle_hash')}", flush=True)

        # Back to the base tree for the next arm. `checkout --force -- studio` reverts TRACKED
        # files only, so `node_modules` and the isolated Node under the build home survive and
        # the second build reuses them. `git clean -fdx -- studio` would delete both and turn a
        # 65 second install into a full npm install.
        sh(["git", "checkout", "--force", "--", "studio"], cwd=str(root), timeout=300)

    manifest = {
        "base_ref": BASE_REF, "arms": {k: list(v) for k, v in ARMS.items()},
        "arm_order": list(ARM_ORDER), "head_marker": HEAD_MARKER,
        "clone": obs["clone"], "preflight": obs["preflight"],
        "states": {n: {k: v for k, v in s.items()
                       if k in ("ref", "ref_landed", "commit", "checkout_ok", "patches",
                                "patch_ok", "patch_steps", "head_marker_present", "dirty_files",
                                "why", "exported_dist", "exported_bundle_hash", "dist")}
                   for n, s in obs["states"].items()},
        "install": {n: {"rc": (s.get("install") or {}).get("rc"),
                        "seconds": (s.get("install") or {}).get("seconds"),
                        "args": (s.get("install") or {}).get("args")}
                    for n, s in obs["states"].items()},
    }
    (dists / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
    obs["build_manifest"] = str(dists / "build_manifest.json")
    obs["arms_built"] = [n for n, s in obs["states"].items() if s.get("exported_dist")]
    obs["arms_not_built"] = [n for n in ARM_ORDER if n not in obs["arms_built"]]
    hashes = {n: obs["states"][n].get("exported_bundle_hash") for n in obs["arms_built"]}
    obs["bundle_hashes"] = hashes
    # TWO ARMS THAT HASH THE SAME ARE ONE ARM. Warned here so the build log says it; the gate in
    # the criteria module is what fails the run.
    if len(set(hashes.values())) < len(hashes):
        print(f"::warning::the arms share a bundle hash, so they are the same tree: {hashes}",
              flush=True)
    print(f"== built {len(obs['arms_built'])}/{len(ARM_ORDER)}: {hashes}", flush=True)
    return 0


def do_measure(args, obs: dict, work: Path, rungs: list[str], reps: dict) -> int:
    obs["inventory"] = inventory()
    if not obs["inventory"]["Xvfb"]:
        obs["fetch_xvfb"] = fetch_xvfb(work)
    # A display DERIVED FROM THIS RUN'S PORT RANGE rather than the shared default. Four ephemeral
    # slots run on one machine, and a job that lands on the same display shares an X server with
    # this one; two WebKitGTK toplevels there occlude each other and the lower one stops being
    # asked to paint, which reads as a frame rate.
    xproc, xinfo = start_xserver(work, obs, display=f":{(args.first_port % 90) + 100}")
    obs["xserver"] = xinfo
    if not xinfo.get("display"):
        obs["fatal"] = "no display server could be started"
        return 0

    py_gi, tried = gi_python()
    obs["gi_python"] = {"chosen": py_gi, "tried": tried}
    try:
        if py_gi is None:
            obs["fatal"] = "no python on this host can import gi + WebKit2 4.1"
            return 0

        dists = Path(args.dist_in)
        mpath = dists / "build_manifest.json"
        if not mpath.is_file():
            obs["fatal"] = (f"no build manifest at {mpath}: the build job produced none, so there "
                            f"is nothing to measure and this is a non-result, not a finding")
            return 0
        manifest = json.loads(mpath.read_text())
        obs["build"] = manifest
        obs["states"] = {n: dict(s) for n, s in manifest["states"].items()}

        # ── ONE BACKEND, INSTALLED ONCE ───────────────────────────────────────────────────────
        shared = work / "backend"
        obs["shared_backend"] = clone_at(args.repo, BASE_REF, shared)
        if not obs["shared_backend"].get("checkout_ok"):
            obs["fatal"] = (f"the shared backend checkout asked for {BASE_REF[:9]} and landed on "
                            f"{str(obs['shared_backend'].get('commit'))[:9]}")
            return 0
        shome = work / "home_shared"
        shome.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        inst = sh(["bash", "install.sh", "--local", *args.install_arg], cwd=str(shared),
                  timeout=args.install_timeout, env={"UNSLOTH_STUDIO_HOME": str(shome)})
        inst["seconds"] = round(time.time() - t0, 1)
        inst["args"] = ["--local", *args.install_arg]
        inst["stdout"] = (inst.get("stdout") or "")[-6000:]
        obs["shared_install"] = inst
        binp = None
        for c in [shome / "bin" / "unsloth", *sorted(shome.glob(".venv*/bin/unsloth"))]:
            if c.exists() and os.access(c, os.X_OK):
                binp = str(c)
                break
        obs["shared_unsloth_bin"] = binp
        obs["zoo"] = zoo_identity(shome)
        if not binp:
            obs["fatal"] = (f"the shared backend install produced no unsloth CLI "
                            f"(rc={inst.get('rc')})")
            return 0
        print(f"== shared backend installed in {inst['seconds']}s: {binp}", flush=True)
        print(f"== unsloth-zoo: {obs['zoo'].get('version')} {obs['zoo'].get('commit_id')}",
              flush=True)

        # ── THE INSTRUMENT, cloned once, hashed, outside every arm's tree ─────────────────────
        instrument = work / "instrument"
        obs["instrument"] = clone_at(args.repo, INSTRUMENT_REF, instrument)
        obs["instrument"]["path"] = str(instrument)
        if not obs["instrument"].get("checkout_ok"):
            obs["fatal"] = (f"the instrument checkout asked for {INSTRUMENT_REF[:9]} and landed "
                            f"on {str(obs['instrument'].get('commit'))[:9]}")
            return 0
        sb = instrument / "tests" / "studio" / "studiobench"
        obs["instrument"]["studiobench_present"] = sb.is_dir()
        if not sb.is_dir():
            obs["fatal"] = f"the pinned instrument has no studiobench package at {sb}"
            return 0
        ihash = tree_hash(sb)
        obs["instrument"]["studiobench_hash"] = ihash
        print(f"== instrument at {obs['instrument']['commit'][:9]} hash={ihash}", flush=True)

        # Each arm becomes a state the measure path understands: its OWN dist, the SHARED bin and
        # home. The bundle hash is re-read here and compared with the build job's, so a dist that
        # changed in transit shows up instead of passing silently.
        usable = []
        for name in ARM_ORDER:
            st = obs["states"].get(name) or {}
            d = dists / name
            st["dist"] = {"path": str(d), "exists": d.is_dir(),
                          "index_html": (d / "index.html").is_file(),
                          "asset_files": len(list((d / "assets").rglob("*")))
                          if (d / "assets").is_dir() else 0}
            st["unsloth_bin"] = binp
            st["home"] = str(shome)
            st["bundle_hash_at_measure"] = bundle_hash(d) if d.is_dir() else None
            st["bundle_hash_matches_build"] = (
                st.get("exported_bundle_hash") == st["bundle_hash_at_measure"])
            obs["states"][name] = st
            if st["dist"]["index_html"] and st.get("patch_ok"):
                usable.append(name)
            else:
                print(f"== arm {name} has no usable dist", flush=True)
        obs["arms_built"] = usable
        obs["arms_not_built"] = [n for n in ARM_ORDER if n not in usable]
        if len(usable) < 2:
            obs["fatal"] = (f"a two-arm differential needs two arms and {usable} arrived with a "
                            f"usable dist")
            return 0

        # ── THE PLAN ──────────────────────────────────────────────────────────────────────────
        #
        # Both arms ADJACENT at each rung, which keeps machine drift out of the arm difference,
        # and the ARM ORDER SWAPPED on alternate repetitions, which keeps "whichever arm went
        # first" out of it. Rungs interleaved within a repetition rather than run to completion
        # one after the other, so a host stall cannot sit entirely on one rung either.
        #
        # 100K first inside each repetition: it is the cheaper rung and the one the 5.02x was
        # measured on, so a run cut short still answers the question it was asked.
        plan: list[tuple[str, str, str]] = []
        for rep in range(1, max(reps.values()) + 1):
            order = list(ARM_ORDER) if rep % 2 == 1 else list(reversed(ARM_ORDER))
            for rung in rungs:
                if rep > reps.get(rung, 0):
                    continue
                for arm in order:
                    plan.append((rung, str(rep), arm))
        obs["plan"] = [{"rung": r, "rep": p, "arm": a} for r, p, a in plan]
        obs["runs"] = []

        def flush() -> None:
            """After every session, not once at the end.

            The end used to be the only write, inside a `finally`: when a workflow step timeout
            fired, the process tree was killed, the `finally` never ran, and hours of exclusive
            GPU produced a zero-byte result with nothing to re-score offline. A partial artifact
            is a first-class result here.
            """
            try:
                args.out.write_text(json.dumps(obs, indent=2))
            except Exception:                                            # noqa: BLE001
                pass

        port = args.first_port
        for rung, rep, arm in plan:
            obs["runs"].append(measure(work, obs["states"][arm], arm, rung, rep, port,
                                       xinfo["display"], py_gi, args.rung_timeout,
                                       instrument, ihash, args.hog_ms, args.hog_period_ms))
            flush()
            # A fresh port per session, monotonically. Not tidiness: assets cache on ORIGIN plus
            # PATH, and vite content-hashes on SOURCE, so a rebuilt bundle can keep its URL and
            # be served from the previous arm's bytes.
            port += 1
            time.sleep(8)

            # EARLY BAIL. The plan puts both arms at the cheaper rung first precisely so an arm
            # that cannot be driven at all shows up in the first two sessions, rather than after
            # another two hours of exclusive GPU proving the same thing ten more times.
            if len(obs["runs"]) == 2 and not any(
                    (r.get("payload") or {}).get("ok") for r in obs["runs"]):
                obs["fatal"] = (
                    "neither arm completed a session at the first rung, so the remaining sessions "
                    "were not attempted: "
                    + "; ".join(f"{r['arm']}/{r['rung']}: "
                                f"{str((r.get('payload') or {}).get('error'))[:160]}"
                                for r in obs["runs"]))
                flush()
                break

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
            # By PID. `pkill -f Xvfb` would match this probe's own command line, and an X server
            # left running outlives the job and holds VRAM on a machine three other slots share.
            try:
                os.kill(xproc.pid, signal.SIGTERM)
                time.sleep(2)
                if xproc.poll() is None:
                    os.kill(xproc.pid, signal.SIGKILL)
            except Exception:                                            # noqa: BLE001
                pass
            obs["xserver"]["stopped_pid"] = xproc.pid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--state", default="host")
    ap.add_argument("--checkout", default="")
    ap.add_argument("--mode", default="measure", choices=["build", "measure"],
                    help="build: ONE clone, both frontends, no GPU and no display. measure: ONE "
                         "backend install, then drive each arm's prebuilt dist. They are separate "
                         "jobs because building holds the exclusive GPU group for work that does "
                         "not touch the GPU.")
    ap.add_argument("--dist-out", default="", help="where build mode writes the per-arm dists")
    ap.add_argument("--dist-in", default="", help="where measure mode reads them from")
    ap.add_argument("--work", default=os.environ.get("AMD_CI_WORK", "/tmp/studio_r9695"))
    ap.add_argument("--repo", default="https://github.com/unslothai/unsloth")
    ap.add_argument("--rungs", default="100K,500K")
    ap.add_argument("--reps", default="100K:5,500K:5")
    ap.add_argument("--first-port", type=int, default=5601)
    ap.add_argument("--install-timeout", type=int, default=5400)
    ap.add_argument("--rung-timeout", type=int, default=1800)
    ap.add_argument("--install-arg", action="append", default=[],
                    help="extra argument for install.sh, applied to BOTH arms and to the shared "
                         "backend. For off-runner rehearsals only; the measured run passes none")
    ap.add_argument("--hog-ms", type=int, default=200)
    ap.add_argument("--hog-period-ms", type=int, default=250)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    work = Path(args.work) / "r9695"
    work.mkdir(parents=True, exist_ok=True)
    rungs = [r.strip() for r in args.rungs.split(",") if r.strip()]
    reps = parse_reps(args.reps, rungs)
    obs: dict = {"mode": args.mode, "rungs": rungs, "reps": reps, "base_ref": BASE_REF,
                 "head_marker": HEAD_MARKER, "patch": PATCH,
                 "arms": {k: list(v) for k, v in ARMS.items()},
                 "arm_order": list(ARM_ORDER), "repo": args.repo,
                 "scene": str(LADDER / "r9695_scene.js"),
                 "bench": str(LADDER / "final_rung_bench.py"),
                 "install_args": ["--local", *args.install_arg],
                 "hog": {"ms": args.hog_ms, "period_ms": args.hog_period_ms},
                 "patch_bytes": {p.name: p.stat().st_size
                                 for p in sorted(LADDER.glob("r9695*.patch"))}}
    try:
        if args.mode == "build":
            return do_build(args, obs, work)
        return do_measure(args, obs, work, rungs, reps)
    finally:
        args.out.write_text(json.dumps(obs, indent=2))


if __name__ == "__main__":
    sys.exit(main())

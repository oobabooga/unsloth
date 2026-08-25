#!/usr/bin/env python3
"""Probe: WHICH PIECE of PR 9477 owns the 4.981x on `action:reasoning_toggle_all`?

Observes only. criteria/studio_p123_attribution.py judges.

WHAT IS BEING SPLIT. At r100K, on the gesture that opens EVERY reasoning pane, upstream main
(`90f85fdbf`) reads 5.1 effective fps at 93% busy and 4,746 ms blocked. Arm `N111` -- all of 9477
with `REASONING_PAGINATION_ENABLED = false` -- reads 25.3 fps at 60% busy. That 4.981x is one lump
and piece 4 cannot be in it, because the flag is off in every arm here. So it belongs to piece 1
(streamed content fidelity), piece 2 (the streaming render schedule and text presentation) and/or
piece 3 (the streaming code policy).

WHY THIS IS RECONCILIATION AND NOT DISCOVERY. The branch already carries an ablation, in
`6adc583a2`: at 250K STREAMED reasoning characters on WebKitGTK 2.50.4, worst main-thread freeze
went base 6,821 ms, null 6,878, the streaming-render rewrite alone 869, plus the plain-code policy
211, plus pagination 79. On that window piece 2 dominates and piece 3 is a clear second.

But that is a STREAMING window and this is a SETTLED one: a thread that finished streaming long
ago, whose panes are then opened. This campaign has already been caught once by exactly that
distinction -- `reasoning_toggle` names a one-pane gesture and an all-panes gesture, and pagination
reads 1.9% on one and 1.913x on the other, both correctly. So the question here is not "what is the
ordering" but "does the published ordering reproduce on this window", and both orderings are
reported side by side rather than one being assumed to carry the other.

THE DESIGN IS A FULL 2^3 FACTORIAL. Eight corners plus upstream main. A subtractive-only design
(remove one piece from the whole) is blind to redundancy: if pieces 2 and 3 are each independently
sufficient, removing either alone changes nothing and the design reports "no piece matters", which
would be false. An additive-only design is blind to a piece that only pays off in company. All
eight corners give every piece FOUR one-piece-at-a-time isolations, one per context of the other
two, in the same spirit as the p4 probe's single one-literal flip.

Piece 3 is carried as TWO mechanisms, because reading the source shows it is two, with very
different user-visible costs: `3a` is `reasoning.tsx` passing `codeHighlighting="plain"`, which
makes every reasoning fence permanently plain AT ANY SIZE and is not a cap at all; `3b` is the pair
of size thresholds in `streaming-code-policy.ts`. Piece 1 sits outside the factorial because it is
a stream-time mechanism and this window is a settled thread, so it gets one isolation rather than
four. Both choices are budget decisions and both are stated here rather than buried.

    arm     p1   p2   p3a  p3b   built as
    main    -    -    -    -     90f85fdbf, no patches
    Z       off  off  off  off   C_tip + all four off-patches      <- CLOSURE CONTROL
    A000    on   off  off  off   C_tip + p2_off + p3a_off + p3b_off
    A100    on   ON   off  off   C_tip + p3a_off + p3b_off
    A010    on   off  ON   off   C_tip + p2_off + p3b_off
    A001    on   off  off  ON    C_tip + p2_off + p3a_off
    A110    on   ON   ON   off   C_tip + p3b_off
    A101    on   ON   off  ON    C_tip + p3a_off
    A011    on   off  ON   ON    C_tip + p2_off
    A111    on   ON   ON   ON    C_tip alone                       <- the 4.981x endpoint

`Z` is the gate that can falsify the whole exercise: with all four mechanisms neutralised it should
behave like `main`. If it does not, something in 9477 that none of the four patches turns off is
carrying part of the lump, and every per-mechanism share below is a share of less than the whole.
The criteria reports that as a number rather than asserting it away. `Z -> A000` is piece 1's only
isolation, and it is an additive one.

EVERY ARM IS ONE UPSTREAM COMMIT PLUS PATCHES. Unlike the p4 probe, nothing here needs a commit
that lives only on a pull request, so `refs/pull/9477/head` is not fetched and no arm can silently
land on a default branch tip. The `HEAD == ref` assertion is kept anyway and is reported as
"asked X, landed Y".

THE JAMMED POSITIVE CONTROL RUNS AT EVERY RUNG, in the same loop as the arms. `--hog-ms 200
--hog-period-ms 250` blocks the page's main thread on a timer. The p4 run took this control
globally -- `ctrl_ok or drop >= 25` -- so r100K's working control licensed r500K, where the JAMMED
arm read 0.14 fps against the clean arm's 0.08 and the ratios were therefore computed inside a
channel that could not resolve anything. The criteria here gates each rung by its own control, on
each channel separately.

`--frame-clock passive`, not `updating`: `begin_updating()` drives the clock itself and read 60.0
fps with the main thread 80% blocked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LADDER = HERE / "studio_ladder"

#: ONE upstream commit under every arm. This is the merge base the 4.981x was measured against.
BASE_REF = "90f85fdbf8fd6f9df1b99aff44f1e45da4c808d0"

#: THE INSTRUMENT: a separate checkout of the same commit, used to drive EVERY arm and never any
#: arm's own tree. `amdv_rung_bench.py` imports the pacer, seeder and frozen corpus from
#: `--sb-root`; handing it the arm's own repo would let the measuring device co-vary with the
#: subject. Hashed as well as path-checked, so "one instrument" is a fact about content.
INSTRUMENT_REF = BASE_REF

#: PIECE 3 IS TWO MECHANISMS, NOT ONE, and they have completely different user-visible costs.
#: Splitting them is not fussiness: reading the source establishes that
#:
#:   3a  `reasoning.tsx` passes `codeHighlighting="plain"`, so EVERY fence inside a reasoning
#:       pane is permanently plain AT ANY SIZE. Main renders a bare `<MarkdownText />` and the
#:       context defaults to "syntax". This is not a cap at all.
#:   3b  `isOversizedStreamingCode` sends a fence down the plain-first path at
#:       `OVERSIZED_OPEN_CODE_CHARS` = 4 KiB -- NOT the 16 KiB constant the campaign has been
#:       quoting -- and `shouldAutoHighlightStreamingCode` then refuses the later upgrade to a
#:       highlighted subtree above `MAX_AUTO_HIGHLIGHT_SOURCE_CODE_UNITS` = 16 KiB.
#:
#: On the gesture under test the panes being opened are reasoning panes, so 3a and 3b act on
#: different fences and folding them into one arm would price whichever happens to dominate and
#: attribute it to both.
#:
#: PIECE 1 SITS OUTSIDE THE FACTORIAL, on a budget argument that is stated rather than hidden.
#: It is a STREAM-TIME mechanism (chat adapter, assistant-content parsing) and this window is a
#: SETTLED thread that finished streaming before the measurement began, so it is the least likely
#: of the four to carry the lump. Sixteen corners would price it with four isolations each; ten
#: arms price the three live candidates with four each and piece 1 with one, which is enough to
#: show whether it moves anything at all.
FACTORS = ("p2", "p3a", "p3b")
PIECES = ("p1",) + FACTORS
#: the patch that turns each mechanism OFF, applied on top of C_tip.patch
OFF_PATCH = {"p1": "p1_off.patch", "p2": "p2_off.patch",
             "p3a": "p3a_off.patch", "p3b": "p3b_off.patch"}
#: the sentinel each of those patches plants, grepped back out of the built tree. A patch that
#: applied to the wrong hunk still reports rc=0, and this is the one fact the run turns on.
SENTINEL = {"p1": "AMDCI_PIECE1_NEUTRALISED",
            "p2": "AMDCI_PIECE2_NEUTRALISED",
            "p3a": "AMDCI_PIECE3A_NEUTRALISED",
            "p3b": "AMDCI_PIECE3B_NEUTRALISED"}

#: the eight corners of the 2^3 over FACTORS. Piece 1 is ON at every corner.
MASKS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
         (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))
REFERENCE = "main"
#: everything 9477 does, neutralised. Should behave like `main`; if it does not, the patches do
#: not span the change and every share below is a share of less than the whole.
CLOSURE = "Z"


def arm_of(mask) -> str:
    return "A" + "".join(str(b) for b in mask)


def _arms() -> "dict[str, tuple[str, tuple[str, ...]]]":
    out: dict[str, tuple[str, tuple[str, ...]]] = {REFERENCE: (BASE_REF, ())}
    for mask in MASKS:
        patches = ["C_tip.patch"]
        # deterministic order, so an arm is a function of its piece set and nothing else
        for i, piece in enumerate(FACTORS):
            if mask[i] == 0:
                patches.append(OFF_PATCH[piece])
        out[arm_of(mask)] = (BASE_REF, tuple(patches))
    out[CLOSURE] = (BASE_REF, ("C_tip.patch", "p1_off.patch", "p2_off.patch",
                               "p3a_off.patch", "p3b_off.patch"))
    return out


ARMS = _arms()
#: `Z` early, right after `main`: it is the arm most likely to expose a patch-composition problem
#: (it stacks all four neutralisations), and finding that out before eight installs is cheap.
ARM_ORDER = (REFERENCE, CLOSURE) + tuple(arm_of(m) for m in MASKS)

EMPTY_STATE = {
    "patch_ok": False, "dirty_files": 0, "piece_mask": None, "pagination_literal": None,
    "install": {"rc": None}, "unsloth_bin": None, "home": None, "repo_root": None,
    "dist": {"path": None, "exists": False, "index_html": False, "asset_files": 0},
}

sys.path.insert(0, str(HERE))
from webkit_paint_probe import fetch_xvfb, inventory, start_xserver  # noqa: E402
from studio_ladder_probe import gi_python, sh  # noqa: E402


def clone_at(repo_url: str, ref: str, dest: Path) -> dict:
    """Clone, then land on `ref` exactly, and record ASKED versus LANDED either way.

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
    out["ref_requested"] = ref
    out["ref_landed"] = out["commit"]
    out["checkout_ok"] = out["commit"] == ref
    return out


def tree_hash(root: Path) -> str:
    """Content hash of a directory tree. Used on the instrument, so that "one instrument for
    every arm" is checkable by CONTENT and not only by resolved path: a path assertion cannot
    see a tree that changed under it between the first arm and the ninth."""
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def read_piece_mask(root: Path) -> tuple:
    """Read each piece's ON/OFF state back out of the SOURCE the arm was built from.

    Not from the patch list: a patch that applied to the wrong hunk still returns rc=0, and the
    whole factorial turns on each arm really carrying the piece set it is labelled with. Each
    `*_off.patch` plants a sentinel comment; a piece is ON when its sentinel is ABSENT.
    """
    src = root / "studio" / "frontend" / "src"
    found = {p: False for p in PIECES}
    if src.is_dir():
        for f in src.rglob("*.ts*"):
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception:                                            # noqa: BLE001
                continue
            for piece in PIECES:
                if SENTINEL[piece] in text:
                    found[piece] = True
    return tuple(not found[p] for p in PIECES)


def build_state(work: Path, name: str, repo_url: str, ref: str, patches: tuple[str, ...],
                install_timeout: int) -> dict:
    """One checkout, patched in order, installed. EVERY return path carries the full key set, so
    a build failure fails this arm's gate rather than aborting the run and discarding the arms
    that did build."""
    root = work / f"state_{name}"
    out: dict = {"name": name, "ref": ref, "patches": list(patches), **EMPTY_STATE}
    out["patch_steps"] = []
    out.update(clone_at(repo_url, ref, root))

    ok = out.get("checkout_ok", False)
    if not ok:
        out["why"] = (f"asked for {ref[:9]} and landed on {str(out.get('commit'))[:9]}; "
                      f"this arm was NOT built")
        return out
    for p in patches:
        path = LADDER / p
        if not path.is_file():
            out["patch_steps"].append({"patch": p, "missing": str(path)})
            ok = False
            break
        # --check BEFORE --apply: a half-applied patch yields an arm that is neither state and
        # installs and measures perfectly happily.
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
    out["piece_mask"] = list(read_piece_mask(root))

    # Piece 4 must be OFF on every arm here, by construction. Read it back rather than trust it:
    # an arm that silently shipped pagination would put piece 4 back into the lump being split.
    flagf = (root / "studio" / "frontend" / "src" / "components" / "assistant-ui"
             / "thread-feature-flags.ts")
    if flagf.is_file():
        for line in flagf.read_text(encoding="utf-8", errors="replace").splitlines():
            if "REASONING_PAGINATION_ENABLED" in line and ("true" in line or "false" in line):
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
            py_gi: str, timeout: int, instrument: Path, instrument_hash: str,
            hog_ms: int = 0) -> dict:
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
           "--sb-root", str(instrument), "--unsloth-bin", st["unsloth_bin"],
           "--python-gi", py_gi,
           # `amdv_scene_p4.js` plus ONE addition: a settled fidelity census taken after every
           # measured window. The measured windows themselves are byte-identical to p4's.
           "--scene", str(LADDER / "amdv_scene_p123.js"),
           "--driver", str(LADDER / "amdv_drive.py"),
           "--frame-clock", "passive",
           "--skip-send",
           "--instrument-hash", instrument_hash,
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


def parse_reps(spec: str, rungs: list[str]) -> dict:
    """`--reps 100K:3,500K:2` or a bare `--reps 2`.

    Per rung, because the rungs cost very different amounts: an r100K measurement takes ~40 s and
    an r500K one ~125 s, and the rung that carries the verdict is the one whose repetition spread
    has to be believable. Spending the budget evenly would buy precision where it is not needed
    and leave the scoring rung with two readings that once disagreed 44.9 against 26.4 fps.
    """
    out = {}
    if ":" not in spec:
        return {r: int(spec) for r in rungs}
    for part in spec.split(","):
        k, _, v = part.partition(":")
        out[k.strip()] = int(v)
    return {r: out.get(r, 2) for r in rungs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--state", default="host")
    ap.add_argument("--checkout", default="")
    ap.add_argument("--work", default=os.environ.get("AMD_CI_WORK", "/tmp/studio_p123"))
    ap.add_argument("--repo", default="https://github.com/unslothai/unsloth")
    ap.add_argument("--rungs", default="100K,500K")
    ap.add_argument("--reps", default="100K:3,500K:2")
    ap.add_argument("--first-port", type=int, default=5461)
    ap.add_argument("--install-timeout", type=int, default=3600)
    ap.add_argument("--rung-timeout", type=int, default=2400)
    ap.add_argument("--hog-ms", type=int, default=200)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    work = Path(args.work) / "p123"
    work.mkdir(parents=True, exist_ok=True)
    rungs = [r.strip() for r in args.rungs.split(",") if r.strip()]
    reps = parse_reps(args.reps, rungs)
    obs: dict = {"rungs": rungs, "reps": reps, "base_ref": BASE_REF,
                 "pieces": list(PIECES), "sentinels": SENTINEL,
                 "arms": {k: list(v[1]) for k, v in ARMS.items()},
                 "arm_refs": {k: v[0] for k, v in ARMS.items()},
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

        # PRE-FLIGHT, BEFORE ANY INSTALL. Nine installs are the expensive part of this job, and
        # every one of them is wasted if a patch does not apply. Each arm's patch stack is
        # dry-run against a throwaway checkout first, so a bad patch costs seconds rather than
        # the whole GPU hold, and it names which patch and which arm.
        pre = work / "preflight"
        obs["preflight"] = {"clone": clone_at(args.repo, BASE_REF, pre)}
        if not obs["preflight"]["clone"].get("checkout_ok"):
            obs["fatal"] = (f"the preflight checkout asked for {BASE_REF[:9]} and landed on "
                            f"{str(obs['preflight']['clone'].get('commit'))[:9]}")
            return 0
        stacks = {}
        for name in ARM_ORDER:
            _, patches = ARMS[name]
            sh(["git", "checkout", "--force", "--detach", BASE_REF], cwd=str(pre), timeout=300)
            sh(["git", "clean", "-fdx", "--", "studio"], cwd=str(pre), timeout=300)
            steps = []
            for p in patches:
                r = sh(["git", "apply", "--check", str(LADDER / p)], cwd=str(pre), timeout=300)
                if r.get("rc") == 0:
                    r = sh(["git", "apply", str(LADDER / p)], cwd=str(pre), timeout=300)
                steps.append({"patch": p, "rc": r.get("rc"),
                              "stderr": (r.get("stderr") or "")[:300]})
            mask = read_piece_mask(pre) if all(s["rc"] == 0 for s in steps) else None
            stacks[name] = {"steps": steps, "ok": all(s["rc"] == 0 for s in steps),
                            "piece_mask": list(mask) if mask else None}
            print(f"== preflight {name}: ok={stacks[name]['ok']} mask={stacks[name]['piece_mask']}",
                  flush=True)
        sh(["git", "checkout", "--force", "--detach", BASE_REF], cwd=str(pre), timeout=300)
        obs["preflight"]["stacks"] = stacks
        shutil.rmtree(pre, ignore_errors=True)
        bad = [n for n, v in stacks.items() if not v["ok"]]
        if bad:
            obs["fatal"] = (f"these arms' patch stacks do not apply on {BASE_REF[:9]}: {bad}. "
                            f"Nothing was installed.")
            print(f"== {obs['fatal']}", flush=True)
            return 0

        # THE INSTRUMENT, cloned once, before any arm, and hashed.
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
        ihash = tree_hash(sb)
        obs["instrument"]["studiobench_hash"] = ihash
        print(f"== instrument: {instrument} at {obs['instrument']['commit'][:9]} "
              f"studiobench_hash={ihash}", flush=True)

        obs["states"] = {}
        for name in ARM_ORDER:
            ref, patches = ARMS[name]
            obs["states"][name] = build_state(work, name, args.repo, ref, patches,
                                              args.install_timeout)
            st = obs["states"][name]
            print(f"== built {name}: asked {ref[:9]} landed {str(st.get('commit'))[:9]} "
                  f"patch_ok={st.get('patch_ok')} mask={st.get('piece_mask')} "
                  f"pagination={st.get('pagination_literal')} "
                  f"install rc={(st.get('install') or {}).get('rc')}", flush=True)

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

        # ── THE JAMMED POSITIVE CONTROL, AT EVERY RUNG, IN THE SAME LOOP AS THE ARMS ─────────
        # A control taken only at the cheapest rung is not a control for the expensive one. The
        # p4 run measured it at both but SCORED it globally, which let a rung whose control read
        # backwards carry the verdict.
        control_arm = REFERENCE if REFERENCE in usable else usable[0]
        for rung in rungs:
            obs["runs"].append(measure(work, obs["states"][control_arm], "JAM", rung, "jam", port,
                                       xinfo["display"], py_gi, args.rung_timeout,
                                       instrument, ihash, hog_ms=args.hog_ms))
            port += 1

        # ── the arms, interleaved within each rep so drift cannot land on whoever ran last ───
        for rep in range(1, max(reps.values()) + 1):
            for rung in rungs:
                if rep > reps[rung]:
                    continue
                for arm in usable:
                    obs["runs"].append(measure(work, obs["states"][arm], arm, rung, str(rep),
                                               port, xinfo["display"], py_gi, args.rung_timeout,
                                               instrument, ihash))
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

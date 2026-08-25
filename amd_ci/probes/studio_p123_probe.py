#!/usr/bin/env python3
"""Probe: WHAT owns the 4.981x on `action:reasoning_toggle_all`, and can we have it for free?

Observes only. criteria/studio_p123_attribution.py judges.

WHAT IS BEING SPLIT. At r100K, on the gesture that opens EVERY reasoning pane, upstream main
(`90f85fdbf`) reads 5.1 effective fps at 93% busy and 4,746 ms blocked. Arm `A11` -- all of PR 9477
with `REASONING_PAGINATION_ENABLED = false` -- reads 25.3 fps at 60% busy. That 4.981x is one lump
and nobody has split it.

Do NOT read that as "piece 4 excluded by construction". The flag being false stops
`selectReasoningMarkdownPage` and the Show more control, but `reasoning-scroll-pin.ts` is wired
unconditionally at `reasoning.tsx:291-296` and no flag gates it. It is inert on this gesture and
live during a stream, which is a claim about this window and not about the module.

RECONCILIATION, NOT DISCOVERY. The branch already carries an ablation, in `6adc583a2`: at 250K
STREAMED reasoning characters on WebKitGTK 2.50.4, worst main-thread freeze went base 6,821 ms,
null 6,878, the streaming-render rewrite alone 869, plus the plain-code policy 211, plus pagination
79. On THAT window piece 2 dominates. But that is a streaming window and this is a settled one: a
thread that finished streaming long ago, whose panes are then opened. This campaign has already
been caught once by exactly that distinction, when `reasoning_toggle` turned out to name a one-pane
gesture and an all-panes gesture and pagination read 1.9% on one and 1.913x on the other, both
correctly. So both orderings are reported side by side and neither is assumed to carry the other.

PIECE 3 IS TWO MECHANISMS, and its only entry point lives in `reasoning.tsx`, a file a naive
taxonomy assigns to piece 4. Any mapping that calls `reasoning.tsx` piece 4's file mis-assigns the
dominant mount-gesture mechanism.

    3a  `reasoning.tsx:344` passes `codeHighlighting="plain"`, so EVERY fence inside a reasoning
        pane is permanently plain AT ANY SIZE. Main renders a bare `<MarkdownText />` and the
        context defaults to "syntax". This is not a cap at all.
    3b  `isOversizedStreamingCode` sends a fence down the plain-first path at
        `OVERSIZED_OPEN_CODE_CHARS` = 4,096 -- NOT the 16 KiB constant this campaign has been
        quoting -- and `shouldAutoHighlightStreamingCode` then refuses the later upgrade above
        `MAX_AUTO_HIGHLIGHT_SOURCE_CODE_UNITS` = 16,384.

THE ARM THAT MATTERS MOST IS `M`, AND IT IS NOT ONE OF 9477'S MECHANISMS.

`code-fence-defer.tsx` is BYTE-IDENTICAL between main and this branch. The per-fence deferral
machinery -- an IntersectionObserver per fence, a ResizeObserver per nested-scroller fence, a
pre-paint `useLayoutEffect` doing `getComputedStyle` ancestor walks and a `getBoundingClientRect`
inside the commit that just inserted the whole trace, registration into a module-global `unreached`
set, a document-wide capturing scroll listener, and a jump path that walks `unreached` and issues
two `flushSync` renders -- is MAIN'S OWN CODE. 9477 does not touch a line of it.

What 9477 does is pass `codeHighlighting="plain"`, which makes `renderPlainBody` non-null, which
passes `enabled = false` into `useFenceReached` at `markdown-text.tsx:849-855`, which turns all of
that machinery off as a SIDE EFFECT while also removing every syntax colour. Those two costs have
never been told apart. `M` is `A00` plus one flip of that `enabled` argument: colours ON, machinery
OFF. If `M` captures the win, the speed is available with no fidelity cost at all and 9477's piece
3 can be dropped entirely. If it does not, the cost is genuinely the tokenizer and the trade is
real. Nobody has run this arm.

THE ARMS.

    arm    p1   p2   p3a  p3b  defer  built as
    main   -    -    -    -    on     90f85fdbf, no patches
    Z      off  off  off  off  on     C_tip + all four off-patches   <- CLOSURE CONTROL
    M      on   off  off  on   OFF    A00 + p3m_off                  <- the decoupling arm
    A00    on   off  off  on   on     C_tip + p2_off + p3a_off
    A10    on   ON   off  on   on     C_tip + p3a_off
    A01    on   off  ON   on   on     C_tip + p2_off
    A11    on   ON   ON   on   on     C_tip alone                    <- the 4.981x endpoint
    N1     on*  ON   ON   on   on     C_tip + p1_off   (* p1 off)    <- predicted null
    N3b    on   ON   ON   off  on     C_tip + p3b_off                <- predicted null

The 2^2 over (p2, p3a) gives each of the two live mechanisms TWO one-mechanism isolations, one in
each context of the other, so a mechanism whose effect depends on its company says so instead of
being averaged. A subtractive-only design cannot see redundancy: if p2 and p3a were each
independently sufficient, removing either alone would change nothing and that design would report
"nothing matters", which would be false.

The decomposition `M` buys:

    A00 -> M     the cost of MAIN's fence-deferral machinery, colours held ON
    M   -> A01   the additional cost of the colours, machinery held OFF
    A00 -> A01   both together, which is all `codeHighlighting="plain"` has ever been measured as

`Z` is the gate that can falsify the whole exercise: with all four of 9477's mechanisms
neutralised it should behave like `main`. If it does not, something in 9477 that none of these
patches turns off is carrying part of the lump, and every share below is a share of less than the
whole. That is reported as a number, not asserted away.

PIECE 1 AND PIECE 3b ARE PREDICTED NULLS, AND THE PREDICTIONS ARE RECORDED IN THE ARTIFACT BEFORE
THE RUN. See `PREDICTED_NULLS`. Writing them down first is what makes a null read as CONFIRMED
INERT rather than as a flip that quietly failed to take, and one of them falsifies the harness
rather than the piece if it comes out wrong.

WHY THE ARMS ARE NEUTRALISATIONS ON TOP OF C RATHER THAN ONE PIECE ADDED TO MAIN. Piece 2's
`streaming-render-schedule.ts` imports piece 3's `streaming-code-policy.ts` and
`fenced-code-provenance.ts`, so "piece 2 alone on main" would drag piece 3 in with it and would not
be one piece. Every arm here is C with something turned off, and the four off-patches are disjoint
BY FILE, so they compose in any order. All 16 subsets were verified to apply and the extreme
corners were built with the real production `tsc -b && vite build` before this was pushed.

EVERY ARM IS ONE UPSTREAM COMMIT PLUS PATCHES. Nothing here needs a commit that lives only on a
pull request, so `refs/pull/9477/head` is not fetched and no arm can silently land on a default
branch tip. The `HEAD == ref` assertion is kept anyway and is reported as "asked X, landed Y".

THE JAMMED POSITIVE CONTROL RUNS AT EVERY RUNG, in the same loop as the arms. `--hog-ms 200
--hog-period-ms 250` blocks the page's main thread on a timer. The p4 run took this control
GLOBALLY -- `ctrl_ok or drop >= 25` -- so r100K's working control licensed r500K, where the
deliberately jammed arm read 0.14 fps against the clean arm's 0.08 and every ratio was therefore
computed inside a channel that could not resolve anything. The criteria here gates each rung by its
own control, on each channel separately, and r500K is expected to come out VOID on that test.

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

#: The two mechanisms that can plausibly move a MOUNT gesture, swept as a 2^2 factorial.
FACTORS = ("p2", "p3a")
PIECES = ("p1", "p2", "p3a", "p3b")
#: the patch that turns each mechanism OFF, applied on top of C_tip.patch
OFF_PATCH = {"p1": "p1_off.patch", "p2": "p2_off.patch",
             "p3a": "p3a_off.patch", "p3b": "p3b_off.patch"}
#: the sentinel each of those patches plants, grepped back out of the built tree. A patch that
#: applied to the wrong hunk still reports rc=0, and this is the one fact the run turns on.
SENTINEL = {"p1": "AMDCI_PIECE1_NEUTRALISED",
            "p2": "AMDCI_PIECE2_NEUTRALISED",
            "p3a": "AMDCI_PIECE3A_NEUTRALISED",
            "p3b": "AMDCI_PIECE3B_NEUTRALISED"}
#: not one of 9477's mechanisms at all: it turns MAIN's fence-deferral machinery off while
#: leaving every colour in place. See `M` below.
DEFER_PATCH = "p3m_off.patch"
DEFER_SENTINEL = "AMDCI_FENCEDEFER_DISABLED"

MASKS = ((0, 0), (1, 0), (0, 1), (1, 1))
REFERENCE = "main"
CLOSURE = "Z"
#: the decoupling arm, and the one most likely to change what ships
DECOUPLE = "M"
BASELINE_ARM = "A00"
#: the two single-mechanism null checks, each one piece away from `A11`
NULL_P1 = "N1"
NULL_P3B = "N3b"


def arm_of(mask) -> str:
    return "A" + "".join(str(b) for b in mask)


def _arms() -> "dict[str, tuple[str, tuple[str, ...]]]":
    out: dict[str, tuple[str, tuple[str, ...]]] = {REFERENCE: (BASE_REF, ())}
    for mask in MASKS:
        patches = ["C_tip.patch"]
        for i, piece in enumerate(FACTORS):
            if mask[i] == 0:
                patches.append(OFF_PATCH[piece])
        out[arm_of(mask)] = (BASE_REF, tuple(patches))
    out[CLOSURE] = (BASE_REF, ("C_tip.patch", "p1_off.patch", "p2_off.patch",
                               "p3a_off.patch", "p3b_off.patch"))
    # `M` is `A00` plus exactly one thing, so `A00 -> M` prices the machinery and nothing else.
    out[DECOUPLE] = (BASE_REF, ("C_tip.patch", "p2_off.patch", "p3a_off.patch", DEFER_PATCH))
    # each is `A11` minus exactly one mechanism, which is what makes a null here a null about
    # that mechanism rather than about the arm
    out[NULL_P1] = (BASE_REF, ("C_tip.patch", "p1_off.patch"))
    out[NULL_P3B] = (BASE_REF, ("C_tip.patch", "p3b_off.patch"))
    return out


ARMS = _arms()
#: ORDER MATTERS, AND IT IS NOT ALPHABETICAL. Arms are measured in this order within every
#: repetition, so a run that is cut short loses the LAST arms rather than a random five. The first
#: five carry the entire recommendation:
#:
#:   main -> A11   the 4.981x lump itself
#:   A00  -> M     MAIN's fence-deferral machinery, colours held ON
#:   M    -> A01   the colours themselves, machinery held OFF
#:   A00  -> A01   both together, which is all `codeHighlighting="plain"` has been measured as
#:
#: `Z`, `A10`, `N1` and `N3b` are the closure control, the p2-in-context edge and the two
#: predicted nulls. All four are worth having and none of them changes what we would ship.
ARM_ORDER = (REFERENCE, BASELINE_ARM, DECOUPLE, arm_of((0, 1)), arm_of((1, 1)),
             arm_of((1, 0)), CLOSURE, NULL_P1, NULL_P3B)

#: PREDICTIONS, RECORDED BEFORE THE RUN so that a null reads as CONFIRMED INERT rather than as a
#: flip that quietly failed to take. Both are mechanistic, both are falsifiable, and one of them
#: falsifies the harness rather than the piece if it comes out wrong.
PREDICTED_NULLS = {
    NULL_P1: ("p1 must read the same as A11. Piece 1 executes only inside the chat adapter's SSE "
              "loop, and this thread is SEEDED through PUT /api/chat/threads/{id}/messages as "
              "finished parts, so that loop never runs. A NON-null here does not mean piece 1 "
              "helps; it means the harness is streaming when it believes it is not, and it "
              "falsifies the harness rather than the piece."),
    NULL_P3B: ("p3b must read the same as A11. The largest fenced code block anywhere in this "
               "thread is 2,781 characters, against OVERSIZED_OPEN_CODE_CHARS = 4,096 and "
               "MAX_AUTO_HIGHLIGHT_SOURCE_CODE_UNITS = 16,384, so both predicates already sit at "
               "their permissive value for every fence in the corpus and neutralising them cannot "
               "change a single render. The settled fence census in the artifact is what makes "
               "this checkable rather than asserted."),
}

EMPTY_STATE = {
    "patch_ok": False, "dirty_files": 0, "piece_mask": None, "fence_defer_enabled": None,
    "pagination_literal": None,
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


def bundle_hash(dist: Path) -> str:
    """Content hash of a built frontend bundle. The SAME definition `amdv_rung_bench.py` uses, so
    the hash recorded at build time and the one recorded at measure time are comparable and a
    dist that changed in between would show up rather than pass silently."""
    h = hashlib.sha256()
    n = 0
    for f in sorted((dist / "assets").rglob("*")) if (dist / "assets").is_dir() else []:
        if f.is_file() and f.suffix in (".js", ".css"):
            h.update(f.name.encode())
            h.update(f.read_bytes())
            n += 1
    idx = dist / "index.html"
    if idx.exists():
        h.update(idx.read_bytes())
    return f"{h.hexdigest()[:16]}({n} files)"


def tree_hash(root: Path) -> str:
    """Content hash of a directory tree. Used on the instrument, so that "one instrument for
    every arm" is checkable by CONTENT and not only by resolved path: a path assertion cannot
    see a tree that changed under it between the first arm and the ninth."""
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def read_sentinels(root: Path) -> dict:
    """Read every mechanism's ON/OFF state back out of the SOURCE the arm was built from.

    Not from the patch list: a patch that applied to the wrong hunk still returns rc=0, and the
    whole design turns on each arm really carrying the mechanism set it is labelled with. Each
    patch plants a sentinel comment; a mechanism is ON when its sentinel is ABSENT.
    """
    src = root / "studio" / "frontend" / "src"
    wanted = dict(SENTINEL)
    wanted["_defer"] = DEFER_SENTINEL
    found = {k: False for k in wanted}
    if src.is_dir():
        for f in src.rglob("*.ts*"):
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception:                                            # noqa: BLE001
                continue
            for key, needle in wanted.items():
                if needle in text:
                    found[key] = True
    return {"piece_mask": [not found[p] for p in PIECES],
            # main's own fence-deferral machinery, which 9477 does not touch: `code-fence-defer`
            # is byte-identical between the two trees. It is ON everywhere except arm `M`.
            "fence_defer_enabled": not found["_defer"]}


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
    out.update(read_sentinels(root))

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
    out["bundle_hash"] = bundle_hash(dist)
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


def do_build(args, obs: dict, work: Path) -> int:
    """BUILD MODE. No GPU, no display, no measurement. Runs in its own job.

    Every arm differs from every other ONLY inside `studio/frontend/src`: outside the frontend,
    `C_tip.patch` touches five TEST files and no backend source, and all four off-patches touch
    nothing but `studio/frontend/src`. So an arm's entire identity is its built `dist`, and the
    backend the measuring job installs can be any one of them. That is asserted below rather than
    assumed, because it is the premise the whole split rests on.

    This also removes a confound rather than only saving time. Nine installs in the measuring job
    meant nine chances for an install to differ between arms; one install for all of them means an
    install flake can no longer land on one arm and be read as that arm's mechanism.
    """
    # PRE-FLIGHT. Every arm's patch stack, dry-run against one throwaway checkout, before any
    # install. Cheap to ask, and it names the arm and the patch.
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
            steps.append({"patch": p, "rc": r.get("rc"), "stderr": (r.get("stderr") or "")[:300]})
        ok = all(s["rc"] == 0 for s in steps)
        sent = read_sentinels(pre) if ok else {}
        # THE PREMISE, CHECKED. Nothing may differ outside the frontend, or one dist cannot stand
        # for one arm and the measuring job's single backend install would silently erase a
        # difference that mattered.
        changed = [l[3:] for l in (sh(["git", "status", "--porcelain"], cwd=str(pre),
                                      timeout=60).get("stdout") or "").splitlines() if l.strip()]
        outside = [c for c in changed if not c.startswith("studio/frontend/")]
        stacks[name] = {"steps": steps, "ok": ok, "piece_mask": sent.get("piece_mask"),
                        "fence_defer_enabled": sent.get("fence_defer_enabled"),
                        "changed_outside_frontend": outside}
        print(f"== preflight {name}: ok={ok} mask={stacks[name]['piece_mask']} "
              f"defer={stacks[name]['fence_defer_enabled']} outside={outside}", flush=True)
    sh(["git", "checkout", "--force", "--detach", BASE_REF], cwd=str(pre), timeout=300)
    obs["preflight"]["stacks"] = stacks
    shutil.rmtree(pre, ignore_errors=True)
    bad = [n for n, v in stacks.items() if not v["ok"]]
    if bad:
        obs["fatal"] = (f"these arms' patch stacks do not apply on {BASE_REF[:9]}: {bad}. "
                        f"Nothing was built.")
        print(f"== {obs['fatal']}", flush=True)
        return 0

    dists = Path(args.dist_out)
    dists.mkdir(parents=True, exist_ok=True)
    obs["states"] = {}
    for name in ARM_ORDER:
        ref, patches = ARMS[name]
        # A build failure fails THIS ARM's gate and the loop continues. Aborting the run on the
        # first bad arm is what threw away three good arms earlier in this campaign.
        try:
            st = build_state(work, name, args.repo, ref, patches, args.install_timeout)
        except Exception as e:                                           # noqa: BLE001
            st = {"name": name, "ref": ref, "patches": list(patches), **EMPTY_STATE,
                  "why": f"{type(e).__name__}: {e}"}
        obs["states"][name] = st
        if (st.get("dist") or {}).get("index_html"):
            dest = dists / name
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(st["dist"]["path"], dest)
            st["exported_dist"] = str(dest)
            st["exported_bundle_hash"] = bundle_hash(dest)
        print(f"== built {name}: asked {ref[:9]} landed {str(st.get('commit'))[:9]} "
              f"patch_ok={st.get('patch_ok')} mask={st.get('piece_mask')} "
              f"defer={st.get('fence_defer_enabled')} "
              f"pagination={st.get('pagination_literal')} "
              f"bundle={st.get('exported_bundle_hash')} "
              f"install rc={(st.get('install') or {}).get('rc')}", flush=True)
        # the arm's own tree is large and there are nine of them on a shared disk
        shutil.rmtree(st.get("repo_root") or "/nonexistent", ignore_errors=True)
        shutil.rmtree(st.get("home") or "/nonexistent", ignore_errors=True)

    manifest = {
        "base_ref": BASE_REF, "arms": {k: list(v[1]) for k, v in ARMS.items()},
        "arm_order": list(ARM_ORDER), "pieces": list(PIECES),
        "predicted_nulls": PREDICTED_NULLS,
        "states": {n: {k: v for k, v in s.items()
                       if k in ("ref", "ref_landed", "commit", "checkout_ok", "patches",
                                "patch_ok", "patch_steps", "piece_mask", "fence_defer_enabled",
                                "pagination_literal", "dirty_files", "why",
                                "exported_dist", "exported_bundle_hash", "dist")}
                   for n, s in obs["states"].items()},
        "install": {n: {"rc": (s.get("install") or {}).get("rc"),
                        "seconds": (s.get("install") or {}).get("seconds")}
                    for n, s in obs["states"].items()},
    }
    (dists / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
    obs["build_manifest"] = str(dists / "build_manifest.json")
    obs["arms_built"] = [n for n, s in obs["states"].items() if s.get("exported_dist")]
    obs["arms_not_built"] = [n for n in ARM_ORDER if n not in obs["arms_built"]]
    print(f"== built {len(obs['arms_built'])}/{len(ARM_ORDER)}: {obs['arms_built']}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--state", default="host")
    ap.add_argument("--checkout", default="")
    ap.add_argument("--mode", default="measure", choices=["build", "measure"],
                    help="build: clone, patch and build every arm's frontend, no GPU and no "
                         "display. measure: install ONE backend and drive each arm's prebuilt "
                         "dist. They are separate jobs because building holds the exclusive GPU "
                         "group for work that does not touch the GPU.")
    ap.add_argument("--dist-out", default="", help="where build mode writes the per-arm dists")
    ap.add_argument("--dist-in", default="", help="where measure mode reads them from")
    ap.add_argument("--work", default=os.environ.get("AMD_CI_WORK", "/tmp/studio_p123"))
    ap.add_argument("--repo", default="https://github.com/unslothai/unsloth")
    ap.add_argument("--rungs", default="100K")
    ap.add_argument("--reps", default="100K:5")
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
    obs: dict = {"mode": args.mode, "rungs": rungs, "reps": reps, "base_ref": BASE_REF,
                 "pieces": list(PIECES), "sentinels": SENTINEL,
                 "defer_sentinel": DEFER_SENTINEL,
                 "predicted_nulls": PREDICTED_NULLS,
                 "arms": {k: list(v[1]) for k, v in ARMS.items()},
                 "arm_order": list(ARM_ORDER),
                 "arm_refs": {k: v[0] for k, v in ARMS.items()},
                 "patch_bytes": {p.name: p.stat().st_size
                                 for p in sorted(LADDER.glob("*.patch"))}}

    if args.mode == "build":
        try:
            return do_build(args, obs, work)
        finally:
            args.out.write_text(json.dumps(obs, indent=2))

    # ── MEASURE MODE ─────────────────────────────────────────────────────────────────────────
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

        dists = Path(args.dist_in)
        mpath = dists / "build_manifest.json"
        if not mpath.is_file():
            obs["fatal"] = (f"no build manifest at {mpath}: the build job did not produce one, so "
                            f"there is nothing to measure and this is a non-result rather than a "
                            f"finding")
            return 0
        manifest = json.loads(mpath.read_text())
        obs["build"] = manifest
        # The build job's integrity facts are carried through verbatim so the criteria's gates
        # read them exactly as if the build had happened here.
        obs["states"] = {n: dict(s) for n, s in manifest["states"].items()}

        # ONE BACKEND, INSTALLED ONCE. Every arm differs only inside `studio/frontend/src`, which
        # the build job asserted per arm, so the backend is common and installing it nine times
        # would only add nine chances for one arm's install to differ from another's.
        shared = work / "backend"
        obs["shared_backend"] = clone_at(args.repo, BASE_REF, shared)
        if not obs["shared_backend"].get("checkout_ok"):
            obs["fatal"] = (f"the shared backend checkout asked for {BASE_REF[:9]} and landed on "
                            f"{str(obs['shared_backend'].get('commit'))[:9]}")
            return 0
        shome = work / "home_shared"
        shome.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        inst = sh(["bash", "install.sh", "--local"], cwd=str(shared),
                  timeout=args.install_timeout, env={"UNSLOTH_STUDIO_HOME": str(shome)})
        inst["seconds"] = round(time.time() - t0, 1)
        inst["stdout"] = (inst.get("stdout") or "")[-3000:]
        obs["shared_install"] = inst
        binp = None
        for c in [shome / "bin" / "unsloth", *sorted(shome.glob(".venv*/bin/unsloth"))]:
            if c.exists() and os.access(c, os.X_OK):
                binp = str(c)
                break
        obs["shared_unsloth_bin"] = binp
        if not binp:
            obs["fatal"] = f"the shared backend install produced no unsloth CLI (rc={inst.get('rc')})"
            return 0
        print(f"== shared backend installed in {inst['seconds']}s: {binp}", flush=True)

        # THE INSTRUMENT, cloned once, hashed, and outside every arm's tree.
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
        # home. The bundle hash is re-read here and compared with the one the build job recorded,
        # so a dist that changed in transit shows up instead of passing silently.
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
                print(f"== arm {name} has no usable dist, skipping it and continuing", flush=True)
        obs["arms_built"] = usable
        obs["arms_not_built"] = [n for n in ARM_ORDER if n not in usable]
        if not usable:
            obs["fatal"] = "no arm arrived with a usable dist"
            return 0

        obs["runs"] = []
        port = args.first_port

        # ── THE JAMMED POSITIVE CONTROL, AT EVERY RUNG ───────────────────────────────────────
        control_arm = REFERENCE if REFERENCE in usable else usable[0]
        for rung in rungs:
            obs["runs"].append(measure(work, obs["states"][control_arm], "JAM", rung, "jam", port,
                                       xinfo["display"], py_gi, args.rung_timeout,
                                       instrument, ihash, hog_ms=args.hog_ms))
            port += 1

        # ── the arms, interleaved within each rep, in ARM_ORDER ───────────────────────────────
        # ARM_ORDER is deliberately not alphabetical: `main`, `A00`, `M`, `A01`, `A11` come first,
        # which is every arm the recommendation needs. A run cut short after five arms still
        # answers the lump, the decoupling and p3a; one cut short after five alphabetical arms
        # would answer nothing.
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

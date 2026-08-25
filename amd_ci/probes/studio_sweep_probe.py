#!/usr/bin/env python3
"""Probe: sweep every commit of PR 9477 between 33d65ce99 and the tip, looking for a PEAK.

Observes only. criteria/studio_sweep_peak.py judges.

THE QUESTION, AND WHY THE EXISTING DATA CANNOT ANSWER IT. Run 32819830754 measured the two
ENDPOINTS of this span on real WebKitGTK 2.52.3 / gfx1151. At r100K on `action:reasoning_toggle_all`
the left endpoint (33d65ce99, pagination live as its author wrote it) read 35.7 effective fps / 45%
busy / 1,888 ms worst, and the right endpoint (tip e48dab138 with the pagination flag forced true)
read 48.5 / 24% / 299 ms. The tip beats the authored commit on every metric at both rungs, so there
is no NET regression across the span.

A two-point measurement cannot see a PEAK IN THE MIDDLE. A commit that introduced a regression
which a later commit then partially repaired would leave BOTH endpoints below some intermediate
point, and the endpoint comparison would report "no regression" while concealing exactly the thing
being asked about. So every commit on the span is measured, not just its ends.

THIS IS A LINEAR SWEEP, NOT `git bisect`. Bisection finds one transition on a boolean predicate.
The object here is a continuous metric that may be non-monotonic, and may have no transition at
all. Bisection would not merely be slower, it would be unable to represent the answer.

THE POINTS. The span `33d65ce99..e48dab138` contains 134 commits by `git log`, because three of
them are merges of `main`. The branch's OWN history is the FIRST-PARENT path, which is nine
commits, and that is what is swept:

    p1  078ac63bc  Merge main into the long-thinking-stream branch, keeping both fence mechanisms
    p2  0b3592ad1  Stop an open fence re-parsing the whole tail twice per chunk     <- parse hot path
    p3  8d94da484  Make the paginated reasoning pane copy CRLF exactly
    p4  6adc583a2  Ship reasoning pagination behind a flag, defaulting off          <- flag appears
    p5  b0f1f51e9  Fix three repo contracts this branch was already failing
    p6  baa4c3224  Stop a regex literal in the reply defeating the open-fence bounding  <- hot path
    p7  adf308cfb  pre-commit autofixes
    p8  88f45b49b  Merge main into chat-perfomance
    p9  e48dab138  Merge latest main into chat-perfomance                           <- the tip

plus `p0` = 33d65ce99 itself, the left endpoint, which is EXCLUSIVE to the span and therefore the
anchor rather than a swept point.

PAGINATION IS HELD ON AT EVERY POINT, so that the flag's introduction at p4 is not confounded with
a code change. Before p4 there is no flag: `reasoning.tsx` hardcodes `paginateReasoning={true}` and
pagination is already live. From p4 onward `REASONING_PAGINATION_ENABLED` exists and defaults
false, and is forced true. The EFFECTIVE state is then read back out of the built source by
resolving `paginateReasoning={X}` through the flag file, and an arm whose effective state is not
`true` fails its own gate. The commit's position in the list is never trusted for this.

THE SECOND FLAG, WHICH IS THE CONFOUND NOBODY NAMED. `GRID_COLLAPSE_REASONING_ENABLED` is FALSE at
p0 and TRUE from p1 onward -- it flips at the very first commit of the span, carried in by the
merge. It is a second reasoning-pane feature flag and it changes exactly where the sweep starts, so
any p0 -> p1 step confounds grid collapse with everything else the merge brought. `p0g` is p0 with
that one literal forced true and nothing else changed, which prices it. Without `p0g` a step at the
first point is uninterpretable, and a step at the first point is precisely what a "peak" would look
like if it were an artifact.

THE SPAN CENSUS, WHICH THE ENDPOINT RUN MEASURED AND NOBODY ATTRIBUTED. The authored commit carries
41,410 `pre span` highlight spans at r100K against main's 11,530. Re-reading that run's own
artifact shows the attribution implied by those two numbers is WRONG: `Aoff` -- the same checkout
with pagination turned OFF -- also carries exactly 41,410, both arms already carry 41,410 in
`census_before` with every pane CLOSED, and at mount they carry 2,690 against main's 170. So 41,410
is a property of the 33d65ce99 CHECKOUT, not of pagination and not of the reasoning panes. Both
9477-on-main arms read 7,259, which is BELOW main. Somewhere on this span the count falls 41,410 ->
7,259, and `pre span` is code-block highlighting, so the commits that rewrote `streaming-code-policy
.ts` (p2 and p6) are the mechanistic candidates. The per-point census answers it directly, and
`p0g` separates the grid-collapse flag from the code path before the sweep even starts.

TWO BUILT-IN NULLS, BOTH FREE, BOTH CHECKABLE AT BUILD TIME. The `studio/frontend` trees of p6 and
p7 are IDENTICAL, and so are those of p8 and p9: `adf308cfb` is a pre-commit hook commit touching
one Python test, and `e48dab138` merges lint tooling only. Neither can change a pixel. They must
therefore produce the same BUNDLE HASH as their predecessor -- a stronger statement than any
measurement -- and must read the same fps. A "peak" that appears at p7 or p9 is proof the channel
is noise-dominated at this number of repetitions, and the sweep would be reporting noise rather
than commits. The bundle-hash table is printed for every point for that reason: two points sharing
one hash cannot differ in the browser, whatever their commits touched.

WHAT AN ARM ACTUALLY IS, AND WHY THE DIST IS THE WHOLE OF IT. Every point's identity is its built
`dist`, and the measuring job pins ONE backend for all of them. That is sound here and it was
checked rather than assumed: `studio/backend/main.py` is byte-identical between 33d65ce99 and the
pinned backend, so the static mount, the SPA catch-all and the `index.html` rewriters are the same
for every point; the route census across the window is purely ADDITIVE, 0 removed and 6 added; and
the 151 distinct `/api` paths the oldest frontend calls are a strict subset of what the pinned
backend serves. The branch itself never touches `studio/backend` at all -- every backend difference
on the span arrives from a merge of `main` and is deliberately excluded by pinning.

The dist is NOT only `src`. `studio/frontend/public/` is copied into `dist/` verbatim and
`index.html` is built into it, and at p8 the merge adds `public/reload-snapshot.js` (872 lines) plus
a third render-blocking classic boot script that paints a restored shell BEFORE first paint and
removes it when React commits. That is a genuine first-paint intervention, it is invisible to a
diff of `studio/frontend/src`, and it lands between p7 and p8. Shipping the whole dist per point
captures it; naming it here stops it being read as "the merge did nothing but lint".

TWO JOBS. Building eleven frontends needs no GPU, and holding the exclusive gfx1151 group while it
happens costs the other three slots for nothing. It also removes a confound: installing once rather
than eleven times means an install flake can no longer land on one point and be read as that
commit's effect. The build job asserts, per arm, what differs outside `studio/frontend/`, because
the measuring job pins ONE backend and that premise has to be checked rather than assumed.

ONE PINNED INSTRUMENT, OUTSIDE EVERY ARM'S TREE. `amdv_rung_bench.py` imports the pacer, seeder and
frozen corpus from `--sb-root`. Handing it an arm's own checkout would measure a 21 August arm with
21 August instruments and a 25 August arm with today's, then report the difference as the effect of
the commits -- the ruler co-varying with the subject, which is the single most dangerous failure
available in this design. `p0`'s tree has no `tests/studio/studiobench` at all, so that mistake
would not even fail loudly for most of the sweep. The instrument is cloned once, its resolved
`__file__` is recorded per run, and its tree is CONTENT-hashed so that "one instrument for all
eleven" is checkable rather than intended.

r100K ONLY, and the time that buys goes into REPETITIONS. At r500K the reference arm and the jammed
control both sit at 0.1 fps, so the channel cannot resolve anything there and the ratios are
direction-only; it also roughly doubles the run. Repetitions are what this design actually needs:
the endpoint run's arm A had a rep spread of 18.5 fps (44.9 against 26.4) where arm B had 0.2, and
a sweep that reported means alone would have manufactured a peak out of that one arm.

THE JAMMED POSITIVE CONTROL runs first, at the rung, in the same loop as the arms.
`--hog-ms 200 --hog-period-ms 250` blocks the page's main thread on a timer. If the frame channel
does not fall a long way under it, the channel cannot report a blocked main thread and no point on
the curve means anything. `1000/p50` reads ~62 fps jammed and unjammed alike on this scene, which
is why the effective rate `1000*raf.n/elapsed` is the metric and the p50 never is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LADDER = HERE / "studio_ladder"

#: The PR's head ref. Every commit swept here exists ONLY on the pull request, not on main, so this
#: ref is fetched explicitly and with real depth. A `--depth 50` clone off main cannot resolve them
#: and `git checkout` then leaves HEAD on the default branch tip, which has already silently turned
#: two arms of this campaign into main.
PR_REF = "refs/pull/9477/head"
FETCH_DEPTH = 400

#: The pinned backend AND the pinned instrument: today's main, the tip's own merge base. One
#: checkout of this serves every arm's dist, and a separate checkout of it provides the pacer,
#: seeder and frozen corpus for every measurement.
BACKEND_REF = "90f85fdbf8fd6f9df1b99aff44f1e45da4c808d0"
INSTRUMENT_REF = BACKEND_REF

#: (arm, commit, force_grid_collapse, what the commit is)
#:
#: `force_grid_collapse` is None everywhere except `p0g`. It is not "set grid collapse true"
#: applied blindly: p1..p9 already have it true, and forcing a literal that is already at the
#: wanted value would be an edit that cannot be told from a no-op in the diff. The value is READ
#: BACK on every arm regardless, and recorded, so the sweep's grid-collapse column is measured
#: rather than assumed.
SWEEP: tuple[tuple[str, str, bool | None, str], ...] = (
    ("p0",  "33d65ce99b40de93361366447a3e34949480008a", None,
     "33d65ce99 as authored: the left endpoint. No pagination flag exists; reasoning.tsx "
     "hardcodes paginateReasoning={true}. GRID_COLLAPSE_REASONING_ENABLED is FALSE here"),
    ("p0g", "33d65ce99b40de93361366447a3e34949480008a", True,
     "33d65ce99 with GRID_COLLAPSE_REASONING_ENABLED forced true and nothing else changed. "
     "The anchor the rest of the sweep is comparable to, and the arm that prices grid collapse"),
    ("p1",  "078ac63bc77bf30e54963a9d74d2264b4285a6f4", None,
     "Merge main into the long-thinking-stream branch, keeping both fence mechanisms"),
    ("p2",  "0b3592ad1cb75c5e2c967fe5a1b98042c1797a2b", None,
     "Stop an open fence re-parsing the whole tail twice per chunk -- PARSE HOT PATH"),
    ("p3",  "8d94da484acaf05176088ffd5e027e2411223ac1", None,
     "Make the paginated reasoning pane copy CRLF exactly"),
    ("p4",  "6adc583a221e64c714f767165dd33f557f96f07a", None,
     "Ship reasoning pagination behind a flag, defaulting off -- THE FLAG APPEARS HERE"),
    ("p5",  "b0f1f51e97db2292d6a4e678dc5f1f20287c5a9a", None,
     "Fix three repo contracts this branch was already failing (1 frontend line, 2 Python tests)"),
    ("p6",  "baa4c322426a19b2c911e32ab00c4a6c2bb401bc", None,
     "Stop a regex literal in the reply defeating the open-fence bounding -- PARSE HOT PATH"),
    ("p7",  "adf308cfbb65676c3bae194e1710e3ba6c3d9bf1", None,
     "pre-commit autofixes: ONE Python test file. FREE NULL -- its studio/frontend tree is "
     "identical to p6's, so it must share p6's bundle hash and its fps"),
    ("p8",  "88f45b49b4931f20b1caafbb9cb08e95d00fa3a8", None,
     "Merge main into chat-perfomance. Adds public/reload-snapshot.js and a third render-blocking "
     "boot script to index.html: a first-paint intervention OUTSIDE studio/frontend/src"),
    ("p9",  "e48dab1386ad16d2f56a086a798e78caf376b66c", None,
     "Merge latest main into chat-perfomance -- THE TIP. Lint tooling only; its studio/frontend "
     "tree is identical to p8's, so no effect may be attributed to this commit"),
)

ARM_ORDER = tuple(a[0] for a in SWEEP)
ARM_REF = {a[0]: a[1] for a in SWEEP}
ARM_GRID = {a[0]: a[2] for a in SWEEP}
ARM_WHAT = {a[0]: a[3] for a in SWEEP}

#: Arms that must be indistinguishable, recorded BEFORE the run so a null reads as CONFIRMED INERT
#: rather than as a change that quietly failed to take.
PREDICTED_NULLS = {
    ("p6", "p7"): ("adf308cfb changes one Python TEST file and no shipped source at all: the "
                   "`studio/frontend` trees of p6 and p7 are IDENTICAL by tree hash. p7 cannot "
                   "differ from p6 in the browser. If the two bundle hashes match, this is settled "
                   "at BUILD time and the measured pair becomes a pure noise estimate for the "
                   "whole sweep. A 'peak' at p7 means the channel cannot resolve this question at "
                   "this number of repetitions."),
    ("p8", "p9"): ("e48dab138 merges lint tooling only -- `.github/workflows/lint-ci.yml`, "
                   "`scripts/lint_duplicate_definitions.py` and its test. The `studio/frontend` "
                   "trees of p8 and p9 are IDENTICAL by tree hash, so the TIP of this PR is "
                   "bit-identical in the browser to its parent. A second free noise estimate, and "
                   "the reason no effect may ever be attributed to the tip commit itself."),
}

#: The file that carries both flags, and the component that consumes them.
FLAGS_REL = Path("studio/frontend/src/components/assistant-ui/thread-feature-flags.ts")
REASONING_REL = Path("studio/frontend/src/components/assistant-ui/reasoning.tsx")

PAGINATION_CONST = "REASONING_PAGINATION_ENABLED"
GRID_CONST = "GRID_COLLAPSE_REASONING_ENABLED"

#: Every key a state dict must carry, whatever went wrong. A build failure must fail its OWN gate
#: and leave the other ten arms measurable: a `KeyError` on one bad arm destroyed three good arms
#: earlier in this campaign.
#: A FACTORY, not a module-level dict. `**EMPTY_STATE` copies the references, not the values, so
#: eleven arms spread from one literal would share one `edits` list and one `dist` dict, and a
#: single in-place append anywhere would appear in all of them. Nothing appends today; eleven arms
#: silently sharing state is not a bug worth leaving one edit away.
def _empty_state() -> dict:
    return {
        "prepared_ok": False, "dirty_files": 0,
        "pagination_effective": None, "pagination_how": None, "grid_collapse": None,
        "edits": [], "install": {"rc": None}, "unsloth_bin": None, "home": None,
        "repo_root": None, "bundle_hash": None,
        "dist": {"path": None, "exists": False, "index_html": False, "asset_files": 0},
    }

sys.path.insert(0, str(HERE))
from webkit_paint_probe import fetch_xvfb, inventory, start_xserver  # noqa: E402
from studio_ladder_probe import gi_python, sh  # noqa: E402


# ── checkout, hashing ────────────────────────────────────────────────────────────────────────────

def clone_at(repo_url: str, ref: str, dest: Path) -> dict:
    """Clone so BOTH main commits and PR-only commits resolve, then land on `ref` EXACTLY.

    ASKED and LANDED are both recorded on every path. A shallow clone that cannot resolve the ref
    leaves HEAD on the default branch tip and every later step succeeds against the wrong tree;
    that has happened here, to the arms that mattered most, and only this assertion caught it.
    """
    out: dict = {"url": repo_url, "ref": ref, "dest": str(dest)}
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    out["clone"] = sh(["git", "clone", "--filter=blob:none", repo_url, str(dest)], timeout=2400)
    if out["clone"].get("rc") != 0:
        out["clone_fallback"] = sh(["git", "clone", repo_url, str(dest)], timeout=3600)
    # The PR head, explicitly: nine of the eleven arms exist nowhere else.
    out["fetch_pr"] = sh(["git", "fetch", "--depth", str(FETCH_DEPTH), "origin",
                          f"{PR_REF}:pr9477"], cwd=str(dest), timeout=1800)
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
    """Content hash of a built frontend bundle, the SAME definition `amdv_rung_bench.py` uses, so
    the hash taken at build time and the one taken at measure time are comparable and a dist that
    changed in transit shows up rather than passing silently."""
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
    """Content hash of a directory tree. Used on the instrument, so "one instrument for every arm"
    is checkable by CONTENT and not only by resolved path: a path assertion cannot see a tree that
    changed under it between the first arm and the eleventh."""
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


# ── the flags: force, then read the EFFECTIVE state back out of the source ───────────────────────

_CONST_RE = r"^(\s*export\s+const\s+{name}\s*(?::\s*boolean\s*)?=\s*)(true|false)(\s*;.*)$"


def set_bool_const(path: Path, name: str, value: bool) -> dict:
    """Rewrite `export const NAME = <bool>;` in place. Reports what it saw and what it wrote.

    A regex bounded to that one declaration, rather than a patch file, because eleven different
    trees carry this file at three different line numbers with different neighbours, and a static
    patch that fails to apply on six of them is a worse instrument than an edit that names the
    line it changed. `git apply` returning rc=0 against the wrong hunk is the failure this avoids.
    """
    rec: dict = {"file": str(path), "const": name, "want": value,
                 "existed": path.is_file(), "matched": 0, "before": None, "after": None}
    if not path.is_file():
        return rec
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    pat = re.compile(_CONST_RE.format(name=re.escape(name)))
    out = []
    for line in lines:
        m = pat.match(line.rstrip("\n"))
        if m:
            rec["matched"] += 1
            rec["before"] = m.group(2)
            rec["after"] = "true" if value else "false"
            nl = "\n" if line.endswith("\n") else ""
            out.append(f"{m.group(1)}{rec['after']}{m.group(3)}{nl}")
        else:
            out.append(line)
    # Exactly one declaration, or refuse. Two matches would mean the value the bundle sees depends
    # on which one wins, and zero means the edit silently did nothing.
    if rec["matched"] == 1:
        path.write_text("".join(out), encoding="utf-8")
        rec["written"] = True
    else:
        rec["written"] = False
    return rec


def read_bool_const(path: Path, name: str) -> str | None:
    """The literal a `export const NAME = <bool>;` declaration carries, or None."""
    if not path.is_file():
        return None
    pat = re.compile(_CONST_RE.format(name=re.escape(name)))
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.match(line)
        if m:
            return m.group(2)
    return None


def read_effective_pagination(root: Path) -> dict:
    """RESOLVE `paginateReasoning={X}` THROUGH THE FLAG FILE. Never trust the commit's position.

    Three shapes occur across this span and they must not be conflated:
      p0..p3  `paginateReasoning={true}`                        -> live, hardcoded, no flag exists
      p4..p9  `paginateReasoning={REASONING_PAGINATION_ENABLED}` -> live iff that const is true
      absent  the prop is not passed at all                     -> unknown, and the arm fails

    The task this probe answers turns entirely on pagination being ON at every point, so "on" is
    established by reading the built source, in both files, and an arm that cannot be resolved to
    a literal `true` is refused rather than measured and quietly averaged in.
    """
    rc = root / REASONING_REL
    fl = root / FLAGS_REL
    out: dict = {"prop": None, "prop_lines": [], "flag_literal": read_bool_const(fl, PAGINATION_CONST),
                 "effective": None, "how": None}
    if not rc.is_file():
        out["how"] = f"{REASONING_REL} does not exist in this tree"
        return out
    pat = re.compile(r"paginateReasoning\s*=\s*\{([^}]*)\}")
    exprs = []
    for line in rc.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line)
        if m:
            exprs.append(m.group(1).strip())
            out["prop_lines"].append(line.strip()[:140])
    if not exprs:
        out["how"] = "reasoning.tsx never passes paginateReasoning"
        return out
    if len(set(exprs)) > 1:
        out["prop"] = exprs
        out["how"] = f"reasoning.tsx passes paginateReasoning with conflicting expressions: {exprs}"
        return out
    expr = exprs[0]
    out["prop"] = expr
    if expr in ("true", "false"):
        out["effective"] = expr
        out["how"] = f"hardcoded {expr} in reasoning.tsx; no flag involved"
    elif expr == PAGINATION_CONST:
        out["effective"] = out["flag_literal"]
        out["how"] = (f"reasoning.tsx passes {PAGINATION_CONST}, which "
                      f"{FLAGS_REL.name} declares as {out['flag_literal']}")
    else:
        out["how"] = f"paginateReasoning={{{expr}}} is neither a literal nor {PAGINATION_CONST}"
    return out


def prepare_flags(root: Path, arm: str) -> dict:
    """Force pagination ON, force grid collapse where this arm asks for it, then READ BOTH BACK.

    The read-back is the fact the whole sweep turns on and it is taken from the source that is
    about to be compiled, not from the edit's return value.
    """
    fl = root / FLAGS_REL
    edits = []
    # Pagination: only where a flag exists to force. Before p4 the prop is hardcoded true and there
    # is nothing to set -- writing a const that the component does not read would look like a
    # successful edit and change nothing in the bundle.
    if read_bool_const(fl, PAGINATION_CONST) is not None:
        edits.append(set_bool_const(fl, PAGINATION_CONST, True))
    want_grid = ARM_GRID.get(arm)
    if want_grid is not None:
        edits.append(set_bool_const(fl, GRID_CONST, want_grid))

    pag = read_effective_pagination(root)
    return {"edits": edits,
            "pagination": pag,
            "pagination_effective": pag.get("effective"),
            "pagination_how": pag.get("how"),
            "grid_collapse": read_bool_const(fl, GRID_CONST),
            "grid_forced": want_grid}


# ── build ────────────────────────────────────────────────────────────────────────────────────────

def build_state(work: Path, arm: str, repo_url: str, ref: str, install_timeout: int) -> dict:
    """One checkout, flags forced and read back, installed, dist hashed.

    EVERY return path carries the full key set, so an arm that fails here fails its own gate and
    the other ten are still measured.
    """
    root = work / f"state_{arm}"
    out: dict = {"name": arm, "ref": ref, "what": ARM_WHAT[arm], **_empty_state()}
    out.update(clone_at(repo_url, ref, root))
    if not out.get("checkout_ok"):
        out["why"] = (f"asked for {ref[:9]} and landed on {str(out.get('commit'))[:9]}; the commit "
                      f"was not fetched, so this arm was NOT built")
        return out

    out.update(prepare_flags(root, arm))
    # THE GATE THIS ARM MUST PASS. Pagination has to be live, established from the source, or the
    # point is not on the curve the question asks about and must not be drawn on it.
    out["prepared_ok"] = out.get("pagination_effective") == "true"
    if not out["prepared_ok"]:
        out["why"] = (f"pagination is not live at {ref[:9]}: {out.get('pagination_how')}. This arm "
                      f"is NOT measured, because a point with pagination off is not on the curve.")
        return out

    out["dirty_files"] = len([l for l in (sh(["git", "status", "--porcelain"], cwd=str(root),
                                             timeout=60).get("stdout") or "").splitlines()
                              if l.strip()])
    # WHAT DIFFERS OUTSIDE THE FRONTEND, against the pinned backend. The measuring job installs ONE
    # backend for all eleven arms, which is only sound if the arms' non-frontend differences cannot
    # reach the browser. That is recorded per arm rather than asserted, so the report can say what
    # was held fixed instead of implying nothing was.
    diff = sh(["git", "diff", "--name-only", BACKEND_REF, "HEAD", "--",
               ".", ":(exclude)studio/frontend"], cwd=str(root), timeout=300)
    names = [n for n in (diff.get("stdout") or "").splitlines() if n.strip()]
    out["differs_from_backend_outside_frontend"] = {
        "count": len(names),
        "non_test_non_ci": [n for n in names
                            if not n.startswith(("tests/", ".github/", "docs/"))][:40],
        "sample": names[:40]}

    # THE SOURCE TREE HASH, taken BEFORE the build and from git rather than from the bundle.
    # `p6==p7` and `p8==p9` are claims about the SOURCE, and a bundle hash cannot distinguish
    # "these trees are identical" from "vite happened to emit identical bytes". Both are recorded
    # so the free nulls are provable at build time rather than inferred from the measurement.
    #
    # `studio/frontend` whole, NOT `studio/frontend/src`: `public/` is copied into `dist/` verbatim
    # and `index.html` is built into it, and the first-paint boot script that lands at p8 lives in
    # exactly that gap. A src-only hash would call p7 -> p8 a null and be wrong.
    for label, spec in (("frontend_tree_hash", "HEAD:studio/frontend"),
                        ("frontend_src_tree_hash", "HEAD:studio/frontend/src"),
                        ("frontend_public_tree_hash", "HEAD:studio/frontend/public")):
        r = sh(["git", "rev-parse", spec], cwd=str(root), timeout=60)
        out[label] = (r.get("stdout") or "").strip() or None
    idx = sh(["git", "hash-object", "studio/frontend/index.html"], cwd=str(root), timeout=60)
    out["frontend_index_html_hash"] = (idx.get("stdout") or "").strip() or None

    home = work / f"home_{arm}"
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


def do_build(args, obs: dict, work: Path) -> int:
    """BUILD MODE. No GPU, no display, no measurement. Its own job."""
    dists = Path(args.dist_out)
    dists.mkdir(parents=True, exist_ok=True)
    obs["states"] = {}
    for arm in ARM_ORDER:
        ref = ARM_REF[arm]
        try:
            st = build_state(work, arm, args.repo, ref, args.install_timeout)
        except Exception as e:                                            # noqa: BLE001
            st = {"name": arm, "ref": ref, "what": ARM_WHAT[arm], **_empty_state(),
                  "why": f"{type(e).__name__}: {e}"}
        obs["states"][arm] = st
        if (st.get("dist") or {}).get("index_html") and st.get("prepared_ok"):
            dest = dists / arm
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(st["dist"]["path"], dest)
            st["exported_dist"] = str(dest)
            st["exported_bundle_hash"] = bundle_hash(dest)
        print(f"== built {arm}: asked {ref[:9]} landed {str(st.get('commit'))[:9]} "
              f"prepared_ok={st.get('prepared_ok')} "
              f"pagination={st.get('pagination_effective')} "
              f"grid={st.get('grid_collapse')} "
              f"bundle={st.get('exported_bundle_hash')} "
              f"install rc={(st.get('install') or {}).get('rc')}", flush=True)
        # Eleven checkouts and eleven venvs on a disk three other slots share.
        shutil.rmtree(st.get("repo_root") or "/nonexistent", ignore_errors=True)
        shutil.rmtree(st.get("home") or "/nonexistent", ignore_errors=True)

    manifest = {
        "backend_ref": BACKEND_REF, "instrument_ref": INSTRUMENT_REF, "pr_ref": PR_REF,
        "arm_order": list(ARM_ORDER), "arm_refs": ARM_REF, "arm_what": ARM_WHAT,
        "predicted_nulls": {f"{a}|{b}": v for (a, b), v in PREDICTED_NULLS.items()},
        "states": {n: {k: v for k, v in s.items()
                       if k in ("ref", "ref_requested", "ref_landed", "commit", "checkout_ok",
                                "what", "prepared_ok", "why", "edits", "pagination",
                                "pagination_effective", "pagination_how", "grid_collapse",
                                "grid_forced", "dirty_files", "bundle_hash",
                                "differs_from_backend_outside_frontend",
                                "frontend_tree_hash", "frontend_src_tree_hash",
                                "frontend_public_tree_hash", "frontend_index_html_hash",
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

    # THE BUNDLE-HASH TABLE, at build time. Two arms sharing a hash cannot differ in the browser,
    # whatever their commits changed, and that is a stronger statement than any measurement.
    groups: dict[str, list[str]] = {}
    for n in obs["arms_built"]:
        groups.setdefault(obs["states"][n]["exported_bundle_hash"], []).append(n)
    obs["bundle_groups"] = groups
    tgroups: dict[str, list[str]] = {}
    for n in obs["arms_built"]:
        tgroups.setdefault(str(obs["states"][n].get("frontend_tree_hash")), []).append(n)
    obs["frontend_tree_groups"] = tgroups
    print(f"== built {len(obs['arms_built'])}/{len(ARM_ORDER)}: {obs['arms_built']}", flush=True)
    for h, arms in groups.items():
        print(f"== bundle {h}: {arms}", flush=True)
    for h, arms in tgroups.items():
        print(f"== frontend tree {str(h)[:12]}: {arms}", flush=True)

    # THE TWO BUILD-TIME CHECKS THIS DESIGN TURNS ON, stated as findings rather than left for a
    # reader to spot in two tables.
    #
    # 1. The predicted nulls: identical SOURCE trees must give identical BUNDLES. If they do not,
    #    the build is not reproducible and no small difference anywhere on the curve can be read.
    for (a, b) in PREDICTED_NULLS:
        sa, sb2 = obs["states"].get(a) or {}, obs["states"].get(b) or {}
        same_tree = (sa.get("frontend_tree_hash") is not None
                     and sa.get("frontend_tree_hash") == sb2.get("frontend_tree_hash"))
        same_bundle = (sa.get("exported_bundle_hash") is not None
                       and sa.get("exported_bundle_hash") == sb2.get("exported_bundle_hash"))
        obs.setdefault("null_checks", {})[f"{a}|{b}"] = {
            "same_frontend_tree": same_tree, "same_bundle": same_bundle,
            "reproducible": same_tree == same_bundle}
        print(f"== null {a} vs {b}: same_source_tree={same_tree} same_bundle={same_bundle}"
              f"{'' if same_tree == same_bundle else '  <-- BUILD IS NOT REPRODUCIBLE'}", flush=True)
    # 2. p0 and p0g are the SAME COMMIT and must therefore share a source tree hash, but they
    #    differ by the forced grid-collapse literal and must therefore NOT share a bundle. A shared
    #    bundle here means the one edit that prices grid collapse never reached the compiler, and
    #    the arm that disambiguates the first step of the curve would be silently a duplicate.
    s0, s0g = obs["states"].get("p0") or {}, obs["states"].get("p0g") or {}
    obs["grid_edit_took"] = {
        "same_commit": s0.get("commit") == s0g.get("commit"),
        "same_source_tree": s0.get("frontend_tree_hash") == s0g.get("frontend_tree_hash"),
        "bundles_differ": (s0.get("exported_bundle_hash") != s0g.get("exported_bundle_hash")),
        "p0_grid": s0.get("grid_collapse"), "p0g_grid": s0g.get("grid_collapse")}
    print(f"== grid edit: p0 grid={s0.get('grid_collapse')} p0g grid={s0g.get('grid_collapse')} "
          f"bundles_differ={obs['grid_edit_took']['bundles_differ']}"
          f"{'' if obs['grid_edit_took']['bundles_differ'] else '  <-- THE GRID EDIT DID NOT TAKE'}",
          flush=True)
    return 0


# ── measure ──────────────────────────────────────────────────────────────────────────────────────

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
           # ONE instrument for every arm, never the arm's own checkout.
           "--sb-root", str(instrument), "--unsloth-bin", st["unsloth_bin"],
           "--python-gi", py_gi,
           # The p4 scene unchanged: it carries `action:reasoning_toggle_all`, which opens EVERY
           # reasoning pane, and a census taken while the panes are OPEN. Both are required here.
           # A census taken with the panes closed cannot tell "the flag never reached the DOM"
           # apart from "the mechanism engaged and did not matter", and the single-pane gesture of
           # the same name mounts 12,149 of 1,148,084 characters, which is unreadable as a frame
           # number.
           "--scene", str(LADDER / "amdv_scene_p4.js"),
           "--driver", str(LADDER / "amdv_drive.py"),
           # PASSIVE. `begin_updating()` drives the clock itself and reads 60.0 fps with the main
           # thread 80% blocked, so under it the presented-frame series cannot carry a claim.
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
        except Exception as e:                                            # noqa: BLE001
            entry["payload_error"] = f"{type(e).__name__}: {e}"
    return entry


def parse_reps(spec: str, rungs: list[str]) -> dict:
    """`--reps 100K:5` or a bare `--reps 5`."""
    if ":" not in spec:
        return {r: int(spec) for r in rungs}
    out = {}
    for part in spec.split(","):
        k, _, v = part.partition(":")
        out[k.strip()] = int(v)
    return {r: out.get(r, 2) for r in rungs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--state", default="host")
    ap.add_argument("--checkout", default="")
    ap.add_argument("--mode", default="measure", choices=["build", "measure"],
                    help="build: clone, force the flags and build every point's frontend, no GPU "
                         "and no display. measure: install ONE backend and drive each point's "
                         "prebuilt dist. Separate jobs, because building holds the exclusive GPU "
                         "group for work that does not touch the GPU.")
    ap.add_argument("--dist-out", default="", help="where build mode writes the per-arm dists")
    ap.add_argument("--dist-in", default="", help="where measure mode reads them from")
    ap.add_argument("--work", default=os.environ.get("AMD_CI_WORK", "/tmp/studio_sweep"))
    ap.add_argument("--repo", default="https://github.com/unslothai/unsloth")
    ap.add_argument("--rungs", default="100K")
    ap.add_argument("--reps", default="100K:5")
    ap.add_argument("--first-port", type=int, default=5461)
    ap.add_argument("--install-timeout", type=int, default=3600)
    ap.add_argument("--rung-timeout", type=int, default=2400)
    ap.add_argument("--hog-ms", type=int, default=200)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    work = Path(args.work) / "sweep"
    work.mkdir(parents=True, exist_ok=True)
    rungs = [r.strip() for r in args.rungs.split(",") if r.strip()]
    reps = parse_reps(args.reps, rungs)
    obs: dict = {"mode": args.mode, "rungs": rungs, "reps": reps,
                 "backend_ref": BACKEND_REF, "instrument_ref": INSTRUMENT_REF, "pr_ref": PR_REF,
                 "arm_order": list(ARM_ORDER), "arm_refs": ARM_REF, "arm_what": ARM_WHAT,
                 "predicted_nulls": {f"{a}|{b}": v for (a, b), v in PREDICTED_NULLS.items()}}

    if args.mode == "build":
        try:
            return do_build(args, obs, work)
        finally:
            args.out.write_text(json.dumps(obs, indent=2))

    # ── MEASURE MODE ─────────────────────────────────────────────────────────────────────────────
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
            obs["fatal"] = (f"no build manifest at {mpath}: the build job produced none, so there "
                            f"is nothing to measure. That is a non-result, not a finding.")
            return 0
        manifest = json.loads(mpath.read_text())
        obs["build"] = manifest
        obs["states"] = {n: dict(s) for n, s in manifest["states"].items()}

        # ONE BACKEND, INSTALLED ONCE, PINNED AT MAIN. Every arm is served as `unsloth studio -f
        # <dist>`, so the backend is a static file server plus the API the seeder posts through.
        # Installing it eleven times would add eleven chances for one point's install to differ
        # from another's and be read as that commit's effect.
        shared = work / "backend"
        obs["shared_backend"] = clone_at(args.repo, BACKEND_REF, shared)
        if not obs["shared_backend"].get("checkout_ok"):
            obs["fatal"] = (f"the shared backend checkout asked for {BACKEND_REF[:9]} and landed "
                            f"on {str(obs['shared_backend'].get('commit'))[:9]}")
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
            obs["fatal"] = (f"the shared backend install produced no unsloth CLI "
                            f"(rc={inst.get('rc')})")
            return 0
        print(f"== shared backend installed in {inst['seconds']}s: {binp}", flush=True)

        # THE INSTRUMENT: cloned once, resolved, content-hashed, and outside every arm's tree.
        instrument = work / "instrument"
        obs["instrument"] = clone_at(args.repo, INSTRUMENT_REF, instrument)
        obs["instrument"]["path"] = str(instrument)
        if not obs["instrument"].get("checkout_ok"):
            obs["fatal"] = (f"the instrument checkout asked for {INSTRUMENT_REF[:9]} and landed on "
                            f"{str(obs['instrument'].get('commit'))[:9]}; refusing to measure any "
                            f"point with an instrument that is not the one declared")
            return 0
        sb = instrument / "tests" / "studio" / "studiobench"
        obs["instrument"]["studiobench_present"] = sb.is_dir()
        if not sb.is_dir():
            obs["fatal"] = f"the pinned instrument has no studiobench package at {sb}"
            return 0
        ihash = tree_hash(sb)
        obs["instrument"]["studiobench_hash"] = ihash
        print(f"== instrument at {obs['instrument']['commit'][:9]} hash={ihash}", flush=True)

        # Each point becomes a state the measure path understands: its OWN dist, the SHARED bin and
        # home. The bundle hash is re-read here and compared with the build job's, so a dist that
        # changed in transit shows up instead of passing silently.
        usable = []
        for arm in ARM_ORDER:
            st = obs["states"].get(arm) or {}
            d = dists / arm
            st["dist"] = {"path": str(d), "exists": d.is_dir(),
                          "index_html": (d / "index.html").is_file(),
                          "asset_files": len(list((d / "assets").rglob("*")))
                          if (d / "assets").is_dir() else 0}
            st["unsloth_bin"] = binp
            st["home"] = str(shome)
            st["bundle_hash_at_measure"] = bundle_hash(d) if d.is_dir() else None
            st["bundle_hash_matches_build"] = (
                st.get("exported_bundle_hash") == st["bundle_hash_at_measure"])
            obs["states"][arm] = st
            if st["dist"]["index_html"] and st.get("prepared_ok"):
                usable.append(arm)
            else:
                print(f"== point {arm} has no usable dist or failed its flag gate, skipping it "
                      f"and continuing", flush=True)
        obs["arms_built"] = usable
        obs["arms_not_built"] = [n for n in ARM_ORDER if n not in usable]
        if not usable:
            obs["fatal"] = "no point arrived with a usable dist"
            return 0

        obs["runs"] = []
        port = args.first_port

        # ── THE JAMMED POSITIVE CONTROL, FIRST, at every rung, in the same loop as the points ────
        control_arm = "p9" if "p9" in usable else usable[-1]
        obs["control_arm"] = control_arm
        for rung in rungs:
            obs["runs"].append(measure(work, obs["states"][control_arm], "JAM", rung, "jam", port,
                                       xinfo["display"], py_gi, args.rung_timeout,
                                       instrument, ihash, hog_ms=args.hog_ms))
            port += 1

        # ── the points, INTERLEAVED within each rep ──────────────────────────────────────────────
        # Interleaved, never blocked: a curve measured arm-by-arm would alias any thermal or
        # cache drift over the run onto commit order, which is exactly the shape a "peak in the
        # middle" would take.
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
                except Exception:                                         # noqa: BLE001
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
            except Exception:                                             # noqa: BLE001
                pass
            obs["xserver"]["stopped_pid"] = xproc.pid
        args.out.write_text(json.dumps(obs, indent=2))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Criteria for PR 8642: the torchcodec "installed but cannot load" report.

MODE is `regression`, and the base leg is VACUOUS BY CONSTRUCTION. That is not a
tolerance widened to rescue a result, it is a property of the change: PR 8642 adds
`studio/backend/tests/test_setup_torchcodec_probe.py`, and `lib/states.py` checks out
each state as a whole worktree, so at the base that file does not exist and there is
nothing at the base to run. `pytest_probe.py` records it in `absent_at_this_state`
rather than letting pytest call it a collection error.

So this run CANNOT and DOES NOT demonstrate the defect at the base. What it can
demonstrate, and what the gates below are built to make non-vacuous, is:

  * the new assertions REALLY EXECUTED on a Windows 11 gfx1151 box, rather than
    erroring in a fixture or skipping wholesale, and
  * no test that passed at the base fails at the head or at the merge.

`criteria/pytest_no_regression.py` is unchanged and still refuses this shape, which
is correct for its stated question ("does this change break a test that used to
pass?" needs a base that ran). This module asks the narrower question the shape
actually admits, and says so in the report instead of quietly relaxing the gate.

Two prior Windows runs are the reason each gate exists:
  * PR 10289: the backend suite collected 25 and SKIPPED all 25 on Windows. A suite
    that only skipped is not evidence, hence the "executed test bodies" gates.
  * PR 8642 run 34815928085: 23 of 25 ERRORED at a session fixture importing
    huggingface_hub, and the observation recorded "0 failed". Hence the setup-error
    gate, and hence the workflow now installs what conftest.py imports.
"""

from __future__ import annotations

TITLE = "PR 8642 torchcodec probe: do the new assertions execute on Windows gfx1151?"
MODE = "regression"

# Authored, not detected. The gaps in the report are NEEDS minus what this host is,
# so anything the CHANGE touches belongs here even when the host cannot reach it.
#   windows  setup.ps1 ships the PowerShell probe; this job runs there, so it is MET
#            and correctly drops out of the gap list.
#   rocm     the report exists for AMD users whose torchcodec cannot load. Whether a
#            ROCm torch exists on the Windows boxes is unmeasured, so this stays a gap.
#   linux    setup.sh is the POSIX installer. Its text is asserted here, but `bash -n`
#            skips on Windows, so the shell was never actually parsed by this run.
#   mlx      setup.sh's in-body `_alarm(60)` deadline exists for stock macOS, which has
#            no coreutils `timeout`. Unreachable from this pool.
NEEDS: list[str] = ["windows", "rocm", "linux", "mlx"]

# pytest exit codes: 0 ok, 1 tests failed, 2 interrupted, 3 internal, 4 usage, 5 none
# collected. Only 0 and 1 mean the suite got as far as judging something.
_RAN_CODES = (0, 1)


def _counts(o: dict) -> tuple[int, int, int, int, int]:
    p = o.get("n_passed", 0) or 0
    f = o.get("n_failed", 0) or 0
    s = o.get("n_skipped", 0) or 0
    e = o.get("n_errors", 0) or 0
    return p, f, s, e, p + f + s + e


def _base_is_legitimately_absent(o: dict) -> bool:
    """True only when EVERY selected path is missing at this state and nothing ran.

    Deliberately narrow. A base that merely failed to collect, or that ran some of the
    selection and not the rest, is a broken run and must not be waved through as "the
    PR added the tests".
    """
    return bool(o.get("absent_at_this_state")) and not o.get("selected") and o.get("rc") is None


def _executed_gates(name: str, o: dict) -> list[tuple[str, bool, str]]:
    p, f, s, e, collected = _counts(o)
    rc = o.get("rc")
    out: list[tuple[str, bool, str]] = []

    detail = o.get("error") or o.get("note") or f"rc={rc}"
    if rc not in _RAN_CODES:
        detail += f"; pytest exit {rc} means it never ran the tests"
        if o.get("stderr_tail"):
            detail += f"; stderr: {str(o['stderr_tail'])[:200]}"
    out.append((f"{name} suite actually ran", rc in _RAN_CODES, detail))

    out.append((f"{name} suite collected tests", collected > 0,
                f"{collected} collected (passed {p}, failed {f}, skipped {s}, errors {e})"))

    # A test that errors in a fixture never reached its body, so it is neither a pass
    # nor a failure, and a criteria that only reads n_failed sees a clean suite.
    out.append((f"{name} suite ran without setup errors", e == 0,
                f"{e} test(s) errored before their body ran"
                + (f": {', '.join(o.get('errors', [])[:5])}" if o.get("errors") else "")))

    # The whole point of renting the hardware. Skips are not observations about it.
    out.append((f"{name} executed test bodies", p + f > 0,
                f"{p + f} of {collected} ran their body, {s} skipped"))

    # Stronger than "at least one": PR 10289 skipped 25 of 25 here, and a run where
    # most of the suite opts out is not evidence about this platform either.
    out.append((f"{name} executed most of the suite, not mostly skips", s * 2 < collected,
                f"{s} skipped of {collected} collected"))

    # The selection has to be the tests the PR adds, at the states that have them.
    out.append((f"{name} found every selected test file", not o.get("absent_at_this_state"),
                "absent: " + ", ".join(o.get("absent_at_this_state") or []) if
                o.get("absent_at_this_state") else "all selected paths present"))
    return out


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []

    base = obs.get("base") or {}
    if _base_is_legitimately_absent(base):
        # Passes, and the evidence string is the disclosure. A reader of this report
        # must not be able to miss that the base proved nothing.
        out.append((
            "base state accounted for", True,
            "VACUOUS: the selection does not exist at the base ("
            + ", ".join(base.get("absent_at_this_state") or [])
            + "), because this PR adds it. The base did NOT reproduce any defect and "
              "this run is not evidence that one existed."))
    else:
        out += _executed_gates("base", base)

    for name in ("head", "merge"):
        o = obs.get(name)
        if o is None:
            if name == "head":
                out.append(("head state present", False, "no head observation"))
            continue
        out += _executed_gates(name, o)
    return out


def table(obs: dict) -> str:
    rows = ["| state | passed | failed | skipped | errored | failing tests |",
            "|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        p, f, s, e, _ = _counts(o)
        failed = ", ".join(f"`{x.split('::')[-1]}`" for x in o.get("failed", [])) or "none"
        errored = ", ".join(f"`{x.split('::')[-1]}`" for x in o.get("errors", []))
        if errored:
            failed = (failed if failed != "none" else "") + f" (errored: {errored})"
        rows.append(f"| {name} | {p} | {f} | {s} | {e} | {failed} |")

    base = obs.get("base") or {}
    notes = []
    if base.get("absent_at_this_state"):
        notes.append(
            "**The base leg is vacuous and the base defect was NOT reproduced.** "
            + ", ".join(f"`{a}`" for a in base["absent_at_this_state"])
            + " is added by this PR, and `states.py` checks out each state as a whole "
              "worktree, so there is nothing at the base to run. This run therefore says "
              "nothing about whether the old code misreported an unloadable torchcodec. "
              "It says only that the new assertions execute and pass on this host.")
    return "\n".join(rows) + ("\n\n" + "\n\n".join(notes) if notes else "")


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b = set(base.get("failed", [])) | set(base.get("errors", []))
    h = set(head.get("failed", [])) | set(head.get("errors", []))
    new = sorted(h - b)
    if new:
        return True, ("failing or erroring at the head: "
                      + ", ".join(f"`{n.split('::')[-1]}`" for n in new)
                      + (" (the base ran none of these, so they are new tests that FAIL "
                         "on this host, not a regression against a passing base)"
                         if _base_is_legitimately_absent(base) else ""))

    p, f, s, _e, collected = _counts(head)
    detail = (f"every one of the {p} executed assertions passed at the head "
              f"({s} skipped of {collected} collected)")
    if _base_is_legitimately_absent(base):
        detail += ("; the base ran nothing at all, so this is NOT a base-versus-head "
                   "comparison and no base defect was demonstrated")
    fixed = sorted(b - h)
    if fixed:
        detail += "; newly passing: " + ", ".join(f"`{n.split('::')[-1]}`" for n in fixed)
    return False, detail

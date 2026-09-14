#!/usr/bin/env python3
"""Criteria: does this change break a test that used to pass?

Compares by test ID, never by count. A PR that adds tests raises the pass count
and can raise the fail count without regressing anything, and a PR that deletes
a failing test lowers both while fixing nothing. Only the set difference of
FAILED ids answers the question asked.

Pairs with probes/pytest_probe.py.
"""

from __future__ import annotations

TITLE = "Backend suites, base versus head"
MODE = "regression"
NEEDS: list[str] = []


# pytest exit codes: 0 ok, 1 tests failed, 2 interrupted, 3 internal error,
# 4 usage error, 5 no tests collected. Only 0 and 1 mean the suite ran and judged
# something; the rest mean it never got that far.
_RAN_CODES = (0, 1)


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        rc = o.get("rc")
        collected = (o.get("n_passed", 0) or 0) + (o.get("n_failed", 0) or 0) \
            + (o.get("n_skipped", 0) or 0) + (o.get("n_errors", 0) or 0)

        # "rc is not None" used to be the whole gate. A usage error (rc 4, empty
        # stdout, reason on stderr) satisfied it, so a run that executed ZERO
        # tests was reported as no regression, green tick and all. Observed on
        # the gfx1151 runner, where the Studio venv has no pytest-timeout and the
        # probe's unconditional --timeout made pytest refuse to start.
        detail = o.get("error") or o.get("note") or f"rc={rc}"
        if rc not in _RAN_CODES:
            detail += f"; pytest exit {rc} means it never ran the tests"
            if o.get("stderr_tail"):
                detail += f"; stderr: {str(o['stderr_tail'])[:200]}"
        out.append((f"{name} suite actually ran", rc in _RAN_CODES, detail))

        # And a clean exit having collected nothing is equally vacuous: comparing
        # zero against zero always shows no regression.
        out.append((f"{name} suite collected tests", collected > 0,
                    f"{collected} collected (passed {o.get('n_passed', 0)}, "
                    f"failed {o.get('n_failed', 0)}, skipped {o.get('n_skipped', 0)}, "
                    f"errors {o.get('n_errors', 0)})"))

        # A test that errors in a fixture never ran its body, so it is neither a
        # pass nor a failure. A suite that errored most of what it selected is not
        # evidence of anything, and without this gate it reads as "0 failed".
        # Observed on the Windows runner: 25 selected, 23 errored on a missing
        # conftest import, recorded as a 2-test suite with nothing wrong.
        n_err = o.get("n_errors", 0) or 0
        out.append((f"{name} suite ran without setup errors", n_err == 0,
                    f"{n_err} test(s) errored before their body ran"
                    + (f": {', '.join(o.get('errors', [])[:5])}" if o.get("errors") else "")))

        # A COUNT with no IDS is not a comparison. head_is_worse works on id sets,
        # so whenever the short summary is missing it compares two empty sets and
        # reports a clean suite no matter how much is red. Observed on the Windows
        # gfx1151 runner (run 34821272189): `-rf -rE` in the probe asked pytest for
        # the error summary ONLY -- its -r is `store`, not `append` -- so 56 failures
        # arrived with zero ids and the verdict was a vacuous NO_REGRESSION.
        n_ids = len(o.get("failed", []) or []) + len(o.get("errors", []) or [])
        n_bad = (o.get("n_failed", 0) or 0) + n_err
        out.append((f"{name} failing tests were identified by id", n_bad == 0 or n_ids > 0,
                    f"{n_bad} failed/errored, {n_ids} id(s) captured"
                    + ("; a count without ids means the short summary was not printed"
                       if n_bad and not n_ids else "")))
    return out


def table(obs: dict) -> str:
    rows = ["| state | passed | failed | skipped | errored | failing tests |",
            "|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        failed = ", ".join(f"`{f.split('::')[-1]}`" for f in o.get("failed", [])) or "none"
        rows.append(f"| {name} | {o.get('n_passed', 0)} | {o.get('n_failed', 0)} "
                    f"| {o.get('n_skipped', 0)} | {o.get('n_errors', 0)} | {failed} |")
    note = ""
    absent = (obs.get("base") or {}).get("absent_at_this_state") or []
    if absent:
        note = ("\n\nTests absent at the base (added by this change), excluded from the "
                "comparison: " + ", ".join(f"`{a}`" for a in absent))
    return "\n".join(rows) + note


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b = set(base.get("failed", [])) | set(base.get("errors", []))
    h = set(head.get("failed", [])) | set(head.get("errors", []))
    new = sorted(h - b)
    fixed = sorted(b - h)
    if new:
        return True, ("newly failing at the head: "
                      + ", ".join(f"`{n.split('::')[-1]}`" for n in new))
    detail = "no test that passed at the base fails at the head"
    if fixed:
        detail += "; also newly passing: " + ", ".join(f"`{n.split('::')[-1]}`" for n in fixed)
    if h:
        detail += (". Note "
                   + ", ".join(f"`{n.split('::')[-1]}`" for n in sorted(h))
                   + " fails at BOTH states, so it pre-dates this change")
    return False, detail

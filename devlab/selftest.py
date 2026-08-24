#!/usr/bin/env python3
"""Self-test for the devlab toolkit.

The toolkit's whole claim is that its verdicts mean something, so the parts that
could quietly turn a non-result into a pass are the parts that need testing:
the VOID rule, the added-tests-are-not-a-regression rule, and the lint rules
that each correspond to a CI run someone already lost.

Run: python devlab/selftest.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "lib"))

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_void_rule() -> None:
    print("\nThe VOID rule (a green head with no base failure is not a pass)")
    diff = _load(ROOT / "lib" / "differential.py")

    class Crit:
        MODE = "differential"
        base_shows_defect = staticmethod(lambda o: o.get("broken", False))
        head_is_fixed = staticmethod(lambda o: not o.get("broken", False))

    v, _ = diff._decide(Crit, {"base": {"broken": False}, "head": {"broken": False}}, [])
    check("base clean + head clean -> VOID, not CONFIRMED", v == "VOID", f"got {v}")

    v, _ = diff._decide(Crit, {"base": {"broken": True}, "head": {"broken": False}}, [])
    check("base broken + head clean -> CONFIRMED", v == "CONFIRMED", f"got {v}")

    v, _ = diff._decide(Crit, {"base": {"broken": True}, "head": {"broken": True}}, [])
    check("base broken + head broken -> FIX_INCOMPLETE", v == "FIX_INCOMPLETE", f"got {v}")

    v, _ = diff._decide(Crit, {"base": {"broken": True}, "head": {"broken": False}},
                        [("a gate", False, "")])
    check("failed gate outranks everything -> INCONCLUSIVE", v == "INCONCLUSIVE", f"got {v}")

    v, _ = diff._decide(Crit, {"head": {"broken": False}}, [])
    check("missing base state -> VOID", v == "VOID", f"got {v}")


def test_added_tests_are_not_a_regression() -> None:
    print("\nAdded tests are not a regression (the trap that misread 237 -> 260)")
    crit = _load(ROOT / "criteria" / "pytest_no_regression.py")

    base = {"failed": [], "n_passed": 237, "n_failed": 0}
    head = {"failed": ["tests/t_new.py::test_added_and_failing"], "n_passed": 260, "n_failed": 1}
    worse, why = crit.head_is_worse(base, head)
    check("a NEW failing test at head is a regression", worse, why)

    base = {"failed": ["tests/t.py::flaky"], "n_passed": 100}
    head = {"failed": ["tests/t.py::flaky"], "n_passed": 123}
    worse, why = crit.head_is_worse(base, head)
    check("same failure at both states is NOT a regression", not worse, why)
    check("and it is reported as pre-dating the change", "pre-dates" in why, why)

    base = {"failed": ["tests/t.py::broken"], "n_passed": 100}
    head = {"failed": [], "n_passed": 101}
    worse, why = crit.head_is_worse(base, head)
    check("a fix is not a regression", not worse, why)
    check("and the newly passing test is named", "newly passing" in why, why)


def test_probe_skips_tests_absent_at_a_state() -> None:
    print("\nA test file added by the PR is absent at base, not a failure there")
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base" / "studio" / "backend" / "tests"
        base.mkdir(parents = True)
        (base.parent / "tests" / "t_old.py").write_text("def test_a():\n    assert True\n")
        out = Path(td) / "obs.json"
        subprocess.run([sys.executable, str(ROOT / "probes" / "pytest_probe.py"),
                        "--state", "base", "--checkout", str(Path(td) / "base"),
                        "--out", str(out), "--tests", "tests/t_old.py", "tests/t_new.py"],
                       capture_output = True)
        obs = json.loads(out.read_text())
        check("absent file is recorded, not run", obs.get("absent_at_this_state") == ["tests/t_new.py"],
              str(obs.get("absent_at_this_state")))
        check("present file is still selected", obs.get("selected") == ["tests/t_old.py"],
              str(obs.get("selected")))
        check("no spurious failure from the absent file", obs.get("n_failed", 0) == 0,
              str(obs.get("n_failed")))


def test_lint_rules_fire() -> None:
    print("\nLint rules fire on the shapes that cost real runs")
    lint = _load(ROOT / "lib" / "lint_workflow.py")

    f = lint.lint_text("x", "set -uo pipefail\nfoo bar\nrc=$?\n")
    check("E001 errexit swallowing rc", any(c == "E001" for c, _, _ in f))
    f = lint.lint_text("x", "set +e\nset -uo pipefail\nfoo bar\nrc=$?\n")
    check("E001 silent once `set +e` present", not any(c == "E001" for c, _, _ in f))

    f = lint.lint_text("x", 'pkill -f "probe_9362.py holder"\n')
    check("E002 self-matching pkill", any(c == "E002" for c, _, _ in f))

    f = lint.lint_text("x", "set -euo pipefail\nicd=$(ls /a/b*.json 2>/dev/null | head -1)\n")
    check("E005 pipefail glob despite 2>/dev/null", any(c == "E005" for c, _, _ in f))
    f = lint.lint_text("x", "set -euo pipefail\nicd=$(ls /a/b*.json 2>/dev/null | head -1 || true)\n")
    check("E005 silent with `|| true`", not any(c == "E005" for c, _, _ in f))

    f = lint.lint_text("x", "cat <<'PY' > f.txt\n  hello\n  PY\n")
    check("E003 indented heredoc terminator", any(c == "E003" for c, _, _ in f))


def main() -> int:
    print("devlab selftest")
    test_void_rule()
    test_added_tests_are_not_a_regression()
    test_probe_skips_tests_absent_at_a_state()
    test_lint_rules_fire()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all self-tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

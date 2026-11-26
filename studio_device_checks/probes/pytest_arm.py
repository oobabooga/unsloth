"""Run one tree's own unit tests under the installed Studio Python and write a per-test outcome table.
usage: pytest_arm.py --out FILE --ini INI -- <pytest args>"""

from __future__ import annotations

import os
import sys

import _dc


class _Collect:
    def __init__(self):
        self.outcomes = {}
        self.errors = {}

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
            self.outcomes[report.nodeid] = report.outcome
            if report.outcome == "failed":
                self.errors[report.nodeid] = str(report.longrepr)[-1500:]

    def pytest_collectreport(self, report):
        if report.outcome == "failed":
            self.outcomes[report.nodeid or "<collection>"] = "collect_error"
            self.errors[report.nodeid or "<collection>"] = str(report.longrepr)[-2500:]


def main() -> None:
    out = _dc.out_path()
    args = sys.argv[3:]
    ini = args[args.index("--ini") + 1]
    targets = args[args.index("--") + 1:]
    rec = {"tree": _dc.enter_tree(), "targets": targets}
    try:
        import pytest
    except Exception as exc:  # noqa: BLE001
        rec["not_run"] = f"pytest is not importable: {_dc.err(exc, 300)}"
        _dc.write(out, rec)
        return
    rec["pytest"] = pytest.__version__
    backend = _dc.tree_backend()
    col = _Collect()
    argv = ["-c", ini, "--rootdir", backend, "-p", "no:cacheprovider", "-q", "-o", "addopts=",
            "-m", "not gpu and not slow and not server and not native_mxc", *targets]
    os.chdir(backend)
    rc = pytest.main(argv, plugins = [col])
    rec["exit_code"] = int(rc)
    rec["outcomes"] = col.outcomes
    rec["errors"] = col.errors
    counts = {}
    for v in col.outcomes.values():
        counts[v] = counts.get(v, 0) + 1
    rec["counts"] = counts
    _dc.write(out, rec)


if __name__ == "__main__":
    main()

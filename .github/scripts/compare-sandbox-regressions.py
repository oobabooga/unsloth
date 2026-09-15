# SPDX-License-Identifier: AGPL-3.0-only
"""Fail on regression tests not already failing at the staging merge base."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
BASE = "4becbbd24d53027e492d9dd05c41ca5bb09f8ece"
EXISTING = ["test_account_tool_confinement.py", "test_bypass_permissions.py",
            "test_sandbox_files_and_storage_roots.py", "test_tool_output_streaming.py"]
ADDED = ["test_sandbox_linux.py", "test_sandbox_macos.py", "test_sandbox_probe.py",
         "test_tool_sandbox_wiring.py"]

def run_tests(root, names, report):
    env = dict(os.environ, PYTHONPATH=str(root / "studio/backend"))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *[f"studio/backend/tests/{name}" for name in names],
         "-q", "--tb=short", "--timeout=120", "-p", "no:cacheprovider", f"--junitxml={report}"],
        cwd=root, env=env,
    )
    if result.returncode not in (0, 1) or not report.exists():
        raise SystemExit(f"Test execution failed without a usable result: {result.returncode}")
    cases = ET.parse(report).findall(".//testcase")
    if not cases or all(case.find("skipped") is not None for case in cases):
        raise SystemExit("No tests executed")
    if any(case.find("error") is not None for case in cases):
        raise SystemExit("Test collection or setup errors must be investigated")
    failures = {(case.attrib["classname"], case.attrib["name"])
                for case in cases if case.find("failure") is not None}
    if result.returncode and not failures:
        raise SystemExit("Pytest failed without a reported test failure")
    return failures

with tempfile.TemporaryDirectory(prefix="sandbox-results-", dir=os.environ["RUNNER_TEMP"]) as tmp:
    results = Path(tmp)
    current = run_tests(ROOT, EXISTING + ADDED, results / "head.xml")
    if not current:
        print("All sandbox regression suites passed")
        raise SystemExit(0)
    baseline = results / "baseline"
    subprocess.run(["git", "fetch", "--depth=1", "origin", BASE], cwd=ROOT, check=True)
    subprocess.run(["git", "worktree", "add", "--detach", str(baseline), BASE], cwd=ROOT, check=True)
    previous = run_tests(baseline, EXISTING, results / "base.xml")
    for name in sorted(current & previous):
        print("ALSO FAILS AT BASE:", "::".join(name))
    for name in sorted(current - previous):
        print("NEW FAILURE:", "::".join(name))
    if current - previous:
        raise SystemExit(1)
    print(f"No new failures; {len(current)} failures also reproduced at base {BASE}")

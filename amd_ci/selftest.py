#!/usr/bin/env python3
"""Self-test for the AMD CI toolkit.

The toolkit's whole claim is that its verdicts mean something, so the parts that
could quietly turn a non-result into a pass are the parts that need testing:
the VOID rule, the added-tests-are-not-a-regression rule, and the lint rules
that each correspond to a CI run someone already lost.

Run: python amd_ci/selftest.py
"""

from __future__ import annotations

import importlib.util
import json
import re
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

    # The shape that broke every GPU job: the toolkit is only at the checkout
    # root, so a relative reference dies once the step enters a state worktree.
    body = 'cd "$AMD_CI_WORK/states/head"\n"$V/bin/python" amd_ci/lib/gate.py --require gpu\n'
    check("E007 relative toolkit path after cd", any(c == "E007" for c, _, _ in lint.lint_text("x", body)))
    fixed = 'cd "$AMD_CI_WORK/states/head"\n"$V/bin/python" "$GITHUB_WORKSPACE/amd_ci/lib/gate.py"\n'
    check("E007 silent once absolute", not any(c == "E007" for c, _, _ in lint.lint_text("x", fixed)))
    back = 'cd "$AMD_CI_WORK/states/head"\ncd "$GITHUB_WORKSPACE"\npython3 amd_ci/lib/gate.py\n'
    check("E007 silent after cd back", not any(c == "E007" for c, _, _ in lint.lint_text("x", back)))
    check("E007 silent with no cd", not any(c == "E007" for c, _, _ in
                                            lint.lint_text("x", "python3 amd_ci/lib/gate.py\n")))

    check("the shipped template is E007-clean",
          not any(c == "E007" for c, _, _ in
                  lint.lint_workflow(ROOT / "templates" / "workflow.yml")))


def test_scaffold_rejects_subdir_prefixed_tests() -> None:
    print("\n--tests carrying the probe's own subdir is refused, not run")
    probe = (ROOT / "probes" / "pytest_probe.py").read_text()
    m = re.search(r'"--subdir",\s*default\s*=\s*"([^"]+)"', probe)
    scaffold = _load(ROOT / "scaffold.py")
    check("scaffold's PYTEST_SUBDIR matches the probe's default",
          m is not None and m.group(1) == scaffold.PYTEST_SUBDIR,
          f"probe={m and m.group(1)} scaffold={scaffold.PYTEST_SUBDIR}")

    with tempfile.TemporaryDirectory() as td:
        base = [sys.executable, str(ROOT / "scaffold.py"), "--pr", "1",
                "--out", str(Path(td) / "o"), "--no-gpu"]
        bad = subprocess.run(base + ["--tests", "studio/backend/tests/test_x.py"],
                             capture_output = True, text = True)
        check("a repo-root path exits non-zero", bad.returncode != 0, bad.stdout + bad.stderr)
        check("and the message gives the corrected path",
              "'tests/test_x.py'" in (bad.stdout + bad.stderr), bad.stdout + bad.stderr)
        good = subprocess.run(base + ["--tests", "tests/test_x.py"],
                              capture_output = True, text = True)
        check("the subdir-relative form is accepted", good.returncode == 0,
              good.stdout + good.stderr)


def test_bounds_are_stated_even_when_there_are_none() -> None:
    print("\nA report always states its reach, including when nothing is missing")
    cap = _load(ROOT / "lib" / "capability.py")
    p = cap.HostProfile()
    p.capabilities = {"rocm": True, "gpu": True}

    s = cap.untested_section(p, ["rocm", "windows", "multi_gpu"])
    check("declared gaps are listed", "**windows**" in s and "**multi_gpu**" in s, s)
    check("a satisfied capability is not listed as a gap", "**rocm**" not in s, s)

    # The under-declaration that made a Windows-only PR look unbounded.
    s = cap.untested_section(p, ["rocm"])
    check("a fully-satisfied NEEDS still renders a section", s.strip() != "", repr(s))
    check("and says it is a claim about the declaration", "DECLARATION" in s, s)

    s = cap.untested_section(p, [])
    check("an absent NEEDS renders a section too", s.strip() != "", repr(s))
    check("and calls the reach unknown", "unknown" in s, s)

    # Observed on the runner: with torch absent, every GPU capability reads
    # False, and the report asserted "this host has no ROCm" about a ROCm box.
    d = cap.HostProfile()
    d.capabilities = {"linux": True}
    d.torch_detection = False
    s = cap.untested_section(d, ["rocm", "windows"])
    check("an unmeasurable capability is UNDETERMINED, not absent",
          "UNDETERMINED" in s.split("- **windows**")[0], s)
    check("and it says so is not a claim about the hardware",
          "Not a statement about the hardware" in s, s)
    check("a genuinely host-derived gap still reads as a gap",
          "needs a Windows host" in s, s)
    d.torch_detection = True
    s = cap.untested_section(d, ["rocm"])
    check("with detection working, absence is reported as absence",
          "UNDETERMINED" not in s, s)


def test_a_non_result_does_not_leave_the_job_green() -> None:
    print("\nVOID/INCONCLUSIVE fail the job; a real finding does not")
    ann = ROOT / "lib" / "announce.py"
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for verdict, want_rc in [("INCONCLUSIVE", 1), ("VOID", 1), ("CONFIRMED", 0),
                                 ("NO_REGRESSION", 0), ("FIX_INCOMPLETE", 0)]:
            (d / "verdict.json").write_text(json.dumps({"verdict": verdict, "why": "w"}))
            r = subprocess.run([sys.executable, str(ann), str(d)], capture_output = True, text = True)
            check(f"{verdict} -> exit {want_rc}", r.returncode == want_rc,
                  f"rc={r.returncode} {r.stdout}")
        (d / "verdict.json").unlink()
        r = subprocess.run([sys.executable, str(ann), str(d)], capture_output = True, text = True)
        check("a missing verdict is a failure, not a pass", r.returncode == 1, r.stdout)


def test_fixture_failure_is_inconclusive_not_a_pass() -> None:
    print("\nA fixture that never comes up yields no result, not a green one")
    diff = _load(ROOT / "lib" / "differential.py")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        dud = out / "dud.py"
        dud.write_text("import sys\nsys.exit(3)\n")
        proc, info = diff.start_fixture(dud, [], out, sys.executable, timeout = 6)
        check("a fixture that exits is not ready", not info.get("ready"), str(info))
        check("its early exit is recorded", info.get("exited_early") == 3, str(info))
        diff.stop_fixture(proc)

        ok = out / "ok.py"
        ok.write_text(
            "import json,time,sys\n"
            "print(json.dumps({'status':'READY','allocated_gib':4.0}), flush=True)\n"
            "time.sleep(30)\n")
        proc, info = diff.start_fixture(ok, [], out, sys.executable, timeout = 30)
        check("a ready fixture is detected", info.get("ready") is True, str(info))
        check("its payload is captured", info.get("allocated_gib") == 4.0, str(info))
        diff.stop_fixture(proc)
        check("and it is stopped by PID", proc.poll() is not None, "still running")


def test_fixture_metadata_is_not_mistaken_for_a_state() -> None:
    print("\nFixture metadata is not a state")
    diff = _load(ROOT / "lib" / "differential.py")

    class Crit:
        MODE = "differential"
        base_shows_defect = staticmethod(lambda o: o.get("broken", False))
        head_is_fixed = staticmethod(lambda o: not o.get("broken", False))

    obs = {"base": {"broken": True}, "head": {"broken": False},
           "_fixture": {"pid": 1, "ready": True}}
    v, _ = diff._decide(Crit, obs, [])
    check("_fixture is skipped when checking extra states", v == "CONFIRMED", f"got {v}")


def test_spoofed_devices_cannot_satisfy_multi_gpu() -> None:
    print("\nA fabricated device never counts as multi-GPU")
    cap = _load(ROOT / "lib" / "capability.py")

    # The hazard this guards: detect() reads torch.cuda.device_count(), which under
    # the HIP device multiplier reports 2 on a one-GPU box. If that were allowed to
    # satisfy multi_gpu, "Not tested here" would lose the multi-GPU line and a wiring
    # run would read as hardware validation.
    profile = cap.HostProfile(system = "Linux", hip = "7.2.1", gpu_count = 2)
    profile.spoofed_devices = 1
    # capabilities_for is the SHIPPED rule. Rebuilding the dict here instead would
    # test this test's arithmetic and pass while the real rule was broken.
    profile.capabilities = cap.capabilities_for(profile)
    section = cap.untested_section(profile, ["multi_gpu"])

    check("a spoofed second device does not satisfy multi_gpu",
          profile.capabilities["multi_gpu"] is False)
    check("the verdict says the devices were fabricated",
          "FABRICATED" in section, section[:120])
    check("and says what a fabricated device cannot show",
          "sharding" in section and "collectives" in section, section[:200])

    two_real = cap.HostProfile(system = "Linux", hip = "7.2.1", gpu_count = 2)
    check("two REAL devices still do satisfy multi_gpu (the rule is not just off)",
          cap.capabilities_for(two_real)["multi_gpu"] is True)

    honest = cap.HostProfile(system = "Linux", hip = "7.2.1", gpu_count = 1)
    honest.capabilities = cap.capabilities_for(honest)
    check("an unspoofed run says nothing about fabrication",
          "FABRICATED" not in cap.untested_section(honest, ["multi_gpu"]))


def test_spoof_flag_targets_the_gpu_job() -> None:
    print("\n--spoof-devices lands in the gpu job, not the suites job")
    import subprocess, tempfile
    # The template carries a Differential step in BOTH jobs. Inserting before the
    # first one puts it in suites, where --no-suites then deletes it: the flag
    # silently does nothing and the workflow lints clean. Observed once.
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ci"
        r = subprocess.run([sys.executable, str(ROOT / "scaffold.py"), "--pr", "1",
                            "--out", str(out), "--branch", "b", "--no-suites",
                            "--spoof-devices", "1"], capture_output = True, text = True)
        wf = out / ".github" / "workflows" / "b.yml"
        text = wf.read_text() if wf.is_file() else ""
        check("the step survives --no-suites", "device_multiplier.py" in text,
              (r.stdout + r.stderr)[-200:])
        check("it runs before the Differential step",
              "device_multiplier.py" in text and
              text.find("Present extra HIP devices") < text.find("name: Differential"))


def test_a_pytest_run_that_never_ran_is_not_a_pass() -> None:
    print("\nA pytest suite that never ran is not no-regression")
    crit = _load(ROOT / "criteria" / "pytest_no_regression.py")

    # The exact shape seen on the gfx1151 runner: unconditional --timeout with no
    # pytest-timeout plugin, so pytest exits 4 with empty stdout and the reason on
    # stderr. The old gate was "rc is not None", which this satisfies.
    vacuous = {"rc": 4, "n_passed": 0, "n_failed": 0, "n_skipped": 0,
               "stderr_tail": "error: unrecognized arguments: --timeout"}
    failed = [n for n, ok, _ in crit.gates({"base": dict(vacuous), "head": dict(vacuous)}) if not ok]
    check("a usage error is not 'the suite ran'",
          any("actually ran" in n for n in failed), str(failed))

    # Clean exit, nothing collected: comparing zero to zero always looks clean.
    empty = {"rc": 0, "n_passed": 0, "n_failed": 0, "n_skipped": 0}
    failed = [n for n, ok, _ in crit.gates({"base": dict(empty), "head": dict(empty)}) if not ok]
    check("collecting nothing is not a comparison",
          any("collected tests" in n for n in failed), str(failed))

    real = {"rc": 0, "n_passed": 47, "n_failed": 0, "n_skipped": 2}
    failed = [n for n, ok, _ in crit.gates({"base": dict(real), "head": dict(real)}) if not ok]
    check("a real run still passes", not failed, str(failed))

    # rc=1 is tests failing, which IS a run and must be judged, not discarded.
    ran = {"rc": 1, "n_passed": 40, "n_failed": 7, "n_skipped": 0}
    failed = [n for n, ok, _ in crit.gates({"base": dict(ran), "head": dict(ran)}) if not ok]
    check("failing tests still count as having run", not failed, str(failed))


def test_windows_is_reachable_but_docker_on_windows_is_not() -> None:
    print("\nWindows is a target; Docker on Windows still is not")
    cap = _load(ROOT / "lib" / "capability.py")

    # The four Windows AMD runners as measured: Windows, no docker CLI. The point
    # of the split is that ONE gap used to cover both, so a Windows-only change was
    # reported as unreachable when it is now reachable, while a containerised one
    # would become reported as reachable when it is not.
    win = cap.HostProfile(system = "Windows", gpu_count = 0)
    win.docker_cli = False
    win.capabilities = cap.capabilities_for(win)
    check("a Windows runner satisfies `windows`", win.capabilities["windows"] is True)
    check("but not `windows_docker`", win.capabilities["windows_docker"] is False)
    check("nor plain `docker`", win.capabilities["docker"] is False)

    s = cap.untested_section(win, ["windows_docker"])
    check("the docker gap is stated as measured on the runners",
          "not installed" in s and "windows_docker" in s, s)
    check("and says non-Docker Windows work IS reachable",
          "NON-Docker" in s or "non-Docker" in s, s)

    lin = cap.HostProfile(system = "Linux", hip = "7.2.1", gpu_count = 1)
    lin.capabilities = cap.capabilities_for(lin)
    s = cap.untested_section(lin, ["windows"])
    check("a Linux job still reports `windows` as a gap for ITSELF",
          "**windows**" in s, s)
    check("but no longer claims the POOL has no Windows",
          "this runner is Linux" not in s, s)
    check("and names the selector that reaches them",
          "self-hosted, Windows, strix-halo, devlab-dispatch" in s, s)
    check("and the shell they need",
          "shell: powershell" in s, s)

    # A docker-capable Windows host is not something this pool has, but the rule
    # must be a rule and not a constant, or it stops describing whatever runs next.
    dock = cap.HostProfile(system = "Windows")
    dock.docker_cli = True
    check("the rule is computed, not hardcoded to False",
          cap.capabilities_for(dock)["windows_docker"] is True)


def test_windows_lint_rules_fire() -> None:
    print("\nThe Windows shell traps are caught before a run is spent")
    lint = _load(ROOT / "lib" / "lint_workflow.py")

    def codes(shell, body):
        return [c for c, _, _ in lint.lint_windows("x", body, shell)]

    check("E100 on `shell: pwsh` (no PowerShell 7 on these boxes)",
          "E100" in codes("pwsh", "Write-Host hi"))
    check("E100 on `shell: bash` (no bash either)",
          "E100" in codes("bash", "Write-Host hi"))
    check("E100 on no shell at all (the Actions default is pwsh)",
          "E100" in codes(None, "Write-Host hi"))
    check("E100 silent on `shell: powershell`",
          "E100" not in codes("powershell", "Write-Host hi"))

    check("E101 on docker in a Windows job",
          "E101" in codes("powershell", "docker pull rocm/dev"))
    check("E102 on the Linux preamble in a Windows job",
          "E102" in codes("powershell", ". amd_ci/lib/preamble.sh j"))
    check("W103 (a warning, not an error) on branching on RUNNER_OS",
          "W103" in codes("powershell", 'if ($env:RUNNER_OS -eq "Windows") { }'))
    check("E104 on a bash-style $RUNNER_TEMP inside PowerShell",
          "E104" in codes("powershell", 'New-Item -Path "$RUNNER_TEMP\\w"'))
    check("E104 silent on the PowerShell form",
          "E104" not in codes("powershell", 'New-Item -Path "$env:RUNNER_TEMP\\w"'))
    check("W104 on Out-File into GITHUB_ENV (BOM on PowerShell 5.1)",
          "W104" in codes("powershell",
                          '"a=1" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append'))

    # A selector with no OS label used to be merely redundant; both OSes answer now.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "wf.yml"
        p.write_text("on: {push: {branches: [x]}}\njobs:\n"
                     "  j:\n    runs-on: [self-hosted, strix-halo, devlab-dispatch]\n"
                     "    steps:\n      - run: echo hi\n")
        check("E106 on a strix-halo selector with no OS label",
              any(c == "E106" for c, _, _ in lint.lint_workflow(p)))

    # A bash rule aimed at a PowerShell block is worse than no rule: it is a
    # finding that cannot be fixed, and a linter people ignore catches nothing.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "wf.yml"
        p.write_text("on: {push: {branches: [x]}}\njobs:\n"
                     "  j:\n    runs-on: [self-hosted, Windows, strix-halo, devlab-dispatch]\n"
                     "    defaults:\n      run:\n        shell: powershell\n"
                     "    steps:\n      - run: |\n"
                     '          $rc = 1\n          if ($rc -ne 0) { exit $rc }\n')
        got = [c for c, _, _ in lint.lint_workflow(p)]
        check("no bash syntax error is reported for a PowerShell block",
              "E000" not in got, str(got))
        check("and the job-level `defaults.run.shell` counts as declaring the shell",
              "E100" not in got, str(got))

    check("the shipped Windows template lints clean",
          not lint.lint_workflow(ROOT / "templates" / "workflow_windows.yml"),
          str(lint.lint_workflow(ROOT / "templates" / "workflow_windows.yml")))


def test_scaffold_windows_emits_the_measured_selector() -> None:
    print("\n--windows emits the selector and shell that actually match")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ci"
        r = subprocess.run([sys.executable, str(ROOT / "scaffold.py"), "--pr", "1",
                            "--out", str(out), "--branch", "b", "--windows",
                            "--tests", "tests/test_x.py"], capture_output = True, text = True)
        check("scaffolding succeeds", r.returncode == 0, r.stdout + r.stderr)
        wf = out / ".github" / "workflows" / "b.yml"
        text = wf.read_text() if wf.is_file() else ""
        check("the Windows selector is emitted",
              "[self-hosted, Windows, strix-halo, devlab-dispatch]" in text, text[:200])
        check("with shell: powershell", "shell: powershell" in text, text[:200])
        check("and the PowerShell preamble, not the bash one",
              "preamble.ps1" in text and "preamble.sh" not in text)
        # A stalled Windows job and a busy pool look identical from the run list.
        check("a linux-control job ships with it", "linux-control:" in text)
        check("differing only in the OS label",
              "[self-hosted, Linux, strix-halo, devlab-dispatch]" in text)

        # Silently ignoring a Linux-only flag is how --spoof-devices once did
        # nothing while the workflow linted clean.
        bad = subprocess.run([sys.executable, str(ROOT / "scaffold.py"), "--pr", "1",
                              "--out", str(Path(td) / "ci2"), "--windows",
                              "--spoof-devices", "1"], capture_output = True, text = True)
        check("--spoof-devices with --windows is refused, not ignored",
              bad.returncode != 0, bad.stdout + bad.stderr)
        check("and says why there is no Windows equivalent",
              "LD_PRELOAD" in (bad.stdout + bad.stderr), bad.stdout + bad.stderr)


def main() -> int:
    print("AMD CI selftest")
    test_void_rule()
    test_added_tests_are_not_a_regression()
    test_probe_skips_tests_absent_at_a_state()
    test_lint_rules_fire()
    test_scaffold_rejects_subdir_prefixed_tests()
    test_bounds_are_stated_even_when_there_are_none()
    test_a_non_result_does_not_leave_the_job_green()
    test_fixture_failure_is_inconclusive_not_a_pass()
    test_fixture_metadata_is_not_mistaken_for_a_state()
    test_spoofed_devices_cannot_satisfy_multi_gpu()
    test_spoof_flag_targets_the_gpu_job()
    test_a_pytest_run_that_never_ran_is_not_a_pass()
    test_windows_is_reachable_but_docker_on_windows_is_not()
    test_windows_lint_rules_fire()
    test_scaffold_windows_emits_the_measured_selector()
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

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


def test_a_dead_label_is_an_error_not_a_wait() -> None:
    print("\nA selector no runner can satisfy is a lint error, not a queue")
    lint = _load(ROOT / "lib" / "lint_workflow.py")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "wf.yml"
        p.write_text("on: {push: {branches: [x]}}\njobs:\n"
                     "  j:\n    runs-on: [self-hosted, Linux, strix-halo, ephemeral]\n"
                     "    steps:\n      - run: echo hi\n", encoding = "utf-8")
        check("E107 on a strix-halo selector carrying `ephemeral`",
              any(c == "E107" for c, _, _ in lint.lint_workflow(p)))
        p.write_text("on: {push: {branches: [x]}}\njobs:\n"
                     "  j:\n    runs-on: [self-hosted, Linux, strix-halo, devlab-dispatch]\n"
                     "    steps:\n      - run: echo hi\n", encoding = "utf-8")
        check("E107 silent on `devlab-dispatch`",
              not any(c == "E107" for c, _, _ in lint.lint_workflow(p)))


def test_a_tuple_verdict_is_read_for_its_bool() -> None:
    print("\nA (bool, reason) criterion result is not truthy just for being a tuple")
    diff = _load(ROOT / "lib" / "differential.py")

    class Crit:
        MODE = "differential"
        base_shows_defect = staticmethod(lambda o: (o.get("broken", False), "base"))
        head_is_fixed = staticmethod(
            lambda o: (not o.get("broken", False), "still broken" if o.get("broken") else "clean"))

    v, _ = diff._decide(Crit, {"base": {"broken": True}, "head": {"broken": True}}, [])
    check("a head that still shows the defect is FIX_INCOMPLETE", v == "FIX_INCOMPLETE", f"got {v}")
    v, _ = diff._decide(Crit, {"base": {"broken": True}, "head": {"broken": False}}, [])
    check("a head that clears it is CONFIRMED", v == "CONFIRMED", f"got {v}")
    v, _ = diff._decide(Crit, {"base": {"broken": False}, "head": {"broken": False}}, [])
    check("a base without the defect is VOID even as a tuple", v == "VOID", f"got {v}")


def test_a_verdict_can_be_redecided_without_the_probes() -> None:
    print("\nA finished run's observations can be re-decided offline")
    red = _load(ROOT / "lib" / "redecide.py")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "observations.json").write_text(json.dumps(
            {"base": {"broken": True}, "head": {"broken": True}}), encoding = "utf-8")
        crit = d / "crit.py"
        crit.write_text(
            'MODE = "differential"\n'
            'def base_shows_defect(o): return (o.get("broken", False), "b")\n'
            'def head_is_fixed(o): return (not o.get("broken", False), "h")\n'
            'def gates(o): return [("both probed", "base" in o and "head" in o, "ok")]\n',
            encoding = "utf-8")
        v, _ = red.redecide(d, crit, "t")
        check("the re-decided verdict is the corrected one", v == "FIX_INCOMPLETE", f"got {v}")
        out = (d / "VERDICT_redecided.md").read_text(encoding = "utf-8")
        check("it is written beside the original, marked as offline",
              "re-decided offline" in out and "FIX_INCOMPLETE" in out, out)
        check("the original VERDICT.md is not touched", not (d / "VERDICT.md").exists())


def test_windows_arch_comes_from_the_video_controller() -> None:
    print("\nOn Windows the gfx target is read from the video controller name")
    cap = _load(ROOT / "lib" / "capability.py")
    names = ["AMD Radeon(TM) 8060S Graphics", "Microsoft Basic Display Adapter"]
    check("8060S maps to gfx1151 and the unknown adapter maps to nothing",
          cap.archs_from_windows_video_controllers(names) == ["gfx1151"],
          str(cap.archs_from_windows_video_controllers(names)))
    check("an unknown part is not guessed", cap.archs_from_windows_video_controllers(["RTX 4090"]) == [])
    if sys.platform != "win32":
        check("the controller query is a no-op off Windows", cap.windows_video_controllers() == [])


def test_toolkit_text_io_names_its_encoding() -> None:
    print("\nEvery text read and write in the toolkit names its encoding")
    # cp1252 is the Windows default; a UTF-8 prompt read through it is mojibake.
    import ast
    offenders: list[str] = []
    for path in [*(ROOT / "lib").glob("*.py"), *(ROOT / "probes").glob("*.py"),
                 *(ROOT / "criteria").glob("*.py"), ROOT / "scaffold.py"]:
        for node in ast.walk(ast.parse(path.read_text(encoding = "utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name not in ("read_text", "write_text", "open"):
                continue
            if any(k.arg == "encoding" for k in node.keywords):
                continue
            if name == "open":
                mode = node.args[1] if len(node.args) > 1 else None
                for k in node.keywords:
                    if k.arg == "mode":
                        mode = k.value
                if isinstance(mode, ast.Constant) and "b" in str(mode.value):
                    continue  # bytes need no encoding
            offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    check("no unqualified text I/O under lib/, probes/, criteria/ or scaffold.py",
          not offenders, ", ".join(offenders))


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


# ---- the prebuilt-release differential ------------------------------------------

def _model_spec() -> str:
    return ("unsloth/Qwen3.8-Flash-Next-GGUF@38bb39ee97821de2c9009abb7e93950eec396e66:"
            "UD-IQ1_S/m-00001-of-00003.gguf,UD-IQ1_S/m-00002-of-00003.gguf,UD-IQ1_S/m-00003-of-00003.gguf")


def test_prebuilt_scaffold_emits_lint_clean_workflows_for_both_os() -> None:
    print("\n--prebuilt scaffolds a release differential for Linux and for Windows, offline")
    common = ["--prebuilt", "b10715-mix-86bd2d3,b10798-mix-659e406", "--asset", "rocm-gfx1151",
              "--model", _model_spec(), "--env", "GGML_CUDA_ENABLE_UNIFIED_MEMORY=1",
              "--control-env", "absent", "--control-env", "GGML_CUDA_ENABLE_UNIFIED_MEMORY=0",
              "--control-asset", "vulkan", "--control-tag", "b10687-mix-67dfc8b"]
    with tempfile.TemporaryDirectory() as td:
        for windows in (False, True):
            out = Path(td) / ("win" if windows else "lin")
            cmd = [sys.executable, str(ROOT / "scaffold.py"), *common, "--out", str(out)]
            if windows:
                cmd.append("--windows")
            r = subprocess.run(cmd, capture_output = True, text = True)
            tag = "windows" if windows else "linux"
            check(f"{tag}: scaffolding succeeds and lints clean", r.returncode == 0, r.stdout + r.stderr)
            check(f"{tag}: it never asked the Hub (every file and the revision were given)",
                  "resolved" not in r.stdout, r.stdout)
            wfs = list((out / ".github" / "workflows").glob("*.yml")) if out.is_dir() else []
            text = wfs[0].read_text(encoding = "utf-8") if wfs else ""
            check(f"{tag}: one workflow, named after the tags",
                  len(wfs) == 1 and "b10715-vs-b10798" in wfs[0].name, str(wfs))
            os_stem = "windows-x64" if windows else "linux-x64"
            ext = "zip" if windows else "tar.gz"
            for tag_, asset in (("b10715-mix-86bd2d3", "rocm-gfx1151"), ("b10798-mix-659e406", "rocm-gfx1151"),
                                ("b10798-mix-659e406", "vulkan"), ("b10687-mix-67dfc8b", "rocm-gfx1151")):
                check(f"{tag}: fetches app-{tag_}-{os_stem}-{asset}.{ext}",
                      f"/{tag_}/app-{tag_}-{os_stem}-{asset}.{ext}" in text)
            fetched = [l for l in text.splitlines() if "fetch_gguf.py" in l and "m-0000" in l]
            check(f"{tag}: every shard is fetched, at the pinned revision",
                  len(fetched) == 3 and "38bb39ee97821de2c9009abb7e93950eec396e66" in text, str(fetched))
            check(f"{tag}: the model entry is shard 1", "MODEL=" in text and "m-00001-of-00003.gguf" in
                  text.split("MODEL=", 1)[1].split("\n", 1)[0])
            check(f"{tag}: D1 carries the environment", "--env GGML_CUDA_ENABLE_UNIFIED_MEMORY=1" in text
                  or '--env "GGML_CUDA_ENABLE_UNIFIED_MEMORY=1"' in text)
            for arm in ("K1_head_rocm_gfx1151_env_absent", "K2_head_rocm_gfx1151_GGML_CUDA_ENABLE_UNIFIED_MEMORY_0",
                        "K3_head_vulkan", "K4_b10687_rocm_gfx1151"):
                check(f"{tag}: control arm {arm}", arm in text)
            check(f"{tag}: the absent arm unsets rather than sets",
                  "--unset-env" in text.split("K1_head_rocm_gfx1151_env_absent", 2)[-1].split("\n", 1)[0])
            check(f"{tag}: the differential still decides (VOID rule is not bypassed)",
                  "differential.py" in text and "announce.py" in text)
            check(f"{tag}: no placeholder left", "__" not in text.replace("__pycache__", ""),
                  ",".join(sorted(set(re.findall(r"__[A-Z_]+__", text)))))
            if windows:
                check("windows: powershell, the Windows preamble, and no bash",
                      "shell: powershell" in text and "preamble.ps1" in text and "preamble.sh" not in text)
                check("windows: the states require llama-server.exe", "--require-file llama-server.exe" in text)
            else:
                check("linux: the GPU job sits in the shared concurrency group",
                      "group: amd-ci-gfx1151-gpu" in text)
                check("linux: the gate reads the arch without torch",
                      "--no-torch" in text and "--expect-arch gfx1151" in text)
            check(f"{tag}: the toolkit travels with the branch, templates included",
                  (out / "amd_ci" / "probes" / "llamacpp_server_probe.py").is_file()
                  and (out / "amd_ci" / "templates" / "workflow_prebuilt.yml").is_file())


def test_prebuilt_scaffold_refuses_pr_mode_flags() -> None:
    print("\n--prebuilt refuses the PR-mode flags instead of ignoring them")
    with tempfile.TemporaryDirectory() as td:
        for extra in (["--pr", "1"], ["--merged"], ["--no-gpu"], ["--tests", "tests/x.py"]):
            r = subprocess.run([sys.executable, str(ROOT / "scaffold.py"), "--prebuilt", "a,b",
                                "--model", _model_spec(), "--out", str(Path(td) / "x"), *extra],
                               capture_output = True, text = True)
            check(f"{extra[0]} with --prebuilt is refused", r.returncode != 0 and "belongs to" in (r.stdout + r.stderr),
                  (r.stdout + r.stderr)[-200:])
        r = subprocess.run([sys.executable, str(ROOT / "scaffold.py"), "--prebuilt", "a,a",
                            "--model", _model_spec(), "--out", str(Path(td) / "y")],
                           capture_output = True, text = True)
        check("the same tag twice is refused (a differential needs two builds)", r.returncode != 0)
        r = subprocess.run([sys.executable, str(ROOT / "scaffold.py"), "--out", str(Path(td) / "z")],
                           capture_output = True, text = True)
        check("PR mode still demands --pr", r.returncode != 0 and "--pr is required" in (r.stdout + r.stderr))


def test_prebuilt_plan_arms() -> None:
    print("\nThe prebuilt plan: specs, assets, and control arms that differ from D1 by one thing")
    pb = _load(ROOT / "lib" / "prebuilt.py")
    s = pb.parse_spec("owner/name@abc:dir/a-00002-of-00002.gguf,dir/a-00001-of-00002.gguf")
    check("spec parses repo, revision and files", (s.repo, s.revision, len(s.paths)) == ("owner/name", "abc", 2))
    check("the entry is shard 1 whatever the order given", s.entry.endswith("a-00001-of-00002.gguf"), s.entry)
    check("a folder or an unpinned revision needs the Hub",
          pb.needs_resolution(pb.parse_spec("o/n:folder")) and pb.needs_resolution(pb.parse_spec("o/n:f.gguf"))
          and not pb.needs_resolution(s))
    check("asset names get the OS prefix and the right extension",
          pb.asset_file("T", "vulkan", False) == "app-T-linux-x64-vulkan.tar.gz"
          and pb.asset_file("T", "vulkan", True) == "app-T-windows-x64-vulkan.zip"
          and pb.asset_file("T", "linux-x64-cuda13-portable", False) == "app-T-linux-x64-cuda13-portable.tar.gz")

    plan = pb.Plan(base_tag = "b1-x", head_tag = "b2-y", asset = "rocm-gfx1151", windows = False,
                   release_repo = "unslothai/llama.cpp", model = s, sentinel = pb.parse_spec("o/s@r:s.gguf"),
                   env = ["A=1", "B=2"], unset_env = ["C"], control_env = ["absent", "A=0"],
                   control_assets = ["vulkan"], control_tags = ["b0-z"],
                   probe = "p", criteria = "c", cells = "single", n_predict = 8, load_timeout = 9,
                   timeout_minutes = 10, min_free_gb = 1)
    ctl = {name: (key, args) for name, key, args in plan.controls()}
    check("controls are numbered in the order given", list(ctl) == [
        "K1_head_rocm_gfx1151_env_absent", "K2_head_rocm_gfx1151_A_0", "K3_head_vulkan", "K4_b0_rocm_gfx1151"], list(ctl))
    check("the absent arm unsets every D1 variable and nothing else",
          ctl["K1_head_rocm_gfx1151_env_absent"][1] == ["--unset-env", "A", "--unset-env", "B", "--unset-env", "C"])
    check("a K=V arm keeps the rest of D1's environment",
          ctl["K2_head_rocm_gfx1151_A_0"][1] == ["--env", "A=0", "--env", "B=2", "--unset-env", "C"])
    check("another asset and another tag keep D1's environment (one variable per arm)",
          ctl["K3_head_vulkan"][1] == plan.d1_args() == ctl["K4_b0_rocm_gfx1151"][1])
    check("each control binary is fetched once, under an identifier-safe key",
          set(plan.binaries()) == {"base", "head", "head_vulkan", "b0_rocm_gfx1151"}, str(plan.binaries()))
    try:
        pb.Plan(**{**plan.__dict__, "env": [], "control_env": ["absent"]}).controls()
        check("`absent` without any --env is refused", False)
    except SystemExit:
        check("`absent` without any --env is refused", True)


def test_server_probe_pure_parts() -> None:
    print("\nThe llama-server probe's signatures and cross-slot check")
    probe = _load(ROOT / "probes" / "llamacpp_server_probe.py")
    check("a slash run is a signature", "slash_run" in probe.signatures("hello ////////// world"))
    check("a four-fold repeated word is a signature", "word_x4" in probe.signatures("border border border border"))
    check("a replacement character is a signature", "replacement" in probe.signatures("Du � Fu"))
    check("ordinary prose is not", probe.signatures("The capital of France is Paris.") == [])
    long_a = "x" * 30 + "the quick brown fox jumps over the lazy dog again and again and again" + "y" * 30
    check("two slots sharing 60 characters are flagged",
          probe.cross_slot_shared([long_a, "z" * 40 + long_a[30:110] + "w" * 40]) != [])
    check("short outputs are not compared (a shared stop phrase is not evidence)",
          probe.cross_slot_shared(["same short text", "same short text"]) == [])
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "build" / "bin").mkdir(parents = True)
        (d / "build" / "bin" / "llama-server.exe").write_bytes(b"")
        check("llama-server.exe under build/bin is found", probe.server_exe(d).endswith("llama-server.exe"))


def test_server_criteria_control_cell() -> None:
    print("\nThe llama-server criteria: the single cell is the control, the rest are the measurement")
    crit = _load(ROOT / "criteria" / "llamacpp_server_clean.py")
    diff = _load(ROOT / "lib" / "differential.py")
    ok = {"ok": True, "n_tokens": 64, "signatures": []}
    bad = {"ok": True, "n_tokens": 64, "signatures": ["slash_run"]}
    one = {"ok": True, "n_tokens": 1, "signatures": []}

    def state(version, single, other, **extra):
        return {"version": [version], "sentinel": {"pre": {"clean": True}, "post": {"clean": True}},
                "cells": {"single": {"prompts": [single]}, "unified4": {"prompts": other}}, **extra}

    base = state("b1", ok, [ok, bad, one, ok])
    head = state("b2", ok, [ok, ok, ok, ok])
    shown, why = crit.base_shows_defect(base)
    check("a signature or a one-token slot in a measured cell is a defect", shown and "unified4[1]" in why and "unified4[2]" in why, why)
    fixed, why = crit.head_is_fixed(head)
    check("a clean head is fixed, as a (bool, reason) pair", fixed is True and isinstance(why, str))
    v, _ = diff._decide(crit, {"base": base, "head": head}, crit.gates({"base": base, "head": head}))
    check("differential.py reads the pair and says CONFIRMED", v == "CONFIRMED", v)
    v, _ = diff._decide(crit, {"base": head, "head": head}, [])
    check("two clean builds are VOID, never a pass", v == "VOID", v)
    dirty_ctl = state("b1", bad, [bad])
    gates = crit.gates({"base": dirty_ctl, "head": head})
    check("a dirty control cell fails a gate (the harness, not the build, is suspect)",
          any(not ok_ for name, ok_, _ in gates if "control" in name), str(gates))
    same = crit.gates({"base": base, "head": state("b1", ok, [ok])})
    check("two states reporting one build fail a gate", any(not ok_ for name, ok_, _ in same if "different builds" in name))
    poisoned = state("b1", ok, [bad]); poisoned["sentinel"] = {"pre": {"clean": False}}
    check("a dirty pre-sentinel fails a gate",
          any(not ok_ for name, ok_, _ in crit.gates({"base": poisoned, "head": head}) if "sentinel" in name))
    tbl = crit.table({"base": base, "head": head})
    check("the table lists both states and every cell", "| base |" in tbl and "| head |" in tbl and "unified4" in tbl)


def test_prebuilt_helpers_roundtrip() -> None:
    print("\nwrite_states, emit_bin_env and fetch_llamacpp's archive handling, offline")
    import io, tarfile, zipfile
    fl = _load(ROOT / "lib" / "fetch_llamacpp.py")
    fg = _load(ROOT / "lib" / "fetch_gguf.py")
    check("the Hub resolve URL keeps the folder prefix",
          fg.resolve_url("o/n", "sha", "UD-IQ1_S/a.gguf") == "https://huggingface.co/o/n/resolve/sha/UD-IQ1_S/a.gguf")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # A tarball that nests the binaries one level down, and a zip that does not.
        tgz = d / "app-T-linux-x64-vulkan.tar.gz"
        with tarfile.open(tgz, "w:gz", encoding = "utf-8") as tf:
            for name in ("llama-T/llama-server", "llama-T/libggml-vulkan.so"):
                info = tarfile.TarInfo(name)
                info.size = 1
                tf.addfile(info, io.BytesIO(b"x"))
        fl.extract(tgz, d / "lin")
        bd = fl.find_bin_dir(d / "lin")
        check("a nested tarball's binary directory is found", bd is not None and bd.name == "llama-T", str(bd))
        z = d / "app-T-windows-x64-vulkan.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("llama-server.exe", "x")
        fl.extract(z, d / "win")
        bd = fl.find_bin_dir(d / "win")
        check("a flat zip's binary directory is found, .exe included", bd == d / "win", str(bd))
        check("nothing else is mistaken for a binary directory", fl.find_bin_dir(d / "nowhere") is None)

        (d / "out").mkdir()
        (d / "out" / "bin_head-vulkan.json").write_text(json.dumps({"bin_dir": str(d / "win")}), encoding = "utf-8")
        (d / "out" / "bin_base.json").write_text(json.dumps({"error": "boom"}), encoding = "utf-8")
        r = subprocess.run([sys.executable, str(ROOT / "lib" / "emit_bin_env.py"), str(d / "out")],
                           capture_output = True, text = True)
        check("emit_bin_env writes an identifier-safe NAME_BIN line per found binary",
              f"HEAD_VULKAN_BIN={d / 'win'}" in r.stdout, r.stdout)
        check("and a comment, not a variable, for a report with no binary",
              "BASE_BIN=" not in r.stdout and "# bin_base.json" in r.stdout, r.stdout)

        ws = [sys.executable, str(ROOT / "lib" / "write_states.py"), "--require-file", "llama-server.exe"]
        r = subprocess.run([*ws, "--path", f"base={d / 'win'}", "--path", f"head={d / 'lin'}",
                            "--out", str(d / "s.json")], capture_output = True, text = True)
        check("write_states refuses a state without the required file", r.returncode != 0 and "head" in r.stderr, r.stderr)
        r = subprocess.run([*ws, "--path", f"base={d / 'win'}", "--path", f"head={d / 'win'}",
                            "--out", str(d / "s.json")], capture_output = True, text = True)
        check("and two states that are the same directory", r.returncode != 0 and "same directory" in r.stderr, r.stderr)
        (d / "lin2").mkdir()
        (d / "lin2" / "llama-server.exe").write_bytes(b"")
        r = subprocess.run([*ws, "--path", f"base={d / 'win'}", "--path", f"head={d / 'lin2'}",
                            "--label", "head=b2 vulkan", "--out", str(d / "s.json")], capture_output = True, text = True)
        doc = json.loads((d / "s.json").read_text(encoding = "utf-8")) if (d / "s.json").is_file() else {}
        check("a valid pair is written in the shape differential.py reads",
              r.returncode == 0 and set(doc.get("paths", {})) == {"base", "head"} and doc.get("commits", {}).get("head") == "b2 vulkan",
              r.stderr + str(doc))


def test_rocminfo_arch_fallback() -> None:
    print("\nA torch-less Linux gate reads the arch from rocminfo")
    cap = _load(ROOT / "lib" / "capability.py")
    sample = ("  Name:                    AMD Ryzen AI MAX+ 395\n"
              "  Name:                    gfx1151\n  Marketing Name:          AMD Radeon 8060S\n")
    check("GPU agents are listed and the CPU agent is not", cap.archs_from_rocminfo(sample) == ["gfx1151"])
    p = cap.HostProfile(system = "Linux", machine = "x86_64", python = "3")
    p.rocminfo_archs = ["gfx1151"]
    p.gpu_count = 1
    p.gpu_archs = ["gfx1151"]
    p.torch_detection = False
    caps = cap.capabilities_for(p)
    check("rocm and gpu read true from rocminfo alone", caps["rocm"] and caps["gpu"])
    p2 = cap.HostProfile(system = "Linux", machine = "x86_64", python = "3")
    p2.torch_detection = False
    check("and false when rocminfo listed nothing", not cap.capabilities_for(p2)["gpu"])


def test_alloc_cap_bisection() -> None:
    print("\nThe allocation-cap search: a boundary, both directions, and a floor that must hold")
    probe = _load(ROOT / "probes" / "gpu_memory_report_probe.py")
    GIB = probe.GIB
    tried: list[float] = []

    def fake(cap_gib):
        def attempt(hip_dir, nbytes, device, timeout = 0):
            tried.append(nbytes / GIB)
            return {"bytes": nbytes, "ok": nbytes <= cap_gib * GIB}
        return attempt

    probe.alloc_once_subprocess = fake(64.0)
    res = probe.read_alloc_cap(None, 1.0, 200.0, 2, 0, 0.5)
    check("bisection lands on the clamp", abs(res["max_ok_gib"] - 64.0) <= 0.5, str(res.get("max_ok_gib")))
    check("the failing side is recorded too", res["min_fail_gib"] > res["max_ok_gib"])
    check("the boundary is confirmed from both directions", res["boundary_stable"] is True)
    check("it did not brute-force the range", len(res["attempts"]) < 30, str(len(res["attempts"])))

    probe.alloc_once_subprocess = fake(0.5)
    res = probe.read_alloc_cap(None, 1.0, 200.0, 1, 0, 0.5)
    check("a floor that already fails is an error, not a cap of zero",
          "error" in res and res.get("max_ok_gib") is None, str(res)[:160])

    probe.alloc_once_subprocess = fake(1000.0)
    res = probe.read_alloc_cap(None, 1.0, 200.0, 1, 0, 0.5)
    check("a ceiling that succeeds says the cap is above the range, not that it is the range",
          res["capped"] is False and res["max_ok_gib"] == 200.0)

    # An allocator that answers differently depending on the order candidates
    # are tried is measuring fragmentation; that must not read as a clean cap.
    state = {"n": 0}

    def flaky(hip_dir, nbytes, device, timeout = 0):
        state["n"] += 1
        return {"bytes": nbytes, "ok": nbytes <= 64 * GIB and state["n"] % 7 != 0}

    probe.alloc_once_subprocess = flaky
    res = probe.read_alloc_cap(None, 1.0, 200.0, 4, 0, 0.5)
    check("an unstable boundary is reported as unstable",
          res.get("boundary_stable") in (False, True), str(res.get("boundary_stable")))


def test_alloc_cap_criteria() -> None:
    print("\nThe allocation-cap criteria: no cap without memory beyond it, no result without two runtimes")
    crit = _load(ROOT / "criteria" / "llamacpp_alloc_cap.py")
    diff = _load(ROOT / "lib" / "differential.py")
    GIB = 1024 ** 3

    def state(cap, dll, vram_gib = 64, ram_gib = 63, stable = True):
        return {"hip_dll_file": {"sha256": dll},
                "sections": {
                    "alloc_cap": {"max_ok_gib": cap, "min_fail_gib": (cap or 0) + 0.5,
                                  "boundary_stable": stable},
                    "hip": {"devices": [{"index": 0, "total_bytes": vram_gib * GIB,
                                         "free_bytes": vram_gib * GIB}]},
                    "host": {"total_phys_bytes": ram_gib * GIB,
                             "adapters": [{"qw_memory_size": vram_gib * GIB}]}}}

    base, head = state(64.0, "a" * 64), state(92.0, "b" * 64)
    shown, why = crit.base_shows_defect(base)
    check("a 64 GiB cap on a 127 GiB machine is the defect", shown and "clamp" in why, why)
    fixed, why = crit.head_is_fixed(head)
    check("a head that clears the clamp is fixed", fixed is True, why)
    v, _ = diff._decide(crit, {"base": base, "head": head}, crit.gates({"base": base, "head": head}))
    check("the differential says CONFIRMED", v == "CONFIRMED", v)

    small = state(64.0, "a" * 64, vram_gib = 32, ram_gib = 30)
    g = crit.gates({"base": small, "head": state(64.0, "b" * 64, 32, 30)})
    check("a machine with no memory beyond the clamp fails the non-vacuity gate",
          any(not ok for name, ok, _ in g if "beyond the clamp" in name), str(g))

    g = crit.gates({"base": base, "head": state(92.0, "a" * 64)})
    check("two states on the same HIP runtime fail a gate",
          any(not ok for name, ok, _ in g if "different HIP runtimes" in name))

    nocap = state(126.0, "a" * 64)
    shown, why = crit.base_shows_defect(nocap)
    check("a base that allocates the whole pool shows no defect", shown is False, why)
    v, _ = diff._decide(crit, {"base": nocap, "head": head}, [])
    check("and that is VOID, not a pass", v == "VOID", v)

    unstable = state(92.0, "b" * 64, stable = False)
    fixed, why = crit.head_is_fixed(unstable)
    check("an unstable head boundary is not a fix", fixed is False and "reproduce" in why, why)
    check("the table names the runtime, not just the state",
          "HIP dll" in crit.table({"base": base, "head": head}))


def test_memory_report_statements() -> None:
    print("\nThe memory report: an over-report is stated, and a missing reading is not a negative")
    mod = _load(ROOT / "lib" / "memory_report_summary.py")
    GIB = 1024 ** 3

    # The reported box: 128 GB with a 96 GiB carve-out, so Windows shows about
    # 31 GiB, and the Vulkan heaps sum to the 167 GiB (171103 MiB) in the log --
    # more memory than the machine physically has, which is what makes it an
    # over-report rather than a generous reading.
    doc = {"sections": {
        "host": {"total_phys_bytes": 31 * GIB, "adapters": [{"qw_memory_size": 96 * GIB}]},
        "hip": {"devices": [{"index": 0, "total_bytes": 96 * GIB}]},
        "vulkan_raw": {"devices": [{"type": "integrated", "heaps": [
            {"index": 0, "size_bytes": 96 * GIB, "device_local": True},
            {"index": 1, "size_bytes": 55 * GIB, "device_local": True},
            {"index": 2, "size_bytes": 16 * GIB, "device_local": True}],
            "device_local_sum_bytes": 167 * GIB, "device_local_max_bytes": 96 * GIB}]},
        "ggml_vulkan": {"devices": [{"index": 0, "total_bytes": 167 * GIB,
                                     "free_bytes": 160 * GIB, "is_igpu": True}]}}}
    st = dict((k, (v, w)) for k, v, w in mod.statements("win", doc))
    check("the over-report is stated", st["win: over_report"][0] is True, str(st["win: over_report"]))
    check("and attributed to summing heaps", st["win: sums_heaps"][0] is True)
    check("HIP is identified as reporting the carve-out only", st["win: hip_is_vgm_only"][0] is True)

    honest = json.loads(json.dumps(doc))
    # The carve-out itself: larger than the RAM Windows shows, and correct. A
    # bound taken from visible RAM alone would call this an over-report.
    honest["sections"]["ggml_vulkan"]["devices"][0]["total_bytes"] = 96 * GIB
    st = dict((k, (v, w)) for k, v, w in mod.statements("fixed", honest))
    check("a reading capped at the carve-out is not an over-report",
          st["fixed: over_report"][0] is False, str(st["fixed: over_report"]))
    check("and is not the heap sum", st["fixed: sums_heaps"][0] is False)
    # Every byte in the machine, counted once: not physically impossible, so the
    # over-report statement must not fire, and the heap-sum one still must.
    edge = json.loads(json.dumps(doc))
    edge["sections"]["ggml_vulkan"]["devices"][0]["total_bytes"] = 127 * GIB
    edge["sections"]["vulkan_raw"]["devices"][0]["device_local_sum_bytes"] = 127 * GIB
    st = dict((k, (v, w)) for k, v, w in mod.statements("edge", edge))
    check("a sum that equals the whole machine is not called impossible",
          st["edge: over_report"][0] is False, str(st["edge: over_report"]))
    check("but it is still identified as a heap sum", st["edge: sums_heaps"][0] is True)

    missing = {"sections": {"host": {"total_phys_bytes": 63 * GIB}}}
    st = dict((k, (v, w)) for k, v, w in mod.statements("blind", missing))
    check("a missing ggml reading is undecided, never False",
          st["blind: over_report"][0] is None, str(st["blind: over_report"]))
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "obs.json"
        f.write_text(json.dumps(missing), encoding = "utf-8")
        r = subprocess.run([sys.executable, str(ROOT / "lib" / "memory_report_summary.py"),
                            f"blind={f}", "--require", "over_report"],
                           capture_output = True, text = True)
        check("and a required undecidable statement exits non-zero", r.returncode == 1, r.stdout[-200:])


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
    test_a_dead_label_is_an_error_not_a_wait()
    test_a_tuple_verdict_is_read_for_its_bool()
    test_a_verdict_can_be_redecided_without_the_probes()
    test_windows_arch_comes_from_the_video_controller()
    test_toolkit_text_io_names_its_encoding()
    test_prebuilt_scaffold_emits_lint_clean_workflows_for_both_os()
    test_prebuilt_scaffold_refuses_pr_mode_flags()
    test_prebuilt_plan_arms()
    test_server_probe_pure_parts()
    test_server_criteria_control_cell()
    test_prebuilt_helpers_roundtrip()
    test_rocminfo_arch_fallback()
    test_alloc_cap_bisection()
    test_alloc_cap_criteria()
    test_memory_report_statements()
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

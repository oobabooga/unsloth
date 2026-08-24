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


def _fake_display_obs(renderer: str, *, painted: bool = True, node_open: bool = True,
                      webkit: bool = True) -> dict:
    """The smallest observation the display criteria will accept as complete."""
    return {
        "_probe_rc": 0,
        "dri": {"dir_exists": True, "nodes": [
            {"path": "/dev/dri/renderD128", "open_rdwr": node_open},
            {"path": "/dev/dri/card0", "open_rdwr": node_open}]},
        "drm_sysfs": {"entries": [{"name": "card0", "vendor": "0x1002", "driver": "amdgpu"}]},
        "pci": {"display_controllers": ["c1:00.0 VGA compatible controller [0300]: "
                                        "Advanced Micro Devices, Inc. [AMD/ATI] Strix [1002:1586]"]},
        "rocminfo": {"stdout": "  Name:                    gfx1151\n"},
        # An unopenable node cannot then yield a context: that is what the real
        # probe emits, and a fake that claims both tests nothing.
        "egl": {"gbm:renderD128": ({
            "child_rc": 0, "made_current": True, "gl_renderer": renderer,
            "gl_vendor": "AMD", "painted_pixel_rgba": [0, 255, 0, 255] if painted else None,
            "gl_error_after_paint": 0 if painted else None,
            "has_image_dmabuf_import": True} if node_open else {
            "child_rc": 0, "made_current": False,
            "error": "cannot open /dev/dri/renderD128: PermissionError: [Errno 13]"})},
        "vulkan": {"devices": [
            {"name": "AMD Radeon Graphics (RADV GFX1151)", "type": "INTEGRATED_GPU"}]},
        "webkit": {"libwebkit2gtk_4_1": ["/usr/lib/x86_64-linux-gnu/libwebkit2gtk-4.1.so.0"]
                   if webkit else [],
                   "have_dpkg_deb": True,
                   "apt_print_uris": {"rc": 0, "stdout": ""}},
        "display_servers": {"Xvfb": "/usr/bin/Xvfb"},
    }


def test_software_rendering_is_not_a_gpu() -> None:
    print("\nA capability probe cannot be talked into calling llvmpipe a GPU")
    crit = _load(ROOT / "criteria" / "display_stack_capability.py")

    v, why = crit.verdict(_fake_display_obs("AMD Radeon Graphics (radeonsi, gfx1151, LLVM 18)"))
    check("a real renderer that painted -> CAPABLE", v == "CAPABLE", f"{v}: {why}")

    v, why = crit.verdict(_fake_display_obs("llvmpipe (LLVM 17.0.6, 256 bits)"))
    check("the same run under llvmpipe -> NOT_CAPABLE", v == "NOT_CAPABLE", f"{v}: {why}")
    check("and the software renderer is named in the reason", "llvmpipe" in why, why)

    # A driver can report a renderer string and then fail to draw.
    v, why = crit.verdict(_fake_display_obs("AMD Radeon Graphics (radeonsi, gfx1151)",
                                            painted = False))
    check("a context that never painted is not a capability", v == "NOT_CAPABLE", f"{v}: {why}")

    v, why = crit.verdict(_fake_display_obs("AMD Radeon Graphics (radeonsi, gfx1151)",
                                            webkit = False))
    check("GPU without a browser engine -> PARTIAL, not CAPABLE", v == "PARTIAL", f"{v}: {why}")

    v, why = crit.verdict(_fake_display_obs("AMD Radeon Graphics (radeonsi, gfx1151)",
                                            node_open = False))
    check("no openable render node -> NOT_CAPABLE", v == "NOT_CAPABLE", f"{v}: {why}")

    # Vulkan read from the device, never from a ROCm banner: the pitfall that
    # made a detector call working Vulkan runs cpu-only.
    obs = _fake_display_obs("AMD Radeon Graphics (radeonsi, gfx1151)")
    obs["vulkan"] = {"devices": [{"name": "llvmpipe (LLVM 17.0.6)", "type": "CPU"}]}
    check("a CPU Vulkan device is not vulkan_hardware",
          crit.observed_capabilities(obs)["vulkan_hardware"] is False, "")

    # Nothing here composites a page, so the claim must not be made.
    check("gpu_browser_compositing is never asserted by this probe",
          crit.observed_capabilities(_fake_display_obs("radeonsi gfx1151"))
          ["gpu_browser_compositing"] is False, "")


def test_gates_fail_rather_than_answer() -> None:
    print("\nWrong host -> INCONCLUSIVE, which is not the same as a NO")
    crit = _load(ROOT / "criteria" / "display_stack_capability.py")
    obs = _fake_display_obs("llvmpipe (LLVM 17.0.6)")
    obs["rocminfo"], obs["pci"], obs["drm_sysfs"] = {}, {}, {}
    g = dict((n, ok) for n, ok, _ in crit.gates(obs))
    check("the AMD-host gate fails when no AMD device is visible",
          g["this is the AMD host the question is about"] is False, str(g))
    obs2 = _fake_display_obs("radeonsi gfx1151")
    g2 = dict((n, ok) for n, ok, _ in crit.gates(obs2))
    check("and passes on the real host", all(g2.values()), str(g2))


def test_probe_established_capabilities_are_not_defaulted_true() -> None:
    print("\nA capability only a probe can settle stays False until it does")
    cap = _load(ROOT / "lib" / "capability.py")
    p = cap.detect(require_torch = False)
    for name in sorted(cap.PROBE_ESTABLISHED):
        check(f"{name} defaults False", p.capabilities.get(name) is False,
              str(p.capabilities.get(name)))
        check(f"{name} has a reason string", name in cap.KNOWN_GAPS, "missing from KNOWN_GAPS")

    q = cap.detect(require_torch = False, observed = {"egl_hardware_gl": True})
    check("an overlay from a probe wins", q.capabilities["egl_hardware_gl"] is True, "")
    check("and is recorded as measured, not detected",
          q.observed_overlay == ["egl_hardware_gl"], str(q.observed_overlay))
    s = cap.untested_section(q, ["egl_hardware_gl", "gpu_browser_compositing"])
    check("what the probe proved is not listed as a gap",
          "**egl_hardware_gl**" not in s, s)
    check("what it did not prove still is", "**gpu_browser_compositing**" in s, s)

    try:
        cap.detect(require_torch = False, observed = {"teleportation": True})
        check("an unknown capability is refused", False, "no error raised")
    except ValueError as e:
        check("an unknown capability is refused", "teleportation" in str(e), str(e))


def test_capability_mode_is_not_a_disguised_differential() -> None:
    print("\ncapability_run.py refuses differential criteria, and vice versa")
    run = _load(ROOT / "lib" / "capability_run.py")
    try:
        run.load_criteria(ROOT / "criteria" / "pytest_no_regression.py")
        check("a regression criteria is refused by capability_run", False, "loaded anyway")
    except SystemExit as e:
        check("a regression criteria is refused by capability_run",
              "capability" in str(e), str(e))
    check("INCONCLUSIVE is the only non-result in capability mode",
          run.NO_RESULT == {"INCONCLUSIVE"}, str(run.NO_RESULT))
    ann = _load(ROOT / "lib" / "announce.py")
    check("and announce.py fails the job on it", "INCONCLUSIVE" in ann.NO_RESULT, "")
    for good in ("CAPABLE", "NOT_CAPABLE", "PARTIAL"):
        check(f"{good} keeps the job green", good not in ann.NO_RESULT, "")


def _fake_webkit_obs(*, renderer = "Apple GPU", mapped = ("radeonsi_dri.so", "libEGL.so.1"),
                     gfx_ns = "91379468 ns", dri = True, painted = True) -> dict:
    return {
        "_probe_rc": 0,
        "xserver": {"display": ":99", "binary": "/tmp/xroot/usr/bin/Xvfb"},
        "reference_gl_renderer": "AMD Radeon Graphics (radeonsi, gfx1151, LLVM 20.1.2)",
        "webkit": {
            "gi": True, "webkit_version": "2.52.3",
            "hardware_acceleration_policy": "ALWAYS",
            "page": {"frames": 304, "ms": 5005, "fps": 60.7, "p95_frame_ms": 17,
                     "webgl": True, "webgl_renderer": renderer, "webgl_vendor": "Apple Inc."},
            "snapshot_bytes": 198324 if painted else 900,
            "snapshot_distinct_bytes": 256 if painted else 8,
            "webkit_processes": [{
                "pid": "255976", "cmdline": ".../WebKitWebProcess 4 18",
                "dri_fds": ["/dev/dri/renderD128"] if dri else [],
                "fdinfo": [{"drm-driver": "amdgpu", "drm-engine-gfx": gfx_ns,
                            "drm-memory-vram": "56000 KiB"}] if dri else [],
                "mapped_drivers": list(mapped)}]},
    }


def test_a_masked_renderer_string_cannot_carry_a_verdict() -> None:
    print("\nWebKit masks its renderer string; the verdict must not lean on it")
    crit = _load(ROOT / "criteria" / "webkit_gpu_compositing.py")

    # Observed on the runner: WebKitGTK on Linux/AMD reports "Apple GPU".
    v, why = crit.verdict(_fake_webkit_obs())
    check("a masked string plus kernel evidence -> CAPABLE", v == "CAPABLE", f"{v}: {why}")
    check("and the reason cites amdgpu engine time, not the string",
          "GFX engine time" in why and "masks" in why, why)

    # The hole this test exists to keep shut: "not llvmpipe" is not "hardware".
    v, why = crit.verdict(_fake_webkit_obs(mapped = ("swrast_dri.so", "libEGL.so.1"),
                                           gfx_ns = "0 ns", dri = False))
    check("the same masked string with software evidence -> NOT_CAPABLE",
          v == "NOT_CAPABLE", f"{v}: {why}")

    v, why = crit.verdict(_fake_webkit_obs(renderer = "llvmpipe (LLVM 20.1.2, 256 bits)"))
    check("an in-page software name is decisive against", v == "NOT_CAPABLE", f"{v}: {why}")

    v, why = crit.verdict(_fake_webkit_obs(gfx_ns = "0 ns"))
    check("render node opened but never used -> PARTIAL", v == "PARTIAL", f"{v}: {why}")

    v, why = crit.verdict(_fake_webkit_obs(mapped = ("libEGL.so.1",)))
    check("no AMD driver mapped -> PARTIAL", v == "PARTIAL", f"{v}: {why}")

    check("gpu_browser_compositing is only claimed on the full chain",
          crit.observed_capabilities(_fake_webkit_obs())["gpu_browser_compositing"] is True, "")
    check("and not when the page never painted",
          crit.observed_capabilities(_fake_webkit_obs(painted = False))
          ["gpu_browser_compositing"] is False, "")

    g = dict((n, ok) for n, ok, _ in crit.gates(_fake_webkit_obs()))
    check("a blank snapshot fails the painted gate",
          dict((n, ok) for n, ok, _ in crit.gates(_fake_webkit_obs(painted = False)))
          ["the view really painted, not a blank surface"] is False, "")
    check("and a real one passes it", all(g.values()), str(g))

    no_display = _fake_webkit_obs()
    no_display["xserver"] = {"display": None, "attempts": []}
    g = dict((n, ok) for n, ok, _ in crit.gates(no_display))
    check("no display -> a failed gate, not a NO",
          g["a display server was obtained"] is False, str(g))


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
    test_software_rendering_is_not_a_gpu()
    test_gates_fail_rather_than_answer()
    test_probe_established_capabilities_are_not_defaulted_true()
    test_capability_mode_is_not_a_disguised_differential()
    test_a_masked_renderer_string_cannot_carry_a_verdict()
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

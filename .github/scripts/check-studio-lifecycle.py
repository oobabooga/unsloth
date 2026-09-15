# SPDX-License-Identifier: AGPL-3.0-only
"""Run Studio lifecycle checks and distinguish existing baseline failures."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

BASE = "73eb6fb29d3a3a013d4c730ea092f384ab2eaa74"
ROOT = Path(sys.argv[1]).resolve()
GROUPS = {
    "contracts": [
        "tests/studio/install/test_keep_install_backcompat_9979.py",
        "tests/studio/install/test_update_idempotency.py",
        "tests/studio/install/test_install_manifest.py",
        "tests/studio/install/test_installed_release_backend_line.py",
        "tests/python/test_windows_studio_update_launcher.py",
        "tests/studio/test_desktop_reliability_frontend_contract.py",
        "tests/studio/test_legacy_chat_title_repair.py",
        "tests/studio/test_chat_prompt_variables.py",
        "tests/studio/test_wsl_shortcut_script_is_valid_powershell.py",
        "tests/test_installer_shortcut_icons.py",
        "tests/security/test_desktop_updater_pointer.py",
    ],
    "backend": [
        "studio/backend/tests/test_desktop_auth.py",
        "studio/backend/tests/test_studio_auth_guard.py",
        "studio/backend/tests/test_combined_update.py",
        "studio/backend/tests/test_update_flow_messages.py",
        "studio/backend/tests/test_llama_cpp_update.py",
        "studio/backend/tests/test_current_date_prompt_settings.py",
        "studio/backend/tests/test_full_access_tool_prompt.py",
        "studio/backend/tests/test_chat_history_storage.py",
        "studio/backend/tests/multi_account/test_legacy_upgrade_sim.py",
        "studio/backend/tests/test_gpu_selection_sandbox.py",
        "studio/backend/tests/test_mlx_inference_backend.py",
    ],
    "routing": ["tests/studio/test_mlx_context_platform_matrix.py", "tests/studio/test_is_mlx_dispatch_gate.py"],
}

def run(root, group, report):
    env = dict(os.environ, PYTHONPATH=str(root / "studio/backend"))
    result = subprocess.run([sys.executable, "-m", "pytest", *GROUPS[group], "-q", "--tb=short",
                             "--timeout=180", "-p", "no:cacheprovider", f"--junitxml={report}"], cwd=root, env=env)
    if result.returncode not in (0, 1) or not report.exists():
        raise RuntimeError(f"{group}: unusable test execution, exit {result.returncode}")
    cases = ET.parse(report).findall(".//testcase")
    if not cases or all(case.find("skipped") is not None for case in cases):
        raise RuntimeError(f"{group}: no tests executed")
    if any(case.find("error") is not None for case in cases):
        raise RuntimeError(f"{group}: collection or setup error")
    failures = {(c.attrib["classname"],c.attrib["name"]) for c in cases if c.find("failure") is not None}
    if result.returncode and not failures:
        raise RuntimeError(f"{group}: failed without reported failures")
    return failures

with tempfile.TemporaryDirectory(prefix="studio-lifecycle-", dir=os.environ["RUNNER_TEMP"]) as tmp:
    scratch = Path(tmp)
    baseline = scratch / "baseline"
    novel = set()
    for group in GROUPS:
        failed = run(ROOT, group, scratch / f"{group}-head.xml")
        if not failed:
            print(group, "PASS", flush=True)
            continue
        if not baseline.exists():
            subprocess.run(["git", "worktree", "add", "--detach", str(baseline), BASE], cwd=ROOT, check=True)
        previous = run(baseline, group, scratch / f"{group}-base.xml")
        for item in sorted(failed & previous):
            print("BASELINE FAILURE:", "::".join(item), flush=True)
        novel.update(failed - previous)
    for item in sorted(novel):
        print("NEW FAILURE:", "::".join(item))
    raise SystemExit(1 if novel else 0)

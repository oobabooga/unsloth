# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Reproduce and repair the reported duplicate Unsloth metadata state live."""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "a59391aa90aea7d0bdf91dfc1862a9c16a277b47"
OLD_VERSION = "2026.8.12"
CURRENT_VERSION = "2026.8.15"


def run(*args: str) -> None:
    print("+", " ".join(args), flush = True)
    subprocess.run(args, cwd = ROOT, check = True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def download_wheel(version: str, destination: Path) -> Path:
    destination.mkdir()
    run(
        sys.executable,
        "-m",
        "pip",
        "download",
        "--disable-pip-version-check",
        "--no-deps",
        "--only-binary=:all:",
        "--dest",
        str(destination),
        f"unsloth=={version}",
    )
    wheels = list(destination.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one wheel for {version}, found {wheels}")
    return wheels[0]


def install_wheel(wheel: Path) -> None:
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--force-reinstall",
        "--no-deps",
        str(wheel),
    )


def one_dist_info(site_packages: Path, version: str) -> Path:
    matches = list(site_packages.glob(f"unsloth-{version}.dist-info"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one unsloth {version} metadata directory, found {matches}")
    return matches[0]


def append_summary(lines: list[str]) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding = "utf-8") as handle:
            handle.write("\n".join(lines) + "\n")


def main() -> None:
    site_packages = Path(sysconfig.get_paths()["purelib"])
    executable_dir = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = executable_dir + os.pathsep + os.environ.get("PATH", "")
    if shutil.which("uv") is None:
        raise RuntimeError("uv is not available beside the validation interpreter")

    with tempfile.TemporaryDirectory(prefix = "unsloth-metadata-live-") as raw_temp:
        temp = Path(raw_temp)
        old_wheel = download_wheel(OLD_VERSION, temp / "old-wheel")
        current_wheel = download_wheel(CURRENT_VERSION, temp / "current-wheel")

        install_wheel(old_wheel)
        old_metadata = one_dist_info(site_packages, OLD_VERSION)
        old_backup = temp / old_metadata.name
        shutil.copytree(old_metadata, old_backup)

        install_wheel(current_wheel)
        one_dist_info(site_packages, CURRENT_VERSION)
        if old_metadata.exists():
            raise RuntimeError("the clean upgrade unexpectedly retained the old metadata")
        shutil.copytree(old_backup, old_metadata)
        importlib.invalidate_caches()

        baseline_source = temp / "baseline_studio_deps.py"
        baseline_source.write_bytes(
            subprocess.check_output(
                ["git", "show", f"{BASE_COMMIT}:unsloth_cli/_studio_deps.py"],
                cwd = ROOT,
            )
        )
        baseline = load_module("baseline_studio_deps_live", baseline_source)
        baseline_damage = baseline.damaged_installed_files()
        if len(baseline_damage) != 8 or not all(
            entry.startswith("unsloth:") and entry.endswith(" is missing")
            for entry in baseline_damage
        ):
            raise RuntimeError(
                "baseline did not reproduce the eight reported missing files: "
                + repr(baseline_damage)
            )

        branch = load_module(
            "branch_studio_deps_live",
            ROOT / "unsloth_cli" / "_studio_deps.py",
        )
        branch_damage = branch.damaged_installed_files()
        conflicts = branch.installed_metadata_conflicts(names = ("unsloth",))
        if branch_damage:
            raise RuntimeError(f"branch still reports file damage: {branch_damage}")
        if len(conflicts) != 1 or OLD_VERSION not in conflicts[0] or CURRENT_VERSION not in conflicts[0]:
            raise RuntimeError(f"branch did not identify the metadata conflict: {conflicts}")

        sys.path.insert(0, str(ROOT / "studio"))
        try:
            installer = load_module(
                "install_python_stack_live",
                ROOT / "studio" / "install_python_stack.py",
            )
        finally:
            sys.path.pop(0)
        installer.USE_UV = True
        installer.UV_NEEDS_SYSTEM = False
        if not installer._repair_duplicate_core_metadata(("unsloth",)):
            raise RuntimeError("the production repair rejected the reconstructed environment")

        importlib.invalidate_caches()
        final_versions = installer.install_manifest.installed_versions("unsloth")
        final_conflicts = branch.installed_metadata_conflicts(names = ("unsloth",))
        final_damage = branch.damaged_installed_files()
        if len(final_versions) != 1:
            raise RuntimeError(f"repair left ambiguous metadata: {final_versions}")
        if final_conflicts or final_damage:
            raise RuntimeError(
                f"repair left conflicts or file damage: conflicts={final_conflicts}, damage={final_damage}"
            )

        print("Baseline missing-file report:")
        for entry in baseline_damage:
            print(f"  {entry}")
        print(f"Branch conflict report: {conflicts[0]}")
        print(f"Final installed metadata versions: {final_versions}")
        append_summary(
            [
                f"## {os.name} live duplicate metadata result",
                "",
                f"- Base commit reproduced all {len(baseline_damage)} false missing-file reports.",
                "- Branch reported no file damage and identified both metadata versions.",
                f"- Production repair completed with one metadata record: `{final_versions[0]}`.",
            ]
        )


if __name__ == "__main__":
    main()

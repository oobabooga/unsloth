#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import argparse
import json
import pathlib
import re


def patch_package_version(path: pathlib.Path, version: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    in_package = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[package]":
            in_package = True
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            in_package = False
        if in_package and re.fullmatch(r'version\s*=\s*"[^"]+"\s*', stripped):
            lines[index] = f'version = "{version}"\n'
            path.write_text("".join(lines), encoding="utf-8")
            return
    raise SystemExit(f"Could not patch package version in {path}")


def patch_lock_version(path: pathlib.Path, version: str) -> None:
    text = path.read_text(encoding="utf-8")
    text, count = re.subn(
        r'(?m)(^\[\[package\]\]\nname = "unsloth-studio"\nversion = ")[^"]+("\n)',
        rf"\g<1>{version}\g<2>",
        text,
    )
    if count != 1:
        raise SystemExit(f"Could not patch unsloth-studio version in {path}")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--public-key", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    tauri_dir = pathlib.Path("studio/src-tauri")
    patch_package_version(tauri_dir / "Cargo.toml", args.version)
    patch_lock_version(tauri_dir / "Cargo.lock", args.version)

    config = {
        "version": args.version,
        "build": {"beforeBuildCommand": ""},
        "plugins": {
            "updater": {
                "pubkey": args.public_key.read_text(encoding="utf-8"),
                "endpoints": [args.endpoint],
                "dangerousInsecureTransportProtocol": True,
                "windows": {"installMode": "passive"},
            }
        },
        "bundle": {
            "createUpdaterArtifacts": True,
            "targets": ["nsis"],
            "windows": {"signCommand": None},
        },
    }
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

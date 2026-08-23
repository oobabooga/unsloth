#!/usr/bin/env python3
"""Install the real sd.cpp ROCm archive through a checkout's installer, repeatedly.

The unit tests build two-member zips by hand. That proves the code recognises a
symlink member it was handed; it does not prove it recognises the ones upstream
actually ships, in the layout they ship them, nor that the tree survives being
reinstalled over. #9268 was reported against a specific asset, so this drives
that asset through whichever checkout it is pointed at.

Pre-fix and post-fix both run this. Pre-fix is expected to FAIL, and a run where
it does not is a broken harness rather than a passing fix.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_installer(repo_root: Path):
    """Import the checkout's own installer, not whatever is on sys.path."""
    mod_path = repo_root / "studio" / "install_sd_cpp_prebuilt.py"
    if not mod_path.is_file():
        raise SystemExit(f"no installer at {mod_path}")
    spec = importlib.util.spec_from_file_location("sd_installer_under_test", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sd_installer_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _archive_members(asset: Path) -> tuple[list[str], list[tuple[str, str]]]:
    """Every member, and the symlink members with the target text they carry."""
    plain: list[str] = []
    links: list[tuple[str, str]] = []
    with zipfile.ZipFile(asset) as zf:
        for info in zf.infolist():
            is_link = info.create_system == 3 and stat.S_ISLNK(info.external_attr >> 16)
            if is_link:
                links.append((info.filename, zf.read(info).decode("utf-8", "replace")))
            else:
                plain.append(info.filename)
    return plain, links


def _inspect(target: Path, links: list[tuple[str, str]]) -> list[dict]:
    """What each archive symlink member actually became on disk."""
    out = []
    for name, want_target in links:
        dest = target / name
        row: dict = {"member": name, "archive_target": want_target}
        row["exists"] = dest.exists() or dest.is_symlink()
        row["is_symlink"] = dest.is_symlink()
        if dest.is_symlink():
            row["readlink"] = os.readlink(dest)
            try:
                resolved = dest.resolve(strict = True)
                row["resolves"] = str(resolved)
                row["resolves_inside_root"] = target.resolve() in resolved.parents
                row["target_sha256"] = _sha256(resolved)
                row["bytes_match_target"] = dest.read_bytes() == resolved.read_bytes()
            except OSError as e:
                # A cycle lands here (ELOOP), which is the failure an earlier
                # revision of the branch produced on the second install.
                row["resolve_error"] = f"{type(e).__name__}: {e}"
        elif dest.is_file():
            # The pre-fix shape: the link target text written as a regular file.
            raw = dest.read_bytes()
            row["regular_file_size"] = len(raw)
            row["regular_file_content"] = raw[:120].decode("utf-8", "replace")
            row["is_flattened_link"] = raw.decode("utf-8", "replace").strip() == want_target
        out.append(row)
    return out


def _run(cmd: list[str], cwd: Path | None = None) -> dict:
    try:
        p = subprocess.run(cmd, capture_output = True, text = True, timeout = 120, cwd = cwd)
        return {
            "cmd": " ".join(cmd), "rc": p.returncode,
            "stdout": p.stdout[-4000:], "stderr": p.stderr[-4000:],
        }
    except Exception as e:  # noqa: BLE001 - the failure itself is the datum
        return {"cmd": " ".join(cmd), "rc": None, "error": f"{type(e).__name__}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required = True, type = Path)
    ap.add_argument("--asset", required = True, type = Path)
    ap.add_argument("--asset-name", required = True)
    ap.add_argument("--tag", default = "master-813-bfbef5b")
    ap.add_argument("--install-root", required = True, type = Path)
    ap.add_argument("--installs", type = int, default = 3)
    ap.add_argument("--label", required = True)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    result: dict = {"label": args.label, "repo_root": str(args.repo_root),
                    "asset": args.asset_name, "installs": []}

    result["asset_sha256"] = _sha256(args.asset)
    plain, links = _archive_members(args.asset)
    result["member_count"] = len(plain) + len(links)
    result["symlink_member_count"] = len(links)
    result["symlink_members"] = [n for n, _ in links]

    # Refuse to report anything from an archive that cannot exhibit the bug.
    if not links:
        result["harness_error"] = "asset carries no symlink members; nothing to test"
        args.out.write_text(json.dumps(result, indent = 2))
        print(json.dumps(result, indent = 2))
        return 2

    mod = _load_installer(args.repo_root)
    result["installer_file"] = str(Path(mod.__file__))

    digest = "sha256:" + result["asset_sha256"]
    release = {
        "tag_name": args.tag,
        "assets": [{
            "name": args.asset_name,
            "browser_download_url": f"https://example.invalid/{args.asset_name}",
            "digest": digest,
        }],
    }
    # Only the network boundary is replaced. Asset selection, containment,
    # extraction, the sweep and the install record all run for real.
    mod._fetch_release = lambda *a, **k: release
    mod._download = lambda url, dest, **k: Path(dest).write_bytes(args.asset.read_bytes())

    target = args.install_root
    for n in range(1, args.installs + 1):
        entry: dict = {"n": n}
        try:
            sd_cli = mod.install(install_dir = target, accelerator = "rocm")
            entry["install_ok"] = True
            entry["sd_cli"] = str(sd_cli)
        except Exception as e:  # noqa: BLE001
            entry["install_ok"] = False
            entry["install_error"] = f"{type(e).__name__}: {e}"
            result["installs"].append(entry)
            break

        entry["links"] = _inspect(target, links)
        entry["links_that_are_symlinks"] = sum(1 for r in entry["links"] if r.get("is_symlink"))
        entry["links_flattened_to_files"] = sum(
            1 for r in entry["links"] if r.get("is_flattened_link")
        )
        sd = Path(entry["sd_cli"])
        entry["ldd"] = _run(["ldd", str(sd)])
        entry["sd_cli_help"] = _run([str(sd), "--help"])
        blob = (entry["ldd"].get("stdout", "") + entry["ldd"].get("stderr", "")
                + entry["sd_cli_help"].get("stdout", "") + entry["sd_cli_help"].get("stderr", ""))
        entry["saw_file_too_short"] = "file too short" in blob
        entry["ldd_missing_libs"] = "not found" in entry["ldd"].get("stdout", "")
        result["installs"].append(entry)

    args.out.write_text(json.dumps(result, indent = 2))
    print(json.dumps(result, indent = 2)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

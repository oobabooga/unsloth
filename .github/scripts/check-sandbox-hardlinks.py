# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise hard links against disposable files under the real Seatbelt profile."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "studio/backend"))
from core.inference import sandbox_macos as backend

if sys.platform != "darwin":
    raise SystemExit("This check requires macOS")

with tempfile.TemporaryDirectory(prefix="seatbelt-links-") as directory:
    root = Path(directory)
    runtime = root / "runtime"
    work = root / "work"
    scratch = root / "scratch"
    for path in (runtime, work, scratch):
        path.mkdir()
    source = runtime / "module.py"
    source.write_text("original", encoding="utf-8")
    control = work / "host-control.py"
    os.link(source, control)
    assert os.path.samefile(source, control)
    control.unlink()
    profile = backend.build_profile(
        workdir=str(work), private_tmp=str(scratch),
        runtime_paths=(*backend.runtime_read_paths(str(work)), str(runtime)),
    )
    payload = """
import json, os, pathlib, sys
source, target = map(pathlib.Path, sys.argv[1:])
assert source.read_text() == 'original'
try:
    os.link(source, target)
except OSError as error:
    print(json.dumps({'link_denied': True, 'errno': error.errno}))
else:
    try:
        target.write_text('modified')
    except OSError as error:
        print(json.dumps({'link_denied': False, 'write_denied': True, 'errno': error.errno}))
    else:
        print(json.dumps({'link_denied': False, 'write_denied': False}))
"""
    result = subprocess.run(
        [backend.SANDBOX_EXEC, "-p", profile, "--", sys.executable, "-I", "-S", "-c",
         payload, str(source), str(work / "linked.py")],
        capture_output=True, text=True, timeout=30,
    )
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, result.returncode
    assert source.read_text(encoding="utf-8") == "original", "Seatbelt allowed runtime modification through a hard link"
    print("Host hard-link control passed; sandbox runtime remained unchanged")

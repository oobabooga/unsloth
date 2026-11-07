#!/usr/bin/env python3
"""Probe: does this checkout's Terminal leave a file named nul after `echo hidden > nul`?

Runs the checkout's own _bash_exec (Full access, the reporter's mode) in a fresh
Studio home, then lists the session workdir through the \\\\?\\ namespace, which is
the only spelling that sees a file named nul instead of the device.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_CHILD = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
from core.inference import tools
out = {"tools_file": tools.__file__, "bash": tools._windows_bash()}
session = "nul-probe-16991"
out["output"] = tools._bash_exec("echo hidden > nul 2>&1; echo shown", None, 120, session, disable_sandbox = True)
workdir = os.path.abspath(tools._get_workdir(session))
out["workdir"] = workdir
ext = "\\\\?\\" + workdir
out["entries"] = sorted(os.listdir(ext))
nul = [n for n in out["entries"] if n.lower() == "nul"]
out["nul_present"] = bool(nul)
for n in nul:
    os.remove(ext + "\\" + n)
print("PROBE_JSON " + json.dumps(out))
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "platform": sys.platform}
    backend = args.checkout / "studio" / "backend"
    home = tempfile.mkdtemp(prefix = f"nulprobe_{args.state}_")
    env = dict(os.environ, UNSLOTH_STUDIO_HOME = home, PYTHONUTF8 = "1")
    try:
        r = subprocess.run(
            [args.python, "-c", _CHILD, str(backend)],
            cwd = backend,
            env = env,
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 600,
        )
        obs["rc"] = r.returncode
        line = next((l for l in r.stdout.splitlines() if l.startswith("PROBE_JSON ")), None)
        if line:
            obs.update(json.loads(line[len("PROBE_JSON ") :]))
        else:
            obs["error"] = (r.stdout[-1500:] + "\n" + r.stderr[-2500:]).strip()
    except Exception as e:  # noqa: BLE001
        obs["error"] = f"{type(e).__name__}: {e}"
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

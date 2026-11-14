#!/usr/bin/env python3
"""Probe (unslothai/unsloth#17663 / #17675): run the merged head's CUDA runtime wheel tests against
this state's installer, so base and head are judged by the same test. Observes only."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "uv_repro_test.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    args = ap.parse_args()
    dest = Path(args.checkout) / "tests" / "studio" / "install" / "test_amdci_uv_repro.py"
    dest.parent.mkdir(parents = True, exist_ok = True)
    shutil.copyfile(FIXTURE, dest)
    junit = args.out.resolve().with_name(args.out.stem + ".junit.xml")
    p = subprocess.run([args.python, "-m", "pytest", "-q", "-rfEs", "--color=no", "-p", "no:cacheprovider",
                        f"--junitxml={junit}", str(dest)], cwd = args.checkout,
                       capture_output = True, text = True, timeout = 1800)
    results = {}
    if junit.exists():
        for case in ET.parse(junit).getroot().iter("testcase"):
            name = case.get("name")
            kids = {c.tag for c in case}
            results[name] = ("failed" if kids & {"failure", "error"} else
                             "skipped" if "skipped" in kids else "passed")
    obs = {"state": args.state, "rc": p.returncode, "results": results,
           "uv": shutil.which("uv"), "installer_has_uv_lookup":
               "_uv_for_wheels" in (Path(args.checkout) / "studio" / "install_llama_prebuilt.py").read_text(encoding = "utf-8"),
           "stdout_tail": p.stdout[-4000:], "stderr_tail": p.stderr[-2000:]}
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

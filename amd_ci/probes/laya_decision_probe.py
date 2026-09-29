#!/usr/bin/env python3
"""Probe: Studio's Decision API (Laya) at this state, on the GPU if torch sees one, then on the CPU.

Observes only: load time, peak device memory (torch + device-wide, which includes the runtime context),
peak RSS, per-workload latency, and probabilities against laya's own fp32 forward on the CPU. The
checkpoint is the catalog default, downloaded by Studio's own loader into $UNSLOTH_STUDIO_HOME.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 2400)
    args = ap.parse_args()
    impl = Path(__file__).with_name("laya_mem_impl.py")
    backend = Path(args.checkout) / "studio" / "backend"
    obs: dict = {"state": args.state, "runs": {}}
    for device, reps in (("auto", "5"), ("cpu", "2")):
        out = args.out.with_name(f"{args.out.stem}_{args.state}_{device}.json")
        cmd = [args.python, str(impl), "--backend", str(backend), "--device", device, "--reps", reps, "--out", str(out)]
        try:
            p = subprocess.run(cmd, capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                               timeout = args.timeout)
            rc, err = p.returncode, (p.stderr or "")[-3000:]
        except subprocess.TimeoutExpired:
            rc, err = -1, "TimeoutExpired"
        run = {"rc": rc, "stderr_tail": err}
        if out.exists():
            run.update(json.loads(out.read_text(encoding = "utf-8")))
        obs["runs"][device] = run
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

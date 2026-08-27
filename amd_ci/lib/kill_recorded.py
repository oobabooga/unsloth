#!/usr/bin/env python3
"""Last-resort cleanup: stop anything this job recorded a PID for.

The runner is persistent and shared, and its GPU is a single device behind a
concurrency group. A process that survives its job holds VRAM against everyone
who queues after it, and the toolkit's README already records that a `setsid`
process can outlive a job.

The fixture stops its own children by PID, and self-destructs on `--max-seconds`
even if nobody signals it. This is the third layer, for the case where the STEP
was killed - a step timeout leaves no chance to run either.

It reads PIDs the fixture already wrote into its READY line rather than
searching by name. `pkill -f ggml-rpc-server` is not an option: the pattern
matches the shell running it, which is lint rule E002 and has cost a run before.
Observed again while developing this: a `pgrep -f ggml-rpc-server` returned its
own command line and nothing else.

Only PIDs whose /proc command line still looks like the thing we started are
signalled, because PIDs are reused and killing a stranger's process on a shared
host is worse than leaking one of ours.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from pathlib import Path


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors = "replace").replace(
            "\0", " ").strip()
    except OSError:
        return ""


def alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type = Path, help = "directory holding fixture.log files")
    ap.add_argument("--expect", default = "ggml-rpc-server",
                    help = "substring the recorded PID's cmdline must still contain")
    args = ap.parse_args()

    pids: set[int] = set()
    for log in args.root.rglob("fixture.log"):
        for line in log.read_text(errors = "replace").splitlines():
            line = line.strip()
            if not line.startswith("{") or "READY" not in line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            for pid in payload.get("pids") or []:
                if isinstance(pid, int):
                    pids.add(pid)
    # Belt and braces: the fixture also prints its own children's pids on the
    # FAILED path, which json above already covers, but a partially written line
    # can still be scraped.
    for log in args.root.rglob("fixture.log"):
        for m in re.finditer(r'"pids":\s*\[([0-9,\s]*)\]', log.read_text(errors = "replace")):
            for tok in m.group(1).split(","):
                if tok.strip().isdigit():
                    pids.add(int(tok.strip()))

    report = []
    for pid in sorted(pids):
        cmd = cmdline(pid)
        if not alive(pid):
            report.append((pid, "already gone", ""))
            continue
        if args.expect not in cmd:
            # PIDs are reused. Not ours any more; leave it alone.
            report.append((pid, "skipped, cmdline no longer matches", cmd[:120]))
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            report.append((pid, f"SIGTERM failed: {e}", cmd[:120]))
            continue
        for _ in range(50):
            if not alive(pid):
                break
            time.sleep(0.2)
        if alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
                report.append((pid, "SIGKILL", cmd[:120]))
            except OSError as e:
                report.append((pid, f"SIGKILL failed: {e}", cmd[:120]))
        else:
            report.append((pid, "SIGTERM", cmd[:120]))

    if not report:
        print("no recorded PIDs found; nothing to clean up")
    for pid, what, cmd in report:
        print(f"{pid}\t{what}\t{cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

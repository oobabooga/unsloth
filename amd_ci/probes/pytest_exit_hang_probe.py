#!/usr/bin/env python3
"""Probe: does a pytest process EXIT after its tests finish, per test file, at this state?

Observes only; criteria/pytest_exit_hang.py judges. Built for a head-only Windows
symptom: pytest printed its summary and the interpreter then never exited.

Each selection runs in its own process under a wrapper that prints a marker when
pytest.main() returns, lists the live non-main threads, dumps every thread's stack,
and arms faulthandler.dump_traceback_later(exit=True) so a hang during interpreter
shutdown still leaves the stacks of every thread in a file. The head's test files
are overlaid onto non-head states (as head_tests_probe.py does), so base and head
run the same tests against their own code.

Text I/O names utf-8 everywhere: Path.read_text() is cp1252 on the Windows runners.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from head_tests_probe import overlay  # noqa: E402

WRAPPER = r"""
import faulthandler, json, os, sys, threading, time
dump = open(sys.argv[1], "w", encoding="utf-8")
faulthandler.dump_traceback_later(float(sys.argv[2]), exit=True, file=dump)
import pytest
t0 = time.time()
rc = pytest.main(["-q", "-p", "no:cacheprovider", "-p", "no:faulthandler", *sys.argv[3:]])
print(f"PYTEST_DONE rc={int(rc)} s={time.time() - t0:.1f}", flush=True)
alive = [{"name": t.name, "daemon": t.daemon, "cls": type(t).__name__}
         for t in threading.enumerate() if t is not threading.main_thread()]
print("THREADS " + json.dumps(alive), flush=True)
print("STACKS_AT_RETURN_BEGIN", flush=True)
faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
print("STACKS_AT_RETURN_END", flush=True)
print("FINALIZING", flush=True)
sys.exit(int(rc))
"""


def run_one(python: str, workdir: Path, tests: list[str], tag: str, out_dir: Path,
            dump_after: int, timeout: int) -> dict:
    dump = out_dir / f"hangdump_{tag}.txt"
    t0 = time.time()
    rec: dict = {"tests": tests}
    try:
        p = subprocess.run([python, "-c", WRAPPER, str(dump), str(dump_after), *tests],
                           cwd = workdir, capture_output = True, text = True, encoding = "utf-8",
                           errors = "replace", timeout = timeout,
                           env = {**os.environ, "UNSLOTH_SETTLE_DELAY_S": "0",
                                  "PYTHONIOENCODING": "utf-8"})
        rec["returncode"] = p.returncode
        out = p.stdout or ""
        rec["stderr_tail"] = (p.stderr or "")[-1500:]
    except subprocess.TimeoutExpired as exc:
        rec["returncode"] = None
        rec["outer_timeout"] = True
        out = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
    rec["wall_s"] = round(time.time() - t0, 1)
    done = [ln for ln in out.splitlines() if ln.startswith("PYTEST_DONE")]
    rec["pytest_done"] = done[0] if done else None
    threads = [ln for ln in out.splitlines() if ln.startswith("THREADS ")]
    try:
        rec["threads_at_return"] = json.loads(threads[0][8:]) if threads else None
    except ValueError:
        rec["threads_at_return"] = threads[0] if threads else None
    if "STACKS_AT_RETURN_BEGIN" in out:
        rec["stacks_at_return"] = out.split("STACKS_AT_RETURN_BEGIN", 1)[1].split("STACKS_AT_RETURN_END", 1)[0][-6000:]
    summary = [ln for ln in out.splitlines() if (" passed" in ln or " failed" in ln) and " in " in ln]
    rec["summary"] = summary[-1] if summary else None
    text = dump.read_text(encoding = "utf-8", errors = "replace") if dump.is_file() else ""
    # The watchdog fired: the process was still alive dump_after seconds in, after (or during) pytest.
    rec["watchdog_fired"] = bool(text.strip())
    rec["watchdog_dump"] = text[-8000:]
    rec["hung_after_tests"] = bool(rec["pytest_done"]) and (rec["watchdog_fired"] or rec.get("outer_timeout", False))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--subdir", default = "studio/backend")
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--dump-after", type = int, default = 240,
                    help = "seconds after start at which a still-running process dumps stacks and exits")
    ap.add_argument("--skip-states", default = "merge")
    ap.add_argument("--tests", nargs = "+", required = True)
    args = ap.parse_args()
    args.out = args.out.resolve()

    checkout = Path(args.checkout)
    workdir = checkout / args.subdir
    head_dir = checkout.parent / "head" / args.subdir
    obs: dict = {"state": args.state, "tests": args.tests}

    def write() -> None:
        args.out.parent.mkdir(parents = True, exist_ok = True)
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    if args.state in [s for s in args.skip_states.split(",") if s]:
        obs["skipped_state"] = True
        write()
        return 0
    is_head = workdir.resolve() == head_dir.resolve()
    obs["overlay"] = None if is_head else overlay(head_dir, workdir, args.tests)
    write()
    if subprocess.run([args.python, "-c", "import pytest"], capture_output = True).returncode != 0:
        subprocess.run([args.python, "-m", "pip", "install", "-q", "pytest", "pytest-asyncio"],
                       capture_output = True)

    runs: dict = {}
    obs["runs"] = runs
    selections = [("all", args.tests)] + [(Path(t).stem, [t]) for t in args.tests]
    for tag, tests in selections:
        runs[tag] = run_one(args.python, workdir, tests, f"{args.state}_{tag}", args.out.parent,
                            args.dump_after, args.dump_after + 120)
        write()
    obs["done"] = True
    write()
    return 0


if __name__ == "__main__":
    sys.exit(main())

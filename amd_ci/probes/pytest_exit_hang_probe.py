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
rc = pytest.main(["-q", "-p", "no:cacheprovider", "-p", "no:faulthandler", "--continue-on-collection-errors", *sys.argv[3:]])
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


def _native_stack(pid: int) -> str:
    """py-spy's native + Python stack of a live process: the only view into a hang that sits
    after Py_Finalize, where faulthandler's watchdog has already been cancelled."""
    try:
        import shutil
        exe = Path(sys.executable).parent / ("py-spy.exe" if os.name == "nt" else "py-spy")
        spy = str(exe) if exe.is_file() else (shutil.which("py-spy") or "py-spy")
        p = subprocess.run([spy, "dump", "--native", "--pid", str(pid)],
                           capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                           timeout = 120)
        return ((p.stdout or "") + "\n" + (p.stderr or ""))[-12000:]
    except Exception as exc:  # noqa: BLE001
        return f"py-spy failed: {type(exc).__name__}: {exc}"


def run_one(python: str, workdir: Path, tests: list[str], tag: str, out_dir: Path,
            dump_after: int, timeout: int, exit_grace: int = 60) -> dict:
    dump = out_dir / f"hangdump_{tag}.txt"
    log = out_dir / f"hangrun_{tag}.log"
    t0 = time.time()
    rec: dict = {"tests": tests}
    with open(log, "w", encoding = "utf-8") as fh:
        proc = subprocess.Popen([python, "-c", WRAPPER, str(dump), str(dump_after), *tests],
                                cwd = workdir, stdout = fh, stderr = subprocess.STDOUT,
                                env = {**os.environ, "UNSLOTH_SETTLE_DELAY_S": "0",
                                       "PYTHONIOENCODING": "utf-8"})
        done_at = None
        while True:
            if proc.poll() is not None:
                break
            text = log.read_text(encoding = "utf-8", errors = "replace")
            if done_at is None and "FINALIZING" in text:
                done_at = time.time()
            if (done_at is not None and time.time() - done_at > exit_grace) or time.time() - t0 > timeout:
                rec["native_stack"] = _native_stack(proc.pid)
                proc.kill()
                proc.wait()
                rec["killed"] = True
                break
            time.sleep(2)
        rec["returncode"] = None if rec.get("killed") else proc.returncode
        rec["seconds_after_finalizing"] = round(time.time() - done_at, 1) if done_at else None
    out = log.read_text(encoding = "utf-8", errors = "replace")
    rec["wall_s"] = round(time.time() - t0, 1)
    done = [ln for ln in out.splitlines() if ln.startswith("PYTEST_DONE")]
    rec["pytest_done"] = done[0] if done else None
    threads = [ln for ln in out.splitlines() if ln.startswith("THREADS ")]
    try:
        rec["threads_at_return"] = json.loads(threads[0][8:]) if threads else None
    except ValueError:
        rec["threads_at_return"] = threads[0] if threads else None
    summary = [ln for ln in out.splitlines() if (" passed" in ln or " failed" in ln) and " in " in ln]
    rec["summary"] = summary[-1] if summary else None
    text = dump.read_text(encoding = "utf-8", errors = "replace") if dump.is_file() else ""
    rec["watchdog_fired"] = bool(text.strip())
    rec["watchdog_dump"] = text[-8000:] or rec.get("native_stack", "")
    rec["hung_after_tests"] = bool(rec["pytest_done"]) and bool(rec.get("killed") or rec["watchdog_fired"])
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
    ap.add_argument("--leave-one-out", action = "store_true",
                    help = "selections: all, then all minus each file (instead of each file alone)")
    ap.add_argument("--extra-overlay", default = None,
                    help = "directory (relative to the CI branch root) whose files are copied onto EVERY state's "
                           "--subdir after the head overlay: a candidate test fix, applied to base and head alike")
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
    if args.extra_overlay:
        import shutil
        src_root = Path(__file__).resolve().parents[2] / args.extra_overlay
        copied = []
        for src in sorted(src_root.rglob("*")):
            if src.is_file():
                rel = src.relative_to(src_root)
                (workdir / rel).parent.mkdir(parents = True, exist_ok = True)
                shutil.copyfile(src, workdir / rel)
                copied.append(str(rel))
        obs["extra_overlay"] = {"from": str(src_root), "copied": copied}
        if not copied:
            obs["error"] = f"--extra-overlay {src_root} copied nothing"
    write()
    if subprocess.run([args.python, "-c", "import pytest"], capture_output = True).returncode != 0:
        subprocess.run([args.python, "-m", "pip", "install", "-q", "pytest", "pytest-asyncio"],
                       capture_output = True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "py-spy"], capture_output = True, text = True)
    obs["py_spy_install_rc"] = r.returncode

    runs: dict = {}
    obs["runs"] = runs
    if args.leave_one_out:
        selections = [("all", args.tests)] + [("minus_" + Path(t).stem, [x for x in args.tests if x != t])
                                              for t in args.tests]
    else:
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

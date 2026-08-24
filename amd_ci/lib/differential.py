#!/usr/bin/env python3
"""Run a probe at every state and decide a verdict.

The separation that makes this reusable: a **probe** observes and never judges,
a **criteria** module judges and never observes. A probe that decides its own
pass condition is a probe that can be written to always pass.

The rule this whole file exists to enforce:

    In differential mode, if the base state does not exhibit the defect, the
    verdict is VOID. Not a pass. A green head leg with no demonstrated base
    failure shows only that the harness ran.

That is deliberately not a per-probe option. A reusable harness whose pass
criteria are negotiable becomes a green-tick generator, and the whole value of
this toolkit is that its results mean something.

Verdicts:
  CONFIRMED       base showed the defect, head does not
  VOID            base did not show the defect, so nothing can be concluded
  FIX_INCOMPLETE  base showed it, head did not fully clear it
  NO_REGRESSION   regression mode: head is no worse than base
  REGRESSION      regression mode: head is worse than base
  INCONCLUSIVE    a non-vacuity gate failed
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from capability import detect, untested_section  # noqa: E402


def load_criteria(path: Path):
    spec = importlib.util.spec_from_file_location("amd_ci_criteria", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for required in ("MODE",):
        if not hasattr(mod, required):
            raise SystemExit(f"criteria module missing {required}")
    return mod


def run_probe(probe: Path, state: str, checkout: str, out_dir: Path,
              python: str, extra: list[str]) -> dict:
    """Probes emit JSON to a FILE, never stdout.

    Importing an application module frequently prints a banner, and a probe that
    writes its result to stdout has that banner concatenated into its JSON. That
    cost a run once already.
    """
    out = out_dir / f"obs_{state}.json"
    log = out_dir / f"probe_{state}.log"
    cmd = [python, str(probe), "--state", state, "--checkout", checkout,
           "--out", str(out), *extra]
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT).returncode
    obs: dict = {"_state": state, "_probe_rc": rc}
    if out.is_file():
        try:
            obs.update(json.loads(out.read_text()))
        except Exception as e:  # noqa: BLE001
            obs["_parse_error"] = f"{type(e).__name__}: {e}"
    else:
        obs["_missing_output"] = True
    return obs


def start_fixture(path: Path, extra: list[str], out_dir: Path, python: str,
                  timeout: int) -> tuple[subprocess.Popen | None, dict]:
    """Start a process that must stay alive across every state's probe.

    A VRAM holder is the motivating case: the condition being measured is
    "another process is holding memory", and that has to be true for the base
    and head observations equally or they are not comparable.

    The fixture signals readiness with one JSON line containing
    `"status": "READY"`. Polling a log for a sentinel beats sleeping, because a
    fixed sleep either wastes time or races.
    """
    log = out_dir / "fixture.log"
    fh = open(log, "w")
    proc = subprocess.Popen([python, str(path), *extra], stdout = fh, stderr = subprocess.STDOUT)
    info: dict = {"pid": proc.pid, "ready": False, "log": str(log)}
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            info["exited_early"] = proc.returncode
            break
        for line in log.read_text(errors = "replace").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                j = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if j.get("status") == "READY":
                info.update(j)
                info["ready"] = True
                break
        if info["ready"]:
            break
        time.sleep(2)
    print(f"fixture ready={info['ready']} pid={proc.pid}", flush = True)
    return proc, info


def stop_fixture(proc: subprocess.Popen) -> None:
    """Stop by PID.

    Never `pkill -f <pattern>`: the pattern matches the shell that runs it, so
    the caller kills itself. That happened once and cost a debugging cycle.
    """
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout = 20)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required = True, type = Path, help = "states.json from states.py")
    ap.add_argument("--probe", required = True, type = Path)
    ap.add_argument("--criteria", required = True, type = Path)
    ap.add_argument("--out-dir", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--title", default = "")
    ap.add_argument("--fixture", type = Path, default = None,
                    help = "program started before probing and stopped after; must print one "
                           "JSON line containing \"status\": \"READY\" when it is up")
    ap.add_argument("--fixture-arg", action = "append", default = [])
    ap.add_argument("--fixture-timeout", type = int, default = 300)
    ap.add_argument("probe_args", nargs = argparse.REMAINDER)
    args = ap.parse_args()

    args.out_dir.mkdir(parents = True, exist_ok = True)
    extra = [a for a in args.probe_args if a != "--"]
    states = json.loads(args.states.read_text())
    crit = load_criteria(args.criteria)

    fixture = None
    fixture_info: dict = {}
    if args.fixture:
        fixture, fixture_info = start_fixture(
            args.fixture, args.fixture_arg, args.out_dir, args.python, args.fixture_timeout)
        if not fixture_info.get("ready"):
            # A differential measured against a fixture that never came up is
            # not a failing result, it is no result.
            (args.out_dir / "VERDICT.md").write_text(
                f"## {args.title}\n\n**INCONCLUSIVE** - the fixture never became ready, so the "
                f"condition being measured was never established.\n")
            print((args.out_dir / "VERDICT.md").read_text())
            return 0

    obs: dict[str, dict] = {}
    try:
        for name, path in states["paths"].items():
            print(f"== probing {name}", flush = True)
            obs[name] = run_probe(args.probe, name, path, args.out_dir, args.python, extra)
    finally:
        if fixture is not None:
            stop_fixture(fixture)
    if fixture_info:
        obs["_fixture"] = fixture_info
    (args.out_dir / "observations.json").write_text(json.dumps(obs, indent = 2))

    profile = detect(require_torch = False)
    lines: list[str] = [f"## {args.title or getattr(crit, 'TITLE', 'AMD CI differential')}", ""]
    short = {k: v[:9] for k, v in states.get("commits", {}).items()}
    lines.append("States: " + ", ".join(f"`{k}` {v}" for k, v in short.items()))
    lines.append("")

    # ---- non-vacuity gates run BEFORE any comparison is shown, so a failed gate
    # cannot be read past.
    gates = list(getattr(crit, "gates", lambda o: [])(obs))
    if gates:
        lines += ["| gate | ok | evidence |", "|---|---|---|"]
        for name, ok, ev in gates:
            lines.append(f"| {name} | {'yes' if ok else 'NO'} | {ev} |")
        lines.append("")

    table = getattr(crit, "table", lambda o: "")(obs)
    verdict, why = _decide(crit, obs, gates)

    if verdict == "INCONCLUSIVE":
        lines.append("**INCONCLUSIVE** - a non-vacuity gate failed. The comparison is not "
                     "trustworthy, and widening tolerances to rescue it would be the easiest "
                     "way to fake this result.")
    else:
        if table:
            lines += [table, ""]
        lines.append(f"**{verdict}** - {why}")

    needed = list(getattr(crit, "NEEDS", []))
    gap = untested_section(profile, needed)
    if gap:
        lines += ["", gap]

    report = "\n".join(lines)
    (args.out_dir / "VERDICT.md").write_text(report)
    (args.out_dir / "verdict.json").write_text(json.dumps(
        {"verdict": verdict, "why": why, "mode": crit.MODE}, indent = 2))
    print(report)
    return 0


def _decide(crit, obs: dict, gates: list) -> tuple[str, str]:
    if any(not ok for _, ok, _ in gates):
        return "INCONCLUSIVE", "a gate failed"

    base, head = obs.get("base"), obs.get("head")
    if base is None or head is None:
        return "VOID", "a required state is missing, so no comparison exists"

    if crit.MODE == "differential":
        shown = crit.base_shows_defect(base)
        if not shown:
            # The rule. Not overridable.
            return "VOID", (
                "the base state did not exhibit the defect, so this run did not reproduce "
                "the problem and cannot speak to whether the change fixes it")
        fixed = crit.head_is_fixed(head)
        # Underscore keys are harness metadata (fixtures), not states.
        extra = [n for n in obs if n not in ("base", "head") and not n.startswith("_")]
        others = all(crit.head_is_fixed(obs[n]) for n in extra) if extra else True
        if fixed and others:
            return "CONFIRMED", "the defect reproduces at the base and is absent at the head"
        return "FIX_INCOMPLETE", "the base reproduced the defect but the head did not clear it"

    if crit.MODE == "regression":
        worse, detail = crit.head_is_worse(base, head)
        if worse:
            return "REGRESSION", detail
        return "NO_REGRESSION", detail or "head is no worse than base"

    return "VOID", f"unknown criteria mode {crit.MODE!r}"


if __name__ == "__main__":
    sys.exit(main())

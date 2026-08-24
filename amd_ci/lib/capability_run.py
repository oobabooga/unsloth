#!/usr/bin/env python3
"""Run one probe once and let a criteria module say what this host can do.

Not every question is a differential. "Does this runner expose a GPU-rendered
browser stack" has no defect and no base/head pair, and the honest way to answer
it is NOT to point two states at the same tree so lib/differential.py has
something to compare: that would manufacture the shape of a comparison without
its content, and the VOID rule exists precisely to stop shapes standing in for
evidence.

So this is a separate, smaller runner with the same discipline:

  * the probe OBSERVES via --out, the criteria JUDGES, and neither does the
    other's job;
  * gates run before anything is shown, and a failed gate is INCONCLUSIVE,
    meaning "we could not tell", which lib/announce.py fails the job on;
  * a plain NO is a FINDING and keeps the job green, because "this host cannot
    do it" is the answer to a capability question, not a broken run;
  * the report ends with what this host could not answer, computed from the
    criteria's NEEDS, overlaid with what the probe actually established.

  python amd_ci/lib/capability_run.py \
      --probe amd_ci/probes/display_stack_probe.py \
      --criteria amd_ci/criteria/display_stack_capability.py \
      --out-dir "$AMD_CI_WORK/out/display"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from capability import detect, untested_section  # noqa: E402

# Same split as lib/announce.py: an unwelcome answer is still an answer.
NO_RESULT = {"INCONCLUSIVE"}


def load_criteria(path: Path):
    spec = importlib.util.spec_from_file_location("amd_ci_criteria", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if getattr(mod, "MODE", None) != "capability":
        raise SystemExit(
            f"{path} declares MODE={getattr(mod, 'MODE', None)!r}; capability_run.py only "
            f"runs MODE='capability'. Use differential.py for base/head criteria.")
    for required in ("gates", "verdict", "NEEDS"):
        if not hasattr(mod, required):
            raise SystemExit(f"criteria module missing {required}")
    return mod


def run_probe(probe: Path, out_dir: Path, python: str, extra: list[str]) -> dict:
    """Probes emit JSON to a FILE, never stdout: an import banner corrupts it."""
    out = out_dir / "obs_host.json"
    log = out_dir / "probe_host.log"
    cmd = [python, str(probe), "--state", "host", "--checkout", "", "--out", str(out), *extra]
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT).returncode
    obs: dict = {"_state": "host", "_probe_rc": rc}
    if out.is_file():
        try:
            obs.update(json.loads(out.read_text()))
        except Exception as e:  # noqa: BLE001
            obs["_parse_error"] = f"{type(e).__name__}: {e}"
    else:
        obs["_missing_output"] = True
    return obs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required = True, type = Path)
    ap.add_argument("--criteria", required = True, type = Path)
    ap.add_argument("--out-dir", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--title", default = "")
    ap.add_argument("probe_args", nargs = argparse.REMAINDER)
    args = ap.parse_args()

    args.out_dir.mkdir(parents = True, exist_ok = True)
    crit = load_criteria(args.criteria)
    extra = [a for a in args.probe_args if a != "--"]

    obs = run_probe(args.probe, args.out_dir, args.python, extra)
    (args.out_dir / "observations.json").write_text(json.dumps(obs, indent = 2))

    lines: list[str] = [f"## {args.title or getattr(crit, 'TITLE', 'AMD CI capability')}", ""]

    gates = list(crit.gates(obs))
    lines += ["| gate | ok | evidence |", "|---|---|---|"]
    for name, ok, ev in gates:
        lines.append(f"| {name} | {'yes' if ok else 'NO'} | {ev} |")
    lines.append("")

    if any(not ok for _, ok, _ in gates):
        v, why = "INCONCLUSIVE", ("a gate failed, so this run could not tell. That is not the "
                                 "same as the host being incapable")
    else:
        v, why = crit.verdict(obs)
        tbl = getattr(crit, "table", lambda o: "")(obs)
        if tbl:
            lines += [tbl, ""]
    lines.append(f"**{v}** - {why}")

    observed = getattr(crit, "observed_capabilities", lambda o: {})(obs) if v != "INCONCLUSIVE" else {}
    profile = detect(require_torch = False, observed = observed)
    needed = list(crit.NEEDS)
    lines += ["", untested_section(profile, needed)]
    if not [n for n in needed if not profile.capabilities.get(n, False)]:
        print(f"::warning::no capability gaps computed (NEEDS={needed or 'unset'}); "
              f"the report will not bound its own reach, so check NEEDS is complete")

    report = "\n".join(lines)
    (args.out_dir / "VERDICT.md").write_text(report)
    (args.out_dir / "verdict.json").write_text(json.dumps(
        {"verdict": v, "why": why, "mode": crit.MODE,
         "observed_capabilities": observed}, indent = 2))
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

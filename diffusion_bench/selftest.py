#!/usr/bin/env python3
"""CPU-only self test: runs specs/selftest_fake.json through matrix.py and score.py and checks the records,
the resume path and the scores. Needs numpy + pillow (lpips / scikit-image optional). Exit 0 = pass."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402


def main() -> int:
    out = C.WS / "temp" / "diffusion_bench" / "selftest"
    for p in sorted(out.rglob("*"), reverse = True):
        p.unlink() if p.is_file() else p.rmdir()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    run = lambda *a: subprocess.run([sys.executable, *map(str, a)], env = env, capture_output = True, text = True)  # noqa: E731
    first = run(HERE / "matrix.py", HERE / "specs/selftest_fake.json", "--out", out, "--no-gate")
    failures = []
    if first.returncode != 1:
        failures.append(f"matrix should exit 1 with one BROKEN cell, got {first.returncode}\n{first.stdout[-2000:]}")
    rec = C.read_record(out / "p1" / "ref")
    if not C.record_ok(rec) or len(rec["renders"]) != 4 or rec.get("step_s_derived") is None:
        failures.append(f"ref record incomplete: {rec and {k: rec.get(k) for k in ('verdict', 'step_s_derived')}}")
    if (C.read_record(out / "p1" / "broken") or {}).get("verdict") != "BROKEN":
        failures.append("broken cell not recorded as BROKEN")
    if len((C.read_record(out / "p2" / "ref") or {}).get("renders", [])) != 2:
        failures.append("pass override n=2 not applied")
    if not (out / "p1" / "video" / "v0.npz").exists():
        failures.append("video frames not saved")
    second = run(HERE / "matrix.py", HERE / "specs/selftest_fake.json", "--out", out, "--no-gate", "--only", "ref")
    if "ok record exists" not in second.stdout:
        failures.append("resume did not skip an ok cell")
    sc = run(HERE / "score.py", out)
    if sc.returncode != 0:
        failures.append(f"score failed: {sc.stderr[-2000:]}")
    else:
        rows = {r["tag"]: r for r in json.loads((out / "p1" / "scores.json").read_text())["rows"]}
        if rows["same"]["identical"] != "4/4":
            failures.append(f"same-seed fake cell should be identical to ref: {rows['same']}")
        if rows["noisy"]["identical"] != "0/4" or (rows["noisy"]["psnr"] or 99) >= 99:
            failures.append(f"noisy cell should differ from ref: {rows['noisy']}")
    # setup_studio: a missing path-shaped source or an invalid ref fails fast, never reaching `git fetch`.
    import setup_studio as S
    for bad, exc in (("/nonexistent/unsloth", FileNotFoundError), ("./missing_checkout", FileNotFoundError),
                     ("not a ref", ValueError)):
        try:
            S.resolve_tree(bad)
            failures.append(f"setup_studio.resolve_tree({bad!r}) did not raise")
        except exc:
            pass
        except Exception as e:  # noqa: BLE001
            failures.append(f"setup_studio.resolve_tree({bad!r}) raised {type(e).__name__}: {e}")
    for f in failures:
        print("FAIL:", f)
    print("selftest", "FAILED" if failures else "passed", f"({out})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

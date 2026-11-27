#!/usr/bin/env python3
"""Probe for unsloth #19900: fp8_lora.rowwise_supported() on ROCm. Observes only.
Per state: (1) one cell process: the real fp8 _scaled_mm calls (tensorwise and row-wise, e4m3fn and e4m3fnuz) on the
device, then unsloth's rowwise_supported() with _scaled_mm counted; (2) pytest of the state's own
tests/test_fp8_no_rowwise_scaled_mm.py -k rowwise_probe_is_cached_per_device."""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True); ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path); ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--work", type = Path, default = None)
    a = ap.parse_args()
    out_dir = a.out.parent; out_dir.mkdir(parents = True, exist_ok = True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([a.checkout] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"; env["PYTHONUNBUFFERED"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    cell_out = out_dir / f"cell_{a.state}.json"
    p = subprocess.run([a.python, "-u", str(HERE / "g103_rowwise_cell.py"), "--out", str(cell_out)], env = env,
                       capture_output = True, text = True, timeout = 900, encoding = "utf-8", errors = "replace")
    obs = {"state": a.state, "cell_rc": p.returncode, "cell_stderr_tail": p.stderr[-1500:]}
    try:
        obs["cell"] = json.loads(cell_out.read_text(encoding = "utf-8"))
    except Exception as e:
        obs["cell"] = {"error": f"no cell output: {e}"}
    t = subprocess.run([a.python, "-m", "pytest", "tests/test_fp8_no_rowwise_scaled_mm.py", "-o", "addopts=", "-p",
                        "no:cacheprovider", "-q", "-rA", "-k", "rowwise_probe_is_cached_per_device"], cwd = a.checkout,
                       env = env, capture_output = True, text = True, timeout = 900, encoding = "utf-8", errors = "replace")
    text = t.stdout + t.stderr
    obs["pytest_rc"] = t.returncode
    obs["pytest_passed"] = sorted(set(re.findall(r"^PASSED (\S+)", text, re.M)))
    obs["pytest_failed"] = sorted(set(re.findall(r"^FAILED (\S+)", text, re.M)))
    obs["pytest_tail"] = text[-1500:]
    a.out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
    return 0

if __name__ == "__main__":
    sys.exit(main())

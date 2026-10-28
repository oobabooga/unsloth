#!/usr/bin/env python3
"""Probe for unsloth main 2026-10-28 (25fd237aad -> 987731ec56) at one state (an unsloth checkout). Observes only.

Cells (today1028_cell.py, one process each): decisions, sft16, sft4, packed (head_dim 256, packing=True),
infer. Then pytest over the day's AMD-relevant test files (files absent at a state recorded as such).
The state's checkout is first on PYTHONPATH so its unsloth shadows the installed one.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = {
    "decisions": ["--mode", "decisions"],
    "sft16": ["--mode", "sft", "--quant", "16", "--head-dim", "64"],
    "sft4": ["--mode", "sft", "--quant", "4", "--head-dim", "64"],
    "packed": ["--mode", "packed", "--quant", "16", "--head-dim", "256"],
    "packed_r2": ["--mode", "packed", "--quant", "16", "--head-dim", "256"],
    "infer": ["--mode", "infer", "--head-dim", "64"],
}
TESTS = [
    "tests/test_block_cudagraphs_auto.py", "tests/test_cross_entropy_chunk_config.py",
    "tests/test_cross_entropy_sm120_chunk_config.py", "tests/test_flex_preferred_shared_memory.py",
    "tests/test_flex_small_smem_tiles.py", "tests/test_fp8_dequant_max_rows.py", "tests/test_fp8_rowwise_direct_route.py",
    "tests/test_hf_packed_attention.py", "tests/test_moe_fallback_tile_dims.py", "tests/test_packed_varlen_over_flex.py",
    "tests/test_sdpa_cudnn_length_bucket_masked.py", "tests/test_multi_lora_experts.py",
    "tests/test_gc_skip_auto_dist_cells.py", "tests/python/test_fused_linear_trl_routes.py",
    "tests/utils/test_packing.py", "tests/test_padding_free_eval.py",
]
CLEAR = ("UNSLOTH_BLOCK_CUDAGRAPHS", "UNSLOTH_BLOCK_COMPILE", "UNSLOTH_PACKED_VARLEN_OVER_FLEX", "UNSLOTH_TORCH_VARLEN",
         "UNSLOTH_FP8_ROWWISE_DIRECT", "UNSLOTH_FP8_DEQUANT_MAX_ROWS", "UNSLOTH_COMPILE_DISABLE", "UNSLOTH_FORCE_FLOAT32",
         "UNSLOTH_HF_PACKED_VARLEN", "UNSLOTH_FLASH_ATTENTION_IMPL", "UNSLOTH_FLEX_ATTENTION_FOR_LARGE_HEAD_DIM")


def _env(checkout: str, extra: dict) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = checkout + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for k in CLEAR:
        env.pop(k, None)
    env.update(extra)
    return env


def _pytest_one(checkout, test, out_dir, state, python, timeout):
    tag = test.replace("/", "_").removesuffix(".py")
    junit = out_dir / f"junit_{state}_{tag}.xml"
    cmd = [python, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", f"--junitxml={junit}", test]
    extra = {"UNSLOTH_COMPILE_LOCATION": str(out_dir / f"compiled_{state}_pytest")}
    try:
        p = subprocess.run(cmd, cwd = checkout, env = _env(checkout, extra), capture_output = True,
                           text = True, timeout = timeout)
        rc, out, err = p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc = "timeout"
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = "TimeoutExpired"
    (out_dir / f"pytest_{state}_{tag}.log").write_text(out + "\n--- stderr ---\n" + err, encoding = "utf-8")
    rec = {"rc": rc, "passed": [], "failed": [], "errors": [], "skipped": {}, "fail_msgs": {}}
    if junit.is_file():
        try:
            for tc in ET.parse(junit).getroot().iter("testcase"):
                tid = f"{tc.get('classname')}::{tc.get('name')}"
                kinds = {c.tag: c for c in tc}
                if "failure" in kinds or "error" in kinds:
                    (rec["failed"] if "failure" in kinds else rec["errors"]).append(tid)
                    k = kinds.get("failure", kinds.get("error"))
                    rec["fail_msgs"][tid] = ((k.get("message") or "") + "\n" + (k.text or "")[-1200:])[:1600]
                elif "skipped" in kinds:
                    rec["skipped"][tid] = (kinds["skipped"].get("message") or "")[:200]
                else:
                    rec["passed"].append(tid)
        except Exception as e:  # noqa: BLE001
            rec["junit_error"] = repr(e)
    else:
        rec["no_junit"] = True
    if not (isinstance(rc, int) and rc in (0, 1, 5)):
        rec["crashed"] = True
        rec["stderr_tail"] = err[-3000:]
        rec["stdout_tail"] = out[-1500:]
    return rec


def run_pytest(checkout: str, out_dir: Path, state: str, python: str, timeout: int) -> dict:
    """One pytest process per file, so a native abort costs only its own file and is named."""
    present = [t for t in TESTS if (Path(checkout) / t).exists()]
    obs: dict = {"selected": present, "absent_at_this_state": [t for t in TESTS if t not in present], "files": {}}
    if not present:
        obs["note"] = "no selected tests exist at this state"
        return obs
    passed, failed, errors, skipped, msgs = [], [], [], {}, {}
    for t in present:
        r = _pytest_one(checkout, t, out_dir, state, python, timeout)
        obs["files"][t] = {k: r.get(k) for k in ("rc", "crashed", "no_junit", "stderr_tail", "stdout_tail", "junit_error")
                           if r.get(k) is not None}
        obs["files"][t]["n"] = len(r["passed"]) + len(r["failed"]) + len(r["errors"]) + len(r["skipped"])
        passed += r["passed"]; failed += r["failed"]; errors += r["errors"]
        skipped.update(r["skipped"]); msgs.update(r["fail_msgs"])
    obs["rc"] = 0 if all(not f.get("crashed") for f in obs["files"].values()) else "crash"
    obs["crashed_files"] = sorted(t for t, f in obs["files"].items() if f.get("crashed"))
    obs.update(passed = sorted(passed), failed = sorted(failed), errors = sorted(errors), skipped = skipped,
               fail_msgs = msgs, n_passed = len(passed), n_failed = len(failed) + len(errors), n_skipped = len(skipped))
    return obs


def run_cell(checkout: str, name: str, out_dir: Path, state: str, python: str) -> dict:
    out, log = out_dir / f"cell_{state}_{name}.json", out_dir / f"cell_{state}_{name}.log"
    argv = list(CELLS[name])
    if name != "decisions":
        hd = argv[argv.index("--head-dim") + 1]
        argv += ["--model-dir", str(out_dir / f"tiny_mistral_hd{hd}")]  # shared by base and head: same bytes
    extra = {"UNSLOTH_COMPILE_LOCATION": str(out_dir / f"compiled_{state}_{name}")}
    cmd = [python, "-u", str(HERE / "today1028_cell.py"), "--out", str(out), *argv]
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run(cmd, cwd = str(out_dir), env = _env(checkout, extra), stdout = fh,
                                stderr = subprocess.STDOUT, timeout = 1500).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    rec: dict = {"rc": rc}
    if out.is_file():
        try:
            rec.update(json.loads(out.read_text(encoding = "utf-8")))
        except Exception as e:  # noqa: BLE001
            rec["parse_error"] = repr(e)
    else:
        rec["missing_output"] = True
    text = log.read_text(encoding = "utf-8", errors = "replace")
    rec["log_tail"] = text[-2500:]
    rec["log_block_graph_notes"] = [ln[:300] for ln in text.splitlines() if "CUDA graphs" in ln][:10]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 900)
    args = ap.parse_args()
    args.out = args.out.resolve()
    out_dir = args.out.parent
    obs: dict = {"state": args.state, "checkout": args.checkout}
    obs["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = args.checkout,
                                     capture_output = True, text = True).stdout.strip()
    obs["cells"] = {}
    if args.state != "merge":
        for name in CELLS:
            obs["cells"][name] = run_cell(args.checkout, name, out_dir, args.state, args.python)
            args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
        obs["pytest"] = run_pytest(args.checkout, out_dir, args.state, args.python, args.timeout)
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Probe for unsloth-zoo PR 2106 (sparse flex attention for MiniMax-M3 / DeepSeek-V3.2 / GLM-5) on ROCm, one state.
Observes only; JSON via --out.

0. Venv (built once, shared by both states, under --venv): ROCm torch from AMD's gfx1151 index, transformers
   --transformers (zoo's ceiling is 5.17.0; the patches no-op below 5.17), the head checkout's other deps, pytest.
   The state's checkout is first on PYTHONPATH (UNSLOTH_IS_PRESENT=1), so its unsloth_zoo shadows any installed one.
1. pytest of --tests as they are (their GPU parity tests skip on ROCm by marker).
2. head only: a copy of tests/test_sparse_attention_flex.py with `torch.version.hip is not None or` removed from the
   `gpu` skip marker, so its flex parity tests (the tests' own tolerances) run on gfx1151.
3. head only: zoo2106_cell.py cells, MiniMax-M3 and GLM-5 tiny models above the budget, flex on / off / fp32,
   one process each, then the compare step.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
AMD_INDEX = "https://repo.amd.com/rocm/whl/gfx1151/"
TORCH_PIN = "torch==2.11.0+rocm7.13.0"
FAMILIES = ("minimax_m3_vl", "glm_moe_dsa")
SKIP_NEEDLE = "torch.version.hip is not None or "
FLEX_TEST = "tests/test_sparse_attention_flex.py"
UNSKIPPED = "tests/test_zz_rocm_unskipped_sparse_attention_flex.py"
CLEAR_ENV = ("UNSLOTH_SPARSE_ATTN_FLEX", "UNSLOTH_SPARSE_INDEXER_CHUNKED", "UNSLOTH_SPARSE_INDEXER_Q_CHUNK",
             "UNSLOTH_SPARSE_ATTN_DSA_MIN_KEYS", "UNSLOTH_SPARSE_ATTN_SKIP_FULL", "UNSLOTH_COMPILE_DISABLE",
             "UNSLOTH_SPARSE_FLEX_VIRTUAL_KV", "TORCHDYNAMO_DISABLE", "PYTEST_ADDOPTS")


def _sh(cmd, log, timeout = 3600, env = None, cwd = None):
    with open(log, "ab") as fh:
        fh.write(("\n$ " + " ".join(cmd) + "\n").encode())
        fh.flush()
        try:
            return subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, timeout = timeout, env = env, cwd = cwd).returncode
        except subprocess.TimeoutExpired:
            return "timeout"
        except FileNotFoundError as e:
            fh.write(f"{e!r}\n".encode())
            return "not_found"


def build_venv(venv: Path, head: str, transformers_pin: str, out_dir: Path) -> dict:
    py = venv / "bin" / "python"
    info: dict = {"venv": str(venv), "python": str(py)}
    marker = venv / ".amd_ci_ready"
    log = out_dir / "venv_build.log"
    if marker.is_file():
        info["reused"] = True
    else:
        uv = shutil.which("uv")
        info["uv"] = uv
        steps = []
        if uv:
            steps.append(("venv", [uv, "venv", "-q", "--seed", "--python", sys.executable, str(venv)]))
            pip = [uv, "pip", "install", "--python", str(py)]
            steps.append(("torch", pip + ["--index-url", AMD_INDEX, "--extra-index-url", "https://pypi.org/simple",
                                         "--index-strategy", "unsafe-best-match", TORCH_PIN]))
            steps.append(("deps", pip + ["--extra-index-url", "https://pypi.org/simple", head,
                                        f"transformers=={transformers_pin}", "pytest", "pytest-timeout"]))
            # unsloth_zoo's import guard is find_spec("unsloth"): the package must exist; it is never imported here.
            steps.append(("unsloth", pip + ["--extra-index-url", "https://pypi.org/simple", "--no-deps", "unsloth"]))
        else:
            steps.append(("venv", [sys.executable, "-m", "venv", str(venv)]))
            pip = [str(py), "-m", "pip", "install", "-q"]
            steps.append(("torch", pip + ["--index-url", AMD_INDEX, "--extra-index-url", "https://pypi.org/simple", TORCH_PIN]))
            steps.append(("deps", pip + ["--extra-index-url", "https://pypi.org/simple", head,
                                        f"transformers=={transformers_pin}", "pytest", "pytest-timeout"]))
            steps.append(("unsloth", pip + ["--extra-index-url", "https://pypi.org/simple", "--no-deps", "unsloth"]))
        info["steps"] = {}
        for name, cmd in steps:
            rc = _sh(cmd, log, timeout = 3000)
            info["steps"][name] = rc
            if rc != 0:
                info["error"] = f"step {name} rc={rc}"
                info["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-3000:]
                return info
        marker.write_text("ok", encoding = "utf-8")
    code = ("import json, torch, transformers, triton; p = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None; "
            "print('AMDCI' + json.dumps({'torch': torch.__version__, 'hip': torch.version.hip, 'transformers': transformers.__version__, "
            "'triton': triton.__version__, 'cuda_available': torch.cuda.is_available(), 'device': p.name if p else None, "
            "'arch': getattr(p, 'gcnArchName', None) if p else None}))")
    p = subprocess.run([str(py), "-c", code], capture_output = True, text = True)
    line = next((ln for ln in (p.stdout or "").splitlines() if ln.startswith("AMDCI")), None)
    info["env"] = json.loads(line[5:]) if line else {"error": (p.stderr or "")[-1500:]}
    pl = subprocess.run([str(py), "-m", "pip", "list"], capture_output = True, text = True)
    info["pip_list"] = [ln for ln in (pl.stdout or "").splitlines()
                        if ln.split(" ")[0].lower() in ("torch", "triton", "transformers", "peft", "trl", "accelerate",
                                                        "torchao", "unsloth", "unsloth-zoo", "unsloth_zoo", "pytest")]
    return info


def _env(checkout: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = checkout + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["UNSLOTH_IS_PRESENT"] = "1"
    for k in CLEAR_ENV:
        env.pop(k, None)
    return env


def run_pytest(python: str, checkout: str, tests: list, out_dir: Path, tag: str, timeout: int) -> dict:
    present = [t for t in tests if (Path(checkout) / t.split("::")[0]).exists()]
    obs: dict = {"selected": present, "absent_at_this_state": [t for t in tests if t not in present]}
    if not present:
        obs["note"] = "no selected tests exist at this state"
        return obs
    junit = out_dir / f"junit_{tag}.xml"
    cmd = [python, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--timeout", "900",
           f"--junitxml={junit}", *present]
    obs["cmd"] = " ".join(cmd)
    try:
        p = subprocess.run(cmd, cwd = checkout, env = _env(checkout), capture_output = True, text = True, timeout = timeout)
        rc, out, err = p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc, err = "timeout", "TimeoutExpired"
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
    obs["rc"] = rc
    (out_dir / f"pytest_{tag}.log").write_text(out + "\n--- stderr ---\n" + err, encoding = "utf-8")
    obs["tail"] = out[-5000:]
    obs["stderr_tail"] = err[-1500:]
    passed, failed, errors, skipped, msgs = [], [], [], {}, {}
    if junit.is_file():
        try:
            for tc in ET.parse(junit).getroot().iter("testcase"):
                tid = f"{tc.get('classname')}::{tc.get('name')}"
                kinds = {c.tag: c for c in tc}
                if "failure" in kinds or "error" in kinds:
                    (failed if "failure" in kinds else errors).append(tid)
                    k = kinds.get("failure", kinds.get("error"))
                    msgs[tid] = ((k.get("message") or "") + "\n" + (k.text or "")[-1500:])[:2000]
                elif "skipped" in kinds:
                    skipped[tid] = (kinds["skipped"].get("message") or "")[:300]
                else:
                    passed.append(tid)
        except Exception as e:  # noqa: BLE001
            obs["junit_error"] = repr(e)
    obs.update(passed = sorted(passed), failed = sorted(failed), errors = sorted(errors), skipped = skipped,
               fail_msgs = msgs, n_passed = len(passed), n_failed = len(failed) + len(errors), n_skipped = len(skipped))
    return obs


def run_cells(python: str, checkout: str, out_dir: Path) -> dict:
    res: dict = {}
    for fam in FAMILIES:
        fam_res: dict = {}
        for mode in ("on", "off", "fp32"):
            out = out_dir / f"cell_{fam}_{mode}.json"
            pt = out_dir / f"cell_{fam}_{mode}.pt"
            log = out_dir / f"cell_{fam}_{mode}.log"
            cmd = [python, "-u", str(HERE / "zoo2106_cell.py"), "--family", fam, "--mode", mode, "--pt", str(pt), "--out", str(out)]
            rc = _sh(cmd, log, timeout = 1800, env = _env(checkout), cwd = checkout)
            rec: dict = {"rc": rc}
            if out.is_file():
                rec.update(json.loads(out.read_text(encoding = "utf-8")))
            else:
                rec["missing_output"] = True
            rec["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-2500:]
            fam_res[mode] = rec
        out = out_dir / f"compare_{fam}.json"
        log = out_dir / f"compare_{fam}.log"
        rc = _sh([python, str(HERE / "zoo2106_cell.py"), "--family", fam, "--compare", str(out_dir), "--out", str(out)],
                 log, timeout = 600, env = _env(checkout), cwd = checkout)
        fam_res["compare"] = json.loads(out.read_text(encoding = "utf-8")) if out.is_file() else {"rc": rc, "missing_output": True}
        res[fam] = fam_res
    return res


TINY_RE = r"(\w+): loss fp32 ([-\d.e]+) dense ([-\d.e]+) flex ([-\d.e]+); max grad rel vs fp32: dense ([-\d.e]+) flex ([-\d.e]+)"


def run_tiny_repeats(python: str, checkout: str, text: str, out_dir: Path, state: str, n: int = 5) -> dict:
    """test_tiny_model_flex_on_vs_off judges a max-over-parameters grad error, one sample. Repeat it n times (one
    process each) and record the numbers it prints, so the criteria judge the median, not one draw."""
    import re
    f = "tests/test_zz_rocm_tiny_repeat.py"
    (Path(checkout) / f).write_text(text, encoding = "utf-8")
    res: dict = {"runs": []}
    for i in range(n):
        cmd = [python, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider", "--timeout", "900", f + "::test_tiny_model_flex_on_vs_off"]
        try:
            p = subprocess.run(cmd, cwd = checkout, env = _env(checkout), capture_output = True, text = True, timeout = 1800)
            out, rc = (p.stdout or "") + (p.stderr or ""), p.returncode
        except subprocess.TimeoutExpired:
            out, rc = "", "timeout"
        (out_dir / f"tiny_repeat_{state}_{i}.log").write_text(out, encoding = "utf-8")
        rows = [{"family": m[0], "loss_fp32": float(m[1]), "loss_dense": float(m[2]), "loss_flex": float(m[3]),
                 "grad_dense": float(m[4]), "grad_flex": float(m[5])} for m in re.findall(TINY_RE, out)]
        res["runs"].append({"rc": rc, "rows": rows, "summary": out.strip().splitlines()[-1:] if out.strip() else []})
    (Path(checkout) / f).unlink()
    return res


COMPILE_TEST = "def test_attention_inside_torch_compile_matches_eager("
AA_FILE = "tests/test_zz_rocm_compile_aa.py"


def run_compile_aa(python: str, checkout: str, text: str, out_dir: Path, state: str, timeout: int) -> dict:
    """test_attention_inside_torch_compile_matches_eager compares compiled flex-on with compiled flex-off (both take
    the model's own path inside a compiled graph). Its A/A twin compiles flex-off twice: if the twin differs as much,
    the gap is compile-to-compile noise, not the patch. Three repeats of each, one process per repeat."""
    start = text.find(COMPILE_TEST)
    deco = text.rfind("\n\n", 0, start) + 2
    end = text.find("\n\n\n", start)
    res: dict = {"found": start > 0 and end > start}
    if not res["found"]:
        return res
    fn = text[deco:end]
    if "got, got_g = run(True, True)" not in fn:
        res["found"] = False
        return res
    aa = fn.replace("def test_attention_inside_torch_compile_matches_eager(", "def test_compile_aa_flex_off_twice(").replace(
        "got, got_g = run(True, True)", "got, got_g = run(False, True)")
    (Path(checkout) / AA_FILE).write_text(text + "\n\n" + aa + "\n", encoding = "utf-8")
    res["repeats"] = []
    for i in range(3):
        r = run_pytest(python, checkout, [AA_FILE + "::test_attention_inside_torch_compile_matches_eager",
                                          AA_FILE + "::test_compile_aa_flex_off_twice"], out_dir, f"{state}_compile_aa{i}", timeout)
        res["repeats"].append({k: r.get(k) for k in ("rc", "passed", "failed", "errors", "n_passed", "n_failed")}
                              | {"fail_msgs": {t: m[:400] for t, m in (r.get("fail_msgs") or {}).items()}})
    (Path(checkout) / AA_FILE).unlink()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--head-checkout", required = True)
    ap.add_argument("--venv", type = Path, default = None)
    ap.add_argument("--use-python", default = None, help = "skip the venv build and use this interpreter (local harness check)")
    ap.add_argument("--transformers", default = "5.17.0")
    ap.add_argument("--tests", nargs = "+", required = True)
    ap.add_argument("--timeout", type = int, default = 3600)
    args = ap.parse_args()
    args.out = args.out.resolve()
    out_dir = args.out.parent / f"{args.state}_artifacts"
    out_dir.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "checkout": args.checkout, "tests": args.tests}
    obs["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = args.checkout, capture_output = True, text = True).stdout.strip()
    obs["has_sparse_blocks"] = (Path(args.checkout) / "unsloth_zoo/flex_attention/sparse_blocks.py").is_file()

    def dump():
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    if args.use_python:
        python = args.use_python
        obs["venv"] = {"python": python, "local": True}
    else:
        obs["venv"] = build_venv(args.venv.resolve(), args.head_checkout, args.transformers, out_dir)
        python = obs["venv"]["python"]
        if obs["venv"].get("error"):
            dump()
            return 0
    if True:
        p = subprocess.run([python, "-c", "import unsloth_zoo; print('AMDCI' + unsloth_zoo.__file__)"],
                           cwd = args.checkout, env = _env(args.checkout), capture_output = True, text = True)
        line = next((ln for ln in (p.stdout or "").splitlines() if ln.startswith("AMDCI")), None)
        obs["unsloth_zoo_file"] = line[5:] if line else None
        obs["unsloth_zoo_import_error"] = None if line else (p.stdout + p.stderr)[-2000:]
    dump()
    if obs["has_sparse_blocks"]:
        obs["cells"] = run_cells(python, args.checkout, out_dir)
        dump()
        src = Path(args.checkout) / FLEX_TEST
        text = src.read_text(encoding = "utf-8")
        obs["unskip_needle_count"] = text.count(SKIP_NEEDLE)
        if obs["unskip_needle_count"]:
            (Path(args.checkout) / UNSKIPPED).write_text(text.replace(SKIP_NEEDLE, ""), encoding = "utf-8")
            obs["pytest_rocm_unskipped"] = run_pytest(python, args.checkout, [UNSKIPPED], out_dir, f"{args.state}_unskipped", args.timeout)
            (Path(args.checkout) / UNSKIPPED).unlink()
            dump()
            obs["tiny_repeats"] = run_tiny_repeats(python, args.checkout, text.replace(SKIP_NEEDLE, ""), out_dir, args.state)
            dump()
            obs["compile_aa"] = run_compile_aa(python, args.checkout, text.replace(SKIP_NEEDLE, ""), out_dir, args.state, args.timeout)
            dump()
    # One pytest process per file: a cross-file import-order effect inside torch._inductor must not read as a
    # base-vs-head difference (the base lacks the flex test file, so the orders differ).
    merged: dict = {"per_file": {}}
    for i, t in enumerate(args.tests):
        r = run_pytest(python, args.checkout, [t], out_dir, f"{args.state}_{i}", args.timeout)
        merged["per_file"][t] = {k: r.get(k) for k in ("rc", "n_passed", "n_failed", "n_skipped", "note")}
        for k in ("passed", "failed", "errors"):
            merged[k] = sorted(set(merged.get(k) or []) | set(r.get(k) or []))
        for k in ("skipped", "fail_msgs"):
            merged.setdefault(k, {}).update(r.get(k) or {})
        merged.setdefault("absent_at_this_state", []).extend(r.get("absent_at_this_state") or [])
        merged.setdefault("stderr_tail", "")
        if r.get("rc") not in (0, 1, None):
            merged["stderr_tail"] += f"[{t}] rc={r.get('rc')} {str(r.get('stderr_tail'))[-300:]} {str(r.get('tail'))[-300:]}"
    rcs = [v["rc"] for v in merged["per_file"].values() if v["rc"] is not None]
    merged["rc"] = next((r for r in rcs if r not in (0, 1)), max(rcs) if rcs else None)
    merged["n_passed"], merged["n_skipped"] = len(merged["passed"]), len(merged["skipped"])
    merged["n_failed"] = len(merged["failed"]) + len(merged["errors"])
    obs["pytest"] = merged
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())

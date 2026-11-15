#!/usr/bin/env python3
"""Tiny Unsloth training on gfx1151 (Linux or Windows) at one state of an unsloth or unsloth-zoo PR, with the
other repo pinned (a companion commit: main, or the paired PR's merge base / head). Observes only; JSON via --out.

0. Environment, shared by every state (marker file):
   Linux:   --venv DIR: a fresh venv, ROCm torch from --torch-index (AMD's gfx1151 index: generic ROCm wheels
            segfault in _grouped_mm on gfx1151), then the training stack.
   Windows: --use-current: the interpreter running this probe (the workflow installed ROCm torch into it and gated
            on it), then the training stack.
   Stack: transformers / trl / peft pins, unsloth's and unsloth-zoo's runtime deps, bitsandbytes, PyPI unsloth and
   unsloth_zoo --no-deps (metadata only: each state's checkouts come first on PYTHONPATH and shadow them).
1. Companion checkout for this state (cloned once, one worktree per commit).
2. train5_cell.py per cell in its own process; --repeat cells run twice at the base (A/A spread).
3. pytest, one process per file, of --tests (state repo) and --companion-tests (companion repo), as they exist at
   this state (absent files recorded).

The merge state, if states.py made one, is recorded and skipped: base vs head is the comparison.
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
PYPI = "https://pypi.org/simple"
WIN = sys.platform == "win32"
STACK = ["accelerate", "datasets>=3.4.1,!=4.0.*,!=4.1.0,!=4.4.*,!=4.5.0,<5.0.0", "sentencepiece>=0.2.0", "protobuf",
         "huggingface_hub>=0.34.0", "bitsandbytes", "einops", "msgspec", "psutil", "tqdm", "regex", "pillow",
         "packaging", "numpy", "wheel", "requests", "filelock", "typing_extensions", "typer>=0.19.0,<0.29", "rich",
         "pydantic", "pyyaml", "json5>=0.9.25", "nest-asyncio", "structlog>=24.1.0", "click>=8.0", "tyro",
         "diffusers", "pytest", "pytest-timeout", "certifi"]
CLEAR = ("UNSLOTH_COMPILE_DISABLE", "UNSLOTH_MOE_BACKEND", "UNSLOTH_DECODE_CUDA_GRAPHS", "UNSLOTH_FORCE_FLOAT32",
         "PYTEST_ADDOPTS", "TORCHDYNAMO_DISABLE", "UNSLOTH_BLOCK_COMPILE", "UNSLOTH_BLOCK_CUDAGRAPHS")


def sh(cmd, log: Path, timeout = 3600, env = None, cwd = None):
    with open(log, "ab") as fh:
        fh.write(("\n$ " + " ".join(map(str, cmd)) + "\n").encode("utf-8"))
        fh.flush()
        try:
            return subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, timeout = timeout, env = env,
                                  cwd = cwd).returncode
        except subprocess.TimeoutExpired:
            return "timeout"
        except FileNotFoundError as e:
            fh.write(f"{e!r}\n".encode("utf-8"))
            return "not_found"


def tail(p: Path, n = 3000):
    return p.read_text(encoding = "utf-8", errors = "replace")[-n:] if p.is_file() else ""


def pip_cmd(py: str):
    uv = shutil.which("uv")
    if uv and not WIN:
        return [uv, "pip", "install", "--python", py, "--index-strategy", "unsafe-best-match"]
    return [py, "-m", "pip", "install", "-q", "--disable-pip-version-check"]


def build_env(args, work: Path) -> dict:
    log = work / "env_build.log"
    if args.use_current:
        py = sys.executable
        marker = Path(py).resolve().parent.parent / ".train5_ready"
    else:
        venv = args.venv.resolve()
        py = str(venv / ("Scripts/python.exe" if WIN else "bin/python"))
        marker = venv / ".train5_ready"
    info: dict = {"python": py, "torch_spec": args.torch_spec, "torch_index": args.torch_index,
                  "transformers": args.transformers, "trl": args.trl, "peft": args.peft}
    if marker.is_file():
        info["reused"] = True
    else:
        steps = []
        if not args.use_current:
            uv = shutil.which("uv")
            steps.append(("venv", [uv, "venv", "-q", "--seed", "--python", sys.executable, str(args.venv.resolve())]
                          if uv and not WIN else [sys.executable, "-m", "venv", str(args.venv.resolve())]))
            steps.append(("torch", pip_cmd(py) + ["--index-url", args.torch_index, "--extra-index-url", PYPI,
                                                  *args.torch_spec.split()]))
        steps.append(("stack", pip_cmd(py) + ["--extra-index-url", PYPI, f"transformers=={args.transformers}",
                                              f"trl=={args.trl}", f"peft=={args.peft}", *STACK,
                                              *(["triton-windows<3.7"] if WIN else []), *args.extra_pip]))
        steps.append(("unsloth_meta", pip_cmd(py) + ["--extra-index-url", PYPI, "--no-deps", "unsloth", "unsloth_zoo"]))
        info["steps"] = {}
        for name, cmd in steps:
            rc = sh(cmd, log, timeout = 3000)
            info["steps"][name] = rc
            if rc != 0:
                info["error"] = f"step {name} rc={rc}"
                info["log_tail"] = tail(log)
                return info
        marker.write_text("ok", encoding = "utf-8")
    code = ("import json, torch, importlib.metadata as m\n"
            "ok = torch.cuda.is_available(); p = torch.cuda.get_device_properties(0) if ok else None\n"
            "d = {'torch': torch.__version__, 'hip': torch.version.hip, 'cuda_available': ok,\n"
            "     'arch': getattr(p, 'gcnArchName', None) if p else None, 'device': p.name if p else None}\n"
            "for k in ('transformers', 'trl', 'peft', 'bitsandbytes', 'triton', 'triton-windows', 'pytorch-triton-rocm',\n"
            "          'triton-rocm', 'torchvision', 'unsloth', 'unsloth_zoo', 'accelerate', 'datasets'):\n"
            "    try: d[k] = m.version(k)\n"
            "    except Exception: d[k] = None\n"
            "print('AMDCI' + json.dumps(d))\n")
    p = subprocess.run([py, "-c", code], capture_output = True, text = True, encoding = "utf-8", errors = "replace")
    line = next((ln for ln in (p.stdout or "").splitlines() if ln.startswith("AMDCI")), None)
    info["env"] = json.loads(line[5:]) if line else {"error": (p.stderr or "")[-1500:]}
    return info


def companion(args, work: Path, state: str) -> str | None:
    if not args.companion_repo:
        return None
    sha = args.companion_head if state == "head" else args.companion_base
    src = work / "companion_repo"
    if not src.exists():
        url = args.companion_repo
        rc = subprocess.run(["git", "clone", "-q", "--filter=blob:none", url, str(src)]).returncode
        if rc != 0:
            raise SystemExit(f"clone {url} failed")
    wt = work / f"companion_{sha[:12]}"
    if not wt.exists():
        subprocess.run(["git", "fetch", "-q", "origin", sha], cwd = src)
        rc = subprocess.run(["git", "worktree", "add", "-q", "--detach", str(wt), sha], cwd = src).returncode
        if rc != 0:
            raise SystemExit(f"worktree {sha} failed")
    return str(wt)


def apply_head_files(spec: Path, roots: dict) -> list:
    import hashlib
    out = []
    for entry in json.loads(spec.read_text(encoding = "utf-8")):
        target = Path(roots[entry["repo"]]) / entry["path"]
        rec = {"repo": entry["repo"], "path": entry["path"]}
        old = target.read_text(encoding = "utf-8").replace("\r\n", "\n") if target.is_file() else None
        sha = hashlib.sha256(old.encode("utf-8")).hexdigest() if old is not None else None
        if entry.get("orig_sha256") and sha != entry["orig_sha256"]:
            rec["error"] = f"{target} is not the PR-head version (sha256 {sha})"
        else:
            target.write_text(entry["content"], encoding = "utf-8", newline = "\n")
            rec["written"] = hashlib.sha256(entry["content"].encode("utf-8")).hexdigest()
        out.append(rec)
    return out


def state_env(paths: list, extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in [*paths, env.get("PYTHONPATH", "")] if p)
    for k in CLEAR:
        env.pop(k, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"
    env.update(extra or {})
    return env


def run_cell(py, cell, tag, paths, out_dir: Path, timeout, model_dir: Path, extra_env: dict | None = None):
    out, log = out_dir / f"cell_{tag}.json", out_dir / f"cell_{tag}.log"
    cdir = out_dir / f"run_{tag}"
    cdir.mkdir(parents = True, exist_ok = True)
    rc = sh([py, "-u", str(HERE / "train5_cell.py"), "--cell", cell, "--out", str(out), "--model-dir", str(model_dir)],
            log, timeout = timeout,
            env = state_env(paths, {"UNSLOTH_COMPILE_LOCATION": str(cdir / "unsloth_compiled_cache"), **(extra_env or {})}),
            cwd = str(cdir))
    rec: dict = {"rc": rc}
    if out.is_file():
        try:
            rec.update(json.loads(out.read_text(encoding = "utf-8")))
        except Exception as e:  # noqa: BLE001
            rec["parse_error"] = repr(e)
    else:
        rec["missing_output"] = True
    rec["log_tail"] = tail(log, 2500)
    return rec


def pytest_one(py, checkout, test, paths, out_dir: Path, tag, timeout):
    junit = out_dir / f"junit_{tag}.xml"
    cmd = [py, "-m", "pytest", "-q", "-rfEs", "--tb=short", "-p", "no:cacheprovider", "-o", "addopts=",
           "--timeout", "900", f"--junitxml={junit}", test]
    try:
        p = subprocess.run(cmd, cwd = checkout, env = state_env(paths, {
            "UNSLOTH_COMPILE_LOCATION": str(out_dir / f"compiled_{tag}")}), capture_output = True, text = True,
            encoding = "utf-8", errors = "replace", timeout = timeout)
        rc, out, err = p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc, err = "timeout", "TimeoutExpired"
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
    (out_dir / f"pytest_{tag}.log").write_text(out + "\n--- stderr ---\n" + err, encoding = "utf-8")
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
    # No junit = the process died before pytest's summary (a native abort, os._exit), whatever rc says.
    if not (isinstance(rc, int) and rc in (0, 1, 5)) or not junit.is_file():
        rec["crashed"] = True
        rec["stderr_tail"], rec["stdout_tail"] = err[-2500:], out[-1500:]
    return rec


def _per_test(py, checkout, test, paths, out_dir: Path, tag, timeout, first):
    c = subprocess.run([py, "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider", "-o", "addopts=", test],
                       cwd = checkout, env = state_env(paths), capture_output = True, text = True, encoding = "utf-8",
                       errors = "replace", timeout = 900)
    ids = [ln.strip() for ln in (c.stdout or "").splitlines() if "::" in ln and not ln.startswith(("<", " "))]
    merged = {"rc": first.get("rc"), "passed": [], "failed": [], "errors": [], "skipped": {}, "fail_msgs": {},
              "per_test": True, "n_ids": len(ids)}
    killers = []
    for j, tid in enumerate(ids):
        r = pytest_one(py, checkout, tid, paths, out_dir, f"{tag}_t{j}", min(timeout, 900))
        if r.get("crashed"):
            name = tid.replace("/", ".").removesuffix(".py") if "::" not in tid else \
                tid.split("::")[0].replace("/", ".").replace("\\", ".").removesuffix(".py") + "::" + tid.split("::", 1)[1]
            merged["errors"].append(name)
            merged["fail_msgs"][name] = f"process died (rc {r.get('rc')}): " + str(r.get("stderr_tail") or r.get("stdout_tail"))[-1200:]
            killers.append(tid)
            continue
        for k in ("passed", "failed", "errors"):
            merged[k] += r[k]
        merged["skipped"].update(r["skipped"])
        merged["fail_msgs"].update(r["fail_msgs"])
    merged["killers"] = killers
    if not ids:
        merged.update(crashed = True, stderr_tail = first.get("stderr_tail"), stdout_tail = first.get("stdout_tail"))
    return merged


def run_pytest(py, checkout, tests, paths, out_dir: Path, state, prefix, timeout):
    present = [t for t in tests if (Path(checkout) / t.split("::")[0]).exists()]
    obs: dict = {"selected": present, "absent_at_this_state": [t for t in tests if t not in present], "files": {},
                 "passed": [], "failed": [], "errors": [], "skipped": {}, "fail_msgs": {}}
    for i, t in enumerate(present):
        # File tag: no ":" (NTFS and upload-artifact refuse it); the test-ID prefix keeps it.
        tag = f"{state}_{'comp' if prefix else ''}{i}"
        r = pytest_one(py, checkout, t, paths, out_dir, tag, timeout)
        if r.get("crashed"):
            # The file killed its process: one process per test, so the killer is named and the rest still count.
            r = _per_test(py, checkout, t, paths, out_dir, tag, timeout, r)
        obs["files"][t] = {k: r.get(k) for k in ("rc", "crashed", "stderr_tail", "junit_error", "per_test", "n_ids", "killers")
                           if r.get(k) is not None}
        for k in ("passed", "failed", "errors"):
            obs[k] += [f"{prefix}{x}" for x in r[k]]
        obs["skipped"].update({f"{prefix}{k}": v for k, v in r["skipped"].items()})
        obs["fail_msgs"].update({f"{prefix}{k}": v for k, v in r["fail_msgs"].items()})
    obs["crashed_files"] = sorted(t for t, f in obs["files"].items() if f.get("crashed"))
    obs.update(n_passed = len(obs["passed"]), n_failed = len(obs["failed"]) + len(obs["errors"]),
               n_skipped = len(obs["skipped"]))
    return obs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--unit", required = True)
    ap.add_argument("--state-repo", choices = ("unsloth", "zoo"), required = True)
    ap.add_argument("--venv", type = Path, default = None)
    ap.add_argument("--use-current", action = "store_true")
    ap.add_argument("--torch-spec", default = "torch==2.11.0+rocm7.13.0 torchvision==0.26.0+rocm7.13.0")
    ap.add_argument("--torch-index", default = "https://repo.amd.com/rocm/whl/gfx1151/")
    ap.add_argument("--transformers", default = "5.17.0")
    ap.add_argument("--trl", default = "1.15.0")
    ap.add_argument("--peft", default = "0.21.3")
    ap.add_argument("--companion-repo", default = None)
    ap.add_argument("--companion-base", default = None)
    ap.add_argument("--companion-head", default = None)
    ap.add_argument("--cells", default = "env,flm16,flm4,vision,moe")
    ap.add_argument("--head-files", type = Path, default = None,
                    help = "JSON list of {repo: unsloth|zoo, path, orig_sha256, content}: written into the HEAD state's "
                         "checkouts (a local fix that is not pushed), after checking the file is still the PR-head version")
    ap.add_argument("--extra-pip", nargs = "*", default = [], help = "added to the stack install (e.g. compressed-tensors)")
    ap.add_argument("--repeat", default = "flm16,moe", help = "cells run twice at the base: the A/A spread")
    ap.add_argument("--tests", nargs = "*", default = [])
    ap.add_argument("--companion-tests", nargs = "*", default = [])
    ap.add_argument("--cell-timeout", type = int, default = 1500)
    ap.add_argument("--test-timeout", type = int, default = 1800)
    args = ap.parse_args()
    if bool(args.venv) == bool(args.use_current):
        raise SystemExit("exactly one of --venv / --use-current")
    args.out = args.out.resolve()
    work = args.out.parent
    out_dir = work / f"{args.state}_artifacts"
    out_dir.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "unit": args.unit, "checkout": args.checkout, "state_repo": args.state_repo,
                 "platform": sys.platform}

    def dump():
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    obs["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = args.checkout, capture_output = True,
                                     text = True, encoding = "utf-8").stdout.strip()
    if args.state not in ("base", "head"):
        obs["skipped"] = "only base and head are compared"
        dump()
        return 0
    obs["env"] = build_env(args, work)
    dump()
    if obs["env"].get("error"):
        return 0
    py = obs["env"]["python"]
    comp = companion(args, work, args.state)
    obs["companion"] = {"repo": args.companion_repo, "checkout": comp,
                        "sha": args.companion_head if args.state == "head" else args.companion_base}
    unsloth_ck, zoo_ck = (args.checkout, comp) if args.state_repo == "unsloth" else (comp, args.checkout)
    obs["paths"] = {"unsloth": unsloth_ck, "zoo": zoo_ck}
    paths = [p for p in (unsloth_ck, zoo_ck) if p]
    if args.head_files and args.state == "head":
        obs["head_files"] = apply_head_files(args.head_files, {"unsloth": unsloth_ck, "zoo": zoo_ck})
        dump()
        if any(r.get("error") for r in obs["head_files"]):
            return 0
    dump()
    # The tiny Llama, built once per job (seeded, no unsloth import): the same bytes for every state.
    model_dir = work / "tiny_llama"
    obs["model_build"] = {k: v for k, v in run_cell(py, "build", f"{args.state}_build", paths, out_dir, 900,
                                                    model_dir).items() if k in ("rc", "stage", "error", "model_dir")}
    if any(c in args.cells.split(",") for c in ("ctnvfp4", "ctmxfp8")):
        # Same bytes for every state: quantized with the PAIRED zoo head's packed_experts (base zoo may lack it).
        builder = companion(args, work, "head") if args.state_repo == "unsloth" else args.checkout
        obs["ct_build"] = {k: v for k, v in run_cell(py, "build_ct", f"{args.state}_build_ct", [builder, unsloth_ck], out_dir, 900,
                                                     model_dir, {"UNSLOTH_IS_PRESENT": "1"}).items()
                           if k in ("rc", "stage", "error", "ct_dirs", "builder_zoo", "log_tail")}
    if any(c.startswith("hd512_") for c in args.cells.split(",")):
        obs["hd512_build"] = {k: v for k, v in run_cell(py, "build_hd512", f"{args.state}_build_hd512", paths, out_dir, 900,
                                                        model_dir).items() if k in ("rc", "stage", "error", "hd512_dirs")}
    obs["cells"] = {}
    for cell in [c for c in args.cells.split(",") if c]:
        obs["cells"][cell] = run_cell(py, cell, f"{args.state}_{cell}", paths, out_dir, args.cell_timeout, model_dir)
        dump()
        if args.state == "base" and cell in args.repeat.split(","):
            obs["cells"][cell + "@aa"] = run_cell(py, cell, f"{args.state}_{cell}_aa", paths, out_dir, args.cell_timeout,
                                                  model_dir)
            dump()
    if args.tests:
        obs["pytest"] = run_pytest(py, args.checkout, args.tests, paths, out_dir, args.state, "", args.test_timeout)
        dump()
    if args.companion_tests and comp:
        obs["pytest_companion"] = run_pytest(py, comp, args.companion_tests, paths, out_dir, args.state, "companion:",
                                             args.test_timeout)
        dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""PR 19378 (Studio multi-GPU training backend) on gfx1151, one state. Observes only; JSON via --out.

0. Environment, built once per job (marker file), the approach of train5_probe.py:
   Linux:   --venv DIR: a fresh venv, ROCm torch from AMD's gfx1151 index, then the stack.
   Windows: --use-current: the interpreter running this probe (the workflow installed ROCm torch into it and
            gated on it), then the stack.
   Stack: Studio backend requirements (studio/backend/requirements/studio.txt), Studio's transformers / trl / peft
   pins, unsloth's runtime deps, unsloth-zoo main (git, --no-deps), PyPI unsloth --no-deps (metadata only: each
   state's checkout comes first on PYTHONPATH and shadows it).
1. p19378_live.py: torch device, where `unsloth` resolves from, and (head) parallel_plan capabilities / plans.
2. studio_mgpu_smoke.py --mode single: Studio's real worker (core.training.worker.run_training_process) trains a
   tiny Qwen3 LoRA for a few steps, no parallel field (today's path). Base runs it twice (A/A spread).
3. pytest of --tests as they exist at this state (absent files recorded), one process per file.
The merge state is recorded and skipped: base vs head is the comparison.
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
STACK = ["accelerate", "sentencepiece>=0.2.0", "protobuf", "bitsandbytes", "einops", "msgspec", "psutil", "tqdm",
         "regex", "pillow", "numpy", "wheel", "requests", "filelock", "typing_extensions", "rich", "pyyaml",
         "tyro", "diffusers", "pytest", "pytest-timeout", "pytest-asyncio", "certifi", "hf_transfer"]
CLEAR = ("UNSLOTH_COMPILE_DISABLE", "UNSLOTH_MOE_BACKEND", "UNSLOTH_FORCE_FLOAT32", "PYTEST_ADDOPTS",
         "TORCHDYNAMO_DISABLE")


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


def build_env(args, work: Path, checkout: str) -> dict:
    log = work / "env_build.log"
    if args.use_current:
        py = sys.executable
        marker = Path(py).resolve().parent.parent / ".p19378_ready"
    else:
        venv = (args.venv or (work / "venv_p19378")).resolve()
        py = str(venv / ("Scripts/python.exe" if WIN else "bin/python"))
        marker = venv / ".p19378_ready"
    info: dict = {"python": py, "torch_spec": args.torch_spec, "torch_index": args.torch_index,
                  "transformers": args.transformers, "trl": args.trl, "peft": args.peft}
    if marker.is_file():
        info["reused"] = True
    else:
        steps = []
        if not args.use_current:
            uv = shutil.which("uv")
            steps.append(("venv", [uv, "venv", "-q", "--seed", "--python", sys.executable, str(venv)]
                          if uv and not WIN else [sys.executable, "-m", "venv", str(venv)]))
            steps.append(("torch", pip_cmd(py) + ["--index-url", args.torch_index, "--extra-index-url", PYPI,
                                                  *args.torch_spec.split()]))
        req = Path(checkout) / "studio" / "backend" / "requirements" / "studio.txt"
        steps.append(("stack", pip_cmd(py) + ["--extra-index-url", PYPI, "-r", str(req),
                                              f"transformers=={args.transformers}", f"trl=={args.trl}",
                                              f"peft=={args.peft}", *STACK,
                                              *(["triton-windows<3.7"] if WIN else [])]))
        steps.append(("unsloth_meta", pip_cmd(py) + ["--extra-index-url", PYPI, "--no-deps", "unsloth",
                                                     "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo"]))
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
            "          'torchvision', 'unsloth', 'unsloth_zoo', 'accelerate', 'datasets', 'fastapi', 'pytest'):\n"
            "    try: d[k] = m.version(k)\n"
            "    except Exception: d[k] = None\n"
            "print('AMDCI' + json.dumps(d))\n")
    p = subprocess.run([py, "-c", code], capture_output = True, text = True, encoding = "utf-8", errors = "replace")
    line = next((ln for ln in (p.stdout or "").splitlines() if ln.startswith("AMDCI")), None)
    info["env"] = json.loads(line[5:]) if line else {"error": (p.stderr or "")[-1500:]}
    return info


def state_env(checkout: str, extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in [checkout, env.get("PYTHONPATH", "")] if p)
    for k in CLEAR:
        env.pop(k, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"
    env.update(extra or {})
    return env


def load_json(p: Path):
    if not p.is_file():
        return {"missing_output": True}
    try:
        return json.loads(p.read_text(encoding = "utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"parse_error": repr(e)}


def run_live(py, checkout, out_dir: Path, tag):
    out, log = out_dir / f"live_{tag}.json", out_dir / f"live_{tag}.log"
    backend = str(Path(checkout) / "studio" / "backend")
    rc = sh([py, "-u", str(HERE / "p19378_live.py"), "--backend-dir", backend, "--out", str(out)], log,
            timeout = 900, env = state_env(checkout, {"UNSLOTH_STUDIO_HOME": str(out_dir / f"studio_home_{tag}")}))
    rec = {"rc": rc, **load_json(out)}
    rec["log_tail"] = tail(log, 1500)
    return rec


def run_smoke(py, checkout, out_dir: Path, tag, args):
    out, log = out_dir / f"smoke_{tag}.json", out_dir / f"smoke_{tag}.log"
    backend = str(Path(checkout) / "studio" / "backend")
    home = out_dir / f"studio_home_{tag}"
    home.mkdir(parents = True, exist_ok = True)
    cmd = [py, "-u", str(HERE / "studio_mgpu_smoke.py"), "--backend-dir", backend, "--mode", "single",
           "--model", args.model, "--steps", str(args.steps), "--batch", "2", "--seq", "256",
           "--output-dir", str(out_dir / f"smoke_work_{tag}"), "--out", str(out), "--timeout", str(args.smoke_timeout)]
    rc = sh(cmd, log, timeout = args.smoke_timeout + 300,
            env = state_env(checkout, {"UNSLOTH_STUDIO_HOME": str(home),
                                       "UNSLOTH_COMPILE_LOCATION": str(out_dir / f"compiled_{tag}")}))
    rec = {"rc": rc, **load_json(out)}
    rec["log_tail"] = tail(log, 4000)
    return rec


def pytest_one(py, checkout, test, out_dir: Path, tag, timeout):
    backend = Path(checkout) / "studio" / "backend"
    junit = out_dir / f"junit_{tag}.xml"
    cmd = [py, "-m", "pytest", "-q", "-rfEs", "--tb=short", "-p", "no:cacheprovider", "-o", "addopts=",
           "--timeout", "600", f"--junitxml={junit}", test]
    try:
        p = subprocess.run(cmd, cwd = str(backend), env = state_env(checkout, {
            "UNSLOTH_STUDIO_HOME": str(out_dir / f"studio_home_pytest_{tag}"),
            "UNSLOTH_COMPILE_LOCATION": str(out_dir / f"compiled_pytest_{tag}")}),
            capture_output = True, text = True, encoding = "utf-8", errors = "replace", timeout = timeout)
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
            # A module-level skip is reported as a testsuite with skipped=1 and no testcase on some pytest versions.
            for ts in ET.parse(junit).getroot().iter("testsuite"):
                rec["suite_counts"] = {k: ts.get(k) for k in ("tests", "failures", "errors", "skipped")}
        except Exception as e:  # noqa: BLE001
            rec["junit_error"] = repr(e)
    if not (isinstance(rc, int) and rc in (0, 1, 5)) or not junit.is_file():
        rec["crashed"] = True
    rec["stdout_tail"], rec["stderr_tail"] = out[-2500:], err[-1500:]
    return rec


def run_pytest(py, checkout, tests, out_dir: Path, state, timeout):
    backend = Path(checkout) / "studio" / "backend"
    present = [t for t in tests if (backend / t.split("::")[0]).exists()]
    obs: dict = {"selected": present, "absent_at_this_state": [t for t in tests if t not in present], "files": {},
                 "passed": [], "failed": [], "errors": [], "skipped": {}, "fail_msgs": {}}
    for i, t in enumerate(present):
        r = pytest_one(py, checkout, t, out_dir, f"{state}_{i}", timeout)
        obs["files"][t] = {k: r.get(k) for k in ("rc", "crashed", "junit_error", "suite_counts", "stdout_tail",
                                                 "stderr_tail") if r.get(k) is not None}
        for k in ("passed", "failed", "errors"):
            obs[k] += r[k]
        obs["skipped"].update(r["skipped"])
        obs["fail_msgs"].update(r["fail_msgs"])
    obs["crashed_files"] = sorted(t for t, f in obs["files"].items() if f.get("crashed"))
    obs.update(n_passed = len(obs["passed"]), n_failed = len(obs["failed"]) + len(obs["errors"]),
               n_skipped = len(obs["skipped"]))
    return obs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--venv", type = Path, default = None)
    ap.add_argument("--use-current", action = "store_true")
    ap.add_argument("--torch-spec", default = "torch==2.11.0+rocm7.13.0 torchvision==0.26.0+rocm7.13.0")
    ap.add_argument("--torch-index", default = "https://repo.amd.com/rocm/whl/gfx1151/")
    # Studio's own pins (studio/backend/requirements/extras-no-deps.txt).
    ap.add_argument("--transformers", default = "5.5.4")
    ap.add_argument("--trl", default = "0.23.1")
    ap.add_argument("--peft", default = "0.18.1")
    ap.add_argument("--model", default = "trl-internal-testing/tiny-Qwen3ForCausalLM")
    ap.add_argument("--steps", type = int, default = 5)
    ap.add_argument("--smoke-timeout", type = int, default = 1800)
    ap.add_argument("--no-aa", action = "store_true")
    ap.add_argument("--tests", nargs = "*", default = [])
    ap.add_argument("--test-timeout", type = int, default = 1500)
    args = ap.parse_args()
    args.out = args.out.resolve()
    work = args.out.parent
    out_dir = work / f"{args.state}_artifacts"
    out_dir.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "checkout": args.checkout, "platform": sys.platform}

    def dump():
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    obs["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = args.checkout, capture_output = True,
                                     text = True, encoding = "utf-8").stdout.strip()
    if args.state not in ("base", "head"):
        obs["skipped"] = "only base and head are compared"
        dump()
        return 0
    obs["env"] = build_env(args, work, args.checkout)
    dump()
    if obs["env"].get("error"):
        return 0
    py = obs["env"]["python"]
    obs["live"] = run_live(py, args.checkout, out_dir, args.state)
    dump()
    obs["smoke"] = run_smoke(py, args.checkout, out_dir, args.state, args)
    dump()
    if args.state == "base" and not args.no_aa:
        obs["smoke_aa"] = run_smoke(py, args.checkout, out_dir, args.state + "_aa", args)
        dump()
    if args.tests:
        obs["pytest"] = run_pytest(py, args.checkout, args.tests, out_dir, args.state, args.test_timeout)
        dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())

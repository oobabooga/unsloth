#!/usr/bin/env python3
"""Shared helpers for unsloth-zoo probes on gfx1151: build one venv (ROCm torch pin + the head checkout's deps,
optionally a transformers override), run pytest per file with junit parsing, run a sub-process cell. Observes only.

The venv is built once under --venv and shared by both states. Each state's checkout is first on PYTHONPATH
(UNSLOTH_IS_PRESENT=1), so its unsloth_zoo shadows the installed one; the probe records which file was imported.

Triton: unsloth-zoo declares `triton>=3.0.0`. A torch whose Triton ships as `triton-rocm` / `pytorch-triton-rocm`
then gets PyPI's CUDA `triton` installed over the same `triton/` package directory. After the deps step the
ROCm Triton torch requires is reinstalled (and the CUDA one removed) when that happened; recorded as `triton_fixup`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

AMD_GFX1151_INDEX = "https://repo.amd.com/rocm/whl/gfx1151/"
PYPI = "https://pypi.org/simple"
CLEAR_ENV = ("UNSLOTH_SPARSE_ATTN_FLEX", "UNSLOTH_SPARSE_INDEXER_CHUNKED", "UNSLOTH_SPARSE_INDEXER_Q_CHUNK",
             "UNSLOTH_SPARSE_ATTN_DSA_MIN_KEYS", "UNSLOTH_SPARSE_ATTN_SKIP_FULL", "UNSLOTH_COMPILE_DISABLE",
             "UNSLOTH_SPARSE_FLEX_VIRTUAL_KV", "TORCHDYNAMO_DISABLE", "PYTEST_ADDOPTS", "TORCHINDUCTOR_USE_RAND4X",
             "TORCH_COMPILE_CACHE_KEY_TAG")
ENV_CODE = (
    "import json, torch\n"
    "ok = torch.cuda.is_available()\n"
    "p = torch.cuda.get_device_properties(0) if ok else None\n"
    "smoke = None\n"
    "if ok:\n"
    "    a = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16)\n"
    "    smoke = float((a @ a).float().abs().sum().item())\n"
    "try:\n"
    "    import triton; tv = triton.__version__\n"
    "except Exception as e:\n"
    "    tv = 'import failed: ' + repr(e)[:200]\n"
    "try:\n"
    "    import transformers; trv = transformers.__version__\n"
    "except Exception as e:\n"
    "    trv = 'import failed: ' + repr(e)[:200]\n"
    "print('AMDCI' + json.dumps({'torch': torch.__version__, 'hip': torch.version.hip, 'transformers': trv,\n"
    "    'triton': tv, 'cuda_available': ok, 'device': p.name if p else None,\n"
    "    'arch': getattr(p, 'gcnArchName', None) if p else None,\n"
    "    'arch_list': torch.cuda.get_arch_list() if ok else [], 'matmul_smoke': smoke}))\n"
)


def sh(cmd, log: Path, timeout = 3600, env = None, cwd = None):
    with open(log, "ab") as fh:
        fh.write(("\n$ " + " ".join(map(str, cmd)) + "\n").encode())
        fh.flush()
        try:
            return subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, timeout = timeout, env = env, cwd = cwd).returncode
        except subprocess.TimeoutExpired:
            return "timeout"
        except FileNotFoundError as e:
            fh.write(f"{e!r}\n".encode())
            return "not_found"


def _torch_triton_requirement(py: str) -> str | None:
    code = ("import importlib.metadata as m\n"
            "r = [x for x in (m.requires('torch') or []) if 'triton' in x.lower() and 'extra ==' not in x]\n"
            "print('AMDCI' + (r[0] if r else ''))\n")
    p = subprocess.run([py, "-c", code], capture_output = True, text = True)
    line = next((ln for ln in (p.stdout or "").splitlines() if ln.startswith("AMDCI")), "AMDCI")
    return line[5:].split(";")[0].strip() or None


def _installed(py: str, dist: str) -> str | None:
    p = subprocess.run([py, "-c", f"import importlib.metadata as m; print('AMDCI' + m.version({dist!r}))"],
                       capture_output = True, text = True)
    line = next((ln for ln in (p.stdout or "").splitlines() if ln.startswith("AMDCI")), None)
    return line[5:] if line else None


def build_venv(venv: Path, head: str, torch_pin: str, torch_index: str, transformers_pin: str | None,
               out_dir: Path) -> dict:
    """torch_index 'amd' = AMD's gfx1151 index; otherwise a full index URL (e.g. download.pytorch.org/whl/rocm7.14)."""
    py = venv / "bin" / "python"
    index = AMD_GFX1151_INDEX if torch_index == "amd" else torch_index
    info: dict = {"venv": str(venv), "python": str(py), "torch_pin": torch_pin, "torch_index": index,
                  "transformers_pin": transformers_pin}
    marker = venv / ".amd_ci_ready"
    log = out_dir / "venv_build.log"
    if marker.is_file():
        info["reused"] = True
    else:
        uv = shutil.which("uv")
        info["uv"] = uv
        if uv:
            mk = [uv, "venv", "-q", "--seed", "--python", sys.executable, str(venv)]
            pip = [uv, "pip", "install", "--python", str(py)]
            strat = ["--index-strategy", "unsafe-best-match"]
        else:
            mk = [sys.executable, "-m", "venv", str(venv)]
            pip = [str(py), "-m", "pip", "install", "-q"]
            strat = []
        steps = [("venv", mk),
                 ("torch", pip + ["--index-url", index, "--extra-index-url", PYPI, *strat, torch_pin]),
                 # Deps resolve against the installed torch (it satisfies zoo's torch range, so it stays).
                 ("deps", pip + ["--extra-index-url", PYPI, head, "pytest", "pytest-timeout"]),
                 # unsloth_zoo's import guard is find_spec("unsloth"): the package must exist; never imported here.
                 ("unsloth", pip + ["--extra-index-url", PYPI, "--no-deps", "unsloth"])]
        if transformers_pin:
            # Above zoo's declared ceiling on purpose (the PR under test targets it); its own deps resolve normally.
            steps.append(("transformers", pip + ["--extra-index-url", PYPI, f"transformers=={transformers_pin}"]))
        info["steps"] = {}
        for name, cmd in steps:
            rc = sh(cmd, log, timeout = 3000)
            info["steps"][name] = rc
            if rc != 0:
                info["error"] = f"step {name} rc={rc}"
                info["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-3000:]
                return info
            if name == "deps":
                info["triton_fixup"] = _fix_triton(str(py), pip, strat, index, log)
                if info["triton_fixup"].get("error"):
                    info["error"] = "triton fixup: " + info["triton_fixup"]["error"]
                    info["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-3000:]
                    return info
        marker.write_text("ok", encoding = "utf-8")
    p = subprocess.run([str(py), "-c", ENV_CODE], capture_output = True, text = True)
    line = next((ln for ln in (p.stdout or "").splitlines() if ln.startswith("AMDCI")), None)
    info["env"] = json.loads(line[5:]) if line else {"error": (p.stderr or "")[-1500:]}
    pl = subprocess.run([str(py), "-m", "pip", "list"], capture_output = True, text = True)
    info["pip_list"] = [ln for ln in (pl.stdout or "").splitlines()
                        if ln.split(" ")[0].lower() in ("torch", "triton", "triton-rocm", "pytorch-triton-rocm",
                                                        "transformers", "peft", "trl", "accelerate", "torchao",
                                                        "unsloth", "unsloth-zoo", "unsloth_zoo", "pytest")]
    return info


def _fix_triton(py: str, pip: list, strat: list, index: str, log: Path) -> dict:
    req = _torch_triton_requirement(py)
    res: dict = {"torch_requires": req}
    if not req:
        return res
    name = re.split(r"[=<>!~ ]", req, 1)[0].strip()
    res["torch_triton_dist"] = name
    if name.lower() == "triton":
        return res
    cuda_triton = _installed(py, "triton")
    res["pypi_triton_installed"] = cuda_triton
    if not cuda_triton:
        return res
    uninstall = (pip[:2] + ["uninstall", "--python", py, "triton"]) if pip[0].endswith("uv") else \
        [py, "-m", "pip", "uninstall", "-y", "triton"]
    res["uninstall_rc"] = sh(uninstall, log, timeout = 600)
    again = "--reinstall" if pip[0].endswith("uv") else "--force-reinstall"
    rc = sh(pip + ["--index-url", index, "--extra-index-url", PYPI, *strat, again, "--no-deps", req],
            log, timeout = 1800)
    res["reinstall_rc"] = rc
    if rc != 0 or res["uninstall_rc"] != 0:
        res["error"] = f"uninstall rc {res['uninstall_rc']} reinstall rc {rc}"
    return res


def state_env(checkout: str, extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = checkout + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["UNSLOTH_IS_PRESENT"] = "1"
    for k in CLEAR_ENV:
        env.pop(k, None)
    env.update(extra or {})
    return env


def zoo_import(python: str, checkout: str) -> dict:
    p = subprocess.run([python, "-c", "import unsloth_zoo; print('AMDCI' + unsloth_zoo.__file__)"],
                       cwd = checkout, env = state_env(checkout), capture_output = True, text = True, timeout = 900)
    line = next((ln for ln in (p.stdout or "").splitlines() if ln.startswith("AMDCI")), None)
    return {"file": line[5:] if line else None, "error": None if line else (p.stdout + p.stderr)[-2500:]}


def run_pytest(python: str, checkout: str, tests: list, out_dir: Path, tag: str, timeout: int) -> dict:
    present = [t for t in tests if (Path(checkout) / t.split("::")[0]).exists()]
    obs: dict = {"selected": present, "absent_at_this_state": [t for t in tests if t not in present]}
    if not present:
        obs["note"] = "no selected tests exist at this state"
        return obs
    junit = out_dir / f"junit_{tag}.xml"
    cmd = [python, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "-o", "addopts=", "--timeout", "900",
           f"--junitxml={junit}", *present]
    obs["cmd"] = " ".join(cmd)
    try:
        p = subprocess.run(cmd, cwd = checkout, env = state_env(checkout), capture_output = True, text = True, timeout = timeout)
        rc, out, err = p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc, err = "timeout", "TimeoutExpired"
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
    obs["rc"] = rc
    (out_dir / f"pytest_{tag}.log").write_text(out + "\n--- stderr ---\n" + err, encoding = "utf-8")
    obs["tail"] = out[-4000:]
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


def run_pytest_per_file(python: str, checkout: str, tests: list, out_dir: Path, state: str, timeout: int) -> dict:
    """One pytest process per file: cross-file import order inside torch._inductor must not read as base vs head."""
    merged: dict = {"per_file": {}, "passed": [], "failed": [], "errors": [], "skipped": {}, "fail_msgs": {},
                    "absent_at_this_state": [], "stderr_tail": ""}
    for i, t in enumerate(tests):
        r = run_pytest(python, checkout, [t], out_dir, f"{state}_{i}", timeout)
        merged["per_file"][t] = {k: r.get(k) for k in ("rc", "n_passed", "n_failed", "n_skipped", "note")}
        for k in ("passed", "failed", "errors"):
            merged[k] = sorted(set(merged[k]) | set(r.get(k) or []))
        for k in ("skipped", "fail_msgs"):
            merged[k].update(r.get(k) or {})
        merged["absent_at_this_state"].extend(r.get("absent_at_this_state") or [])
        if r.get("rc") not in (0, 1, None):
            merged["stderr_tail"] += f"[{t}] rc={r.get('rc')} {str(r.get('stderr_tail'))[-300:]} {str(r.get('tail'))[-300:]}"
    rcs = [v["rc"] for v in merged["per_file"].values() if v["rc"] is not None]
    merged["rc"] = next((r for r in rcs if r not in (0, 1)), max(rcs) if rcs else None)
    merged["n_passed"], merged["n_skipped"] = len(merged["passed"]), len(merged["skipped"])
    merged["n_failed"] = len(merged["failed"]) + len(merged["errors"])
    return merged


def run_cell(python: str, script: Path, args: list, checkout: str, out_dir: Path, tag: str, timeout = 1800,
             extra_env: dict | None = None) -> dict:
    """A cell writes its JSON to --out; a crash (HIP fault, timeout) is recorded, not raised."""
    out = out_dir / f"cell_{tag}.json"
    log = out_dir / f"cell_{tag}.log"
    rc = sh([python, "-u", str(script), *args, "--out", str(out)], log, timeout = timeout,
            env = state_env(checkout, extra_env), cwd = checkout)
    rec: dict = {"rc": rc}
    if out.is_file():
        try:
            rec.update(json.loads(out.read_text(encoding = "utf-8")))
        except Exception as e:  # noqa: BLE001
            rec["parse_error"] = repr(e)
    else:
        rec["missing_output"] = True
    rec["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-2500:] if log.is_file() else ""
    return rec

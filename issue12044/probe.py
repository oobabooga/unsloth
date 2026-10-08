#!/usr/bin/env python3
"""Probe for unslothai/unsloth#12044 on native Windows 11 x64, Python 3.12. Observes only.

The git states (PR 12151 merge commit vs its parent) are labels: each state maps to the
RELEASED unsloth / unsloth-zoo pair it corresponds to, installed from PyPI.

  base -> 2026.9.14 (torch cap <2.13.0)    head -> 2026.10.2 (torch cap <2.15.0)

Leg 1, every state, spoof stripped, no GPU: fresh venv, torch==2.14.0+cu130 from the cu130
index, then the unsloth pair with torch pinned in the same command. Records the resolver
outcome, freeze, triton-windows / bitsandbytes / xformers, and import outcomes.

Leg 2, head only, NVIDIA spoof active, real gfx1151 underneath: fresh venv with the ROCm
torch unsloth's own install.ps1 picks on Windows (2.11.0+rocm7.14.0 from AMD's multi-arch
index), the head pair, then a tiny GPT-OSS q_proj/v_proj LoRA (r=4, alpha=8) for 5 SFT steps,
arms default and UNSLOTH_COMPILE_DISABLE=1. Wiring only: HIP kernels, not CUDA.

Writes JSON to --out; never prints the result to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PAIRS = {"base": "2026.9.14", "head": "2026.10.2"}
CU130 = "https://download.pytorch.org/whl/cu130"
PYPI = "https://pypi.org/simple"
ROCM_INDEX = "https://repo.amd.com/rocm/whl-multi-arch/"
ROCM_TORCH = "torch[device-gfx1151]==2.11.0+rocm7.14.0"
ROCM_VISION = "torchvision[device-gfx1151]==0.26.0+rocm7.14.0"
TORCH_CU130 = "torch==2.14.0+cu130"
VISION_CU130 = "torchvision==0.29.0+cu130"
HERE = Path(__file__).resolve().parent


def _tail(text: str, n: int = 4000) -> str:
    return text[-n:] if text else ""


def run(cmd: list[str], env: dict, log: Path, timeout: int) -> dict:
    t0 = time.time()
    try:
        cp = subprocess.run(cmd, env = env, capture_output = True, timeout = timeout)
        out = cp.stdout.decode("utf-8", "replace") + cp.stderr.decode("utf-8", "replace")
        rc = cp.returncode
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b"") + (e.stderr or b"")).decode("utf-8", "replace") + "\n<TIMEOUT>"
        rc = "timeout"
    except Exception as e:  # noqa: BLE001
        out, rc = f"{type(e).__name__}: {e}", "spawn_error"
    with open(log, "a", encoding = "utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n{out}\n[rc={rc}]\n")
    return {"rc": rc, "seconds": round(time.time() - t0, 1), "tail": _tail(out)}


def spoof_free_env() -> dict:
    """The job env has the NVIDIA spoof on PYTHONPATH / PATH. Leg 1 must see the real host."""
    env = dict(os.environ)
    def is_spoof(entry: str) -> bool:
        p = Path(entry)
        try:
            sc = p / "sitecustomize.py"
            if sc.is_file() and "_amd_ci_real" in sc.read_text(encoding = "utf-8", errors = "replace"):
                return True
            return (p / "nvidia-smi.cmd").is_file()
        except OSError:
            return False
    for key in ("PYTHONPATH", "PATH"):
        parts = [x for x in env.get(key, "").split(os.pathsep) if x and not is_spoof(x)]
        env[key] = os.pathsep.join(parts)
    env.pop("AMD_CI_SPOOFED_VENDOR", None)
    env["HIP_VISIBLE_DEVICES"] = ""
    env["ROCR_VISIBLE_DEVICES"] = ""
    return env


def base_python() -> str:
    # The job's AMD_CI_PY is a venv; build fresh venvs from the interpreter under it.
    return getattr(sys, "_base_executable", None) or sys.executable


def make_venv(path: Path, env: dict, log: Path) -> tuple[str, dict]:
    r = run([base_python(), "-m", "venv", str(path)], env, log, 600)
    py = str(path / "Scripts" / "python.exe") if os.name == "nt" else str(path / "bin" / "python")
    if r["rc"] == 0:
        run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], env, log, 600)
    return py, r


INSPECT = r'''
import json, sys, importlib.metadata as md
res = {}
def ver(n):
    try: return md.version(n)
    except Exception: return None
for n in ("torch","torchvision","unsloth","unsloth_zoo","triton","triton-windows","bitsandbytes",
          "xformers","transformers","trl","peft","accelerate","datasets","torchao"):
    res["v_" + n] = ver(n)
try:
    res["unsloth_extras"] = sorted(md.metadata("unsloth").get_all("Provides-Extra") or [])
except Exception as e:
    res["unsloth_extras"] = None
try:
    res["unsloth_requires_torch"] = [r for r in (md.requires("unsloth") or []) if r.startswith("torch")]
except Exception:
    res["unsloth_requires_torch"] = None
json.dump(res, open(sys.argv[1], "w", encoding="utf-8"))
'''

IMPORT_ONE = r'''
import json, sys, traceback
mod, out = sys.argv[1], sys.argv[2]
res = {"module": mod}
try:
    m = __import__(mod)
    res["ok"] = True
    res["version"] = str(getattr(m, "__version__", None))
    if mod == "torch":
        res["cuda"] = m.version.cuda; res["hip"] = getattr(m.version, "hip", None)
        res["is_available"] = bool(m.cuda.is_available())
        res["real_hip"] = getattr(m.version, "_amd_ci_real_hip", None)
        try:
            res["device_name"] = m.cuda.get_device_name(0) if res["is_available"] else None
        except Exception as e:
            res["device_name"] = f"{type(e).__name__}: {e}"
except BaseException as e:
    res["ok"] = False
    res["exc_type"] = type(e).__name__
    res["exc"] = str(e)[:1500]
    res["tb_tail"] = traceback.format_exc()[-2500:]
json.dump(res, open(out, "w", encoding="utf-8"))
'''


def py_json(py: str, code: str, args: list[str], env: dict, log: Path, out: Path, timeout = 900) -> dict:
    script = out.with_suffix(".py")
    script.write_text(code, encoding = "utf-8")
    r = run([py, str(script), *args, str(out)], env, log, timeout)
    if out.is_file():
        try:
            return json.loads(out.read_text(encoding = "utf-8"))
        except Exception as e:  # noqa: BLE001
            return {"_parse_error": str(e), "_run": r}
    return {"_missing": True, "_run": r}


def long_paths_enabled():
    if os.name != "nt":
        return None
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem")
        return int(winreg.QueryValueEx(k, "LongPathsEnabled")[0])
    except Exception:
        return None


def leg1(state: str, work: Path, log: Path) -> dict:
    v = PAIRS[state]
    env = spoof_free_env()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    res: dict = {"pair": v}
    vdir = work / f"v1{state[0]}"
    py, r = make_venv(vdir, env, log)
    res["venv"] = r
    if r["rc"] != 0:
        return res
    res["torch_install"] = run([py, "-m", "pip", "install", TORCH_CU130, VISION_CU130,
                                "--index-url", CU130, "--extra-index-url", PYPI], env, log, 3600)
    res["pair_install"] = run([py, "-m", "pip", "install", f"unsloth=={v}", f"unsloth-zoo=={v}",
                               TORCH_CU130, VISION_CU130, "--extra-index-url", CU130], env, log, 3600)
    t = res["pair_install"]["tail"]
    res["resolver_conflict"] = ("ResolutionImpossible" in t) or ("conflicting dependencies" in t) \
        or ("Cannot install" in t)
    fr = run([py, "-m", "pip", "freeze"], env, log, 300)
    res["freeze"] = fr["tail"] if fr["rc"] == 0 else None
    res["inspect"] = py_json(py, INSPECT, [], env, log, work / f"inspect1_{state}.json")
    res["imports"] = {m: py_json(py, IMPORT_ONE, [m], env, log, work / f"imp1_{state}_{m}.json")
                      for m in ("torch", "triton", "bitsandbytes", "unsloth")}
    return res


def _requires_without_torch(py: str, env: dict, log: Path, work: Path) -> list[str]:
    code = r'''
import json, sys, importlib.metadata as md
from packaging.requirements import Requirement
skip = {"torch","torchvision","torchaudio","bitsandbytes","xformers","triton","triton-windows",
        "unsloth","unsloth-zoo","unsloth_zoo"}
out = []
for d in ("unsloth","unsloth_zoo"):
    for r in md.requires(d) or []:
        q = Requirement(r)
        if q.name.lower().replace("_","-") in {s.replace("_","-") for s in skip}: continue
        if q.marker is not None and not q.marker.evaluate({"extra": ""}): continue
        q.marker = None
        out.append(str(q))
json.dump(sorted(set(out)), open(sys.argv[1], "w", encoding="utf-8"))
'''
    got = py_json(py, code, [], env, log, work / "reqs2.json")
    return got if isinstance(got, list) else []


def leg2(work: Path, log: Path) -> dict:
    v = PAIRS["head"]
    env = dict(os.environ)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    res: dict = {"pair": v, "spoof_marker": env.get("AMD_CI_SPOOFED_VENDOR")}
    vdir = work / "v2"
    py, r = make_venv(vdir, env, log)
    res["venv"] = r
    if r["rc"] != 0:
        return res
    idx = ["--index-url", ROCM_INDEX, "--extra-index-url", PYPI]
    res["torch_install"] = run([py, "-m", "pip", "install", ROCM_TORCH, ROCM_VISION, *idx], env, log, 3600)
    res["pair_install"] = run([py, "-m", "pip", "install", f"unsloth=={v}", f"unsloth-zoo=={v}",
                               "torch==2.11.0+rocm7.14.0", "torchvision==0.26.0+rocm7.14.0", *idx],
                              env, log, 3600)
    res["pair_install_mode"] = "with_deps"
    if res["pair_install"]["rc"] != 0:
        # Keep the ROCm torch: install the pair without deps, then its runtime deps minus torch-family.
        res["pair_install_nodeps"] = run([py, "-m", "pip", "install", "--no-deps", f"unsloth=={v}",
                                          f"unsloth-zoo=={v}", "packaging"], env, log, 1200)
        reqs = _requires_without_torch(py, env, log, work)
        res["fallback_reqs"] = reqs
        res["fallback_install"] = run([py, "-m", "pip", "install", *reqs, "--extra-index-url", PYPI],
                                      env, log, 3600) if reqs else {"rc": "no_reqs"}
        res["pair_install_mode"] = "no_deps_fallback"
    fr = run([py, "-m", "pip", "freeze"], env, log, 300)
    res["freeze"] = fr["tail"] if fr["rc"] == 0 else None
    res["inspect"] = py_json(py, INSPECT, [], env, log, work / "inspect2.json")
    res["torch_import"] = py_json(py, IMPORT_ONE, ["torch"], env, log, work / "imp2_torch.json")
    res["bnb_import"] = py_json(py, IMPORT_ONE, ["bitsandbytes"], env, log, work / "imp2_bnb.json")
    train = HERE / "train_tiny_gptoss.py"
    arms = {}
    for arm, extra in (("default", {}), ("compile_disabled", {"UNSLOTH_COMPILE_DISABLE": "1"})):
        aenv = dict(env)
        aenv.update(extra)
        aenv["ARM"] = arm
        aenv["UNSLOTH_ENABLE_LOGGING"] = "1"
        aenv["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"
        aenv["HF_HOME"] = str(work / "hf")
        aenv["UNSLOTH_COMPILE_LOCATION"] = str(work / f"ucc_{arm}")
        out = work / f"train_{arm}.json"
        rr = run([py, str(train), str(out), str(work / f"trainer_{arm}")], aenv, log, 2400)
        obs = {}
        if out.is_file():
            try:
                obs = json.loads(out.read_text(encoding = "utf-8"))
            except Exception as e:  # noqa: BLE001
                obs = {"_parse_error": str(e)}
        obs["_run"] = {"rc": rr["rc"], "seconds": rr["seconds"], "tail": rr["tail"]}
        arms[arm] = obs
    res["arms"] = arms
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", default = "")
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    root = Path(os.environ.get("AMD_CI_WORK") or os.environ.get("RUNNER_TEMP") or ".")
    work = root / f"i12044_{args.state}"
    work.mkdir(parents = True, exist_ok = True)
    log = args.out.parent / f"probe_detail_{args.state}.log"
    obs: dict = {
        "state": args.state, "checkout": args.checkout, "pair": PAIRS.get(args.state),
        "python": sys.version, "platform": sys.platform, "base_python": base_python(),
        "long_paths_enabled": long_paths_enabled(), "work": str(work),
    }
    if args.state not in PAIRS:
        obs["error"] = f"no release pair for state {args.state!r}"
    else:
        try:
            obs["leg1"] = leg1(args.state, work, log)
        except Exception as e:  # noqa: BLE001
            obs["leg1"] = {"probe_error": f"{type(e).__name__}: {e}"}
        if args.state == "head":
            try:
                obs["leg2"] = leg2(work, log)
            except Exception as e:  # noqa: BLE001
                obs["leg2"] = {"probe_error": f"{type(e).__name__}: {e}"}
        else:
            obs["leg2"] = {"not_run": "leg 2 targets the head release only"}
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

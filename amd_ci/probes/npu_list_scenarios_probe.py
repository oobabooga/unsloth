#!/usr/bin/env python3
"""unsloth#18763 on a real XDNA2 NPU: what FastFlowLM 1.0.13 prints around `flm list --json` under
the reporter's conditions, and whether this checkout's Flm.catalog() lists through it.

Scenarios: models pulled by 1.0.13; models pulled by 1.0.12 then listed by 1.0.13 (the update the
reporter describes); an 8 MiB memlock limit (Fedora's default); a second flm holding the NPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

SHA_1013 = "7c2ee388f23503f323d6c21beca8401e7ef675ce700a92f2c786ca6581322a72"
MEMLOCK_8M = 8 * 1024 * 1024


def work() -> Path:
    root = Path(os.environ.get("AMD_CI_WORK") or tempfile.gettempdir()) / "npu_scen"
    root.mkdir(parents = True, exist_ok = True)
    return root


def install(version: str) -> Path:
    """flm under <work>/flm_<v>/bin/flm/npu, the layout Flm(cache_dir) expects."""
    root = work() / f"flm_{version}"
    target = root / "bin" / "flm" / "npu"
    if (target / "flm").is_file():
        return root
    name = f"fastflowlm_{version}_linux.tar.gz"
    archive = work() / name
    if not archive.is_file():
        urllib.request.urlretrieve(f"https://github.com/ROCm/FastFlowLM/releases/download/v{version}/{name}", archive)
    if version == "1.0.13" and hashlib.sha256(archive.read_bytes()).hexdigest() != SHA_1013:
        raise SystemExit("1.0.13 archive does not match Studio's pin")
    scratch = work() / f"x_{version}"
    shutil.rmtree(scratch, ignore_errors = True)
    with tarfile.open(archive) as t:
        try:
            t.extractall(scratch, filter = "fully_trusted")
        except TypeError:
            t.extractall(scratch)
    found = sorted(scratch.rglob("flm-real"), key = lambda p: len(p.parts))[0].parent
    target.parent.mkdir(parents = True, exist_ok = True)
    shutil.move(str(found), str(target))
    return root


def env_for(models: Path) -> dict:
    return dict(os.environ, FLM_MODEL_PATH = str(models), FLM_DISABLE_UPDATE_CHECK = "1")


def sh(cmd: list[str], env: dict, memlock: int | None = None, timeout: int = 900) -> dict:
    def limit():
        if memlock is not None:
            import resource

            _soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (memlock, hard))

    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                           env = env, timeout = timeout, preexec_fn = limit)
        return {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "s": round(time.time() - t0, 1)}
    except Exception as e:
        return {"rc": None, "stdout": "", "stderr": f"{type(e).__name__}: {e}", "s": round(time.time() - t0, 1)}


def split(stdout: str) -> dict:
    start = stdout.find("{")
    try:
        value, end = json.JSONDecoder().raw_decode(stdout, start)
        return {"before": stdout[:start][-500:], "after": stdout[end:][:1500], "chars": end - start,
                "installed": sum(1 for m in value.get("models") or [] if m.get("installed"))}
    except Exception as e:
        return {"parse_error": str(e), "head": stdout[:500]}


CATALOG = r"""
import json, sys, types, importlib.util, resource
from pathlib import Path
checkout, root, memlock = sys.argv[1], sys.argv[2], int(sys.argv[3])
if memlock:
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (memlock, resource.getrlimit(resource.RLIMIT_MEMLOCK)[1]))
if importlib.util.find_spec("httpx") is None:
    sys.modules["httpx"] = types.ModuleType("httpx")
spec = importlib.util.spec_from_file_location("fh", Path(checkout) / "studio/backend/core/inference/flm_host.py")
fh = importlib.util.module_from_spec(spec); spec.loader.exec_module(fh)
try:
    rows = fh.Flm(Path(root)).catalog()
    print("CATALOG " + json.dumps({"ok": True, "n": len(rows), "installed": sum(1 for r in rows if r.get("downloaded") or r.get("installed"))}))
except BaseException as e:
    print("CATALOG " + json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"[:1500]}))
"""


def catalog(checkout: Path, root: Path, models: Path, memlock: int | None) -> dict:
    r = sh([sys.executable, "-c", CATALOG, str(checkout), str(root), str(memlock or 0)], env_for(models))
    line = next((l for l in r["stdout"].splitlines() if l.startswith("CATALOG ")), None)
    return json.loads(line[8:]) if line else {"ok": False, "error": "no result: " + (r["stderr"] or r["stdout"])[-800:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    obs: dict = {"state": args.state,
                 "host": {"dev_accel": sorted(os.listdir("/dev/accel")) if os.path.isdir("/dev/accel") else [],
                          "amdxdna": os.path.isdir("/sys/module/amdxdna")}}
    try:
        new = install("1.0.13")
        old = install("1.0.12")
        flm_new = str(new / "bin/flm/npu/flm")
        flm_old = str(old / "bin/flm/npu/flm")
        m_new, m_old = work() / "models_1013", work() / "models_1012"
        pulls = obs.setdefault("pulls", {})
        # Pulls are shared by both states; the first state does them.
        for flm, models, tag in ((flm_new, m_new, "embed-gemma:300m"), (flm_new, m_new, "qwen3:0.6b"),
                                 (flm_old, m_old, "embed-gemma:300m"), (flm_old, m_old, "qwen3:0.6b")):
            key = f"{Path(flm).parents[3].name}:{tag}"
            marker = models / f".pulled_{tag.replace(':', '_')}"
            if not marker.is_file():
                r = sh([flm, "pull", tag], env_for(models))
                pulls[key] = {"rc": r["rc"], "s": r["s"], "tail": (r["stdout"] + r["stderr"])[-400:]}
                if r["rc"] == 0:
                    models.mkdir(parents = True, exist_ok = True)
                    marker.write_text("ok", encoding = "utf-8")
        scen = {}
        for name, models, memlock in (("empty", work() / "models_empty", None), ("pulled_by_1013", m_new, None),
                                      ("pulled_by_1012", m_old, None), ("pulled_by_1013_memlock8m", m_new, MEMLOCK_8M)):
            r = sh([flm_new, "list", "--json"], env_for(models), memlock = memlock)
            v = sh([flm_new, "validate", "--json"], env_for(models), memlock = memlock)
            scen[name] = {"list": {"rc": r["rc"], "stderr": r["stderr"][:2000], **split(r["stdout"])},
                          "validate": {"rc": v["rc"], "stdout": v["stdout"][:1500], "stderr": v["stderr"][:800]},
                          "catalog": catalog(args.checkout.resolve(), new, models, memlock)}
        # A second flm holding the NPU while Studio lists.
        server = subprocess.Popen([flm_new, "serve", "qwen3:0.6b", "--port", "52625"], env = env_for(m_new),
                                  stdout = open(work() / f"serve_{args.state}.log", "w"), stderr = subprocess.STDOUT)
        try:
            time.sleep(45)
            r = sh([flm_new, "list", "--json"], env_for(m_new))
            scen["npu_busy"] = {"serve_alive": server.poll() is None,
                                "list": {"rc": r["rc"], "stderr": r["stderr"][:2000], **split(r["stdout"])},
                                "catalog": catalog(args.checkout.resolve(), new, m_new, None),
                                "serve_log": (work() / f"serve_{args.state}.log").read_text("utf-8", "replace")[-1200:]}
        finally:
            server.terminate()
            try:
                server.wait(30)
            except subprocess.TimeoutExpired:
                server.kill()
        obs["scenarios"] = scen
    except BaseException as e:
        import traceback

        obs["fatal"] = f"{type(e).__name__}: {e}"
        obs["tb"] = traceback.format_exc()[-2500:]
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

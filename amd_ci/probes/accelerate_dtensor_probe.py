#!/usr/bin/env python3
"""Probe: which accelerate does this checkout select, and can it start a trainer?

accelerate 1.15.0 made `Accelerator.prepare_model` evaluate `model_has_dtensor(model)`
unconditionally while computing `device_placement` (accelerator.py:1802). That imports
`torch.distributed.tensor` and so `torch._C._distributed_c10d`, which AMD's Windows ROCm
wheels do not ship, so training dies at trainer start (huggingface/accelerate#4249).

Per state this does three things and judges none of them:

1. Describes the torch actually on this host: version, whether `torch._C._distributed_c10d`
   imports, whether `torch.distributed.is_available()`. Without a build that lacks c10d
   there is no defect to show, and that is the criteria module's problem, not this one's.
2. Resolves what accelerate version THIS checkout's constraints.txt selects, by handing the
   file to pip as `-c` and reading the resolution back. The repo diff is the independent
   variable, so the resolution has to be done from the checkout rather than assumed.
3. Installs that version into a private --target and runs
   `Accelerator().prepare_model(nn.Linear(4, 4))` in a subprocess against it.

The prepare_model run is a subprocess on purpose: a failed `torch.distributed` import
leaves partially-initialised modules in sys.modules, so doing it in-process would make the
second state's reading depend on the first's.

Pairs with criteria/accelerate_starts_a_trainer.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CONSTRAINTS = Path("studio") / "backend" / "requirements" / "single-env" / "constraints.txt"

# Run in the child, against whatever accelerate PYTHONPATH puts first.
_PREPARE = r"""
import json, sys
out = {}
try:
    import accelerate, torch, torch.nn as nn
    out["accelerate"] = accelerate.__version__
    out["torch"] = torch.__version__
    accelerate.Accelerator(cpu = True).prepare_model(nn.Linear(4, 4))
    out["prepare_ok"] = True
except Exception as e:
    out["prepare_ok"] = False
    out["prepare_error"] = "%s: %s" % (type(e).__name__, e)
print(json.dumps(out))
"""


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output = True, text = True, timeout = 1800, **kw)


def describe_torch() -> dict:
    """What torch is on this host, and does it carry a distributed backend?"""
    code = (
        "import json;out={}\n"
        "try:\n"
        " import torch;out['torch']=torch.__version__\n"
        " out['hip']=getattr(getattr(torch,'version',None),'hip',None)\n"
        " out['cuda_available']=torch.cuda.is_available()\n"
        "except Exception as e:\n"
        " out['torch_error']='%s: %s'%(type(e).__name__,e)\n"
        "try:\n"
        " import torch._C._distributed_c10d;out['has_c10d']=True\n"
        "except Exception as e:\n"
        " out['has_c10d']=False;out['c10d_error']='%s: %s'%(type(e).__name__,e)\n"
        "try:\n"
        " import torch.distributed as d;out['dist_available']=bool(d.is_available())\n"
        "except Exception as e:\n"
        " out['dist_available']=False;out['dist_error']='%s: %s'%(type(e).__name__,e)\n"
        "print(json.dumps(out))\n"
    )
    r = _run([sys.executable, "-c", code])
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"torch_probe_error": (r.stderr or r.stdout or "no output")[-400:]}


def resolve_accelerate(constraints: Path) -> tuple[str | None, str | None]:
    """What version does pip pick for `accelerate` under this checkout's constraints?"""
    r = _run([sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
              "-q", "--report", "-", "-c", str(constraints), "accelerate"])
    if r.returncode != 0:
        return None, f"pip exit {r.returncode}: {(r.stderr or r.stdout)[-400:]}"
    try:
        report = json.loads(r.stdout)
    except Exception as e:  # noqa: BLE001
        return None, f"unparseable report: {type(e).__name__}: {e}"
    for item in report.get("install", []):
        md = item.get("metadata") or {}
        if (md.get("name") or "").lower() == "accelerate":
            return md.get("version"), None
    return None, "accelerate absent from the resolution"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "platform": sys.platform}
    obs.update(describe_torch())

    cfile = args.checkout / CONSTRAINTS
    obs["constraints_present"] = cfile.is_file()
    if cfile.is_file():
        # Record the accelerate line verbatim so the verdict shows what differed.
        obs["constraint_line"] = next(
            (ln.strip() for ln in cfile.read_text(encoding = "utf-8").splitlines()
             if ln.strip().startswith("accelerate")), "")
        version, err = resolve_accelerate(cfile)
        obs["selected_accelerate"] = version
        if err:
            obs["resolve_error"] = err
    else:
        obs["resolve_error"] = f"no constraints.txt at {cfile}"

    version = obs.get("selected_accelerate")
    if version:
        # A private --target per state, so neither reading can see the other's install.
        target = Path(os.environ.get("RUNNER_TEMP", ".")) / f"acc_{args.state}_{version}"
        inst = _run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                     "--target", str(target), f"accelerate=={version}"])
        if inst.returncode != 0:
            obs["install_error"] = f"pip exit {inst.returncode}: {(inst.stderr or inst.stdout)[-400:]}"
        else:
            env = dict(os.environ, PYTHONPATH = str(target))
            r = _run([sys.executable, "-c", _PREPARE], env = env)
            try:
                obs.update(json.loads(r.stdout.strip().splitlines()[-1]))
            except Exception:  # noqa: BLE001
                obs["prepare_ok"] = False
                obs["prepare_error"] = f"probe child produced no JSON: {(r.stderr or r.stdout)[-400:]}"

    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

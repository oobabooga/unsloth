#!/usr/bin/env python3
"""Probe (PR 18355): on the real card, which precision route does each DiT family get from this checkout.

Observes only, per env case (unset, UNSLOTH_STUDIO_DIT_FP16=0/force, UNSLOTH_STUDIO_ROCM_BF16=0), one fresh
subprocess each (the predicates read env and torch at call time, but a fresh process rules out caching).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

FAMILIES = ["flux.2-klein", "z-image", "wan2.2-ti2v-5b", "flux.1", "qwen-image"]
ENVS = {"unset": {}, "dit_fp16_0": {"UNSLOTH_STUDIO_DIT_FP16": "0"},
        "dit_fp16_force": {"UNSLOTH_STUDIO_DIT_FP16": "force"}, "rocm_bf16_0": {"UNSLOTH_STUDIO_ROCM_BF16": "0"}}
CHILD = r'''
import json, sys, types
sys.path.insert(0, sys.argv[1])
import torch
import core.training.diffusion_train_common as C
out = {"file": C.__file__, "torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
       "cuda_available": torch.cuda.is_available(), "has_fallback_fn": hasattr(C, "dit_fp16_fallback"),
       "arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if torch.cuda.is_available() else None,
       "device": C.resolve_train_device(), "flow_bf16_trainable": C.flow_bf16_trainable(), "fam": {}}
for f in json.loads(sys.argv[2]):
    cfg = types.SimpleNamespace(resolved_family=f, mixed_precision="bf16")
    out["fam"][f] = {
        "fallback": getattr(C, "dit_fp16_fallback", lambda _f: False)(f),
        "bf16_unsupported_reason": C.bf16_unsupported_reason(f),
        "effective_mixed_precision": C.effective_mixed_precision(cfg),
        "preflight": {m: C.training_precision_preflight_error(f, m) for m in ("nf4", "auto", "bf16", "int8", "fp8")},
    }
modes, rec = C.train_precision_modes()
out["train_precision_modes"] = [modes, rec]
print("PROBE_JSON " + json.dumps(out, default=str))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    a = ap.parse_args()
    backend = str(a.checkout / "studio" / "backend")
    obs: dict = {"state": a.state, "cases": {}}
    for name, extra in ENVS.items():
        env = {k: v for k, v in os.environ.items() if k not in ("UNSLOTH_STUDIO_DIT_FP16", "UNSLOTH_STUDIO_ROCM_BF16")}
        env.update(extra)
        p = subprocess.run([sys.executable, "-c", CHILD, backend, json.dumps(FAMILIES)], env = env, cwd = backend,
                           capture_output = True, text = True, timeout = 600)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("PROBE_JSON ")), None)
        obs["cases"][name] = json.loads(line[len("PROBE_JSON "):]) if line else {"error": (p.stderr or p.stdout)[-2000:], "rc": p.returncode}
    a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

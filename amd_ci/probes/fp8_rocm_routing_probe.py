#!/usr/bin/env python3
"""Probe: which branch does the fp8 block dequant helper pick, on real ROCm hardware.

Observes only. For each state's checkout it records, at the card's real compute
capability and at several spoofed ones, whether _blockwise_weight_dequant_any_shape
routed to the triton kernel or to the torch expansion, plus a checksum of the values.

It also records what the majority-only predicate `capability[0] < 9` WOULD have decided
at each capability. That is an observation, not a judgement: the criteria module decides
whether any of it matters.

The triton entry point is stubbed while routing is observed, so this works on cards that
cannot compile fp8e4nv. Values are measured separately, without the stub.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

INNER = r'''
import json, os, sys
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
sys.path.insert(0, sys.argv[1])
out = {}
try:
    import torch
    out["torch"] = torch.__version__
    out["hip"] = torch.version.hip
    out["cuda_available"] = bool(torch.cuda.is_available())
    if out["cuda_available"]:
        out["gpu"] = torch.cuda.get_device_name(0)
        out["real_capability"] = list(torch.cuda.get_device_capability())
    import unsloth  # noqa: F401
    from unsloth.kernels import fp8
    out["fp8_file"] = fp8.__file__
    out["has_chunked_fallback"] = hasattr(fp8, "_torch_blockwise_dequant")
except Exception as e:
    out["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    print("PROBE_JSON " + json.dumps(out)); raise SystemExit(0)

real_cap = tuple(out["real_capability"])
# Deduped: on gfx1151 the real capability IS (11, 5), so a naive list would collapse from
# six entries to five keys and any "observed N capabilities" gate counting the list would
# fail on the very host it was written for.
CAPS = []
for c in [real_cap, (8, 0), (8, 6), (8, 9), (9, 0), (11, 5)]:
    if c not in CAPS:
        CAPS.append(c)
out["caps_requested"] = ["%d.%d" % c for c in CAPS]
routes, checksums, errors = {}, {}, {}
_orig = fp8.weight_dequant_block
_real_getcap = torch.cuda.get_device_capability

for cap in CAPS:
    key = "%d.%d" % cap
    calls = []
    def _stub(x, s, block_size = 128, dtype = torch.bfloat16, _c = calls):
        _c.append(1)
        return torch.empty(x.shape, dtype = dtype, device = x.device)
    try:
        torch.manual_seed(0)
        w = (torch.randn(256, 256, device = "cuda") * 0.4).to(torch.float8_e4m3fn)
        s = torch.rand(2, 2, device = "cuda", dtype = torch.float32) + 0.5
        # routing, with the kernel stubbed so pre-sm89 cards can be asked too
        torch.cuda.get_device_capability = lambda *a, **k: cap
        fp8.weight_dequant_block = _stub
        fp8._blockwise_weight_dequant_any_shape(w, s, [128, 128], torch.bfloat16)
        routes[key] = "triton" if calls else "torch"
        # values, for real this time
        fp8.weight_dequant_block = _orig
        y = fp8._blockwise_weight_dequant_any_shape(w, s, [128, 128], torch.bfloat16)
        torch.cuda.synchronize()
        checksums[key] = round(float(y.float().sum()), 4)
    except Exception as e:
        errors[key] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    finally:
        fp8.weight_dequant_block = _orig
        torch.cuda.get_device_capability = _real_getcap

out["routes"] = routes
out["checksums"] = checksums
out["errors"] = errors
# what the major-only predicate would have said here, for the record
out["major_only_would_divert"] = {
    "%d.%d" % c: bool(torch.version.hip is None and c[0] < 9) for c in CAPS
}
out["full_tuple_would_divert"] = {
    "%d.%d" % c: bool(torch.version.hip is None and c < (8, 9)) for c in CAPS
}
print("PROBE_JSON " + json.dumps(out))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 900)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "checkout": args.checkout}
    runner = Path(args.checkout) / "_fp8_rocm_inner.py"
    try:
        runner.write_text(INNER, encoding = "utf-8")
        # stdout carries import banners, so the payload is a tagged line, never raw stdout
        proc = subprocess.run([args.python, str(runner), args.checkout],
                              capture_output = True, text = True, timeout = args.timeout)
        obs["rc"] = proc.returncode
        line = [ln for ln in (proc.stdout + proc.stderr).splitlines()
                if ln.startswith("PROBE_JSON ")]
        if line:
            obs.update(json.loads(line[-1][len("PROBE_JSON "):]))
        else:
            obs["error"] = "probe produced no PROBE_JSON line"
            obs["stdout_tail"] = proc.stdout[-1500:]
            obs["stderr_tail"] = proc.stderr[-1500:]
    except subprocess.TimeoutExpired:
        obs["error"] = f"probe timed out after {args.timeout}s"
    except Exception as e:  # noqa: BLE001
        obs["error"] = f"{type(e).__name__}: {e}"
    finally:
        runner.unlink(missing_ok = True)

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

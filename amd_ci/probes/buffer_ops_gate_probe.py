#!/usr/bin/env python3
"""Probe: what `import unsloth` from this checkout does to Triton's buffer ops on this GPU.

Observes only. In a fresh subprocess with the checkout first on sys.path and a private
TMPDIR / TRITON_HOME, it records the env after import, Triton's effective
amd.use_buffer_ops, the visible archs, whether a ten-line Triton kernel writes its output,
and which Inductor cache directory a torch.compile call fills.
"""

from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

CHILD = r'''
import json, os, sys
out = {}
try:
    import unsloth
    out["unsloth_file"] = unsloth.__file__
except Exception as e:
    out["import_error"] = f"{type(e).__name__}: {e}"
    print("@@PROBE@@" + json.dumps(out)); sys.exit(0)
out["env"] = {k: os.environ.get(k) for k in ("AMDGCN_USE_BUFFER_OPS", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")}
import torch
out["torch"] = torch.__version__
out["hip"] = getattr(torch.version, "hip", None)
try:
    from unsloth.device_type import hip_visible_archs
    out["archs"] = hip_visible_archs()
except Exception as e:
    out["archs"] = [torch.cuda.get_device_properties(i).gcnArchName for i in range(torch.cuda.device_count())]
try:
    import triton, triton.language as tl
    out["triton"] = triton.__version__
    try:
        import triton.knobs as k
        out["use_buffer_ops"] = bool(k.amd.use_buffer_ops)
    except Exception as e:
        out["use_buffer_ops"] = f"unreadable: {e}"
    @triton.jit
    def k2(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = offs < n
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask = m) * 2, mask = m)
    x = torch.arange(1024, device = "cuda", dtype = torch.float32); y = torch.full_like(x, -1)
    k2[(4,)](x, y, 1024, BLOCK = 256); torch.cuda.synchronize()
    out["kernel_correct"] = int((y == x * 2).sum().item())
except Exception as e:
    out["kernel_error"] = f"{type(e).__name__}: {e}"
try:
    f = torch.compile(lambda t: torch.sin(t) * 2 + 1)
    t = torch.randn(64, device = "cuda")
    out["compile_ok"] = bool(torch.allclose(f(t), torch.sin(t) * 2 + 1, atol = 1e-5))
except Exception as e:
    out["compile_error"] = f"{type(e).__name__}: {e}"[:300]
tmp = os.environ["TMPDIR"]
out["inductor_dirs"] = {d: sum(len(fs) for _, _, fs in os.walk(os.path.join(tmp, d)))
                        for d in sorted(os.listdir(tmp)) if d.startswith("torchinductor")}
print("@@PROBE@@" + json.dumps(out))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    args, _ = ap.parse_known_args()
    obs: dict = {"state": args.state}
    work = Path(tempfile.mkdtemp(prefix = f"bufops_{args.state}_", dir = os.environ.get("RUNNER_TEMP")))
    (work / "tmp").mkdir(); (work / "th").mkdir()
    env = dict(os.environ)
    for k in ("AMDGCN_USE_BUFFER_OPS", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        env.pop(k, None)
    env.update(TMPDIR = str(work / "tmp"), TEMP = str(work / "tmp"), TMP = str(work / "tmp"),
               TRITON_HOME = str(work / "th"),
               PYTHONPATH = args.checkout + os.pathsep + env.get("PYTHONPATH", ""))
    # @triton.jit reads its source with inspect, so the child has to live in a real file.
    child = work / "child.py"
    child.write_text(CHILD, encoding = "utf-8")
    p = subprocess.run([args.python, str(child)], env = env, cwd = str(work),
                       capture_output = True, text = True, timeout = 1800)
    obs["rc"] = p.returncode
    line = [l for l in p.stdout.splitlines() if l.startswith("@@PROBE@@")]
    if line:
        obs.update(json.loads(line[-1][len("@@PROBE@@"):]))
    else:
        obs["error"] = "child printed no result"
        obs["stderr_tail"] = p.stderr[-2000:]
    obs["checkout_imported"] = str(obs.get("unsloth_file", "")).startswith(str(Path(args.checkout).resolve())) \
        or str(obs.get("unsloth_file", "")).startswith(args.checkout)
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

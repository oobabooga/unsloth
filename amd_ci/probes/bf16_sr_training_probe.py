import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def observe(args):
    try:
        import torch
    except ImportError:
        if not args.bootstrap_torch:
            raise
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
        importlib.invalidate_caches()
        import torch
    device = args.device
    n = 65536
    source = Path(args.checkout) / "unsloth/optimizers/bf16_adamw.py"
    p = torch.nn.Parameter(torch.ones(n, dtype=torch.bfloat16, device=device))
    p.grad = torch.full_like(p, .1)
    reference = torch.nn.Parameter(p.detach().float())
    reference.grad = p.grad.float()
    control = torch.optim.AdamW([reference], lr=1e-5, weight_decay=0)
    if source.is_file():
        spec = importlib.util.spec_from_file_location("bf16_sr_probe_runtime", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        optimizer = module.AdamWBF16([p], lr=1e-5, weight_decay=0)
    else:
        optimizer = torch.optim.AdamW([p], lr=1e-5, weight_decay=0)
    optimizer.step()
    control.step()
    observation = {"device": str(p.device), "torch": torch.__version__, "hip": torch.version.hip,
                   "elements": n, "api": source.is_file(), "changed": int((p != 1).sum()),
                   "reference_changed": int((reference != 1).sum()), "finite": bool(torch.isfinite(p).all()),
                   "variance_dtype": str(optimizer.state[p]["exp_avg_sq"].dtype)}
    if source.is_file():
        marker = "gpu" if device == "cuda" else "not gpu"
        test = subprocess.run([sys.executable, "-m", "pytest", "tests/test_bf16_adamw.py", "-q", "-o", "addopts=", "-m", marker],
                              cwd=args.checkout, capture_output=True, text=True, timeout=600)
        observation.update(tests_rc=test.returncode, tests_tail=(test.stdout + test.stderr)[-4000:])
    return observation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--bootstrap-torch", action="store_true")
    args = parser.parse_args()
    try:
        output = observe(args)
    except Exception as error:
        output = {"error": repr(error)}
    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")

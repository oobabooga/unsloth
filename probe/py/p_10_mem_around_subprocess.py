import subprocess
import sys
import torch

print("before =", torch.cuda.mem_get_info(0), flush=True)
r = subprocess.run(
    [sys.executable, "-c", "import torch; print(torch.cuda.mem_get_info(0))"],
    capture_output=True, text=True,
)
print("child rc =", r.returncode, "out =", r.stdout.strip(), "err =", r.stderr.strip()[-400:], flush=True)
print("after =", torch.cuda.mem_get_info(0), flush=True)

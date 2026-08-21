import os
import sys
import torch

print("parent context:", torch.cuda.mem_get_info(0), flush=True)
pid = os.fork()
if pid == 0:
    try:
        print("child mem_get_info:", torch.cuda.mem_get_info(0), flush=True)
    except BaseException as e:  # noqa: BLE001
        print("child raised:", repr(e), flush=True)
    os._exit(0)
_, status = os.waitpid(pid, 0)
print("fork child status =", status, "signal =", status & 0x7F, "exit =", status >> 8)
sys.stdout.flush()

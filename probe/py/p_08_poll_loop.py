import time
import torch

for i in range(120):
    free, total = torch.cuda.mem_get_info(0)
    if i % 30 == 0:
        print(i, free, total, flush=True)
    time.sleep(0.02)
print("poll loop finished cleanly")

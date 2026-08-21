import threading
import torch

errs = []


def work(i):
    try:
        for _ in range(25):
            torch.cuda.mem_get_info(0)
    except BaseException as e:  # noqa: BLE001
        errs.append((i, repr(e)))


ts = [threading.Thread(target=work, args=(i,)) for i in range(8)]
for t in ts:
    t.start()
for t in ts:
    t.join()
print("errors:", errs)

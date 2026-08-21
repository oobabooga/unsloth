import threading
import torch

out = {}


def work():
    try:
        out["props"] = str(torch.cuda.get_device_properties(0))
        out["mem"] = torch.cuda.mem_get_info(0)
    except BaseException as e:  # noqa: BLE001
        out["error"] = repr(e)


t = threading.Thread(target=work)
t.start()
t.join()
print(out)

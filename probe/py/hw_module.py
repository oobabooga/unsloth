import logging
import os
import sys

sys.path.insert(0, os.getcwd())
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format="%(levelname)s %(name)s: %(message)s")


def step(label, fn):
    print("=== " + label, flush=True)
    try:
        print("   ->", fn(), flush=True)
    except BaseException as e:  # noqa: BLE001
        print("   RAISED", repr(e), flush=True)


from utils.hardware import hardware as H  # noqa: E402

step("detect_hardware", H.detect_hardware)
print("IS_ROCM =", H.IS_ROCM, flush=True)
step("_torch_get_device_inventory([0])", lambda: H._torch_get_device_inventory([0]))
step("_torch_get_per_device_info([0])", lambda: H._torch_get_per_device_info([0]))
step("get_visible_gpu_utilization", H.get_visible_gpu_utilization)
step("get_gpu_memory_info", H.get_gpu_memory_info)
step("trusted_mem_get_info", H.trusted_mem_get_info)
try:
    import torch
    props = torch.cuda.get_device_properties(0)
    step("_rocm_props_total_is_carve_out", lambda: H._rocm_props_total_is_carve_out(props))
except BaseException as e:  # noqa: BLE001
    print("props unavailable:", repr(e))
print("hw_module.py finished cleanly", flush=True)

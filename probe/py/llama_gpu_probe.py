"""Exercise the exact GPU-memory probe the model dropdown / load path uses."""
import logging
import os
import shutil
import sys

sys.path.insert(0, os.getcwd())
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format="%(levelname)s %(name)s: %(message)s")

print("amd-smi on PATH:", shutil.which("amd-smi"), flush=True)


def step(label, fn):
    print("=== " + label, flush=True)
    try:
        print("   ->", fn(), flush=True)
    except BaseException as e:  # noqa: BLE001
        print("   RAISED", repr(e), flush=True)


from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

step("_find_llama_server_binary", LlamaCppBackend._find_llama_server_binary)
step("_is_vulkan_backend", LlamaCppBackend._is_vulkan_backend)
step("_rocm_hip_is_reachable", LlamaCppBackend._rocm_hip_is_reachable)
step("_rocm_hip_device_count", LlamaCppBackend._rocm_hip_device_count)
step("_rocm_unified_memory_gpu_ids", LlamaCppBackend._rocm_unified_memory_gpu_ids)
step("_rocm_total_memory_mib_by_physical_id", LlamaCppBackend._rocm_total_memory_mib_by_physical_id)
step("_get_gpu_memory_amd_smi", lambda: LlamaCppBackend._get_gpu_memory_amd_smi(None))
step("_get_gpu_memory", LlamaCppBackend._get_gpu_memory)
step("_get_gpu_memory(for_llama_server=True)", lambda: LlamaCppBackend._get_gpu_memory(for_llama_server=True))
step("_available_system_memory_mib", LlamaCppBackend._available_system_memory_mib)

print("=== repeat _get_gpu_memory 5x (poll shape)", flush=True)
for i in range(5):
    print("   ", i, LlamaCppBackend._get_gpu_memory(), flush=True)

print("llama_gpu_probe.py finished cleanly", flush=True)

import glob, importlib, os, sys, traceback
backend = sys.argv[1]; label = sys.argv[2]
sys.path.insert(0, backend)
import torch
MiB = 1 << 20
def sysfs():
    out = {}
    for d in glob.glob("/sys/class/drm/card*/device"):
        if os.path.basename(os.path.realpath(os.path.join(d, "driver"))) != "amdgpu": continue
        for f in ("mem_info_vram_total", "mem_info_vram_used", "mem_info_gtt_total", "mem_info_gtt_used"):
            try: out[f] = int(open(os.path.join(d, f)).read()) // MiB
            except OSError: pass
    return out
free, total = torch.cuda.mem_get_info(0)
p = torch.cuda.get_device_properties(0)
print(f"[{label}] torch.mem_get_info free={free//MiB} MiB total={total//MiB} MiB props.total={p.total_memory//MiB} MiB pci={getattr(p,'pci_domain_id',None)}:{getattr(p,'pci_bus_id',None)}:{getattr(p,'pci_device_id',None)}")
print(f"[{label}] sysfs {sysfs()}")
try:
    hw = importlib.import_module("utils.hardware.hardware")
except Exception:
    traceback.print_exc(); sys.exit(0)
try:
    k = hw.rocm_linux_kernel_free_mib(0)
except TypeError:
    k = hw.rocm_linux_kernel_free_mib(0)
print(f"[{label}] rocm_linux_kernel_free_mib(0) = {k}")
try:
    print(f"[{label}] rocm_linux_kernel_free_mib(0, total_bytes=driver total) = {hw.rocm_linux_kernel_free_mib(0, total_bytes=total)}")
except TypeError:
    print(f"[{label}] (no total_bytes parameter on this revision)")
tf, tt = hw.trusted_mem_get_info(0)
print(f"[{label}] trusted_mem_get_info -> free={tf//MiB} MiB total={tt//MiB} MiB")

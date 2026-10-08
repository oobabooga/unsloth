#!/usr/bin/env python3
"""Apply the rocm-systems#11299 wave count to libhsakmt queues.c (KFD's 32 waves/CU below gfx1250)."""
import sys

path = sys.argv[1]
text = open(path, encoding = "utf-8").read()
old = ("\telse\n"
       "\t\twave_num = cu_num * node->NumSIMDPerCU * node->MaxWavesPerSIMD;\n")
new = ("\telse if (gfxv < GFX_VERSION_GFX1250)\n"
       "\t\twave_num = cu_num * 32;\n"
       "\telse\n"
       "\t\twave_num = cu_num * node->NumSIMDPerCU * node->MaxWavesPerSIMD;\n")
if new in text:
    sys.exit("pinned rocm-systems already carries the fix: delete this step")
if text.count(old) != 1:
    sys.exit("get_num_waves() changed upstream: re-check against kfd_queue.c")
open(path, "w", encoding = "utf-8").write(text.replace(old, new))
print(f"patched {path}")

#!/usr/bin/env python3
"""Fixture: hold amdgpu GTT through RADV so sysfs free GTT sits at a target, for every state's probe alike.

usage: gtt_hold_fixture.py --gtt-free-gib X

Allocates host-visible, non-device-local Vulkan memory (GTT on radv) in 1 GiB chunks and touches it, until the
sysfs reading (sum of mem_info_gtt_total - mem_info_gtt_used over amdgpu cards) reaches the target. Prints one JSON
line with "status": "READY" and the achieved reading, then sleeps until terminated. Exits non-zero without READY when
it cannot reach the target within 0.5 GiB, so a differential never runs against a condition that was not established.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import sys
import time

G = 1024**3


def gtt_reading() -> tuple[int, int]:
    tot = used = 0
    for path in glob.glob("/sys/class/drm/card*/device/mem_info_gtt_total"):
        dev = path.rsplit("/", 1)[0]
        try:
            with open(dev + "/vendor", encoding = "utf-8") as fh:
                if fh.read().strip() != "0x1002":
                    continue
            with open(path, encoding = "utf-8") as fh:
                tot += int(fh.read().strip())
            with open(dev + "/mem_info_gtt_used", encoding = "utf-8") as fh:
                used += int(fh.read().strip())
        except (OSError, ValueError):
            continue
    return tot, used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtt-free-gib", type = float, required = True)
    args = ap.parse_args()
    import vulkan as vk  # noqa: PLC0415

    inst = vk.vkCreateInstance(
        vk.VkInstanceCreateInfo(pApplicationInfo = vk.VkApplicationInfo(apiVersion = vk.VK_MAKE_VERSION(1, 2, 0))),
        None,
    )
    pdev = vk.vkEnumeratePhysicalDevices(inst)[0]
    props = vk.vkGetPhysicalDeviceMemoryProperties(pdev)
    hv = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    dl = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    gtt_types = []
    for i in range(props.memoryTypeCount):
        t = props.memoryTypes[i]
        h = props.memoryHeaps[t.heapIndex]
        print(f"type {i} flags=0x{t.propertyFlags:x} heap {t.heapIndex} size={h.size / G:.2f} GiB", flush = True)
        if (t.propertyFlags & hv) == hv and not (t.propertyFlags & dl):
            gtt_types.append((h.size, i))
    if not gtt_types:
        print("no host-visible non-device-local memory type", flush = True)
        return 2
    mtype = max(gtt_types)[1]
    qci = vk.VkDeviceQueueCreateInfo(queueFamilyIndex = 0, queueCount = 1, pQueuePriorities = [1.0])
    dev = vk.vkCreateDevice(pdev, vk.VkDeviceCreateInfo(queueCreateInfoCount = 1, pQueueCreateInfos = [qci]), None)
    held = []
    tot, used = gtt_reading()
    need = int(tot - used - args.gtt_free_gib * G)
    print(f"gtt total={tot / G:.2f} used={used / G:.2f} GiB; allocating {max(need, 0) / G:.2f} GiB", flush = True)
    while need > 0:
        n = min(G, need)
        mem = vk.vkAllocateMemory(dev, vk.VkMemoryAllocateInfo(allocationSize = n, memoryTypeIndex = mtype), None)
        ptr = vk.vkMapMemory(dev, mem, 0, n, 0)
        ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(ptr)), 1, n)
        held.append(mem)
        need -= n
    time.sleep(2)
    tot, used = gtt_reading()
    free = (tot - used) / G
    ok = abs(free - args.gtt_free_gib) <= 0.5
    print(json.dumps({"status": "READY" if ok else "MISSED", "target_gtt_free_gib": args.gtt_free_gib,
                      "gtt_total_gib": round(tot / G, 2), "gtt_free_gib": round(free, 2),
                      "held_gib": len(held)}), flush = True)
    if not ok:
        return 3
    while True:
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())

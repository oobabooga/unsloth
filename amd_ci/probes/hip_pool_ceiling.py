"""Where does hipMalloc actually stop on an AMD APU?

unsloth#9884 gates GGML_CUDA_ENABLE_UNIFIED_MEMORY on a comparison of two pools:
plain allocation is assumed to draw the VRAM carve-out, managed allocation to
draw host RAM, so the flag is enabled when host RAM is the larger of the two.

That premise is load bearing and untested. lemonade-sdk/lemonade#3377 reports
Strix Halo hosts with carve-outs of 0.5 to 4 GB against 100 to 120 GB of GTT,
where models needing 16 GB resident nonetheless RUN. If hipMalloc reaches GTT,
the carve-out is not the ceiling and the gate compares the wrong number.

This probe measures the ceiling instead of assuming it. It reads what the driver
reports, then binary searches the largest single allocation that actually
succeeds, and prints both so they can be compared directly.

The runner has no compiler, so HIP is driven through ctypes against
libamdhip64.so rather than a C++ test.

Two deliberate limits, stated because they bound what the output can support:

  - hipMalloc failure is clean. The driver returns hipErrorOutOfMemory and
    nothing is disturbed, so searching the ceiling is safe.
  - hipMallocManaged is NOT touched after allocation. Managed memory can be
    reserved lazily without physical backing, so the number here is a
    reservation ceiling, not a residency ceiling. Faulting in tens of GiB to
    find the real one risks an OOM kill on a shared CI host, which is a poor
    trade for a number this probe does not need.
"""

import ctypes
import json
import os
import sys

HIP_SUCCESS = 0
GIB = 1 << 30


def load_hip():
    for name in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def driver_reported():
    """What the kernel driver says, straight from sysfs.

    These are the same files lemonade#3377 cites, so the numbers here are
    directly comparable to the carve-outs in that thread.
    """
    out = {}
    for card in ("card0", "card1"):
        base = f"/sys/class/drm/{card}/device"
        if not os.path.isdir(base):
            continue
        for key in ("mem_info_vram_total", "mem_info_gtt_total",
                    "mem_info_vis_vram_total", "mem_info_vram_used",
                    "mem_info_gtt_used"):
            try:
                with open(f"{base}/{key}") as fh:
                    out[f"{card}.{key}"] = int(fh.read().strip())
            except (OSError, ValueError):
                pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith(("MemTotal:", "MemAvailable:")):
                    k, v = line.split(":")
                    out[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    return out


def try_alloc(hip, nbytes, managed):
    ptr = ctypes.c_void_p()
    if managed:
        # hipMallocManaged(ptr, size, flags); 1 = hipMemAttachGlobal
        rc = hip.hipMallocManaged(ctypes.byref(ptr), ctypes.c_size_t(nbytes),
                                  ctypes.c_uint(1))
    else:
        rc = hip.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
    if rc == HIP_SUCCESS and ptr.value:
        hip.hipFree(ptr)
        return True
    # Clear the sticky error so the next probe is not poisoned by this one.
    hip.hipGetLastError()
    return False


def ceiling(hip, managed):
    """Largest single allocation that succeeds, to 1 GiB resolution.

    Doubling first, then bisecting, so a host with a large pool is not probed
    one gigabyte at a time.
    """
    if not try_alloc(hip, GIB, managed):
        return 0
    lo = 1
    hi = 2
    while try_alloc(hip, hi * GIB, managed):
        lo = hi
        hi *= 2
        if hi > 1024:
            break
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if try_alloc(hip, mid * GIB, managed):
            lo = mid
        else:
            hi = mid
    return lo


def main():
    result = {"driver_reported": driver_reported()}

    hip = load_hip()
    if hip is None:
        result["error"] = "libamdhip64.so not loadable; no measurement taken"
        print(json.dumps(result, indent=2))
        return 1

    free_b = ctypes.c_size_t()
    total_b = ctypes.c_size_t()
    if hip.hipMemGetInfo(ctypes.byref(free_b), ctypes.byref(total_b)) == HIP_SUCCESS:
        result["hipMemGetInfo_free"] = free_b.value
        result["hipMemGetInfo_total"] = total_b.value

    result["hipMalloc_ceiling_gib"] = ceiling(hip, managed=False)
    result["hipMallocManaged_ceiling_gib"] = ceiling(hip, managed=True)
    result["managed_ceiling_caveat"] = (
        "reservation only; the allocation was never touched, so this is not a "
        "residency ceiling"
    )

    d = result["driver_reported"]
    vram = d.get("card0.mem_info_vram_total") or d.get("card1.mem_info_vram_total")
    gtt = d.get("card0.mem_info_gtt_total") or d.get("card1.mem_info_gtt_total")
    plain = result["hipMalloc_ceiling_gib"]

    # The whole point of the probe: does plain allocation stay inside the
    # carve-out, or does it reach past it into GTT?
    if vram and plain:
        vram_gib = vram / GIB
        result["vram_carveout_gib"] = round(vram_gib, 2)
        result["gtt_total_gib"] = round(gtt / GIB, 2) if gtt else None
        if plain > vram_gib * 1.05:
            result["finding"] = (
                f"hipMalloc reached {plain} GiB, PAST the {vram_gib:.1f} GiB "
                f"carve-out. The carve-out is not the ceiling, so gating on it "
                f"compares the wrong number."
            )
        else:
            result["finding"] = (
                f"hipMalloc stopped at {plain} GiB, within the "
                f"{vram_gib:.1f} GiB carve-out. Consistent with the carve-out "
                f"being the ceiling on THIS host."
            )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

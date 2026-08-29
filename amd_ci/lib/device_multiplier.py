#!/usr/bin/env python3
"""Present more HIP devices to torch than the host physically has.

A gfx1151 runner has one GPU, so every multi-GPU code path in the tree was
unreachable here and got reported as a permanent gap. It is not permanent. torch
reaches the device list through the HIP runtime, and HIP is a shared library, so an
LD_PRELOAD shim can inflate the count and fold the extra ordinal onto the real device.
Measured on this runner: `torch.cuda.device_count()` is 2, `cuda:1` accepts real
tensors, and a real matmul runs on it.

WHAT THIS IS FOR. Wiring. Selection logic, arch gates, device-index bookkeeping, and
code that assumes `device_count() == 1`. Those are real bugs and this reaches them.

WHAT IT IS NOT. The phantom device IS the real device wearing another number. Shared
memory, shared compute, serialised execution. So it says nothing about:
  - sharding, which becomes one GPU talking to itself and OOMs at half the expected size
  - throughput or scaling, which are meaningless when both "devices" are one
  - collectives, where two ranks on one device deadlock or silently agree
  - mixed architectures, since both devices report the same arch
Peer access is deliberately reported UNAVAILABLE so torch stages through the host
rather than aliasing device-to-itself behind your back.

<critical>It only fools HIP.</critical> amd-smi, rocm-smi, sysfs, KFD topology and
rocminfo all still see one GPU. Studio's own `utils/hardware/hardware.py` enumerates
through amd-smi, NOT through torch, so this shim does not move it at all and a probe
that needs Studio to see two devices must stub the amd-smi binary as well. Whichever
boundary the code under test actually reads is the one that has to be faked.

<critical>Symbol coverage is exactly what torch drives.</critical> Every HIP call that
takes a device ordinal needs a wrapper; the count is the easy part. A consumer that
calls one this file does not wrap (vLLM, bitsandbytes, RCCL) gets the real function
with an out-of-range ordinal and dies with hipErrorInvalidDevice 101. That is how the
first version of this failed, inside rocBLAS. Add the wrapper, do not widen `demap`.

Linux only: LD_PRELOAD does not exist on Windows or macOS. NVIDIA is untried; the
equivalent would wrap cuDeviceGet / cuDeviceGetAttribute / cuDevicePrimaryCtxRetain.

Honesty coupling: activating this sets AMD_CI_SPOOFED_DEVICES, and `capability.py`
reads that marker and REFUSES to let the spoofed count satisfy `multi_gpu`. Without
that, a spoofed run would drop the multi-GPU line from "Not tested here" and read as
hardware validation. Do not set the marker by hand and do not remove that coupling.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# Set whenever the shim is active. capability.py keys off this; see the module docstring.
ENV_MARKER = "AMD_CI_SPOOFED_DEVICES"

SHIM_C = r'''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

static int   g_extra   = 1;   /* how many phantom devices to add */
static int   g_real    = -1;  /* real count, filled on first query */
static __thread int g_current = 0;  /* the ordinal the app THINKS is current */

/* The real dlsym, resolved by version because dlsym cannot look itself up. Every
   internal lookup in this file MUST go through xdlsym: this shim also exports its own
   dlsym (see below), so a plain dlsym(RTLD_NEXT, ...) here would re-enter the wrapper
   and, for any hooked name, return the wrapper itself -- unbounded recursion. */
static void *(*real_dlsym)(void *, const char *) = NULL;

static void *xdlsym(const char *name) {
    if (!real_dlsym) real_dlsym = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    return real_dlsym ? real_dlsym(RTLD_NEXT, name) : NULL;
}

static int  (*real_GetDeviceCount)(int *) = NULL;
static int  (*real_SetDevice)(int) = NULL;

static void init_once(void) {
    if (!real_GetDeviceCount) {
        real_GetDeviceCount = xdlsym("hipGetDeviceCount");
        real_SetDevice      = xdlsym("hipSetDevice");
        const char *e = getenv("SHIM_EXTRA_DEVICES");
        if (e) g_extra = atoi(e);
    }
    if (g_real < 0 && real_GetDeviceCount) {
        int c = 0;
        if (real_GetDeviceCount(&c) == 0) g_real = c;
    }
}

/* Map a virtual ordinal onto a real one. Everything at or above the real count folds
   onto device 0: the phantom device IS device 0 wearing another number. */
static int demap(int dev) {
    if (g_real > 0 && dev >= g_real) return 0;
    return dev;
}

int hipGetDeviceCount(int *count) {
    init_once();
    if (!real_GetDeviceCount) return 101; /* hipErrorInvalidDevice */
    int rc = real_GetDeviceCount(count);
    if (rc == 0 && count) {
        g_real = *count;
        *count = *count + g_extra;
    }
    return rc;
}

int hipSetDevice(int dev) {
    init_once();
    g_current = dev;
    return real_SetDevice ? real_SetDevice(demap(dev)) : 101;
}

int hipGetDevice(int *dev) {
    init_once();
    if (!dev) return 101;
    /* Report the virtual ordinal back, so the caller's bookkeeping stays consistent
       with what it set. Returning the demapped one makes torch believe its
       set_device silently failed. */
    *dev = g_current;
    return 0;
}

/* --- The downstream per-device APIs. -----------------------------------------------
   The first run of this probe shimmed only the count, and rocBLAS aborted the process
   on hipGetDeviceProperties(&prop, 1) with hipErrorInvalidDevice. That is the exact
   trap the HAMi work documents: an inflated count without remapped ordinals breaks the
   moment anything asks about the phantom device. Each wrapper below is one API that
   takes a device ordinal, demapped onto the real device.

   hipGetDevicePropertiesR0600 is the name actually exported since ROCm 6.0; the
   unversioned spelling is kept for older stacks. Whichever is absent simply never
   binds, so emitting both is safe. */

#define WRAP_DEV_FIRST(name, type1)                                        \
    static int (*real_##name)(type1, int) = NULL;                          \
    int name(type1 a, int dev) {                                           \
        init_once();                                                       \
        if (!real_##name) real_##name = xdlsym(#name);           \
        if (!real_##name) return 101;                                      \
        return real_##name(a, demap(dev));                                 \
    }

WRAP_DEV_FIRST(hipGetDevicePropertiesR0600, void *)
WRAP_DEV_FIRST(hipGetDeviceProperties,      void *)
WRAP_DEV_FIRST(hipDeviceTotalMem,           size_t *)
WRAP_DEV_FIRST(hipDeviceGetDefaultMemPool,  void *)
WRAP_DEV_FIRST(hipDeviceGetMemPool,         void *)
WRAP_DEV_FIRST(hipDeviceGet,                int *)

static int (*real_hipDeviceGetAttribute)(int *, int, int) = NULL;
int hipDeviceGetAttribute(int *v, int attr, int dev) {
    init_once();
    if (!real_hipDeviceGetAttribute)
        real_hipDeviceGetAttribute = xdlsym("hipDeviceGetAttribute");
    if (!real_hipDeviceGetAttribute) return 101;
    return real_hipDeviceGetAttribute(v, attr, demap(dev));
}

static int (*real_hipDeviceGetName)(char *, int, int) = NULL;
int hipDeviceGetName(char *name, int len, int dev) {
    init_once();
    if (!real_hipDeviceGetName)
        real_hipDeviceGetName = xdlsym("hipDeviceGetName");
    if (!real_hipDeviceGetName) return 101;
    return real_hipDeviceGetName(name, len, demap(dev));
}

static int (*real_hipDeviceGetPCIBusId)(char *, int, int) = NULL;
int hipDeviceGetPCIBusId(char *id, int len, int dev) {
    init_once();
    if (!real_hipDeviceGetPCIBusId)
        real_hipDeviceGetPCIBusId = xdlsym("hipDeviceGetPCIBusId");
    if (!real_hipDeviceGetPCIBusId) return 101;
    return real_hipDeviceGetPCIBusId(id, len, demap(dev));
}

static int (*real_hipDeviceGetUuid)(void *, int) = NULL;
int hipDeviceGetUuid(void *uuid, int dev) {
    init_once();
    if (!real_hipDeviceGetUuid)
        real_hipDeviceGetUuid = xdlsym("hipDeviceGetUuid");
    if (!real_hipDeviceGetUuid) return 101;
    return real_hipDeviceGetUuid(uuid, demap(dev));
}

/* Peer access is deliberately reported as UNAVAILABLE between the real and phantom
   device. They are the same physical GPU, so a P2P copy would be device-to-itself;
   saying no makes torch stage through the host, which is correct behaviour rather
   than a silent aliasing bug. */
int hipDeviceCanAccessPeer(int *canAccess, int dev, int peer) {
    init_once();
    if (canAccess) *canAccess = (demap(dev) == demap(peer)) ? 0 : 0;
    return 0;
}

/* --- dlsym interception -------------------------------------------------------------
   LD_PRELOAD only rewrites bindings the dynamic linker resolves. torch's driver_api
   layer dlopen()s libamdhip64 and pulls symbols out with dlsym(handle, name), which
   returns the library's own function and walks straight past the shim -- the same
   reason a ctypes CDLL is not interposed. The previous run got device_count 2 and then
   "CUDA driver error: 101" on the first allocation, which is that bypass.

   So intercept dlsym itself and hand back the wrappers. real dlsym cannot be found with
   dlsym, hence dlvsym against the glibc version.

   Gated on SHIM_HOOK_DLSYM so the run can attribute any change to this hook alone. */
static int g_hook_dlsym = -1;

/* Forward declarations: the wrappers below are defined after this hook but named
   inside it. */
int hipDevicePrimaryCtxRetain(void *, int);
int hipDevicePrimaryCtxRelease(int);
int hipDevicePrimaryCtxSetFlags(int, unsigned);
int hipDevicePrimaryCtxGetState(int, unsigned *, int *);

static void dlsym_init(void) {
    if (!real_dlsym)
        real_dlsym = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    if (g_hook_dlsym < 0) {
        const char *e = getenv("SHIM_HOOK_DLSYM");
        g_hook_dlsym = (e && e[0] == '1') ? 1 : 0;
    }
}

void *dlsym(void *handle, const char *name) {
    dlsym_init();
    if (!real_dlsym) return NULL;
    if (g_hook_dlsym && name) {
        if (!__builtin_strcmp(name, "hipGetDeviceCount"))       return (void *) hipGetDeviceCount;
        if (!__builtin_strcmp(name, "hipSetDevice"))            return (void *) hipSetDevice;
        if (!__builtin_strcmp(name, "hipGetDevice"))            return (void *) hipGetDevice;
        if (!__builtin_strcmp(name, "hipGetDevicePropertiesR0600")) return (void *) hipGetDevicePropertiesR0600;
        if (!__builtin_strcmp(name, "hipGetDeviceProperties"))  return (void *) hipGetDeviceProperties;
        if (!__builtin_strcmp(name, "hipDeviceGetAttribute"))   return (void *) hipDeviceGetAttribute;
        if (!__builtin_strcmp(name, "hipDeviceTotalMem"))       return (void *) hipDeviceTotalMem;
        if (!__builtin_strcmp(name, "hipDeviceGet"))            return (void *) hipDeviceGet;
        if (!__builtin_strcmp(name, "hipDeviceGetName"))        return (void *) hipDeviceGetName;
        if (!__builtin_strcmp(name, "hipDeviceGetUuid"))        return (void *) hipDeviceGetUuid;
        if (!__builtin_strcmp(name, "hipDeviceGetPCIBusId"))    return (void *) hipDeviceGetPCIBusId;
        if (!__builtin_strcmp(name, "hipDeviceCanAccessPeer"))  return (void *) hipDeviceCanAccessPeer;
        if (!__builtin_strcmp(name, "hipDeviceGetDefaultMemPool")) return (void *) hipDeviceGetDefaultMemPool;
        if (!__builtin_strcmp(name, "hipDeviceGetMemPool"))     return (void *) hipDeviceGetMemPool;
        if (!__builtin_strcmp(name, "hipDevicePrimaryCtxRetain"))  return (void *) hipDevicePrimaryCtxRetain;
        if (!__builtin_strcmp(name, "hipDevicePrimaryCtxRelease")) return (void *) hipDevicePrimaryCtxRelease;
        if (!__builtin_strcmp(name, "hipDevicePrimaryCtxSetFlags"))return (void *) hipDevicePrimaryCtxSetFlags;
        if (!__builtin_strcmp(name, "hipDevicePrimaryCtxGetState"))return (void *) hipDevicePrimaryCtxGetState;
    }
    return real_dlsym(handle, name);
}

/* Primary-context calls take a hipDevice_t, which in HIP is the ordinal itself. */
static int (*real_hipDevicePrimaryCtxRetain)(void *, int) = NULL;
int hipDevicePrimaryCtxRetain(void *ctx, int dev) {
    init_once();
    if (!real_hipDevicePrimaryCtxRetain)
        real_hipDevicePrimaryCtxRetain = xdlsym("hipDevicePrimaryCtxRetain");
    if (!real_hipDevicePrimaryCtxRetain) return 101;
    return real_hipDevicePrimaryCtxRetain(ctx, demap(dev));
}

static int (*real_hipDevicePrimaryCtxRelease)(int) = NULL;
int hipDevicePrimaryCtxRelease(int dev) {
    init_once();
    if (!real_hipDevicePrimaryCtxRelease)
        real_hipDevicePrimaryCtxRelease = xdlsym("hipDevicePrimaryCtxRelease");
    if (!real_hipDevicePrimaryCtxRelease) return 101;
    return real_hipDevicePrimaryCtxRelease(demap(dev));
}

static int (*real_hipDevicePrimaryCtxSetFlags)(int, unsigned) = NULL;
int hipDevicePrimaryCtxSetFlags(int dev, unsigned flags) {
    init_once();
    if (!real_hipDevicePrimaryCtxSetFlags)
        real_hipDevicePrimaryCtxSetFlags = xdlsym("hipDevicePrimaryCtxSetFlags");
    if (!real_hipDevicePrimaryCtxSetFlags) return 101;
    return real_hipDevicePrimaryCtxSetFlags(demap(dev), flags);
}

static int (*real_hipDevicePrimaryCtxGetState)(int, unsigned *, int *) = NULL;
int hipDevicePrimaryCtxGetState(int dev, unsigned *flags, int *active) {
    init_once();
    if (!real_hipDevicePrimaryCtxGetState)
        real_hipDevicePrimaryCtxGetState = xdlsym("hipDevicePrimaryCtxGetState");
    if (!real_hipDevicePrimaryCtxGetState) return 101;
    return real_hipDevicePrimaryCtxGetState(demap(dev), flags, active);
}
'''


def available() -> tuple[bool, str]:
    """(can this host build the shim, why not)."""
    if os.name != "posix":
        return False, "LD_PRELOAD is POSIX only"
    if not shutil.which("gcc"):
        return False, "no gcc to build the shim"
    return True, ""


def build(dest_dir: Path, cc: str = "gcc") -> Path:
    """Compile the shim into dest_dir and return the .so path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents = True, exist_ok = True)
    src = dest_dir / "hip_device_multiplier.c"
    so = dest_dir / "libhip_device_multiplier.so"
    src.write_text(SHIM_C, encoding = "utf-8")
    r = subprocess.run([cc, "-shared", "-fPIC", "-O2", "-o", str(so), str(src), "-ldl"],
                       capture_output = True, text = True)
    if r.returncode != 0:
        raise RuntimeError(f"shim build failed:\n{r.stderr[-2000:]}")
    return so


def env_for(so: Path, extra_devices: int = 1, base: dict | None = None,
            hook_dlsym: bool = False) -> dict:
    """An environment with the shim active.

    `hook_dlsym` additionally intercepts dlsym, for consumers that dlopen the HIP
    library and resolve symbols through the handle (LD_PRELOAD does not rewrite those).
    torch does NOT need it: the run that first passed had it off, and the four
    primary-context wrappers were what actually fixed allocation. It is off by default
    so a consumer that needs it turns it on deliberately.
    """
    env = dict(os.environ if base is None else base)
    pre = env.get("LD_PRELOAD")
    env["LD_PRELOAD"] = f"{so}{os.pathsep}{pre}" if pre else str(so)
    env["SHIM_EXTRA_DEVICES"] = str(extra_devices)
    env[ENV_MARKER] = str(extra_devices)
    if hook_dlsym:
        env["SHIM_HOOK_DLSYM"] = "1"
    return env


def spoofed_count() -> int:
    """How many devices are fabricated in THIS process, 0 when the shim is off."""
    try:
        return max(0, int(os.environ.get(ENV_MARKER, "0")))
    except ValueError:
        return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description = __doc__.split("\n")[0])
    ap.add_argument("--build-into", type = Path,
                    help = "compile the shim into this directory")
    ap.add_argument("--extra", type = int, default = 1,
                    help = "how many phantom devices to add")
    ap.add_argument("--github-env", action = "store_true",
                    help = "append the activating variables to $GITHUB_ENV")
    ap.add_argument("--hook-dlsym", action = "store_true")
    args = ap.parse_args()

    ok, why = available()
    if not args.build_into:
        print(f"buildable: {ok}" + (f" ({why})" if why else ""))
        print(f"spoofed in this process: {spoofed_count()}")
        return 0
    if not ok:
        raise SystemExit(f"cannot build the device multiplier: {why}")

    so = build(args.build_into)
    print(f"built {so}")

    if args.github_env:
        gh = os.environ.get("GITHUB_ENV")
        if not gh:
            raise SystemExit("--github-env outside GitHub Actions: $GITHUB_ENV is unset")
        pre = os.environ.get("LD_PRELOAD")
        lines = [
            f"LD_PRELOAD={so}{os.pathsep}{pre}" if pre else f"LD_PRELOAD={so}",
            f"SHIM_EXTRA_DEVICES={args.extra}",
            # Read by capability.py, which then refuses to let these devices satisfy
            # multi_gpu. Activation and disclosure are one action on purpose.
            f"{ENV_MARKER}={args.extra}",
        ]
        if args.hook_dlsym:
            lines.append("SHIM_HOOK_DLSYM=1")
        with open(gh, "a", encoding = "utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"activated: +{args.extra} phantom device(s); {ENV_MARKER} set so the "
              f"verdict declares them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

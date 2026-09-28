#!/usr/bin/env python3
"""Present more HIP devices to torch than the host physically has.

torch reaches its device list through the HIP runtime, and HIP is a shared library, so
an LD_PRELOAD shim can inflate the count and fold the extra ordinals onto the real
device. Measured on the gfx1151 runner: `torch.cuda.device_count()` is 2, `cuda:1`
accepts real tensors, and a real matmul runs on it.

**For wiring only.** Selection logic, arch gates, device-index bookkeeping, code
assuming `device_count() == 1`. The phantom device IS the real GPU wearing another
number: shared memory, shared compute, serialised. It gives a confident wrong answer,
not an error, if a probe measures sharding, throughput, collectives or mixed
architectures.

Two limits before reaching for it:

- **It only fools HIP.** amd-smi, rocm-smi, sysfs, KFD and rocminfo still see one GPU.
  Studio's `utils/hardware/hardware.py` enumerates through amd-smi, not torch, so this
  does not move it at all; that needs the amd-smi binary stubbed as well.
- **Symbol coverage is only what torch drives.** A consumer calling an ordinal-taking
  HIP function the shim does not wrap (vLLM, bitsandbytes, RCCL) gets the real one with
  an out-of-range ordinal and dies with hipErrorInvalidDevice 101. Add the wrapper.
  If it resolves symbols via `dlsym` on a `dlopen` handle, LD_PRELOAD cannot reach it
  and the shim would also need to export `dlsym`; torch does not need this.

Linux only. NVIDIA is untried: the equivalent wraps cuDeviceGet / cuDeviceGetAttribute
/ cuDevicePrimaryCtxRetain.

<critical>Activating this sets AMD_CI_SPOOFED_DEVICES, and capability.py subtracts those
devices so `multi_gpu` stays UNMET and the verdict names the fabrication. Without that
coupling a spoofed run drops the multi-GPU gap and reads as hardware validation. Do not
set the marker by hand.</critical>
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
#include <stdlib.h>

/* Report g_extra more HIP devices than exist, folding the extra ordinals onto device 0.
   Every HIP call taking a device ordinal needs a wrapper: the count is the easy part.
   An unwrapped one receives the phantom ordinal and returns hipErrorInvalidDevice 101,
   which is how the first version died inside rocBLAS. Add a wrapper, never widen
   demap(). */

static int g_extra = 1;
static int g_real  = -1;
static __thread int g_current = 0;   /* the ordinal the caller THINKS is current */

static int (*real_count)(int *) = NULL;
static int (*real_set)(int) = NULL;

static void *xdlsym(const char *name) { return dlsym(RTLD_NEXT, name); }

static void init_once(void) {
    if (!real_count) {
        real_count = xdlsym("hipGetDeviceCount");
        real_set   = xdlsym("hipSetDevice");
        const char *e = getenv("SHIM_EXTRA_DEVICES");
        if (e) g_extra = atoi(e);
    }
    if (g_real < 0 && real_count) {
        int c = 0;
        if (real_count(&c) == 0) g_real = c;
    }
}

static int demap(int dev) { return (g_real > 0 && dev >= g_real) ? 0 : dev; }

int hipGetDeviceCount(int *count) {
    init_once();
    if (!real_count) return 101;
    int rc = real_count(count);
    if (rc == 0 && count) { g_real = *count; *count += g_extra; }
    return rc;
}

int hipSetDevice(int dev) {
    init_once();
    g_current = dev;
    return real_set ? real_set(demap(dev)) : 101;
}

/* Report the VIRTUAL ordinal back. Returning the demapped one makes torch believe its
   set_device silently failed. */
int hipGetDevice(int *dev) {
    init_once();
    if (!dev) return 101;
    *dev = g_current;
    return 0;
}

/* hipGetDevicePropertiesR0600 is the name exported since ROCm 6.0; the unversioned
   spelling covers older stacks. Whichever is absent never binds. */
#define WRAP1(name, t1)                                            \
    static int (*real_##name)(t1, int) = NULL;                     \
    int name(t1 a, int dev) {                                      \
        init_once();                                               \
        if (!real_##name) real_##name = xdlsym(#name);             \
        return real_##name ? real_##name(a, demap(dev)) : 101;     \
    }

#define WRAP2(name, t1, t2)                                        \
    static int (*real_##name)(t1, t2, int) = NULL;                 \
    int name(t1 a, t2 b, int dev) {                                \
        init_once();                                               \
        if (!real_##name) real_##name = xdlsym(#name);             \
        return real_##name ? real_##name(a, b, demap(dev)) : 101;  \
    }

WRAP1(hipGetDevicePropertiesR0600, void *)
WRAP1(hipGetDeviceProperties,      void *)
WRAP1(hipDeviceTotalMem,           size_t *)
WRAP1(hipDeviceGetDefaultMemPool,  void *)
WRAP1(hipDeviceGetMemPool,         void *)
WRAP1(hipDeviceGet,                int *)
WRAP1(hipDeviceGetUuid,            void *)
WRAP1(hipDevicePrimaryCtxRetain,   void *)
WRAP2(hipDeviceGetName,            char *, int)
WRAP2(hipDeviceGetPCIBusId,        char *, int)

/* Allocation on the phantom device fails with driver error 101 without these: the
   primary-context calls take a hipDevice_t, which in HIP is the ordinal itself. */
#define WRAP_DEV_ONLY(name, extra_t)                               \
    static int (*real_##name)(int, extra_t) = NULL;                \
    int name(int dev, extra_t x) {                                 \
        init_once();                                               \
        if (!real_##name) real_##name = xdlsym(#name);             \
        return real_##name ? real_##name(demap(dev), x) : 101;     \
    }

WRAP_DEV_ONLY(hipDevicePrimaryCtxSetFlags, unsigned)

static int (*real_release)(int) = NULL;
int hipDevicePrimaryCtxRelease(int dev) {
    init_once();
    if (!real_release) real_release = xdlsym("hipDevicePrimaryCtxRelease");
    return real_release ? real_release(demap(dev)) : 101;
}

static int (*real_attr)(int *, int, int) = NULL;
int hipDeviceGetAttribute(int *v, int attr, int dev) {
    init_once();
    if (!real_attr) real_attr = xdlsym("hipDeviceGetAttribute");
    return real_attr ? real_attr(v, attr, demap(dev)) : 101;
}

static int (*real_state)(int, unsigned *, int *) = NULL;
int hipDevicePrimaryCtxGetState(int dev, unsigned *flags, int *active) {
    init_once();
    if (!real_state) real_state = xdlsym("hipDevicePrimaryCtxGetState");
    return real_state ? real_state(demap(dev), flags, active) : 101;
}

/* Report peer access UNAVAILABLE: the two ordinals are one GPU, so a P2P copy would be
   device-to-itself. Saying no makes torch stage through the host instead of aliasing. */
int hipDeviceCanAccessPeer(int *canAccess, int dev, int peer) {
    (void) dev; (void) peer;
    init_once();
    if (canAccess) *canAccess = 0;
    return 0;
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


def env_for(so: Path, extra_devices: int = 1, base: dict | None = None) -> dict:
    """An environment with the shim active."""
    env = dict(os.environ if base is None else base)
    pre = env.get("LD_PRELOAD")
    env["LD_PRELOAD"] = f"{so}{os.pathsep}{pre}" if pre else str(so)
    env["SHIM_EXTRA_DEVICES"] = str(extra_devices)
    env[ENV_MARKER] = str(extra_devices)
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
        with open(gh, "a", encoding = "utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"activated: +{args.extra} phantom device(s); {ENV_MARKER} set so the "
              f"verdict declares them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Probe: what display / render stack does this host actually expose?

Observes only. It answers "what is here", never "is that good enough": the
question of whether a stack is sufficient to host a browser perf venue belongs
to criteria/display_stack_capability.py, and a probe that decided it for itself
could be written so the answer is always yes.

The decisive observation is the GL renderer STRING obtained from a context that
was really made current, not the presence of a driver package. A host with
mesa installed and no usable device still reports an ICD file, still has
`libEGL.so.1`, and still yields `llvmpipe` once you ask it to render. The
existing `vulkan` capability in lib/capability.py is exactly that weaker kind of
claim, keyed on an ICD filename; this probe supersedes it by enumerating a
physical device and reading its name and type.

Everything that can segfault (EGL, GBM, Vulkan) runs in a child process, so a
driver that dies on this hardware is recorded as a crash rather than taking the
observation with it.

  python display_stack_probe.py --out obs.json
  python display_stack_probe.py --egl-worker --node /dev/dri/renderD128 \
      --platform gbm --out egl.json      # internal, re-entrant
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import grp
import json
import os
import platform
import pwd
import shutil
import subprocess
import sys
from pathlib import Path

TIMEOUT = 60


# --------------------------------------------------------------------------
# helpers


def sh(cmd: list[str], timeout: int = TIMEOUT) -> dict:
    """Run a command, never raise. A missing tool is an observation."""
    exe = shutil.which(cmd[0])
    if exe is None:
        return {"present": False}
    try:
        r = subprocess.run(cmd, capture_output = True, text = True, timeout = timeout)
        return {
            "present": True, "path": exe, "rc": r.returncode,
            "stdout": r.stdout[-20000:], "stderr": r.stderr[-4000:],
        }
    except Exception as e:  # noqa: BLE001
        return {"present": True, "path": exe, "error": f"{type(e).__name__}: {e}"}


def read(path: str) -> str | None:
    try:
        return Path(path).read_text(errors = "replace").strip()
    except Exception:  # noqa: BLE001
        return None


def _name(fn, i):
    try:
        return fn(i)
    except Exception:  # noqa: BLE001
        return str(i)


# --------------------------------------------------------------------------
# 1. /dev/dri


def observe_dri_nodes() -> dict:
    out: dict = {"dir_exists": os.path.isdir("/dev/dri"), "nodes": []}
    for path in sorted(glob.glob("/dev/dri/*")) + sorted(glob.glob("/dev/dri/by-path/*")):
        entry: dict = {"path": path}
        try:
            st = os.stat(path)
            entry.update({
                "mode": oct(st.st_mode & 0o7777),
                "uid": st.st_uid, "gid": st.st_gid,
                "owner": _name(lambda i: pwd.getpwuid(i).pw_name, st.st_uid),
                "group": _name(lambda i: grp.getgrgid(i).gr_name, st.st_gid),
                "is_char_device": os.path.exists(path) and not os.path.isdir(path),
            })
        except Exception as e:  # noqa: BLE001
            entry["stat_error"] = f"{type(e).__name__}: {e}"
        entry["access_r"] = os.access(path, os.R_OK)
        entry["access_w"] = os.access(path, os.W_OK)
        # os.access consults the real uid and lies under some ACL setups; the
        # only claim worth making is whether an open actually succeeds.
        if not os.path.isdir(path):
            try:
                fd = os.open(path, os.O_RDWR)
                os.close(fd)
                entry["open_rdwr"] = True
            except Exception as e:  # noqa: BLE001
                entry["open_rdwr"] = False
                entry["open_error"] = f"{type(e).__name__}: {e}"
        out["nodes"].append(entry)
    return out


def observe_drm_sysfs() -> dict:
    out: dict = {"entries": []}
    for p in sorted(glob.glob("/sys/class/drm/*")):
        e: dict = {"name": os.path.basename(p)}
        e["vendor"] = read(f"{p}/device/vendor")
        e["device"] = read(f"{p}/device/device")
        try:
            e["driver"] = os.path.basename(os.readlink(f"{p}/device/driver"))
        except Exception:  # noqa: BLE001
            e["driver"] = None
        e["status"] = read(f"{p}/status")
        out["entries"].append(e)
    return out


# --------------------------------------------------------------------------
# 2. PCI


def observe_pci() -> dict:
    r = sh(["lspci", "-nnk"])
    if r.get("rc") == 0:
        want = ("vga", "display", "3d controller")
        keep, take = [], 0
        for line in r["stdout"].splitlines():
            low = line.lower()
            if line and not line[0].isspace():
                take = 4 if any(w in low for w in want) else 0
            if take:
                keep.append(line)
                take -= 1
        r["display_controllers"] = keep
    return r


# --------------------------------------------------------------------------
# 3/5. EGL, GBM, GL renderer string.  Runs in a child; see module docstring.

EGL_PLATFORM_GBM_KHR = 0x31D7
EGL_PLATFORM_SURFACELESS_MESA = 0x31DD
EGL_PLATFORM_DEVICE_EXT = 0x313F

EGL_VENDOR, EGL_VERSION_S, EGL_EXTENSIONS, EGL_CLIENT_APIS = 0x3053, 0x3054, 0x3055, 0x308D
EGL_OPENGL_ES_API = 0x30A0
EGL_SURFACE_TYPE, EGL_PBUFFER_BIT = 0x3033, 0x0001
EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT = 0x3040, 0x0004
EGL_RED_SIZE, EGL_GREEN_SIZE, EGL_BLUE_SIZE, EGL_ALPHA_SIZE = 0x3024, 0x3023, 0x3022, 0x3021
EGL_WIDTH, EGL_HEIGHT, EGL_NONE = 0x3057, 0x3056, 0x3038
EGL_CONTEXT_CLIENT_VERSION = 0x3098

GL_VENDOR, GL_RENDERER, GL_VERSION, GL_SL_VERSION = 0x1F00, 0x1F01, 0x1F02, 0x8B8C
GL_COLOR_BUFFER_BIT = 0x4000
GL_RGBA, GL_UNSIGNED_BYTE = 0x1908, 0x1401


def egl_worker(node: str | None, plat: str) -> dict:
    """Bring up EGL and read the renderer string off a context made current."""
    o: dict = {"platform": plat, "node": node}
    fd = None
    try:
        egl = ctypes.CDLL("libEGL.so.1")
    except Exception as e:  # noqa: BLE001
        o["error"] = f"libEGL.so.1 not loadable: {e}"
        return o
    o["libEGL"] = True

    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    egl.eglQueryString.restype = ctypes.c_char_p
    egl.eglQueryString.argtypes = [ctypes.c_void_p, ctypes.c_int]
    egl.eglGetError.restype = ctypes.c_int

    def qs(dpy, name):
        v = egl.eglQueryString(dpy, name)
        return v.decode() if v else None

    o["client_extensions"] = qs(None, EGL_EXTENSIONS)

    addr = egl.eglGetProcAddress(b"eglGetPlatformDisplayEXT")
    if not addr:
        o["error"] = "eglGetPlatformDisplayEXT unavailable; cannot select a platform"
        return o
    proto = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p,
                             ctypes.POINTER(ctypes.c_int))
    get_platform_display = proto(addr)

    native = None
    if plat == "gbm":
        try:
            gbm = ctypes.CDLL("libgbm.so.1")
        except Exception as e:  # noqa: BLE001
            o["error"] = f"libgbm.so.1 not loadable: {e}"
            return o
        gbm.gbm_create_device.restype = ctypes.c_void_p
        gbm.gbm_create_device.argtypes = [ctypes.c_int]
        try:
            fd = os.open(node, os.O_RDWR)
        except Exception as e:  # noqa: BLE001
            o["error"] = f"cannot open {node}: {type(e).__name__}: {e}"
            return o
        native = gbm.gbm_create_device(fd)
        o["gbm_device"] = bool(native)
        if not native:
            o["error"] = f"gbm_create_device failed on {node}"
            os.close(fd)
            return o
        try:
            gbm.gbm_device_get_backend_name.restype = ctypes.c_char_p
            gbm.gbm_device_get_backend_name.argtypes = [ctypes.c_void_p]
            bn = gbm.gbm_device_get_backend_name(ctypes.c_void_p(native))
            o["gbm_backend"] = bn.decode() if bn else None
        except Exception:  # noqa: BLE001
            pass
        enum = EGL_PLATFORM_GBM_KHR
    elif plat == "surfaceless":
        enum = EGL_PLATFORM_SURFACELESS_MESA
    elif plat == "device":
        enum = EGL_PLATFORM_DEVICE_EXT
    else:
        o["error"] = f"unknown platform {plat}"
        return o

    dpy = get_platform_display(enum, ctypes.c_void_p(native), None)
    o["display"] = bool(dpy)
    if not dpy:
        o["error"] = f"eglGetPlatformDisplayEXT returned EGL_NO_DISPLAY (0x{egl.eglGetError():x})"
        return o

    major, minor = ctypes.c_int(), ctypes.c_int()
    if not egl.eglInitialize(ctypes.c_void_p(dpy), ctypes.byref(major), ctypes.byref(minor)):
        o["error"] = f"eglInitialize failed (0x{egl.eglGetError():x})"
        return o
    o["egl_version"] = f"{major.value}.{minor.value}"
    o["egl_vendor"] = qs(ctypes.c_void_p(dpy), EGL_VENDOR)
    o["egl_version_string"] = qs(ctypes.c_void_p(dpy), EGL_VERSION_S)
    o["egl_client_apis"] = qs(ctypes.c_void_p(dpy), EGL_CLIENT_APIS)
    exts = qs(ctypes.c_void_p(dpy), EGL_EXTENSIONS) or ""
    o["egl_display_extensions"] = exts
    # WebKitGTK's accelerated compositing path imports rendered frames as dmabuf
    # textures; without these two it falls back to a software path even when a
    # GPU context exists.
    o["has_image_dmabuf_import"] = "EGL_EXT_image_dma_buf_import" in exts
    o["has_image_dmabuf_import_modifiers"] = "EGL_EXT_image_dma_buf_import_modifiers" in exts
    o["has_surfaceless_context"] = "EGL_KHR_surfaceless_context" in exts

    egl.eglBindAPI(EGL_OPENGL_ES_API)

    attribs = (ctypes.c_int * 13)(
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_NONE)
    cfg = ctypes.c_void_p()
    n = ctypes.c_int()
    ok = egl.eglChooseConfig(ctypes.c_void_p(dpy), attribs, ctypes.byref(cfg), 1,
                             ctypes.byref(n))
    o["configs"] = n.value
    if not ok or n.value < 1:
        o["error"] = f"eglChooseConfig found no pbuffer-capable config (0x{egl.eglGetError():x})"
        return o

    ctx_attr = (ctypes.c_int * 3)(EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE)
    egl.eglCreateContext.restype = ctypes.c_void_p
    ctx = egl.eglCreateContext(ctypes.c_void_p(dpy), cfg, None, ctx_attr)
    o["context"] = bool(ctx)
    if not ctx:
        o["error"] = f"eglCreateContext failed (0x{egl.eglGetError():x})"
        return o

    egl.eglCreatePbufferSurface.restype = ctypes.c_void_p
    surf_attr = (ctypes.c_int * 5)(EGL_WIDTH, 64, EGL_HEIGHT, 64, EGL_NONE)
    surf = egl.eglCreatePbufferSurface(ctypes.c_void_p(dpy), cfg, surf_attr)
    o["pbuffer"] = bool(surf)
    made = egl.eglMakeCurrent(ctypes.c_void_p(dpy), ctypes.c_void_p(surf),
                              ctypes.c_void_p(surf), ctypes.c_void_p(ctx))
    if not made and o["has_surfaceless_context"]:
        made = egl.eglMakeCurrent(ctypes.c_void_p(dpy), None, None, ctypes.c_void_p(ctx))
        o["made_current_surfaceless"] = bool(made)
    o["made_current"] = bool(made)
    if not made:
        o["error"] = f"eglMakeCurrent failed (0x{egl.eglGetError():x})"
        return o

    try:
        gl = ctypes.CDLL("libGLESv2.so.2")
    except Exception as e:  # noqa: BLE001
        o["error"] = f"libGLESv2.so.2 not loadable: {e}"
        return o
    gl.glGetString.restype = ctypes.c_char_p
    gl.glGetString.argtypes = [ctypes.c_uint]

    def gs(name):
        v = gl.glGetString(name)
        return v.decode() if v else None

    o["gl_vendor"] = gs(GL_VENDOR)
    o["gl_renderer"] = gs(GL_RENDERER)      # the decisive one
    o["gl_version"] = gs(GL_VERSION)
    o["gl_shading_language"] = gs(GL_SL_VERSION)

    # 7 (at the GL layer): the smallest possible thing that paints. A renderer
    # string can be reported by a driver that then fails to draw, so clear to a
    # known colour and read the pixel back.
    if surf:
        try:
            gl.glClearColor(ctypes.c_float(0.0), ctypes.c_float(1.0),
                            ctypes.c_float(0.0), ctypes.c_float(1.0))
            gl.glClear(ctypes.c_uint(GL_COLOR_BUFFER_BIT))
            gl.glFinish()
            buf = (ctypes.c_ubyte * 4)()
            gl.glReadPixels(0, 0, 1, 1, ctypes.c_uint(GL_RGBA),
                            ctypes.c_uint(GL_UNSIGNED_BYTE), buf)
            o["painted_pixel_rgba"] = list(buf)
            o["gl_error_after_paint"] = int(gl.glGetError())
        except Exception as e:  # noqa: BLE001
            o["paint_error"] = f"{type(e).__name__}: {e}"

    try:
        egl.eglMakeCurrent(ctypes.c_void_p(dpy), None, None, None)
        egl.eglTerminate(ctypes.c_void_p(dpy))
    except Exception:  # noqa: BLE001
        pass
    if fd is not None:
        try:
            os.close(fd)
        except Exception:  # noqa: BLE001
            pass
    return o


# --------------------------------------------------------------------------
# 4. Vulkan, read from an enumerated physical device rather than an ICD file.

VK_TYPES = {0: "OTHER", 1: "INTEGRATED_GPU", 2: "DISCRETE_GPU",
            3: "VIRTUAL_GPU", 4: "CPU"}


class VkApplicationInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("pApplicationName", ctypes.c_char_p), ("applicationVersion", ctypes.c_uint),
                ("pEngineName", ctypes.c_char_p), ("engineVersion", ctypes.c_uint),
                ("apiVersion", ctypes.c_uint)]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("flags", ctypes.c_uint), ("pApplicationInfo", ctypes.c_void_p),
                ("enabledLayerCount", ctypes.c_uint), ("ppEnabledLayerNames", ctypes.c_void_p),
                ("enabledExtensionCount", ctypes.c_uint),
                ("ppEnabledExtensionNames", ctypes.c_void_p)]


class VkPhysicalDevicePropertiesHead(ctypes.Structure):
    """Only the leading fields; `limits` follows and is not needed here."""
    _fields_ = [("apiVersion", ctypes.c_uint), ("driverVersion", ctypes.c_uint),
                ("vendorID", ctypes.c_uint), ("deviceID", ctypes.c_uint),
                ("deviceType", ctypes.c_uint), ("deviceName", ctypes.c_char * 256),
                ("pipelineCacheUUID", ctypes.c_ubyte * 16),
                ("_tail", ctypes.c_ubyte * 4096)]


def vulkan_worker() -> dict:
    o: dict = {}
    try:
        vk = ctypes.CDLL("libvulkan.so.1")
    except Exception as e:  # noqa: BLE001
        o["error"] = f"libvulkan.so.1 not loadable: {e}"
        return o
    o["libvulkan"] = True

    app = VkApplicationInfo(0, None, b"amd_ci_display_probe", 1, b"none", 1,
                            (1 << 22) | (0 << 12) | 0)
    ci = VkInstanceCreateInfo(1, None, 0, ctypes.cast(ctypes.byref(app), ctypes.c_void_p),
                              0, None, 0, None)
    inst = ctypes.c_void_p()
    rc = vk.vkCreateInstance(ctypes.byref(ci), None, ctypes.byref(inst))
    o["vkCreateInstance_rc"] = rc
    if rc != 0:
        o["error"] = f"vkCreateInstance failed with VkResult {rc}"
        return o

    count = ctypes.c_uint(0)
    vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(count), None)
    o["physical_device_count"] = count.value
    if count.value == 0:
        o["error"] = "an instance was created but no physical device was enumerated"
        return o
    arr = (ctypes.c_void_p * count.value)()
    vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(count), arr)

    o["devices"] = []
    for i in range(count.value):
        props = VkPhysicalDevicePropertiesHead()
        vk.vkGetPhysicalDeviceProperties(arr[i], ctypes.byref(props))
        a = props.apiVersion
        o["devices"].append({
            "name": props.deviceName.decode(errors = "replace"),
            "type": VK_TYPES.get(props.deviceType, str(props.deviceType)),
            "vendor_id": hex(props.vendorID), "device_id": hex(props.deviceID),
            "api_version": f"{(a >> 22) & 0x7F}.{(a >> 12) & 0x3FF}.{a & 0xFFF}",
            "driver_version_raw": props.driverVersion,
        })
    return o


# --------------------------------------------------------------------------
# 6. WebKitGTK reachability


SO_DIRS = ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/lib/x86_64-linux-gnu",
           "/usr/local/lib/x86_64-linux-gnu", "/usr/local/lib"]


def find_so(patterns: list[str]) -> list[str]:
    hits: list[str] = []
    for d in SO_DIRS:
        for pat in patterns:
            hits += sorted(glob.glob(os.path.join(d, pat)))
    return sorted(set(hits))


def observe_webkit() -> dict:
    o: dict = {}
    o["libwebkit2gtk_4_1"] = find_so(["libwebkit2gtk-4.1.so*"])
    o["libwebkit2gtk_4_0"] = find_so(["libwebkit2gtk-4.0.so*"])
    o["libwebkitgtk_6_0"] = find_so(["libwebkitgtk-6.0.so*"])
    o["libwpe"] = find_so(["libWPEWebKit*.so*", "libwpe-1.0.so*"])
    o["libgtk"] = find_so(["libgtk-3.so*", "libgtk-4.so*"])
    o["typelibs"] = sorted(set(
        glob.glob("/usr/lib/*/girepository-1.0/WebKit*.typelib")
        + glob.glob("/usr/lib/girepository-1.0/WebKit*.typelib")))
    o["ldconfig_webkit"] = [
        ln.strip() for ln in (sh(["ldconfig", "-p"]).get("stdout") or "").splitlines()
        if "webkit" in ln.lower() or "libwpe" in ln.lower()]
    o["pkg_config_webkit2gtk_4_1"] = sh(["pkg-config", "--modversion", "webkit2gtk-4.1"])
    o["dpkg_webkit"] = sh(["dpkg-query", "-W", "-f=${Package} ${Version} ${Status}\\n",
                           "libwebkit2gtk-4.1-0", "libwebkit2gtk-4.0-37",
                           "gir1.2-webkit2-4.1", "libgtk-3-0", "libgtk-3-0t64"])
    o["python_gi"] = sh([sys.executable, "-c",
                         "import gi; print(gi.__version__, gi.__file__)"])
    # Can a non-root user fetch the package at all? --print-uris mutates nothing
    # and distinguishes "no such package / no sources" from "no permission".
    o["apt_print_uris"] = sh(["apt-get", "-y", "--print-uris", "install",
                              "libwebkit2gtk-4.1-0"], timeout = 120)
    o["apt_policy"] = sh(["apt-cache", "policy", "libwebkit2gtk-4.1-0"], timeout = 60)
    o["have_dpkg_deb"] = shutil.which("dpkg-deb") is not None
    o["sources_list"] = sorted(glob.glob("/etc/apt/sources.list.d/*")) + (
        ["/etc/apt/sources.list"] if os.path.exists("/etc/apt/sources.list") else [])
    return o


def observe_display_servers() -> dict:
    o: dict = {}
    for tool in ("Xvfb", "Xwayland", "weston", "cage", "sway", "wayfire", "labwc",
                 "xvfb-run", "glxinfo", "eglinfo", "vulkaninfo", "drm_info",
                 "wlr-randr", "dbus-run-session", "xauth"):
        o[tool] = shutil.which(tool)
    o["env"] = {k: os.environ.get(k) for k in
                ("DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "XDG_SESSION_TYPE",
                 "LIBGL_ALWAYS_SOFTWARE", "MESA_LOADER_DRIVER_OVERRIDE")}
    xrd = os.environ.get("XDG_RUNTIME_DIR")
    o["xdg_runtime_dir_usable"] = bool(xrd and os.path.isdir(xrd) and os.access(xrd, os.W_OK))
    o["logind_seats"] = sorted(glob.glob("/run/systemd/seats/*"))
    o["browsers"] = {n: shutil.which(n) for n in
                     ("chromium", "chromium-browser", "google-chrome", "firefox")}
    o["playwright_cache"] = sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/*")))
    return o


# --------------------------------------------------------------------------


def child(args_extra: list[str], out: Path) -> dict:
    """Run this file again for the crash-prone parts."""
    cmd = [sys.executable, os.path.abspath(__file__), *args_extra, "--out", str(out)]
    try:
        r = subprocess.run(cmd, capture_output = True, text = True, timeout = TIMEOUT + 60)
    except Exception as e:  # noqa: BLE001
        return {"child_error": f"{type(e).__name__}: {e}"}
    d: dict = {"child_rc": r.returncode}
    if r.returncode < 0:
        d["crashed_with_signal"] = -r.returncode
    if out.is_file():
        try:
            d.update(json.loads(out.read_text()))
        except Exception as e:  # noqa: BLE001
            d["parse_error"] = f"{type(e).__name__}: {e}"
    if r.stderr.strip():
        d["child_stderr"] = r.stderr[-2000:]
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--state", default = "single")
    ap.add_argument("--checkout", default = "")
    ap.add_argument("--egl-worker", action = "store_true")
    ap.add_argument("--vulkan-worker", action = "store_true")
    ap.add_argument("--node", default = None)
    ap.add_argument("--platform", default = "gbm")
    args = ap.parse_args()
    args.out.parent.mkdir(parents = True, exist_ok = True)

    if args.egl_worker:
        args.out.write_text(json.dumps(egl_worker(args.node, args.platform), indent = 2))
        return 0
    if args.vulkan_worker:
        args.out.write_text(json.dumps(vulkan_worker(), indent = 2))
        return 0

    tmp = args.out.parent
    obs: dict = {
        "state": args.state,
        "host": {
            "hostname": platform.node(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "uid": os.getuid(), "user": _name(lambda i: pwd.getpwuid(i).pw_name, os.getuid()),
            "groups": sorted(_name(lambda i: grp.getgrgid(i).gr_name, g)
                             for g in os.getgroups()),
            "os_release": read("/etc/os-release"),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "runner_temp": os.environ.get("RUNNER_TEMP"),
        },
    }

    obs["dri"] = observe_dri_nodes()
    obs["drm_sysfs"] = observe_drm_sysfs()
    obs["pci"] = observe_pci()
    obs["drm_info"] = sh(["drm_info"])
    obs["rocminfo"] = sh(["rocminfo"], timeout = 120)
    obs["display_servers"] = observe_display_servers()
    obs["webkit"] = observe_webkit()

    obs["vulkan_icd_files"] = sorted(glob.glob("/usr/share/vulkan/icd.d/*.json")
                                     + glob.glob("/etc/vulkan/icd.d/*.json"))
    obs["mesa_dri_drivers"] = sorted(
        glob.glob("/usr/lib/x86_64-linux-gnu/dri/*.so") + glob.glob("/usr/lib/dri/*.so"))

    # ---- the crash-prone parts, each in a child
    obs["vulkan"] = child(["--vulkan-worker"], tmp / "_vk.json")
    obs["vulkaninfo_summary"] = sh(["vulkaninfo", "--summary"], timeout = 120)
    obs["glxinfo"] = sh(["glxinfo", "-B"])
    obs["eglinfo"] = sh(["eglinfo"])

    obs["egl"] = {}
    render_nodes = [n["path"] for n in obs["dri"]["nodes"]
                    if os.path.basename(n["path"]).startswith("renderD")]
    for node in render_nodes:
        key = f"gbm:{os.path.basename(node)}"
        obs["egl"][key] = child(["--egl-worker", "--platform", "gbm", "--node", node],
                                tmp / f"_egl_{os.path.basename(node)}.json")
    for plat in ("surfaceless", "device"):
        obs["egl"][plat] = child(["--egl-worker", "--platform", plat],
                                 tmp / f"_egl_{plat}.json")

    args.out.write_text(json.dumps(obs, indent = 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Probe: bring up WebKitGTK on this host and see what it renders WITH.

Observes only. It reports what happened at each step and never decides whether
that is good enough; criteria/webkit_gpu_compositing.py does that.

The question it exists to settle cannot be answered by inspection. A host can
have libwebkit2gtk-4.1, a render node, and a hardware EGL context, and still
composite pages on the CPU, because the engine picks its own driver in its own
process. So the reading that matters is taken from INSIDE the page:
`WEBGL_debug_renderer_info` reports the renderer WebKit's own GL context got,
and it is the same string the standalone EGL probe reads, which makes the two
directly comparable. Two independent corroborations are collected beside it:
which of WebKit's auxiliary processes hold /dev/dri/renderD128 open, and what
amdgpu's per-fd counters say those processes did.

It also has to solve the display problem: GTK needs a display server, this host
has no X server, no compositor and no root. The attempts are recorded in order
so that "we could not get a display" is never silently reported as "WebKit
cannot render here".

  python webkit_paint_probe.py --out obs.json --work /path/to/scratch
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# Packages fetched with `apt-get download`, which needs no root: it writes the
# .deb into the cwd and reads only the lists. dpkg-deb -x then unpacks it into a
# private prefix under $RUNNER_TEMP, which the runner wipes between jobs.
XVFB_PKGS = ["xvfb", "xserver-common", "x11-xkb-utils", "xkb-data", "xfonts-base",
             "libxfont2", "libunwind8", "libpixman-1-0", "libxkbfile1"]

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;background:#101010;overflow:hidden}
.card{position:absolute;width:180px;height:120px;border-radius:12px;
  background:linear-gradient(135deg,#3aa,#a3a);will-change:transform;
  transform:translateZ(0);animation:spin 1.4s linear infinite}
@keyframes spin{from{transform:translateZ(0) rotate(0)}to{transform:translateZ(0) rotate(360deg)}}
</style></head><body>
<div id=host></div>
<script>
var host=document.getElementById('host');
for(var i=0;i<24;i++){var d=document.createElement('div');d.className='card';
  d.style.left=(40+(i%6)*200)+'px';d.style.top=(30+Math.floor(i/6)*160)+'px';
  d.style.animationDelay=(i*0.05)+'s';host.appendChild(d);}
var n=0,t0=null,last=null,deltas=[];
function frame(t){ if(t0===null){t0=t;last=t;} n++; deltas.push(t-last); last=t;
  if(t-t0<__MS__){requestAnimationFrame(frame);} else {finish(t-t0);} }
function finish(ms){
  var renderer=null,vendor=null,ok=false,err=null;
  try{ var c=document.createElement('canvas');
    var gl=c.getContext('webgl')||c.getContext('experimental-webgl');
    if(gl){ ok=true; var e=gl.getExtension('WEBGL_debug_renderer_info');
      renderer=e?gl.getParameter(e.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER);
      vendor=e?gl.getParameter(e.UNMASKED_VENDOR_WEBGL):gl.getParameter(gl.VENDOR); } }
  catch(x){ err=''+x; }
  deltas.sort(function(a,b){return a-b;});
  var p95=deltas.length?deltas[Math.min(deltas.length-1,Math.floor(deltas.length*0.95))]:null;
  document.title='RESULT'+JSON.stringify({frames:n,ms:ms,fps:n/(ms/1000),
    p95_frame_ms:p95,webgl:ok,webgl_renderer:renderer,webgl_vendor:vendor,webgl_error:err,
    ua:navigator.userAgent});
}
requestAnimationFrame(frame);
</script></body></html>"""


def sh(cmd: list[str], cwd: str | None = None, timeout: int = 300, env: dict | None = None) -> dict:
    exe = shutil.which(cmd[0]) if not os.path.isabs(cmd[0]) else cmd[0]
    if exe is None or (os.path.isabs(cmd[0]) and not os.path.exists(cmd[0])):
        return {"present": False, "cmd": cmd}
    try:
        e = dict(os.environ)
        if env:
            e.update(env)
        r = subprocess.run(cmd, cwd = cwd, capture_output = True, text = True,
                           timeout = timeout, env = e)
        return {"present": True, "cmd": cmd, "rc": r.returncode,
                "stdout": r.stdout[-8000:], "stderr": r.stderr[-8000:]}
    except Exception as ex:  # noqa: BLE001
        return {"present": True, "cmd": cmd, "error": f"{type(ex).__name__}: {ex}"}


# --------------------------------------------------------------------------
# inventory


def inventory() -> dict:
    typelibs = sorted(os.path.basename(p) for p in
                      glob.glob("/usr/lib/*/girepository-1.0/*.typelib")
                      + glob.glob("/usr/lib/girepository-1.0/*.typelib"))
    return {
        "typelibs_gtk_webkit": [t for t in typelibs
                                if t.startswith(("Gtk-", "WebKit", "Gdk-", "Soup-"))],
        "n_typelibs": len(typelibs),
        "Xvfb": shutil.which("Xvfb"),
        "Xwayland": shutil.which("Xwayland"),
        "weston": shutil.which("weston"),
        "cage": shutil.which("cage"),
        "bwrap": shutil.which("bwrap"),
        "dbus_run_session": shutil.which("dbus-run-session"),
        "xkb_data": os.path.isdir("/usr/share/X11/xkb"),
        "x11_fonts": sorted(glob.glob("/usr/share/fonts/X11/*"))[:5],
        "webkit_processes_dir": sorted(
            glob.glob("/usr/lib/*/webkit2gtk-4.1/*") + glob.glob("/usr/libexec/webkit2gtk-4.1/*")),
        "xdg_runtime_dir": os.environ.get("XDG_RUNTIME_DIR"),
    }


# --------------------------------------------------------------------------
# display


def fetch_xvfb(work: Path) -> dict:
    """apt-get download + dpkg-deb -x, both of which work without root."""
    out: dict = {"downloads": {}, "prefix": str(work / "xroot")}
    pkgs = work / "pkgs"
    pkgs.mkdir(parents = True, exist_ok = True)
    root = work / "xroot"
    root.mkdir(parents = True, exist_ok = True)
    for pkg in XVFB_PKGS:
        r = sh(["apt-get", "download", pkg], cwd = str(pkgs), timeout = 300)
        out["downloads"][pkg] = {"rc": r.get("rc"), "err": (r.get("stderr") or "")[-400:]}
    debs = sorted(glob.glob(str(pkgs / "*.deb")))
    out["debs"] = [os.path.basename(d) for d in debs]
    out["extracted"] = []
    for d in debs:
        r = sh(["dpkg-deb", "-x", d, str(root)], timeout = 300)
        out["extracted"].append({"deb": os.path.basename(d), "rc": r.get("rc")})
    out["xvfb_binary"] = next(iter(glob.glob(str(root / "usr/bin/Xvfb"))), None)
    return out


def start_xserver(work: Path, obs: dict) -> tuple[subprocess.Popen | None, dict]:
    """Start an X server and return it plus the DISPLAY it is on."""
    info: dict = {"attempts": []}
    root = work / "xroot"
    candidates = [shutil.which("Xvfb"), str(root / "usr/bin/Xvfb")]
    display = ":99"
    log = work / "xvfb.log"
    for cand in candidates:
        if not cand or not os.path.exists(cand):
            info["attempts"].append({"binary": cand, "exists": False})
            continue
        env = {
            "LD_LIBRARY_PATH": f"{root}/usr/lib/x86_64-linux-gnu:{root}/usr/lib:"
                               f"{os.environ.get('LD_LIBRARY_PATH', '')}",
            "XKB_BINDIR": f"{root}/usr/bin",
        }
        cmd = [cand, display, "-screen", "0", "1280x800x24", "-nolisten", "tcp",
               "-xkbdir", (f"{root}/usr/share/X11/xkb"
                           if (root / "usr/share/X11/xkb").is_dir() else "/usr/share/X11/xkb")]
        with open(log, "a") as fh:
            fh.write(f"\n== {cmd}\n")
            fh.flush()
            e = dict(os.environ)
            e.update(env)
            try:
                proc = subprocess.Popen(cmd, stdout = fh, stderr = subprocess.STDOUT, env = e)
            except Exception as ex:  # noqa: BLE001
                info["attempts"].append({"binary": cand, "spawn_error": str(ex)})
                continue
        # Poll for the socket rather than sleeping a fixed interval.
        sock = f"/tmp/.X11-unix/X{display.lstrip(':')}"
        for _ in range(60):
            if proc.poll() is not None:
                break
            if os.path.exists(sock):
                info["attempts"].append({"binary": cand, "started": True, "display": display})
                info["display"] = display
                info["binary"] = cand
                obs["xvfb_log"] = log.read_text(errors = "replace")[-4000:]
                return proc, info
            time.sleep(0.5)
        info["attempts"].append({"binary": cand, "started": False,
                                 "exit": proc.poll(),
                                 "log": log.read_text(errors = "replace")[-2000:]})
        if proc.poll() is None:
            proc.kill()
    obs["xvfb_log"] = log.read_text(errors = "replace")[-4000:] if log.exists() else None
    return None, info


# --------------------------------------------------------------------------
# WebKit child: creates the view, paints, reports through the window title.


def webkit_child(out: Path, ms: int, snapshot: Path) -> int:
    res: dict = {"gi": False}
    try:
        import gi
        gi.require_version("Gtk", "3.0")
        gi.require_version("WebKit2", "4.1")
        from gi.repository import GLib, Gtk, WebKit2  # noqa: PLC0415
        res["gi"] = True
        res["webkit_version"] = f"{WebKit2.get_major_version()}.{WebKit2.get_minor_version()}." \
                                f"{WebKit2.get_micro_version()}"
    except Exception as e:  # noqa: BLE001
        res["import_error"] = f"{type(e).__name__}: {e}"
        out.write_text(json.dumps(res, indent = 2))
        return 0

    win = Gtk.Window()
    win.set_default_size(1280, 800)
    view = WebKit2.WebView()
    st = view.get_settings()
    st.set_property("enable-webgl", True)
    st.set_property("enable-developer-extras", True)
    try:
        # Default is ON_DEMAND; ALWAYS is what a desktop app with an animating
        # page ends up on, and it is the policy under test.
        st.set_property("hardware-acceleration-policy",
                        WebKit2.HardwareAccelerationPolicy.ALWAYS)
        res["hardware_acceleration_policy"] = "ALWAYS"
    except Exception as e:  # noqa: BLE001
        res["hardware_acceleration_policy_error"] = f"{type(e).__name__}: {e}"
    win.add(view)
    win.show_all()

    state = {"done": False}

    def descendants(root_pid: int) -> set[str]:
        """Only OUR WebKit processes.

        Four ephemeral slots share this machine, so a bare scan for "WebKit" in
        a command line can attribute another job's browser to this reading.
        """
        parent: dict[str, str] = {}
        for p in glob.glob("/proc/[0-9]*"):
            try:
                stat = Path(p, "stat").read_text(errors = "replace")
                ppid = stat.rsplit(")", 1)[1].split()[1]
                parent[os.path.basename(p)] = ppid
            except Exception:  # noqa: BLE001
                continue
        out = set()
        for pid in parent:
            seen, cur = 0, pid
            while cur in parent and seen < 40:
                if cur == str(root_pid):
                    out.add(pid)
                    break
                cur = parent[cur]
                seen += 1
        return out

    def collect_processes():
        procs = []
        mine = descendants(os.getpid()) | {str(os.getpid())}
        for p in glob.glob("/proc/[0-9]*"):
            if os.path.basename(p) not in mine:
                continue
            try:
                cmd = Path(p, "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                    errors = "replace").strip()
            except Exception:  # noqa: BLE001
                continue
            if "WebKit" not in cmd and "webkit" not in cmd:
                continue
            pid = os.path.basename(p)
            entry = {"pid": pid, "cmdline": cmd[:200], "dri_fds": [], "fdinfo": [],
                     "mapped_drivers": []}
            # Which mesa driver the engine's own process loaded. WebKitGTK
            # returns a masked "Apple GPU" from WEBGL_debug_renderer_info on
            # every port, so the in-page string cannot name the device; a
            # mapped radeonsi_dri.so or swrast_dri.so can, and the engine does
            # not choose what its own address space says.
            try:
                maps = Path(p, "maps").read_text(errors = "replace")
                names = set()
                for ln in maps.splitlines():
                    path = ln.split(" ", 5)[-1].strip() if " " in ln else ""
                    base = os.path.basename(path)
                    if base.endswith("_dri.so") or any(
                            t in base for t in ("radeonsi", "llvmpipe", "swrast", "libEGL",
                                                "libgbm", "libvulkan", "libGLX", "libGLESv2")):
                        names.add(base)
                entry["mapped_drivers"] = sorted(names)
            except Exception:  # noqa: BLE001
                pass
            for fd in glob.glob(f"{p}/fd/*"):
                try:
                    tgt = os.readlink(fd)
                except Exception:  # noqa: BLE001
                    continue
                if tgt.startswith("/dev/dri/"):
                    entry["dri_fds"].append(tgt)
                    try:
                        txt = Path(p, "fdinfo", os.path.basename(fd)).read_text(errors = "replace")
                        entry["fdinfo"].append({
                            k: v.strip() for k, v in
                            (ln.split(":", 1) for ln in txt.splitlines() if ":" in ln)
                            if k.startswith("drm-")})
                    except Exception:  # noqa: BLE001
                        pass
            procs.append(entry)
        return procs

    def finish(reason):
        if state["done"]:
            return
        state["done"] = True
        res["finish_reason"] = reason
        res["webkit_processes"] = collect_processes()

        def record_png(path):
            data = path.read_bytes()
            res["snapshot_bytes"] = len(data)
            res["snapshot_b64_head"] = base64.b64encode(data[:64]).decode()
            # Proof it is not a blank page: a flat surface compresses to a very
            # small number of distinct bytes, a painted one does not.
            res["snapshot_distinct_bytes"] = len(set(data[:200000]))

        def pixbuf_fallback():
            """Capture the real X window, so a missing pycairo is not a dead end."""
            try:
                gi.require_version("GdkPixbuf", "2.0")
                from gi.repository import Gdk  # noqa: PLC0415
                gw = win.get_window()
                w, h = gw.get_width(), gw.get_height()
                pb = Gdk.pixbuf_get_from_window(gw, 0, 0, w, h)
                alt = snapshot.with_name(snapshot.stem + "_window.png")
                pb.savev(str(alt), "png", [], [])
                record_png(alt)
                res["snapshot_source"] = "gdk_pixbuf_get_from_window"
            except Exception as e:  # noqa: BLE001
                res["snapshot_fallback_error"] = f"{type(e).__name__}: {e}"

        def done():
            out.write_text(json.dumps(res, indent = 2))
            Gtk.main_quit()

        def snap_done(v, task):
            try:
                surf = v.get_snapshot_finish(task)
                surf.write_to_png(str(snapshot))
                record_png(snapshot)
                res["snapshot_source"] = "webkit_get_snapshot"
            except Exception as e:  # noqa: BLE001
                res["snapshot_error"] = f"{type(e).__name__}: {e}"
                pixbuf_fallback()
            done()

        # If the snapshot callback never fires, quit anyway rather than sitting
        # in the main loop until the outer timeout kills the process.
        GLib.timeout_add(20000, lambda: (pixbuf_fallback(), done(), False)[2]
                         if not res.get("snapshot_bytes") else False)
        try:
            view.get_snapshot(WebKit2.SnapshotRegion.VISIBLE,
                              WebKit2.SnapshotOptions.NONE, None, snap_done)
        except Exception as e:  # noqa: BLE001
            res["snapshot_error"] = f"{type(e).__name__}: {e}"
            pixbuf_fallback()
            done()

    def on_title(v, _p):
        t = v.get_title() or ""
        if t.startswith("RESULT"):
            try:
                res["page"] = json.loads(t[len("RESULT"):])
            except Exception as e:  # noqa: BLE001
                res["page_parse_error"] = f"{type(e).__name__}: {e}"
            finish("page reported")

    def on_load(v, ev):
        if ev == WebKit2.LoadEvent.FINISHED:
            res["load_finished"] = True

    view.connect("notify::title", on_title)
    view.connect("load-changed", on_load)
    view.connect("web-process-terminated",
                 lambda v, r: (res.update({"web_process_terminated": str(r)}), finish("crash")))
    GLib.timeout_add(ms + 30000, lambda: (finish("timeout"), False)[1])

    view.load_html(PAGE.replace("__MS__", str(ms)), "file:///")
    Gtk.main()
    if not out.exists():
        out.write_text(json.dumps(res, indent = 2))
    return 0


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--state", default = "host")
    ap.add_argument("--checkout", default = "")
    ap.add_argument("--work", default = os.environ.get("AMD_CI_WORK", "/tmp/webkit_probe"))
    ap.add_argument("--ms", type = int, default = 5000)
    ap.add_argument("--webkit-child", action = "store_true")
    ap.add_argument("--snapshot", type = Path, default = None)
    args = ap.parse_args()
    args.out.parent.mkdir(parents = True, exist_ok = True)

    if args.webkit_child:
        return webkit_child(args.out, args.ms, args.snapshot or (args.out.parent / "page.png"))

    work = Path(args.work) / "webkit"
    work.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "inventory": inventory()}

    if not obs["inventory"]["Xvfb"]:
        obs["fetch_xvfb"] = fetch_xvfb(work)

    xproc, xinfo = start_xserver(work, obs)
    obs["xserver"] = xinfo

    try:
        if not xinfo.get("display"):
            obs["webkit"] = {"skipped": "no display server could be started"}
        else:
            child_out = args.out.parent / "webkit_child.json"
            snap = args.out.parent / "page.png"
            cmd = ["dbus-run-session", "--"] if shutil.which("dbus-run-session") else []
            cmd += [sys.executable, os.path.abspath(__file__), "--webkit-child",
                    "--out", str(child_out), "--ms", str(args.ms), "--snapshot", str(snap)]
            env = {"DISPLAY": xinfo["display"], "GDK_BACKEND": "x11",
                   "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR", str(work))}
            r = sh(cmd, timeout = args.ms // 1000 + 180, env = env)
            obs["webkit_run"] = {k: r.get(k) for k in ("rc", "stdout", "stderr", "present")}
            if child_out.is_file():
                try:
                    obs["webkit"] = json.loads(child_out.read_text())
                except Exception as e:  # noqa: BLE001
                    obs["webkit"] = {"parse_error": f"{type(e).__name__}: {e}"}
            else:
                obs["webkit"] = {"missing_output": True}

            # The same reading the standalone EGL probe takes, from the same
            # host, so the two strings can be compared directly.
            obs["reference_gl"] = sh([sys.executable,
                                      str(Path(__file__).parent / "display_stack_probe.py"),
                                      "--egl-worker", "--platform", "surfaceless",
                                      "--out", str(args.out.parent / "_ref_egl.json")])
            ref = args.out.parent / "_ref_egl.json"
            if ref.is_file():
                try:
                    obs["reference_gl_renderer"] = json.loads(ref.read_text()).get("gl_renderer")
                except Exception:  # noqa: BLE001
                    pass
    finally:
        # By PID. `pkill -f Xvfb` would match this probe's own command line, and
        # an X server left running would outlive the job and hold the GPU.
        if xproc is not None and xproc.poll() is None:
            try:
                os.kill(xproc.pid, signal.SIGTERM)
                time.sleep(2)
                if xproc.poll() is None:
                    os.kill(xproc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
            obs["xserver"]["stopped_pid"] = xproc.pid

    args.out.write_text(json.dumps(obs, indent = 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

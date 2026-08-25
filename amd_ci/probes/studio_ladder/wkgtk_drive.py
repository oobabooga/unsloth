#!/usr/bin/python3.12
"""Drive a URL in REAL WebKitGTK (webkit2gtk-4.1), the engine wry/Tauri embeds on Linux.

Not Playwright WebKit. Not Chromium. This is libwebkit2gtk-4.1.so, the same shared
library Unsloth Desktop loads on Linux.

Contract with the page:
  - --init-script  JS injected at document-start (before app JS runs)
  - --script       JS evaluated once, after load finishes
  - the page ends the run by calling
        window.webkit.messageHandlers.bench.postMessage(JSON.stringify(payload))
    with an object carrying {"__done": true, ...}; that object is written to --out.
  - any other posted message is appended to --out-log as one JSON line.

Usage:
  DISPLAY=:77 /usr/bin/python3.12 wkgtk_drive.py --url http://... --init-script a.js \
      --script b.js --out result.json --timeout 300
"""
import argparse, json, os, sys, time

import gi
gi.require_version("Gtk", "3.0")
gi.require_version("WebKit2", "4.1")
from gi.repository import Gtk, WebKit2, GLib  # noqa: E402


def read(p):
    if not p:
        return None
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--init-script", default=None)
    ap.add_argument("--script", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-log", default=None)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--after-load-delay", type=float, default=2.0)
    # The Desktop app runs a FORCED SOFTWARE RENDERER on Linux. Under WebKitGTK the knob that
    # decides whether every composite goes through that software GL stack is the hardware
    # acceleration policy, so it has to be settable rather than left at whatever the headless
    # X server negotiates.
    ap.add_argument("--accel", choices=["always", "never", "ondemand", "default"],
                    default="default")
    args = ap.parse_args()

    state = {"done": False, "exit": 1}
    logf = open(args.out_log, "a", encoding="utf-8") if args.out_log else None

    def emit(tag, obj):
        line = json.dumps({"t": time.time(), "tag": tag, "v": obj})
        print(line, flush=True)
        if logf:
            logf.write(line + "\n")
            logf.flush()

    ucm = WebKit2.UserContentManager()
    if args.init_script:
        ucm.add_script(
            WebKit2.UserScript.new(
                read(args.init_script),
                WebKit2.UserContentInjectedFrames.TOP_FRAME,
                WebKit2.UserScriptInjectionTime.START,
                None,
                None,
            )
        )

    ctx = WebKit2.WebContext.get_default()
    view = WebKit2.WebView.new_with_user_content_manager(ucm)

    settings = view.get_settings()
    if args.accel != "default":
        pol = {"always": WebKit2.HardwareAccelerationPolicy.ALWAYS,
               "never": WebKit2.HardwareAccelerationPolicy.NEVER}.get(args.accel)
        if pol is None:
            pol = getattr(WebKit2.HardwareAccelerationPolicy, "ON_DEMAND", None)
        if pol is not None:
            settings.set_hardware_acceleration_policy(pol)
    settings.set_enable_developer_extras(True)
    settings.set_enable_write_console_messages_to_stdout(True)
    settings.set_javascript_can_access_clipboard(True)

    def on_msg(_ucm, js_result):
        try:
            # WebKitGTK 2.40+: JavaScriptCore.Value
            val = js_result.to_string() if hasattr(js_result, "to_string") else js_result.get_js_value().to_string()
        except Exception as e:  # pragma: no cover
            emit("msg_error", repr(e))
            return
        try:
            obj = json.loads(val)
        except Exception:
            emit("msg_raw", val)
            return
        if isinstance(obj, dict) and obj.get("__done"):
            gaps = [round((b - a) * 1000, 1) for a, b in zip(draw_ts, draw_ts[1:])]
            obj["widget_draws"] = {
                "n": len(draw_ts),
                "first": draw_ts[0] if draw_ts else None,
                "last": draw_ts[-1] if draw_ts else None,
                "gaps_ms": gaps,
            }
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(obj, f)
            emit("done", {"bytes": len(val)})
            state["done"] = True
            state["exit"] = 0
            Gtk.main_quit()
        else:
            emit("msg", obj)

    ucm.connect("script-message-received::bench", on_msg)
    if not ucm.register_script_message_handler("bench"):
        print("FATAL: could not register script message handler", file=sys.stderr)
        sys.exit(2)

    # A PRESENTED-FRAME channel that does not go through rAF.
    #
    # requestAnimationFrame in a headless X server is not vsync locked: gaps of 8-9 ms are
    # recorded, which is 120 Hz on a screen that has no refresh rate at all. So rAF here measures
    # MAIN THREAD AVAILABILITY, not frames a user would see. GtkWidget::draw fires once per
    # repaint of the widget, so counting it gives the second number, and the two disagreeing is
    # itself the finding.
    # GdkFrameClock::after-paint, one emission per RENDERED FRAME of the toplevel. Measured at
    # 59.2-59.5/s with a 16.7 ms median on an idle page, i.e. a real 60 Hz presentation clock,
    # while rAF in the same process reports gaps of 8-9 ms. They are different quantities and only
    # this one is what a user sees.
    #
    # GtkWidget::draw was tried first and read zero. That was an artefact: PyGObject cannot
    # marshal cairo.Context, so the handler raised before its first statement. after-paint takes
    # no arguments and marshals cleanly. Positive control in scripts/wk_draw_probe.py.
    draw_ts = []

    win = Gtk.Window()
    win.set_default_size(args.width, args.height)
    win.add(view)
    win.show_all()
    win.connect("destroy", Gtk.main_quit)

    def on_load(_v, event):
        if event == WebKit2.LoadEvent.FINISHED:
            emit("load_finished", view.get_uri())
            if args.script:
                def run_it():
                    view.evaluate_javascript(read(args.script), -1, None, None, None,
                                             lambda v, r, u: None, None)
                    return False
                GLib.timeout_add(int(args.after_load_delay * 1000), run_it)

    view.connect("load-changed", on_load)

    fc_state = {"wired": False}

    def wire_frame_clock():
        if fc_state["wired"]:
            return False
        gw = win.get_window()
        if gw is None:
            return True
        fc = gw.get_frame_clock()
        if fc is None:
            return True
        fc.connect("after-paint", lambda *a: draw_ts.append(time.time()))
        fc.begin_updating()
        fc_state["wired"] = True
        emit("frame_clock_wired", True)
        return False

    GLib.timeout_add(200, wire_frame_clock)

    def on_fail(_v, _ev, uri, err):
        emit("load_failed", {"uri": uri, "err": str(err)})
        return False

    view.connect("load-failed", on_fail)

    def bail():
        if not state["done"]:
            emit("timeout", args.timeout)
            state["exit"] = 3
            Gtk.main_quit()
        return False

    GLib.timeout_add(int(args.timeout * 1000), bail)

    view.load_uri(args.url)
    emit("start", {"url": args.url, "webkit": "%d.%d.%d" % (
        WebKit2.get_major_version(), WebKit2.get_minor_version(), WebKit2.get_micro_version())})
    Gtk.main()
    sys.exit(state["exit"])


if __name__ == "__main__":
    main()

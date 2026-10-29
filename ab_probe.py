"""RD-845 A/B driver: run one sd.cpp test-launch scenario against a Windows ROCm bundle on a host
without the HIP SDK, watching for loader hard-error boxes while it runs.

usage: ab_probe.py <backend_dir> <label> <pristine_tree> <out_dir> <scenario>
"""

import ctypes
import ctypes.wintypes as wt
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

backend, label, pristine, out_dir, scenario = sys.argv[1:6]
pristine = Path(pristine)
out_dir = Path(out_dir)
out_dir.mkdir(parents = True, exist_ok = True)
sys.path.insert(0, backend)

k32 = ctypes.WinDLL("kernel32")
user32 = ctypes.WinDLL("user32")
inherited_mode = k32.GetErrorMode()
# A Studio launched from the desktop starts with the default mode; do not inherit the runner's.
k32.SetErrorMode(0)

WNDENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)


def _text(hwnd):
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def _class(hwnd):
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def dialogs():
    """Visible top-level dialog-class windows: {hwnd: (title, [child texts])}."""
    found = {}

    def top(hwnd, _):
        if user32.IsWindowVisible(hwnd) and _class(hwnd) == "#32770":
            texts = []

            def child(ch, _):
                t = _text(ch)
                if t:
                    texts.append(t)
                return True

            user32.EnumChildWindows(hwnd, WNDENUMPROC(child), 0)
            found[hwnd] = (_text(hwnd), texts)
        return True

    user32.EnumWindows(WNDENUMPROC(top), 0)
    return found


class Watcher(threading.Thread):
    def __init__(self):
        super().__init__(daemon = True)
        self.baseline = set(dialogs())
        self.seen = {}
        self.t0 = time.monotonic()
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            for hwnd, (title, texts) in dialogs().items():
                if hwnd in self.baseline or hwnd in self.seen:
                    continue
                at = round(time.monotonic() - self.t0, 2)
                self.seen[hwnd] = {"at_s": at, "title": title, "text": texts}
                print(f"  [{label}/{scenario}] DIALOG at {at}s: {title!r} {texts}", flush = True)
                try:
                    from PIL import ImageGrab

                    shot = out_dir / f"{label}-{scenario}-dialog{len(self.seen)}.png"
                    ImageGrab.grab(all_screens = True).save(shot)
                except Exception as exc:  # noqa: BLE001
                    print(f"  screenshot failed: {exc}", flush = True)
            time.sleep(0.2)

    def finish(self):
        self.stop.set()
        self.join()
        # Dismiss what we opened so the next scenario starts clean.
        for hwnd in self.seen:
            user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
        time.sleep(1)
        return list(self.seen.values())


def popup_events(since_ms):
    q = (
        "*[System[Provider[@Name='Application Popup'] and "
        f"TimeCreated[timediff(@SystemTime) <= {since_ms}]]]"
    )
    r = subprocess.run(
        ["wevtutil", "qe", "System", f"/q:{q}", "/f:text", "/rd:true"],
        capture_output = True, text = True, errors = "replace",
    )
    return [
        line.strip() for line in r.stdout.splitlines() if line.strip().startswith("Description")
        or "hipblas" in line.lower() or "System Error" in line
    ]


from core.inference import sd_cpp_backend as bk  # noqa: E402
from core.inference import sd_cpp_engine as eng  # noqa: E402

root = eng.managed_install_root()


def fresh_tree():
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(pristine, root)


def record_accelerator():
    try:
        return json.loads((root / ".unsloth-sd-cpp-install.json").read_text()).get("accelerator")
    except Exception as exc:  # noqa: BLE001
        return f"<no record: {exc}>"


def run_scenario():
    out = {}
    fresh_tree()
    server = eng.find_sd_server_binary()
    cli = eng.find_sd_cpp_binary()
    out["seeded_server"] = server
    out["seeded_record_accelerator"] = record_accelerator()
    if scenario == "probe_server":
        out["result"] = bk._server_binary_runnable(server)
    elif scenario == "probe_output":
        out["result"] = bk._sd_cpp_probe_output(cli, "--help")
    elif scenario == "engine_probes":
        out["identifies"] = eng.sd_cpp_binary_identifies(cli)
        out["version"] = eng.SdCppEngine(binary = cli).version()
    elif scenario in ("upgrade_to_vulkan", "forced_rocm"):
        if scenario == "forced_rocm":
            os.environ["UNSLOTH_SD_CPP_WINDOWS_ROCM"] = "1"
        acc = bk.preferred_accelerator("rocm")
        out["requested_accelerator"] = acc
        s = bk.ensure_sd_server_binary(accelerator = acc)
        out["ensure_server_returned"] = s
        out["returned_is_seeded_file"] = bool(s) and s == server and Path(s).is_file()
        out["record_after_server"] = record_accelerator()
        if s:
            # The router probes the returned server again before using it (diffusion_engine_router.py:915).
            out["router_recheck_runnable"] = bk._server_binary_runnable(s)
        c = bk.ensure_sd_cpp_binary(accelerator = acc)
        out["ensure_cli_returned"] = c
        out["record_after_cli"] = record_accelerator()
    else:
        raise SystemExit(f"unknown scenario {scenario}")
    return out


print(f"== {label} / {scenario}: inherited error mode {inherited_mode:#x}, reset to 0; managed root {root}", flush = True)
watcher = Watcher()
watcher.start()
t0 = time.monotonic()
try:
    out = run_scenario()
except Exception as exc:  # noqa: BLE001
    out = {"exception": repr(exc)}
elapsed = round(time.monotonic() - t0, 2)
seen = watcher.finish()
sid = wt.DWORD()
k32.ProcessIdToSessionId(os.getpid(), ctypes.byref(sid))
out.update(
    session_id = sid.value,
    label = label,
    scenario = scenario,
    elapsed_s = elapsed,
    dialogs = seen,
    popup_events = popup_events(int(elapsed * 1000) + 5000),
    error_mode_after = hex(k32.GetErrorMode()),
)
print(json.dumps(out, indent = 2, default = str), flush = True)
with open(out_dir / "results.jsonl", "a", encoding = "utf-8") as fh:
    fh.write(json.dumps(out, default = str) + "\n")

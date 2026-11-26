#!/usr/bin/env python3
"""M8 (b) step 2: one Studio load (probes/r8/e2e_arm.py, fresh process) with the native cache's per-tensor mappings
as on main (``--access copy``: mmap ACCESS_COPY) or read-only (``--access read``: the same function, ACCESS_READ).
Counts ``_map_tensor`` calls (engagement: 0 means the load never took the mapped path) and samples RssAnon /
Private_Dirty every 0.2 s for the peak. Everything after ``--`` goes to e2e_arm.py unchanged.

  run_mmap_load.py --root B --access copy|read --side-out F -- <e2e_arm args without --root>
"""
from __future__ import annotations

import inspect
import json
import os
import runpy
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _rss() -> dict:
    out = {}
    try:
        for ln in open("/proc/self/status", encoding = "utf-8"):
            if ln.startswith(("RssAnon", "RssFile")):
                k, _, v = ln.partition(":")
                out[k] = int(v.split()[0]) * 1024
    except Exception:  # noqa: BLE001
        pass
    return out


def main() -> int:
    argv = sys.argv[1:]
    sep = argv.index("--")
    own, rest = argv[:sep], argv[sep + 1:]
    opts = dict(zip(own[::2], own[1::2]))
    root, access, side = opts["--root"], opts["--access"], Path(opts["--side-out"])
    sys.path.insert(0, root)
    rec = {"access": access, "map_calls": 0, "peak": {}}
    from core.inference import diffusion_native_cache as NC
    orig = NC._map_tensor
    fn = orig
    if access == "read":
        src = inspect.getsource(orig)
        if src.count("access = mmap.ACCESS_COPY") != 1:
            rec["error"] = "tree's _map_tensor does not map ACCESS_COPY once"
        else:
            ns = dict(NC.__dict__)
            exec(compile(src.replace("access = mmap.ACCESS_COPY", "access = mmap.ACCESS_READ"), NC.__file__, "exec"),
                 ns)
            fn = ns["_map_tensor"]

    def counted(*a, **k):
        rec["map_calls"] += 1
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*a, **k)
    NC._map_tensor = counted
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            for k, v in _rss().items():
                rec["peak"][k] = max(rec["peak"].get(k, 0), v)
            time.sleep(0.2)
    threading.Thread(target = sampler, daemon = True).start()
    real_exit = os._exit

    def finish():
        stop.set()
        rec["end"] = _rss()
        side.write_text(json.dumps(rec), encoding = "utf-8")

    def exit_hook(code):  # e2e_arm ends with os._exit, which skips finally blocks
        finish()
        real_exit(code)
    os._exit = exit_hook
    sys.argv = [str(HERE / "e2e_arm.py"), "--root", root, *rest]
    rc = 0
    try:
        runpy.run_path(str(HERE / "e2e_arm.py"), run_name = "__main__")
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 0
    finally:
        finish()
    return rc


if __name__ == "__main__":
    sys.exit(main())

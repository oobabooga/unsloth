#!/usr/bin/env python3
"""run_cell.py with the q21 profile hook attached to the Studio denoiser instance after load (ComfyUI cells get it
from the custom node in the server process). Same CLI as run_cell.py."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import q21_prof_hook  # noqa: E402
import run_cell  # noqa: E402
from backends import studio_inproc  # noqa: E402

_orig_load = studio_inproc.StudioBackend.load


def _load(self):
    st = _orig_load(self)
    hits = []
    for obj in gc.get_objects():
        try:
            if type(obj).__name__ == "QwenImage21Transformer2DModel":
                hits.append(obj)
        except Exception:  # noqa: BLE001
            continue
    attached = sum(q21_prof_hook.attach_instance(m, "studio") for m in hits)
    print(f"q21_prof: Studio denoisers found {len(hits)}, wrapped {attached}", flush = True)
    st["q21_prof_wrapped"] = attached
    return st


studio_inproc.StudioBackend.load = _load

if __name__ == "__main__":
    rc = run_cell.main()
    q21_prof_hook.flush()
    raise SystemExit(rc)

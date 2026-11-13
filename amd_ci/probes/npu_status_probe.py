#!/usr/bin/env python3
"""Probe: what this checkout's NPU backend reports on the real host (status() starts nothing)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    obs: dict = {"state": args.state}
    backend = args.checkout / "studio" / "backend"
    sys.path.insert(0, str(backend))
    for stale in [m for m in sys.modules if m.startswith(("core.", "utils."))]:
        del sys.modules[stale]
    try:
        import core.inference.npu_backend as nb  # noqa: PLC0415
        obs["module_file"] = nb.__file__
        st = nb.get_npu_backend().status()
        obs["status"] = st
        obs["footer_would_show"] = bool(st.get("supported"))
    except Exception as e:  # noqa: BLE001
        obs["error"] = f"{type(e).__name__}: {e}"
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

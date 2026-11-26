#!/usr/bin/env python3
"""C15-3 arm wrapper (observes only): runs c153_vendor_check.py (a copy of the conv3d productisation ticket's
conv3d_vendor_check.py) in THIS process, against the vendored ROCm/aiter Triton conv3d shipped in c153_pkg (the
productisation branch's files, whose gfx1151 tables are ROCm/aiter PR #6973 at 065f9f99c9).

  --tables pr6973   the vendored tables as shipped (gfx1151, PR #6973)
  --tables gfx1100  the same kernels with aiter's gfx1100 conv3d tables (aiter 07933277b5, the PR's base) swapped in
                    for gfx1151: the "without the 8 tables" arm. Engagement: the table in force is recorded.
Every other argument goes to the check (--mode decode|ltxdecode --arm stock|triton --vae-dir D --out J --save T).
The wrapper writes its own facts to <out>.tables.json."""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    argv = sys.argv[1:]
    tables = "pr6973"
    if "--tables" in argv:
        i = argv.index("--tables")
        tables = argv[i + 1]
        del argv[i:i + 2]
    out = argv[argv.index("--out") + 1]
    pkg = str(HERE / "c153_pkg")
    sys.path.insert(0, pkg)
    facts = {"tables": tables, "pkg": pkg}
    from core.inference.rocm_triton_conv3d import _config as C
    if tables == "gfx1100":
        C.TABLES["gfx1151"] = json.loads((HERE / "c153_gfx1100_tables.json").read_text(encoding = "utf-8"))
    facts["any_1x1x1_in_force"] = C.TABLES["gfx1151"]["CONV3D-1X1X1"]["any"]
    facts["families"] = sorted(C.TABLES["gfx1151"])
    Path(out + ".tables.json").write_text(json.dumps(facts), encoding = "utf-8")
    sys.argv = [str(HERE / "c153_vendor_check.py"), "--backend", pkg, *argv]
    try:
        runpy.run_path(str(HERE / "c153_vendor_check.py"), run_name = "__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
